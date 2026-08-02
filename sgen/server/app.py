"""Local web UI for sgen.

Binds to 127.0.0.1 only. Media is never uploaded or copied — the browser picks
paths and the server reads them in place, which keeps multi-gigabyte files off
the wire and personal footage where it already lives.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import string
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import library, settings
from ..config import WORK_DIR, Config, enforce_offline
from .jobs import Job, JobQueue, cloud_provider, build_config

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _asset_version() -> str:
    """Short hash of the frontend files, so edits always bust the cache."""
    marker = 0.0
    for name in ("index.html", "app.js", "style.css"):
        path = STATIC_DIR / name
        if path.exists():
            marker = max(marker, path.stat().st_mtime)
    return f"{int(marker)}"

VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".webm", ".mpg",
    ".mpeg", ".mts", ".m2ts", ".3gp", ".ts", ".vob",
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#1f2937"/>'
    b'<rect x="5" y="19" width="22" height="3" rx="1.5" fill="#e5e7eb"/>'
    b'<rect x="9" y="24" width="14" height="3" rx="1.5" fill="#9ca3af"/>'
    b"</svg>"
)


# --------------------------------------------------------------------------- #
# event fan-out
# --------------------------------------------------------------------------- #

class EventHub:
    """Broadcasts job updates to every connected browser via SSE."""

    def __init__(self) -> None:
        self._subscribers: list[Queue] = []

    def subscribe(self) -> Queue:
        q: Queue = Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                # Slow or dead client: drop it rather than block the worker.
                self.unsubscribe(q)


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #

class SubmitRequest(BaseModel):
    paths: list[str]
    out_dir: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    # Skip files that already have complete subtitles for these settings. What
    # counts as complete is decided from the files on disk, so closing the app or
    # restarting the machine loses nothing.
    skip_done: bool = False
    recursive: bool = True


class ScanRequest(BaseModel):
    folder: str
    options: dict[str, Any] = Field(default_factory=dict)
    out_dir: str | None = None
    recursive: bool = True


class FolderConfigRequest(BaseModel):
    """One file's per-file settings, written to the folder's own config file."""

    folder: str
    path: str
    values: dict[str, Any] = Field(default_factory=dict)


class RegateRequest(BaseModel):
    gating: dict[str, Any] = Field(default_factory=dict)
    cues: dict[str, Any] = Field(default_factory=dict)
    keep_suppressed: bool = False


class SaveRequest(BaseModel):
    cues: list[dict[str, Any]]
    formats: list[str] = Field(default_factory=lambda: ["srt", "vtt"])
    language: str | None = None
    out_dir: str | None = None
    romanize: bool = False


class OnlineTranslateRequest(BaseModel):
    provider: str = "google"
    language: str = "en"
    formats: list[str] = Field(default_factory=lambda: ["srt", "vtt"])
    out_dir: str | None = None


class KeysRequest(BaseModel):
    google: str | None = None
    deepl: str | None = None
    deepl_plan: str | None = None


class TranslateDefaultRequest(BaseModel):
    """Remember a translation choice, so it applies without being re-picked."""

    auto: bool
    provider: str = ""
    target: str = ""


class ForgetAllRequest(BaseModel):
    """Deleting every cached transcript is not something to do by accident."""

    confirm: bool = False


class ProfileRequest(BaseModel):
    gating: dict[str, Any] = Field(default_factory=dict)
    cues: dict[str, Any] = Field(default_factory=dict)


class ImportTranslationRequest(BaseModel):
    text: str
    language: str = "en"
    formats: list[str] = Field(default_factory=lambda: ["srt", "vtt"])
    keep_untranslated: bool = True
    out_dir: str | None = None


def _out_dir(requested: str | None) -> Path | None:
    """Where to write. Falls back to defaults.out_dir in the settings file.

    None means "next to the source file", which is what both an empty request
    field and an empty setting mean.
    """
    if requested and requested.strip():
        return Path(requested.strip())
    configured = settings.load_or_default().defaults.out_dir.strip()
    return Path(configured) if configured else None


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

def create_app() -> FastAPI:
    enforce_offline()

    hub = EventHub()
    queue = JobQueue(on_change=lambda job: hub.publish({"type": "job", "job": job.to_dict()}))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        queue.shutdown()

    app = FastAPI(title="sgen", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.hub = hub
    app.state.queue = queue

    # ----------------------------------------------------------------- pages

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the page, never cached, with version-stamped asset URLs.

        Without this the browser can hold a stale index.html while /static/app.js
        revalidates via ETag and updates. New JS against old HTML means a
        module-scope element lookup returns null, the script dies on the spot,
        and unrelated things — the file browser most visibly — stop responding.
        Stamping the asset URLs makes a changed file a changed URL.
        """
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        stamp = _asset_version()
        html = html.replace("/static/style.css", f"/static/style.css?v={stamp}")
        html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )

    @app.get("/favicon.ico")
    def favicon() -> Response:
        """A tab icon, and one fewer 404 in the console while debugging."""
        return Response(content=FAVICON, media_type="image/svg+xml")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------ filesystem

    @app.get("/api/drives")
    def drives() -> dict[str, Any]:
        found = []
        if os.name == "nt":
            for letter in string.ascii_uppercase:
                root = Path(f"{letter}:/")
                if root.exists():
                    found.append(str(root))
        else:
            found.append("/")
        return {
            "drives": found,
            "home": str(Path.home()),
            "cwd": str(Path.cwd()),
        }

    @app.get("/api/browse")
    def browse(path: str = "") -> dict[str, Any]:
        target = Path(path) if path else Path.home()
        try:
            target = target.resolve()
        except OSError as exc:
            raise HTTPException(400, f"bad path: {exc}") from exc
        if not target.is_dir():
            raise HTTPException(400, f"not a directory: {target}")

        dirs, files = [], []
        try:
            for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
                try:
                    if entry.is_dir():
                        if entry.name.startswith((".", "$")):
                            continue
                        dirs.append({"name": entry.name, "path": str(entry)})
                    elif entry.suffix.lower() in MEDIA_SUFFIXES:
                        files.append({
                            "name": entry.name,
                            "path": str(entry),
                            "size": entry.stat().st_size,
                        })
                except OSError:
                    continue  # permission denied on a single entry
        except PermissionError as exc:
            raise HTTPException(403, f"permission denied: {exc}") from exc

        return {
            "path": str(target),
            "parent": str(target.parent) if target.parent != target else None,
            "dirs": dirs,
            "files": files,
        }

    # ------------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": queue.list()}

    def _settings_for(folder: Path, base: dict[str, Any]):
        """Build a per-file (config, translate target, overrides) resolver.

        Each file is judged and run with its own settings, so one folder can hold
        songs, Hindi footage and English clips without three separate runs.
        """
        from .. import folderconf

        overrides = folderconf.load(folder)

        def resolve(source: Path):
            key = folderconf.key_for(folder, source)
            mine = overrides.get(key) or overrides.get(source.name) or {}
            options = folderconf.apply_to_options(base, mine)
            return build_config(options), _translate_target_for(options), mine

        return resolve, overrides

    def _translate_target_for(options: dict[str, Any]) -> str | None:
        """The language a translation was asked for, or None.

        Needed to judge "finished": a Russian file with only Russian subtitles is
        done if nothing asked for English, and unfinished if something did.
        """
        cfg = build_config(options)
        if cloud_provider(options) or cfg.translate_to_english:
            return (options.get("translate_target") or cfg.translate_target or "en").lower()
        return None

    @app.post("/api/scan")
    def scan(req: ScanRequest) -> dict[str, Any]:
        """What a folder still needs, judged from the files in it."""
        from .. import resume

        from .. import folderconf

        folder = Path(req.folder)
        resolve, _ = _settings_for(folder, req.options)
        try:
            result = resume.scan_folder(
                folder,
                build_config(req.options),
                out_dir=_out_dir(req.out_dir),
                translate_target=_translate_target_for(req.options),
                recursive=req.recursive,
                settings_for=resolve,
            )
        except NotADirectoryError as exc:
            raise HTTPException(400, f"not a folder: {exc}") from exc
        payload = result.to_dict()
        payload["summary"] = resume.summarise(result)
        payload["config_file"] = str(folderconf.config_path(folder))
        return payload

    @app.post("/api/folder-config/reset")
    def reset_folder_config(req: ScanRequest) -> dict[str, Any]:
        """Put every file in this folder back on the panel's settings."""
        from .. import folderconf

        folder = Path(req.folder)
        if not folder.is_dir():
            raise HTTPException(400, f"not a folder: {folder}")
        try:
            path, cleared = folderconf.clear_all(folder)
        except OSError as exc:
            raise HTTPException(400, f"could not remove {path}: {exc}") from exc
        return {"ok": True, "cleared": cleared, "path": str(path)}

    @app.post("/api/folder-config")
    def set_folder_config(req: FolderConfigRequest) -> dict[str, Any]:
        """Set one file's per-file settings, in the folder's own config file."""
        from .. import folderconf

        folder = Path(req.folder)
        if not folder.is_dir():
            raise HTTPException(400, f"not a folder: {folder}")
        source = Path(req.path)
        try:
            overrides = folderconf.set_for_file(folder, source, req.values)
        except folderconf.FolderConfigError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(400, f"could not write the folder settings: {exc}") from exc
        return {
            "ok": True,
            "path": str(folderconf.config_path(folder)),
            "overrides": overrides,
        }

    @app.post("/api/jobs")
    def submit(req: SubmitRequest) -> dict[str, Any]:
        from .. import resume

        from .. import folderconf

        out_dir = _out_dir(req.out_dir)
        requested = [Path(p) for p in req.paths]
        expanded = resume.iter_media(requested, req.recursive)
        if not expanded:
            raise HTTPException(400, "no media files found in selection")

        # Per-file settings come from the folder each file lives in, so they
        # apply whether the folder or the file was picked.
        def options_for(source: Path) -> dict[str, Any]:
            override = folderconf.for_file(source.parent, source)
            if not override:
                for base in requested:
                    if base.is_dir():
                        found = folderconf.for_file(base, source)
                        if found:
                            override = found
                            break
            return folderconf.apply_to_options(req.options, override)

        skipped: list[dict[str, Any]] = []
        if req.skip_done and not req.options.get("overwrite"):
            keep: list[Path] = []
            for source in expanded:
                options = options_for(source)
                status = resume.classify(
                    source, build_config(options), out_dir=out_dir,
                    translate_target=_translate_target_for(options),
                )
                (keep if status.needs_work else skipped).append(
                    source if status.needs_work else status.to_dict()
                )
            expanded = keep

        created = [
            queue.submit(p, out_dir, options_for(p)).to_dict() for p in expanded
        ]
        return {"jobs": created, "skipped": skipped, "skipped_count": len(skipped)}

    @app.delete("/api/jobs/{job_id}")
    def cancel(job_id: str) -> dict[str, Any]:
        if not queue.cancel(job_id):
            raise HTTPException(409, "job is not cancellable (already running or finished)")
        return {"ok": True}

    @app.post("/api/jobs/clear")
    def clear() -> dict[str, Any]:
        return {"removed": queue.clear_finished()}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        q = hub.subscribe()

        async def stream():
            try:
                yield f"data: {json.dumps({'type': 'hello', 'jobs': queue.list()})}\n\n"
                while True:
                    try:
                        payload = q.get_nowait()
                        yield f"data: {json.dumps(payload)}\n\n"
                    except Empty:
                        await asyncio.sleep(0.15)
                        yield ": ping\n\n"  # keep the connection warm
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --------------------------------------------------------------- results

    def _sidecar(content_id: str) -> Path:
        path = WORK_DIR / content_id / "transcript.sgen.json"
        if not path.exists():
            raise HTTPException(404, f"no transcript for {content_id}")
        return path

    @app.get("/api/result/{content_id}")
    def result(content_id: str) -> dict[str, Any]:
        from ..write import load_sidecar

        data = load_sidecar(_sidecar(content_id))
        edits = WORK_DIR / content_id / "edits.json"
        return {
            "source": data.get("source", {}),
            "language": data.get("language"),
            "language_probability": data.get("language_probability"),
            "duration": data.get("audio_duration"),
            "model": data.get("model"),
            "gating": data.get("gating"),
            "config": data.get("config"),
            "segments": data.get("segments", []),
            "cues": data.get("cues", []),
            "edited": json.loads(edits.read_text(encoding="utf-8")) if edits.exists() else None,
        }

    @app.get("/api/audio/{content_id}")
    def audio(content_id: str) -> FileResponse:
        path = WORK_DIR / content_id / "audio.16k.wav"
        if not path.exists():
            raise HTTPException(404, "no extracted audio")
        return FileResponse(str(path), media_type="audio/wav")

    @app.post("/api/result/{content_id}/regate")
    def regate(content_id: str, req: RegateRequest) -> dict[str, Any]:
        """Re-run gating and cue building from the sidecar.

        No GPU and no model: this is what makes threshold tuning interactive.
        """
        from .. import cues as cues_mod
        from .. import gating, resegment
        from ..pipeline import _segments_from_sidecar
        from ..write import load_sidecar

        data = load_sidecar(_sidecar(content_id))
        segments = _segments_from_sidecar(data)
        if not segments:
            raise HTTPException(400, "sidecar has no segments")

        cfg = build_config({
            "gating": req.gating,
            "cues": req.cues,
            "keep_suppressed": req.keep_suppressed,
        })

        if cfg.asr.resegment:
            segments = resegment.split(
                segments,
                max_silence=cfg.asr.resegment_max_silence,
                max_duration=cfg.asr.resegment_max_duration,
            )
        stats = gating.apply(segments, cfg.gating)
        built = cues_mod.build(
            segments, cfg.cues, include_suppressed=not cfg.gating.drop_suppressed
        )

        return {
            "cues": [c.to_dict() for c in built],
            "segments": [s.to_dict() for s in segments],
            "stats": {
                "total": stats.total,
                "kept": stats.kept,
                "suppressed": stats.suppressed,
                "reasons": dict(stats.reasons),
                "summary": stats.summary(),
            },
        }

    @app.post("/api/result/{content_id}/save")
    def save(content_id: str, req: SaveRequest) -> dict[str, Any]:
        """Write edited cues to subtitle files and persist the edits."""
        from ..cues import Cue
        from ..write import load_sidecar, write_subtitles

        data = load_sidecar(_sidecar(content_id))
        source = Path(data["source"]["path"])
        language = req.language or data.get("language") or "und"

        cues = []
        for raw in req.cues:
            lines = raw.get("lines")
            if lines is None:
                lines = [l for l in (raw.get("text") or "").split("\n") if l.strip()]
            if not lines:
                continue
            cues.append(
                Cue(
                    start=float(raw["start"]),
                    end=float(raw["end"]),
                    lines=lines,
                    warnings=list(raw.get("warnings") or []),
                )
            )
        cues.sort(key=lambda c: c.start)

        base_dir = _out_dir(req.out_dir) or source.parent
        out_base = base_dir / source.stem
        cfg = Config()
        written = write_subtitles(cues, out_base, req.formats, language, cfg.encoding)
        if req.romanize:
            from ..write import write_romanized

            written += write_romanized(cues, out_base, req.formats, language, cfg.encoding)

        # Persist edits so reopening the file shows the human version.
        edits = WORK_DIR / content_id / "edits.json"
        edits.write_text(
            json.dumps([c.to_dict() for c in cues], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {"written": [str(p) for p in written], "cue_count": len(cues)}

    @app.get("/api/library")
    def library_index(limit: int = 300) -> dict[str, Any]:
        """Every transcript on disk, newest first.

        The job queue is in-memory, so restarting the server used to lose track
        of everything already transcribed — and with it the Translate button,
        which only appears on a finished file. The sidecars persist, so the
        library is read from them instead of from queue state.
        """
        items = [e.to_dict() for e in library.entries(WORK_DIR, limit)]
        return {"items": items, "bytes": sum(i["size"] for i in items)}

    @app.delete("/api/library/{content_id}")
    def library_forget(content_id: str) -> dict[str, Any]:
        """Forget one file: delete the transcript and audio the app cached.

        The subtitle files next to the video are left alone — they are what the
        user came here for. This removes the copy the app kept.
        """
        try:
            result = library.forget(WORK_DIR, content_id, queue.active_paths())
        except library.LibraryError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"ok": True, **result}

    @app.post("/api/library/forget-all")
    def library_forget_all(req: ForgetAllRequest) -> dict[str, Any]:
        """Forget everything, including runs that never finished.

        An interrupted transcription leaves extracted audio with no sidecar: not
        listed anywhere, still a recording of someone.

        The explicit `confirm` is the wire-level guard on an irreversible bulk
        delete: the two clicks in the UI stop a person doing this by accident,
        and this stops a bare POST — a replay, a stray fetch, a curl in the wrong
        window — doing it without one.
        """
        if not req.confirm:
            raise HTTPException(400, "forgetting everything needs confirm: true")
        removed = library.forget_all(WORK_DIR, queue.active_paths())
        log.warning(
            "forget-all: removed %d cache entries, freed %d bytes, kept %s",
            removed["removed"], removed["freed"], removed["kept"] or "nothing",
        )
        return {"ok": True, **removed}

    # --------------------------------------------- online translation (opt-in)

    @app.get("/api/translate/providers")
    def translate_providers() -> dict[str, Any]:
        from .. import online

        from .. import translate as mt

        # Ask DeepL what it supports when there is a key to ask with; a built-in
        # list was two years stale and hid languages the service had gained.
        deepl_targets = sorted(online.DEEPL_FALLBACK_TARGETS)
        keys = online.load_keys()
        if keys.deepl:
            try:
                deepl_targets = sorted(
                    online.DeepLTranslator(keys.deepl, keys.deepl_plan).targets()
                )
            except online.TranslationError:
                pass

        return {
            "configured": online.configured(),
            "keys_path": str(online.keys_path()),
            "key_source": settings.load_or_default().key_source,
            "deepl_targets": deepl_targets,
            # What each engine can actually translate into, so the language list
            # can stop offering choices that are then refused. Google's coverage
            # is broader than every language this app lists, so it gets no
            # restriction rather than a list that would go stale.
            "targets": {
                "deepl": deepl_targets,
                "local": sorted(mt.NLLB_CODES),
                "google": [],          # empty means "no restriction"
            },
        }

    @app.post("/api/translate/keys")
    def set_keys(req: KeysRequest) -> dict[str, Any]:
        """Store API keys in the gitignored settings file.

        Only what the request names is written, so a key supplied through the
        environment is not quietly copied onto disk, and everything else in the
        file — comments included — is left alone.
        """
        from .. import online, settings as user_settings

        path = user_settings.update_api_keys(
            google=req.google.strip() if req.google is not None else None,
            deepl=req.deepl.strip() if req.deepl is not None else None,
            deepl_plan=req.deepl_plan if req.deepl_plan in ("free", "pro") else None,
        )
        return {"ok": True, "path": str(path), "configured": online.configured()}

    @app.post("/api/translate/default")
    def set_translate_default(req: TranslateDefaultRequest) -> dict[str, Any]:
        """Persist "always translate" into settings.local.yaml.

        Written to the same file the user edits by hand, so the choice survives a
        restart and is visible where every other default lives — rather than
        hiding in browser storage where nobody would find it again.
        """
        from .. import online

        values: dict[str, Any] = {"defaults.translate.auto": req.auto}
        if req.provider:
            if req.provider not in ("google", "deepl", "local"):
                raise HTTPException(400, f"unknown provider {req.provider!r}")
            values["defaults.translate.provider"] = req.provider
        # Refuse a combination that cannot run, rather than storing it and
        # failing on every file from now on. Checked against DeepL's own list
        # when there is a key, so a language it has gained is not refused here.
        if req.provider == "deepl" and req.target:
            keys = online.load_keys()
            reachable = (
                online.DeepLTranslator(keys.deepl, keys.deepl_plan).can_target(req.target)
                if keys.deepl
                else online.DeepLTranslator.supports(req.target)
            )
            if not reachable:
                raise HTTPException(
                    400,
                    f"DeepL does not translate into {req.target!r} — use Google "
                    "for this language.",
                )
        if req.target:
            values["defaults.translate.target"] = req.target
        try:
            path = settings.set_values(values)
        except settings.SettingsError as exc:
            raise HTTPException(400, str(exc)) from exc
        user = settings.load_or_default()
        return {
            "ok": True,
            "path": str(path),
            "auto": user.defaults.translate.auto,
            "provider": user.defaults.translate.provider,
            "target": user.defaults.translate.target,
        }

    @app.post("/api/translate/test")
    def test_key(req: OnlineTranslateRequest) -> dict[str, Any]:
        """Translate one short word, so a bad key fails here and not mid-file."""
        from .. import online

        try:
            provider = online.get_translator(req.provider)
            out = provider.translate_texts(["hello"], "en", "de")
        except online.TranslationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "provider": req.provider, "sample": out[0] if out else ""}

    @app.post("/api/result/{content_id}/translate-online")
    def translate_online(content_id: str, req: OnlineTranslateRequest) -> dict[str, Any]:
        """Translate a stored transcript through Google or DeepL.

        Explicit only: nothing here is reachable from a normal transcription run.
        """
        from .. import cloud, online

        cues, data = _current_cues(content_id)
        source = (data.get("language") or "").lower()
        target = (req.language or "en").lower()

        try:
            provider = cloud.resolve(req.provider, source, target)
        except cloud.NotPossible as exc:
            raise HTTPException(400, str(exc)) from exc

        cfg = Config.load(data.get("config", {}).get("profile") or "home-video")
        try:
            result = cloud.translate(
                cues, provider,
                provider_name=req.provider,
                source_language=source,
                target_language=target,
                source_path=Path(data["source"]["path"]),
                formats=req.formats,
                out_dir=_out_dir(req.out_dir),
                cfg=cfg,
            )
        except cloud.NotPossible as exc:
            raise HTTPException(400, str(exc)) from exc
        except online.TranslationError as exc:
            raise HTTPException(502, str(exc)) from exc
        return result.to_dict()

    # ------------------------------------------------- external translation

    def _current_cues(content_id: str) -> list:
        """Cues as they stand: hand-edited if edits exist, else generated."""
        from ..cues import Cue
        from ..write import load_sidecar

        data = load_sidecar(_sidecar(content_id))
        edits = WORK_DIR / content_id / "edits.json"
        raw = (
            json.loads(edits.read_text(encoding="utf-8"))
            if edits.exists()
            else data.get("cues", [])
        )
        return [
            Cue(
                start=c["start"],
                end=c["end"],
                lines=c.get("lines") or [c.get("text", "")],
                warnings=list(c.get("warnings") or []),
            )
            for c in raw
        ], data

    @app.get("/api/result/{content_id}/export-text")
    def export_text(content_id: str) -> dict[str, Any]:
        """Numbered subtitle text, for pasting into any external translator."""
        from .. import roundtrip

        cues, data = _current_cues(content_id)
        return {
            "text": roundtrip.export_text(cues),
            "cue_count": len(cues),
            "language": data.get("language"),
        }

    @app.post("/api/result/{content_id}/import-translation")
    def import_translation(
        content_id: str, req: ImportTranslationRequest
    ) -> dict[str, Any]:
        """Apply externally translated text onto the original timings."""
        from .. import roundtrip
        from ..write import write_subtitles

        cues, data = _current_cues(content_id)
        if not cues:
            raise HTTPException(400, "no cues to translate")

        translated, report = roundtrip.apply_translation(
            cues, req.text, keep_untranslated=req.keep_untranslated
        )
        if not report.ok:
            raise HTTPException(
                400,
                "Could not match any translated line to a cue. Paste the "
                "translation with its numbers intact, or keep one line per cue.",
            )

        cfg = Config.load(data.get("config", {}).get("profile") or "home-video")
        rebroken = roundtrip.rebreak(translated, cfg.cues)

        source = Path(data["source"]["path"])
        base_dir = _out_dir(req.out_dir) or source.parent
        written = write_subtitles(
            rebroken, base_dir / source.stem, req.formats, req.language, cfg.encoding
        )
        return {
            "written": [str(p) for p in written],
            "cue_count": len(rebroken),
            "matched": report.matched,
            "total": report.total,
            "missing": report.missing,
            "method": report.method,
            "summary": report.summary(),
        }

    # -------------------------------------------------------------- profiles

    @app.post("/api/profile/{name}")
    def save_profile(name: str, req: ProfileRequest) -> dict[str, Any]:
        """Persist tuned thresholds back into the profile YAML."""
        import yaml

        if not name.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(400, "bad profile name")

        path = Path(__file__).parent.parent.parent / "profiles" / f"{name}.yaml"
        if not path.exists():
            raise HTTPException(404, f"no profile {name}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # Validate against the dataclasses before writing, so a bad key from the
        # browser cannot produce a profile that fails to load next run.
        reference = Config()
        for section, incoming in (("gating", req.gating), ("cues", req.cues)):
            target = getattr(reference, section)
            for key in incoming:
                if not hasattr(target, key):
                    raise HTTPException(400, f"unknown {section} option: {key}")
            data.setdefault(section, {}).update(incoming)

        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return {"ok": True, "path": str(path)}

    # ----------------------------------------------------------------- meta

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Identify this process, so a new launch can stop the previous one.

        The pid comes from the server itself rather than from parsing netstat:
        it names exactly the process answering on this port, which is what makes
        it safe to kill. Cheap enough to be probed on twenty ports at startup.
        """
        return {"app": "sgen", "pid": os.getpid(), "assets": _asset_version()}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        from .. import models as registry

        user = settings.load_or_default()
        try:
            cfg = Config.load(user.defaults.profile)
        except FileNotFoundError:
            # A typo in the settings file must not take the whole page down.
            user.error = user.error or (
                f"unknown profile {user.defaults.profile!r} in "
                f"{user.path.name} — using home-video"
            )
            cfg = Config.load("home-video")
        profiles = sorted(
            p.stem for p in (Path(__file__).parent.parent.parent / "profiles").glob("*.yaml")
        )
        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu = {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1)}
        except Exception:
            pass

        return {
            "profiles": profiles,
            "models": {k: v for k, v in registry.available().items() if v},
            "gpu": gpu,
            "defaults": {
                "profile": cfg.profile or "home-video",
                "model": cfg.asr.model,
                "batch_size": cfg.asr.batch_size,
                "beam_size": cfg.asr.beam_size,
                "gating": cfg.gating.__dict__,
                "cues": cfg.cues.__dict__,
                "formats": list(user.defaults.formats or cfg.formats),
                # From settings.local.yaml, so the controls start where this
                # machine wants them rather than where the code does.
                "language": user.defaults.language,
                "hotwords": user.defaults.hotwords,
                "romanize": user.defaults.romanize,
                "romanize_script": user.defaults.romanize_script,
                "keep_suppressed": user.defaults.keep_suppressed,
                "out_dir": user.defaults.out_dir,
                "translate_provider": user.defaults.translate.provider,
                "translate_target": user.defaults.translate.target,
                "translate_auto": user.defaults.translate.auto,
            },
            "settings": {
                "path": str(user.path),
                "exists": user.exists,
                "error": user.error,
            },
            "work_dir": str(WORK_DIR),
        }

    return app
