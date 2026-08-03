"""Job queue for the UI.

A single worker thread owns the GPU and the ASR model, exactly as the CLI batch
path does — two concurrent transcriptions on an 8 GB card would fragment VRAM
and lose more than they gain. The queue serializes work; the UI stays responsive
because it only ever reads state.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..config import Config

log = logging.getLogger(__name__)

Status = Literal["queued", "running", "paused", "done", "failed", "cancelled"]


class Cancelled(Exception):
    """Raised inside the worker when the user stops the file being processed.

    Thrown from the progress callback, which the pipeline calls once per decoded
    segment — so the decode stops within a second or so, at a point where nothing
    is half-written. Subtitles are written atomically, so there is no partial file
    to clean up either.
    """

STAGES = ("probe", "extract", "detect", "transcribe", "gate", "translate", "write")
# Rough share of wall time per stage, for a single overall progress number.
# "translate" is an optional second decode pass, so it carries real weight.
STAGE_WEIGHTS = {
    "probe": 0.02,
    "extract": 0.06,
    "detect": 0.04,
    "transcribe": 0.56,
    "gate": 0.02,
    "translate": 0.26,
    "write": 0.04,
}


@dataclass
class Job:
    id: str
    source: Path
    out_dir: Path | None
    options: dict[str, Any]
    status: Status = "queued"
    stage: str = ""
    stage_fraction: float = 0.0
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    # Filled in on success
    cloud_note: str = ""      # outcome of an optional cloud translation pass
    content_id: str = ""
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    cue_count: int = 0
    suppressed_count: int = 0
    gate_summary: str = ""
    outputs: list[str] = field(default_factory=list)
    # File-level verdict: whether the result is credible at all.
    suspect: bool = False
    qc_warnings: list[str] = field(default_factory=list)
    qc_notes: list[str] = field(default_factory=list)
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started:
            elapsed = (self.finished or time.time()) - self.started
        return {
            "id": self.id,
            "name": self.source.name,
            "path": str(self.source),
            "status": self.status,
            "stage": self.stage,
            "stage_fraction": round(self.stage_fraction, 3),
            "progress": round(self.progress, 3),
            "error": self.error,
            "elapsed": round(elapsed, 1) if elapsed else None,
            "speed": (
                round(self.duration / elapsed, 1)
                if elapsed and elapsed > 0 and self.duration
                else None
            ),
            "content_id": self.content_id,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "cue_count": self.cue_count,
            "suppressed_count": self.suppressed_count,
            "gate_summary": self.gate_summary,
            "outputs": self.outputs,
            "options": self.options,
            "suspect": self.suspect,
            "qc_warnings": self.qc_warnings,
            "qc_notes": self.qc_notes,
            "coverage": round(self.coverage, 4),
            "cloud_note": self.cloud_note,
        }


class JobQueue:
    """Thread-safe queue with a single GPU worker."""

    def __init__(self, on_change: Callable[[Job], None] | None = None):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._pending: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._cancelled: set[str] = set()
        # Set means "running". Cleared holds the worker between segments, so a
        # pause stops the file being processed rather than only the queue.
        self._resume = threading.Event()
        self._resume.set()
        self._on_change = on_change
        # The loaded pipeline, kept alive between jobs when options match.
        self._pipeline = None
        self._pipeline_key: str | None = None

    # ---------------------------------------------------------------- public

    def submit(self, source: Path, out_dir: Path | None, options: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], source=source, out_dir=out_dir, options=options)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._pending.put(job.id)
        self._notify(job)
        self._ensure_worker()
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[i].to_dict() for i in self._order]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Stop a job, queued or running.

        A queued job is dropped outright. A running one is asked to stop and does
        so at the next segment boundary — cooperative rather than killed, because
        tearing down a decode mid-CUDA-call is what leaves the card in a state the
        next job inherits.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status in ("done", "failed", "cancelled"):
                return False
            self._cancelled.add(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                job.finished = time.time()
                self._notify(job)
            else:
                # Leave the status alone: the worker sets it when it actually
                # stops, so the UI never claims it stopped before it has.
                self._resume.set()      # a paused job has to wake up to notice
            return True

    # ------------------------------------------------------------------ pause

    def pause(self) -> None:
        """Hold the worker. The file being processed stops where it is, keeps its
        progress, and continues from there on resume."""
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def _hold(self, job: Job | None = None) -> None:
        """Block while paused. Called between segments, so a pause takes effect
        inside a file rather than only between files."""
        if self._resume.is_set():
            return
        if job is not None and job.status == "running":
            job.status = "paused"
            self._notify(job)
        while not self._resume.wait(timeout=0.5):
            if self._stop.is_set():
                return
        if job is not None and job.status == "paused":
            job.status = "running"
            self._notify(job)

    def active_paths(self) -> list[str]:
        """Sources of jobs not yet finished.

        Their work folders are in use: deleting one mid-run would fail the job
        with a confusing error somewhere deep in the pipeline.
        """
        with self._lock:
            return [
                str(self._jobs[i].source)
                for i in self._order
                if self._jobs[i].status in ("queued", "running", "paused")
            ]

    def clear_finished(self) -> int:
        with self._lock:
            remove = [
                i for i in self._order
                if self._jobs[i].status in ("done", "failed", "cancelled")
            ]
            for i in remove:
                self._jobs.pop(i, None)
                self._order.remove(i)
            return len(remove)

    def shutdown(self) -> None:
        self._stop.set()
        self._resume.set()     # a paused worker has to wake up to shut down
        self._pending.put("")  # unblock the worker
        if self._worker:
            self._worker.join(timeout=10)
        self._release_pipeline()

    # --------------------------------------------------------------- worker

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run, name="sgen-gpu-worker", daemon=True
            )
            self._worker.start()

    def _notify(self, job: Job) -> None:
        if self._on_change:
            try:
                self._on_change(job)
            except Exception:
                log.debug("progress listener failed", exc_info=True)

    def _release_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:
                log.debug("pipeline close failed", exc_info=True)
            self._pipeline = None
            self._pipeline_key = None

    def _pipeline_for(self, options: dict[str, Any]):
        """Reuse the loaded model when ASR settings are unchanged."""
        from ..pipeline import Pipeline

        cfg = build_config(options)
        key = "|".join(
            str(getattr(cfg.asr, field))
            for field in ("model", "compute_type", "device", "batched", "batch_size")
        )
        if self._pipeline is not None and self._pipeline_key == key:
            # Same model: swap in the new config so per-job gate/cue settings
            # take effect without paying to reload weights.
            self._pipeline.cfg = cfg
            return self._pipeline

        self._release_pipeline()
        self._pipeline = Pipeline(cfg)
        self._pipeline_key = key
        return self._pipeline

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._pending.get(timeout=1.0)
            except queue.Empty:
                # Idle: let go of the VRAM so other apps can use the card.
                self._release_pipeline()
                continue

            if not job_id or self._stop.is_set():
                break
            self._hold()          # paused: do not start the next file either
            if self._stop.is_set():
                break
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job_id in self._cancelled:
                    continue

            self._execute(job)

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        job.stage = "probe"
        self._notify(job)

        last_push = 0.0

        def progress(stage: str, fraction: float) -> None:
            nonlocal last_push
            # The pipeline calls this once per decoded segment, which makes it the
            # natural place to stop or hold: often enough to feel immediate, and
            # always between units of work rather than inside one.
            self._hold(job)
            if job.id in self._cancelled:
                raise Cancelled()
            job.stage = stage
            job.stage_fraction = fraction
            job.progress = _overall(stage, fraction)
            # Throttle: transcribe fires per segment and would flood the stream.
            now = time.time()
            if fraction >= 1.0 or now - last_push > 0.25:
                last_push = now
                self._notify(job)

        try:
            pipeline = self._pipeline_for(job.options)
            result = pipeline.process(
                job.source,
                out_dir=job.out_dir,
                overwrite=bool(job.options.get("overwrite")),
                progress=progress,
            )
        except Cancelled:
            log.info("job %s stopped on request", job.id)
            job.status = "cancelled"
            job.stage = ""
            job.finished = time.time()
            self._notify(job)
            return
        except Exception as exc:
            log.exception("job %s failed", job.id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished = time.time()
            job.progress = 1.0
            self._notify(job)
            return

        job.content_id = result.content_id
        job.language = result.language
        job.language_probability = result.language_probability
        job.duration = result.duration
        job.cue_count = len(result.cues)
        job.suppressed_count = result.suppressed_count
        job.gate_summary = result.gate_summary
        job.outputs = [str(p) for p in result.outputs]

        # Cloud translation, only if it was asked for in Settings. Deliberately
        # after the subtitles are on disk: the transcription is the valuable
        # part, and a rejected API key must not lose it.
        if cloud_provider(job.options):
            job.stage = "translate"
            job.stage_fraction = 0.5
            self._notify(job)
            note, extra = self._cloud_translate(job, result)
            job.cloud_note = note
            job.outputs.extend(str(p) for p in extra)

        job.status = "done"
        job.progress = 1.0
        job.stage = "done"
        job.finished = time.time()
        if result.verdict:
            job.suspect = result.verdict.suspect
            job.qc_warnings = list(result.verdict.warnings)
            job.qc_notes = list(result.verdict.notes)
            job.coverage = result.verdict.coverage
        self._notify(job)


    def _cloud_translate(self, job: Job, result) -> tuple[str, list]:
        """Translate the finished subtitles through Google or DeepL.

        Never raises: the transcription already succeeded, so a translation
        failure is a note on an otherwise good result, not a failed job. The
        note is shown on the file's row — failing silently would look exactly
        like the setting doing nothing.
        """
        from .. import cloud, online

        provider_name = cloud_provider(job.options)
        target = (job.options.get("translate_target") or "en").lower()
        source = (result.language or "").lower()
        cfg = build_config(job.options)

        # Nothing to translate, and saying so should not read as a failure. This
        # is the normal case for most home video once "translate automatically"
        # is on: the English files are simply left alone.
        if source and source == target:
            return f"already in {target} — nothing to translate", []

        try:
            provider = cloud.resolve(provider_name, source, target)
            translated = cloud.translate(
                result.cues, provider,
                provider_name=provider_name,
                source_language=source,
                target_language=target,
                source_path=job.source,
                formats=cfg.formats,
                out_dir=job.out_dir,
                cfg=cfg,
            )
        except cloud.NotPossible as exc:
            return f"not translated: {exc}", []
        except online.TranslationError as exc:
            log.warning("cloud translation failed for %s: %s", job.source.name, exc)
            return f"translation failed: {exc}", []
        except Exception as exc:                      # never lose the transcript
            log.exception("cloud translation crashed")
            return f"translation failed: {type(exc).__name__}: {exc}", []

        return (
            f"translated to {translated.language} via {provider_name} "
            f"({translated.cue_count} subtitles, {translated.characters} characters)",
            translated.written,
        )


def cloud_provider(options: dict[str, Any]) -> str:
    """Which cloud provider this job should use, if any.

    An explicit choice in the request wins. Otherwise `defaults.translate.auto`
    in settings.local.yaml decides, which is what makes "translate anything that
    isn't English" a setting rather than something to remember per run.
    """
    chosen = options.get("cloud_provider")
    if chosen:
        return str(chosen)
    if "cloud_provider" in options:
        return ""            # explicitly none: the UI said "transcribe only"

    from .. import settings

    translate = settings.load_or_default().defaults.translate
    if translate.auto and translate.provider in ("google", "deepl"):
        return translate.provider
    return ""


def _overall(stage: str, fraction: float) -> float:
    done = 0.0
    for name in STAGES:
        if name == stage:
            return min(1.0, done + STAGE_WEIGHTS[name] * fraction)
        done += STAGE_WEIGHTS[name]
    return min(1.0, done)


def build_config(options: dict[str, Any]) -> Config:
    """Turn UI options into a Config.

    Anything the request does not mention falls back to settings.local.yaml, so
    a machine default set once applies to API callers too and not only to the
    controls in the browser.
    """
    from .. import settings

    user = settings.load_or_default()
    cfg = Config.load(options.get("profile") or user.defaults.profile)
    settings.apply_defaults(cfg, user)

    if options.get("model"):
        cfg.asr.model = options["model"]
    if options.get("compute_type"):
        cfg.asr.compute_type = options["compute_type"]
    if options.get("batch_size"):
        cfg.asr.batch_size = int(options["batch_size"])
    if options.get("beam_size"):
        cfg.asr.beam_size = int(options["beam_size"])
    # "" / "auto" means detect. Present-but-empty is a choice the UI made, so
    # only a missing key keeps whatever the settings file asked for.
    if "language" in options:
        cfg.asr.language = (options["language"] or "").strip() or None
    if "hotwords" in options:
        cfg.asr.hotwords = (options["hotwords"] or "").strip() or None
    if options.get("formats"):
        cfg.formats = tuple(options["formats"])
    if "keep_suppressed" in options:
        cfg.gating.drop_suppressed = not bool(options["keep_suppressed"])
    if "romanize" in options:
        cfg.romanize = bool(options["romanize"])
    if options.get("romanize_script") in ("latin", "urdu", "both"):
        cfg.romanize_script = options["romanize_script"]
    # "Translate automatically" with the offline model chosen. The pipeline
    # already skips files that are in the target language.
    if "translate" not in options:
        translate = user.defaults.translate
        cfg.translate_to_english = translate.auto and translate.provider == "local"
    if "translate" in options:
        cfg.translate_to_english = bool(options["translate"])
    if options.get("translate_engine") in ("auto", "nllb", "whisper"):
        cfg.translate_engine = options["translate_engine"]
    if options.get("translate_target"):
        cfg.translate_target = options["translate_target"]
    if "gating" in options and isinstance(options["gating"], dict):
        for key, value in options["gating"].items():
            if hasattr(cfg.gating, key):
                setattr(cfg.gating, key, value)
    if "cues" in options and isinstance(options["cues"], dict):
        for key, value in options["cues"].items():
            if hasattr(cfg.cues, key):
                setattr(cfg.cues, key, value)

    return cfg
