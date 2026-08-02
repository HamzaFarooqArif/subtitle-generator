"""Write subtitle files and the JSON sidecar."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pysubs2

from .asr import Transcript
from .config import Config
from .cues import Cue

__all__ = [
    "write_subtitles", "write_romanized", "write_urdu", "write_second_script",
    "romanize_or_explain", "write_sidecar", "load_sidecar", "SIDECAR_VERSION",
]

SIDECAR_VERSION = 1


def _save_atomically(
    subs: pysubs2.SSAFile, path: Path, fmt: str, encoding: str
) -> None:
    """Write to a temporary file, then rename over the target.

    This is what makes "the file exists, therefore that video is finished" a
    safe thing to believe. Saving in place leaves a truncated but perfectly
    plausible subtitle file if the machine loses power mid-write, and a resumable
    batch would then skip that video forever. `os.replace` is atomic within a
    volume, so a reader sees either the old file or the whole new one.
    """
    temp = path.with_name(path.name + ".part")
    try:
        subs.save(str(temp), format_=fmt, encoding=encoding)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _to_ssafile(cues: Sequence[Cue]) -> pysubs2.SSAFile:
    subs = pysubs2.SSAFile()
    for cue in cues:
        subs.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=cue.start),
                end=pysubs2.make_time(s=cue.end),
                text=cue.text.replace("\n", r"\N"),
            )
        )
    return subs


def write_subtitles(
    cues: Sequence[Cue],
    out_base: Path,
    formats: Sequence[str],
    language: str,
    encoding: str = "utf-8-sig",
) -> list[Path]:
    """Write `out_base.<lang>.<ext>` for each requested format."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    subs = _to_ssafile(cues)
    written: list[Path] = []
    for fmt in formats:
        path = out_base.with_suffix("")
        path = path.parent / f"{path.name}.{language}.{fmt}"
        _save_atomically(subs, path, fmt, encoding)
        written.append(path)
    return written


def write_romanized(
    cues: Sequence[Cue],
    out_base: Path,
    formats: Sequence[str],
    language: str,
    encoding: str = "utf-8-sig",
) -> list[Path]:
    """Write Latin-script copies as `<name>.<lang>-Latn.<ext>`.

    Returns an empty list when the language has no romanization support, so the
    caller does not need to check first.
    """
    from . import translit

    if not translit.supported(language):
        return []

    romanized = [
        Cue(
            start=cue.start,
            end=cue.end,
            lines=translit.romanize_lines(cue.lines, language),
            warnings=list(cue.warnings),
        )
        for cue in cues
    ]
    # BCP-47 style: hi-Latn is Hindi written in Latin script.
    return write_subtitles(romanized, out_base, formats, f"{language}-Latn", encoding)


def write_urdu(
    cues: Sequence[Cue],
    out_base: Path,
    formats: Sequence[str],
    language: str,
    encoding: str = "utf-8-sig",
) -> list[Path]:
    """Write Urdu-script copies as `<name>.<lang>-Arab.<ext>`.

    Hindi only, and empty for anything else — see `sgen.urdu`.
    """
    from . import urdu

    if not urdu.supported(language):
        return []

    converted = [
        Cue(
            start=cue.start,
            end=cue.end,
            lines=urdu.convert_lines(cue.lines),
            warnings=list(cue.warnings),
        )
        for cue in cues
    ]
    # BCP-47 again: hi-Arab is Hindi written in Arabic script, which is Urdu.
    return write_subtitles(converted, out_base, formats, f"{language}-Arab", encoding)


# What each choice is called, for the sentence explaining why it did nothing.
_SCRIPT_NAMES = {
    "latin": ("Latin-script", "Indic scripts and Cyrillic"),
    "urdu": ("Urdu-script", "Hindi"),
}


def write_second_script(
    cues: Sequence[Cue],
    out_base: Path,
    formats: Sequence[str],
    language: str,
    encoding: str = "utf-8-sig",
    script: str = "latin",
) -> tuple[list[Path], list[str]]:
    """Write the requested second script(s), and say why any produced nothing.

    Returning an empty list was enough for the caller but not for the user: a
    ticked box produced no file, no error and no explanation, and the run looked
    identical either way. The notes travel on the QC verdict, which the UI, the
    CLI and the sidecar all already show.
    """
    wanted = ("latin", "urdu") if script == "both" else (script,)
    writers = {"latin": write_romanized, "urdu": write_urdu}

    paths: list[Path] = []
    notes: list[str] = []
    for name in wanted:
        writer = writers.get(name)
        if writer is None:
            notes.append(f"{name!r} is not a script this can write.")
            continue
        written = writer(cues, out_base, formats, language, encoding)
        if written:
            paths.extend(written)
            continue
        label, available = _SCRIPT_NAMES[name]
        notes.append(
            f"{label} subtitles were asked for, but there is none for "
            f"{language!r} — only the original script was written. Available for "
            f"{available}."
        )
    return paths, notes


def romanize_or_explain(
    cues: Sequence[Cue],
    out_base: Path,
    formats: Sequence[str],
    language: str,
    encoding: str = "utf-8-sig",
) -> tuple[list[Path], str]:
    """Latin script only. Kept because it reads better at the one call site that
    wants exactly that."""
    paths, notes = write_second_script(
        cues, out_base, formats, language, encoding, "latin"
    )
    return paths, notes[0] if notes else ""


def write_sidecar(
    path: Path,
    *,
    source: Path,
    content_id: str,
    config: Config,
    transcript: Transcript,
    cues: Sequence[Cue],
    gate_summary: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist everything needed to reformat without re-transcribing."""
    payload = {
        "sidecar_version": SIDECAR_VERSION,
        "source": {"path": str(source), "name": source.name, "content_id": content_id},
        "model": transcript.model,
        "language": transcript.language,
        "language_probability": transcript.language_probability,
        "audio_duration": transcript.duration,
        "gating": gate_summary,
        "config": _jsonable(config.to_dict()),
        "segments": [s.to_dict() for s in transcript.segments],
        "cues": [c.to_dict() for c in cues],
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_sidecar(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
