"""Write subtitle files and the JSON sidecar."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pysubs2

from .asr import Transcript
from .config import Config
from .cues import Cue

__all__ = [
    "write_subtitles", "write_romanized", "write_sidecar", "load_sidecar",
    "SIDECAR_VERSION",
]

SIDECAR_VERSION = 1


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
        subs.save(str(path), format_=fmt, encoding=encoding)
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
