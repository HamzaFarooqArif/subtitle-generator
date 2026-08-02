"""Decide what a folder still needs, by looking only at the files in it.

Batch a folder of fifty videos and the machine will be restarted before it
finishes. The state that survives that has to be the output itself — subtitle
files next to the videos — not a job list in memory or a database that can
disagree with the disk. So this module answers one question per video: *is there
already a complete result for this, given the current settings?*

Three things make that answerable without any bookkeeping:

1. **Outputs are written atomically** (see `write._save_atomically`), so a file
   that exists is a file that finished. Without that, a power cut mid-write
   leaves a plausible-looking truncated subtitle file and the video is skipped
   forever — the exact failure this design is meant to avoid.
2. **The language is in the filename** — `clip.ru.srt`, `clip.en.srt` — so a
   translation that was requested but never produced is visible as an absence.
3. Files that predate atomic writes, or that something else truncated, are
   **validated** rather than trusted: a subtitle with no readable cue counts as
   missing, so it gets produced again.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

from .config import Config, WORK_DIR

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".webm", ".mpg",
    ".mpeg", ".mts", ".m2ts", ".3gp", ".ts", ".vob",
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

State = Literal["done", "pending", "translate", "damaged"]

# "clip.ru.srt" -> ru; "clip.hi-Latn.srt" -> hi-Latn. Anything else is somebody
# else's file and is ignored rather than mistaken for ours.
_LANG_TAG = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,8})?$")


@dataclass
class FileStatus:
    source: Path
    state: State
    reason: str
    languages: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    damaged: list[Path] = field(default_factory=list)
    # Whether the stored transcript is still around, which decides whether a
    # missing translation costs a GPU pass or only an API call.
    has_transcript: bool = False

    @property
    def needs_work(self) -> bool:
        return self.state != "done"

    def to_dict(self) -> dict:
        return {
            "path": str(self.source),
            "name": self.source.name,
            "state": self.state,
            "reason": self.reason,
            "languages": self.languages,
            "outputs": [str(p) for p in self.outputs],
            "damaged": [str(p) for p in self.damaged],
            "has_transcript": self.has_transcript,
        }


@dataclass
class Scan:
    folder: Path
    files: list[FileStatus]

    @property
    def counts(self) -> dict[str, int]:
        counts = {"done": 0, "pending": 0, "translate": 0, "damaged": 0}
        for item in self.files:
            counts[item.state] += 1
        return counts

    @property
    def todo(self) -> list[FileStatus]:
        return [f for f in self.files if f.needs_work]

    def to_dict(self) -> dict:
        return {
            "folder": str(self.folder),
            "total": len(self.files),
            "counts": self.counts,
            "files": [f.to_dict() for f in self.files],
        }


# --------------------------------------------------------------------------- #
# finding media
# --------------------------------------------------------------------------- #

def media_files(folder: Path, recursive: bool = True) -> list[Path]:
    """Every media file in a folder, sorted so a batch is reproducible."""
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    walk = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        p for p in walk
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )


# --------------------------------------------------------------------------- #
# reading what is already there
# --------------------------------------------------------------------------- #

def existing_subtitles(
    source: Path, formats: Sequence[str], out_dir: Path | None = None
) -> dict[str, dict[str, Path]]:
    """Subtitles already written for this source, as {language: {format: path}}.

    Matched by the naming convention this app writes: `<stem>.<lang>.<fmt>`. A
    hand-made `clip.srt` with no language tag is not claimed as ours — deleting
    or counting somebody else's file would be worse than redoing the work.
    """
    directory = out_dir or source.parent
    found: dict[str, dict[str, Path]] = {}
    if not directory.is_dir():
        return found
    stem = source.stem
    for fmt in formats:
        for path in directory.glob(f"{glob_escape(stem)}.*.{fmt}"):
            tag = path.name[len(stem) + 1 : -(len(fmt) + 1)]
            if _LANG_TAG.match(tag):
                found.setdefault(tag, {})[fmt] = path
    return found


def glob_escape(text: str) -> str:
    """Filenames routinely contain [ and ], which glob would read as a class."""
    return re.sub(r"([\[\]])", r"[\1]", text)


def is_complete(path: Path) -> bool:
    """Does this look like a finished subtitle file?

    Cheap structural check, not a parse: at least one timing line, and text after
    the last one. Anything written by this app since atomic saves cannot fail
    this; anything that does fail was truncated by something else and should be
    produced again rather than trusted.
    """
    try:
        if path.stat().st_size == 0:
            return False
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False

    if "-->" not in text:
        return False
    tail = text.rstrip()
    last_arrow = tail.rfind("-->")
    remainder = tail[last_arrow:]
    # After the final timestamp there must be a line of actual text.
    lines = [line for line in remainder.splitlines()[1:] if line.strip()]
    return bool(lines)


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #

def classify(
    source: Path,
    cfg: Config,
    *,
    out_dir: Path | None = None,
    translate_target: str | None = None,
    work_dir: Path | None = None,
) -> FileStatus:
    """Decide whether this file still needs work, from the disk alone.

    `translate_target` is the language a translation was asked for, or None if
    none was. It cannot be inferred from the file: whether to translate is a
    setting, and only the caller knows it.
    """
    formats = tuple(cfg.formats) or ("srt",)
    subtitles = existing_subtitles(source, formats, out_dir)

    damaged: list[Path] = []
    good: dict[str, dict[str, Path]] = {}
    for language, by_format in subtitles.items():
        usable = {f: p for f, p in by_format.items() if is_complete(p)}
        damaged.extend(p for f, p in by_format.items() if f not in usable)
        if usable:
            good[language] = usable

    outputs = [p for by_format in good.values() for p in by_format.values()]
    # Romanized companions are extras, not evidence of a finished transcription.
    spoken = sorted(lang for lang in good if not lang.endswith("-Latn"))

    if damaged:
        return FileStatus(
            source=source, state="damaged",
            reason=f"{len(damaged)} subtitle file(s) look truncated — probably "
                   "interrupted; will be produced again",
            languages=spoken, outputs=outputs, damaged=damaged,
        )

    if not spoken:
        return FileStatus(source=source, state="pending",
                          reason="no subtitles yet", outputs=outputs)

    # Every requested format has to be present for the language that was decoded.
    missing_formats = [
        f for f in formats if any(f not in good[lang] for lang in spoken)
    ]
    if missing_formats:
        return FileStatus(
            source=source, state="pending",
            reason=f"missing {', '.join('.' + f for f in missing_formats)}",
            languages=spoken, outputs=outputs,
        )

    target = (translate_target or "").lower()
    if target and target not in spoken:
        has_transcript = _transcript_exists(source, work_dir)
        return FileStatus(
            source=source, state="translate",
            reason=(
                f"transcribed as {', '.join(spoken)} but no {target} translation"
                + ("" if has_transcript else " (transcript gone, needs a full pass)")
            ),
            languages=spoken, outputs=outputs, has_transcript=has_transcript,
        )

    return FileStatus(
        source=source, state="done",
        reason=f"{', '.join(spoken)} subtitles present",
        languages=spoken, outputs=outputs,
    )


def _transcript_exists(source: Path, work_dir: Path | None) -> bool:
    """Is the stored transcript still available for this source?

    Only an optimisation: with it, a missing translation costs one API call
    instead of a GPU pass. Its absence never makes a file "done".
    """
    root = work_dir or WORK_DIR
    if not root.is_dir():
        return False
    try:
        for sidecar in root.glob("*/transcript.sgen.json"):
            from .write import load_sidecar

            try:
                data = load_sidecar(sidecar)
            except (OSError, ValueError):
                continue
            if Path(data.get("source", {}).get("path", "")) == source:
                return True
    except OSError:
        return False
    return False


def scan_folder(
    folder: Path,
    cfg: Config,
    *,
    out_dir: Path | None = None,
    translate_target: str | None = None,
    recursive: bool = True,
    work_dir: Path | None = None,
) -> Scan:
    """Classify every media file in a folder."""
    files = [
        classify(source, cfg, out_dir=out_dir, translate_target=translate_target,
                 work_dir=work_dir)
        for source in media_files(folder, recursive)
    ]
    return Scan(folder=folder, files=files)


def summarise(scan: Scan) -> str:
    counts = scan.counts
    parts = [
        f"{len(scan.files)} media file(s)",
        f"{counts['done']} already done",
    ]
    if counts["pending"]:
        parts.append(f"{counts['pending']} to transcribe")
    if counts["translate"]:
        parts.append(f"{counts['translate']} needing translation only")
    if counts["damaged"]:
        parts.append(f"{counts['damaged']} interrupted, will be redone")
    return " · ".join(parts)


def pending_paths(scan: Scan) -> list[Path]:
    return [f.source for f in scan.todo]


def iter_media(paths: Iterable[Path], recursive: bool = True) -> list[Path]:
    """Expand a mixed selection of files and folders into media files."""
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(media_files(path, recursive))
        elif path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            out.append(path)
    return out
