"""What the app remembers about the files it has transcribed — and how to make
it forget.

Everything the UI shows as "already transcribed" is read from `work/`, one
folder per file, keyed by content id:

    work/06f677d0590710db/
      transcript.sgen.json   the transcript, its timings and the source path
      audio.16k.wav          the extracted audio
      edits.json             hand edits, if any

There is no database and no index. That is deliberate — the same reasoning as
resumability — but it has a consequence worth naming: the sidecar holds the
**full text of what was said** and the path it came from, and the WAV holds the
audio itself. For home video that is a private record sitting in a cache
directory, so removing it has to be a button and not a chore.

Forgetting an entry deletes that folder. It does **not** touch the subtitle
files written next to the video: those are the point of the app, the user asked
for them, and deleting the thing someone came here to produce would be a
surprise. What goes is the copy the app kept for itself.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SIDECAR_NAME = "transcript.sgen.json"
AUDIO_NAME = "audio.16k.wav"
EDITS_NAME = "edits.json"

# The files that make a directory under work/ ours to delete. A folder holding
# anything else is left alone: work/ is inside the project, and "forget
# everything" must not become "delete whatever is in this directory".
ENTRY_MARKERS = (SIDECAR_NAME, AUDIO_NAME, EDITS_NAME)

# Content ids are hex digests, but be liberal in what we accept and strict about
# what it may resolve to — the id arrives from a URL.
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LibraryError(ValueError):
    """The id names nothing we are willing to touch, or the delete failed."""


@dataclass
class Entry:
    """One transcribed file, as the UI needs to show it."""

    content_id: str
    name: str
    path: str
    language: str | None
    cue_count: int
    duration: float
    modified: float
    source_exists: bool
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_id": self.content_id,
            "name": self.name,
            "path": self.path,
            "language": self.language,
            "cue_count": self.cue_count,
            "duration": self.duration,
            "modified": self.modified,
            "source_exists": self.source_exists,
            "size": self.size,
        }


def entry_dir(work: Path, content_id: str) -> Path:
    """Resolve an id to its folder, refusing anything outside `work/`.

    The id comes off a URL, so `..`, an absolute path or a symlink pointing
    elsewhere all have to be dead ends before a recursive delete sees them.
    """
    if not SAFE_ID.match(content_id or ""):
        raise LibraryError(f"not a valid id: {content_id!r}")
    candidate = (work / content_id).resolve()
    if candidate.parent != work.resolve():
        raise LibraryError(f"{content_id!r} is not inside the work folder")
    return candidate


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def read_entry(sidecar: Path) -> Entry | None:
    """One entry from its sidecar, or None if the file is unusable.

    A half-written or hand-mangled sidecar is skipped rather than fatal: one bad
    file must not empty the list.
    """
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("cues"):
        return None
    source = data.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    folder = sidecar.parent
    path = str(source.get("path") or "")
    try:
        modified = sidecar.stat().st_mtime
    except OSError:
        return None
    return Entry(
        content_id=folder.name,
        name=str(source.get("name") or folder.name),
        path=path,
        language=data.get("language"),
        cue_count=len(data.get("cues") or []),
        duration=float(data.get("audio_duration") or 0.0),
        modified=modified,
        source_exists=bool(path) and Path(path).exists(),
        size=_dir_size(folder),
    )


def entries(work: Path, limit: int | None = None) -> list[Entry]:
    """Every readable transcript under `work/`, newest first."""
    if not work.exists():
        return []
    found = [e for e in (read_entry(s) for s in work.glob(f"*/{SIDECAR_NAME}")) if e]
    found.sort(key=lambda e: -e.modified)
    return found[:limit] if limit else found


def is_entry_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in ENTRY_MARKERS)


def entry_dirs(work: Path) -> list[Path]:
    """Cache folders under `work/`, including runs that never finished.

    An interrupted transcription leaves the extracted audio behind with no
    sidecar. It is invisible in the list but it is still a recording of someone,
    so "forget everything" has to include it.
    """
    if not work.exists():
        return []
    return sorted(p for p in work.iterdir() if is_entry_dir(p))


def describe(work: Path, content_id: str) -> Entry | None:
    """What an id refers to, for confirming before deleting it."""
    return read_entry(entry_dir(work, content_id) / SIDECAR_NAME)


def _remove(folder: Path) -> int:
    """Delete a cache folder, returning the bytes reclaimed."""
    size = _dir_size(folder)
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        # Windows keeps a handle on the WAV while a job is decoding it, and
        # ignore_errors would otherwise report a silent success on a folder that
        # is still there.
        raise LibraryError(
            f"could not delete {folder.name} — a file in it is still open. "
            "Wait for the job using it to finish, then try again."
        )
    return size


def forget(work: Path, content_id: str, protected: Iterable[str] = ()) -> dict[str, Any]:
    """Forget one file: delete its cache folder.

    `protected` is the set of source paths currently being transcribed. Pulling
    the work folder out from under a running job would fail the job with an
    obscure error, so it is refused with a clear one instead.
    """
    folder = entry_dir(work, content_id)
    if not folder.exists():
        raise LibraryError(f"nothing here to forget: {content_id}")

    entry = read_entry(folder / SIDECAR_NAME)
    name = entry.name if entry else content_id
    if entry and entry.path and _same_path(entry.path, protected):
        raise LibraryError(f"{name} is being transcribed right now")

    freed = _remove(folder)
    log.info("forgot %s (%s), freed %d bytes", content_id, name, freed)
    return {"content_id": content_id, "name": name, "freed": freed}


def forget_all(work: Path, protected: Iterable[str] = ()) -> dict[str, Any]:
    """Forget everything. Reports what it removed, and what it could not."""
    busy = {_norm(p) for p in protected}
    removed, freed, kept = 0, 0, []
    for folder in entry_dirs(work):
        entry = read_entry(folder / SIDECAR_NAME)
        if entry and entry.path and _norm(entry.path) in busy:
            kept.append(entry.name)
            continue
        try:
            freed += _remove(folder)
            removed += 1
        except LibraryError as exc:
            log.warning("%s", exc)
            kept.append(folder.name)
    return {"removed": removed, "freed": freed, "kept": kept}


def _norm(path: str) -> str:
    """Compare paths the way the filesystem does, not the way strings do."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()


def _same_path(path: str, others: Iterable[str]) -> bool:
    target = _norm(path)
    return any(_norm(other) == target for other in others)
