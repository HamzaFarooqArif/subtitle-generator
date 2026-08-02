"""Locating and invoking ffmpeg/ffprobe.

winget installs ffmpeg into a versioned Links/package directory that is not on
PATH until the shell restarts, so resolution falls back to known locations.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


_WINGET_GLOBS = [
    Path.home() / "AppData/Local/Microsoft/WinGet/Links",
    Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
    Path("C:/Program Files/ffmpeg/bin"),
    Path("C:/ffmpeg/bin"),
]


@functools.lru_cache(maxsize=4)
def resolve(tool: str) -> str:
    """Return an absolute path to ffmpeg or ffprobe."""
    found = shutil.which(tool)
    if found:
        return found

    exe = f"{tool}.exe"
    for base in _WINGET_GLOBS:
        if not base.exists():
            continue
        direct = base / exe
        if direct.exists():
            return str(direct)
        for candidate in base.rglob(exe):
            return str(candidate)

    raise FfmpegMissing(
        f"Could not find {tool}. Install it with:  winget install Gyan.FFmpeg\n"
        "then restart the shell (or it will be picked up from the WinGet "
        "Links directory automatically)."
    )


def run(tool: str, args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [resolve(tool), *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError(
            f"{tool} failed (exit {proc.returncode}):\n" + "\n".join(tail)
        )
    return proc
