"""Make the pip-installed CUDA libraries loadable on Windows.

CTranslate2 >= 4.5 links cuDNN 9. The `nvidia-cudnn-cu12` / `nvidia-cublas-cu12`
wheels put their DLLs inside site-packages rather than on PATH, so importing
ctranslate2 fails with a bare "DLL load failed" or
"Could not locate cudnn_ops64_9.dll" unless those directories are registered
first. This must run before ctranslate2 is imported.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_NVIDIA_PACKAGES = ("cudnn", "cublas", "cuda_runtime", "cufft")
_prepared = False


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for base in sys.path:
        nvidia = Path(base) / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in _NVIDIA_PACKAGES:
            for sub in ("bin", "lib"):
                candidate = nvidia / package / sub
                if candidate.is_dir():
                    dirs.append(candidate)
    # torch ships its own copies of most CUDA DLLs; a useful fallback.
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            dirs.append(torch_lib)
    except ImportError:
        pass
    return dirs


def prepare() -> list[Path]:
    """Register CUDA DLL directories. Idempotent; safe on non-Windows."""
    global _prepared
    registered: list[Path] = []
    if _prepared or sys.platform != "win32":
        _prepared = True
        return registered

    for directory in _candidate_dirs():
        try:
            os.add_dll_directory(str(directory))
            registered.append(directory)
            log.debug("registered CUDA DLL dir: %s", directory)
        except (OSError, AttributeError) as exc:
            log.debug("could not register %s: %s", directory, exc)

    _prepared = True
    return registered
