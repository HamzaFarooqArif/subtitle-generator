"""Resolve model names to local directories.

Everything must resolve without network access. `sgen models pull` populates
these paths once; `sgen models verify` proves the offline property holds.
"""

from __future__ import annotations

from pathlib import Path

from .config import MODELS_DIR

# Friendly name -> (local converted dir, upstream repo used by `models pull`)
CT2_MODELS: dict[str, tuple[str, str]] = {
    "large-v3": ("ct2/large-v3-float16", "Systran/faster-whisper-large-v3"),
    "large-v3-int8": ("ct2/large-v3-int8_float16", "Systran/faster-whisper-large-v3"),
    "large-v3-turbo": ("ct2/large-v3-turbo-float16", "deepdml/faster-whisper-large-v3-turbo-ct2"),
}

# Text translation models, converted to CTranslate2 so they reuse the inference
# engine already installed for ASR. Tokenizer files are copied into the same
# directory during conversion, so one path serves both.
MT_MODELS: dict[str, tuple[str, str]] = {
    "nllb-1.3b": ("ct2/nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-1.3B"),
    "nllb-600m": ("ct2/nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M"),
}


def mt_path(name: str) -> str:
    """Return the local CTranslate2 directory for a translation model."""
    if name not in MT_MODELS:
        candidate = Path(name)
        if candidate.exists():
            return str(candidate)
        raise ModelMissing(f"unknown translation model {name!r}")

    rel, repo = MT_MODELS[name]
    local = MODELS_DIR / rel
    if (local / "model.bin").exists():
        return str(local)
    raise ModelMissing(
        f"Translation model '{name}' is not present locally.\n"
        f"  expected: {local}\n"
        f"Run:  sgen models pull --translation"
    )


def best_mt_model() -> str:
    """Pick the best translation model present, largest first.

    Lets the user ask for translation without knowing model names, and lets a
    bigger model be dropped in later without changing any config.
    """
    for name in ("nllb-1.3b", "nllb-600m"):
        try:
            mt_path(name)
            return name
        except ModelMissing:
            continue
    raise ModelMissing(
        "No translation model is installed.\nRun:  sgen models pull --translation"
    )


def mt_available() -> dict[str, bool]:
    result = {}
    for name in MT_MODELS:
        try:
            mt_path(name)
            result[name] = True
        except ModelMissing:
            result[name] = False
    return result


# Models `sgen models pull` fetches and `sgen models verify` insists on.
# large-v3-int8 is an opt-in VRAM fallback: convert it locally with
#   ct2-transformers-converter --model openai/whisper-large-v3 \
#     --quantization int8_float16 --output_dir models/ct2/large-v3-int8_float16
REQUIRED = ("large-v3", "large-v3-turbo")

# Languages this project targets, per the code-switching requirement.
TARGET_LANGUAGES = ("en", "de", "es")


class ModelMissing(RuntimeError):
    pass


def ct2_path(name: str) -> str:
    """Return a local CT2 model directory, or the bare name for hub cache lookup."""
    if name not in CT2_MODELS:
        # Allow an explicit path or repo id to pass through unchanged.
        candidate = Path(name)
        if candidate.exists():
            return str(candidate)
        return name

    rel, repo = CT2_MODELS[name]
    local = MODELS_DIR / rel
    if (local / "model.bin").exists():
        return str(local)

    # Fall back to a hub snapshot already sitting in the local cache.
    snapshots = MODELS_DIR / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
    if snapshots.is_dir():
        for snap in snapshots.iterdir():
            if (snap / "model.bin").exists():
                return str(snap)

    raise ModelMissing(
        f"Model '{name}' is not present locally.\n"
        f"  expected: {local}\n"
        f"  or a cached snapshot of {repo}\n"
        f"Run:  sgen models pull"
    )


def available() -> dict[str, bool]:
    result = {}
    for name in CT2_MODELS:
        try:
            ct2_path(name)
            result[name] = True
        except ModelMissing:
            result[name] = False
    return result
