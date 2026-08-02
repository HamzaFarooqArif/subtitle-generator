"""Model acquisition. This is the only module allowed to use the network."""

from __future__ import annotations

import os
from pathlib import Path

from .config import MODELS_DIR
from .models import CT2_MODELS

# WhisperX's default alignment models for our target languages. English uses a
# torchaudio bundle instead of a hub repo.
ALIGN_REPOS = {
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
}


def _prepare_env() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)
    # Explicitly clear offline flags: this command is meant to fetch.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Xet-backed transfers stalled at zero bytes on large weight files here;
    # the classic resolver is slower but actually completes.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def pull_ct2(name: str, console=None) -> Path:
    from huggingface_hub import snapshot_download

    rel, repo = CT2_MODELS[name]
    if console:
        console.print(f"[cyan]pulling[/] {name} <- {repo}")
    path = snapshot_download(repo_id=repo, local_dir=str(MODELS_DIR / rel))
    return Path(path)


def pull_align(lang: str, console=None) -> None:
    from huggingface_hub import snapshot_download

    repo = ALIGN_REPOS[lang]
    if console:
        console.print(f"[cyan]pulling[/] align:{lang} <- {repo}")
    snapshot_download(repo_id=repo)


def pull_english_align(console=None) -> None:
    """torchaudio's WAV2VEC2_ASR_BASE_960H, cached under models/torch."""
    os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
    try:
        import torchaudio

        if console:
            console.print("[cyan]pulling[/] align:en <- torchaudio WAV2VEC2_ASR_BASE_960H")
        torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    except Exception as exc:  # non-fatal; alignment is optional
        if console:
            console.print(f"[yellow]align:en unavailable:[/] {exc}")


def pull_translation(name: str = "nllb-1.3b", console=None) -> Path:
    """Download an NLLB model and convert it to CTranslate2.

    Conversion copies the tokenizer files alongside the converted weights, so the
    runtime needs a single directory. int8_float16 keeps a 1.3B model around
    1.4 GB, which leaves plenty of room on an 8 GB card.
    """
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    from .models import MT_MODELS

    rel, repo = MT_MODELS[name]
    out_dir = MODELS_DIR / rel
    if (out_dir / "model.bin").exists():
        if console:
            console.print(f"[dim]{name} already converted[/]")
        return out_dir

    if console:
        console.print(f"[cyan]pulling[/] {name} <- {repo}")
    source = snapshot_download(repo_id=repo)

    if console:
        console.print(f"[cyan]converting[/] {name} -> CTranslate2 (int8_float16)")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", source,
            "--output_dir", str(out_dir),
            "--quantization", "int8_float16",
            "--copy_files", "tokenizer.json", "sentencepiece.bpe.model",
            "special_tokens_map.json", "tokenizer_config.json",
            "--force",
        ],
        check=True,
    )
    return out_dir


def pull_all(*, include_turbo: bool = True, include_align: bool = True, console=None) -> None:
    _prepare_env()

    wanted = ["large-v3"]
    if include_turbo:
        wanted.append("large-v3-turbo")

    for name in wanted:
        pull_ct2(name, console=console)

    if include_align:
        pull_english_align(console=console)
        for lang in ALIGN_REPOS:
            try:
                pull_align(lang, console=console)
            except Exception as exc:
                if console:
                    console.print(f"[yellow]align:{lang} failed:[/] {exc}")

    if console:
        console.print("[green]done.[/] Models live in " + str(MODELS_DIR))
