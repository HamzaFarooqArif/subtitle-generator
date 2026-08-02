"""Extract normalized 16 kHz mono PCM audio for the ASR stage."""

from __future__ import annotations

from pathlib import Path

from . import ffmpeg
from .config import AudioConfig
from .probe import AudioStream, MediaInfo


def _filters(cfg: AudioConfig, stream: AudioStream) -> str:
    chain: list[str] = []

    if cfg.downmix == "center" and stream.channels >= 5:
        # Only meaningful for 5.1+; consumer recordings have no discrete
        # dialogue channel, so this is off by default.
        chain.append("pan=mono|c0=FC")

    if cfg.normalize == "speechnorm":
        # Per-frame speech leveller. Handles the case this pipeline actually
        # faces: a whisper at -45 dBFS a second after a shout at -6 dBFS.
        # loudnorm would average those together and leave the whisper inaudible
        # to the model.
        chain.append("speechnorm=e=12.5:r=0.0001:l=1")
    elif cfg.normalize == "loudnorm":
        chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    chain.append(f"aresample={cfg.sample_rate}:resampler=soxr:precision=28")
    return ",".join(chain)


def extract_audio(
    info: MediaInfo,
    stream: AudioStream,
    dest: Path,
    cfg: AudioConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Decode `stream` to a mono 16-bit WAV at `dest`."""
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial.wav")

    ffmpeg.run(
        "ffmpeg",
        [
            "-hide_banner", "-nostdin", "-y",
            "-i", str(info.path),
            "-vn", "-sn", "-dn",
            "-map", f"0:{stream.index}",
            "-af", _filters(cfg, stream),
            "-ac", "1",
            "-ar", str(cfg.sample_rate),
            "-c:a", "pcm_s16le",
            str(tmp),
        ],
        capture=False,
    )

    tmp.replace(dest)
    return dest
