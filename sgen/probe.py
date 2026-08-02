"""Inspect a media file: audio streams, duration, stable content id."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg

# Tracks we never want to transcribe even when they are the only tagged option.
_SKIP_TITLE_TOKENS = ("commentary", "description", "audiodescription", "sap")


@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int
    layout: str | None
    language: str | None
    title: str | None
    default: bool

    @property
    def is_undesirable(self) -> bool:
        title = (self.title or "").lower()
        return any(token in title for token in _SKIP_TITLE_TOKENS)


@dataclass
class MediaInfo:
    path: Path
    duration: float
    audio_streams: list[AudioStream]
    content_id: str

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_streams)


def content_id(path: Path, chunk: int = 8 << 20) -> str:
    """Cheap stable id: size plus head and tail bytes.

    Hashing multi-gigabyte video in full would dominate runtime, and this is
    only used to key the work/ cache.
    """
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(chunk))
        if size > chunk * 2:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()[:16]


def probe(path: Path) -> MediaInfo:
    proc = ffmpeg.run(
        "ffprobe",
        [
            "-v", "error",
            "-show_entries", "format=duration",
            "-show_streams", "-select_streams", "a",
            "-of", "json",
            str(path),
        ],
    )
    data = json.loads(proc.stdout)

    duration = float(data.get("format", {}).get("duration") or 0.0)
    streams: list[AudioStream] = []
    for stream in data.get("streams", []):
        tags = stream.get("tags", {}) or {}
        disposition = stream.get("disposition", {}) or {}
        streams.append(
            AudioStream(
                index=int(stream["index"]),
                codec=stream.get("codec_name", "?"),
                channels=int(stream.get("channels") or 0),
                layout=stream.get("channel_layout"),
                language=(tags.get("language") or tags.get("LANGUAGE") or "").lower() or None,
                title=tags.get("title"),
                default=bool(disposition.get("default")),
            )
        )

    if duration <= 0 and streams:
        # Some phone-camera containers omit format duration; fall back to a decode.
        proc = ffmpeg.run(
            "ffprobe",
            ["-v", "error", "-select_streams", f"a:0", "-show_entries",
             "stream=duration", "-of", "csv=p=0", str(path)],
        )
        try:
            duration = float(proc.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            duration = 0.0

    return MediaInfo(
        path=path,
        duration=duration,
        audio_streams=streams,
        content_id=content_id(path),
    )


def choose_stream(info: MediaInfo, prefer_language: str | None = None) -> AudioStream:
    """Pick the dialogue track.

    Home video normally has exactly one track, so this mostly matters for the
    occasional file that came off a camcorder with a second mic channel.
    """
    if not info.audio_streams:
        raise ValueError(f"{info.path.name} has no audio stream")

    candidates = [s for s in info.audio_streams if not s.is_undesirable] or info.audio_streams

    def score(stream: AudioStream) -> tuple:
        return (
            prefer_language is not None and stream.language == prefer_language,
            stream.default,
            stream.channels,
            -stream.index,
        )

    return max(candidates, key=score)
