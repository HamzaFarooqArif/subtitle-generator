"""Speech recognition via faster-whisper / CTranslate2."""

from __future__ import annotations

import gc
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import models
from .config import AsrConfig

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text,
                "probability": self.probability}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    temperature: float = 0.0
    language: str | None = None
    # Populated by the gating stage.
    suppressed: bool = False
    suppress_reason: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def mean_word_prob(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "language": self.language,
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "compression_ratio": self.compression_ratio,
            "temperature": self.temperature,
            "mean_word_prob": self.mean_word_prob,
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model: str

    @property
    def kept(self) -> list[Segment]:
        return [s for s in self.segments if not s.suppressed]


def _filter_kwargs(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the installed faster-whisper version does not accept.

    The batched and sequential transcribe signatures differ, and both have
    churned across releases. Filtering beats pinning an exact version.
    """
    try:
        allowed = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return kwargs
    dropped = sorted(set(kwargs) - allowed)
    if dropped:
        log.debug("dropping unsupported transcribe kwargs: %s", ", ".join(dropped))
    return {k: v for k, v in kwargs.items() if k in allowed}


class Recognizer:
    """Owns the ASR model. Load once, reuse across a whole batch."""

    def __init__(self, cfg: AsrConfig):
        from . import cuda

        cuda.prepare()  # must precede the ctranslate2 import on Windows
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.cfg = cfg
        self.model_path = models.ct2_path(cfg.model)
        log.info("loading %s (%s) on %s", cfg.model, cfg.compute_type, cfg.device)
        self._model = WhisperModel(
            self.model_path,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        self._batched = BatchedInferencePipeline(model=self._model) if cfg.batched else None

    def close(self) -> None:
        """Release VRAM. Required before loading an alignment model on 8 GB."""
        self._batched = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _kwargs(self, language: str | None, task: str = "transcribe") -> dict[str, Any]:
        cfg = self.cfg
        return {
            "language": language,
            "task": task,
            "beam_size": cfg.beam_size,
            "best_of": cfg.best_of,
            "patience": cfg.patience,
            "temperature": list(cfg.temperature),
            "compression_ratio_threshold": cfg.compression_ratio_threshold,
            "log_prob_threshold": cfg.log_prob_threshold,
            "no_speech_threshold": cfg.no_speech_threshold,
            "condition_on_previous_text": cfg.condition_on_previous_text,
            "repetition_penalty": cfg.repetition_penalty,
            "word_timestamps": cfg.word_timestamps,
            "hotwords": cfg.hotwords,
            "vad_filter": cfg.vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": cfg.vad_min_silence_ms,
                "speech_pad_ms": cfg.vad_speech_pad_ms,
            },
        }

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        clip_offset: float = 0.0,
        progress: Callable[[float], None] | None = None,
        vad_filter: bool | None = None,
        force_sequential: bool = False,
        task: str = "transcribe",
    ) -> Transcript:
        """Transcribe a WAV. `clip_offset` is added to all timings.

        `progress` is called with a 0..1 fraction as segments arrive. The
        underlying call is a generator, so consuming it lazily is what makes
        live progress possible at all.
        """
        lang = language or self.cfg.language
        kwargs = self._kwargs(lang, task=task)
        if vad_filter is not None:
            kwargs["vad_filter"] = vad_filter

        # BatchedInferencePipeline derives its work units from VAD and raises
        # "No clip timestamps found" if VAD is off, so decoding the whole file
        # unconditionally requires the sequential path.
        use_batched = (
            self._batched is not None
            and not force_sequential
            and kwargs.get("vad_filter", True)
        )
        if use_batched:
            kwargs["batch_size"] = self.cfg.batch_size
            target = self._batched.transcribe
        else:
            target = self._model.transcribe

        segments_iter, info = target(str(audio), **_filter_kwargs(target, kwargs))
        total = float(getattr(info, "duration", 0.0) or 0.0)

        segments = []
        for segment in _convert(segments_iter, clip_offset, getattr(info, "language", lang)):
            segments.append(segment)
            if progress and total > 0:
                progress(min(1.0, (segment.end - clip_offset) / total))
        if progress:
            progress(1.0)

        return Transcript(
            segments=segments,
            language=getattr(info, "language", lang) or "unknown",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            model=self.cfg.model,
        )

    def detect_language(
        self, audio: Path, *, segments: int = 8
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Identify the language, sampling across the file.

        Returns (language, probability, alternatives). VAD is deliberately left
        OFF: on music or singing it rejects nearly everything, which would leave
        detection guessing from a fraction of a second — the failure that
        produced a Turkish transcript for a Hindi song.

        The probability matters as much as the label. A confident-looking result
        at 21% is a coin flip and callers must be able to see that.
        """
        from faster_whisper.audio import decode_audio

        samples = decode_audio(str(audio), sampling_rate=16_000)
        language, probability, alternatives = self._model.detect_language(
            audio=samples,
            vad_filter=False,
            language_detection_segments=segments,
            language_detection_threshold=0.5,
        )
        return language, float(probability), [(l, float(p)) for l, p in alternatives[:5]]


def _convert(raw: Iterable, offset: float, language: str | None) -> Iterable[Segment]:
    for s in raw:
        words = [
            Word(
                start=float(w.start) + offset,
                end=float(w.end) + offset,
                text=w.word,
                probability=float(getattr(w, "probability", 0.0) or 0.0),
            )
            for w in (getattr(s, "words", None) or [])
        ]
        yield Segment(
            start=float(s.start) + offset,
            end=float(s.end) + offset,
            text=(s.text or "").strip(),
            words=words,
            avg_logprob=float(getattr(s, "avg_logprob", 0.0) or 0.0),
            no_speech_prob=float(getattr(s, "no_speech_prob", 0.0) or 0.0),
            compression_ratio=float(getattr(s, "compression_ratio", 1.0) or 1.0),
            temperature=float(getattr(s, "temperature", 0.0) or 0.0),
            language=language,
        )
