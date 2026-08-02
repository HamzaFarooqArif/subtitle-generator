"""Configuration objects and profiles.

Everything the pipeline does is driven by a Config instance so that a run is
reproducible from the sidecar JSON alone.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
WORK_DIR = REPO_ROOT / "work"


def enforce_offline() -> None:
    """Hard-disable network access for the model libraries.

    Called at the start of every run. Personal footage should never leave the
    machine, and an accidental hub lookup is the only way that could happen.
    """
    os.environ.setdefault("HF_HOME", str(MODELS_DIR))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    # "speechnorm" tracks wildly varying levels far better than loudnorm on
    # handheld / phone recordings, which is what home video actually is.
    normalize: Literal["speechnorm", "loudnorm", "none"] = "speechnorm"
    # Downmix rather than picking a channel: consumer cameras put dialogue in
    # both channels, so there is no center channel to isolate.
    downmix: Literal["mono", "center"] = "mono"
    keep_wav: bool = True


@dataclass
class AsrConfig:
    model: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    batched: bool = True
    batch_size: int = 8
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = False
    repetition_penalty: float = 1.05
    word_timestamps: bool = True
    hotwords: str | None = None
    language: str | None = None  # None => detect
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 200
    # Split decode windows to sentence granularity before gating, so one
    # aggregate signal cannot discard half a minute of correct speech.
    resegment: bool = True
    resegment_max_silence: float = 0.7
    resegment_max_duration: float = 12.0


@dataclass
class GatingConfig:
    """Thresholds for suppressing non-lexical / hallucinated output.

    Tuned for audio containing breathing, whispering and non-speech
    vocalization, where Whisper's failure mode is inventing fluent text rather
    than returning nothing.
    """

    enabled: bool = True
    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    # A segment failing BOTH of the above is suppressed. Failing either alone
    # is only suppressed past these harder limits:
    hard_no_speech_prob: float = 0.85
    hard_avg_logprob: float = -1.6
    max_compression_ratio: float = 2.4
    min_words_per_second: float = 0.4
    min_span_for_wps_check: float = 2.5
    max_repeat_of_neighbour: int = 2
    min_mean_word_prob: float = 0.35
    drop_suppressed: bool = True  # False => keep them, marked, for review


@dataclass
class QcConfig:
    """File-level plausibility checks.

    Per-segment gating cannot catch a file that failed wholesale, because the
    few segments that survive look locally fine. These thresholds decide when a
    result is not credible as a whole.
    """

    enabled: bool = True
    # Below this fraction of the audio covered by subtitles, something is wrong.
    min_coverage: float = 0.15
    min_segments: int = 1
    min_duration_for_checks: float = 20.0
    min_language_confidence: float = 0.5
    max_suppressed_fraction: float = 0.6
    # If speech detection rejected almost the whole file, decode it again with
    # VAD off. Singing over dense instrumentation reads as non-speech to Silero,
    # which otherwise silently discards the entire file.
    retry_without_vad: bool = True
    retry_coverage_threshold: float = 0.15


@dataclass
class CueConfig:
    max_lines: int = 2
    max_chars_per_line: int = 42
    target_cps: float = 17.0
    max_cps: float = 20.0
    min_duration: float = 0.833
    max_duration: float = 7.0
    min_gap: float = 0.083
    lead_in: float = 0.05
    lead_out: float = 0.15
    max_silence_within_cue: float = 0.7
    min_line_fill: float = 0.4  # reject breaks leaving a line this empty
    # Fold single-word runts into a neighbour. Unpunctuated text (song lyrics
    # especially) otherwise strands cues reading one word.
    merge_short_cues: bool = True
    min_cue_chars: int = 12


@dataclass
class Config:
    profile: str = "home-video"
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    qc: QcConfig = field(default_factory=QcConfig)
    cues: CueConfig = field(default_factory=CueConfig)
    formats: tuple[str, ...] = ("srt", "vtt")
    # UTF-8 *with* BOM by default. The output is correct UTF-8 either way, but
    # without a BOM Windows tools (Notepad, PowerShell's Get-Content, several
    # players) guess cp1252 and render German umlauts and Spanish accents as
    # mojibake. Every target language here has non-ASCII characters, so the BOM
    # earns its keep. Set "utf-8" if a player rejects it.
    encoding: str = "utf-8-sig"
    # Also write the subtitles in a second script, for readers who speak the
    # language but don't read the script it is written in.
    romanize: bool = False
    # Which second script. "latin" is <lang>-Latn.srt (नमस्ते -> "namaste") and
    # works for Indic scripts and Cyrillic. "urdu" is <lang>-Arab.srt and is
    # Hindi only — Hindi and Urdu are one language in two alphabets.
    romanize_script: Literal["latin", "urdu", "both"] = "latin"
    # Also write translated subtitles.
    translate_to_english: bool = False
    # "nllb" translates the transcript text with a dedicated model — clearly
    # better on ordinary speech. "whisper" translates from audio, needs no extra
    # model, and wins on text with no sentence punctuation (sung lyrics), where
    # text translation has no sentence boundaries to work with and rambles.
    # "auto" measures the punctuation and picks. Falls back to whisper if the
    # NLLB model is not installed.
    translate_engine: Literal["auto", "nllb", "whisper"] = "auto"
    # "auto" uses the largest translation model installed.
    translate_model: str = "auto"
    translate_target: str = "en"
    work_dir: Path = WORK_DIR

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, profile: str | None = None) -> "Config":
        cfg = cls()
        if not profile:
            return cfg
        path = REPO_ROOT / "profiles" / f"{profile}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No such profile: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg.profile = profile
        for section in ("audio", "asr", "gating", "qc", "cues"):
            if section in data:
                current = getattr(cfg, section)
                for key, value in data[section].items():
                    if not hasattr(current, key):
                        raise ValueError(f"Unknown {section} option: {key}")
                    if key == "temperature" and isinstance(value, list):
                        value = tuple(value)
                    setattr(current, key, value)
        if "formats" in data:
            cfg.formats = tuple(data["formats"])
        if "encoding" in data:
            cfg.encoding = data["encoding"]
        if "romanize" in data:
            cfg.romanize = bool(data["romanize"])
        if data.get("romanize_script"):
            cfg.romanize_script = str(data["romanize_script"])
        if "translate_to_english" in data:
            cfg.translate_to_english = bool(data["translate_to_english"])
        for key in ("translate_engine", "translate_model", "translate_target"):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg
