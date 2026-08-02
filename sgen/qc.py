"""File-level sanity checks.

Every gate in `gating.py` is per-segment, which leaves a hole: a file where
transcription essentially failed can still pass, because the one or two segments
that survived look locally plausible. A 239-second song that produced a single
0.5-second cue was reported as a success — that was the failure this module
exists to catch.

These checks look at the file as a whole and ask whether the result is credible
at all, independent of whether any individual segment looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .asr import Segment
from .config import QcConfig
from .cues import Cue


@dataclass
class Verdict:
    """Whether a transcript is credible as a whole."""

    coverage: float = 0.0             # fraction of audio covered by final cues
    asr_coverage: float = 0.0         # fraction covered before gating
    language_confidence: float = 1.0
    suppressed_fraction: float = 0.0
    segment_count: int = 0
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def suspect(self) -> bool:
        return bool(self.warnings)

    def to_dict(self) -> dict:
        return {
            "suspect": self.suspect,
            "coverage": round(self.coverage, 4),
            "asr_coverage": round(self.asr_coverage, 4),
            "language_confidence": round(self.language_confidence, 4),
            "suppressed_fraction": round(self.suppressed_fraction, 4),
            "segment_count": self.segment_count,
            "warnings": self.warnings,
            "notes": self.notes,
        }


def speech_span(segments: Sequence[Segment]) -> float:
    """Time covered by actual words, not by segment boundaries.

    A batched decode can return a single segment whose start/end straddle the
    whole file while it contains only a handful of words — one real case spanned
    1409 of 1618 seconds and held about 12 seconds of speech. Measuring segment
    spans therefore overstates coverage enormously, and any threshold built on it
    silently passes files that produced almost nothing. Words are the honest
    unit; fall back to the segment only when word timings are absent.
    """
    items: list = []
    for segment in segments:
        if segment.words:
            items.extend(segment.words)
        else:
            items.append(segment)
    return _span(items)


def _span(items: Sequence) -> float:
    """Total covered time, merging overlaps so coverage can't exceed duration."""
    spans = sorted((float(i.start), float(i.end)) for i in items if i.end > i.start)
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start)


def evaluate(
    segments: Sequence[Segment],
    cues: Sequence[Cue],
    duration: float,
    language_confidence: float,
    cfg: QcConfig,
) -> Verdict:
    """Judge whether this transcript is plausible for a file of this length."""
    verdict = Verdict(
        language_confidence=language_confidence,
        segment_count=len(segments),
        duration=duration,
    )
    if duration <= 0:
        return verdict

    # Word-level for ASR output (segment spans lie — see speech_span), plain
    # spans for cues, which are already tight around their words.
    verdict.asr_coverage = speech_span(segments) / duration
    verdict.coverage = _span(cues) / duration
    total = len(segments)
    suppressed = sum(1 for s in segments if s.suppressed)
    verdict.suppressed_fraction = (suppressed / total) if total else 0.0

    if duration >= cfg.min_duration_for_checks:
        if verdict.asr_coverage < cfg.min_coverage:
            # Genuine failure: the model itself found almost no speech.
            verdict.warnings.append("asr_found_little_speech")
            verdict.notes.append(
                f"Speech recognition found only {verdict.asr_coverage:.1%} of the audio. "
                "If this file is music or singing, voice activity detection is the "
                "likely cause — try the 'music' profile, which disables it."
            )
            if verdict.coverage < cfg.min_coverage:
                verdict.warnings.append("low_coverage")
                verdict.notes.append(
                    f"Subtitles cover only {verdict.coverage:.1%} of {duration:.0f}s "
                    "of audio. Most of this file produced nothing."
                )
        elif verdict.coverage < cfg.min_coverage:
            # Speech *was* found; the gate removed it. That is a different
            # problem with a different fix, and saying "produced nothing" here
            # would be wrong — it produced plenty and then discarded it.
            verdict.warnings.append("mostly_gated")
            verdict.notes.append(
                f"Speech was found across {verdict.asr_coverage:.0%} of the audio, but "
                f"{verdict.suppressed_fraction:.0%} of segments were gated as "
                f"hallucinated or non-lexical, leaving subtitles on only "
                f"{verdict.coverage:.0%}. For audio that is largely non-speech "
                "vocalization this may be correct — verify in Tune gate, or re-run "
                "with --keep-suppressed, before assuming speech was lost."
            )

        if total and total <= cfg.min_segments and duration > 60:
            verdict.warnings.append("too_few_segments")
            verdict.notes.append(
                f"Only {total} segment(s) for {duration:.0f}s of audio."
            )

    if language_confidence and language_confidence < cfg.min_language_confidence:
        verdict.warnings.append("low_language_confidence")
        verdict.notes.append(
            f"Language was detected at only {language_confidence:.0%} confidence. "
            "Pin the language explicitly — a wrong guess produces confident nonsense."
        )

    if total and verdict.suppressed_fraction > cfg.max_suppressed_fraction:
        verdict.warnings.append("heavily_gated")
        verdict.notes.append(
            f"{verdict.suppressed_fraction:.0%} of segments were suppressed as "
            "hallucinated or non-lexical. Review with --keep-suppressed if that "
            "seems too aggressive."
        )

    return verdict
