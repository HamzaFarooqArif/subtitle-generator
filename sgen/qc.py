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

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .asr import Segment
from .config import QcConfig
from .cues import Cue

# Writing systems, for deciding whether a transcript is written in one alphabet.
#
# Deliberately not translit.SCRIPT_RANGES: that table answers "which script can I
# romanize", so it excludes Latin by design and stops at the scripts with a
# transliteration scheme. This one has to see everything a decode might emit,
# Latin and Arabic included, because the point is to notice a letter that does
# not belong.
_SCRIPTS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))),
    ("Greek", ((0x0370, 0x03FF),)),
    ("Cyrillic", ((0x0400, 0x052F),)),
    ("Hebrew", ((0x0590, 0x05FF),)),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("Devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("Bengali", ((0x0980, 0x09FF),)),
    ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    ("Gujarati", ((0x0A80, 0x0AFF),)),
    ("Oriya", ((0x0B00, 0x0B7F),)),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    ("Kannada", ((0x0C80, 0x0CFF),)),
    ("Malayalam", ((0x0D00, 0x0D7F),)),
    ("Sinhala", ((0x0D80, 0x0DFF),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
    ("Japanese", ((0x3040, 0x30FF),)),
    ("Han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))),
    ("Hangul", ((0x1100, 0x11FF), (0xAC00, 0xD7AF))),
)


def script_of_char(ch: str) -> str | None:
    """Which writing system a character belongs to, or None if it is neutral.

    Digits, spaces and punctuation are neutral: they appear in every script and
    say nothing about which one a line is written in.
    """
    cp = ord(ch)
    for name, ranges in _SCRIPTS:
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return None


def scripts_in(text: str) -> Counter:
    """Character counts per writing system."""
    return Counter(s for s in (script_of_char(c) for c in text) if s)


def _word_scripts(word: str) -> set[str]:
    return {s for s in (script_of_char(c) for c in word) if s}


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


def _sample(words: Sequence[str], limit: int = 6) -> str:
    shown = list(dict.fromkeys(words))[:limit]
    more = "" if len(set(words)) <= limit else ", …"
    return ", ".join(f"“{w}”" for w in shown) + more


def check_scripts(cues: Sequence[Cue], verdict: Verdict, cfg: QcConfig) -> None:
    """Flag a transcript that is not written in one alphabet.

    Per-segment gating cannot see this: every metric it has is acoustic, so a
    decode that emitted an English word in the middle of a Punjabi song, or spelt
    one letter of a Gurmukhi word in Devanagari, scores perfectly well. Two real
    cases passed with `suspect: false` and no warnings — one Punjabi file
    containing “shipped” and “तੇਰੇ”, and one that mixed Gurmukhi, Devanagari and
    Cyrillic throughout. Both were obvious on sight and invisible to the checks.
    """
    words = [w for cue in cues for line in cue.lines for w in line.split() if w.strip()]
    if not words:
        return

    # A word spelt in two alphabets at once. No orthography does this, so unlike
    # the checks below there is no legitimate case to make room for.
    mixed = [w for w in words if len(_word_scripts(w)) > 1]
    if mixed:
        verdict.warnings.append("mixed_script_words")
        verdict.notes.append(
            f"{len(mixed)} word(s) are spelt in two alphabets at once: {_sample(mixed)}. "
            "No language writes a word that way, so this is a decode failure rather "
            "than unusual spelling."
        )

    counts = scripts_in(" ".join(words))
    if not counts:
        return
    dominant = counts.most_common(1)[0][0]

    # Whole words in another alphabet. A handful is corruption; a lot is someone
    # genuinely switching language, which is not this module's business — so this
    # fires only on the sprinkle.
    foreign = [
        w for w in words
        if (scripts := _word_scripts(w)) and len(scripts) == 1 and dominant not in scripts
    ]
    limit = max(1, math.ceil(cfg.max_foreign_word_fraction * len(words)))
    if foreign and len(foreign) <= limit:
        verdict.warnings.append("foreign_script_words")
        verdict.notes.append(
            f"{len(foreign)} of {len(words)} words are in a different alphabet from "
            f"the rest of the file: {_sample(foreign)}. A few stray words usually "
            "means the decoder guessed rather than heard. More than "
            f"{cfg.max_foreign_word_fraction:.0%} would be treated as genuinely "
            "mixed speech and not reported."
        )

    # Two alphabets in quantity, neither of them Latin. Latin is excluded because
    # it mixes legitimately with everything — Hinglish, brand names, place names.
    letters = sum(n for s, n in counts.items() if s != "Latin")
    substantial = sorted(
        s for s, n in counts.items()
        if s != "Latin" and letters and n / letters >= 0.02
    )
    if len(substantial) > 1:
        verdict.warnings.append("several_scripts")
        verdict.notes.append(
            f"The transcript is written in {len(substantial)} different alphabets "
            f"({', '.join(substantial)}). One language is written in one of them, so "
            "the language was probably detected wrongly — pin it and run again."
        )


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

    if cfg.check_scripts:
        check_scripts(cues, verdict, cfg)

    if total and verdict.suppressed_fraction > cfg.max_suppressed_fraction:
        verdict.warnings.append("heavily_gated")
        verdict.notes.append(
            f"{verdict.suppressed_fraction:.0%} of segments were suppressed as "
            "hallucinated or non-lexical. Review with --keep-suppressed if that "
            "seems too aggressive."
        )

    return verdict
