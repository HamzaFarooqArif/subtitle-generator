"""Translate subtitles with an external translator, keeping the timings.

Local translation models do not match a production system like Google Translate,
particularly for Hindi. Rather than pretend otherwise, this exports the subtitle
text in a form any translator accepts, and imports the result back onto the
original cue timings — so the transcription and timing work (which is good) is
kept, and only the translation is outsourced.

Lines are numbered because translators are unreliable about preserving line
counts: they merge short lines, split long ones, and drop empty ones. Numbers
survive that and let each translation be matched back to its own cue. Positional
matching is the fallback when the numbers do not come back.

Note the privacy consequence: pasting text into an online translator sends it to
that service. That is fine for a film or a song, and worth a thought for personal
footage — the transcription itself never leaves the machine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from .cues import Cue

log = logging.getLogger(__name__)

# "1. text", "1) text", "1 - text", "[1] text"
_NUMBERED = re.compile(r"^\s*[\[(]?(\d{1,5})[\])]?\s*[.)\-:–—]?\s+(.*\S)\s*$")


def export_text(cues: Sequence[Cue], numbered: bool = True) -> str:
    """Render cues as one numbered line each, ready to paste into a translator.

    Internal line breaks are flattened to a space: they exist for reading on
    screen, and preserving them would make the translator treat one cue as two.
    """
    lines = []
    for index, cue in enumerate(cues, 1):
        text = " ".join(cue.lines).strip()
        if not text:
            continue
        lines.append(f"{index}. {text}" if numbered else text)
    return "\n".join(lines)


@dataclass
class ImportReport:
    matched: int = 0
    total: int = 0
    missing: list[int] = None            # type: ignore[assignment]
    method: str = "numbered"

    def __post_init__(self) -> None:
        if self.missing is None:
            self.missing = []

    @property
    def ok(self) -> bool:
        return self.matched > 0

    def summary(self) -> str:
        base = f"{self.matched}/{self.total} cues translated ({self.method})"
        if self.missing:
            shown = ", ".join(str(n) for n in self.missing[:8])
            more = "…" if len(self.missing) > 8 else ""
            base += f"; untranslated: {shown}{more}"
        return base


def parse_translation(text: str, expected: int) -> tuple[dict[int, str], str]:
    """Parse translated text into {cue number: text}.

    Returns the mapping and the method used, so the caller can tell the user
    whether numbering survived the round trip or positional matching was needed.
    """
    raw_lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in raw_lines if l]

    mapping: dict[int, str] = {}
    for line in lines:
        match = _NUMBERED.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected:
            # A translator may split one cue across two numbered lines; join.
            body = match.group(2).strip()
            mapping[number] = f"{mapping[number]} {body}".strip() if number in mapping else body

    # Numbering is trustworthy only if most cues came back with a number.
    if len(mapping) >= max(1, int(expected * 0.6)):
        return mapping, "numbered"

    # Fall back to position, which is only valid if the count still matches.
    if len(lines) == expected:
        return {i: l for i, l in enumerate(lines, 1)}, "positional"

    stripped = [
        _NUMBERED.match(l).group(2) if _NUMBERED.match(l) else l for l in lines
    ]
    if len(stripped) == expected:
        return {i: l for i, l in enumerate(stripped, 1)}, "positional"

    return mapping, "numbered (incomplete)"


def apply_translation(
    cues: Sequence[Cue],
    text: str,
    *,
    keep_untranslated: bool = True,
) -> tuple[list[Cue], ImportReport]:
    """Replace cue text with its translation, keeping every timing unchanged."""
    numbered = [(i, c) for i, c in enumerate(cues, 1) if " ".join(c.lines).strip()]
    mapping, method = parse_translation(text, len(numbered))

    report = ImportReport(total=len(numbered), method=method)
    out: list[Cue] = []
    for index, cue in numbered:
        translated = mapping.get(index, "").strip()
        if translated:
            report.matched += 1
            out.append(
                Cue(start=cue.start, end=cue.end, lines=[translated], warnings=[])
            )
        else:
            report.missing.append(index)
            if keep_untranslated:
                out.append(
                    Cue(
                        start=cue.start,
                        end=cue.end,
                        lines=list(cue.lines),
                        warnings=[*cue.warnings, "untranslated"],
                    )
                )
    return out, report


def rebreak(cues: Sequence[Cue], cfg) -> list[Cue]:
    """Re-run line breaking and reading-speed rules on imported text.

    Translated text is a different length from the source, so the original line
    breaks no longer fit. Imported cues arrive as a single line and go back
    through the normal cue builder.
    """
    from . import cues as cues_mod
    from .asr import Segment, Word

    segments: list[Segment] = []
    for cue in cues:
        text = " ".join(cue.lines).strip()
        if not text:
            continue
        tokens = text.split()
        duration = max(0.001, cue.end - cue.start)
        total = sum(len(t) for t in tokens) or 1
        words: list[Word] = []
        cursor = cue.start
        for token in tokens:
            span = duration * (len(token) / total)
            words.append(Word(cursor, min(cue.end, cursor + span), " " + token, 1.0))
            cursor += span
        segments.append(
            Segment(start=cue.start, end=cue.end, text=text, words=words)
        )
    return cues_mod.build(segments, cfg)
