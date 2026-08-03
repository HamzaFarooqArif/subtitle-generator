"""Turn word-level ASR output into readable subtitle cues.

Whisper segments are decode windows, not cues: too long, broken at attention
boundaries rather than clause boundaries, and indifferent to reading speed.
Cues are therefore rebuilt from words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .asr import Segment, Word
from .config import CueConfig

# Sentence-final punctuation across the scripts this pipeline sees. Devanagari
# uses the danda, not the full stop — without it, Hindi text looks entirely
# unpunctuated and cues get broken on character count alone, mid-phrase.
TERMINAL = ".!?…।॥۔。！？"
CLAUSE = ",;:،؛、，"
DASHES = "—–-"

# Break *before* these; they open a clause.
CONJUNCTIONS = {
    # en
    "and", "but", "or", "so", "because", "that", "which", "who", "when",
    "while", "if", "then", "as", "though", "although", "since", "unless",
    # de
    "und", "aber", "oder", "weil", "dass", "wenn", "als", "obwohl", "damit",
    "sondern", "denn",
    # es
    "y", "pero", "o", "porque", "que", "cuando", "si", "como", "aunque",
    "mientras", "pues",
}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
    "into", "over", "under", "after", "before", "between",
    "an", "auf", "zu", "für", "mit", "von", "bei", "über", "nach", "aus",
    "durch", "gegen", "ohne", "um", "vor",
    "a", "de", "con", "por", "para", "sobre", "entre", "hasta", "desde",
    "en", "sin",
}

# Never break immediately after these; they bind to the following word.
DETERMINERS = {
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "no", "some", "any",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
    "einer", "mein", "dein", "sein", "ihr", "unser", "kein", "keine",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "mi", "tu", "su",
    "sus", "mis", "tus", "del", "al",
}

_WORD_RE = re.compile(r"\w", re.UNICODE)
_MULTISPACE = re.compile(r"\s+")


@dataclass
class Cue:
    start: float
    end: float
    lines: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def flat(self) -> str:
        return " ".join(self.lines)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cps(self) -> float:
        if self.duration <= 0:
            return float("inf")
        return len(self.flat) / self.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "lines": self.lines,
            "cps": round(self.cps, 2),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# word extraction
# --------------------------------------------------------------------------- #

def _words_from(segments: Sequence[Segment]) -> list[Word]:
    """Flatten to words, synthesizing timings where the model gave none."""
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(w for w in seg.words if w.text.strip())
            continue
        # Fall back to proportional distribution across the segment.
        tokens = seg.text.split()
        if not tokens or seg.duration <= 0:
            continue
        total = sum(len(t) for t in tokens)
        cursor = seg.start
        for token in tokens:
            span = seg.duration * (len(token) / total)
            # Leading space keeps the raw-join convention below intact.
            words.append(Word(cursor, cursor + span, " " + token, seg.mean_word_prob))
            cursor += span
    return sorted(words, key=lambda w: w.start)


def _render(words: Sequence[Word]) -> str:
    """Reconstruct text from words.

    Word tokens carry their own leading whitespace ("' brother'", "\"'clock\"").
    They must be joined raw and trimmed — stripping each token and rejoining on
    a space inserts a space before every apostrophe and contraction, turning
    "o'clock" into "o 'clock". That corrupts the text and, because it changes
    the length, also breaks the line-fitting arithmetic below.
    """
    return _MULTISPACE.sub(" ", "".join(w.text for w in words)).strip()


def _clean(token: str) -> str:
    return token.strip()


def _bare(token: str) -> str:
    return re.sub(r"[^\w'’-]", "", token.strip().lower(), flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# break scoring
# --------------------------------------------------------------------------- #

def _break_score(words: list[Word], i: int) -> float:
    """Desirability of breaking between words[i] and words[i + 1]."""
    if i < 0 or i + 1 >= len(words):
        return -1e6

    left = _clean(words[i].text)
    right = _clean(words[i + 1].text)
    score = 0.0

    if left and left[-1] in TERMINAL:
        score += 100
    elif left and left[-1] in CLAUSE:
        score += 60
    elif left and left[-1] in DASHES:
        score += 45

    rb = _bare(right)
    if rb in CONJUNCTIONS:
        score += 30
    elif rb in PREPOSITIONS:
        score += 20

    lb = _bare(left)
    if lb in DETERMINERS:
        score -= 90
    if left.endswith("-"):
        score -= 70
    # Don't split a numeric run ("3 000", "12 45").
    if lb.isdigit() and rb.isdigit():
        score -= 50
    # Avoid splitting a capitalized name pair.
    if left[:1].isupper() and right[:1].isupper() and lb not in ("i",):
        score -= 15

    return score


def _chars(words: Sequence[Word]) -> int:
    return len(_render(words))


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #

def _sentences(words: list[Word], cfg: CueConfig) -> list[list[Word]]:
    """Split words at sentence ends and at long silences."""
    out: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            if gap > cfg.max_silence_within_cue:
                out.append(current)
                current = []
        current.append(word)
        token = _clean(word.text)
        # Break after terminal punctuation, but not on a bare "." or an
        # abbreviation-like single character before the dot.
        if token and token[-1] in TERMINAL and len(token.rstrip(TERMINAL)) > 1:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _pack(unit: list[Word], cfg: CueConfig) -> list[list[Word]]:
    """Split one sentence into chunks that each fit a cue."""
    budget = cfg.max_lines * cfg.max_chars_per_line
    chunks: list[list[Word]] = []
    start = 0

    while start < len(unit):
        # Furthest end that satisfies chars and duration.
        end = start
        while end + 1 < len(unit):
            nxt = unit[start : end + 2]
            if _chars(nxt) > budget:
                break
            if nxt[-1].end - nxt[0].start > cfg.max_duration:
                break
            end += 1

        if end < len(unit) - 1:
            # Back off to the best break point in the last third of the chunk.
            lo = start + max(1, int((end - start) * 0.55))
            best, best_score = end, _break_score(unit, end)
            for i in range(end, lo - 1, -1):
                score = _break_score(unit, i) - 0.15 * (end - i)
                if score > best_score:
                    best, best_score = i, score
            end = best

            # Orphan control. Sung or slowly-spoken words are long in time, so a
            # chunk can hit max_duration while still far short of the character
            # budget — leaving a one-word tail that is then too late to merge
            # (merging would exceed max_duration). Give the tail enough words to
            # stand on its own, as long as this chunk stays viable too.
            if cfg.merge_short_cues:
                while (
                    end + 1 < len(unit)
                    and _chars(unit[end + 1 :]) < cfg.min_cue_chars
                    and end > start
                    and _chars(unit[start:end]) >= cfg.min_cue_chars
                ):
                    end -= 1

        chunks.append(unit[start : end + 1])
        start = end + 1

    return chunks


def _break_lines(words: Sequence[Word], cfg: CueConfig) -> tuple[list[str], list[str]]:
    """Lay a cue's words out over at most cfg.max_lines lines."""
    words = list(words)
    flat = _render(words)
    warnings: list[str] = []

    if len(flat) <= cfg.max_chars_per_line:
        return [flat], warnings

    limit = cfg.max_chars_per_line
    floor = cfg.min_line_fill * limit
    mid = len(flat) / 2

    best: tuple[float, int] | None = None
    for i in range(len(words) - 1):
        left = _render(words[: i + 1])
        right = _render(words[i + 1 :])
        if len(left) > limit or len(right) > limit:
            continue
        if len(left) < floor or len(right) < floor:
            continue
        # Prefer a good syntactic break, then balance.
        cost = -_break_score(words, i) + abs(len(left) - mid) * 0.8
        if best is None or cost < best[0]:
            best = (cost, i)

    if best is None:
        # No two-line layout fits. Hard-wrap and flag for the editor.
        warnings.append("overlong")
        lines: list[str] = []
        current: list[Word] = []
        for word in words:
            if current and len(_render(current + [word])) > limit:
                lines.append(_render(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(_render(current))
        if len(lines) > cfg.max_lines:
            warnings.append(f"{len(lines)}_lines")
        return lines, warnings

    split = best[1] + 1
    return [_render(words[:split]), _render(words[split:])], warnings


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #

def _merge_runts(chunks: list[list[Word]], cfg: CueConfig) -> list[list[Word]]:
    """Fold tiny cues into a neighbour.

    Unpunctuated text (song lyrics especially) packs on character count alone and
    leaves single-word runts stranded — a cue reading just "है" is noise on
    screen. Merge a runt into whichever neighbour it fits, preferring the
    previous one so the phrase reads in order.
    """
    if not cfg.merge_short_cues or len(chunks) < 2:
        return chunks

    budget = cfg.max_lines * cfg.max_chars_per_line
    limit = cfg.min_cue_chars

    def fits(a: list[Word], b: list[Word]) -> bool:
        if _chars(a + b) > budget:
            return False
        return (b[-1].end - a[0].start) <= cfg.max_duration

    merged: list[list[Word]] = []
    for chunk in chunks:
        if (
            merged
            and _chars(chunk) < limit
            and fits(merged[-1], chunk)
            and (chunk[0].start - merged[-1][-1].end) <= cfg.max_silence_within_cue
        ):
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)

    # A runt at the very front has no previous neighbour; push it into the next.
    if len(merged) > 1 and _chars(merged[0]) < limit and fits(merged[0], merged[1]):
        merged[1] = merged[0] + merged[1]
        merged.pop(0)

    return merged


def _apply_timing(cues: list[Cue], cfg: CueConfig) -> None:
    for i, cue in enumerate(cues):
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        ceiling = (nxt.start - cfg.min_gap) if nxt else cue.end + cfg.max_duration

        # Extend a too-short or too-fast cue into the following silence.
        needed_for_speed = len(cue.flat) / cfg.target_cps
        target_end = cue.start + max(cfg.min_duration, needed_for_speed)
        if target_end > cue.end:
            cue.end = min(target_end, ceiling, cue.start + cfg.max_duration)

        if cue.duration > cfg.max_duration:
            cue.end = cue.start + cfg.max_duration

        if nxt and cue.end > nxt.start - cfg.min_gap:
            cue.end = max(cue.start + 0.2, nxt.start - cfg.min_gap)

        if cue.duration < cfg.min_duration:
            cue.warnings.append("short")
        if cue.cps > cfg.max_cps:
            cue.warnings.append(f"cps_{cue.cps:.0f}")


def _chunks_for(words: Sequence[Word], cfg: CueConfig) -> list[list[Word]]:
    """Sentence-split, pack to the line limits, then fold single-word runts in."""
    if not words:
        return []
    chunks: list[list[Word]] = []
    for sentence in _sentences(list(words), cfg):
        chunks.extend(c for c in _pack(sentence, cfg) if c)
    return _merge_runts(chunks, cfg)


def build(
    segments: Sequence[Segment],
    cfg: CueConfig,
    *,
    include_suppressed: bool = False,
) -> list[Cue]:
    """Build final cues from ASR segments.

    `include_suppressed` keeps gated segments in the output so an editor can
    review what the gate removed rather than trusting it blindly.
    """
    usable = segments if include_suppressed else [s for s in segments if not s.suppressed]
    if not _words_from(usable):
        return []

    if cfg.respect_segment_boundaries:
        # Sung lines are the units. Flattening everything into one word stream and
        # re-packing to 42 characters straddles them, so a chorus repeated three
        # times survives as text but stops looking like three identical lines —
        # which is the whole point of subtitling a song.
        chunks: list[list[Word]] = []
        for segment in usable:
            chunks.extend(_chunks_for(_words_from([segment]), cfg))
    else:
        chunks = _chunks_for(_words_from(usable), cfg)

    cues: list[Cue] = []
    for chunk in chunks:
        lines, warnings = _break_lines(chunk, cfg)
        if not any(_WORD_RE.search(line) for line in lines):
            continue
        cues.append(
            Cue(
                start=max(0.0, chunk[0].start - cfg.lead_in),
                end=chunk[-1].end + cfg.lead_out,
                lines=lines,
                warnings=warnings,
            )
        )

    cues.sort(key=lambda c: c.start)
    _apply_timing(cues, cfg)
    return cues
