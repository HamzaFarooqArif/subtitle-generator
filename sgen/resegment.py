"""Split ASR segments to sentence granularity before gating.

Whisper's batched pipeline happily returns a single 35-second segment covering
continuous speech. Gating at that granularity is destructive: one aggregate
signal tripping (a compression ratio pushed over threshold by a genuinely
repeated phrase, say) discards half a minute of correct transcription.

Splitting to sentences first makes every suppression decision proportionate, and
makes the per-segment metrics meaningful — a compression ratio computed over one
sentence says something about that sentence, whereas one computed over thirty
seconds says almost nothing about any part of it.
"""

from __future__ import annotations

import zlib

from .asr import Segment, Word

# Must match cues.TERMINAL: Devanagari danda, Urdu/Arabic and CJK stops
# included, or non-Latin scripts read as having no sentence structure at all.
TERMINAL = ".!?…।॥۔。！？"


def compression_ratio(text: str) -> float:
    """Same measure faster-whisper uses, recomputed for a text slice."""
    data = text.encode("utf-8")
    if not data:
        return 1.0
    return len(data) / len(zlib.compress(data))


def _render(words: list[Word]) -> str:
    """Words carry their own leading whitespace; join raw and trim."""
    return "".join(w.text for w in words).strip()


def _emit(parent: Segment, words: list[Word]) -> Segment | None:
    text = _render(words)
    if not text:
        return None
    probs = [w.probability for w in words if w.probability]
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=list(words),
        # Recomputed for this slice: the whole point of splitting.
        compression_ratio=compression_ratio(text),
        # Inherited — these are properties of the decode window, not the
        # sentence, so they stay as the parent's weaker evidence.
        avg_logprob=parent.avg_logprob,
        no_speech_prob=parent.no_speech_prob,
        temperature=parent.temperature,
        language=parent.language,
    )


def split(
    segments: list[Segment],
    *,
    max_silence: float = 0.7,
    max_duration: float = 12.0,
) -> list[Segment]:
    """Return segments split at sentence ends, long silences and a hard cap."""
    out: list[Segment] = []

    for segment in segments:
        if not segment.words:
            # Nothing to split on; keep as-is.
            out.append(segment)
            continue

        current: list[Word] = []
        for word in segment.words:
            if current:
                gap = word.start - current[-1].end
                span = word.end - current[0].start
                if gap > max_silence or span > max_duration:
                    piece = _emit(segment, current)
                    if piece:
                        out.append(piece)
                    current = []

            current.append(word)

            token = word.text.strip()
            # Break after terminal punctuation, but not on an abbreviation or a
            # decimal point ("4 p.m.", "3.5") where the token is a single char.
            if token and token[-1] in TERMINAL and len(token.rstrip(TERMINAL)) > 1:
                piece = _emit(segment, current)
                if piece:
                    out.append(piece)
                current = []

        if current:
            piece = _emit(segment, current)
            if piece:
                out.append(piece)

    return out
