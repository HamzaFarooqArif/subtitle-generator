"""Suppress hallucinated and non-lexical output.

Whisper's failure mode on breathing, whispering and non-speech vocalization is
not silence — it is fluent, confident, entirely invented text. Left alone it
produces subtitles that look correct and are fabricated, which is worse than
producing nothing. This module is the defence.

Nothing is deleted here. Segments are marked with `suppressed` and a reason so
the sidecar keeps the full record and `--keep-suppressed` can surface them for
review.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .asr import Segment
from .config import GatingConfig

# Phrases Whisper emits over non-speech audio, from its training data's
# subtitle-file provenance. Matched against normalized text.
HALLUCINATION_PHRASES = (
    # English
    "thanks for watching", "thank you for watching", "please subscribe",
    "subscribe to my channel", "like and subscribe", "see you next time",
    "subtitles by", "subtitled by", "captioning by", "amara.org",
    "transcription by", "translated by",
    # German
    "vielen dank fur das zuschauen", "danke fur das zuschauen",
    "untertitel von", "untertitelung des zdf", "untertitel im auftrag des zdf",
    "copyright wdr", "abonniert diesen kanal", "bis zum nachsten mal",
    # Spanish
    "gracias por ver", "gracias por su atencion", "suscribete",
    "subtitulos por", "subtitulos realizados por", "mas informacion",
    # Turkish. Whisper emits these over music and noise remarkably often —
    # "Sağolun" is the phrase it produced for an entire Hindi song, and Turkish
    # is a frequent wrong guess when language detection has nothing to work with.
    "sagolun", "sag olun", "tesekkurler", "tesekkur ederim", "altyazi",
    "altyazi m k", "abone olmayi unutmayin", "izlediginiz icin tesekkurler",
    # Other languages that show up as confident misdetections
    # Kept as full phrases: a bare "字幕" or "субтитры" would match almost any
    # short segment in those languages and delete real speech.
    "amara org", "subs by", "sous-titres par", "sottotitoli a cura",
    "napisy stworzone", "спасибо за просмотр", "субтитры сделал",
    # Russian subtitle credits Whisper reproduces over non-speech. Seen in the
    # wild: "Субтитры сделал DimaTorzok", "Редактор субтитров А.Семкин".
    "редактор субтитров", "корректор субтитров", "субтитры подготовил",
    "субтитры и перевод", "перевод и субтитры", "динамичная музыка",
    "заключительная музыка",
    "ご視聴ありがとうございました", "구독과 좋아요", "gracias por la traduccion",
    # Music / sound-effect placeholders
    "[music]", "[musica]", "[musik]", "(music)", "(musik)", "(musica)",
    "[applause]", "[laughter]", "[silence]", "[sonido]", "[muzik]",
)

# Segments made up entirely of these are non-lexical vocalization, not speech.
NON_LEXICAL = {
    "ah", "aah", "aaah", "ha", "hah", "huh", "hm", "hmm", "hmmm", "mm", "mmm",
    "mhm", "mmhm", "uh", "uhh", "uhm", "um", "umm", "oh", "ooh", "oooh", "ow",
    "ugh", "eh", "ehh", "err", "aw", "aww", "wow", "phew", "shh", "sh", "psh",
    "ay", "aiy", "uy", "ea", "yeah", "ja", "si", "oui", "mmn", "nn", "ngh",
    "ah-ah", "uh-uh", "uh-huh", "mm-hmm",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def _is_hallucination_phrase(text: str) -> bool:
    norm = normalize(text)
    if not norm:
        return True
    for phrase in HALLUCINATION_PHRASES:
        target = normalize(phrase)
        if not target:
            continue
        # Short segments must match closely; long ones only need to contain it.
        if target in norm and (len(norm) < len(target) * 2.5 or len(norm) < 40):
            return True
    return False


def _is_non_lexical(text: str) -> bool:
    tokens = normalize(text).split()
    if not tokens:
        return True
    return all(t in NON_LEXICAL for t in tokens)


def _repeat_loop(text: str, max_ratio: float = 0.6) -> bool:
    """Detect a single token or short phrase dominating the segment."""
    tokens = normalize(text).split()
    if len(tokens) < 4:
        return False
    counts = Counter(tokens)
    if counts.most_common(1)[0][1] / len(tokens) >= max_ratio:
        return True
    # Repeated bigram, e.g. "oh my oh my oh my".
    bigrams = Counter(zip(tokens, tokens[1:]))
    if bigrams and bigrams.most_common(1)[0][1] >= max(3, len(tokens) // 4):
        return True
    return False


@dataclass
class GateStats:
    total: int = 0
    suppressed: int = 0
    reasons: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()

    @property
    def kept(self) -> int:
        return self.total - self.suppressed

    def summary(self) -> str:
        if not self.total:
            return "no segments"
        pct = 100.0 * self.suppressed / self.total
        detail = ", ".join(f"{r}={n}" for r, n in self.reasons.most_common())
        return f"{self.kept}/{self.total} kept ({pct:.1f}% suppressed)" + (
            f" [{detail}]" if detail else ""
        )


def apply(segments: list[Segment], cfg: GatingConfig) -> GateStats:
    """Mark suppressed segments in place and return statistics."""
    stats = GateStats(total=len(segments))
    if not cfg.enabled:
        return stats

    norm_texts = [normalize(s.text) for s in segments]

    for i, seg in enumerate(segments):
        reason = _reason_for(segments, norm_texts, i, cfg)
        if reason:
            seg.suppressed = True
            seg.suppress_reason = reason
            stats.suppressed += 1
            stats.reasons[reason] += 1

    return stats


def _reason_for(
    segments: list[Segment], norm_texts: list[str], i: int, cfg: GatingConfig
) -> str | None:
    seg = segments[i]
    text = seg.text.strip()

    if not text:
        return "empty"

    if _is_hallucination_phrase(text):
        return "hallucination_phrase"

    if seg.compression_ratio > cfg.max_compression_ratio:
        return "repetition"

    if _repeat_loop(text):
        return "repeat_loop"

    # Non-lexical vocalization: only suppressed when the model was also unsure.
    # A clearly-articulated "yeah" is real speech and should survive.
    if _is_non_lexical(text) and (
        seg.no_speech_prob > cfg.max_no_speech_prob
        or seg.mean_word_prob < cfg.min_mean_word_prob
    ):
        return "non_lexical"

    if seg.no_speech_prob > cfg.hard_no_speech_prob:
        return "no_speech"

    if seg.avg_logprob < cfg.hard_avg_logprob:
        return "very_low_confidence"

    if seg.no_speech_prob > cfg.max_no_speech_prob and seg.avg_logprob < cfg.min_avg_logprob:
        return "low_confidence_no_speech"

    if seg.words and seg.mean_word_prob < cfg.min_mean_word_prob:
        return "low_word_confidence"

    # A handful of words smeared across a long span is the signature of the
    # model reaching for lexical content in audio that has none.
    if seg.duration >= cfg.min_span_for_wps_check:
        wps = len(text.split()) / seg.duration
        if wps < cfg.min_words_per_second:
            return "sparse_text"

    # Identical text recurring across neighbours without intervening silence.
    window = norm_texts[max(0, i - cfg.max_repeat_of_neighbour) : i]
    if norm_texts[i] and window.count(norm_texts[i]) >= cfg.max_repeat_of_neighbour:
        return "duplicate_neighbour"

    return None
