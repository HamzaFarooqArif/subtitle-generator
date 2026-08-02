"""Translate transcript text with a dedicated translation model.

Whisper can translate speech to English on its own, but it does so from 30-second
audio chunks and translation is a byproduct of its training rather than its
purpose. A dedicated model working on the *text* is markedly better — which is
what you see when you paste subtitles into an online translator and the result
beats Whisper's own output.

Two things make the difference, and only one of them is the model:

1. **Sentence context.** Cues are split for reading, not for meaning; half a
   sentence translates badly. Text is regrouped into complete sentences,
   translated, and only then redistributed across cue timings.
2. **A translation-specific model.** NLLB-200 converted to CTranslate2, so it
   reuses the inference engine already installed for ASR.

Because this works on text, it runs from the sidecar with no audio decode, it can
translate hand-edited subtitles, and it can target languages other than English.
"""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .asr import Segment, Word

log = logging.getLogger(__name__)

# Whisper language code -> NLLB (FLORES-200) code.
NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn", "hi": "hin_Deva", "de": "deu_Latn", "es": "spa_Latn",
    "fr": "fra_Latn", "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn",
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn",
    "tr": "tur_Latn", "ar": "arb_Arab", "ur": "urd_Arab", "fa": "pes_Arab",
    "he": "heb_Hebr", "el": "ell_Grek", "sv": "swe_Latn", "da": "dan_Latn",
    "no": "nob_Latn", "fi": "fin_Latn", "hu": "hun_Latn", "ro": "ron_Latn",
    "bg": "bul_Cyrl", "hr": "hrv_Latn", "sr": "srp_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "lt": "lit_Latn", "lv": "lvs_Latn", "et": "est_Latn",
    "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva",
    "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "ne": "npi_Deva", "si": "sin_Sinh", "as": "asm_Beng", "or": "ory_Orya",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans", "vi": "vie_Latn",
    "th": "tha_Thai", "id": "ind_Latn", "ms": "zsm_Latn", "tl": "tgl_Latn",
    "sw": "swh_Latn", "af": "afr_Latn", "ca": "cat_Latn", "eu": "eus_Latn",
    "is": "isl_Latn", "hy": "hye_Armn", "ka": "kat_Geor", "az": "azj_Latn",
    "kk": "kaz_Cyrl", "sq": "als_Latn", "mk": "mkd_Cyrl", "bs": "bos_Latn",
}

TERMINAL = ".!?…।॥۔。！？"
_SENTENCE_END = re.compile(rf"[{re.escape(TERMINAL)}][\"'”’)\]]*\s*$")
_CLOSERS = "\"'”’)]»」』"
# Fullwidth stops end a sentence on their own: CJK does not put a space after
# them, so requiring whitespace (which is what protects "3.5" and "p.m.") would
# never find a boundary.
_ALWAYS_TERMINAL = "。！？"


def split_sentences(text: str) -> list[str]:
    """Break text into single sentences for translation.

    NLLB translates one sentence at a time and drops everything after the first
    sentence boundary, so feeding it two sentences loses one of them.

    Scanned rather than regex-split: a sentence only ends when terminal
    punctuation is followed by whitespace or end-of-text, which keeps "3.5" and
    "p.m." intact.
    """
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        current.append(char)
        i += 1
        if char not in TERMINAL:
            continue
        # Absorb any closing quotes or brackets that belong to this sentence.
        while i < len(text) and text[i] in _CLOSERS:
            current.append(text[i])
            i += 1
        if char in _ALWAYS_TERMINAL or i >= len(text) or text[i].isspace():
            out.append("".join(current).strip())
            current = []

    if current:
        out.append("".join(current).strip())
    return [s for s in out if s] or [text]


_CONTRACTION = re.compile(r"\s+([’']\s?(?:s|t|m|re|ve|ll|d)\b)", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")


def _clean_output(text: str) -> str:
    """Tidy detokenization artifacts from the translation model.

    Sentencepiece detokenization leaves gaps like "I 'm not joking" and " ,",
    and NLLB sometimes prefixes a dialogue dash that was not in the source.
    """
    text = re.sub(r"^[-–—•]\s*", "", text.strip())
    text = _CONTRACTION.sub(lambda m: m.group(1).replace(" ", ""), text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def supported(language: str | None) -> bool:
    return (language or "").lower() in NLLB_CODES


@dataclass
class _Group:
    """One or more consecutive segments forming a complete sentence."""

    text: str
    start: float
    end: float
    members: list[Segment]


def group_sentences(
    segments: Sequence[Segment],
    max_chars: int = 200,
    max_span: float = 8.0,
) -> list[_Group]:
    """Merge consecutive segments until each group ends a sentence.

    Resegmentation already splits on sentence punctuation, so normally each
    segment is a sentence and grouping is a no-op. Fragments get joined so the
    model sees a whole clause.

    **Unless nothing is punctuated at all.** Merging song lyrics — which have no
    punctuation anywhere — produced a 400-character run-on that the model
    answered with repetitive nonsense. Each line translated on its own is far
    better, so when no segment ends a sentence, no merging happens.
    """
    texts = [s.text.strip() for s in segments if s.text.strip()]
    if texts and not any(t[-1] in TERMINAL for t in texts):
        return [
            _Group(s.text.strip(), s.start, s.end, [s])
            for s in segments
            if s.text.strip()
        ]

    groups: list[_Group] = []
    pending: list[Segment] = []

    def flush() -> None:
        if not pending:
            return
        text = " ".join(s.text.strip() for s in pending if s.text.strip()).strip()
        if text:
            groups.append(_Group(text, pending[0].start, pending[-1].end, list(pending)))
        pending.clear()

    for segment in segments:
        if not segment.text.strip():
            continue
        # Flush before the span gets long. Merging five unpunctuated cues over
        # 26 seconds produced a run-on that translated far worse than the same
        # lines translated individually, and then had to be chopped back across
        # cues anyway. Genuine sentence fragments are short and adjacent.
        if pending and (segment.end - pending[0].start) > max_span:
            flush()
        pending.append(segment)
        joined = " ".join(s.text.strip() for s in pending)
        if _SENTENCE_END.search(segment.text.strip()) or len(joined) >= max_chars:
            flush()
    flush()
    return groups


class Translator:
    """NLLB-200 via CTranslate2. Load once, reuse across a batch."""

    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path,
        *,
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ):
        from . import cuda

        cuda.prepare()
        import ctranslate2
        import transformers

        self._tokenizer_dir = str(tokenizer_dir)
        log.info("loading translation model (%s)", compute_type)
        self._translator = ctranslate2.Translator(
            str(model_dir), device=device, compute_type=compute_type
        )
        self._tokenizer_cls = transformers.AutoTokenizer
        self._tokenizer = None
        self._src: str | None = None

    def _tokenizer_for(self, src_code: str):
        if self._tokenizer is None or self._src != src_code:
            self._tokenizer = self._tokenizer_cls.from_pretrained(
                self._tokenizer_dir, src_lang=src_code
            )
            self._src = src_code
        return self._tokenizer

    def close(self) -> None:
        self._translator = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def translate_texts(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str = "en",
        *,
        beam_size: int = 4,
        max_batch: int = 16,
    ) -> list[str]:
        src = NLLB_CODES.get(source_language.lower())
        tgt = NLLB_CODES.get(target_language.lower())
        if src is None or tgt is None:
            raise ValueError(
                f"unsupported language pair {source_language!r} -> {target_language!r}"
            )

        tokenizer = self._tokenizer_for(src)

        # NLLB is trained on single sentences and silently stops at the first
        # sentence boundary: "Where are you going? Come with me." came back as
        # "Where are you going?", losing half the line. Every input is therefore
        # split into sentences, translated individually, and rejoined.
        pieces: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            sentences = split_sentences(text)
            spans.append((len(pieces), len(pieces) + len(sentences)))
            pieces.extend(sentences)

        if not pieces:
            return ["" for _ in texts]

        sources = [
            tokenizer.convert_ids_to_tokens(tokenizer.encode(p)) for p in pieces
        ]
        results = self._translator.translate_batch(
            sources,
            target_prefix=[[tgt]] * len(sources),
            beam_size=beam_size,
            max_batch_size=max_batch,
            # Subtitles are short; a generous cap avoids truncating long lines.
            max_decoding_length=512,
        )

        translated: list[str] = []
        for result in results:
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == tgt:
                tokens = tokens[1:]  # drop the forced target-language token
            decoded = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
            ).strip()
            translated.append(_clean_output(decoded))

        return [" ".join(t for t in translated[lo:hi] if t).strip() for lo, hi in spans]

    def translate_segments(
        self,
        segments: Sequence[Segment],
        source_language: str,
        target_language: str = "en",
        *,
        cue_cfg=None,
    ) -> list[Segment]:
        return translate_segments(
            self, segments, source_language, target_language, cue_cfg=cue_cfg
        )


def translate_segments(
    translator,
    segments: Sequence[Segment],
    source_language: str,
    target_language: str = "en",
    *,
    cue_cfg=None,
) -> list[Segment]:
    """Translate segments with any backend, returning them on the same timeline.

    `translator` only needs a `translate_texts(texts, src, tgt) -> list[str]`
    method, so the local NLLB model and the online providers share every other
    step: sentence grouping, timing, and the cue builder that follows.
    """
    groups = group_sentences(segments)
    if not groups:
        return []

    translations = translator.translate_texts(
        [g.text for g in groups], source_language, target_language
    )

    timing = {}
    if cue_cfg is not None:
        timing = {
            "target_cps": cue_cfg.target_cps,
            "max_duration": cue_cfg.max_duration,
            "min_duration": max(1.0, cue_cfg.min_duration),
        }

    out: list[Segment] = []
    for group, text in zip(groups, translations):
        text = (text or "").strip()
        if not text:
            continue
        out.extend(_redistribute(group, text, **timing))
    return out


def _redistribute(group: _Group, text: str, **timing) -> list[Segment]:
    """Return the translation as one segment spanning the group's time.

    An earlier version apportioned the translated words across the group's source
    segments by character share. That was wrong in a way that dominated the
    output: word order differs between languages, so cutting the English at a
    position derived from the source produced fragments —

        27  to tell you nothing,
        28  nothing Yes, of
        29  course

    Emitting one segment over the whole span and letting the cue builder split it
    is strictly better: it breaks at clause boundaries, enforces line length and
    reading speed, and refuses to leave one-word orphans. The span is unchanged,
    so timing still tracks the speech.
    """
    members = [m for m in group.members if m.text.strip()]
    return [_make_segment(group.start, group.end, text, members, **timing)]


def _make_segment(
    start: float,
    end: float,
    text: str,
    members: Sequence[Segment],
    *,
    target_cps: float = 15.0,
    max_duration: float = 7.0,
    min_duration: float = 1.0,
) -> Segment:
    """Build a segment whose words occupy only the time the text needs.

    Spreading the words evenly across the whole source span looks harmless and
    is not: a 29-character translation stretched over 12 seconds exceeds
    max_duration, so the cue builder splits it and you get

        Yeah, I've / missed you / so much.

    The span came from merged source segments and is longer than the sentence
    needs. Sizing the segment by reading speed instead — anchored at the start of
    the span, never longer than the span — keeps short translations in one cue and
    still lets genuinely long ones split.
    """
    tokens = text.split()
    span = max(0.001, end - start)
    needed = len(text) / max(1.0, target_cps)

    if needed > max_duration:
        # Genuinely long: keep the whole span so the cue builder can split it
        # into several cues at sensible break points.
        duration = span
    else:
        # Fits in one cue: take only the time it needs, never more than the span.
        duration = min(span, max(min_duration, needed))

    end = start + duration
    total = sum(len(t) for t in tokens) or 1
    words: list[Word] = []
    cursor = start
    for token in tokens:
        span = duration * (len(token) / total)
        words.append(Word(cursor, min(end, cursor + span), " " + token, 1.0))
        cursor += span

    probability = (
        min((m.mean_word_prob for m in members if m.words), default=1.0) if members else 1.0
    )
    return Segment(
        start=start,
        end=end,
        text=text,
        words=words,
        # Confidence describes the source audio, not the translation. Carry the
        # source's value so downstream QC has something meaningful, but never let
        # it gate the translation — that decision belongs to the source pass.
        avg_logprob=members[0].avg_logprob if members else 0.0,
        no_speech_prob=0.0,
        compression_ratio=1.0,
        language="translated",
    )
