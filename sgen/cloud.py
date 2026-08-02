"""Translate an existing transcript through Google or DeepL.

Two callers share this: the Translate panel, which does one finished file on
demand, and a transcription job, which can run it immediately after writing the
subtitles. Same code either way, so a file translated during a run and the same
file translated afterwards cannot come out different.

Nothing here happens on its own. A job only reaches this if the user chose a
cloud provider in Settings, and that choice is what sends transcript text to a
third party — the single point in the pipeline where anything leaves the machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import cues as cues_mod
from . import online
from . import roundtrip
from . import translate as mt
from .asr import Segment, Word
from .config import Config
from .write import write_subtitles

log = logging.getLogger(__name__)

# One request carries this much of the numbered transcript. Large enough that a
# scene stays together, small enough to stay well inside request limits.
DOCUMENT_CHUNK_CHARS = 2500
# Below this share of cues matched back by number, the document was mangled and
# the per-cue path is used instead.
MIN_MATCH_RATIO = 0.8


class NotPossible(ValueError):
    """The request cannot work as asked — a missing key, or a pair the provider
    does not have. Distinct from a service failure: nothing was sent, and
    retrying unchanged will fail the same way."""


@dataclass
class Translated:
    written: list[Path]
    cue_count: int
    provider: str
    language: str
    characters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": [str(p) for p in self.written],
            "cue_count": self.cue_count,
            "provider": self.provider,
            "language": self.language,
            "characters": self.characters,
        }


def resolve(provider_name: str, source_language: str, target_language: str):
    """Check the request is possible and return the provider.

    Everything that can be known without spending a request is checked here, so
    a mistake costs nothing and reads as a mistake rather than as an outage.
    """
    source = (source_language or "").lower()
    target = (target_language or "en").lower()
    if source and source == target:
        raise NotPossible(f"already in {target!r}")

    try:
        provider = online.get_translator(provider_name)
    except online.TranslationError as exc:
        raise NotPossible(str(exc)) from exc

    # Asked of DeepL, not assumed: its language list grows, and a built-in one
    # already refused pairs the service supports.
    if provider_name == "deepl" and not provider.can_target(target):
        raise NotPossible(
            f"DeepL does not translate into {target!r} — use Google for this language."
        )
    return provider


def translate(
    cues: Sequence,
    provider,
    *,
    provider_name: str,
    source_language: str,
    target_language: str,
    source_path: Path,
    formats: Sequence[str],
    out_dir: Path | None = None,
    cfg: Config | None = None,
) -> Translated:
    """Translate cues and write subtitle files beside (or into) the target dir.

    Raises online.TranslationError if the service itself fails.
    """
    if not cues:
        raise NotPossible("no subtitles to translate")

    cfg = cfg or Config()
    target = (target_language or "en").lower()
    source = (source_language or "").lower()

    built = _as_document(cues, provider, source, target, cfg)
    if built is None:
        built = _cue_by_cue(cues, provider, source, target, cfg)

    base_dir = out_dir or source_path.parent
    written = write_subtitles(
        built, base_dir / source_path.stem, tuple(formats), target, cfg.encoding
    )
    return Translated(
        written=list(written),
        cue_count=len(built),
        provider=provider_name,
        language=target,
        characters=sum(len(" ".join(c.lines)) for c in cues),
    )


def _as_document(cues, provider, source: str, target: str, cfg: Config):
    """Translate the whole numbered transcript at once, the way a person would.

    This is the difference the user could see. Both APIs translate each item in
    a request independently, so sending one cue per item means every line is
    translated as if it were the only sentence in existence: "мастер по
    интернету / до приходил пришел / отрабатывает" came back as "web developer /
    You were coming—you're here / working". Pasting the same transcript into the
    web UI produced "The internet technician / He came by earlier / He's working
    on it", because there the model saw the conversation.

    So the transcript goes as numbered lines in one request, and the numbers put
    it back on the timings. Returns None if the numbering did not survive, which
    is the one thing that could silently scramble a file.
    """
    document = roundtrip.export_text(cues)
    chunks = _chunks(document)
    try:
        replies = provider.translate_texts(chunks, source, target)
    except online.TranslationError:
        raise
    joined = "\n".join(replies)

    applied, report = roundtrip.apply_translation(cues, joined, keep_untranslated=False)
    if not applied or report.matched < max(1, int(report.total * MIN_MATCH_RATIO)):
        log.warning(
            "numbering survived for only %d of %d cues; translating cue by cue",
            report.matched, report.total,
        )
        return None
    log.info("translated as one document: %d/%d cues matched by %s",
             report.matched, report.total, report.method)
    return roundtrip.rebreak(applied, cfg.cues)


def _chunks(document: str, limit: int = DOCUMENT_CHUNK_CHARS) -> list[str]:
    """Split on line boundaries only, so no cue is ever cut in half."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in document.splitlines():
        if current and size + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _cue_by_cue(cues, provider, source: str, target: str, cfg: Config):
    """Fallback: one request item per cue.

    Alignment cannot fail here — each translation belongs to the cue it came
    from — at the cost of the context that makes the document path better.
    """
    segments = []
    for cue in cues:
        text = " ".join(cue.lines).strip()
        if not text:
            continue
        segments.append(
            Segment(start=cue.start, end=cue.end, text=text,
                    words=[Word(cue.start, cue.end, " " + text, 1.0)])
        )
    if not segments:
        raise NotPossible("no subtitle text to translate")

    translated = mt.translate_segments(
        provider, segments, source, target, cue_cfg=cfg.cues
    )
    built = cues_mod.build(translated, cfg.cues)
    if not built:
        raise online.TranslationError("the service returned nothing usable")
    return built
