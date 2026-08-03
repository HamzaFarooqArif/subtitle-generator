"""Stage orchestration: probe -> extract -> detect -> asr -> gate -> cues -> write."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import cues as cues_mod
from . import extract, gating, probe, qc, resegment, write
from .asr import Recognizer, Segment, Transcript, Word
from .models import ModelMissing
from .config import Config
from .cues import Cue

log = logging.getLogger(__name__)


@dataclass
class Result:
    source: Path
    cues: list[Cue]
    language: str
    language_probability: float
    duration: float
    gate_summary: str
    suppressed_count: int
    outputs: list[Path] = field(default_factory=list)
    sidecar: Optional[Path] = None
    content_id: str = ""
    audio: Optional[Path] = None
    verdict: Optional["qc.Verdict"] = None


class Pipeline:
    """Holds the ASR model across a batch so it loads exactly once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._recognizer: Recognizer | None = None
        # Output stems already used this run, so two sources sharing a stem
        # (clip.mp4 and clip.wav in one --out-dir) cannot silently overwrite
        # each other's subtitles.
        self._claimed: dict[Path, str] = {}

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def recognizer(self) -> Recognizer:
        if self._recognizer is None:
            self._recognizer = Recognizer(self.cfg.asr)
        return self._recognizer

    def close(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def _claim(self, out_base: Path, source: Path, content_id: str) -> Path:
        """Reserve an output stem, disambiguating collisions by source suffix."""
        owner = self._claimed.get(out_base)
        if owner is None or owner == content_id:
            self._claimed[out_base] = content_id
            return out_base

        suffix = source.suffix.lstrip(".").lower() or "file"
        candidate = out_base.with_name(f"{out_base.name}_{suffix}")
        n = 2
        while self._claimed.get(candidate, content_id) != content_id:
            candidate = out_base.with_name(f"{out_base.name}_{suffix}{n}")
            n += 1
        log.warning(
            "output name collision for %s; writing as %s", out_base.name, candidate.name
        )
        self._claimed[candidate] = content_id
        return candidate

    def _translate(
        self,
        wav: Path,
        language: str | None,
        transcript: Transcript,
        built: Sequence[Cue],
        out_base: Path,
        no_vad: bool,
        report: Callable[[str, float], None],
    ) -> list[Path]:
        """Produce translated subtitles, preferring text translation.

        Text translation on the transcript beats Whisper's own speech
        translation clearly enough to be the default: Whisper works from
        30-second audio chunks with no sentence context, which is exactly what
        makes its output read worse than pasting the subtitles into a translator.
        """
        cfg = self.cfg
        engine = cfg.translate_engine
        if engine == "auto":
            engine = _pick_translate_engine(transcript.segments)

        if engine == "nllb":
            try:
                return self._translate_text(transcript, out_base, report)
            except ModelMissing as exc:
                log.warning(
                    "%s — falling back to Whisper's speech translation, which is "
                    "noticeably weaker. Run: sgen models pull --translation", exc
                )
            except Exception:
                log.warning("text translation failed; falling back to Whisper",
                            exc_info=True)

        return self._translate_pass(
            wav, language, out_base, no_vad, report,
            speech_regions=[(c.start, c.end) for c in built],
        )

    def _translate_text(
        self,
        transcript: Transcript,
        out_base: Path,
        report: Callable[[str, float], None],
    ) -> list[Path]:
        """Translate the transcript text with NLLB, then rebuild cues."""
        from . import models as registry
        from . import translate as mt

        cfg = self.cfg
        source = transcript.language
        target = cfg.translate_target
        if not mt.supported(source) or not mt.supported(target):
            raise ValueError(f"unsupported pair {source!r} -> {target!r}")

        name = (
            registry.best_mt_model()
            if cfg.translate_model == "auto"
            else cfg.translate_model
        )
        model_dir = registry.mt_path(name)

        report("translate")
        # 8 GB card: release the ASR weights before loading the translator.
        self.close()

        translator = mt.Translator(
            model_dir, model_dir, device=cfg.asr.device, compute_type="int8_float16"
        )
        try:
            kept = [s for s in transcript.segments if not s.suppressed and s.text.strip()]
            log.info(
                "translating %d segments %s -> %s with %s",
                len(kept), source, target, name,
            )
            translated = translator.translate_segments(
                kept, source, target, cue_cfg=cfg.cues
            )
        finally:
            translator.close()
        report("translate", 1.0)

        if not translated:
            log.warning("translation produced nothing")
            return []

        # No gating: the source pass already decided what is real speech, and
        # confidence numbers from the audio say nothing about translation quality.
        cues = cues_mod.build(translated, cfg.cues)
        if not cues:
            return []
        return write.write_subtitles(cues, out_base, cfg.formats, target, cfg.encoding)

    def _translate_pass(
        self,
        wav: Path,
        language: str | None,
        out_base: Path,
        no_vad: bool,
        report: Callable[[str, float], None],
        speech_regions: Sequence[tuple[float, float]] = (),
    ) -> list[Path]:
        """Decode the audio again as English, via Whisper's translate task.

        Whisper translates speech directly to English, so this produces
        audio-aligned English cues in one pass with no extra model. `no_vad`
        carries over the decision made for the transcription pass — if voice
        activity detection had to be disabled to see this audio at all, the
        translation pass would hit exactly the same wall.
        """
        cfg = self.cfg
        report("translate")
        log.info("translating to English (task=translate)")
        try:
            english = self.recognizer.transcribe(
                wav,
                language=language,
                task="translate",
                progress=lambda f: report("translate", f),
                vad_filter=False if no_vad else None,
                force_sequential=no_vad,
            )
        except Exception as exc:
            # A failed translation must not lose the transcription we already have.
            log.warning("translation pass failed: %s", exc)
            return []

        if cfg.asr.resegment:
            english.segments = resegment.split(
                english.segments,
                max_silence=cfg.asr.resegment_max_silence,
                max_duration=cfg.asr.resegment_max_duration,
            )

        # Gate the translation on WHERE the transcription pass found speech, not
        # on the translation's own confidence numbers. Those numbers describe the
        # source audio, which for music or breathy speech is exactly the audio
        # the transcription pass already had to make allowances for — re-judging
        # it here suppressed 40% of a translation whose transcription was fine,
        # truncating lines mid-sentence. Confidence checks are therefore relaxed
        # to the two unambiguous ones (learned boilerplate, degenerate output).
        relaxed = replace(
            cfg.gating,
            max_no_speech_prob=1.01,
            hard_no_speech_prob=1.01,
            min_avg_logprob=-99.0,
            hard_avg_logprob=-99.0,
            min_mean_word_prob=0.0,
            min_words_per_second=0.0,
        )
        stats = gating.apply(english.segments, relaxed)
        if speech_regions:
            before = len(english.segments)
            english.segments = [
                s for s in english.segments if _overlaps(s, speech_regions)
            ]
            dropped = before - len(english.segments)
            if dropped:
                log.debug("dropped %d translated segments outside kept speech", dropped)
        log.info("translation gating: %s", stats.summary())
        english_cues = cues_mod.build(
            english.segments, cfg.cues, include_suppressed=not cfg.gating.drop_suppressed
        )
        if not english_cues:
            log.warning("translation produced no cues; skipping English output")
            return []

        return write.write_subtitles(
            english_cues, out_base, cfg.formats, "en", cfg.encoding
        )

    def _fill_gaps(
        self,
        wav: Path,
        language: str | None,
        transcript,
        duration: float,
        work: Path,
        report,
    ) -> float:
        """Decode long empty stretches again, each on its own. Returns seconds won.

        Whisper picks its own first timestamp inside every 30-second window, and
        when it picks a late one the audio before it is never transcribed at all —
        no warning, no empty segment, just absence. Measured on a Punjabi song: 80
        of 184 seconds produced nothing, and the three largest gaps each
        transcribed correctly when handed over as a clip. Repeated lines suffer
        most, because the repetition makes a window look like a failed decode.

        Handing the span back on its own removes the surrounding context that
        caused the skip. Anything recovered still goes through gating, which is
        what protects an instrumental passage from being filled with invention.
        """
        cfg = self.cfg
        clips = work / "gaps"
        won = 0.0
        # A recovered clip usually leaves a smaller gap of its own — the decode
        # skips the start of the clip exactly as it skipped the start of the
        # window. A second round picks those up; a third finds almost nothing and
        # is only more chances to hallucinate.
        for round_number in range(cfg.asr.gap_rounds):
            gaps = find_gaps(transcript.segments, duration, cfg.asr.gap_min_seconds)
            if not gaps:
                break
            log.info(
                "%d stretch(es) came back empty (%.0fs total); decoding them again",
                len(gaps), sum(end - start for start, end in gaps),
            )
            recovered: list = []
            for index, (start, end) in enumerate(gaps):
                report("transcribe", index / max(1, len(gaps)))
                clip = clips / f"r{round_number}g{index:02d}.wav"
                try:
                    extract.slice_audio(wav, clip, start, end - start)
                    piece = self.recognizer.transcribe(
                        clip, language=language, clip_offset=start,
                        vad_filter=False, force_sequential=True,
                    )
                except Exception:
                    log.warning(
                        "could not re-decode %.1f-%.1fs", start, end, exc_info=True
                    )
                    continue
                found = [s for s in piece.segments if s.text.strip()]
                if found:
                    log.info(
                        "  %.1f-%.1fs recovered %d segment(s): %s",
                        start, end, len(found), found[0].text.strip()[:60],
                    )
                    recovered.extend(found)

            if not recovered:
                break
            transcript.segments.extend(recovered)
            transcript.segments.sort(key=lambda s: s.start)
            won += sum(s.end - s.start for s in recovered)
        return won

    def process(
        self,
        source: Path,
        *,
        out_dir: Path | None = None,
        overwrite: bool = False,
        progress: Callable[[str, float], None] | None = None,
    ) -> Result:
        """Run the full pipeline.

        `progress(stage, fraction)` is called as each stage advances, so a UI
        can show where a long file actually is.
        """
        cfg = self.cfg

        def report(stage: str, fraction: float = 0.0) -> None:
            if progress:
                progress(stage, fraction)

        report("probe")
        info = probe.probe(source)
        if not info.has_audio:
            raise ValueError("no audio stream")
        stream = probe.choose_stream(info, prefer_language=cfg.asr.language)
        log.debug(
            "audio stream %d: %s %dch %s",
            stream.index, stream.codec, stream.channels, stream.layout or "-",
        )

        report("extract")
        work = cfg.work_dir / info.content_id
        wav = extract.extract_audio(
            info, stream, work / "audio.16k.wav", cfg.audio, overwrite=overwrite
        )

        language = cfg.asr.language
        confidence = 1.0
        alternatives: list[tuple[str, float]] = []
        if not language:
            report("detect")
            language, confidence, alternatives = self.recognizer.detect_language(wav)
            log.info("detected language: %s (%.0f%%)", language, confidence * 100)
            if confidence < cfg.qc.min_language_confidence:
                log.warning(
                    "language detection is unreliable here: %s at only %.0f%% "
                    "(next: %s). Pin the language explicitly.",
                    language, confidence * 100,
                    ", ".join(f"{l} {p:.0%}" for l, p in alternatives[1:4]) or "-",
                )

        report("transcribe")
        transcript = self.recognizer.transcribe(
            wav,
            language=language,
            progress=lambda f: report("transcribe", f),
        )

        # If speech detection rejected nearly the whole file, decode it again
        # with VAD off. Singing over dense instrumentation reads as non-speech to
        # Silero, which otherwise discards the entire file silently.
        audio_duration = transcript.duration or info.duration
        retried = False
        if (
            cfg.qc.enabled
            and cfg.qc.retry_without_vad
            and cfg.asr.vad_filter
            and audio_duration >= cfg.qc.min_duration_for_checks
        ):
            # Word-level coverage: a batched decode may hand back one segment
            # spanning the whole file that contains only seconds of speech, so
            # segment spans would report ~87% here and skip the retry entirely.
            covered = qc.speech_span(transcript.segments)
            if audio_duration > 0 and covered / audio_duration < cfg.qc.retry_coverage_threshold:
                log.warning(
                    "speech detection kept only %.1f%% of the audio; retrying "
                    "with VAD disabled",
                    100 * covered / audio_duration,
                )
                report("transcribe", 0.0)
                second = self.recognizer.transcribe(
                    wav,
                    language=language,
                    progress=lambda f: report("transcribe", f),
                    vad_filter=False,
                    force_sequential=True,
                )
                recovered = qc.speech_span(second.segments)
                if recovered > covered:
                    transcript = second
                    retried = True
                    log.info(
                        "no-VAD pass recovered %.1f%% coverage (was %.1f%%)",
                        100 * recovered / audio_duration,
                        100 * covered / audio_duration,
                    )
        filled = 0.0
        if cfg.asr.fill_gaps:
            filled = self._fill_gaps(
                wav, language, transcript, audio_duration, work, report
            )

        if not cfg.asr.language:
            transcript.language_probability = confidence
        if not transcript.duration:
            transcript.duration = info.duration

        report("gate")
        # Split to sentence granularity first, so gating decisions are
        # proportionate and the per-segment metrics mean something.
        if cfg.asr.resegment:
            before = len(transcript.segments)
            transcript.segments = resegment.split(
                transcript.segments,
                max_silence=cfg.asr.resegment_max_silence,
                max_duration=cfg.asr.resegment_max_duration,
            )
            log.debug("resegmented %d -> %d", before, len(transcript.segments))

        stats = gating.apply(transcript.segments, cfg.gating)
        log.info("gating: %s", stats.summary())

        built = cues_mod.build(
            transcript.segments,
            cfg.cues,
            include_suppressed=not cfg.gating.drop_suppressed,
        )

        # Judge the result as a whole. Per-segment gating cannot see that a file
        # produced almost nothing, because the survivors look locally fine.
        verdict = qc.evaluate(
            transcript.segments,
            built,
            transcript.duration or info.duration,
            transcript.language_probability,
            cfg.qc,
        )
        if retried:
            verdict.notes.insert(0, "Voice activity detection rejected nearly the "
                                    "whole file, so it was decoded again with VAD off.")
        truncated = self.recognizer.hotwords_note()
        if truncated:
            verdict.notes.append(truncated)
        for note in verdict.notes:
            log.warning("qc: %s", note)

        report("write")
        base_dir = out_dir if out_dir else source.parent
        out_base = self._claim(base_dir / source.stem, source, info.content_id)
        outputs = write.write_subtitles(
            built, out_base, cfg.formats, transcript.language, cfg.encoding
        )
        if cfg.romanize:
            extra_outputs, notes = write.write_second_script(
                built, out_base, cfg.formats, transcript.language, cfg.encoding,
                cfg.romanize_script,
            )
            outputs.extend(extra_outputs)
            for note in notes:
                # Silence here meant a ticked box that did nothing: the run
                # looked identical whether or not the script was available.
                verdict.notes.append(note)
                log.warning("qc: %s", note)

        if cfg.translate_to_english and transcript.language != cfg.translate_target:
            outputs.extend(
                self._translate(
                    wav, language, transcript, built, out_base, retried, report
                )
            )

        sidecar = write.write_sidecar(
            work / "transcript.sgen.json",
            source=source,
            content_id=info.content_id,
            config=cfg,
            transcript=transcript,
            cues=built,
            gate_summary=stats.summary(),
            extra={
                "audio": {
                    "wav": str(wav),
                    "stream_index": stream.index,
                    "channels": stream.channels,
                },
                "qc": verdict.to_dict(),
                "language_detection": {
                    "detected": language,
                    "confidence": confidence,
                    "alternatives": [{"language": l, "probability": p} for l, p in alternatives],
                    "pinned": bool(cfg.asr.language),
                },
                "retried_without_vad": retried,
                "gap_seconds_recovered": round(filled, 2),
            },
        )

        return Result(
            source=source,
            cues=built,
            language=transcript.language,
            language_probability=transcript.language_probability,
            duration=transcript.duration or info.duration,
            gate_summary=stats.summary(),
            suppressed_count=stats.suppressed,
            outputs=outputs,
            sidecar=sidecar,
            content_id=info.content_id,
            audio=wav,
            verdict=verdict,
        )


_TERMINALS = ".!?…।॥۔。！？"


def find_gaps(
    segments: Sequence[Segment], duration: float, min_seconds: float
) -> list[tuple[float, float]]:
    """Stretches of at least `min_seconds` that no segment covers.

    Includes the head and the tail: a decode that starts thirty seconds in has
    lost thirty seconds, and nothing downstream would ever notice.
    """
    if duration <= 0:
        return []
    spans = sorted((s.start, s.end) for s in segments)
    gaps: list[tuple[float, float]] = []
    edge = 0.0
    for start, end in spans:
        if start - edge >= min_seconds:
            gaps.append((edge, start))
        edge = max(edge, end)
    if duration - edge >= min_seconds:
        gaps.append((edge, duration))
    return gaps


def punctuation_density(segments: Sequence[Segment]) -> float:
    """Fraction of non-empty segments ending in sentence punctuation."""
    texts = [s.text.strip() for s in segments if not s.suppressed and s.text.strip()]
    if not texts:
        return 0.0
    ended = sum(1 for t in texts if t[-1] in _TERMINALS)
    return ended / len(texts)


def _pick_translate_engine(segments: Sequence[Segment], threshold: float = 0.3) -> str:
    """Choose a translation engine from how punctuated the transcript is.

    Text translation needs sentence boundaries; given a wall of unpunctuated
    text (sung lyrics) it rambles and repeats, while Whisper's audio translation
    stays coherent because it segments on the audio instead. Measured on a
    Bollywood track the text path produced repetitive nonsense where Whisper was
    fine, and on punctuated conversation the ranking reverses decisively.
    """
    density = punctuation_density(segments)
    engine = "nllb" if density >= threshold else "whisper"
    log.info(
        "translation engine: %s (%.0f%% of segments end in sentence punctuation)",
        engine, density * 100,
    )
    return engine


def _overlaps(segment: Segment, regions: Sequence[tuple[float, float]]) -> bool:
    """Whether `segment` shares any time with a region where speech was kept."""
    return any(segment.start < end and segment.end > start for start, end in regions)


def _segments_from_sidecar(data: dict) -> list[Segment]:
    segments: list[Segment] = []
    for raw in data.get("segments", []):
        segments.append(
            Segment(
                start=raw["start"],
                end=raw["end"],
                text=raw.get("text", ""),
                words=[
                    Word(w["start"], w["end"], w["text"], w.get("probability", 0.0))
                    for w in raw.get("words", [])
                ],
                avg_logprob=raw.get("avg_logprob", 0.0),
                no_speech_prob=raw.get("no_speech_prob", 0.0),
                compression_ratio=raw.get("compression_ratio", 1.0),
                temperature=raw.get("temperature", 0.0),
                language=raw.get("language"),
                suppressed=raw.get("suppressed", False),
                suppress_reason=raw.get("suppress_reason"),
            )
        )
    return segments


def _cues_from_sidecar(sidecar: Path) -> list[Cue]:
    """Load cues as they stand — hand-edited if edits exist, else generated."""
    import json

    data = write.load_sidecar(sidecar)
    edits = sidecar.parent / "edits.json"
    raw = (
        json.loads(edits.read_text(encoding="utf-8"))
        if edits.exists()
        else data.get("cues", [])
    )
    return [
        Cue(
            start=c["start"],
            end=c["end"],
            lines=c.get("lines") or [c.get("text", "")],
            warnings=list(c.get("warnings") or []),
        )
        for c in raw
    ]


def reformat_from_sidecar(sidecar: Path, cfg: Config) -> list[Path]:
    """Rebuild cues and subtitle files from a sidecar — no GPU needed."""
    data = write.load_sidecar(sidecar)
    segments = _segments_from_sidecar(data)
    if not segments:
        raise ValueError(f"{sidecar} contains no segments")

    # Re-run resegmentation and gating so threshold changes take effect too.
    if cfg.asr.resegment:
        segments = resegment.split(
            segments,
            max_silence=cfg.asr.resegment_max_silence,
            max_duration=cfg.asr.resegment_max_duration,
        )
    gating.apply(segments, cfg.gating)
    built = cues_mod.build(
        segments, cfg.cues, include_suppressed=not cfg.gating.drop_suppressed
    )

    source = Path(data["source"]["path"])
    out_base = source.parent / source.stem
    language = data.get("language", "und")
    outputs = write.write_subtitles(built, out_base, cfg.formats, language, cfg.encoding)
    # Also here, not only in a full run: the transcript is all romanization
    # needs, so a file already on disk should not have to go back to the GPU to
    # gain its Latin-script copy.
    if cfg.romanize:
        extra, notes = write.write_second_script(
            built, out_base, cfg.formats, language, cfg.encoding, cfg.romanize_script
        )
        outputs.extend(extra)
        for note in notes:
            log.warning("%s", note)
    return outputs
