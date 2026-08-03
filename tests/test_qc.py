"""File-level plausibility checks.

Regression origin: a 239-second Hindi song produced one 0.54-second cue of
Turkish text and was reported as a success, because every gate in gating.py is
per-segment and that single segment looked locally fine.
"""

from sgen.asr import Segment, Word
from sgen.config import QcConfig
from sgen.cues import Cue
from sgen import qc

CFG = QcConfig()


def seg(start, end, text="some words here", suppressed=False):
    words = [Word(start, end, " " + t, 0.9) for t in text.split()]
    s = Segment(start=start, end=end, text=text, words=words)
    s.suppressed = suppressed
    return s


def cue(start, end, text="some words here"):
    return Cue(start=start, end=end, lines=[text])


def test_healthy_file_is_not_suspect():
    segments = [seg(i * 10, i * 10 + 9) for i in range(20)]
    cues = [cue(i * 10, i * 10 + 9) for i in range(20)]
    v = qc.evaluate(segments, cues, 200.0, 0.99, CFG)
    assert not v.suspect, v.warnings
    assert v.coverage > 0.85


def test_the_hindi_song_case_is_flagged():
    """One 0.54 s cue in a 239 s file: the exact failure that motivated this."""
    segments = [seg(116.86, 117.40, "Sağolun.")]
    cues = [cue(116.86, 117.40, "Sağolun.")]
    v = qc.evaluate(segments, cues, 238.56, 0.2119, CFG)

    assert v.suspect
    assert "low_coverage" in v.warnings
    assert "asr_found_little_speech" in v.warnings
    assert "too_few_segments" in v.warnings
    assert "low_language_confidence" in v.warnings
    assert v.coverage < 0.01
    assert any("music" in n or "voice activity" in n for n in v.notes)


def test_low_language_confidence_flagged_even_when_coverage_is_fine():
    segments = [seg(i * 10, i * 10 + 9) for i in range(20)]
    cues = [cue(i * 10, i * 10 + 9) for i in range(20)]
    v = qc.evaluate(segments, cues, 200.0, 0.21, CFG)
    assert "low_language_confidence" in v.warnings
    assert "low_coverage" not in v.warnings


def test_gating_loss_is_reported_as_gating_not_as_silence():
    """Speech found but gated away is a different problem from speech not found.

    Saying "most of this file produced nothing" when the model produced plenty
    and the gate discarded it points the user at the wrong fix.
    """
    # Healthy ASR coverage (32%), but 75% of segments gated -> cue coverage 8%.
    segments = [seg(i * 25, i * 25 + 8, suppressed=i % 4 != 0) for i in range(40)]
    cues = [cue(i * 25, i * 25 + 8) for i in range(40) if i % 4 == 0]
    v = qc.evaluate(segments, cues, 1000.0, 0.91, CFG)
    assert v.asr_coverage > CFG.min_coverage, v.asr_coverage
    assert v.coverage < CFG.min_coverage, v.coverage

    assert "mostly_gated" in v.warnings
    assert "asr_found_little_speech" not in v.warnings
    assert "low_coverage" not in v.warnings
    assert any("gated" in n for n in v.notes)
    assert not any("produced nothing" in n for n in v.notes)


def test_genuine_asr_failure_still_says_produced_nothing():
    segments = [seg(116.86, 117.40, "Sağolun.")]
    cues = [cue(116.86, 117.40, "Sağolun.")]
    v = qc.evaluate(segments, cues, 238.56, 0.21, CFG)
    assert "asr_found_little_speech" in v.warnings
    assert "low_coverage" in v.warnings
    assert "mostly_gated" not in v.warnings


def test_heavily_gated_file_flagged():
    segments = [seg(i * 10, i * 10 + 9, suppressed=i > 2) for i in range(20)]
    cues = [cue(i * 10, i * 10 + 9) for i in range(3)]
    v = qc.evaluate(segments, cues, 200.0, 0.99, CFG)
    assert "heavily_gated" in v.warnings


def test_short_clips_are_not_judged_on_coverage():
    """A 5-second clip with one short cue is normal, not a failure."""
    segments = [seg(1.0, 1.6)]
    cues = [cue(1.0, 1.6)]
    v = qc.evaluate(segments, cues, 5.0, 0.99, CFG)
    assert "low_coverage" not in v.warnings
    assert "too_few_segments" not in v.warnings


def test_overlapping_spans_do_not_inflate_coverage():
    """Merged spans must never let coverage exceed 100%."""
    cues = [cue(0, 60), cue(10, 70), cue(20, 80)]
    v = qc.evaluate([seg(0, 80)], cues, 100.0, 0.99, CFG)
    assert v.coverage <= 1.0
    assert abs(v.coverage - 0.8) < 1e-6


def test_zero_duration_is_handled():
    v = qc.evaluate([], [], 0.0, 0.0, CFG)
    assert v.coverage == 0.0
    assert not v.notes  # nothing meaningful to say about a zero-length file


def test_disabled_checks_via_thresholds():
    """The music profile relaxes these; make sure that actually takes effect."""
    lenient = QcConfig(min_coverage=0.0, max_suppressed_fraction=1.0,
                       min_language_confidence=0.0, min_segments=0)
    segments = [seg(116.86, 117.40, "Sağolun.")]
    cues = [cue(116.86, 117.40, "Sağolun.")]
    v = qc.evaluate(segments, cues, 238.56, 0.21, lenient)
    assert not v.suspect, v.warnings


def test_notes_accompany_every_warning():
    segments = [seg(116.86, 117.40)]
    cues = [cue(116.86, 117.40)]
    v = qc.evaluate(segments, cues, 238.56, 0.21, CFG)
    # Each warning should be explained to the user, not just named.
    assert len(v.notes) >= len(v.warnings) - 1
    assert all(isinstance(n, str) and n for n in v.notes)


def test_verdict_serializes():
    v = qc.evaluate([seg(0, 50)], [cue(0, 50)], 100.0, 0.9, CFG)
    d = v.to_dict()
    assert set(d) >= {"suspect", "coverage", "warnings", "notes"}
    assert isinstance(d["suspect"], bool)


def test_segment_spans_lie_word_spans_dont():
    """Regression: a batched decode returned ONE segment spanning 1409 s of a
    1618 s file while containing ~12 s of words. Measuring the segment span put
    coverage at 87%, so the no-VAD retry was skipped and 26x more speech was
    thrown away. Coverage must be measured from words.
    """
    words = [Word(34.0 + i, 34.5 + i, " word", 0.9) for i in range(12)]
    giant = Segment(start=34.58, end=1443.68, text="a few words", words=words)

    assert qc._span([giant]) > 1400              # what the old check saw
    assert qc.speech_span([giant]) < 15          # what is actually there

    v = qc.evaluate([giant], [], 1618.26, 0.91, CFG)
    assert v.asr_coverage < 0.02
    assert "asr_found_little_speech" in v.warnings


def test_speech_span_falls_back_to_segment_without_words():
    seg = Segment(start=10.0, end=20.0, text="no word timings", words=[])
    assert qc.speech_span([seg]) == 10.0


def test_speech_span_merges_overlapping_words():
    words = [Word(0.0, 5.0, " a", 0.9), Word(2.0, 7.0, " b", 0.9)]
    seg = Segment(start=0.0, end=7.0, text="a b", words=words)
    assert abs(qc.speech_span([seg]) - 7.0) < 1e-6


def test_sagolun_is_now_blacklisted():
    """The specific hallucination that slipped through must be caught."""
    from sgen import gating
    from sgen.config import GatingConfig

    s = Segment(start=116.86, end=117.40, text="Sağolun.",
                words=[Word(116.86, 117.40, " Sağolun.", 0.47)],
                avg_logprob=-0.74, no_speech_prob=0.74, compression_ratio=0.53)
    gating.apply([s], GatingConfig())
    assert s.suppressed
    assert s.suppress_reason == "hallucination_phrase"


# --------------------------------------------------------------------------- #
# gaps where the decoder produced nothing
#
# Whisper picks its own first timestamp inside every 30-second window, and a late
# pick means the audio before it is never transcribed — no warning, no empty
# segment, just absence. Measured on a Punjabi song: 80 of 184 seconds produced
# nothing, including the repeated title hook, and each gap transcribed correctly
# when handed back as a clip of its own.
# --------------------------------------------------------------------------- #

def _span(start, end, text="x"):
    from sgen.asr import Segment

    return Segment(start=start, end=end, text=text)


def test_no_gaps_in_continuous_speech():
    from sgen.pipeline import find_gaps

    segments = [_span(0, 10), _span(10, 20), _span(20, 30)]
    assert find_gaps(segments, 30.0, 6.0) == []


def test_a_long_hole_in_the_middle_is_found():
    from sgen.pipeline import find_gaps

    segments = [_span(0, 14.5), _span(38.6, 43.3)]
    assert find_gaps(segments, 43.3, 6.0) == [(14.5, 38.6)]


def test_short_holes_are_left_alone():
    """Two seconds of room noise decoded again is where hallucinations come
    from; only stretches worth a second pass are returned."""
    from sgen.pipeline import find_gaps

    segments = [_span(0, 10), _span(13, 20)]
    assert find_gaps(segments, 20.0, 6.0) == []


def test_the_head_counts():
    """A decode that starts thirty seconds in has lost thirty seconds, and
    nothing downstream would notice."""
    from sgen.pipeline import find_gaps

    assert find_gaps([_span(30, 40)], 40.0, 6.0) == [(0.0, 30.0)]


def test_the_tail_counts():
    from sgen.pipeline import find_gaps

    assert find_gaps([_span(0, 10)], 40.0, 6.0) == [(10.0, 40.0)]


def test_overlapping_segments_do_not_invent_a_gap():
    from sgen.pipeline import find_gaps

    segments = [_span(0, 20), _span(5, 12), _span(19, 30)]
    assert find_gaps(segments, 30.0, 6.0) == []


def test_an_empty_transcript_is_one_whole_gap():
    from sgen.pipeline import find_gaps

    assert find_gaps([], 100.0, 6.0) == [(0.0, 100.0)]


def test_unknown_duration_is_not_a_gap():
    from sgen.pipeline import find_gaps

    assert find_gaps([_span(0, 10)], 0.0, 6.0) == []


def test_songs_allow_a_chorus_to_repeat():
    """A chorus is the same line three or four times over; the default reads the
    third as a decode loop and deletes it."""
    from sgen.config import Config

    assert Config.load("music").gating.max_repeat_of_neighbour > Config().gating.max_repeat_of_neighbour


# --------------------------------------------------------------------------- #
# One alphabet per transcript
#
# Every other check here is acoustic, so a decode that put an English word into a
# Punjabi song, or spelt one letter of a Gurmukhi word in Devanagari, scored
# perfectly and was written out with suspect: false. Both cases are below.
# --------------------------------------------------------------------------- #

PUNJABI = "ਤੇਰੇ ਪੀਛੇ ਆ ਗਵਾਚੀ ਨਾਲ ਪਿਆਰ ਕਰਨਾ ਸੋਹਣਾ ਮੁੰਡਿਆ ਵੇ"


def _scripts(lines, cfg=None):
    verdict = qc.Verdict()
    qc.check_scripts([Cue(start=0, end=1, lines=[l]) for l in lines],
                     verdict, cfg or QcConfig())
    return verdict


def test_a_clean_transcript_passes():
    assert not _scripts([PUNJABI] * 4).suspect
    assert not _scripts(["खैरियत पूछो कभी तो कैफियत पूछो"]).suspect
    assert not _scripts(["The quick brown fox", "jumped over it"]).suspect


def test_a_word_spelt_in_two_alphabets_is_flagged():
    """The real case: “तੇਰੇ” — one Devanagari letter inside a Gurmukhi word."""
    verdict = _scripts([PUNJABI, "तੇਰੇ ਪੀਛੇ ਆ ਗਵਾਚੀ"] + [PUNJABI] * 3)
    assert "mixed_script_words" in verdict.warnings
    assert "तੇਰੇ" in verdict.notes[0]


def test_latin_letters_inside_an_indic_word_are_flagged():
    """Also real, from the same file: “wzglਾ” and “ਬੀਨਾourtਸੀ”."""
    assert "mixed_script_words" in _scripts([PUNJABI, "wzglਾ ਬੀਨਾourtਸੀ"]).warnings


def test_a_stray_foreign_word_is_flagged():
    """“shipped”, in the middle of a Punjabi song."""
    verdict = _scripts(["ਵੇ ਤੁ ਲ shipped ਵੇ ਮੈ ਲਾਚੀ"] + [PUNJABI] * 4)
    assert "foreign_script_words" in verdict.warnings
    assert "shipped" in " ".join(verdict.notes)


def test_genuine_code_switching_is_not_a_fault():
    """Hinglish is real speech, not a decode failure. Too many English words to
    be corruption means it is someone talking that way."""
    verdict = _scripts([
        "मैंने phone किया और WhatsApp पे message bheja",
        "office se ghar aaya फिर मैंने खाना खाया",
    ])
    assert "foreign_script_words" not in verdict.warnings


def test_the_threshold_is_what_separates_the_two():
    """Same words, different share of the file — only the sprinkle is reported."""
    sprinkle = ["one two three"] + [PUNJABI] * 20
    assert "foreign_script_words" in _scripts(sprinkle).warnings
    assert "foreign_script_words" not in _scripts(["one two three", PUNJABI]).warnings


def test_two_indic_alphabets_in_quantity_are_flagged():
    """The Khairiyat failure: Gurmukhi and Devanagari throughout, and QC said
    nothing because every segment looked acoustically fine."""
    verdict = _scripts([PUNJABI, "खैरियत पूछो कभी तो कैफियत पूछो मेरा दिल"] * 3)
    assert "several_scripts" in verdict.warnings


def test_latin_alongside_an_indic_script_is_not_several_scripts():
    """Latin mixes legitimately with everything — brand names, place names."""
    verdict = _scripts(["मैंने WhatsApp पे message bheja aur phone kiya",
                        "Thomas ne Kreuzberg mein photo liya tha"])
    assert "several_scripts" not in verdict.warnings


def test_the_check_can_be_turned_off():
    cfg = QcConfig(check_scripts=False)
    verdict = qc.evaluate([seg(0, 10)], [cue(0, 10, "तੇਰੇ wzglਾ")], 100.0, 1.0, cfg)
    assert "mixed_script_words" not in verdict.warnings


def test_evaluate_wires_the_check_in():
    """It has to run from evaluate(), not only when called directly."""
    verdict = qc.evaluate(
        [seg(0, 90)], [cue(0, 90, "तੇਰੇ ਪੀਛੇ ਆ ਗਵਾਚੀ")], 100.0, 1.0, QcConfig()
    )
    assert "mixed_script_words" in verdict.warnings
    assert verdict.suspect


def test_every_script_warning_has_a_note():
    verdict = _scripts([PUNJABI, "तੇਰੇ wzglਾ shipped"] + [PUNJABI] * 3)
    assert len(verdict.notes) >= len(verdict.warnings)


def test_punctuation_and_digits_are_not_a_script():
    assert not _scripts(["ਤੇਰੇ 42 ਪੀਛੇ, ਆ — ਗਵਾਚੀ! 3.14"]).suspect
