"""Text translation: sentence splitting, timing redistribution, model wiring.

The sentence-splitting tests matter most. NLLB translates one sentence at a time
and silently drops everything after the first sentence boundary, so
"Where are you going? Come with me." came back as "Where are you going?" until
inputs were split.
"""

import pytest

from sgen import translate as mt
from sgen.asr import Segment, Word


# --------------------------------------------------------------------------- #
# sentence splitting
# --------------------------------------------------------------------------- #

def test_splits_on_terminal_punctuation():
    assert mt.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_splits_devanagari_danda_and_question_mark():
    out = mt.split_sentences("तुम कहाँ जा रहे हो? मेरे साथ चलो।")
    assert out == ["तुम कहाँ जा रहे हो?", "मेरे साथ चलो।"]


def test_does_not_split_decimals():
    assert mt.split_sentences("It cost 3.5 million.") == ["It cost 3.5 million."]


def test_does_not_split_abbreviations_midword():
    """A stop followed by a non-space is not a sentence boundary."""
    assert len(mt.split_sentences("Meet me at 4p.m. sharp.")) <= 2


def test_unpunctuated_text_is_one_sentence():
    assert mt.split_sentences("no punctuation here") == ["no punctuation here"]


def test_empty_input():
    assert mt.split_sentences("") == []
    assert mt.split_sentences("   ") == []


def test_closing_quotes_stay_with_their_sentence():
    out = mt.split_sentences('She said "go home." Then she left.')
    assert out[0].endswith('"')
    assert len(out) == 2


def test_cjk_and_arabic_stops():
    assert len(mt.split_sentences("これは一つ。これは二つ。")) == 2
    assert len(mt.split_sentences("جملة اولى۔ جملة ثانية۔")) == 2


# --------------------------------------------------------------------------- #
# language support
# --------------------------------------------------------------------------- #

def test_supported_languages():
    assert mt.supported("hi")
    assert mt.supported("de")
    assert mt.supported("en")
    assert not mt.supported("xx")
    assert not mt.supported(None)


def test_nllb_codes_are_flores_format():
    for code in ("hi", "de", "es", "en"):
        assert "_" in mt.NLLB_CODES[code]


# --------------------------------------------------------------------------- #
# sentence grouping and timing redistribution
# --------------------------------------------------------------------------- #

def seg(start, end, text):
    words = [Word(start, end, " " + t, 0.9) for t in text.split()]
    return Segment(start=start, end=end, text=text, words=words)


def test_fragments_are_grouped_into_a_sentence():
    """Unpunctuated fragments must be joined so the model sees a whole clause."""
    segments = [seg(0, 2, "तेरे बिन एक दिन"), seg(2, 4, "जैसे सौ साल है।")]
    groups = mt.group_sentences(segments)
    assert len(groups) == 1
    assert groups[0].start == 0 and groups[0].end == 4
    assert "सौ साल" in groups[0].text


def test_complete_sentences_stay_separate():
    segments = [seg(0, 2, "Erste Sache."), seg(2, 4, "Zweite Sache.")]
    assert len(mt.group_sentences(segments)) == 2


def test_grouping_respects_a_length_cap():
    segments = [seg(i, i + 1, "word " * 20) for i in range(10)]
    groups = mt.group_sentences(segments, max_chars=200)
    assert len(groups) > 1


def test_grouping_stops_before_a_long_time_span():
    """Merging unpunctuated cues over 26 s produced a run-on that translated
    far worse than the same lines translated individually, and had to be chopped
    back across cues afterwards. Real sentence fragments are short and adjacent.
    """
    # Six unpunctuated 6-second segments: 36 s if merged blindly.
    segments = [seg(i * 6, i * 6 + 6, "some unpunctuated speech here") for i in range(6)]
    groups = mt.group_sentences(segments, max_span=8.0)
    assert len(groups) > 1
    for group in groups:
        assert group.end - group.start <= 8.0 + 6.0, (group.start, group.end)


def test_short_translation_over_a_long_span_is_not_split():
    """A 29-character line stretched across 12 s exceeded max_duration and got
    split into "Yeah, I've / missed you / so much." Short text must take only the
    time it needs, anchored at the start of the span.
    """
    from sgen import cues as cues_mod
    from sgen.config import CueConfig

    cfg = CueConfig()
    members = [seg(0, 6, "aaa"), seg(6, 12, "bbb")]
    group = mt._Group("aaa bbb", 0.0, 12.0, members)
    out = mt._redistribute(group, "Yeah, I've missed you so much.",
                           target_cps=cfg.target_cps, max_duration=cfg.max_duration,
                           min_duration=max(1.0, cfg.min_duration))
    assert len(out) == 1
    assert out[0].end - out[0].start <= cfg.max_duration
    built = cues_mod.build(out, cfg)
    assert len(built) == 1, [c.flat for c in built]
    assert built[0].flat == "Yeah, I've missed you so much."


def test_long_translation_keeps_the_full_span_so_it_can_split():
    from sgen.config import CueConfig

    cfg = CueConfig()
    members = [seg(0, 10, "x"), seg(10, 20, "y")]
    group = mt._Group("x y", 0.0, 20.0, members)
    long_text = " ".join(["word"] * 60)
    out = mt._redistribute(group, long_text, target_cps=cfg.target_cps,
                           max_duration=cfg.max_duration, min_duration=1.0)
    assert out[0].end == 20.0


def test_translated_cues_do_not_split_sentences(monkeypatch):
    """End-to-end structural check: no cue may end mid-sentence."""
    from sgen import cues as cues_mod
    from sgen.config import CueConfig

    cfg = CueConfig()
    groups = [
        (mt._Group("a", 0.0, 12.0, [seg(0, 12, "a")]), "Yeah, I've missed you so much."),
        (mt._Group("b", 13.0, 25.0, [seg(13, 25, "b")]), "I don't know who's making that noise."),
        (mt._Group("c", 26.0, 34.0, [seg(26, 34, "c")]), "I decided to eat this one."),
    ]
    segments = []
    for group, text in groups:
        segments.extend(mt._redistribute(group, text, target_cps=cfg.target_cps,
                                         max_duration=cfg.max_duration, min_duration=1.0))
    built = cues_mod.build(segments, cfg)
    splits = sum(
        1 for a, b in zip(built, built[1:])
        if a.flat and a.flat[-1] not in ".!?…" and b.flat[:1].islower()
    )
    assert splits == 0, [c.flat for c in built]


def test_redistribution_returns_one_segment_per_group():
    """Chopping the translation by source character share produced fragments
    like "nothing Yes, of" / "course" — word order differs between languages, so
    a cut position derived from the source is meaningless in the target.
    """
    members = [seg(0, 2, "aa"), seg(2, 4, "bb"), seg(4, 6, "cc")]
    group = mt._Group("aa bb cc", 0.0, 6.0, members)
    out = mt._redistribute(group, "one two three four five six")
    assert len(out) == 1
    assert out[0].start == 0.0 and out[0].end <= 6.0
    assert out[0].text == "one two three four five six"


def test_redistribution_starts_at_the_span_and_never_exceeds_it():
    """Timing contract: anchored at the group start, never past the group end.

    The end is no longer forced to the span end — short text takes only the time
    it needs, which is what stops the cue builder fragmenting it.
    """
    members = [seg(0, 2, "aaa aaa"), seg(2, 4, "bbb bbb"), seg(4, 6, "ccc ccc")]
    group = mt._Group("aaa aaa bbb bbb ccc ccc", 0.0, 6.0, members)
    out = mt._redistribute(group, "one two three four five six")
    assert out[0].start == 0.0
    assert out[-1].end <= 6.0
    for piece in out:
        assert piece.text.strip()


def test_redistribution_never_drops_words():
    members = [seg(0, 2, "aa"), seg(2, 4, "bb"), seg(4, 6, "cc")]
    group = mt._Group("aa bb cc", 0.0, 6.0, members)
    text = "alpha beta gamma delta epsilon"
    out = mt._redistribute(group, text)
    joined = " ".join(p.text for p in out).split()
    assert joined == text.split()


def test_short_translation_stays_as_one_segment():
    members = [seg(0, 2, "aa"), seg(2, 4, "bb"), seg(4, 6, "cc")]
    group = mt._Group("aa bb cc", 0.0, 6.0, members)
    out = mt._redistribute(group, "hi")
    assert len(out) == 1
    assert out[0].start == 0.0
    assert 0.0 < out[0].end <= 6.0


def test_single_member_group():
    members = [seg(1.0, 3.0, "hallo")]
    group = mt._Group("hallo", 1.0, 3.0, members)
    out = mt._redistribute(group, "hello there")
    assert len(out) == 1
    assert out[0].start == 1.0
    assert out[0].end <= 3.0


def test_generated_segments_carry_word_timings():
    """Cue building needs word timings; without them timing is synthesized badly."""
    members = [seg(0, 4, "test")]
    group = mt._Group("test", 0.0, 4.0, members)
    out = mt._redistribute(group, "one two three")
    assert out[0].words
    assert out[0].words[0].start >= 0.0
    assert out[0].words[-1].end <= 4.0
    for a, b in zip(out[0].words, out[0].words[1:]):
        assert a.start <= b.start


def test_translated_segments_are_not_gated_by_source_confidence():
    """no_speech_prob must be neutral: it describes audio, not translation."""
    members = [seg(0, 4, "x")]
    members[0].no_speech_prob = 0.95
    group = mt._Group("x", 0.0, 4.0, members)
    out = mt._redistribute(group, "translated text here")
    assert out[0].no_speech_prob == 0.0


# --------------------------------------------------------------------------- #
# config wiring
# --------------------------------------------------------------------------- #

def test_config_defaults_to_automatic_engine_choice():
    """Neither engine wins everywhere, so the default measures and picks."""
    from sgen.config import Config

    cfg = Config()
    assert cfg.translate_engine == "auto"
    assert cfg.translate_target == "en"
    assert cfg.translate_model == "auto"


def test_ui_options_map_to_config():
    from sgen.server.jobs import build_config

    cfg = build_config({"translate": True, "translate_engine": "whisper",
                        "translate_target": "de"})
    assert cfg.translate_to_english is True
    assert cfg.translate_engine == "whisper"
    assert cfg.translate_target == "de"


def test_invalid_engine_is_ignored_not_crashing():
    from sgen.server.jobs import build_config

    assert build_config({"translate_engine": "bogus"}).translate_engine == "auto"


def test_output_cleanup_fixes_detokenization_gaps():
    """Sentencepiece detokenization leaves "I 'm not joking"."""
    assert mt._clean_output("I 'm not joking") == "I'm not joking"
    assert mt._clean_output("It 's fine , really .") == "It's fine, really."
    assert mt._clean_output("don 't touch it") == "don't touch it"


def test_output_cleanup_strips_invented_dialogue_dash():
    assert mt._clean_output("- Good morning to you.") == "Good morning to you."
    assert mt._clean_output("— Hello") == "Hello"


def test_output_cleanup_leaves_normal_text_alone():
    for text in ("This is fine.", "Rock 'n' roll", "A (parenthetical) aside."):
        assert mt._clean_output(text) == text


def test_unpunctuated_segments_are_not_merged():
    """Merging unpunctuated lyrics into a run-on made the model repeat itself."""
    lyrics = [seg(i, i + 1, f"line {i} of the song") for i in range(6)]
    groups = mt.group_sentences(lyrics)
    assert len(groups) == 6, "each unpunctuated line must translate on its own"
    assert groups[0].text == "line 0 of the song"


def test_punctuated_fragments_still_merge():
    segments = [seg(0, 2, "tere bin ek din"), seg(2, 4, "jaise sau saal hai.")]
    assert len(mt.group_sentences(segments)) == 1


def test_punctuation_density():
    from sgen.pipeline import punctuation_density

    assert punctuation_density([seg(0, 1, "One."), seg(1, 2, "Two.")]) == 1.0
    assert punctuation_density([seg(0, 1, "no stop"), seg(1, 2, "none here")]) == 0.0
    assert punctuation_density([seg(0, 1, "One."), seg(1, 2, "no stop")]) == 0.5
    assert punctuation_density([]) == 0.0


def test_punctuation_density_ignores_suppressed():
    from sgen.pipeline import punctuation_density

    a, b = seg(0, 1, "One."), seg(1, 2, "no stop")
    b.suppressed = True
    assert punctuation_density([a, b]) == 1.0


def test_auto_engine_picks_text_translation_for_prose():
    from sgen.pipeline import _pick_translate_engine

    prose = [seg(i, i + 1, f"Sentence number {i}.") for i in range(6)]
    assert _pick_translate_engine(prose) == "nllb"


def test_auto_engine_picks_audio_translation_for_unpunctuated_lyrics():
    """Text translation rambles without sentence boundaries; Whisper does not."""
    from sgen.pipeline import _pick_translate_engine

    lyrics = [seg(i, i + 1, "khairiyat poocho kabhi to") for i in range(8)]
    assert _pick_translate_engine(lyrics) == "whisper"


def test_auto_engine_threshold_boundary():
    from sgen.pipeline import _pick_translate_engine

    # 2 of 5 punctuated = 0.4, above the 0.3 default.
    mixed = [seg(0, 1, "One."), seg(1, 2, "Two."), seg(2, 3, "three"),
             seg(3, 4, "four"), seg(4, 5, "five")]
    assert _pick_translate_engine(mixed) == "nllb"
    assert _pick_translate_engine(mixed, threshold=0.5) == "whisper"


def test_best_mt_model_prefers_the_larger_one(monkeypatch):
    from sgen import models

    monkeypatch.setattr(models, "mt_path", lambda name: f"/fake/{name}")
    assert models.best_mt_model() == "nllb-1.3b"


def test_best_mt_model_falls_back_to_smaller(monkeypatch):
    from sgen import models

    def only_600m(name):
        if name == "nllb-600m":
            return "/fake/600m"
        raise models.ModelMissing(name)

    monkeypatch.setattr(models, "mt_path", only_600m)
    assert models.best_mt_model() == "nllb-600m"


def test_best_mt_model_raises_when_none_installed(monkeypatch):
    from sgen import models

    def none(name):
        raise models.ModelMissing(name)

    monkeypatch.setattr(models, "mt_path", none)
    with pytest.raises(models.ModelMissing):
        models.best_mt_model()


def test_missing_translation_model_raises_model_missing():
    from sgen import models

    with pytest.raises(models.ModelMissing):
        models.mt_path("nllb-does-not-exist")
