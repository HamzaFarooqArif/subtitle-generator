"""Non-Latin script handling and orphan-cue merging.

Regression origin: Hindi lyrics produced a cue containing the single word "है",
because Devanagari ends sentences with the danda (।) rather than a full stop, so
the text looked entirely unpunctuated and packed on character count alone.
"""

from sgen.asr import Segment, Word
from sgen.config import CueConfig
from sgen import cues as cues_mod
from sgen import resegment

CFG = CueConfig()


def make_segment(text, start=0.0, step=0.5):
    tokens = text.split()
    words = [
        Word(start + i * step, start + (i + 1) * step, " " + t, 0.9)
        for i, t in enumerate(tokens)
    ]
    return Segment(start=start, end=start + len(tokens) * step, text=text, words=words)


def test_danda_is_treated_as_sentence_end():
    seg = make_segment("पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।")
    pieces = resegment.split([seg])
    assert len(pieces) == 3, [p.text for p in pieces]
    assert pieces[0].text.endswith("।")


def test_danda_splits_cues_too():
    seg = make_segment("पहला वाक्य है। दूसरा वाक्य है।")
    built = cues_mod.build([seg], CFG)
    assert len(built) >= 2
    assert built[0].flat.endswith("।")


def test_cjk_and_arabic_stops_recognized():
    for text, stop in (("これは 一つ。 これは 二つ。", "。"), ("جملة اولى۔ جملة ثانية۔", "۔")):
        seg = make_segment(text)
        pieces = resegment.split([seg])
        assert len(pieces) == 2, (text, [p.text for p in pieces])


def test_single_word_runt_is_merged_into_neighbour():
    """The exact 'है' orphan from the Khairiyat output."""
    seg = make_segment("तेरे बिन एक दिन जैसे सौ साल है")
    built = cues_mod.build([seg], CFG)
    for cue in built:
        assert len(cue.flat) >= CFG.min_cue_chars or len(built) == 1, cue.flat


def test_runt_merge_respects_line_budget():
    """Merging must never push a cue past the two-line character budget."""
    long_text = " ".join(["palabra"] * 14) + " y"
    seg = make_segment(long_text)
    built = cues_mod.build([seg], CFG)
    budget = CFG.max_lines * CFG.max_chars_per_line
    for cue in built:
        assert len(cue.flat) <= budget, (len(cue.flat), cue.flat)


def test_runt_merge_respects_max_duration():
    seg = make_segment(" ".join(["word"] * 40), step=0.6)
    for cue in cues_mod.build([seg], CFG):
        assert cue.duration <= CFG.max_duration + 1e-6


def test_runt_merge_can_be_disabled():
    cfg = CueConfig(merge_short_cues=False)
    seg = make_segment("तेरे बिन एक दिन जैसे सौ साल है")
    disabled = cues_mod.build([seg], cfg)
    enabled = cues_mod.build([seg], CFG)
    assert len(enabled) <= len(disabled)


def test_leading_runt_folded_into_the_next_cue():
    """A runt at position 0 has no previous neighbour to merge into."""
    words = [Word(0.0, 0.4, " है", 0.9)]
    words += [
        Word(0.5 + i * 0.4, 0.9 + i * 0.4, " " + t, 0.9)
        for i, t in enumerate("तुम्हारे बिन दिवाने का क्या हाल".split())
    ]
    seg = Segment(start=0.0, end=words[-1].end, text="है तुम्हारे बिन दिवाने का क्या हाल",
                  words=words)
    built = cues_mod.build([seg], CFG)
    assert built[0].flat.startswith("है तुम्हारे"), built[0].flat


def test_no_cue_is_dropped_by_merging():
    """Merging must preserve every word, not silently discard runts."""
    seg = make_segment("पहला वाक्य है। दो। तीसरा वाक्य यहाँ है।")
    built = cues_mod.build([seg], CFG)
    joined = " ".join(c.flat for c in built)
    for token in seg.text.split():
        assert token in joined, token


def test_orphan_control_when_words_are_long_in_time():
    """Sung words are long, so a chunk hits max_duration far short of the
    character budget — and the one-word tail is then too late to merge, because
    merging would exceed max_duration. Packing must avoid creating it.
    """
    # ~1.5 s per word: eight words is 12 s, well past max_duration of 7 s.
    seg = make_segment("तेरे बिन एक दिन जैसे सौ साल है", step=1.5)
    built = cues_mod.build([seg], CFG)
    assert built
    for cue in built:
        assert cue.duration <= CFG.max_duration + 1e-6
        assert len(cue.flat) >= CFG.min_cue_chars, cue.flat


def test_orphan_control_does_not_drop_words():
    seg = make_segment("तेरे बिन एक दिन जैसे सौ साल है", step=1.5)
    joined = " ".join(c.flat for c in cues_mod.build([seg], CFG))
    for token in seg.text.split():
        assert token in joined, token


def test_latin_behaviour_unchanged():
    seg = make_segment("This is the first one. And this is the second one.")
    built = cues_mod.build([seg], CFG)
    assert len(built) >= 2
    assert built[0].flat.endswith(".")
