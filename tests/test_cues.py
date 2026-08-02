from sgen.asr import Segment, Word
from sgen.config import CueConfig
from sgen import cues as cues_mod

CFG = CueConfig()


def make_segment(text, start=0.0, wps=2.5, prob=0.9):
    """Build a segment with evenly spaced word timings.

    Word tokens carry a leading space, matching what faster-whisper emits.
    """
    tokens = text.split()
    step = 1.0 / wps
    words = [
        Word(start + i * step, start + (i + 1) * step, " " + t, prob)
        for i, t in enumerate(tokens)
    ]
    return Segment(
        start=start, end=start + len(tokens) * step, text=text, words=words,
        avg_logprob=-0.2, no_speech_prob=0.05,
    )


def build(text, **kw):
    return cues_mod.build([make_segment(text, **kw)], CFG)


def test_short_line_stays_on_one_line():
    cues = build("Come over here.")
    assert len(cues) == 1
    assert cues[0].lines == ["Come over here."]


def test_long_text_breaks_into_two_lines_within_limit():
    cues = build("I really think we should go back inside now because it is getting cold")
    for cue in cues:
        assert len(cue.lines) <= CFG.max_lines
        for line in cue.lines:
            assert len(line) <= CFG.max_chars_per_line, line


def test_break_prefers_clause_punctuation():
    cues = build("She turned around slowly, and then she started laughing again")
    lines = cues[0].lines
    assert len(lines) == 2
    assert lines[0].endswith(","), lines


def test_never_breaks_after_a_determiner():
    cues = build("He carefully placed it down onto the enormous wooden kitchen table")
    for cue in cues:
        if len(cue.lines) == 2:
            last = cue.lines[0].split()[-1].lower().strip(".,")
            assert last not in cues_mod.DETERMINERS, cue.lines


def test_sentences_split_into_separate_cues():
    cues = build("This is the first one. And this is the second one.")
    assert len(cues) >= 2
    assert cues[0].flat.endswith(".")


def test_long_silence_splits_cues():
    a = make_segment("Look at this", start=0.0)
    b = make_segment("over there", start=a.end + 3.0)
    cues = cues_mod.build([a, b], CFG)
    assert len(cues) == 2


def test_reading_speed_extends_short_fast_cue():
    # Many characters spoken very fast: the cue must be stretched toward the
    # target CPS rather than left unreadable.
    cues = build("absolutely unbelievable circumstances surrounding everything", wps=9.0)
    for cue in cues:
        assert cue.cps <= CFG.max_cps + 1e-6 or "cps_" in " ".join(cue.warnings)


def test_min_duration_respected_where_room_exists():
    cues = build("Hey", wps=8.0)
    assert cues[0].duration >= CFG.min_duration - 1e-6


def test_max_duration_never_exceeded():
    long_text = " ".join(["word"] * 120)
    for cue in build(long_text, wps=1.2):
        assert cue.duration <= CFG.max_duration + 1e-6


def test_cues_are_ordered_and_never_overlap():
    long_text = (
        "We walked down to the river in the evening. It was colder than we "
        "expected, so we turned back early. Nobody said very much on the way home."
    )
    cues = build(long_text)
    assert len(cues) >= 3
    for a, b in zip(cues, cues[1:]):
        assert a.start <= b.start
        assert a.end <= b.start, (a.text, b.text)
        assert b.start - a.end >= CFG.min_gap - 1e-6


def test_suppressed_segments_excluded_by_default():
    good = make_segment("This part is fine.", start=0.0)
    bad = make_segment("Thanks for watching", start=5.0)
    bad.suppressed = True
    bad.suppress_reason = "hallucination_phrase"

    kept = cues_mod.build([good, bad], CFG)
    assert all("watching" not in c.flat for c in kept)

    with_all = cues_mod.build([good, bad], CFG, include_suppressed=True)
    assert any("watching" in c.flat for c in with_all)


def test_words_synthesized_when_model_gave_none():
    seg = Segment(start=0.0, end=4.0, text="No word level timings here", words=[])
    cues = cues_mod.build([seg], CFG)
    assert cues
    assert cues[0].start >= 0.0
    assert cues[0].end <= 4.0 + CFG.lead_out + CFG.max_duration


def test_empty_input_yields_no_cues():
    assert cues_mod.build([], CFG) == []


def test_apostrophes_are_not_split_by_a_space():
    """Regression: word tokens carry leading whitespace and must be joined raw.

    Stripping each token and rejoining on a space produced "o 'clock", which
    both corrupted the text and inflated the length enough to break two-line
    layout, spilling cues onto a third line.
    """
    words = [
        Word(0.0, 0.4, " meet", 0.9),
        Word(0.4, 0.8, " us", 0.9),
        Word(0.8, 1.2, " at", 0.9),
        Word(1.2, 1.6, " four", 0.9),
        Word(1.6, 2.0, " o", 0.9),
        Word(2.0, 2.4, "'clock", 0.9),
        Word(2.4, 2.8, " today.", 0.9),
    ]
    seg = Segment(start=0.0, end=2.8, text="meet us at four o'clock today.", words=words)
    cues = cues_mod.build([seg], CFG)
    assert cues[0].flat == "meet us at four o'clock today."
    assert " 'clock" not in cues[0].flat


def test_long_sentence_with_contraction_stays_within_two_lines():
    words_text = "My brother Thomas is going to meet us there at about 4 o'clock in the afternoon."
    tokens = words_text.split()
    words = [
        Word(i * 0.4, (i + 1) * 0.4, (" " if not t.startswith("'") else "") + t, 0.9)
        for i, t in enumerate(tokens)
    ]
    seg = Segment(start=0.0, end=len(tokens) * 0.4, text=words_text, words=words)
    cues = cues_mod.build([seg], CFG)
    assert len(cues) == 1
    assert len(cues[0].lines) == 2, cues[0].lines
    assert "overlong" not in cues[0].warnings
