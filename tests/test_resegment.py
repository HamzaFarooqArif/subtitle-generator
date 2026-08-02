from sgen.asr import Segment, Word
from sgen import resegment


def words_for(text, start=0.0, step=0.4):
    tokens = text.split()
    return [
        Word(start + i * step, start + (i + 1) * step, " " + t, 0.9)
        for i, t in enumerate(tokens)
    ]


def big_segment(text, **kw):
    words = words_for(text, **kw)
    return Segment(
        start=words[0].start, end=words[-1].end, text=text, words=words,
        avg_logprob=-0.2, no_speech_prob=0.05, compression_ratio=2.9,
    )


def test_splits_at_sentence_boundaries():
    seg = big_segment("This is one. This is two. This is three.")
    pieces = resegment.split([seg])
    assert len(pieces) == 3
    assert pieces[0].text == "This is one."
    assert pieces[2].text == "This is three."


def test_compression_ratio_recomputed_per_sentence():
    """The parent's ratio must not be inherited — that is the whole point.

    A long window whose ratio is pushed over threshold by repetition elsewhere
    would otherwise take every sentence in it down.
    """
    seg = big_segment("A perfectly ordinary sentence here. Another ordinary one.")
    pieces = resegment.split([seg])
    assert seg.compression_ratio == 2.9
    for piece in pieces:
        assert piece.compression_ratio < 2.4, (piece.text, piece.compression_ratio)


def test_window_level_metrics_are_inherited():
    seg = big_segment("Something spoken here. And more of it.")
    for piece in resegment.split([seg]):
        assert piece.no_speech_prob == seg.no_speech_prob
        assert piece.avg_logprob == seg.avg_logprob


def test_splits_on_long_silence_without_punctuation():
    words = words_for("look over there") + words_for("and then this", start=6.0)
    seg = Segment(start=0.0, end=words[-1].end, text="look over there and then this",
                  words=words)
    pieces = resegment.split([seg], max_silence=0.7)
    assert len(pieces) == 2
    assert pieces[0].text == "look over there"


def test_hard_duration_cap_splits_runaway_windows():
    seg = big_segment(" ".join(["word"] * 100), step=0.4)  # 40 s, no punctuation
    pieces = resegment.split([seg], max_duration=12.0)
    assert len(pieces) >= 3
    for piece in pieces:
        assert piece.end - piece.start <= 12.0 + 0.4


def test_does_not_split_decimals_or_initials():
    seg = big_segment("It cost 3.5 million in total.")
    pieces = resegment.split([seg])
    assert len(pieces) == 1


def test_segments_without_words_pass_through():
    seg = Segment(start=0.0, end=3.0, text="no words here", words=[])
    assert resegment.split([seg]) == [seg]


def test_text_rejoined_without_spurious_spaces():
    words = [
        Word(0.0, 0.4, " it", 0.9),
        Word(0.4, 0.8, "'s", 0.9),
        Word(0.8, 1.2, " fine.", 0.9),
    ]
    seg = Segment(start=0.0, end=1.2, text="it's fine.", words=words)
    pieces = resegment.split([seg])
    assert pieces[0].text == "it's fine."


def test_timings_come_from_words_not_parent():
    seg = big_segment("First one. Second one.")
    pieces = resegment.split([seg])
    assert pieces[0].start == seg.words[0].start
    assert pieces[1].end == seg.words[-1].end
    assert pieces[0].end <= pieces[1].start


def test_compression_ratio_of_empty_text():
    assert resegment.compression_ratio("") == 1.0
