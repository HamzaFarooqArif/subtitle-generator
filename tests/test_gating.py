from sgen.asr import Segment, Word
from sgen.config import GatingConfig
from sgen import gating


def seg(text, *, start=0.0, end=2.0, no_speech=0.1, logprob=-0.3,
        compression=1.5, word_prob=0.9):
    tokens = text.split() or ["x"]
    step = (end - start) / len(tokens)
    words = [
        Word(start + i * step, start + (i + 1) * step, t, word_prob)
        for i, t in enumerate(tokens)
    ]
    return Segment(
        start=start, end=end, text=text, words=words,
        avg_logprob=logprob, no_speech_prob=no_speech,
        compression_ratio=compression,
    )


CFG = GatingConfig()


def reasons(segments):
    gating.apply(segments, CFG)
    return [s.suppress_reason for s in segments]


def test_clean_speech_survives():
    segments = [seg("We should head back before it gets dark.")]
    assert reasons(segments) == [None]
    assert not segments[0].suppressed


def test_known_hallucination_phrases_suppressed():
    for text in [
        "Thanks for watching!",
        "Untertitel von ZDF, 2020",
        "Subtítulos realizados por la comunidad de Amara.org",
        "[Music]",
    ]:
        assert reasons([seg(text)]) == ["hallucination_phrase"], text


def test_non_lexical_suppressed_only_when_model_unsure():
    # Non-speech vocalization: low word confidence -> suppressed.
    assert reasons([seg("Ah. Ah. Mmm.", word_prob=0.2)])[0] is not None
    # A clearly articulated "Yeah" is real speech and must survive.
    assert reasons([seg("Yeah", word_prob=0.95, no_speech=0.05)]) == [None]


def test_repeat_loop_suppressed():
    text = "oh my god oh my god oh my god oh my god"
    assert reasons([seg(text)]) == ["repeat_loop"]


def test_high_compression_ratio_suppressed():
    assert reasons([seg("something something", compression=3.1)]) == ["repetition"]


def test_sparse_text_over_long_span_suppressed():
    # Two words smeared over nine seconds: the signature of the model reaching
    # for lexical content in audio that has none.
    assert reasons([seg("oh yes", start=0.0, end=9.0)]) == ["sparse_text"]


def test_hard_no_speech_probability_suppressed():
    assert reasons([seg("I think so", no_speech=0.95)]) == ["no_speech"]


def test_very_low_confidence_suppressed():
    assert reasons([seg("indistinct words here", logprob=-2.0)]) == ["very_low_confidence"]


def test_duplicate_neighbour_suppressed():
    segments = [
        seg("You know what I mean", start=0, end=2),
        seg("You know what I mean", start=2, end=4),
        seg("You know what I mean", start=4, end=6),
    ]
    got = reasons(segments)
    assert got[0] is None
    assert got[2] == "duplicate_neighbour"


def test_gate_can_be_disabled():
    cfg = GatingConfig(enabled=False)
    segments = [seg("Thanks for watching!")]
    stats = gating.apply(segments, cfg)
    assert stats.suppressed == 0
    assert not segments[0].suppressed


def test_stats_summary_counts():
    segments = [seg("Thanks for watching!"), seg("A real sentence here.")]
    stats = gating.apply(segments, CFG)
    assert stats.total == 2
    assert stats.suppressed == 1
    assert stats.kept == 1
    assert "hallucination_phrase=1" in stats.summary()


def test_normalize_strips_accents_and_punctuation():
    assert gating.normalize("¡Gracias, por ver!") == "gracias por ver"
