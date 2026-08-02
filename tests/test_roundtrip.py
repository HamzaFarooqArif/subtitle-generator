"""External-translation round trip.

Local models do not match a production translator, so the text can be exported,
translated anywhere, and imported back onto the original timings. Translators are
unreliable about line counts, which is why lines are numbered.
"""

import pytest

from sgen import roundtrip
from sgen.config import CueConfig
from sgen.cues import Cue

CUES = [
    Cue(start=1.0, end=3.0, lines=["खैरियत पूछो कभी तो", "कैफियत पूछो"]),
    Cue(start=3.2, end=5.5, lines=["तुम्हारे बिन दिवाने का क्या हाल है"]),
    Cue(start=6.0, end=8.0, lines=["ये दूरियां फिल्हाल है"]),
]


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #

def test_export_numbers_each_cue():
    out = roundtrip.export_text(CUES).splitlines()
    assert len(out) == 3
    assert out[0].startswith("1. ")
    assert out[2].startswith("3. ")


def test_export_flattens_internal_line_breaks():
    """Two display lines are one utterance; a translator must see one line."""
    first = roundtrip.export_text(CUES).splitlines()[0]
    assert "\n" not in first
    assert "कभी तो कैफियत" in first


def test_export_skips_empty_cues():
    cues = [*CUES, Cue(start=9.0, end=10.0, lines=[""])]
    assert len(roundtrip.export_text(cues).splitlines()) == 3


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #

def test_import_numbered_translation():
    text = "1. Ask about the wellbeing\n2. How is the madman without you\n3. These distances remain"
    out, report = roundtrip.apply_translation(CUES, text)
    assert report.matched == 3
    assert report.method == "numbered"
    assert out[0].lines == ["Ask about the wellbeing"]
    assert not report.missing


def test_import_preserves_every_timing():
    text = "1. one\n2. two\n3. three"
    out, _ = roundtrip.apply_translation(CUES, text)
    for original, translated in zip(CUES, out):
        assert translated.start == original.start
        assert translated.end == original.end


def test_import_tolerates_alternative_numbering_styles():
    for text in (
        "1) one\n2) two\n3) three",
        "[1] one\n[2] two\n[3] three",
        "1 - one\n2 - two\n3 - three",
        "1. one\n\n2. two\n\n3. three",
    ):
        out, report = roundtrip.apply_translation(CUES, text)
        assert report.matched == 3, text


def test_import_falls_back_to_position_without_numbers():
    """Some translators strip the numbers; line count still saves us."""
    text = "one\ntwo\nthree"
    out, report = roundtrip.apply_translation(CUES, text)
    assert report.matched == 3
    assert report.method == "positional"
    assert out[1].lines == ["two"]


def test_import_reports_missing_lines():
    text = "1. one\n3. three"
    out, report = roundtrip.apply_translation(CUES, text)
    assert report.matched == 2
    assert report.missing == [2]
    assert "untranslated" in " ".join(str(c.warnings) for c in out)


def test_untranslated_cue_keeps_source_text_by_default():
    out, _ = roundtrip.apply_translation(CUES, "1. one\n3. three")
    assert len(out) == 3
    assert "तुम्हारे" in out[1].flat
    assert "untranslated" in out[1].warnings


def test_untranslated_cue_can_be_dropped():
    out, _ = roundtrip.apply_translation(CUES, "1. one\n3. three",
                                         keep_untranslated=False)
    assert len(out) == 2


def test_import_rejects_completely_unmatchable_text():
    out, report = roundtrip.apply_translation(CUES, "this is not aligned at all")
    assert not report.ok or report.matched < 3


def test_translator_merging_a_split_line_is_rejoined():
    """A translator may wrap one numbered cue onto two lines."""
    text = "1. Ask about\n1. the wellbeing\n2. two\n3. three"
    out, report = roundtrip.apply_translation(CUES, text)
    assert "Ask about the wellbeing" == out[0].flat


def test_out_of_range_numbers_ignored():
    text = "1. one\n2. two\n3. three\n99. stray line"
    out, report = roundtrip.apply_translation(CUES, text)
    assert report.matched == 3
    assert len(out) == 3


def test_report_summary_is_informative():
    _, report = roundtrip.apply_translation(CUES, "1. one\n3. three")
    summary = report.summary()
    assert "2/3" in summary
    assert "untranslated" in summary


# --------------------------------------------------------------------------- #
# re-breaking after import
# --------------------------------------------------------------------------- #

def test_rebreak_applies_line_limits_to_translated_text():
    """Translated text is a different length, so line breaks must be redone."""
    long_line = "This is a considerably longer English rendering of the original line."
    cues = [Cue(start=0.0, end=5.0, lines=[long_line])]
    cfg = CueConfig()
    out = roundtrip.rebreak(cues, cfg)
    assert out
    for cue in out:
        for line in cue.lines:
            assert len(line) <= cfg.max_chars_per_line or "overlong" in cue.warnings


def test_rebreak_keeps_text_within_the_original_span():
    cues = [Cue(start=10.0, end=14.0, lines=["Some translated text here now"])]
    out = roundtrip.rebreak(cues, CueConfig())
    assert out[0].start >= 10.0 - 0.1
    assert out[-1].end <= 14.0 + 0.2


def test_rebreak_loses_no_words():
    text = "one two three four five six seven eight nine ten"
    cues = [Cue(start=0.0, end=6.0, lines=[text])]
    joined = " ".join(c.flat for c in roundtrip.rebreak(cues, CueConfig()))
    for word in text.split():
        assert word in joined


def test_rebreak_handles_empty_input():
    assert roundtrip.rebreak([], CueConfig()) == []
