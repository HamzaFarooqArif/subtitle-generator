"""Subtitle file writing, with attention to non-ASCII correctness."""

from __future__ import annotations

from sgen.cues import Cue
from sgen.write import write_subtitles

CUES = [
    Cue(start=1.0, end=3.0, lines=["Der Wetterbericht sagt,", "dass es später regnen wird."]),
    Cue(start=4.0, end=6.0, lines=["Nos vemos mañana por la mañana."]),
]


def test_writes_requested_formats(tmp_path):
    paths = write_subtitles(CUES, tmp_path / "clip", ["srt", "vtt"], "de")
    assert [p.name for p in paths] == ["clip.de.srt", "clip.de.vtt"]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_utf8_bom_written_by_default(tmp_path):
    """Windows tools guess cp1252 without a BOM and mangle umlauts."""
    path = write_subtitles(CUES, tmp_path / "clip", ["srt"], "de")[0]
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_bom_can_be_disabled(tmp_path):
    path = write_subtitles(CUES, tmp_path / "clip", ["srt"], "de", "utf-8")[0]
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_non_ascii_round_trips_exactly(tmp_path):
    path = write_subtitles(CUES, tmp_path / "clip", ["srt"], "de")[0]
    body = path.read_text(encoding="utf-8-sig")
    assert "später" in body
    assert "mañana" in body
    assert "Ã" not in body  # the mojibake signature


def test_line_breaks_preserved_in_srt(tmp_path):
    path = write_subtitles(CUES, tmp_path / "clip", ["srt"], "de")[0]
    body = path.read_text(encoding="utf-8-sig")
    assert "Der Wetterbericht sagt,\ndass es später regnen wird." in body.replace("\r\n", "\n")


def test_timecodes_match_cue_times(tmp_path):
    path = write_subtitles(CUES, tmp_path / "clip", ["srt"], "de")[0]
    body = path.read_text(encoding="utf-8-sig")
    assert "00:00:01,000 --> 00:00:03,000" in body
    assert "00:00:04,000 --> 00:00:06,000" in body


def test_empty_cue_list_still_writes_a_file(tmp_path):
    paths = write_subtitles([], tmp_path / "clip", ["srt"], "en")
    assert paths[0].exists()
