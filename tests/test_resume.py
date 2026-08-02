"""Resuming a folder batch with no stored state.

The rule under test: whether a video is finished is decided from the subtitle
files next to it, so closing the app, killing the process or restarting the
machine costs nothing. The dangerous case is the opposite of skipping too little
— skipping a file whose subtitles were left half-written, which would mean that
video is never done and nothing says so.
"""

from pathlib import Path

import pytest

from sgen import resume
from sgen.config import Config
from sgen.cues import Cue
from sgen.write import write_subtitles


@pytest.fixture
def cfg():
    config = Config()
    config.formats = ("srt",)
    return config


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.mkv").write_bytes(b"x")
    (tmp_path / "c.wav").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not media", encoding="utf-8")
    return tmp_path


def subtitle(path: Path, language: str, cues: int = 2, fmt: str = "srt") -> Path:
    made = [
        Cue(start=float(i) * 2, end=float(i) * 2 + 1.5, lines=[f"line {i}"])
        for i in range(cues)
    ]
    return write_subtitles(made, path.with_suffix(""), (fmt,), language)[0]


# --------------------------------------------------------------------------- #
# finding the work
# --------------------------------------------------------------------------- #

def test_media_files_ignores_everything_else(folder):
    names = [p.name for p in resume.media_files(folder)]
    assert names == ["a.mp4", "b.mkv", "c.wav"]


def test_subfolders_are_included_and_can_be_excluded(folder):
    nested = folder / "trip" / "day2"
    nested.mkdir(parents=True)
    (nested / "d.mp4").write_bytes(b"x")
    assert len(resume.media_files(folder, recursive=True)) == 4
    assert len(resume.media_files(folder, recursive=False)) == 3


def test_order_is_stable(folder):
    """A batch that is interrupted and resumed should proceed predictably."""
    assert resume.media_files(folder) == resume.media_files(folder)


def test_a_missing_folder_is_an_error(tmp_path):
    with pytest.raises(NotADirectoryError):
        resume.media_files(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# what counts as finished
# --------------------------------------------------------------------------- #

def test_no_subtitles_means_pending(folder, cfg):
    status = resume.classify(folder / "a.mp4", cfg)
    assert status.state == "pending"
    assert status.needs_work


def test_subtitles_present_means_done(folder, cfg):
    subtitle(folder / "a.mp4", "en")
    status = resume.classify(folder / "a.mp4", cfg)
    assert status.state == "done"
    assert not status.needs_work
    assert status.languages == ["en"]


def test_the_language_in_the_filename_is_read(folder, cfg):
    subtitle(folder / "a.mp4", "ru")
    assert resume.classify(folder / "a.mp4", cfg).languages == ["ru"]


def test_a_missing_format_is_not_finished(folder):
    """Asking for srt and vtt but finding only srt means the run was cut short."""
    config = Config()
    config.formats = ("srt", "vtt")
    subtitle(folder / "a.mp4", "en", fmt="srt")
    status = resume.classify(folder / "a.mp4", config)
    assert status.state == "pending"
    assert ".vtt" in status.reason


def test_somebody_elses_subtitle_file_is_not_claimed(folder, cfg):
    """A hand-made clip.srt with no language tag is not ours; treating it as
    finished would silently skip the video."""
    (folder / "a.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    assert resume.classify(folder / "a.mp4", cfg).state == "pending"


def test_a_romanized_companion_is_not_evidence_of_a_transcript(folder, cfg):
    subtitle(folder / "a.mp4", "hi-Latn")
    status = resume.classify(folder / "a.mp4", cfg)
    assert status.state == "pending", "hi-Latn alone means the hi file is missing"


def test_brackets_in_filenames_do_not_break_matching(tmp_path, cfg):
    """glob would read [1080p] as a character class and match nothing."""
    source = tmp_path / "holiday [1080p] (2024).mp4"
    source.write_bytes(b"x")
    subtitle(source, "de")
    assert resume.classify(source, cfg).state == "done"


def test_out_dir_is_searched_when_output_goes_elsewhere(folder, cfg, tmp_path):
    out = tmp_path / "subs"
    out.mkdir()
    subtitle(out / "a.mp4", "en")
    assert resume.classify(folder / "a.mp4", cfg).state == "pending"
    assert resume.classify(folder / "a.mp4", cfg, out_dir=out).state == "done"


# --------------------------------------------------------------------------- #
# interrupted work — the case this exists for
# --------------------------------------------------------------------------- #

def test_a_truncated_subtitle_is_redone_not_trusted(folder, cfg):
    """The failure that would make a video never finish: a half-written file
    that looks like a real one. Atomic writes prevent it going forward; this is
    the guard for anything already on disk."""
    path = subtitle(folder / "a.mp4", "en")
    text = path.read_text(encoding="utf-8-sig")
    # Cut inside the final timestamp line, so no text follows it — which is what
    # a write interrupted mid-file leaves behind.
    cut = text[: text.rindex("-->") + 5]
    path.write_text(cut, encoding="utf-8-sig")
    assert "line 1" not in cut, "the last cue's text must be missing"

    status = resume.classify(folder / "a.mp4", cfg)
    assert status.state == "damaged"
    assert status.damaged == [path]
    assert "interrupted" in status.reason


def test_an_empty_subtitle_file_is_damaged(folder, cfg):
    (folder / "a.en.srt").write_text("", encoding="utf-8")
    assert resume.classify(folder / "a.mp4", cfg).state == "damaged"


def test_a_file_with_no_cues_is_damaged(folder, cfg):
    (folder / "a.en.srt").write_text("﻿\n", encoding="utf-8")
    assert resume.classify(folder / "a.mp4", cfg).state == "damaged"


def test_writes_are_atomic(folder, cfg, monkeypatch):
    """If saving dies partway, no subtitle file may appear at all — a partial one
    would be indistinguishable from a finished one on the next run."""
    import pysubs2

    def explode(self, path, **kwargs):
        Path(path).write_text("1\n00:00:00,000 -->", encoding="utf-8")
        raise OSError("power cut")

    monkeypatch.setattr(pysubs2.SSAFile, "save", explode)
    with pytest.raises(OSError):
        subtitle(folder / "a.mp4", "en")

    assert not (folder / "a.en.srt").exists(), "no half-written file left behind"
    assert not list(folder.glob("*.part")), "temporary file cleaned up"
    assert resume.classify(folder / "a.mp4", cfg).state == "pending"


# --------------------------------------------------------------------------- #
# translation as part of "finished"
# --------------------------------------------------------------------------- #

def test_a_foreign_file_without_its_translation_is_not_finished(folder, cfg):
    subtitle(folder / "a.mp4", "ru")
    status = resume.classify(folder / "a.mp4", cfg, translate_target="en")
    assert status.state == "translate"
    assert "no en translation" in status.reason


def test_both_files_present_means_finished(folder, cfg):
    subtitle(folder / "a.mp4", "ru")
    subtitle(folder / "a.mp4", "en")
    status = resume.classify(folder / "a.mp4", cfg, translate_target="en")
    assert status.state == "done"


def test_an_english_file_needs_no_translation(folder, cfg):
    """Most home video is already in the target language; asking for a
    translation must not make those files eternally unfinished."""
    subtitle(folder / "a.mp4", "en")
    assert resume.classify(folder / "a.mp4", cfg, translate_target="en").state == "done"


def test_translation_is_only_expected_when_it_was_asked_for(folder, cfg):
    subtitle(folder / "a.mp4", "ru")
    assert resume.classify(folder / "a.mp4", cfg).state == "done"
    assert resume.classify(folder / "a.mp4", cfg, translate_target="en").state == "translate"


def test_a_lost_transcript_is_reported(folder, cfg, tmp_path):
    """Whether the sidecar survives decides API call vs full GPU pass; it never
    decides whether the file is done."""
    subtitle(folder / "a.mp4", "ru")
    status = resume.classify(folder / "a.mp4", cfg, translate_target="en",
                             work_dir=tmp_path / "empty-work")
    assert status.state == "translate"
    assert status.has_transcript is False
    assert "needs a full pass" in status.reason


# --------------------------------------------------------------------------- #
# scanning a whole folder
# --------------------------------------------------------------------------- #

def test_scan_counts_a_mixed_folder(folder, cfg):
    subtitle(folder / "a.mp4", "en")                       # done
    subtitle(folder / "b.mkv", "ru")                       # needs translation
    (folder / "c.en.srt").write_text("", encoding="utf-8")  # damaged

    scan = resume.scan_folder(folder, cfg, translate_target="en")
    assert scan.counts == {"done": 1, "pending": 0, "translate": 1, "damaged": 1}
    assert {f.source.name for f in scan.todo} == {"b.mkv", "c.wav"}


def test_scan_summary_reads_like_a_sentence(folder, cfg):
    subtitle(folder / "a.mp4", "en")
    summary = resume.summarise(resume.scan_folder(folder, cfg))
    assert "3 media file(s)" in summary
    assert "1 already done" in summary
    assert "2 to transcribe" in summary


def test_rerunning_after_everything_is_done_finds_nothing(folder, cfg):
    for name in ("a.mp4", "b.mkv", "c.wav"):
        subtitle(folder / name, "en")
    scan = resume.scan_folder(folder, cfg)
    assert scan.todo == []
    assert scan.counts["done"] == 3


def test_the_batch_survives_a_restart(folder, cfg):
    """The whole point: a fresh process with no memory of the last one reaches
    the same conclusion, because the conclusion is on disk."""
    files = resume.media_files(folder)
    # First "run": two of three finish.
    for source in files[:2]:
        subtitle(source, "en")

    # Second "run" — nothing carried over but the folder itself.
    scan = resume.scan_folder(folder, cfg)
    assert [f.source.name for f in scan.todo] == ["c.wav"]
