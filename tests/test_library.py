"""Forgetting a file.

The list of past transcripts is a record of private footage: the sidecar holds
the full text of what was said and the path it came from, the WAV holds the audio
itself. So removing an entry has to be reachable, precise about what it deletes,
and impossible to aim at anything outside `work/`.
"""

import json
from pathlib import Path

import pytest

from sgen import library


def make_entry(
    work: Path, content_id: str, name: str = "clip.mp4", *, source: Path | None = None,
    cues: int = 2, audio: bool = True,
) -> Path:
    folder = work / content_id
    folder.mkdir(parents=True, exist_ok=True)
    path = source if source is not None else work.parent / name
    (folder / library.SIDECAR_NAME).write_text(json.dumps({
        "source": {"path": str(path), "name": name, "content_id": content_id},
        "language": "hi",
        "audio_duration": 238.5,
        "cues": [{"start": i, "end": i + 1, "lines": ["secret"]} for i in range(cues)],
    }), encoding="utf-8")
    if audio:
        (folder / library.AUDIO_NAME).write_bytes(b"\0" * 2048)
    return folder


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def test_no_work_folder_means_an_empty_list(work):
    assert library.entries(work) == []


def test_an_entry_reports_what_the_ui_shows(work, tmp_path):
    source = tmp_path / "beach.mp4"
    source.write_bytes(b"x")
    make_entry(work, "aaaa1111", "beach.mp4", source=source)

    (entry,) = library.entries(work)
    assert entry.content_id == "aaaa1111"
    assert entry.name == "beach.mp4"
    assert entry.cue_count == 2
    assert entry.source_exists is True
    assert entry.size > 2000, "the cached audio is the bulk of it, and worth showing"


def test_a_moved_source_is_still_listed(work):
    """The transcript is the valuable part; it outlives the file it came from."""
    make_entry(work, "aaaa1111", source=Path("Z:/gone/clip.mp4"))
    (entry,) = library.entries(work)
    assert entry.source_exists is False


def test_a_transcript_with_no_cues_is_not_an_entry(work):
    folder = work / "bbbb2222"
    folder.mkdir(parents=True)
    (folder / library.SIDECAR_NAME).write_text('{"cues": []}', encoding="utf-8")
    assert library.entries(work) == []


def test_a_broken_sidecar_does_not_empty_the_list(work):
    make_entry(work, "aaaa1111")
    bad = work / "bbbb2222"
    bad.mkdir()
    (bad / library.SIDECAR_NAME).write_text("{ truncated", encoding="utf-8")
    assert [e.content_id for e in library.entries(work)] == ["aaaa1111"]


def test_newest_first(work):
    make_entry(work, "old11111")
    make_entry(work, "new22222")
    sidecar = work / "new22222" / library.SIDECAR_NAME
    import os
    os.utime(sidecar, (2_000_000_000, 2_000_000_000))
    assert [e.content_id for e in library.entries(work)][0] == "new22222"


# --------------------------------------------------------------------------- #
# refusing to delete the wrong thing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    "..", "../..", "a/b", "a\\b", "/etc", "C:\\Windows", "", "x" * 65, "a;b",
])
def test_only_a_plain_id_can_name_a_folder(work, bad):
    """The id arrives from a URL and ends up in a recursive delete."""
    work.mkdir(parents=True)
    with pytest.raises(library.LibraryError):
        library.entry_dir(work, bad)


def test_a_symlink_out_of_the_work_folder_is_refused(work, tmp_path):
    work.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        (work / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(library.LibraryError, match="not inside"):
        library.entry_dir(work, "escape")
    assert outside.exists()


def test_forgetting_something_that_is_not_there_says_so(work):
    work.mkdir(parents=True)
    with pytest.raises(library.LibraryError, match="nothing here"):
        library.forget(work, "aaaa1111")


def test_a_file_being_transcribed_is_not_forgotten_under_it(work, tmp_path):
    """Deleting the work folder mid-run fails the job somewhere deep and
    unhelpful. Refuse it up front instead."""
    source = tmp_path / "running.mp4"
    source.write_bytes(b"x")
    make_entry(work, "aaaa1111", "running.mp4", source=source)

    with pytest.raises(library.LibraryError, match="being transcribed"):
        library.forget(work, "aaaa1111", protected=[str(source)])
    assert (work / "aaaa1111").exists()


def test_the_busy_check_compares_paths_not_strings(work, tmp_path):
    source = tmp_path / "running.mp4"
    source.write_bytes(b"x")
    make_entry(work, "aaaa1111", source=source)
    spelled_differently = str(tmp_path / "." / "running.mp4")

    with pytest.raises(library.LibraryError):
        library.forget(work, "aaaa1111", protected=[spelled_differently])


# --------------------------------------------------------------------------- #
# forgetting
# --------------------------------------------------------------------------- #

def test_forgetting_removes_the_transcript_and_the_audio(work):
    folder = make_entry(work, "aaaa1111", "beach.mp4")
    result = library.forget(work, "aaaa1111")

    assert not folder.exists()
    assert result["name"] == "beach.mp4", "the toast should name the file, not a hash"
    assert result["freed"] > 2000
    assert library.entries(work) == []


def test_forgetting_leaves_the_subtitle_files_alone(work, tmp_path):
    """What the user came here for stays. What the app kept for itself goes."""
    source = tmp_path / "beach.mp4"
    source.write_bytes(b"x")
    subtitle = tmp_path / "beach.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    make_entry(work, "aaaa1111", "beach.mp4", source=source)

    library.forget(work, "aaaa1111")
    assert subtitle.exists() and source.exists()


def test_forgetting_one_leaves_the_others(work):
    make_entry(work, "aaaa1111")
    make_entry(work, "bbbb2222")
    library.forget(work, "aaaa1111")
    assert [e.content_id for e in library.entries(work)] == ["bbbb2222"]


def test_forget_all_clears_the_cache(work):
    make_entry(work, "aaaa1111")
    make_entry(work, "bbbb2222")
    result = library.forget_all(work)
    assert result["removed"] == 2 and result["freed"] > 4000
    assert result["kept"] == []
    assert library.entries(work) == []


def test_forget_all_includes_a_run_that_never_finished(work):
    """An interrupted transcription leaves the extracted audio with no sidecar:
    invisible in the list, still a recording of someone."""
    orphan = work / "cccc3333"
    orphan.mkdir(parents=True)
    (orphan / library.AUDIO_NAME).write_bytes(b"\0" * 1024)

    assert library.entries(work) == [], "not listed anywhere"
    assert library.forget_all(work)["removed"] == 1
    assert not orphan.exists()


def test_forget_all_keeps_a_file_that_is_being_transcribed(work, tmp_path):
    source = tmp_path / "running.mp4"
    source.write_bytes(b"x")
    make_entry(work, "aaaa1111", "running.mp4", source=source)
    make_entry(work, "bbbb2222", "other.mp4")

    result = library.forget_all(work, protected=[str(source)])
    assert result["removed"] == 1
    assert result["kept"] == ["running.mp4"]
    assert (work / "aaaa1111").exists() and not (work / "bbbb2222").exists()


def test_forget_all_will_not_delete_unrelated_folders(work):
    """work/ lives inside the project. "Forget everything" is about this app's
    cache, not about whatever else is in the directory."""
    make_entry(work, "aaaa1111")
    mine = work / "my-notes"
    mine.mkdir()
    (mine / "todo.txt").write_text("keep me", encoding="utf-8")

    assert library.forget_all(work)["removed"] == 1
    assert (mine / "todo.txt").read_text(encoding="utf-8") == "keep me"


def test_forget_all_on_an_empty_cache_is_harmless(work):
    assert library.forget_all(work) == {"removed": 0, "freed": 0, "kept": []}


def test_describing_an_id_before_deleting_it(work):
    make_entry(work, "aaaa1111", "beach.mp4")
    assert library.describe(work, "aaaa1111").name == "beach.mp4"
    assert library.describe(work, "bbbb2222") is None
