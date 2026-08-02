"""Per-file settings, stored beside the videos.

A folder is rarely uniform: songs want the `music` profile, Hindi footage wants
Latin-script subtitles, and a clip already in English wants no translation. These
overrides live in `sgen.folder.yaml` in the folder itself — the same reasoning as
resumability, that state kept in the app cannot survive a restart and state kept
in a database can disagree with the disk.
"""

from pathlib import Path

import pytest

from sgen import folderconf


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "song.mp4").write_bytes(b"x")
    (tmp_path / "beach.mp4").write_bytes(b"x")
    sub = tmp_path / "trip"
    sub.mkdir()
    (sub / "song.mp4").write_bytes(b"x")      # same name, different file
    return tmp_path


def write(folder: Path, body: str) -> Path:
    path = folderconf.config_path(folder)
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def test_no_config_means_no_overrides(folder):
    assert folderconf.load(folder) == {}
    assert folderconf.for_file(folder, folder / "song.mp4") == {}


def test_overrides_are_read_per_file(folder):
    write(folder, """
files:
  "song.mp4":
    profile: music
    romanize: true
  "beach.mp4":
    translate: none
""")
    assert folderconf.for_file(folder, folder / "song.mp4") == {
        "profile": "music", "romanize": True}
    assert folderconf.for_file(folder, folder / "beach.mp4") == {"translate": "none"}


def test_a_subfolder_file_is_addressed_by_its_relative_path(folder):
    """Two files can share a name; the override must not leak between them."""
    write(folder, """
files:
  "trip/song.mp4":
    profile: verbatim
""")
    assert folderconf.for_file(folder, folder / "trip" / "song.mp4") == {
        "profile": "verbatim"}
    assert folderconf.for_file(folder, folder / "song.mp4") == {}


def test_backslashes_in_the_file_are_accepted(folder):
    """People will write Windows paths in a Windows folder."""
    write(folder, 'files:\n  "trip\\\\song.mp4":\n    profile: music\n')
    assert folderconf.for_file(folder, folder / "trip" / "song.mp4") == {
        "profile": "music"}


def test_unknown_settings_are_ignored_not_fatal(folder):
    """A batch of fifty videos must not fail because of a typo in an optional file."""
    write(folder, """
files:
  "song.mp4":
    profile: music
    modle: large-v3
""")
    assert folderconf.for_file(folder, folder / "song.mp4") == {"profile": "music"}


def test_broken_yaml_is_survivable(folder):
    write(folder, "files: [not a mapping")
    assert folderconf.load(folder) == {}


def test_problems_are_reportable(folder):
    write(folder, """
files:
  "song.mp4":
    modle: large-v3
    translate: maybe
""")
    problems = folderconf.problems(folder)
    assert any("unknown setting" in p for p in problems)
    assert any("translate must be one of" in p for p in problems)


def test_a_valid_file_has_no_problems(folder):
    write(folder, 'files:\n  "song.mp4":\n    profile: music\n')
    assert folderconf.problems(folder) == []


# --------------------------------------------------------------------------- #
# turning overrides into run options
# --------------------------------------------------------------------------- #

def test_nothing_overridden_leaves_the_options_alone():
    base = {"profile": "home-video", "romanize": False, "cloud_provider": "deepl"}
    assert folderconf.apply_to_options(base, {}) == base


def test_profile_and_romanize_are_applied():
    out = folderconf.apply_to_options(
        {"profile": "home-video", "romanize": False},
        {"profile": "music", "romanize": True})
    assert out["profile"] == "music" and out["romanize"] is True


def test_translate_none_turns_both_paths_off():
    """The one file you do not want translated, in a folder where everything else
    is."""
    out = folderconf.apply_to_options(
        {"cloud_provider": "deepl", "translate": True}, {"translate": "none"})
    assert out["cloud_provider"] == "" and out["translate"] is False


def test_translate_can_name_a_provider_per_file():
    out = folderconf.apply_to_options({"cloud_provider": ""}, {"translate": "google"})
    assert out["cloud_provider"] == "google" and out["translate"] is False


def test_translate_local_selects_the_offline_model():
    out = folderconf.apply_to_options({"cloud_provider": "deepl"}, {"translate": "local"})
    assert out["cloud_provider"] == "" and out["translate"] is True


def test_an_empty_translate_defers_to_the_panel():
    base = {"cloud_provider": "deepl", "translate": False}
    assert folderconf.apply_to_options(base, {"translate": ""}) == base


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

def test_setting_an_override_creates_a_readable_file(folder):
    folderconf.set_for_file(folder, folder / "song.mp4",
                            {"profile": "music", "romanize": True})
    text = folderconf.config_path(folder).read_text(encoding="utf-8")
    assert "song.mp4" in text and "music" in text
    assert text.startswith("#"), "the file explains itself to whoever opens it"
    assert folderconf.for_file(folder, folder / "song.mp4")["profile"] == "music"


def test_overrides_for_other_files_survive_an_edit(folder):
    folderconf.set_for_file(folder, folder / "song.mp4", {"profile": "music"})
    folderconf.set_for_file(folder, folder / "beach.mp4", {"translate": "none"})
    assert folderconf.for_file(folder, folder / "song.mp4") == {"profile": "music"}
    assert folderconf.for_file(folder, folder / "beach.mp4") == {"translate": "none"}


def test_clearing_removes_only_that_file(folder):
    folderconf.set_for_file(folder, folder / "song.mp4", {"profile": "music"})
    folderconf.set_for_file(folder, folder / "beach.mp4", {"translate": "none"})
    folderconf.set_for_file(folder, folder / "song.mp4", {})
    assert folderconf.for_file(folder, folder / "song.mp4") == {}
    assert folderconf.for_file(folder, folder / "beach.mp4") == {"translate": "none"}


def test_the_file_is_removed_when_nothing_is_overridden(folder):
    folderconf.set_for_file(folder, folder / "song.mp4", {"profile": "music"})
    folderconf.set_for_file(folder, folder / "song.mp4", {})
    assert not folderconf.config_path(folder).exists(), \
        "an empty config file says nothing and is just clutter"


def test_empty_values_are_not_stored(folder):
    """"as in settings" is the absence of an override, not an override to ""."""
    folderconf.set_for_file(folder, folder / "song.mp4",
                            {"profile": "", "translate": "none"})
    assert folderconf.for_file(folder, folder / "song.mp4") == {"translate": "none"}


def test_an_invalid_setting_is_refused_before_writing(folder):
    with pytest.raises(folderconf.FolderConfigError, match="not a per-file setting"):
        folderconf.set_for_file(folder, folder / "song.mp4", {"model": "large-v3"})
    with pytest.raises(folderconf.FolderConfigError, match="translate choice"):
        folderconf.set_for_file(folder, folder / "song.mp4", {"translate": "bing"})
    assert not folderconf.config_path(folder).exists()


def test_a_round_trip_survives_reloading(folder):
    """The point of keeping this on disk: a new process reads the same thing."""
    folderconf.set_for_file(folder, folder / "trip" / "song.mp4",
                            {"profile": "music", "translate": "none"})
    reloaded = folderconf.load(folder)
    assert reloaded["trip/song.mp4"] == {"profile": "music", "translate": "none"}


# --------------------------------------------------------------------------- #
# effect on what counts as finished
# --------------------------------------------------------------------------- #

def test_a_file_with_translation_off_is_finished_without_a_translation(folder):
    """The reason the scan has to be per-file: with translation on for the folder,
    a Hindi song that opted out is done once its Hindi subtitles exist."""
    from sgen import resume
    from sgen.config import Config
    from sgen.cues import Cue
    from sgen.write import write_subtitles

    cfg = Config()
    cfg.formats = ("srt",)
    write_subtitles([Cue(start=0.0, end=1.0, lines=["नमस्ते"])],
                    folder / "song", ("srt",), "hi")

    # Folder-wide: English translation wanted -> unfinished.
    assert resume.classify(folder / "song.mp4", cfg,
                           translate_target="en").state == "translate"

    # This file opts out -> finished.
    folderconf.set_for_file(folder, folder / "song.mp4", {"translate": "none"})
    options = folderconf.apply_to_options(
        {"cloud_provider": "deepl"}, folderconf.for_file(folder, folder / "song.mp4"))
    target = None if not options.get("cloud_provider") else "en"
    assert resume.classify(folder / "song.mp4", cfg,
                           translate_target=target).state == "done"


def test_the_config_file_is_not_mistaken_for_media(folder):
    from sgen import resume

    folderconf.set_for_file(folder, folder / "song.mp4", {"profile": "music"})
    names = [p.name for p in resume.media_files(folder)]
    assert folderconf.CONFIG_NAME not in names
