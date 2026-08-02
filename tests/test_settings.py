"""The user-editable settings file.

Two things matter here beyond "it parses". First, a missing or broken file must
never stop a transcription — the pipeline does not need any of this. Second,
writes must not eat the file: it is meant to be maintained by hand, so comments,
ordering and unrelated properties have to survive the UI saving a key into it.
"""

import pytest

from sgen import settings


@pytest.fixture
def path(tmp_path, monkeypatch):
    target = tmp_path / "settings.local.yaml"
    monkeypatch.setenv("SGEN_SETTINGS", str(target))
    for variable in settings.ENV_KEYS.values():
        monkeypatch.delenv(variable, raising=False)
    return target


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def test_missing_file_is_not_an_error(path):
    user = settings.load()
    assert not user.exists
    assert user.defaults.profile == "home-video"
    assert user.server.port == 8420
    assert user.api_keys.google == ""


def test_partial_file_keeps_defaults_for_everything_else(path):
    path.write_text("defaults:\n  profile: music\n", encoding="utf-8")
    user = settings.load()
    assert user.defaults.profile == "music"
    assert user.defaults.formats == ("srt",)
    assert user.server.host == "127.0.0.1"


def test_vtt_is_off_by_default(path):
    """WebVTT is for browser players; beside a video it is a duplicate file."""
    assert settings.Defaults().formats == ("srt",)
    assert "vtt" not in settings.load().defaults.formats


def test_nested_section_loads(path):
    path.write_text(
        "defaults:\n  translate:\n    provider: deepl\n    target: de\n",
        encoding="utf-8",
    )
    user = settings.load()
    assert user.defaults.translate.provider == "deepl"
    assert user.defaults.translate.target == "de"


def test_provided_distinguishes_a_setting_from_a_default(path):
    path.write_text("defaults:\n  romanize: true\n", encoding="utf-8")
    user = settings.load()
    assert user.given("defaults.romanize")
    assert not user.given("defaults.formats")


@pytest.mark.parametrize("body,needle", [
    ("defaults:\n  profil: music\n", "unknown option 'profil'"),
    ("nonsense:\n  x: 1\n", "unknown section"),
    ("defaults:\n  romanize: sometimes\n", "must be true or false"),
    ("server:\n  port: eight\n", "must be a whole number"),
    ("defaults:\n  formats: srt\n", "must be a list"),
    ("defaults: 3\n", "must be a mapping"),
    ("- a\n- b\n", "mapping of sections"),
    ("defaults:\n  profile: [\n", "not valid YAML"),
])
def test_mistakes_are_reported_not_ignored(path, body, needle):
    """A typo that is silently dropped looks exactly like a setting that fails."""
    path.write_text(body, encoding="utf-8")
    with pytest.raises(settings.SettingsError, match=needle):
        settings.load()


def test_a_broken_file_still_yields_working_defaults(path):
    path.write_text("defaults:\n  profil: music\n", encoding="utf-8")
    user = settings.load_or_default()
    assert user.error                      # reported...
    assert user.defaults.profile == "home-video"   # ...but usable


def test_environment_beats_the_file(path, monkeypatch):
    path.write_text('api_keys:\n  google: "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("SGEN_GOOGLE_API_KEY", "from-env")
    user = settings.load()
    assert user.api_keys.google == "from-env"
    assert user.key_source["google"] == "SGEN_GOOGLE_API_KEY"


def test_key_source_names_the_file_when_that_is_where_it_came_from(path):
    path.write_text('api_keys:\n  google: "abc"\n', encoding="utf-8")
    assert settings.load().key_source["google"] == str(path)


def test_keys_are_redacted_before_going_to_the_browser(path):
    path.write_text('api_keys:\n  google: "AIzaSECRETVALUE"\n', encoding="utf-8")
    shown = settings.load().to_dict()["api_keys"]["google"]
    assert "SECRET" not in shown
    assert shown.endswith("ALUE")


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

def test_init_creates_a_commented_file(path):
    created_path, created = settings.ensure_file()
    assert created and created_path.exists()
    text = created_path.read_text(encoding="utf-8")
    assert "api_keys:" in text and "#" in text
    settings.load()                        # the template must be valid
    assert settings.ensure_file()[1] is False   # second call is a no-op


def test_saving_a_key_preserves_comments_and_other_settings(path):
    path.write_text(
        "# my notes\n"
        "api_keys:\n"
        "  # where to get this\n"
        '  google: ""\n'
        "\n"
        "defaults:\n"
        "  profile: verbatim   # keep everything\n",
        encoding="utf-8",
    )
    settings.update_api_keys(google="AIza-123")
    text = path.read_text(encoding="utf-8")

    assert "# my notes" in text
    assert "# where to get this" in text
    assert "# keep everything" in text, "a note beside a value must survive"
    user = settings.load()
    assert user.api_keys.google == "AIza-123"
    assert user.defaults.profile == "verbatim"


def test_writing_adds_missing_sections_to_a_minimal_file(path):
    path.write_text("defaults:\n  profile: music\n", encoding="utf-8")
    settings.set_values({
        "api_keys.deepl": "abc:fx",
        "defaults.translate.target": "de",
        "server.port": 8500,
    })
    user = settings.load()
    assert user.api_keys.deepl == "abc:fx"
    assert user.defaults.translate.target == "de"
    assert user.server.port == 8500
    assert user.defaults.profile == "music"


def test_a_deepl_key_survives_the_yaml_round_trip(path):
    """Free keys end in ':fx', which unquoted YAML reads as a nested mapping."""
    settings.update_api_keys(deepl="9f8e7d6c-1234-abcd-5678-0f1e2d3c4b5a:fx")
    assert settings.load().api_keys.deepl.endswith(":fx")


def test_clearing_a_key_is_possible(path):
    settings.update_api_keys(google="AIza-123")
    settings.update_api_keys(google="")
    assert settings.load().api_keys.google == ""


def test_none_leaves_a_key_untouched(path):
    settings.update_api_keys(google="AIza-123", deepl="d-1")
    settings.update_api_keys(deepl="d-2")           # google not mentioned
    user = settings.load()
    assert user.api_keys.google == "AIza-123"
    assert user.api_keys.deepl == "d-2"


def test_a_bad_property_is_rejected_before_the_file_is_touched(path):
    path.write_text("defaults:\n  profile: music\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(settings.SettingsError):
        settings.set_values({"defaults.profile": "music", "defaults.nope": "x"})
    assert path.read_text(encoding="utf-8") == before


def test_repeated_writes_do_not_duplicate_lines(path):
    settings.update_api_keys(google="one")
    settings.update_api_keys(google="two")
    text = path.read_text(encoding="utf-8")
    assert text.count("google:") == 1
    assert settings.load().api_keys.google == "two"


# --------------------------------------------------------------------------- #
# command-line assignment
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("defaults.profile=music", ("defaults.profile", "music")),
    ("defaults.romanize=true", ("defaults.romanize", True)),
    ("server.port=8500", ("server.port", 8500)),
    ("api_keys.google=", ("api_keys.google", "")),
    ("api_keys.deepl=abc:fx", ("api_keys.deepl", "abc:fx")),
])
def test_parse_assignment(text, expected):
    assert settings.parse_assignment(text) == expected


@pytest.mark.parametrize("text", [
    "defaults.profile",            # no '='
    "defaults.nope=1",             # unknown property
    "defaults=music",              # a section, not a property
    "server.port=abc",             # wrong type
    "defaults.profile.deep=x",     # not a section
])
def test_bad_assignments_are_refused(text):
    with pytest.raises(settings.SettingsError):
        settings.parse_assignment(text)


# --------------------------------------------------------------------------- #
# effect on a run
# --------------------------------------------------------------------------- #

def test_settings_overlay_a_profile(path):
    from sgen.config import Config

    path.write_text(
        "defaults:\n"
        "  language: de\n"
        "  hotwords: Kreuzberg\n"
        "  romanize: true\n"
        "  formats: [vtt]\n",
        encoding="utf-8",
    )
    cfg = settings.apply_defaults(Config.load("home-video"))
    assert cfg.asr.language == "de"
    assert cfg.asr.hotwords == "Kreuzberg"
    assert cfg.romanize is True
    assert cfg.formats == ("vtt",)


def test_unmentioned_properties_leave_the_profile_alone(path):
    from sgen.config import Config

    path.write_text("defaults:\n  profile: home-video\n", encoding="utf-8")
    cfg = settings.apply_defaults(Config.load("home-video"))
    assert cfg.asr.language is None          # still detect
    assert cfg.formats == ("srt", "vtt")
    assert cfg.romanize is False


def test_empty_language_setting_means_detect(path):
    from sgen.config import Config

    path.write_text('defaults:\n  language: ""\n', encoding="utf-8")
    cfg = settings.apply_defaults(Config.load("home-video"))
    assert cfg.asr.language is None


def test_ui_options_win_over_settings(path):
    from sgen.server.jobs import build_config

    path.write_text("defaults:\n  profile: music\n  language: de\n", encoding="utf-8")
    cfg = build_config({"profile": "verbatim", "language": "es"})
    assert cfg.profile == "verbatim"
    assert cfg.asr.language == "es"


def test_settings_fill_in_what_the_request_omits(path):
    from sgen.server.jobs import build_config

    path.write_text(
        "defaults:\n  profile: verbatim\n  language: de\n  hotwords: Oaxaca\n",
        encoding="utf-8",
    )
    cfg = build_config({})
    assert cfg.profile == "verbatim"
    assert cfg.asr.language == "de"
    assert cfg.asr.hotwords == "Oaxaca"


def test_the_example_file_is_committed_and_valid():
    """The template is the documentation; it must stay loadable."""
    assert settings.TEMPLATE_PATH.exists()
    user = settings.load(settings.TEMPLATE_PATH)
    assert user.defaults.profile == "home-video"
    assert user.api_keys.google == ""
