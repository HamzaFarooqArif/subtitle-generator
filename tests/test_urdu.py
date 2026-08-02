"""Hindi written in Urdu script.

Hindi and Urdu are one spoken language with two alphabets. The bar here is the
same as for the Latin path — what a reader would actually recognise — but the
failure mode is different: Devanagari lost the Perso-Arabic letter distinctions,
so a word list carries the vocabulary a table cannot reach.
"""

import pytest

from sgen import urdu


# Real lines from the Tera Ban Jaunga and Khairiyat transcripts.
LINES = [
    ("मेरा हक है इश्क मेरा तू पे शक है", "میرا حق ہے عشق میرا تو پے شک ہے"),
    ("साथ छोड़ूंगा ना तेरे", "ساتھ چھوڑوں گا نا تیرے"),
    ("या खुदा से मांग लाओंगा", "یا خدا سے مانگ لاؤں گا"),
    ("मैं तेरा बन जाऊँगा", "میں تیرا بن جاؤں گا"),
    ("खैरियत पूछो कभी तो कैफियत पूछो", "خیریت پوچھو کبھی تو کیفیت پوچھو"),
]


@pytest.mark.parametrize("hindi,expected", LINES)
def test_known_lines(hindi, expected):
    assert urdu.convert(hindi) == expected


def test_the_word_list_beats_the_table():
    """The point of the list: these are spelled by etymology, not by sound.

    A table gives ہک for हक, which is phonetically right and looks wrong to any
    reader.
    """
    assert urdu.convert("हक") == "حق"
    assert urdu.convert("कसम") == "قسم"
    assert urdu.convert("खातिर") == "خاطر"
    assert urdu.convert("मंजिल") == "منزل"


def test_a_word_outside_the_list_still_comes_out_readable():
    """The known limit, stated as a test rather than left as a surprise: no
    crash, no gap, just naive spelling."""
    out = urdu.convert("तकल्लुफ")       # not in the list
    assert out and not any("ऀ" <= c <= "ॿ" for c in out)


def test_nukta_and_cluster_spellings_find_the_same_word():
    """Whisper rarely writes nuktas and varies on nasals, so मंज़िल, मंजिल and
    मन्जिल all have to reach the same entry."""
    assert urdu.convert("मंज़िल") == urdu.convert("मंजिल") == urdu.convert("मन्जिल")


def test_the_future_tense_becomes_two_words():
    """Urdu writes जाऊँगा as جاؤں گا — a nasal and a space that Devanagari joins.
    Nearly every line of a song is a future verb, so this rule carries the file.
    """
    assert urdu.convert("जाऊँगा") == "جاؤں گا"
    assert urdu.convert("छोड़ूंगा") == "چھوڑوں گا"
    assert urdu.convert("बनाऊँगा") == "بناؤں گا"
    assert urdu.convert("करेंगे") == "کریں گے"


def test_short_vowels_are_not_written_and_long_ones_are():
    """The core of Urdu spelling: दिल is دل, नाम is نام."""
    assert urdu.convert("दिल") == "دل"
    assert urdu.convert("नाम") == "نام"
    assert urdu.convert("राम") == "رام"


def test_a_doubled_consonant_is_one_letter():
    assert urdu.convert("लक्खा") == "لکھا"


def test_aspirates_and_retroflexes():
    assert urdu.convert("साथ") == "ساتھ"       # थ is تھ
    assert urdu.convert("पूछो") == "پوچھو"     # छ is چھ
    assert urdu.convert("बड़ा") == "بڑا"        # ड़ is ڑ


def test_nasal_at_the_end_and_in_the_middle():
    assert urdu.convert("में") == "میں"        # ں at the end
    assert urdu.convert("मांग") == "مانگ"      # ن inside


def test_english_words_and_numbers_pass_through():
    assert urdu.convert("Hello दोस्त 2019") == "Hello دوست 2019"


def test_only_hindi_is_offered():
    """Marathi in Urdu script would be a curiosity; Punjabi's Shahmukhi is a
    different script block and not implemented."""
    assert urdu.supported("hi")
    assert not urdu.supported("mr")
    assert not urdu.supported("ru")
    assert not urdu.supported(None)


def test_empty_input_is_safe():
    assert urdu.convert("") == ""
    assert urdu.convert("   ") == "   "
    assert urdu.convert_lines([]) == []


# --------------------------------------------------------------------------- #
# writing files
# --------------------------------------------------------------------------- #

def test_written_as_hi_arab(tmp_path):
    """BCP-47: hi-Arab is Hindi in Arabic script, matching hi-Latn."""
    from sgen.cues import Cue
    from sgen.write import write_urdu

    cues = [Cue(start=1.0, end=3.0, lines=["मेरा हक है"])]
    paths = write_urdu(cues, tmp_path / "song", ["srt"], "hi")
    assert [p.name for p in paths] == ["song.hi-Arab.srt"]
    body = paths[0].read_text(encoding="utf-8-sig")
    assert "میرا حق ہے" in body


def test_nothing_written_for_a_language_it_cannot_do(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_urdu

    cues = [Cue(start=1.0, end=3.0, lines=["Тихо"])]
    assert write_urdu(cues, tmp_path / "clip", ["srt"], "ru") == []


def test_both_scripts_at_once(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_second_script

    cues = [Cue(start=1.0, end=3.0, lines=["मेरा नाम राम है"])]
    paths, notes = write_second_script(
        cues, tmp_path / "song", ["srt"], "hi", script="both"
    )
    assert sorted(p.name for p in paths) == ["song.hi-Arab.srt", "song.hi-Latn.srt"]
    assert notes == []


def test_asking_for_urdu_on_a_russian_file_says_why(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_second_script

    cues = [Cue(start=1.0, end=3.0, lines=["Тихо, может звонит."])]
    paths, notes = write_second_script(
        cues, tmp_path / "clip", ["srt"], "ru", script="urdu"
    )
    assert paths == []
    assert notes and "Urdu-script" in notes[0] and "Hindi" in notes[0]


def test_both_reports_only_the_half_that_failed(tmp_path):
    """Russian has a Latin romanizer and no Urdu one, so "both" should produce
    one file and one explanation — not silence, and not nothing."""
    from sgen.cues import Cue
    from sgen.write import write_second_script

    cues = [Cue(start=1.0, end=3.0, lines=["Тихо"])]
    paths, notes = write_second_script(
        cues, tmp_path / "clip", ["srt"], "ru", script="both"
    )
    assert [p.name for p in paths] == ["clip.ru-Latn.srt"]
    assert len(notes) == 1 and "Urdu-script" in notes[0]


# --------------------------------------------------------------------------- #
# reaching it from a run
# --------------------------------------------------------------------------- #

def test_the_choice_reaches_a_run_from_the_ui(monkeypatch, tmp_path):
    """The path the app actually uses: UI options -> Config -> pipeline."""
    from sgen.config import Config

    assert Config().romanize_script == "latin", "the safe default"

    monkeypatch.setenv("SGEN_SETTINGS", str(tmp_path / "none.yaml"))
    from sgen.server.jobs import build_config

    cfg = build_config({"romanize": True, "romanize_script": "urdu"})
    assert cfg.romanize is True and cfg.romanize_script == "urdu"
    # And a value nobody implements is ignored rather than passed down.
    assert build_config({"romanize_script": "klingon"}).romanize_script == "latin"


def test_a_bad_choice_in_the_settings_file_is_refused(tmp_path):
    """A typo that is silently ignored looks exactly like a setting that does
    not work."""
    from sgen import settings

    path = tmp_path / "settings.local.yaml"
    path.write_text("defaults:\n  romanize_script: urdoo\n", encoding="utf-8")
    with pytest.raises(settings.SettingsError, match="romanize_script"):
        settings.load(path)


def test_a_bad_choice_is_refused_before_it_is_written(tmp_path):
    """`sgen config --set` validates first: it should not be possible to leave a
    value in the file that the next load will reject."""
    from sgen import settings

    path = tmp_path / "settings.local.yaml"
    settings.ensure_file(path)
    with pytest.raises(settings.SettingsError, match="latin, urdu, both"):
        settings.set_values({"defaults.romanize_script": "nastaliq"}, path)
    assert "nastaliq" not in path.read_text(encoding="utf-8")
    settings.set_values({"defaults.romanize_script": "urdu"}, path)
    assert settings.load(path).defaults.romanize_script == "urdu"


def test_one_file_can_choose_its_own_script(tmp_path):
    from sgen import folderconf

    (tmp_path / "song.mp4").write_bytes(b"x")
    folderconf.set_for_file(tmp_path, tmp_path / "song.mp4", {"romanize_script": "urdu"})
    options = folderconf.apply_to_options(
        {"romanize": False, "romanize_script": "latin"},
        folderconf.for_file(tmp_path, tmp_path / "song.mp4"),
    )
    assert options["romanize_script"] == "urdu"
    assert options["romanize"] is True, "asking for a script means wanting one written"


def test_an_unknown_script_is_refused_per_file(tmp_path):
    from sgen import folderconf

    (tmp_path / "song.mp4").write_bytes(b"x")
    with pytest.raises(folderconf.FolderConfigError, match="script"):
        folderconf.set_for_file(
            tmp_path, tmp_path / "song.mp4", {"romanize_script": "nastaliq"}
        )
