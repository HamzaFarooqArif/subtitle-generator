"""Romanization of non-Latin subtitle text.

For readers who speak a language but do not read its script. The bar is not
scholarly accuracy but readability: what a Hindi speaker would actually type.
"""

import pytest

from sgen import translit

pytest.importorskip("indic_transliteration")


# Real lines from the Khairiyat transcript, plus common test words.
HINDI_CASES = [
    ("खैरियत पूछो कभी तो कैफियत पूछो", "Khairiyat puchho kabhi to kaifiyat puchho"),
    ("तुम्हारे बिन दिवाने का क्या हाल है", "Tumhaare bin divaane ka kya haal hai"),
    ("अन्जाम है तै मेरा", "Anjaam hai tai mera"),
    ("ये दूरियां फिल्हाल है", "Ye duriyaan filhaal hai"),
    ("अच्छा सच्चा प्यार", "Achchha sachcha pyaar"),
]


@pytest.mark.parametrize("devanagari,expected", HINDI_CASES)
def test_known_hindi_lines(devanagari, expected):
    assert translit.romanize(devanagari, "hi") == expected


def test_namaste():
    assert translit.romanize("नमस्ते", "hi") == "Namaste"


def test_final_schwa_deleted():
    """The single biggest readability rule: राम is "raam", not "raama"."""
    assert translit.romanize("राम", "hi") == "Raam"
    assert translit.romanize("दिल", "hi") == "Dil"
    assert translit.romanize("दर्द", "hi") == "Dard"
    assert translit.romanize("इस", "hi") == "Is"


def test_long_final_vowel_shortened_not_deleted():
    """"kaa" must become "ka", not vanish to "k"."""
    assert translit.romanize("का", "hi") == "Ka"
    assert translit.romanize("क्या", "hi") == "Kya"
    assert translit.romanize("ना", "hi") == "Na"


def test_conjuncts_are_not_mangled():
    """Regression: aksharamukha's RomanReadable turns अन्जाम into "anyaama"."""
    assert "anjaam" in translit.romanize("अन्जाम", "hi").lower()
    assert "y" not in translit.romanize("अन्जाम", "hi").lower()


def test_pha_reads_as_f_in_hindi():
    assert "kaifiyat" in translit.romanize("कैफियत", "hi").lower()
    assert "ph" not in translit.romanize("कैफियत", "hi").lower()


def test_danda_becomes_a_full_stop():
    out = translit.romanize("मेरा नाम राम है। नमस्ते।", "hi")
    assert out.count(".") == 2, out
    assert "।" not in out


def test_english_words_in_mixed_text_survive_untouched():
    """A blanket ph->f rule would turn "phone" into "fone"."""
    out = translit.romanize("hello नमस्ते my phone", "hi")
    assert "phone" in out
    assert "namaste" in out
    assert "fone" not in out


def test_latin_text_passes_through_unchanged():
    text = "This is already English."
    assert translit.romanize(text, "en") == text
    assert translit.romanize(text, None) == text


def test_empty_and_whitespace_safe():
    assert translit.romanize("", "hi") == ""
    assert translit.romanize("   ", "hi") == "   "


def test_script_detection():
    assert translit.script_of("नमस्ते") == "Devanagari"
    assert translit.script_of("வணக்கம்") == "Tamil"
    assert translit.script_of("hello") is None


def test_language_without_support_is_unchanged():
    """Arabic and Japanese are not handled; they must pass through, not crash."""
    for text, lang in (("مرحبا", "ar"), ("こんにちは", "ja"), ("Γειά σου", "el")):
        assert translit.romanize(text, lang) == text


def test_supported_reports_accurately():
    assert translit.supported("hi")
    assert translit.supported("ta")
    assert translit.supported("ru")
    assert not translit.supported("en")
    assert not translit.supported("ar")
    assert not translit.supported(None)


def test_other_indic_scripts_produce_latin():
    """Tamil and Bengali should romanize even if tuned less finely than Hindi."""
    for text, lang in (("வணக்கம்", "ta"), ("নমস্কার", "bn")):
        out = translit.romanize(text, lang)
        assert out != text
        assert out.isascii(), out


def test_romanize_lines_preserves_structure():
    lines = ["खैरियत पूछो", "कभी तो कैफियत पूछो"]
    out = translit.romanize_lines(lines, "hi")
    assert len(out) == 2
    assert all(l.isascii() for l in out)


def test_conventional_spellings_applied():
    """में transliterates to "men", which collides with an English word."""
    out = translit.romanize("इस दर्द में", "hi")
    assert "mein" in out
    assert " men" not in out


# --------------------------------------------------------------------------- #
# Nukta letters
#
# The dot under a consonant that borrows a sound the script has no letter for.
# Getting it wrong does not produce obvious garbage — it produces a real word
# spelled with the letter next door, which reads as plausible and is therefore
# the error most likely to go unnoticed.
# --------------------------------------------------------------------------- #

# (source, language, must appear, must NOT appear, why)
NUKTA_CASES = [
    ("ਸ਼ਾਮ", "pa", "shaam", "saam", "ਸ਼ is sh, not s"),
    ("ਜ਼ਿੰਦਗੀ", "pa", "zindagi", "jindagi", "ਜ਼ is z, not j"),
    ("ਖ਼ਾਸ", "pa", "khaas", "kaas", "ਖ਼ keeps its aspiration"),
    ("ਫ਼ੌਜ", "pa", "fauj", "phauj", "ਫ਼ is f"),
    ("ਗ਼ਜ਼ਲ", "pa", "ghazal", "gazal", "two nuktas in one word"),
    ("ਵੜਿਆ", "pa", "vadia", ".d", "ੜ is a letter, not a dot"),
    ("ख़ास", "hi", "khaas", "kaas", "ख़ keeps its aspiration"),
    ("ग़ज़ल", "hi", "ghazal", "gazal", "ग़ is gh, ज़ is z"),
    ("बड़ा", "hi", "bada", ".d", "ड़ is a letter, not a dot"),
    ("पढ़ना", "hi", "padh", ".dh", "ढ़ is a letter, not a dot"),
    ("छोड़ूंगा", "hi", "chhodunga", ".", "the dot never reaches the reader"),
    ("इश्क़", "hi", "ishq", "ishk", "क़ is q"),
    ("मंज़र", "hi", "manzar", "manjar", "ज़ is z, not j"),
]


@pytest.mark.parametrize("text,lang,wanted,unwanted,why", NUKTA_CASES)
def test_nukta_letters_keep_their_own_sound(text, lang, wanted, unwanted, why):
    out = translit.romanize(text, lang).lower()
    assert wanted in out, f"{why}: {text} -> {out!r}"
    assert unwanted not in out, f"{why}: {text} -> {out!r}"


@pytest.mark.parametrize("text,lang", [(c[0], c[1]) for c in NUKTA_CASES])
def test_no_nukta_survives_into_latin_text(text, lang):
    """A mark from the source script in supposedly-Latin output is a bug.

    sanscript's Gurmukhi scheme has no rule for the nukta, so it used to leave
    the raw combining character in the result: ਸ਼ਾਮ came out "Sa਼aam".
    """
    out = translit.romanize(text, lang)
    assert out.isascii(), out


def test_precomposed_and_decomposed_nuktas_agree():
    """The same letter encoded two ways must romanize the same way.

    ज़ is either one codepoint or ज followed by a combining nukta, and a
    transcript can contain either.
    """
    import unicodedata

    for text, lang in (("ज़िंदगी", "hi"), ("ख़ास", "hi"), ("ਜ਼ਿੰਦਗੀ", "pa")):
        composed = unicodedata.normalize("NFC", text)
        decomposed = unicodedata.normalize("NFD", text)
        assert translit.romanize(composed, lang) == translit.romanize(decomposed, lang)


def test_a_nukta_does_not_change_its_neighbours():
    """The fix must not reach beyond the letter carrying the dot."""
    # ङ (ITRANS "~N"/"NG") shares a letter with the ग़ rule; ਸ next to ਸ਼ must stay s.
    assert translit.romanize("ਸਾਸ ਸ਼ਾਮ", "pa").lower() == "saas shaam"
    assert "ng" in translit.romanize("रंग", "hi").lower()


# --------------------------------------------------------------------------- #
# Cyrillic
#
# The scheme is BGN/PCGN for the East Slavic languages — the one an English
# reader already knows from names and street signs — the official streamlined
# system for Bulgarian, and for Serbian and Macedonian the Latin alphabet those
# languages already use.
# --------------------------------------------------------------------------- #

# Lines from a real Russian transcript, plus the words that exercise the rules.
RUSSIAN_CASES = [
    ("Тихо, может звонит.", "Tikho, mozhet zvonit."),
    ("Да, любимый?", "Da, lyubimyy?"),
    ("Музыка Секунду.", "Muzyka Sekundu."),
    ("Я в Москве.", "Ya v Moskve."),
    ("Юрий Гагарин", "Yuriy Gagarin"),
    ("Санкт-Петербург", "Sankt-Peterburg"),
    ("щи", "shchi"),
    ("хорошо", "khorosho"),
    ("цена", "tsena"),
]


@pytest.mark.parametrize("cyrillic,expected", RUSSIAN_CASES)
def test_known_russian_lines(cyrillic, expected):
    assert translit.romanize(cyrillic, "ru") == expected


def test_ye_at_the_start_of_a_word_and_after_a_vowel():
    """е is /je/ where a vowel or nothing precedes it, /e/ after a consonant."""
    assert translit.romanize("Елена", "ru") == "Yelena"
    assert translit.romanize("моей", "ru") == "moyey"
    assert translit.romanize("Петербург", "ru") == "Peterburg"   # not Pyeterburg


def test_yo_is_o_after_a_hushing_consonant():
    """жёлтый is "zholtyy" — "zhyoltyy" is nobody's spelling."""
    assert translit.romanize("жёлтый", "ru") == "zholtyy"
    assert translit.romanize("ёлка", "ru") == "yolka"


def test_soft_and_hard_signs_disappear():
    assert translit.romanize("день", "ru") == "den"
    assert translit.romanize("объект", "ru") == "obyekt"


def test_capitalisation_survives():
    assert translit.romanize("МОСКВА", "ru") == "MOSKVA"
    assert translit.romanize("Москва", "ru") == "Moskva"
    # A multi-letter replacement takes the case of the letter it replaces.
    assert translit.romanize("Щи", "ru") == "Shchi"
    assert translit.romanize("ЩИ", "ru") == "SHCHI"


def test_ukrainian_has_its_own_letters():
    assert translit.romanize("Привіт", "uk") == "Pryvit"      # и is y, і is i
    assert translit.romanize("Їжа", "uk") == "Yizha"
    assert translit.romanize("гарно", "uk") == "harno"        # г is h, not g


def test_bulgarian_differs_from_russian():
    assert translit.romanize("България", "bg") == "Balgariya"   # ъ is a vowel
    assert translit.romanize("Щастие", "bg") == "Shtastie"      # щ is sht
    assert translit.romanize("Елена", "bg") == "Elena"          # no ye rule


def test_serbian_and_macedonian_use_their_own_latin():
    assert translit.romanize("Његош", "sr") == "Njegoš"
    assert translit.romanize("Београд", "sr") == "Beograd"
    assert translit.romanize("Скопје", "mk") == "Skopje"


def test_english_and_numbers_inside_cyrillic_survive():
    assert translit.romanize("Москва OK 2019", "ru") == "Moskva OK 2019"


def test_cyrillic_is_detected_without_a_language_code():
    assert translit.script_of("привет") == "Cyrillic"
    assert translit.romanize("привет") == "privet"


def test_the_indic_path_still_works():
    """The Cyrillic branch returns before sanscript is ever consulted; make sure
    it did not swallow the languages the module was written for."""
    assert translit.romanize("नमस्ते", "hi") == "Namaste"


# --------------------------------------------------------------------------- #
# file writing
# --------------------------------------------------------------------------- #

def test_write_romanized_creates_latn_files(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_romanized

    cues = [Cue(start=1.0, end=3.0, lines=["खैरियत पूछो", "कभी तो कैफियत पूछो"])]
    paths = write_romanized(cues, tmp_path / "song", ["srt"], "hi")
    assert [p.name for p in paths] == ["song.hi-Latn.srt"]
    body = paths[0].read_text(encoding="utf-8-sig")
    assert "Khairiyat" in body
    assert "खैरियत" not in body


def test_write_romanized_skips_unsupported_language(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_romanized

    cues = [Cue(start=1.0, end=3.0, lines=["Hello there"])]
    assert write_romanized(cues, tmp_path / "clip", ["srt"], "en") == []


def test_write_romanized_handles_russian(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_romanized

    cues = [Cue(start=1.0, end=3.0, lines=["Тихо, может звонит."])]
    paths = write_romanized(cues, tmp_path / "clip", ["srt"], "ru")
    assert [p.name for p in paths] == ["clip.ru-Latn.srt"]
    body = paths[0].read_text(encoding="utf-8-sig")
    assert "Tikho, mozhet zvonit." in body
    assert "Тихо" not in body


def test_a_language_with_no_romanizer_says_so(tmp_path):
    """A ticked box that quietly does nothing is worse than one that says why."""
    from sgen.cues import Cue
    from sgen.write import romanize_or_explain

    cues = [Cue(start=1.0, end=3.0, lines=["こんにちは"])]
    paths, note = romanize_or_explain(cues, tmp_path / "clip", ["srt"], "ja")
    assert paths == []
    assert "ja" in note and "Latin-script" in note
    assert "Cyrillic" in note, "it should say what is available"


def test_nothing_is_said_when_it_worked(tmp_path):
    from sgen.cues import Cue
    from sgen.write import romanize_or_explain

    cues = [Cue(start=1.0, end=3.0, lines=["Да, любимый?"])]
    paths, note = romanize_or_explain(cues, tmp_path / "clip", ["srt"], "ru")
    assert [p.name for p in paths] == ["clip.ru-Latn.srt"]
    assert note == ""


def test_romanized_keeps_timings(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_romanized

    cues = [Cue(start=12.5, end=15.25, lines=["नमस्ते"])]
    body = write_romanized(cues, tmp_path / "x", ["srt"], "hi")[0].read_text(
        encoding="utf-8-sig"
    )
    assert "00:00:12,500 --> 00:00:15,250" in body
