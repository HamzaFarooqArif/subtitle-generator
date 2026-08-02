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
    """Arabic and Cyrillic are not handled; they must pass through, not crash."""
    for text, lang in (("مرحبا", "ar"), ("привет", "ru"), ("こんにちは", "ja")):
        assert translit.romanize(text, lang) == text


def test_supported_reports_accurately():
    assert translit.supported("hi")
    assert translit.supported("ta")
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


def test_romanized_keeps_timings(tmp_path):
    from sgen.cues import Cue
    from sgen.write import write_romanized

    cues = [Cue(start=12.5, end=15.25, lines=["नमस्ते"])]
    body = write_romanized(cues, tmp_path / "x", ["srt"], "hi")[0].read_text(
        encoding="utf-8-sig"
    )
    assert "00:00:12,500 --> 00:00:15,250" in body
