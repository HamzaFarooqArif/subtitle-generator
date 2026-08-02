"""Romanize non-Latin subtitle text.

For readers who speak a language but not its script: नमस्ते -> "namaste".
This is transliteration, not translation — the words are unchanged, only the
letters they are written with.

Why the output needs post-processing rather than a library scheme straight:

* Strict transliteration keeps the inherent vowel that Hindi does not pronounce,
  giving "khairiyata", "naama raama", "isa darda" where a Hindi speaker writes
  "khairiyat", "naam raam", "is dard". Word-final schwa deletion is the single
  biggest readability difference, and no scheme applies it.
* Aksharamukha's "RomanReadable" is diacritic-free but mangles conjuncts —
  अन्जाम becomes "anyaama" instead of "anjaam". ITRANS gets conjuncts right, so
  ITRANS is the base and its capital-letter long vowels are mapped down here.
* फ is /f/ in Hindi, so ITRANS "ph" reads better as "f" ("kaifiyat", not
  "kaiphiyata").

Rules are applied per token, and only to tokens actually containing the source
script, so English words mixed into Hinglish text pass through untouched — a
blanket ph->f would otherwise turn "phone" into "fone".
"""

from __future__ import annotations

import re
from typing import Callable

# Unicode block ranges per script we can romanize.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Oriya": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Sinhala": [(0x0D80, 0x0DFF)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
}

# Whisper language code -> sanscript source scheme name.
LANGUAGE_SCRIPTS: dict[str, str] = {
    "hi": "Devanagari", "mr": "Devanagari", "ne": "Devanagari", "sa": "Devanagari",
    "bn": "Bengali", "as": "Bengali",
    "pa": "Gurmukhi",
    "gu": "Gujarati",
    "or": "Oriya",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "si": "Sinhala",
    "ru": "Cyrillic", "uk": "Cyrillic", "be": "Cyrillic",
    "bg": "Cyrillic", "sr": "Cyrillic", "mk": "Cyrillic",
}

# Languages whose romanization benefits from Hindi-style schwa deletion.
_SCHWA_DELETING = {"hi", "mr", "ne", "pa", "gu", "bn"}

_TOKEN = re.compile(r"(\s+)")


def script_of(text: str) -> str | None:
    """Dominant non-Latin script in `text`, or None if it is already Latin."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] = counts.get(script, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def _has_script(token: str, script: str) -> bool:
    ranges = SCRIPT_RANGES[script]
    return any(any(lo <= ord(ch) <= hi for lo, hi in ranges) for ch in token)


# --------------------------------------------------------------------------- #
# ITRANS -> Hinglish conventions
# --------------------------------------------------------------------------- #

# Order matters: multi-character sequences before single characters.
_ITRANS_MAP: list[tuple[str, str]] = [
    ("RRi", "ri"), ("RRI", "ri"), ("LLi", "li"),
    ("Ch", "chh"),          # छ -> chh, keeping च as ch
    ("Sh", "sh"), ("shh", "sh"),
    ("~n", "n"), ("~N", "n"), ("NG", "ng"),
    ("A", "aa"),            # long vowels: ITRANS capitalizes them
    ("I", "i"), ("U", "u"),
    ("E", "e"), ("O", "o"),
    ("M", "n"),             # anusvara reads as a nasal n
    ("H", "h"),             # visarga
    ("D", "d"), ("T", "t"), ("N", "n"), ("S", "sh"), ("L", "l"), ("R", "r"),
    (".n", "n"), (".m", "m"), (".h", ""),
    # ITRANS renders the danda as a pipe. It is sentence-final punctuation, so it
    # becomes a full stop rather than being dropped.
    ("||", "."), ("|", "."),
    ("^", ""), ("'", ""),
]

# Conventional Hinglish spellings for very common words where a mechanical
# transliteration reads awkwardly. Deliberately tiny — this is a convention
# list, not a dictionary, and every entry earns its place by frequency.
_CONVENTIONAL = {
    "men": "mein",     # में — otherwise collides with the English word "men"
    "hun": "hoon",     # हूँ
    "kuchh": "kuch",
    "yah": "yeh",
    "vah": "voh",
    "kah": "keh",
}

_VOWELS = set("aeiou")


def _apply_itrans_map(token: str) -> str:
    for src, dst in _ITRANS_MAP:
        token = token.replace(src, dst)
    return token


def _delete_final_schwa(word: str) -> str:
    """Drop the inherent vowel Hindi does not pronounce at the end of a word.

    "khairiyata" -> "khairiyat", "raama" -> "raam", "dila" -> "dil".
    A final long vowel is shortened instead ("kaa" -> "ka", "sachchaa" ->
    "sachcha") and never deleted, or particles would vanish entirely.
    """
    core = word
    trailing = ""
    # Keep punctuation attached to the word out of the way.
    while core and not core[-1].isalpha():
        trailing = core[-1] + trailing
        core = core[:-1]

    if len(core) >= 3 and core.endswith("aa"):
        return core[:-1] + trailing
    if len(core) >= 3 and core.endswith("a") and core[-2] not in _VOWELS:
        return core[:-1] + trailing
    return core + trailing


def _hindi_polish(word: str) -> str:
    # फ is /f/ in Hindi; "kaifiyat" reads better than "kaiphiyata".
    word = word.replace("ph", "f").replace("Ph", "F")
    word = re.sub(r"([aeiou])\1{2,}", r"\1\1", word)  # collapse runaway vowels

    # Apply conventional spellings to the alphabetic core, preserving any
    # punctuation the word carries.
    match = re.match(r"^(\W*)(\w+)(\W*)$", word, re.UNICODE)
    if match:
        head, core, tail = match.groups()
        replacement = _CONVENTIONAL.get(core.lower())
        if replacement:
            if core[:1].isupper():
                replacement = replacement.capitalize()
            word = head + replacement + tail
    return word


# --------------------------------------------------------------------------- #
# Cyrillic
#
# Not sanscript's job, and not a schwa problem either — a letter table plus a
# couple of context rules is the whole of it. The schemes chosen are the ones a
# reader of English already recognises from names and street signs (BGN/PCGN for
# the East Slavic languages, the official streamlined system for Bulgarian) and,
# for Serbian and Macedonian, the Latin alphabet those languages already use.
# --------------------------------------------------------------------------- #

_RU = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# The Ukrainian apostrophe is dropped, but only the typographic ones: mapping
# ASCII ' too would turn an English "don't" in mixed text into "dont".
_UK = {
    **_RU,
    "г": "h", "ґ": "g", "и": "y", "і": "i", "ї": "yi", "є": "ye",
    "е": "e", "щ": "shch", "’": "", "ʼ": "",
}

_BE = {**_RU, "г": "h", "і": "i", "ў": "w", "’": "", "ʼ": ""}

# Bulgarian has no ye rule, ъ is a vowel, and щ is "sht".
_BG = {**_RU, "х": "h", "щ": "sht", "ъ": "a", "ь": "y", "ё": "yo"}

# Gaj's Latin: not a transliteration so much as the language's other alphabet.
_SR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}

_MK = {**_SR, "ѓ": "gj", "ќ": "kj", "ѕ": "dz", "ђ": "gj", "ћ": "kj"}

CYRILLIC_TABLES: dict[str, dict[str, str]] = {
    "ru": _RU, "uk": _UK, "be": _BE, "bg": _BG, "sr": _SR, "mk": _MK,
}

# Languages where "е" is /je/ at the start of a word and after a vowel.
_YE_LANGUAGES = {"ru", "be"}
_CYRILLIC_VOWELS = set("аеёиоуыэюяіїєўАЕЁИОУЫЭЮЯІЇЄЎ")
# After these, "ё" is /o/ — "жёлтый" is "zholtyy", not "zhyoltyy".
_HUSHING = set("жчшщ")


def _cyrillic_case(source: str, latin: str, run_is_caps: bool) -> str:
    """Carry the original letter's case onto a replacement of any length."""
    if not source.isupper() or not latin:
        return latin
    return latin.upper() if run_is_caps else latin[0].upper() + latin[1:]


def _romanize_cyrillic(text: str, language: str) -> str:
    table = CYRILLIC_TABLES.get(language, _RU)
    ye_rule = language in _YE_LANGUAGES
    out: list[str] = []

    for i, ch in enumerate(text):
        latin = table.get(ch.lower())
        if latin is None:
            out.append(ch)
            continue

        low = ch.lower()
        before = text[i - 1] if i else ""
        if ye_rule and low == "е" and (
            not before or before.lower() not in table
            or before in _CYRILLIC_VOWELS or before.lower() in ("ь", "ъ", "й")
        ):
            latin = "ye"          # Елена -> Yelena, моей -> moyey
        elif low == "ё" and before.lower() in _HUSHING:
            latin = "o"           # жёлтый -> zholtyy

        after = text[i + 1] if i + 1 < len(text) else ""
        caps_run = ch.isupper() and (
            after.isupper() and after.lower() in table
            or before.isupper() and before.lower() in table
        )
        out.append(_cyrillic_case(ch, latin, caps_run))

    return "".join(out)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

class TransliterationUnavailable(RuntimeError):
    pass


def _transliterator(script: str) -> Callable[[str], str]:
    try:
        from indic_transliteration import sanscript
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise TransliterationUnavailable(
            "indic-transliteration is not installed. Run: pip install -r requirements.txt"
        ) from exc

    source = getattr(sanscript, script.upper(), None)
    if source is None:
        raise TransliterationUnavailable(f"no sanscript scheme for {script}")
    return lambda text: sanscript.transliterate(text, source, sanscript.ITRANS)


def romanize(text: str, language: str | None = None) -> str:
    """Return `text` in Latin letters, or unchanged if it already is.

    Tokens without the source script pass through untouched, so English words in
    mixed Hinglish text are preserved exactly.
    """
    if not text or not text.strip():
        return text

    script = LANGUAGE_SCRIPTS.get((language or "").lower()) or script_of(text)
    if script is None or script not in SCRIPT_RANGES:
        return text

    if script == "Cyrillic":
        # Russian unless told otherwise: it is the language most Cyrillic text
        # detected without a language code turns out to be.
        return _romanize_cyrillic(text, (language or "ru").lower())

    try:
        convert = _transliterator(script)
    except TransliterationUnavailable:
        raise

    schwa = (language or "").lower() in _SCHWA_DELETING or (
        language is None and script == "Devanagari"
    )

    out: list[str] = []
    for part in _TOKEN.split(text):
        if not part or part.isspace() or not _has_script(part, script):
            out.append(part)
            continue
        word = _apply_itrans_map(convert(part))
        if schwa:
            word = _delete_final_schwa(word)
            word = _hindi_polish(word)
        out.append(word)

    result = "".join(out)
    # Sentence-initial capital, matching how the native-script line reads.
    return result[:1].upper() + result[1:] if result else result


def supported(language: str | None) -> bool:
    """Whether romanization is available for this language."""
    return (language or "").lower() in LANGUAGE_SCRIPTS


def romanize_lines(lines: list[str], language: str | None = None) -> list[str]:
    return [romanize(line, language) for line in lines]
