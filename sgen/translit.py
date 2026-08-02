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
