"""Write a Hindi transcript in Urdu script.

Hindi and Urdu are one spoken language with two alphabets, so this is the same
operation as the Latin path in `translit` — transliteration, not translation. It
is harder in one specific way, and the difficulty decides the whole design:

    Devanagari -> Latin is many-to-one, and safe.
    Devanagari -> Urdu is one-to-many, and guesses.

Urdu keeps Perso-Arabic spelling for Perso-Arabic words. /z/ is ز, ذ, ض or ظ;
/s/ is س, ص or ث; /t/ is ت or ط — chosen by the word's origin, not its sound.
Devanagari collapsed all of that centuries ago: it writes स for all three /s/
letters and क for both /k/ and /q/. Going back needs a lexicon, not a table.

So there are three parts, in order of how much they matter:

1. A **word list** for Perso-Arabic vocabulary, which in film dialogue is most of
   the content words. حق, not ہک. It is a convention list, not a dictionary: a
   word that is not in it comes out phonetically right and orthographically
   naive, and that is the known limit of this module.
2. A **rule for the future tense**, because Urdu writes it as two words with a
   nasal — जाऊँगा is جاؤں گا, not جاونگا — and Devanagari joins it. No letter
   table can produce that space or that hamza, and nearly every line of a song
   is a future verb.
3. The **letter tables**, which are the easy part.

Only Hindi is offered. Marathi or Nepali in Urdu script would be a curiosity
rather than something anyone reads, and Punjabi's Gurmukhi -> Shahmukhi is the
same idea in a different script block, which this does not implement.
"""

from __future__ import annotations

import re

# Hindi only — see the module docstring.
LANGUAGES = {"hi"}

CONSONANTS = {
    "क": "ک", "ख": "کھ", "ग": "گ", "घ": "گھ", "ङ": "ن",
    "च": "چ", "छ": "چھ", "ज": "ج", "झ": "جھ", "ञ": "ن",
    "ट": "ٹ", "ठ": "ٹھ", "ड": "ڈ", "ढ": "ڈھ", "ण": "ن",
    "त": "ت", "थ": "تھ", "द": "د", "ध": "دھ", "न": "ن",
    "प": "پ", "फ": "پھ", "ब": "ب", "भ": "بھ", "म": "م",
    "य": "ی", "र": "ر", "ल": "ل", "व": "و",
    "श": "ش", "ष": "ش", "स": "س", "ह": "ہ", "ळ": "ل",
    # Nukta forms. Hindi keeps the Perso-Arabic sounds when it bothers to write
    # the dot; Whisper usually does not, which is the problem the word list is
    # there to absorb.
    "क़": "ق", "ख़": "خ", "ग़": "غ", "ज़": "ز", "ड़": "ڑ", "ढ़": "ڑھ", "फ़": "ف",
}

# Independent vowels, which in practice only ever begin a word.
INITIAL_VOWELS = {
    "अ": "ا", "आ": "آ", "इ": "ا", "ई": "ای", "उ": "ا", "ऊ": "او",
    "ए": "اے", "ऐ": "اے", "ओ": "او", "औ": "او", "ऋ": "ر",
}

# Vowel signs. Urdu writes the long vowels and leaves the short ones out
# entirely, which is why so much of this table is an empty string.
MATRAS = {
    "ा": "ا", "ि": "", "ी": "ی", "ु": "", "ू": "و", "ृ": "ر",
    "े": "ی", "ै": "ی", "ो": "و", "ौ": "و",
}
# ے and آ are word-final and word-initial letters respectively; inside a word the
# same sounds are written ی and ا.
MATRAS_FINAL = {**MATRAS, "े": "ے", "ै": "ے"}

VIRAMA = "्"
NASAL_FINAL = "ں"
NASAL_MEDIAL = "ن"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# --------------------------------------------------------------------------- #
# the words a table cannot get right
# --------------------------------------------------------------------------- #

WORDS = {
    # Perso-Arabic vocabulary: the letters are chosen by etymology, so a table
    # cannot reach them. Collected from real film transcripts.
    "खैरियत": "خیریت", "कैफियत": "کیفیت", "फिल्हाल": "فی الحال",
    "अन्जाम": "انجام", "इन्तज़ार": "انتظار", "इंतजार": "انتظار",
    "दिल": "دل", "दर्द": "درد", "प्यार": "پیار", "इश्क": "عشق",
    "ज़िन्दगी": "زندگی", "जिंदगी": "زندگی", "मोहब्बत": "محبت",
    "खुदा": "خدا", "दुआ": "دعا", "सच": "سچ", "वक्त": "وقت",
    "हाल": "حال", "सफर": "سفر", "मंज़िल": "منزل", "तन्हा": "تنہا",
    "याद": "یاد", "ख्वाब": "خواب", "जान": "جان", "जानम": "جانم",
    "सनम": "صنم", "आसमान": "آسمان", "जमीन": "زمین", "हिम्मत": "ہمت",
    "फिक्र": "فکر", "नसीब": "نصیب", "किस्मत": "قسمت", "मुश्किल": "مشکل",
    "सुबह": "صبح", "शाम": "شام", "रात": "رات", "दुनिया": "دنیا",
    "बेवफा": "بیوفا", "वफा": "وفا", "जुदा": "جدا", "मजबूर": "مجبور",
    "हक": "حق", "कसम": "قسم", "खातिर": "خاطر", "तकदीर": "تقدیر",
    "तकदीरां": "تقدیراں", "शक": "شک", "बात": "بات", "बातें": "باتیں",
    "वादा": "وعدہ", "इरादा": "ارادہ", "जरूरी": "ضروری", "जरूर": "ضرور",
    "मतलब": "مطلب", "तसवीर": "تصویر", "हकीकत": "حقیقت", "अफसाना": "افسانہ",
    "इजाजत": "اجازت", "हसरत": "حسرت", "कातिल": "قاتل", "नजर": "نظر",
    "नजरें": "نظریں", "गजल": "غزل", "जज्बात": "جذبات", "शराब": "شراب",
    "मुसाफिर": "مسافر", "आखिर": "آخر", "जख्म": "زخم", "गम": "غم",
    "इंसान": "انسان", "खामोश": "خاموش", "मासूम": "معصوم", "मजाक": "مذاق",
    # Hindi फ stands for both /f/ (loanwords) and /pʰ/ (native words), and the
    # two take different Urdu letters. The table defaults to پھ, so the /f/ words
    # are listed and the native ones are listed to be safe.
    "फिर": "پھر", "फूल": "پھول", "फल": "پھل", "फेंक": "پھینک",
    "काफिर": "کافر", "फकीर": "فقیر", "फर्क": "فرق", "फैसला": "فیصلہ",
    "तूफान": "طوفان", "खिलाफ": "خلاف", "फरियाद": "فریاد", "फना": "فنا",
    # Grammar words. High frequency, and several are irregular.
    "है": "ہے", "हैं": "ہیں", "हूँ": "ہوں", "हूं": "ہوں", "था": "تھا",
    "थी": "تھی", "थे": "تھے", "और": "اور", "क्या": "کیا", "क्यों": "کیوں",
    "नहीं": "نہیں", "मैं": "میں", "में": "میں", "मेरा": "میرا",
    "मेरी": "میری", "मेरे": "میرے", "तेरा": "تیرا", "तेरी": "تیری",
    "तेरे": "تیرے", "तुम": "تم", "हम": "ہم", "ये": "یہ", "यह": "یہ",
    "वो": "وہ", "वह": "وہ", "को": "کو", "का": "کا", "की": "کی",
    "के": "کے", "से": "سے", "पे": "پے", "पर": "پر", "भी": "بھی",
    "तो": "تو", "ही": "ہی", "कभी": "کبھی", "अब": "اب", "जब": "جب",
    "कुछ": "کچھ", "बिन": "بن", "बिना": "بنا", "दूरियां": "دوریاں",
    "पूछो": "پوچھو", "बताओ": "بتاؤ", "कहो": "کہو", "सुनो": "سنو",
    "साथ": "ساتھ", "अपना": "اپنا", "अपनी": "اپنی", "हर": "ہر",
}

# Whisper's Devanagari varies in ways that hide a word from the list: it rarely
# writes nuktas (मंज़िल -> मंजिल) and spells a nasal either as anusvara or as a
# full consonant with virama (मंजिल / मन्जिल). Normalise both away on each side.
_NUKTA = "़"
_CLUSTER_NASAL = re.compile(r"न्(?=[जझचछतथदधटठडढ])|म्(?=[पफबभ])")


def _normal(word: str) -> str:
    word = word.replace(_NUKTA, "").replace("ँ", "ं")
    return _CLUSTER_NASAL.sub("ं", word)


_LOOKUP = {_normal(k): v for k, v in WORDS.items()}

# The future tense: Urdu writes it as two words with a nasal, Devanagari joins
# it. जाऊँगा -> جاؤں گا.
FUTURE = [
    ("ऊँगा", "ؤں گا"), ("ऊंगा", "ؤں گا"), ("ूंगा", "وں گا"), ("ूँगा", "وں گا"),
    ("ओंगा", "ؤں گا"),
    ("ऊँगी", "ؤں گی"), ("ऊंगी", "ؤں گی"), ("ूंगी", "وں گی"),
    ("एंगे", "یں گے"), ("ेंगे", "یں گے"), ("ेंगी", "یں گی"),
    ("ूंगे", "وں گے"), ("ोगे", "و گے"), ("ोगी", "و گی"),
]


def supported(language: str | None) -> bool:
    """Whether Urdu script can be produced for this language."""
    return (language or "").lower() in LANGUAGES


def convert_word(word: str) -> str:
    """One Devanagari word in Urdu script."""
    known = _LOOKUP.get(_normal(word))
    if known:
        return known

    for ending, urdu in FUTURE:
        if word.endswith(ending) and len(word) > len(ending):
            stem = convert_word(word[: -len(ending)])
            # ؤ carries the vowel that a stem ending in ا would otherwise repeat.
            if urdu.startswith("ؤ") and not stem.endswith("ا"):
                urdu = "و" + urdu[1:]
            return stem + urdu

    out: list[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""

        if nxt == _NUKTA and (ch + nxt) in CONSONANTS:
            out.append(CONSONANTS[ch + nxt])
            i += 2
            continue
        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
            i += 1
            continue
        if ch in INITIAL_VOWELS:
            # Only a word-initial vowel takes its alif.
            out.append(INITIAL_VOWELS[ch] if not out
                       else INITIAL_VOWELS[ch].lstrip("اآ") or "ا")
            i += 1
            continue
        if ch in MATRAS:
            rest = word[i + 1:].replace("ं", "").replace("ँ", "")
            table = MATRAS_FINAL if not DEVANAGARI.search(rest) else MATRAS
            out.append(table[ch])
            i += 1
            continue
        if ch in ("ं", "ँ"):
            final = not DEVANAGARI.search(word[i + 1:])
            out.append(NASAL_FINAL if final else NASAL_MEDIAL)
            i += 1
            continue
        if ch == VIRAMA:      # a cluster: Urdu simply writes the letters together
            i += 1
            continue
        if ch == "ः":
            out.append("ہ")
            i += 1
            continue

        out.append(ch)
        i += 1

    # A doubled consonant is one letter in Urdu: लक्खा is لکھا, not لککھا.
    return re.sub(r"(\S)\1", r"\1", "".join(out))


def convert(text: str) -> str:
    """Convert only the tokens that contain Devanagari.

    English words inside Hinglish speech are left exactly as they are, the same
    rule the Latin path follows.
    """
    if not text or not text.strip():
        return text
    return "".join(
        convert_word(part) if DEVANAGARI.search(part) else part
        for part in re.split(r"(\s+|-)", text)
    )


def convert_lines(lines: list[str]) -> list[str]:
    return [convert(line) for line in lines]
