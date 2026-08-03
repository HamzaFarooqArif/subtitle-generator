"""Write a Hindi or Punjabi transcript in Urdu script.

Two languages, one reason: both are written in two alphabets, and the Perso-Arabic
one is what a reader on the other side of a border reads.

    Hindi (Devanagari)   <->  Urdu (Perso-Arabic)
    Punjabi (Gurmukhi)   <->  Punjabi (Shahmukhi)

This is transliteration, not translation — same words, different letters. It is
harder than the Latin path in one specific way, and that difficulty decides the
whole design:

    Devanagari -> Latin is many-to-one, and safe.
    Devanagari -> Urdu is one-to-many, and guesses.

Urdu and Shahmukhi keep Perso-Arabic spelling for Perso-Arabic words. /z/ is ز,
ذ, ض or ظ; /s/ is س, ص or ث; /t/ is ت or ط — selected by the word's etymology,
not by its sound. Devanagari and Gurmukhi both discarded those distinctions
centuries ago: Devanagari writes स for all three /s/ letters, Gurmukhi ਸ. **The
information needed to spell correctly is not in the input.**

So each script gets three parts, in order of how much they matter:

1. A **word list** for Perso-Arabic vocabulary, which in film dialogue is most of
   the content words. حق, not the phonetically-correct-and-wrong ہک. It is a
   convention list, not a dictionary: a word outside it comes out sounding right
   and spelled naively, and that is the known limit of this module.
2. A **rule for the future tense**, because Urdu writes it as two words with a
   nasal — जाऊँगा is جاؤں گا, ਜਾਵਾਂਗਾ is جاواں گا — and both source scripts join
   it. No letter table can produce that space, and nearly every line of a song
   lyric is a future verb.
3. The **letter tables**, which are the easy part.

Only these two. Marathi and Nepali are also written in Devanagari and would
convert mechanically, but nobody reads Marathi in Urdu letters, the word list is
Hindi-Urdu vocabulary, and the future rule is Hindi morphology — so the output
would be worse in ways its reader could not see. Sindhi and Kashmiri are
Perso-Arabic languages too, but Whisper's support for them is too thin to build
on.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Devanagari (Hindi)
# --------------------------------------------------------------------------- #

DEVA_CONSONANTS = {
    "क": "ک", "ख": "کھ", "ग": "گ", "घ": "گھ", "ङ": "ن",
    "च": "چ", "छ": "چھ", "ज": "ج", "झ": "جھ", "ञ": "ن",
    "ट": "ٹ", "ठ": "ٹھ", "ड": "ڈ", "ढ": "ڈھ", "ण": "ن",
    "त": "ت", "थ": "تھ", "द": "د", "ध": "دھ", "न": "ن",
    "प": "پ", "फ": "پھ", "ब": "ب", "भ": "بھ", "म": "م",
    "य": "ی", "र": "ر", "ल": "ل", "व": "و",
    "श": "ش", "ष": "ش", "स": "س", "ह": "ہ", "ळ": "ل",
    # Nukta forms. Hindi keeps the Perso-Arabic sounds when it writes the dot;
    # Whisper usually does not, which is what the word list absorbs.
    "क़": "ق", "ख़": "خ", "ग़": "غ", "ज़": "ز", "ड़": "ڑ", "ढ़": "ڑھ", "फ़": "ف",
}

DEVA_INITIAL_VOWELS = {
    "अ": "ا", "आ": "آ", "इ": "ا", "ई": "ای", "उ": "ا", "ऊ": "او",
    "ए": "اے", "ऐ": "اے", "ओ": "او", "औ": "او", "ऋ": "ر",
}

# Word-initial e/ai before a consonant is alif + ye, not alif + bari-ye: एक is
# ایک and ऐसा is ایسا, where "اےک" is not a word. اے on its own is the vocative,
# which is why the plain table above keeps it for a vowel standing alone.
DEVA_INITIAL_BEFORE_CONSONANT = {"ए": "ای", "ऐ": "ای"}

# A vowel meeting another vowel takes a hamza seat: हुई is ہوئی, not ہوی, and
# गाओ is گاؤ. Without this the two vowels run together into one letter and the
# word stops being readable.
DEVA_VOWELS_AFTER_VOWEL = {
    "अ": "ا", "आ": "ا", "इ": "ئ", "ई": "ئی", "उ": "ؤ", "ऊ": "ؤ",
    "ए": "ئے", "ऐ": "ئے", "ओ": "ؤ", "औ": "ؤ", "ऋ": "ر",
}

# Urdu writes the long vowels and leaves the short ones out entirely, which is
# why so much of this table is an empty string.
DEVA_MATRAS = {
    "ा": "ا", "ि": "", "ी": "ی", "ु": "", "ू": "و", "ृ": "ر",
    "े": "ی", "ै": "ی", "ो": "و", "ौ": "و",
}

DEVA_FUTURE = [
    ("ऊँगा", "ؤں گا"), ("ऊंगा", "ؤں گا"), ("ूंगा", "وں گا"), ("ूँगा", "وں گا"),
    ("ओंगा", "ؤں گا"),
    ("ऊँगी", "ؤں گی"), ("ऊंगी", "ؤں گی"), ("ूंगी", "وں گی"),
    ("एंगे", "یں گے"), ("ेंगे", "یں گے"), ("ेंगी", "یں گی"),
    ("ूंगे", "وں گے"), ("ोगे", "و گے"), ("ोगी", "و گی"),
]

DEVA_WORDS = {
    # Perso-Arabic vocabulary: spelled by etymology, so out of a table's reach.
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
    # Measured wrong on the Khairiyat transcript: the letter tables gave ہیسیت
    # for a word Urdu spells with ح and ث, and دوانے for one that needs the ی.
    "हैसियत": "حیثیت", "हसियत": "حیثیت", "दिवाने": "دیوانے",
    "दीवाने": "دیوانے", "दीवाना": "دیوانہ", "दिवाना": "دیوانہ",
    "खुश": "خوش", "मंजर": "منظر", "मंज़र": "منظر", "इश्क़": "عشق",
    "ज़िंदगी": "زندگی", "खुशहाल": "خوشحال",
    # फ stands for both /f/ (loanwords) and /pʰ/ (native), and the two take
    # different Urdu letters. The table defaults to پھ, so both are listed.
    "फिर": "پھر", "फूल": "پھول", "फल": "پھل", "फेंक": "پھینک",
    "काफिर": "کافر", "फकीर": "فقیر", "फर्क": "فرق", "फैसला": "فیصلہ",
    "तूफान": "طوفان", "खिलाफ": "خلاف", "फरियाद": "فریاد", "फना": "فنا",
    # Grammar words. High frequency, several irregular.
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

# --------------------------------------------------------------------------- #
# Gurmukhi (Punjabi)
#
# The same relationship, and the one that made Punjabi worth adding: Gurmukhi and
# Shahmukhi are two alphabets for one language, split by a border rather than by
# speech. Punjabi in Shahmukhi is what a reader in Lahore reads.
# --------------------------------------------------------------------------- #

GURU_CONSONANTS = {
    "ਸ": "س", "ਹ": "ہ",
    "ਕ": "ک", "ਖ": "کھ", "ਗ": "گ", "ਘ": "گھ", "ਙ": "ن",
    "ਚ": "چ", "ਛ": "چھ", "ਜ": "ج", "ਝ": "جھ", "ਞ": "ن",
    "ਟ": "ٹ", "ਠ": "ٹھ", "ਡ": "ڈ", "ਢ": "ڈھ", "ਣ": "ن",
    "ਤ": "ت", "ਥ": "تھ", "ਦ": "د", "ਧ": "دھ", "ਨ": "ن",
    "ਪ": "پ", "ਫ": "پھ", "ਬ": "ب", "ਭ": "بھ", "ਮ": "م",
    "ਯ": "ی", "ਰ": "ر", "ਲ": "ل", "ਵ": "و", "ੜ": "ڑ",
    # Precomposed nukta letters, which Gurmukhi text does use for loanwords.
    "ਸ਼": "ش", "ਖ਼": "خ", "ਗ਼": "غ", "ਜ਼": "ز", "ਫ਼": "ف", "ਲ਼": "ل",
}

GURU_INITIAL_VOWELS = {
    "ਅ": "ا", "ਆ": "آ", "ਇ": "ا", "ਈ": "ای", "ਉ": "ا", "ਊ": "او",
    "ਏ": "اے", "ਐ": "اے", "ਓ": "او", "ਔ": "او",
}

GURU_INITIAL_BEFORE_CONSONANT = {"ਏ": "ای", "ਐ": "ای"}

GURU_VOWELS_AFTER_VOWEL = {
    "ਅ": "ا", "ਆ": "ا", "ਇ": "ئ", "ਈ": "ئی", "ਉ": "ؤ", "ਊ": "ؤ",
    "ਏ": "ئے", "ਐ": "ئے", "ਓ": "ؤ", "ਔ": "ؤ",
}

GURU_MATRAS = {
    "ਾ": "ا", "ਿ": "", "ੀ": "ی", "ੁ": "", "ੂ": "و",
    "ੇ": "ی", "ੈ": "ی", "ੋ": "و", "ੌ": "و",
}

# Punjabi's future is -ਾਂਗਾ, and Shahmukhi splits it the same way Urdu does.
GURU_FUTURE = [
    ("ਾਂਗਾ", "اں گا"), ("ਾਂਗੀ", "اں گی"), ("ਾਂਗੇ", "اں گے"),
    ("ੇਗਾ", "ے گا"), ("ੇਗੀ", "ے گی"), ("ਣਗੇ", "ن گے"), ("ਣਗੀਆਂ", "ن گیاں"),
]

GURU_WORDS = {
    # Perso-Arabic vocabulary. Punjabi shares most of it with Urdu, which is why
    # the same approach transfers.
    "ਇਸ਼ਕ": "عشق", "ਦਿਲ": "دل", "ਦਰਦ": "درد", "ਪਿਆਰ": "پیار",
    "ਖੁਦਾ": "خدا", "ਖ਼ੁਦਾ": "خدا", "ਰੱਬ": "رب", "ਦੁਆ": "دعا",
    "ਵਕਤ": "وقت", "ਵਖ਼ਤ": "وقت", "ਹੱਕ": "حق", "ਕਸਮ": "قسم",
    "ਨਸੀਬ": "نصیب", "ਤਕਦੀਰ": "تقدیر", "ਮੰਜ਼ਿਲ": "منزل", "ਸਫ਼ਰ": "سفر",
    "ਜ਼ਿੰਦਗੀ": "زندگی", "ਮੁਹੱਬਤ": "محبت", "ਸ਼ਰਾਬ": "شراب", "ਗ਼ਮ": "غم",
    "ਯਾਦ": "یاد", "ਯਾਰ": "یار", "ਖ਼ਿਆਲ": "خیال", "ਦੁਨੀਆ": "دنیا",
    "ਦੁਨੀਆਂ": "دنیاں", "ਹਵਾ": "ہوا", "ਸ਼ਾਮ": "شام", "ਰਾਤ": "رات",
    "ਸੱਚ": "سچ", "ਇੰਤਜ਼ਾਰ": "انتظار", "ਜਾਨ": "جان", "ਸ਼ੁਕਰ": "شکر",
    "ਮਜਬੂਰ": "مجبور", "ਵਫ਼ਾ": "وفا", "ਬੇਵਫ਼ਾ": "بیوفا", "ਜੁਦਾ": "جدا",
    "ਆਸਮਾਨ": "آسمان", "ਜ਼ਮੀਨ": "زمین", "ਇਨਸਾਨ": "انسان", "ਮਾਸੂਮ": "معصوم",
    "ਮੁਸ਼ਕਿਲ": "مشکل", "ਖ਼ਾਮੋਸ਼": "خاموش", "ਸ਼ੁਰੂ": "شروع",
    "ਤਕਦੀਰ": "تقدیر", "ਤਕਦੀਰਾਂ": "تقدیراں", "ਕਿਸਮਤ": "قسمت",
    "ਸ਼ੌਕ": "شوق", "ਫ਼ਿਕਰ": "فکر", "ਹਾਲ": "حال", "ਹਿੰਮਤ": "ہمت",
    "ਜ਼ਖ਼ਮ": "زخم", "ਨਜ਼ਰ": "نظر", "ਗ਼ਜ਼ਲ": "غزل", "ਮਹਿਬੂਬ": "محبوب",
    # Grammar words, and the ones whose spelling is fixed by convention.
    "ਹੈ": "ہے", "ਹਨ": "ہن", "ਸੀ": "سی", "ਸਨ": "سن", "ਹੋ": "ہو",
    "ਮੈਂ": "میں", "ਤੂੰ": "توں", "ਤੁਸੀਂ": "تسیں", "ਅਸੀਂ": "اسیں",
    "ਮੇਰਾ": "میرا", "ਮੇਰੀ": "میری", "ਮੇਰੇ": "میرے", "ਤੇਰਾ": "تیرا",
    "ਤੇਰੀ": "تیری", "ਤੇਰੇ": "تیرے", "ਦਾ": "دا", "ਦੀ": "دی", "ਦੇ": "دے",
    "ਨੂੰ": "نوں", "ਤੋਂ": "توں", "ਨਾਲ": "نال", "ਵਿੱਚ": "وچ", "ਵਿਚ": "وچ",
    "ਕੀ": "کی", "ਕਿਉਂ": "کیوں", "ਨਹੀਂ": "نہیں", "ਨਾ": "نہ", "ਹੁਣ": "ہن",
    "ਤੇ": "تے", "ਵੀ": "وی", "ਹੀ": "ہی", "ਜੇ": "جے", "ਪਰ": "پر",
    "ਗਿਆ": "گیا", "ਗਈ": "گئی", "ਕਰ": "کر", "ਕਰਦਾ": "کردا", "ਕਿਹਾ": "کہا",
    "ਸਾਥ": "ساتھ", "ਆਪਣਾ": "اپنا", "ਆਪਣੀ": "اپنی", "ਹਰ": "ہر",
    "ਬਹੁਤ": "بہت", "ਕੁਝ": "کجھ", "ਸਭ": "سب", "ਕੋਈ": "کوئی",
    # Vowel clusters whose conventional spelling differs from the mechanical one.
    "ਹੋਇਆ": "ہویا", "ਆਇਆ": "آیا", "ਲਿਆ": "لیا", "ਗਾਇਆ": "گایا",
    "ਪਿਆ": "پیا", "ਦਿਆਂ": "دیاں", "ਜਾਂਦਾ": "جاندا", "ਰਹਿੰਦਾ": "رہندا",
}

# --------------------------------------------------------------------------- #
# script definitions
# --------------------------------------------------------------------------- #

_NUKTA_DEVA = "़"
_NUKTA_GURU = "਼"

# Whisper's output varies in ways that hide a word from the list: it rarely
# writes nuktas, and it spells a nasal either as a diacritic or as a full
# consonant with virama. Normalise both away on each side of the lookup.
_DEVA_CLUSTER_NASAL = re.compile(r"न्(?=[जझचछतथदधटठडढ])|म्(?=[पफबभ])")
_GURU_CLUSTER_NASAL = re.compile(r"ਨ੍(?=[ਜਝਚਛਤਥਦਧਟਠਡਢ])|ਮ੍(?=[ਪਫਬਭ])")

# Precomposed nukta letters have decomposed twins; fold them together so the word
# list only needs one spelling.
_GURU_FOLD = {
    "ਸ਼": "ਸ਼", "ਖ਼": "ਖ਼", "ਗ਼": "ਗ਼", "ਜ਼": "ਜ਼", "ਫ਼": "ਫ਼", "ਲ਼": "ਲ਼",
}


def _normal_deva(word: str) -> str:
    word = word.replace(_NUKTA_DEVA, "").replace("ँ", "ं")
    return _DEVA_CLUSTER_NASAL.sub("ं", word)


def _normal_guru(word: str) -> str:
    for decomposed, precomposed in _GURU_FOLD.items():
        word = word.replace(decomposed, precomposed)
    # The addak marks gemination, which Shahmukhi does not write, and tippi and
    # bindi are the same nasal.
    word = word.replace("ੱ", "").replace("ਂ", "ੰ")
    return _GURU_CLUSTER_NASAL.sub("ੰ", word)


SCRIPTS: dict[str, dict[str, Any]] = {
    "devanagari": {
        "block": re.compile(r"[ऀ-ॿ]"),
        "consonants": DEVA_CONSONANTS,
        "vowels": DEVA_INITIAL_VOWELS,
        "vowels_before_consonant": DEVA_INITIAL_BEFORE_CONSONANT,
        "vowels_after_vowel": DEVA_VOWELS_AFTER_VOWEL,
        "matras": DEVA_MATRAS,
        "short_before_vowel": {"ि": "ی", "ु": "و"},
        # ے and آ are word-final and word-initial letters; inside a word the same
        # sounds are written ی and ا.
        "matras_final": {**DEVA_MATRAS, "े": "ے", "ै": "ے"},
        "nasals": ("ं", "ँ"),
        "virama": "्",
        "visarga": "ः",
        "nukta": _NUKTA_DEVA,
        "future": DEVA_FUTURE,
        "words": {_normal_deva(k): v for k, v in DEVA_WORDS.items()},
        "normalise": _normal_deva,
    },
    "gurmukhi": {
        "block": re.compile(r"[਀-੿]"),
        "consonants": GURU_CONSONANTS,
        "vowels": GURU_INITIAL_VOWELS,
        "vowels_before_consonant": GURU_INITIAL_BEFORE_CONSONANT,
        "vowels_after_vowel": GURU_VOWELS_AFTER_VOWEL,
        "matras": GURU_MATRAS,
        "short_before_vowel": {"ਿ": "ی", "ੁ": "و"},
        "matras_final": {**GURU_MATRAS, "ੇ": "ے", "ੈ": "ے"},
        "nasals": ("ੰ", "ਂ"),
        "virama": "੍",
        "visarga": "ਃ",
        "nukta": _NUKTA_GURU,
        "future": GURU_FUTURE,
        "words": {_normal_guru(k): v for k, v in GURU_WORDS.items()},
        "normalise": _normal_guru,
        # The addak doubles the following consonant. Shahmukhi writes the letter
        # once, so it is simply dropped.
        "drop": "ੱ",
    },
}

# Letters that are vowels in Perso-Arabic script, for deciding whether the next
# vowel needs a hamza seat.
VOWEL_LETTERS = set("اآوىیےؤئ")

# Which language uses which source script.
LANGUAGES = {"hi": "devanagari", "pa": "gurmukhi"}


def supported(language: str | None) -> bool:
    """Whether Urdu/Shahmukhi can be produced for this language."""
    return (language or "").lower() in LANGUAGES


def _script_for(language: str | None) -> dict[str, Any]:
    return SCRIPTS[LANGUAGES.get((language or "hi").lower(), "devanagari")]


def convert_word(word: str, language: str = "hi") -> str:
    """One word into Perso-Arabic script."""
    script = _script_for(language)
    known = script["words"].get(script["normalise"](word))
    if known:
        return known

    for ending, target in script["future"]:
        if word.endswith(ending) and len(word) > len(ending):
            stem = convert_word(word[: -len(ending)], language)
            # ؤ carries the vowel a stem ending in ا would otherwise repeat.
            if target.startswith("ؤ") and not stem.endswith("ا"):
                target = "و" + target[1:]
            return stem + target

    block, consonants = script["block"], script["consonants"]
    out: list[str] = []
    # Where a short vowel was left unwritten, and what it would have been.
    dropped: list[tuple[int, str]] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""

        if nxt == script["nukta"] and (ch + nxt) in consonants:
            out.append(consonants[ch + nxt])
            i += 2
            continue
        if ch in consonants:
            out.append(consonants[ch])
            i += 1
            continue
        if ch in script["vowels"]:
            # Word-initial takes alif; anywhere else the vowel needs a hamza
            # seat, or it merges with the letter before it and the word loses a
            # syllable — गई is گئی, not گی.
            if out:
                out.append(script["vowels_after_vowel"][ch])
            elif nxt in consonants and ch in script["vowels_before_consonant"]:
                out.append(script["vowels_before_consonant"][ch])
            else:
                out.append(script["vowels"][ch])
            i += 1
            continue
        if ch in script["matras"]:
            rest = word[i + 1:]
            for nasal in script["nasals"]:
                rest = rest.replace(nasal, "")
            table = script["matras_final"] if not block.search(rest) else script["matras"]
            written = table[ch]
            # A short vowel is normally left out, but it has to be written when it
            # carries the vowel that follows: हुई is ہوئی, not ہئی.
            if not written and nxt in script["vowels"]:
                written = script["short_before_vowel"].get(ch, "")
            # Also before a nasal, which cannot sit straight after its own
            # consonant: ਜਾਨੁਂ is جانوں, and جانں is not readable as anything.
            elif not written and nxt in script["nasals"]:
                written = script["short_before_vowel"].get(ch, "")
            if not written:
                # Remember where it went, in case the word turns out to have no
                # vowel at all — see below.
                dropped.append((len(out), script["short_before_vowel"].get(ch, "")))
            out.append(written)
            i += 1
            continue
        if ch in script["nasals"]:
            final = not block.search(word[i + 1:])
            out.append("ں" if final else "ن")
            i += 1
            continue
        if ch == script["virama"] or ch == script.get("drop"):
            i += 1                     # a cluster: the letters are simply joined
            continue
        if ch == script["visarga"]:
            out.append("ہ")
            i += 1
            continue

        out.append(ch)
        i += 1

    # Urdu leaves short vowels unwritten, which is correct — ਪਾਸੁ really is پاس.
    # But a word whose only vowel was short is then left as bare consonants, and
    # ਤੁ came out as the single letter ت, which cannot be read as a word at all.
    # Writing the short vowel is the lesser evil in exactly that case.
    if dropped and not any(ch in VOWEL_LETTERS for ch in out):
        for position, replacement in dropped:
            out[position] = replacement

    # A doubled consonant is one letter: लक्खा is لکھا, not لککھا.
    return re.sub(r"(\S)\1", r"\1", "".join(out))


def convert(text: str, language: str = "hi") -> str:
    """Convert only the tokens written in the source script.

    English words inside mixed speech are left exactly as they are, the same rule
    the Latin path follows.
    """
    if not text or not text.strip():
        return text
    block = _script_for(language)["block"]
    return "".join(
        convert_word(part, language) if block.search(part) else part
        for part in re.split(r"(\s+|-)", text)
    )


def convert_lines(lines: list[str], language: str = "hi") -> list[str]:
    return [convert(line, language) for line in lines]
