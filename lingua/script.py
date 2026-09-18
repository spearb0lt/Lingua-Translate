"""Script detection and romanized-Indic detection.

Two jobs:

1. ``detect_script`` - which writing system is this text in? Pure Unicode
   range lookup, no dependencies.

2. ``detect_romanized`` - the "mera naam rahul" problem. Text is in Latin
   letters but the language is Indic. We score the text against small
   lexicons of very common function words. This is deliberately a cheap
   heuristic: when we are online, Google's detector (which returns hi-Latn)
   is more reliable and the router prefers it. The heuristic exists so the
   fully-offline path still handles romanized input.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Unicode ranges, ordered so the first hit wins.
_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0370, 0x03FF, "Greek"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x1000, 0x109F, "Myanmar"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1C50, 0x1C7F, "OlChiki"),
    (0xABC0, 0xABFF, "MeeteiMayek"),
    (0x3040, 0x309F, "Japanese"),   # hiragana
    (0x30A0, 0x30FF, "Japanese"),   # katakana
    (0xAC00, 0xD7AF, "Hangul"),
    (0x1100, 0x11FF, "Hangul"),
    (0x4E00, 0x9FFF, "Han"),
    (0x3400, 0x4DBF, "Han"),
    (0x0041, 0x024F, "Latin"),
]

# Script -> the language we assume when we see it and nothing else tells us.
SCRIPT_DEFAULT_LANG = {
    "Devanagari": "hi", "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu",
    "Odia": "or", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Sinhala": "si", "Arabic": "ur", "Cyrillic": "ru",
    "Greek": "el", "Hebrew": "he", "Thai": "th", "Myanmar": "my",
    "Khmer": "km", "Ethiopic": "am", "OlChiki": "sat", "MeeteiMayek": "mni",
    "Japanese": "ja", "Hangul": "ko", "Han": "zh", "Latin": "en",
}

_WORD_RE = re.compile(r"[a-z']+")


def detect_script(text: str) -> str:
    """Dominant writing system of ``text``, ignoring digits and punctuation."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for lo, hi, name in _RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            name = unicodedata.name(ch, "").split()[0].title() if ch.isalpha() else ""
            if name:
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Unknown"
    return max(counts, key=counts.get)


def script_ratio(text: str, script: str) -> float:
    """Fraction of alphabetic characters that belong to ``script``."""
    total = hits = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        for lo, hi, name in _RANGES:
            if lo <= cp <= hi:
                if name == script:
                    hits += 1
                break
    return hits / total if total else 0.0


# --- Romanized Indic detection -------------------------------------------
# Very common function words, written the way people actually type them.
# Only words that are rare-to-absent in English earn a point; ambiguous ones
# ("main", "is", "to", "me", "us", "the") are deliberately left out.
_ROMAN_LEXICON: dict[str, set[str]] = {
    "hi": {
        "hai", "hain", "tha", "thi", "the", "hoon", "hu", "hun", "ho", "hota",
        "hoti", "hote", "kya", "kyun", "kyon", "kaise", "kaisa", "kaisi", "kaha",
        "kahan", "kab", "kaun", "kitna", "kitne", "kitni", "nahi", "nahin", "naa",
        "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski", "unka",
        "hamara", "hamari", "tumhara", "aap", "aapka", "aapko", "tum", "tumhe",
        "yeh", "ye", "woh", "wo", "iska", "isko", "usko", "mujhe", "mujhko",
        "humko", "hume", "unko", "unhe", "kisi", "koi", "kuch", "kuchh", "sab",
        "sabhi", "bhi", "aur", "lekin", "par", "magar", "kyunki", "agar", "toh",
        "phir", "abhi", "kabhi", "hamesha", "bahut", "bohot", "thoda", "zyada",
        "accha", "achha", "acha", "theek", "thik", "bura", "naam", "kaam",
        "ghar", "paani", "khana", "dost", "pyaar", "dil", "zindagi", "waqt",
        "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "karna", "karta",
        "karti", "karte", "kiya", "diya", "liya", "chahiye", "sakta", "sakte",
        "sakti", "jaana", "jata", "aata", "aaya", "bola", "kehna", "kaha",
        "saath", "liye", "wala", "wali", "waale", "jaise", "waise", "itna",
        "utna", "yahan", "wahan", "andar", "bahar", "upar", "neeche", "peeche",
        "aage", "paas", "dur", "jaldi", "dheere", "matlab", "shayad", "bilkul",
        "haan", "nai", "arre", "yaar", "bhai", "didi", "beta", "bhaiya",
    },
    "mr": {
        "ahe", "aahe", "nahi", "mala", "tula", "tyala", "majha", "majhi",
        "tujha", "amcha", "tumcha", "kay", "kasa", "kashi", "kuthe", "kevha",
        "ani", "pan", "tar", "mhanun", "khup", "thoda", "changla", "aaplya",
        "karto", "karte", "hoto", "hoti", "zala", "zali", "gela", "geli",
    },
    "bn": {
        "ami", "tumi", "apni", "amar", "tomar", "apnar", "tar", "amader",
        "tomader", "ache", "achi", "acho", "chilo", "hobe", "korbo", "korchi",
        "kore", "kintu", "ebong", "tobe", "kano", "kotha", "kothay", "kemon",
        "bhalo", "kharap", "onek", "ektu", "naam", "bhai", "dada", "didi",
        "keno", "jodi", "tahole", "ekhon", "takhon", "aaj", "kal",
    },
    "ta": {
        "naan", "nee", "neenga", "avan", "aval", "avanga", "enna", "eppadi",
        "enge", "eppo", "yaar", "illai", "illa", "irukku", "irukken", "iruku",
        "vanakkam", "nandri", "romba", "konjam", "nalla", "ketta", "peru",
        "veedu", "thanni", "saapadu", "seri", "aama", "paathu", "ponga",
    },
    "te": {
        "nenu", "nuvvu", "meeru", "atanu", "aame", "emi", "enti", "ela",
        "ekkada", "eppudu", "evaru", "ledu", "undi", "unnanu", "unnaru",
        "chala", "koncham", "manchi", "peru", "illu", "neeru", "tinu",
        "sare", "avunu", "kada", "kani", "mariyu", "endukante",
    },
    "gu": {
        "chhe", "che", "hu", "tu", "tame", "ame", "maru", "taru", "amaru",
        "shu", "kem", "kya", "kyare", "kon", "nathi", "ghanu", "thodu",
        "saru", "kharab", "naam", "ghar", "pani", "khavanu", "bhai",
    },
    "pa": {
        "main", "tusi", "asi", "mera", "tuhada", "sada", "hai", "han",
        "nahi", "kive", "kithe", "kado", "kaun", "bahut", "changa", "mada",
        "naam", "ghar", "pani", "veer", "paaji", "sat", "sri", "akal",
    },
    "kn": {
        "naanu", "neenu", "neevu", "avanu", "avalu", "enu", "hege", "elli",
        "yavaga", "yaaru", "illa", "ide", "iddene", "tumba", "swalpa",
        "chennagide", "hesaru", "mane", "neeru", "oota", "sari", "howdu",
    },
    "ml": {
        "njan", "nee", "ningal", "avan", "aval", "enthu", "engane", "evide",
        "eppol", "aaru", "illa", "undu", "othiri", "kurachu", "nalla",
        "peru", "veedu", "vellam", "bhakshanam", "sheri", "athe", "ente",
    },
    "ur": {
        "hai", "hain", "tha", "thi", "hoon", "kya", "kyun", "kaise", "kahan",
        "nahi", "nahin", "mera", "meri", "tera", "aap", "aapka", "tum",
        "yeh", "woh", "mujhe", "aur", "lekin", "magar", "agar", "phir",
        "bohot", "bahut", "thoda", "acha", "theek", "naam", "kaam", "ghar",
        "shukriya", "salaam", "khuda", "hafiz", "zaroor", "bilkul",
    },
}

# Words in the lexicons that are also ordinary English -- never counted.
_ENGLISH_CLASH = {
    "the", "he", "she", "to", "me", "us", "is", "are", "was", "his", "her",
    "par", "tar", "main", "pan", "han", "sat", "sari", "chill", "ho", "hu",
    "no", "so", "do", "go", "be", "at", "in", "on", "up", "kind", "core",
    "sabhi", "die", "bola", "cost", "kal", "ache", "acho", "tera", "peru",
}


@dataclass
class ScriptInfo:
    script: str                 # dominant writing system
    romanized: bool             # Latin letters, but an Indic language
    guess: str | None           # best-guess language code
    confidence: float           # 0..1, only meaningful when romanized
    latin_ratio: float


def detect_romanized(text: str) -> tuple[str | None, float]:
    """Best-guess Indic language for Latin-script text, with a 0..1 score.

    Returns ``(None, 0.0)`` when the text looks like plain English.
    """
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1]
    if not words:
        return None, 0.0
    scores: dict[str, int] = {}
    for lang, lex in _ROMAN_LEXICON.items():
        hits = sum(1 for w in words if w in lex and w not in _ENGLISH_CLASH)
        if hits:
            scores[lang] = hits
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    # Confidence: what share of the words were recognisably Indic function
    # words. Two hits in a short sentence is already a strong signal.
    ratio = scores[best] / len(words)
    confidence = min(1.0, ratio * 2.0 + (0.2 if scores[best] >= 2 else 0.0))
    return best, round(confidence, 3)


def analyse(text: str, declared: str | None = None) -> ScriptInfo:
    """Full analysis of an input string.

    ``declared`` is the source language the caller asked for, if any. When the
    caller explicitly says "Hindi" but the text is in Latin letters, that is
    itself proof of romanized input and we skip the heuristic.
    """
    script = detect_script(text)
    latin = script_ratio(text, "Latin")

    if script != "Latin":
        # Several languages share most scripts, so seeing the script is not
        # the same as knowing the language. Try to tell them apart, and report
        # honestly how sure we are -- claiming 1.0 here used to make the
        # router skip asking a provider that could actually detect.
        guess, confidence = disambiguate_script(text, script)
        if guess:
            return ScriptInfo(script, False, guess, confidence, latin)
        default = SCRIPT_DEFAULT_LANG.get(script)
        # A script only one language uses is certain; a shared one is a guess.
        sole_user = script not in _NATIVE_LEXICON
        return ScriptInfo(script, False, default, 0.95 if sole_user else 0.4, latin)

    from .languages import get as _get
    declared_lang = _get(declared)
    if declared_lang is not None and declared_lang.code != "en" and declared_lang.xlit:
        # User picked an Indic language and typed Latin -- definitionally romanized.
        return ScriptInfo("Latin", True, declared_lang.code, 1.0, latin)

    guess, confidence = detect_romanized(text)
    if guess and confidence >= 0.25:
        return ScriptInfo("Latin", True, guess, confidence, latin)
    return ScriptInfo("Latin", False, "en", confidence, latin)


# --- output sanity ---------------------------------------------------------
# Scripts that may legitimately appear in text of a given script. Latin is
# allowed everywhere: brand names, acronyms and untranslated proper nouns show
# up in every language. Japanese mixes kana with Han; Korean mixes Hangul with
# Han.
_COMPATIBLE_SCRIPTS: dict[str, set[str]] = {
    "Japanese": {"Japanese", "Han"},
    "Han": {"Han", "Japanese"},
    "Hangul": {"Hangul", "Han"},
}

#: Below this share of alphabetic characters, foreign script is treated as
#: incidental rather than a defect.
_FOREIGN_SCRIPT_TOLERANCE = 0.02


def foreign_script_ratio(text: str, target_script: str) -> tuple[float, str | None]:
    """How much of ``text`` is in a script that does not belong there.

    Returns ``(ratio, worst_script)``. Latin is never counted, because names
    and loanwords legitimately appear in Latin letters in every language.
    """
    allowed = {target_script, "Latin"} | _COMPATIBLE_SCRIPTS.get(target_script, set())
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        for lo, hi, name in _RANGES:
            if lo <= cp <= hi:
                if name not in allowed:
                    counts[name] = counts.get(name, 0) + 1
                break
    if not total or not counts:
        return 0.0, None
    worst = max(counts, key=counts.get)
    return counts[worst] / total, worst


def looks_like_wrong_script(text: str, target_script: str,
                            tolerance: float = _FOREIGN_SCRIPT_TOLERANCE
                            ) -> tuple[bool, str]:
    """True when a translation contains script that cannot belong to it.

    Small models garble proper nouns across scripts -- NLLB-600M rendered
    "Rahul" into Japanese as ``ラहुल``, splicing Devanagari into katakana, even
    when handed clean English input. The output is objectively broken and
    cheap to detect, so the router can fall through to another provider rather
    than serving it.
    """
    if not text.strip():
        return False, ""
    ratio, worst = foreign_script_ratio(text, target_script)
    if worst and ratio > tolerance:
        return True, f"{worst} characters in {target_script} output ({ratio:.0%})"
    return False, ""


# --- disambiguating languages that share a script -------------------------
# Script alone does not identify a language. Devanagari carries Hindi,
# Marathi, Nepali, Sanskrit, Maithili, Bodo, Dogri and Konkani; guessing
# "Hindi" for all of them is not a rounding error, it changes meaning:
# Marathi "माझे नाव राहुल आहे" read as Hindi becomes "My boat is Rahul",
# because नाव is "boat" in Hindi and "name" in Marathi.
#
# These lexicons key on the words that differ most between the languages --
# copulas, pronouns and conjunctions -- because those appear in almost every
# sentence and almost never cross over.
_NATIVE_LEXICON: dict[str, dict[str, set[str]]] = {
    "Devanagari": {
        "hi": {"है", "हैं", "था", "थी", "थे", "हूँ", "हूं", "और", "मेरा", "मेरी",
               "मैं", "नहीं", "यह", "वह", "किया", "हुआ", "करना", "रहा", "रही",
               "को", "का", "की", "के", "से", "में", "पर", "कोई", "कुछ", "बहुत",
               "क्या", "कैसे", "कहाँ", "लेकिन", "अगर", "साथ", "लिए"},
        "mr": {"आहे", "आहेत", "होता", "होती", "होते", "माझे", "माझा", "माझी",
               "तुझे", "आणि", "मी", "तू", "आम्ही", "तुम्ही", "त्याने", "मला",
               "तुला", "त्याला", "नाही", "काय", "कसे", "कुठे", "खूप", "पण",
               "किंवा", "चांगले", "करतो", "करते", "झाले", "यांनी", "आपण"},
        "ne": {"छ", "छन्", "छु", "हो", "हुन्", "थियो", "भयो", "मेरो", "तिम्रो",
               "म", "तपाईं", "हामी", "गर्छ", "गर्ने", "भएको", "होइन", "पनि",
               "लाई", "बाट", "मा", "को", "कि", "तर", "धेरै", "राम्रो", "अनि"},
        "sa": {"अस्ति", "सन्ति", "भवति", "च", "एव", "इति", "तत्", "सः", "सा",
               "मम", "तव", "अहम्", "त्वम्", "गच्छति", "करोति", "किम्", "न",
               "वा", "यत्", "तथा", "अपि", "एतत्"},
        "mai": {"अछि", "छथि", "छै", "हमर", "अहाँ", "गेल", "केलक", "रहल", "भेल",
                "हमरा", "अपन", "कियै", "मुदा"},
        "brx": {"आंनि", "आं", "बे", "बिनि", "दं", "नों", "मोन", "जायो", "थां"},
        "gom": {"आसा", "म्हजे", "म्हजो", "तुजे", "हांव", "आनी", "ना", "कित्ले",
                "बरें", "खंय"},
        "doi": {"ऐ", "तुसां", "असां", "दा", "दी", "दे", "गेया", "कीता", "होंदा"},
    },
    "Bengali": {
        # Assamese is separated below by its own letters; these are the word
        # level backstops.
        "bn": {"আমি", "তুমি", "আপনি", "আমার", "তোমার", "আছে", "ছিল", "হবে",
               "করে", "কিন্তু", "এবং", "কেন", "কোথায়", "ভালো", "অনেক", "নাম"},
        "as": {"মোৰ", "আৰু", "হৈছে", "আছিল", "কৰি", "তেওঁ", "মই", "তুমি",
               "কিন্তু", "ভাল", "নাম", "হয়"},
    },
    "Arabic": {
        "ur": {"ہے", "ہیں", "تھا", "تھی", "اور", "میرا", "میں", "نہیں", "کیا",
               "کہاں", "لیکن", "اگر", "بہت", "ساتھ", "کے", "کی", "کا"},
        "sd": {"آهي", "آهن", "هو", "هئي", "۽", "منهنجو", "مان", "ناهي", "ڇا",
               "پر", "گهڻو", "سان"},
        "ks": {"چھُ", "چھِ", "آسُن", "مےٚ", "تہٕ", "کیا", "ییٚلہ"},
    },
}

# Characters that belong to exactly one language sharing a script. These are
# far stronger evidence than any word list: Assamese writes ৰ and ৱ where
# Bengali writes র and ব, and Sindhi and Kashmiri each have letters Urdu
# never uses.
_SCRIPT_SIGNATURE_CHARS: dict[str, dict[str, str]] = {
    "Bengali": {"\u09f0": "as", "\u09f1": "as"},
    "Arabic": {
        # Sindhi-only letters
        "\u067b": "sd", "\u0680": "sd", "\u067a": "sd", "\u068a": "sd",
        "\u068f": "sd", "\u068c": "sd", "\u068d": "sd", "\u06a6": "sd",
        "\u06aa": "sd", "\u06b3": "sd", "\u06b1": "sd", "\u06bb": "sd",
        # Kashmiri-only vowel signs
        "\u0672": "ks", "\u06c6": "ks", "\u06c4": "ks", "\u0620": "ks",
    },
}


def disambiguate_script(text: str, script: str) -> tuple[str | None, float]:
    """Pick the language within a shared script. Returns (code, confidence).

    Confidence is deliberately honest: a short sentence with one matching
    word is weak evidence, and the router uses that to decide whether to
    trust us or defer to a provider that can detect for itself.
    """
    families = _NATIVE_LEXICON.get(script)
    if not families:
        return None, 0.0

    # Signature characters first -- they are decisive when present.
    signatures = _SCRIPT_SIGNATURE_CHARS.get(script, {})
    if signatures:
        hits: dict[str, int] = {}
        for ch in text:
            lang = signatures.get(ch)
            if lang:
                hits[lang] = hits.get(lang, 0) + 1
        if hits:
            best = max(hits, key=hits.get)
            return best, 0.95

    words = [w for w in re.split(r"[\s।॥,.!?;:()\"']+", text) if w]
    if not words:
        return None, 0.0

    scores: dict[str, int] = {}
    for lang, lexicon in families.items():
        hit = sum(1 for w in words if w in lexicon)
        if hit:
            scores[lang] = hit
    if not scores:
        return None, 0.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_hits = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    # Confidence grows with how many words matched and how clearly the winner
    # beat the next candidate.
    margin = (best_hits - runner_up) / max(best_hits, 1)
    coverage = min(1.0, best_hits / max(len(words) * 0.3, 1))
    confidence = 0.5 + 0.35 * margin + 0.15 * coverage
    return best, round(min(confidence, 0.93), 3)
