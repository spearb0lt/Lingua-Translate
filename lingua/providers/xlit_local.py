"""Offline romanized-Indic -> native script.

Google Input Tools (see ``inputtools.py``) is much better at this and is what
we use whenever there is a network. This module exists so the fully-offline
path still does something sensible with "mera naam rahul".

Why it is built the way it is: rule-based transliteration libraries expect a
strict scheme (ITRANS, HK, IAST) where vowel length is written out. People do
not type that way -- they write "mera", not "meraa", so plain ITRANS gives
मेर instead of मेरा. Two layers fix most of it:

1. A curated lexicon of the most common words, which in casual Hindi text are
   overwhelmingly function words (hai, mera, aur, nahi, kya...). These are
   exactly the words rules get wrong and exactly the words that recur most.
2. Rule-based transliteration plus cleanup for everything else, mainly
   stripping the trailing halant that ITRANS leaves on consonant-final words
   (रहुल् -> रहुल).

The result is approximate, and ``LocalTransliterator.approximate`` is True so
callers can say so in the UI rather than pretending otherwise.
"""

from __future__ import annotations

import re

from ..languages import Language
from ..types import ProviderError
from .base import Transliterator

# Common Hindi/Hinglish words -> Devanagari. Function words first, because
# they carry the grammar and are what rule-based conversion mangles.
HINGLISH: dict[str, str] = {
    # to be / auxiliaries
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी", "the": "थे",
    "hoon": "हूँ", "hun": "हूँ", "hu": "हूँ", "ho": "हो", "hota": "होता",
    "hoti": "होती", "hote": "होते", "hoga": "होगा", "hogi": "होगी",
    # question words
    "kya": "क्या", "kyun": "क्यों", "kyon": "क्यों", "kaise": "कैसे",
    "kaisa": "कैसा", "kaisi": "कैसी", "kahan": "कहाँ", "kaha": "कहा",
    "kab": "कब", "kaun": "कौन", "kitna": "कितना", "kitne": "कितने",
    "kitni": "कितनी", "kisko": "किसको", "kiska": "किसका",
    # negation
    "nahi": "नहीं", "nahin": "नहीं", "na": "ना", "mat": "मत", "bina": "बिना",
    # pronouns and possessives
    "main": "मैं", "mai": "मैं", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "tera": "तेरा", "teri": "तेरी", "tere": "तेरे", "tu": "तू", "tum": "तुम",
    "tumhe": "तुम्हें", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी",
    "aap": "आप", "aapka": "आपका", "aapki": "आपकी", "aapko": "आपको",
    "hum": "हम", "humko": "हमको", "hume": "हमें", "hamara": "हमारा",
    "hamari": "हमारी", "uska": "उसका", "uski": "उसकी", "usko": "उसको",
    "unka": "उनका", "unki": "उनकी", "unko": "उनको", "unhe": "उन्हें",
    "mujhe": "मुझे", "mujhko": "मुझको", "iska": "इसका", "iski": "इसकी",
    "isko": "इसको", "yeh": "यह", "ye": "ये", "woh": "वह", "wo": "वो",
    "koi": "कोई", "kisi": "किसी", "kuch": "कुछ", "kuchh": "कुछ",
    "sab": "सब", "sabhi": "सभी", "apna": "अपना", "apni": "अपनी",
    "apne": "अपने", "khud": "खुद",
    # postpositions and particles
    "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "ne": "ने",
    "mein": "में", "me": "में", "par": "पर", "pe": "पे", "tak": "तक",
    "hi": "ही", "bhi": "भी", "aur": "और", "ya": "या", "lekin": "लेकिन",
    "magar": "मगर", "kyunki": "क्योंकि", "agar": "अगर", "toh": "तो",
    "to": "तो", "phir": "फिर", "jab": "जब", "tab": "तब", "jo": "जो",
    "liye": "लिए", "saath": "साथ", "sath": "साथ", "wala": "वाला",
    "wali": "वाली", "waale": "वाले", "jaise": "जैसे", "waise": "वैसे",
    "isliye": "इसलिए", "matlab": "मतलब", "yani": "यानी",
    # time and place
    "abhi": "अभी", "kabhi": "कभी", "hamesha": "हमेशा", "aaj": "आज",
    "kal": "कल", "parso": "परसों", "subah": "सुबह", "shaam": "शाम",
    "raat": "रात", "din": "दिन", "saal": "साल", "mahina": "महीना",
    "hafta": "हफ़्ता", "samay": "समय", "waqt": "वक़्त", "yahan": "यहाँ",
    "wahan": "वहाँ", "andar": "अंदर", "bahar": "बाहर", "upar": "ऊपर",
    "neeche": "नीचे", "peeche": "पीछे", "aage": "आगे", "paas": "पास",
    "dur": "दूर", "har": "हर",
    # quantity and quality
    "bahut": "बहुत", "bohot": "बहुत", "thoda": "थोड़ा", "zyada": "ज़्यादा",
    "accha": "अच्छा", "achha": "अच्छा", "acha": "अच्छा", "achcha": "अच्छा",
    "theek": "ठीक", "thik": "ठीक", "bura": "बुरा", "sahi": "सही",
    "galat": "ग़लत", "naya": "नया", "purana": "पुराना", "bada": "बड़ा",
    "chota": "छोटा", "jaldi": "जल्दी", "dheere": "धीरे", "bilkul": "बिल्कुल",
    "shayad": "शायद", "itna": "इतना", "utna": "उतना",
    # very common verbs
    "raha": "रहा", "rahi": "रही", "rahe": "रहे", "gaya": "गया",
    "gayi": "गई", "gaye": "गए", "karna": "करना", "karta": "करता",
    "karti": "करती", "karte": "करते", "karo": "करो", "kiya": "किया",
    "diya": "दिया", "dena": "देना", "liya": "लिया", "lena": "लेना",
    "chahiye": "चाहिए", "sakta": "सकता", "sakte": "सकते", "sakti": "सकती",
    "jaana": "जाना", "jana": "जाना", "jata": "जाता", "jaunga": "जाऊँगा",
    "aata": "आता", "aaya": "आया", "aana": "आना", "bola": "बोला",
    "bolna": "बोलना", "kehna": "कहना", "dekha": "देखा", "dekhna": "देखना",
    "sunna": "सुनना", "likhna": "लिखना", "padhna": "पढ़ना", "samajh": "समझ",
    "mila": "मिला", "milna": "मिलना", "peeta": "पीता", "khana": "खाना",
    # everyday nouns
    "naam": "नाम", "kaam": "काम", "ghar": "घर", "paani": "पानी",
    "pani": "पानी", "chai": "चाय", "dost": "दोस्त", "pyaar": "प्यार",
    "pyar": "प्यार", "mohabbat": "मोहब्बत", "dil": "दिल",
    "zindagi": "ज़िंदगी", "duniya": "दुनिया", "baat": "बात",
    "kitaab": "किताब", "kitaabein": "किताबें", "pasand": "पसंद",
    "mausam": "मौसम", "school": "स्कूल", "college": "कॉलेज",
    "khush": "खुश", "paisa": "पैसा", "log": "लोग", "aadmi": "आदमी",
    # interjections and address
    "haan": "हाँ", "han": "हाँ", "arre": "अरे", "yaar": "यार",
    "bhai": "भाई", "bhaiya": "भैया", "didi": "दीदी", "beta": "बेटा",
    "ji": "जी", "namaste": "नमस्ते", "shukriya": "शुक्रिया",
    "dhanyavaad": "धन्यवाद",
    # place names that come up constantly
    "dilli": "दिल्ली", "delhi": "दिल्ली", "mumbai": "मुंबई",
    "bharat": "भारत", "hindustan": "हिंदुस्तान",
}

# Scheme conversion targets per language, for the rule-based fallback.
_SANSCRIPT_TARGET = {
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "sa": "devanagari", "brx": "devanagari", "doi": "devanagari",
    "gom": "devanagari", "mai": "devanagari",
    "bn": "bengali", "as": "bengali", "gu": "gujarati", "pa": "gurmukhi",
    "or": "oriya", "ta": "tamil", "te": "telugu", "kn": "kannada",
    "ml": "malayalam",
}

_TOKEN = re.compile(r"(\w+|\W+)", re.UNICODE)
_TRAILING_HALANT = re.compile("्$|্$|્$|੍$|୍$|"
                              "்$|్$|್$|്$")


def _strip_halant(word: str) -> str:
    """ITRANS leaves a halant on consonant-final words; real spelling drops it."""
    return _TRAILING_HALANT.sub("", word)


class LocalTransliterator(Transliterator):
    """Offline, approximate. Prefer InputToolsTransliterator when online."""

    name = "local_xlit"
    offline = True
    approximate = True

    def available(self) -> bool:
        try:
            import indic_transliteration  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, language: Language) -> bool:
        return language.code in _SANSCRIPT_TARGET

    async def to_native(self, text: str, language: Language) -> str:
        if not self.supports(language):
            raise ProviderError(self.name,
                                f"no offline transliteration for {language.code}",
                                permanent=True)
        if not self.available():
            raise ProviderError(self.name,
                                "indic-transliteration not installed",
                                permanent=True)

        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        target = getattr(sanscript, _SANSCRIPT_TARGET[language.code].upper())
        use_lexicon = _SANSCRIPT_TARGET[language.code] == "devanagari"

        out: list[str] = []
        for token in _TOKEN.findall(text):
            if not token.strip() or not token.isalnum():
                out.append(token)
                continue
            lower = token.lower()
            if use_lexicon and lower in HINGLISH:
                out.append(HINGLISH[lower])
                continue
            try:
                converted = transliterate(lower, sanscript.ITRANS, target)
            except Exception:
                out.append(token)
                continue
            out.append(_strip_halant(converted))
        return "".join(out)
