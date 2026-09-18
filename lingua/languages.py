"""Single source of truth for languages and how each provider names them.

Every provider speaks a different dialect of language codes: IndicTrans2 and
NLLB use FLORES-200 tags (hin_Deva), Google uses ISO-639-1 (hi), Whisper uses
its own short list, and Edge TTS uses BCP-47 locales (hi-IN). Keeping the
mapping in one table means the router never has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str            # our canonical id (ISO-639-1 where one exists, else 639-3)
    name: str            # English name
    native: str          # endonym, shown in the picker
    script: str          # default writing system (see lingua.script)
    flores: str | None   # FLORES-200 tag used by IndicTrans2 / NLLB
    google: str | None   # Google Translate code
    whisper: str | None  # Whisper code, None if Whisper cannot transcribe it
    tts_locale: str | None   # BCP-47 locale prefix for Edge TTS voice lookup
    indic: bool = False  # one of the 22 scheduled Indian languages
    it2: bool = False    # translatable by IndicTrans2
    nllb: bool = True    # present in NLLB-200
    xlit: bool = False   # romanized input is common enough to auto-detect

    @property
    def label(self) -> str:
        if self.native == self.name:
            return self.name
        return f"{self.name} ({self.native})"


def _ind(code, name, native, script, flores, google, whisper, tts, xlit=False):
    return Language(code, name, native, script, flores, google, whisper, tts,
                    indic=True, it2=True, nllb=True, xlit=xlit)


# --- English + the 22 scheduled Indian languages (IndicTrans2 territory) ---
# xlit=True marks the ones where romanized input ("mera naam rahul") is common
# enough that we actively detect it and offer a transliteration.
_INDIAN: list[Language] = [
    Language("en", "English", "English", "Latin", "eng_Latn", "en", "en", "en-IN",
             indic=False, it2=True, nllb=True),
    _ind("as",  "Assamese",  "অসমীয়া",      "Bengali",     "asm_Beng", "as",       "as",  None,    True),
    _ind("bn",  "Bengali",   "বাংলা",        "Bengali",     "ben_Beng", "bn",       "bn",  "bn-IN", True),
    _ind("brx", "Bodo",      "बड़ो",          "Devanagari",  "brx_Deva", "brx",      None,  None),
    _ind("doi", "Dogri",     "डोगरी",        "Devanagari",  "doi_Deva", "doi",      None,  None),
    _ind("gom", "Konkani",   "कोंकणी",       "Devanagari",  "gom_Deva", "gom",      None,  None),
    _ind("gu",  "Gujarati",  "ગુજરાતી",       "Gujarati",    "guj_Gujr", "gu",       "gu",  "gu-IN", True),
    _ind("hi",  "Hindi",     "हिन्दी",        "Devanagari",  "hin_Deva", "hi",       "hi",  "hi-IN", True),
    _ind("kn",  "Kannada",   "ಕನ್ನಡ",        "Kannada",     "kan_Knda", "kn",       "kn",  "kn-IN", True),
    _ind("ks",  "Kashmiri",  "کٲشُر",         "Arabic",      "kas_Arab", "ks",       None,  None),
    _ind("mai", "Maithili",  "मैथिली",        "Devanagari",  "mai_Deva", "mai",      None,  None),
    _ind("ml",  "Malayalam", "മലയാളം",       "Malayalam",   "mal_Mlym", "ml",       "ml",  "ml-IN", True),
    _ind("mni", "Manipuri",  "ꯃꯤꯇꯩꯂꯣꯟ",      "MeeteiMayek", "mni_Mtei", "mni-Mtei", None,  None),
    _ind("mr",  "Marathi",   "मराठी",         "Devanagari",  "mar_Deva", "mr",       "mr",  "mr-IN", True),
    _ind("ne",  "Nepali",    "नेपाली",        "Devanagari",  "npi_Deva", "ne",       "ne",  "ne-NP", True),
    _ind("or",  "Odia",      "ଓଡ଼ିଆ",         "Odia",        "ory_Orya", "or",       None,  None,    True),
    # No Punjabi voice exists on the Edge endpoint (checked against the live
    # list: the only South Asian locales are bn, gu, hi, kn, ml, mr, ne, si,
    # ta, te, ur and en-IN). Claiming "pa-IN" here made the UI advertise
    # text-to-speech that would always fail, so it is None until a voice ships.
    _ind("pa",  "Punjabi",   "ਪੰਜਾਬੀ",        "Gurmukhi",    "pan_Guru", "pa",       "pa",  None,    True),
    _ind("sa",  "Sanskrit",  "संस्कृतम्",      "Devanagari",  "san_Deva", "sa",       "sa",  None,    True),
    _ind("sat", "Santali",   "ᱥᱟᱱᱛᱟᱲᱤ",       "OlChiki",     "sat_Olck", "sat",      None,  None),
    _ind("sd",  "Sindhi",    "سنڌي",          "Arabic",      "snd_Arab", "sd",       "sd",  None),
    _ind("ta",  "Tamil",     "தமிழ்",         "Tamil",       "tam_Taml", "ta",       "ta",  "ta-IN", True),
    _ind("te",  "Telugu",    "తెలుగు",        "Telugu",      "tel_Telu", "te",       "te",  "te-IN", True),
    _ind("ur",  "Urdu",      "اردو",          "Arabic",      "urd_Arab", "ur",       "ur",  "ur-IN", True),
]

# --- Everything else: NLLB-200 locally, Google / free APIs online ---
_FOREIGN: list[Language] = [
    Language("ar",    "Arabic",     "العربية",           "Arabic",   "arb_Arab", "ar",    "ar", "ar-SA"),
    Language("zh",    "Chinese",    "简体中文",            "Han",      "zho_Hans", "zh-CN", "zh", "zh-CN"),
    Language("zh-TW", "Chinese (Traditional)", "繁體中文", "Han",      "zho_Hant", "zh-TW", "zh", "zh-TW"),
    Language("ja",    "Japanese",   "日本語",              "Japanese", "jpn_Jpan", "ja",    "ja", "ja-JP"),
    Language("ko",    "Korean",     "한국어",              "Hangul",   "kor_Hang", "ko",    "ko", "ko-KR"),
    Language("es",    "Spanish",    "Español",            "Latin",    "spa_Latn", "es",    "es", "es-ES"),
    Language("fr",    "French",     "Français",           "Latin",    "fra_Latn", "fr",    "fr", "fr-FR"),
    Language("de",    "German",     "Deutsch",            "Latin",    "deu_Latn", "de",    "de", "de-DE"),
    Language("pt",    "Portuguese", "Português",          "Latin",    "por_Latn", "pt",    "pt", "pt-BR"),
    Language("ru",    "Russian",    "Русский",            "Cyrillic", "rus_Cyrl", "ru",    "ru", "ru-RU"),
    Language("it",    "Italian",    "Italiano",           "Latin",    "ita_Latn", "it",    "it", "it-IT"),
    Language("nl",    "Dutch",      "Nederlands",         "Latin",    "nld_Latn", "nl",    "nl", "nl-NL"),
    Language("tr",    "Turkish",    "Türkçe",             "Latin",    "tur_Latn", "tr",    "tr", "tr-TR"),
    Language("vi",    "Vietnamese", "Tiếng Việt",         "Latin",    "vie_Latn", "vi",    "vi", "vi-VN"),
    Language("th",    "Thai",       "ไทย",                "Thai",     "tha_Thai", "th",    "th", "th-TH"),
    Language("id",    "Indonesian", "Bahasa Indonesia",   "Latin",    "ind_Latn", "id",    "id", "id-ID"),
    Language("ms",    "Malay",      "Bahasa Melayu",      "Latin",    "zsm_Latn", "ms",    "ms", "ms-MY"),
    Language("pl",    "Polish",     "Polski",             "Latin",    "pol_Latn", "pl",    "pl", "pl-PL"),
    Language("uk",    "Ukrainian",  "Українська",         "Cyrillic", "ukr_Cyrl", "uk",    "uk", "uk-UA"),
    Language("fa",    "Persian",    "فارسی",              "Arabic",   "pes_Arab", "fa",    "fa", "fa-IR"),
    Language("he",    "Hebrew",     "עברית",              "Hebrew",   "heb_Hebr", "iw",    "he", "he-IL"),
    Language("el",    "Greek",      "Ελληνικά",           "Greek",    "ell_Grek", "el",    "el", "el-GR"),
    Language("sv",    "Swedish",    "Svenska",            "Latin",    "swe_Latn", "sv",    "sv", "sv-SE"),
    Language("ro",    "Romanian",   "Română",             "Latin",    "ron_Latn", "ro",    "ro", "ro-RO"),
    Language("hu",    "Hungarian",  "Magyar",             "Latin",    "hun_Latn", "hu",    "hu", "hu-HU"),
    Language("cs",    "Czech",      "Čeština",            "Latin",    "ces_Latn", "cs",    "cs", "cs-CZ"),
    Language("da",    "Danish",     "Dansk",              "Latin",    "dan_Latn", "da",    "da", "da-DK"),
    Language("fi",    "Finnish",    "Suomi",              "Latin",    "fin_Latn", "fi",    "fi", "fi-FI"),
    Language("no",    "Norwegian",  "Norsk",              "Latin",    "nob_Latn", "no",    "no", "nb-NO"),
    Language("tl",    "Filipino",   "Filipino",           "Latin",    "tgl_Latn", "tl",    "tl", "fil-PH"),
    Language("my",    "Burmese",    "မြန်မာ",              "Myanmar",  "mya_Mymr", "my",    "my", "my-MM"),
    Language("km",    "Khmer",      "ខ្មែរ",                "Khmer",    "khm_Khmr", "km",    "km", "km-KH"),
    Language("si",    "Sinhala",    "සිංහල",              "Sinhala",  "sin_Sinh", "si",    "si", "si-LK"),
    Language("am",    "Amharic",    "አማርኛ",               "Ethiopic", "amh_Ethi", "am",    "am", "am-ET"),
    Language("sw",    "Swahili",    "Kiswahili",          "Latin",    "swh_Latn", "sw",    "sw", "sw-KE"),
    Language("ha",    "Hausa",      "Hausa",              "Latin",    "hau_Latn", "ha",    "ha", None),
    Language("yo",    "Yoruba",     "Yorùbá",             "Latin",    "yor_Latn", "yo",    "yo", None),
    Language("zu",    "Zulu",       "isiZulu",            "Latin",    "zul_Latn", "zu",    None, "zu-ZA"),
]

LANGUAGES: list[Language] = _INDIAN + _FOREIGN
BY_CODE: dict[str, Language] = {l.code: l for l in LANGUAGES}
_BY_GOOGLE: dict[str, Language] = {l.google: l for l in LANGUAGES if l.google}
_BY_FLORES: dict[str, Language] = {l.flores: l for l in LANGUAGES if l.flores}

AUTO = "auto"

# Google returns these for text written in Latin letters that is really an
# Indian language -- exactly the "mera naam rahul" case.
_GOOGLE_ROMAN_ALIASES = {
    "hi-Latn": "hi", "bn-Latn": "bn", "ta-Latn": "ta", "te-Latn": "te",
    "mr-Latn": "mr", "gu-Latn": "gu", "kn-Latn": "kn", "ml-Latn": "ml",
    "pa-Latn": "pa", "ur-Latn": "ur", "ne-Latn": "ne", "sa-Latn": "sa",
}


def get(code: str | None) -> Language | None:
    """Resolve any code we might be handed to a Language."""
    if not code or code == AUTO:
        return None
    if code in BY_CODE:
        return BY_CODE[code]
    if code in _GOOGLE_ROMAN_ALIASES:
        return BY_CODE[_GOOGLE_ROMAN_ALIASES[code]]
    if code in _BY_FLORES:
        return _BY_FLORES[code]
    if code in _BY_GOOGLE:
        return _BY_GOOGLE[code]
    base = code.replace("_", "-").split("-")[0].lower()
    return BY_CODE.get(base) or _BY_GOOGLE.get(base)


def require(code: str) -> Language:
    lang = get(code)
    if lang is None:
        raise KeyError(f"unknown language code: {code!r}")
    return lang


def from_google(code: str) -> tuple[Language | None, bool]:
    """Map a Google detection result to (language, was_romanized)."""
    if code in _GOOGLE_ROMAN_ALIASES:
        return BY_CODE[_GOOGLE_ROMAN_ALIASES[code]], True
    return get(code), False


def indic_pair(src: Language, tgt: Language) -> bool:
    """True when IndicTrans2 can handle this pair directly."""
    return src.it2 and tgt.it2 and (src.indic or tgt.indic) and src.code != tgt.code


def picker() -> list[dict]:
    """Payload for the language dropdowns, Indian languages grouped first."""
    def row(l: Language) -> dict:
        return {
            "code": l.code, "name": l.name, "native": l.native, "label": l.label,
            "script": l.script, "indic": l.indic, "xlit": l.xlit,
            "asr": l.whisper is not None, "tts": l.tts_locale is not None,
        }
    return [
        {"group": "Indian languages", "items": [row(l) for l in _INDIAN]},
        {"group": "Other languages", "items": [row(l) for l in _FOREIGN]},
    ]
