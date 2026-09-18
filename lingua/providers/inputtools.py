"""Romanized Indic -> native script, via Google Input Tools.

This is the service behind Google's transliteration keyboards. It is keyless
and handles the loose, unstandardised romanisation people actually type:

    mera naam rahul  ->  मेरा नाम राहुल

We use it for two things: showing the user what we understood their Hinglish
to mean, and feeding native-script text into IndicTrans2, which has no
romanized input mode of its own.
"""

from __future__ import annotations

import json

import httpx

from ..languages import Language
from ..types import ProviderError
from .base import Transliterator

_ENDPOINT = "https://inputtools.google.com/request"

# Input Tools is tuned for keyboard-sized input. Long paragraphs come back
# truncated or degraded, so we transliterate a few words at a time.
_WORDS_PER_CALL = 12

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
}

# Languages Input Tools can transliterate into, keyed by our canonical code.
_ITC_SUPPORTED = {
    "hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "ur",
    "ne", "sa", "as", "si", "am", "ar", "fa", "el", "ru", "sr", "ti",
}


def _chunks(words: list[str], size: int) -> list[list[str]]:
    return [words[i:i + size] for i in range(0, len(words), size)]


class InputToolsTransliterator(Transliterator):
    name = "google_inputtools"
    offline = False

    def __init__(self, timeout: float = 12.0):
        self._client = httpx.AsyncClient(timeout=timeout, headers=_HEADERS)

    def supports(self, language: Language) -> bool:
        return language.code in _ITC_SUPPORTED

    async def to_native(self, text: str, language: Language) -> str:
        if not self.supports(language):
            raise ProviderError(self.name,
                                f"no transliteration for {language.code}",
                                permanent=True)

        out: list[str] = []
        for group in _chunks(text.split(), _WORDS_PER_CALL):
            phrase = " ".join(group)
            params = {
                "text": phrase,
                "itc": f"{language.code}-t-i0-und",
                "num": "1", "cp": "0", "cs": "1",
                "ie": "utf-8", "oe": "utf-8", "app": "translation",
            }
            try:
                resp = await self._client.get(_ENDPOINT, params=params)
                resp.raise_for_status()
                data = json.loads(resp.text)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                raise ProviderError(self.name, str(exc)) from exc

            if not isinstance(data, list) or not data or data[0] != "SUCCESS":
                raise ProviderError(self.name, f"service returned {data[:1]}")

            try:
                candidates = data[1][0][1]
                out.append(candidates[0] if candidates else phrase)
            except (IndexError, KeyError, TypeError):
                out.append(phrase)

        return " ".join(out)

    async def close(self) -> None:
        await self._client.aclose()
