"""Free translation APIs that need no credit card.

These sit below the Google web endpoints because they are slower and weaker,
but they are documented services with published terms, so they are the right
thing to fall back to when the keyless endpoints rate-limit.

``MyMemoryProvider``
    Free translation memory. ~5k words/day anonymously, ~50k/day if you set
    ``MYMEMORY_EMAIL``. Documented and intended for this use.

``LibreTranslateProvider``
    Fully open-source (AGPL) and self-hostable. Public instances come and go
    and mostly require keys now, so this is off unless you point
    ``LIBRETRANSLATE_URL`` at an instance -- including one you run yourself,
    which makes it a genuinely free, unlimited, self-owned fallback.
"""

from __future__ import annotations

import os

import httpx

from ..languages import Language, get as get_lang
from ..types import ProviderError, Tier
from .base import Provider, ProviderOutput
from .google_free import chunk_text

# MyMemory rejects very long q values.
_MM_CHUNK = 480


class MyMemoryProvider(Provider):
    name = "mymemory"
    tier = Tier.FREE_API
    blurb = "MyMemory translation memory, free tier"
    offline = False

    def __init__(self, timeout: float = 20.0, email: str | None = None):
        self._client = httpx.AsyncClient(timeout=timeout)
        self._email = email or os.getenv("MYMEMORY_EMAIL") or None

    def supports(self, source: Language | None, target: Language) -> bool:
        # MyMemory needs an explicit pair; it has no auto-detect.
        return source is not None and source.google is not None \
            and target.google is not None and source.code != target.code

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        if source is None:
            raise ProviderError(self.name, "needs an explicit source language",
                                permanent=True)
        pieces: list[str] = []
        for piece in chunk_text(text, _MM_CHUNK):
            params = {"q": piece, "langpair": f"{source.google}|{target.google}"}
            if self._email:
                params["de"] = self._email
            try:
                resp = await self._client.get(
                    "https://api.mymemory.translated.net/get", params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise ProviderError(self.name, str(exc)) from exc

            status = data.get("responseStatus")
            if status not in (200, "200"):
                raise ProviderError(
                    self.name, f"status {status}: {data.get('responseDetails')}")
            translated = (data.get("responseData") or {}).get("translatedText")
            if not translated:
                raise ProviderError(self.name, "empty translation")
            pieces.append(translated)

        return ProviderOutput(text=" ".join(pieces).strip(),
                              detected_source=source.code)

    async def close(self) -> None:
        await self._client.aclose()


class LibreTranslateProvider(Provider):
    name = "libretranslate"
    tier = Tier.FREE_API
    blurb = "LibreTranslate (AGPL, self-hostable)"
    offline = False
    detects = True

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 timeout: float = 30.0):
        self._url = (url or os.getenv("LIBRETRANSLATE_URL") or "").rstrip("/")
        self._key = api_key or os.getenv("LIBRETRANSLATE_API_KEY") or None
        self._client = httpx.AsyncClient(timeout=timeout)

    def available(self) -> bool:
        return bool(self._url)

    def supports(self, source: Language | None, target: Language) -> bool:
        return bool(self._url) and target.google is not None

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        payload = {
            "q": text,
            "source": source.google if source else "auto",
            "target": target.google,
            "format": "text",
        }
        if self._key:
            payload["api_key"] = self._key
        try:
            resp = await self._client.post(f"{self._url}/translate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(self.name, str(exc)) from exc

        translated = data.get("translatedText")
        if not translated:
            raise ProviderError(self.name, f"no translation in response: {data}")

        detected = None
        det = data.get("detectedLanguage")
        if isinstance(det, dict):
            lang = get_lang(det.get("language"))
            detected = lang.code if lang else None

        return ProviderOutput(text=translated,
                              detected_source=detected or (source.code if source else None))

    async def close(self) -> None:
        await self._client.aclose()
