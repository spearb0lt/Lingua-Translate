"""Google's keyless web-translate endpoints.

Two independent paths to the same free service, because each fails
differently:

``GoogleFreeProvider`` (gtx)
    The endpoint translate.google.com itself calls. Richest response: it
    reports ``hi-Latn`` for romanized Hindi, which is how we detect that
    "mera naam rahul hai" is Hindi typed in Latin letters. Google blocks
    plain-Python HTTP clients by TLS fingerprint, so this needs ``curl_cffi``
    to present a browser handshake.

``GoogleDictProvider`` (clients5)
    The endpoint the Chrome dictionary extension uses. Plain ``httpx`` works,
    so this survives when ``curl_cffi`` is unavailable. It gives no romanized
    detection and is measurably weaker on long code-mixed input, so it sits
    below the gtx path in the chain.

Both are undocumented endpoints: they can rate-limit by IP and are not
covered by any commercial agreement. Fine for a project demo, which is why
the router prefers local models first and treats these as tier 1.
"""

from __future__ import annotations

import json
import re

import httpx

from ..languages import Language, from_google
from ..types import ProviderError, Tier
from .base import Provider, ProviderOutput

try:  # curl_cffi is optional: without it we fall back to GoogleDictProvider
    from curl_cffi.requests import AsyncSession as _CurlSession
except Exception:  # pragma: no cover - depends on install
    _CurlSession = None

_GTX_HOSTS = [
    "https://translate.googleapis.com/translate_a/single",
    "https://translate.google.com/translate_a/single",
]
_DICT_HOST = "https://clients5.google.com/translate_a/t"

_CHUNK_CHARS = 1500
_SENT_SPLIT = re.compile(r"(?<=[.!?।。])\s+")
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def chunk_text(text: str, limit: int = _CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries, never mid-word, into <= limit pieces."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for sentence in _SENT_SPLIT.split(text):
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            buf = ""
            for word in sentence.split(" "):
                if len(buf) + len(word) + 1 > limit:
                    chunks.append(buf)
                    buf = word
                else:
                    buf = f"{buf} {word}" if buf else word
            current = buf
            continue
        if len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        chunks.append(current)
    return chunks


def _parse_gtx(payload: str) -> tuple[str, str | None, bool]:
    """(translation, detected source, was_romanized) from a gtx response."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderError("google_free", f"unparseable response: {exc}") from exc

    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ProviderError("google_free", "unexpected response shape")

    text = "".join(seg[0] for seg in data[0]
                   if isinstance(seg, list) and seg and isinstance(seg[0], str))
    detected = data[2] if len(data) > 2 and isinstance(data[2], str) else None

    # data[8] holds finer-grained detection including the "-Latn" variants
    # that tell us the input was romanized: [['hi'], None, [1], ['hi-Latn']]
    romanized = False
    if len(data) > 8 and isinstance(data[8], list) and len(data[8]) > 3:
        variants = data[8][3]
        if isinstance(variants, list) and variants and isinstance(variants[0], str):
            if variants[0].endswith("-Latn"):
                detected = variants[0]
                romanized = True
    return text, detected, romanized


def _parse_dict(payload: str) -> tuple[str, str | None]:
    """(translation, detected source) from a clients5 response.

    The shape depends on whether a source language was given:
      sl=auto  -> [["translated text","hi"]]
      sl=as    -> ["translated text"]
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderError("google_dict", f"unparseable response: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise ProviderError("google_dict", "unexpected response shape")

    if isinstance(data[0], str):                       # explicit source form
        return "".join(x for x in data if isinstance(x, str)), None
    if isinstance(data[0], list) and data[0]:          # auto-detect form
        rows = data if isinstance(data[0][0], str) else data[0]
        text = "".join(r[0] for r in rows if isinstance(r, list) and r)
        lang = next((r[1] for r in rows
                     if isinstance(r, list) and len(r) > 1 and isinstance(r[1], str)), None)
        return text, lang
    raise ProviderError("google_dict", "unexpected response shape")


class GoogleFreeProvider(Provider):
    """gtx endpoint via a browser-impersonating TLS handshake."""

    name = "google_free"
    tier = Tier.FREE_WEB
    blurb = "Google web endpoint (keyless), detects romanized Indic"
    offline = False
    detects = True

    def __init__(self, timeout: float = 20.0, impersonate: str = "chrome"):
        self._timeout = timeout
        self._impersonate = impersonate
        self._session = None

    def available(self) -> bool:
        return _CurlSession is not None

    def supports(self, source: Language | None, target: Language) -> bool:
        if target.google is None:
            return False
        return source is None or source.google is not None

    def _get_session(self):
        if self._session is None:
            if _CurlSession is None:
                raise ProviderError(self.name, "curl_cffi not installed",
                                    permanent=True)
            self._session = _CurlSession(impersonate=self._impersonate,
                                         timeout=self._timeout)
        return self._session

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        session = self._get_session()
        base = {"client": "gtx", "sl": source.google if source else "auto",
                "tl": target.google, "dt": "t", "ie": "UTF-8", "oe": "UTF-8"}

        pieces: list[str] = []
        detected: str | None = None
        romanized = False
        last_error: Exception | None = None

        for piece in chunk_text(text):
            served = False
            for host in _GTX_HOSTS:
                try:
                    resp = await session.get(host, params=dict(base, q=piece))
                    if resp.status_code == 429:
                        raise ProviderError(self.name, "rate limited (HTTP 429)")
                    if resp.status_code != 200:
                        raise ProviderError(self.name, f"HTTP {resp.status_code}")
                    out, det, rom = _parse_gtx(resp.text)
                except Exception as exc:  # network, parse or provider error
                    last_error = exc
                    continue
                pieces.append(out)
                detected = detected or det
                romanized = romanized or rom
                served = True
                break
            if not served:
                raise ProviderError(self.name, f"all hosts failed ({last_error})")

        lang, alias_rom = from_google(detected) if detected else (None, False)
        return ProviderOutput(
            text=" ".join(p.strip() for p in pieces).strip(),
            detected_source=lang.code if lang else None,
            romanized=romanized or alias_rom,
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class GoogleDictProvider(Provider):
    """clients5 endpoint, reachable with plain httpx. No romanized detection."""

    name = "google_dict"
    tier = Tier.FREE_WEB
    blurb = "Google dictionary endpoint (keyless), no extra dependencies"
    offline = False
    detects = True

    def __init__(self, timeout: float = 20.0):
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": _UA, "Accept": "*/*"})

    def supports(self, source: Language | None, target: Language) -> bool:
        if target.google is None:
            return False
        return source is None or source.google is not None

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        params_base = {"client": "dict-chrome-ex",
                       "sl": source.google if source else "auto",
                       "tl": target.google}
        pieces: list[str] = []
        detected: str | None = None

        for piece in chunk_text(text):
            try:
                resp = await self._client.get(_DICT_HOST,
                                              params=dict(params_base, q=piece))
                if resp.status_code == 429:
                    raise ProviderError(self.name, "rate limited (HTTP 429)")
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(self.name, str(exc)) from exc
            out, det = _parse_dict(resp.text)
            pieces.append(out)
            detected = detected or det

        lang, rom = from_google(detected) if detected else (None, False)
        return ProviderOutput(
            text=" ".join(p.strip() for p in pieces).strip(),
            detected_source=lang.code if lang else None,
            romanized=rom,
        )

    async def close(self) -> None:
        await self._client.aclose()
