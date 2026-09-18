"""Text-to-speech engines, tried in order by the chain in ``tts.py``.

Three engines, because no single one covers the 22 scheduled languages:

``EdgeEngine``   keyless Microsoft endpoint. Best quality by a wide margin
                 (neural voices), but only 11 of our 23 Indian languages, and
                 it needs a network.
``GoogleEngine`` keyless Google translate_tts. Adds Punjabi over Edge and
                 nothing else, but is a useful second network path.
``MMSEngine``    Meta MMS-TTS, VITS, 36M parameters per language (~150 MB).
                 Runs **offline** on CPU, and is the only thing that speaks
                 Assamese, Odia and Maithili at all -- plus Hausa and Yoruba,
                 the two foreign languages Edge misses.

Ordering note: the translation ladder puts local first, for cost and privacy.
Speech is deliberately the other way round by default, because MMS is audibly
more robotic than an Edge neural voice and a demo lives or dies on that.
``LINGUA_TTS_ORDER`` changes it, and asking for the LOCAL tier still forces
MMS, so offline behaviour stays available.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import urllib.parse
import wave

import httpx

from lingua.languages import Language, get as get_lang

log = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_VOICE_LIST_TTL = 60 * 60 * 6


class VoiceResult:
    __slots__ = ("audio", "mime", "voice", "engine", "offline")

    def __init__(self, audio: bytes, mime: str, voice: str, engine: str,
                 offline: bool):
        self.audio = audio
        self.mime = mime
        self.voice = voice
        self.engine = engine
        self.offline = offline


class TTSEngine:
    name = "abstract"
    offline = False
    quality = 0          # rough ranking, higher is better; used for ordering

    def available(self) -> bool:
        return True

    async def supports(self, language: Language) -> bool:
        return False

    async def voices_for(self, language: Language) -> list[dict]:
        return []

    async def synthesize(self, text: str, language: Language, *,
                         voice: str | None = None,
                         gender: str | None = None,
                         rate: str | None = None) -> VoiceResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass


# --------------------------------------------------------------------------
class EdgeEngine(TTSEngine):
    """Microsoft Edge read-aloud voices. Keyless, online, best quality."""

    name = "edge"
    offline = False
    quality = 100

    def __init__(self) -> None:
        self._voices: list[dict] | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return False
        return True

    async def all_voices(self, refresh: bool = False) -> list[dict]:
        if not self.available():
            return []
        fresh = time.time() - self._fetched_at < _VOICE_LIST_TTL
        if self._voices is not None and fresh and not refresh:
            return self._voices
        async with self._lock:
            if self._voices is not None and fresh and not refresh:
                return self._voices
            import edge_tts
            try:
                raw = await edge_tts.list_voices()
            except Exception as exc:
                log.warning("edge voice list failed: %s", exc)
                return self._voices or []
            self._voices = [{"name": v["ShortName"], "locale": v["Locale"],
                             "gender": v.get("Gender", ""),
                             "label": v.get("FriendlyName", v["ShortName"]),
                             "engine": self.name}
                            for v in raw]
            self._fetched_at = time.time()
            log.info("cached %d Edge voices", len(self._voices))
            return self._voices

    async def voices_for(self, language: Language) -> list[dict]:
        if not language.tts_locale:
            return []
        everything = await self.all_voices()
        prefix = language.tts_locale.lower()
        base = prefix.split("-")[0]
        exact = [v for v in everything if v["locale"].lower() == prefix]
        kin = [v for v in everything
               if v["locale"].lower().startswith(base + "-") and v not in exact]
        return exact + kin

    async def supports(self, language: Language) -> bool:
        return bool(await self.voices_for(language))

    async def synthesize(self, text, language, *, voice=None, gender=None,
                         rate=None) -> VoiceResult:
        import edge_tts

        chosen = voice
        if not chosen:
            options = await self.voices_for(language)
            if not options:
                raise LookupError(f"no Edge voice for {language.code}")
            chosen = options[0]["name"]
            if gender:
                want = gender.strip().lower()
                chosen = next((v["name"] for v in options
                               if v["gender"].lower() == want), chosen)

        kwargs = {"rate": rate} if rate else {}
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, chosen, **kwargs).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        if not audio:
            raise RuntimeError(f"Edge voice {chosen} returned no audio")
        return VoiceResult(bytes(audio), "audio/mpeg", chosen, self.name, False)


# --------------------------------------------------------------------------
class GoogleEngine(TTSEngine):
    """Google translate_tts. Keyless, online. Adds Punjabi over Edge."""

    name = "google"
    offline = False
    quality = 70

    # Checked against the live endpoint: every other language it serves is
    # already covered by Edge, so this is here for Punjabi and for redundancy.
    _ENDPOINT = "https://translate.google.com/translate_tts"
    _MAX_CHARS = 190           # the endpoint truncates long q values

    def __init__(self, timeout: float = 20.0):
        self._client = httpx.AsyncClient(timeout=timeout,
                                         headers={"User-Agent": _UA})
        self._checked: dict[str, bool] = {}

    async def supports(self, language: Language) -> bool:
        if not language.google:
            return False
        if language.code in self._checked:
            return self._checked[language.code]
        try:
            result = await self._fetch("test", language.google)
            ok = bool(result)
        except Exception:
            ok = False
        self._checked[language.code] = ok
        return ok

    async def voices_for(self, language: Language) -> list[dict]:
        if not await self.supports(language):
            return []
        return [{"name": f"google-{language.google}", "locale": language.google,
                 "gender": "", "label": f"Google {language.name}",
                 "engine": self.name}]

    async def _fetch(self, text: str, code: str) -> bytes:
        params = {"ie": "UTF-8", "q": text, "tl": code, "client": "tw-ob"}
        url = f"{self._ENDPOINT}?{urllib.parse.urlencode(params)}"
        resp = await self._client.get(url)
        if resp.status_code != 200 or not resp.content.startswith(b"\xff"):
            raise RuntimeError(f"google tts HTTP {resp.status_code}")
        return resp.content

    async def synthesize(self, text, language, *, voice=None, gender=None,
                         rate=None) -> VoiceResult:
        if not language.google:
            raise LookupError(f"no Google TTS for {language.code}")
        # Long text has to be sent in pieces and concatenated; MP3 frames
        # concatenate cleanly so no re-encoding is needed.
        pieces: list[bytes] = []
        for part in _split_for_tts(text, self._MAX_CHARS):
            pieces.append(await self._fetch(part, language.google))
        if not pieces:
            raise RuntimeError("google tts returned nothing")
        return VoiceResult(b"".join(pieces), "audio/mpeg",
                           f"google-{language.google}", self.name, False)

    async def close(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------------
# Meta MMS-TTS. Verified present for these; the rest 404 on the Hub.
MMS_REPOS = {
    "en": "eng", "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin",
    "kn": "kan", "mai": "mai", "ml": "mal", "mr": "mar", "or": "ory",
    "pa": "pan", "ta": "tam", "te": "tel",
    "es": "spa", "fr": "fra", "de": "deu", "pt": "por", "ru": "rus",
    "ar": "ara", "sw": "swh", "ha": "hau", "yo": "yor", "th": "tha",
    "vi": "vie", "id": "ind", "tr": "tur",
}


class MMSEngine(TTSEngine):
    """Meta MMS-TTS (VITS). Offline, tiny, and the only voice for some languages.

    36M parameters per language, about 150 MB on disk, so we keep a couple
    resident and evict the rest rather than holding 22 of them.
    """

    name = "mms"
    offline = True
    quality = 40

    def __init__(self, capacity: int | None = None):
        from lingua.providers._local_util import ModelCache
        self._cache = ModelCache(capacity or int(os.getenv("LINGUA_TTS_CACHE", "2")))
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import VitsModel  # noqa: F401
        except ImportError:
            return False
        return True

    async def supports(self, language: Language) -> bool:
        return self.available() and language.code in MMS_REPOS

    async def voices_for(self, language: Language) -> list[dict]:
        if not await self.supports(language):
            return []
        return [{"name": f"mms-{MMS_REPOS[language.code]}",
                 "locale": language.code, "gender": "",
                 "label": f"MMS {language.name} (offline)", "engine": self.name}]

    def _repo(self, language: Language) -> str:
        return f"facebook/mms-tts-{MMS_REPOS[language.code]}"

    def _load(self, repo: str):
        from transformers import AutoTokenizer, VitsModel
        log.info("loading %s (MMS-TTS)", repo)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo)
        model.eval()
        return tokenizer, model

    def _synth_sync(self, text: str, repo: str) -> tuple[bytes, str]:
        import torch

        entry = self._cache.get(repo)
        if entry is None:
            entry = self._load(repo)
            self._cache.put(repo, entry)
        tokenizer, model = entry

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().detach().cpu().numpy()
        peak = float(abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.97          # normalise, VITS output is quiet
        pcm = (audio * 32767).astype("<i2").tobytes()

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(model.config.sampling_rate))
            handle.writeframes(pcm)
        return buffer.getvalue(), "audio/wav"

    async def synthesize(self, text, language, *, voice=None, gender=None,
                         rate=None) -> VoiceResult:
        if not await self.supports(language):
            raise LookupError(f"no MMS voice for {language.code}")
        repo = self._repo(language)
        async with self._lock:
            audio, mime = await asyncio.to_thread(self._synth_sync, text, repo)
        return VoiceResult(audio, mime, f"mms-{MMS_REPOS[language.code]}",
                           self.name, True)

    async def close(self) -> None:
        self._cache.clear()


def _split_for_tts(text: str, limit: int) -> list[str]:
    """Break text into <= limit pieces on word boundaries."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    out, buf = [], ""
    for word in text.split():
        if len(buf) + len(word) + 1 > limit and buf:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}" if buf else word
    if buf:
        out.append(buf)
    return out
