"""Text-to-speech chain.

Picks the first engine that can actually speak the requested language, in a
configurable order, and falls through when one fails. Same shape as the
translation router: the caller asks for a language, not an engine.

Coverage after chaining all three (measured, not assumed):
  Indian languages   15 of 23   (Edge 11, + Punjabi/Assamese/Odia/Maithili)
  Foreign languages  38 of 38   (Edge 36, + Hausa/Yoruba from MMS)

Still silent, because no public voice model exists for them anywhere:
Bodo, Dogri, Konkani, Kashmiri, Manipuri, Sanskrit, Santali, Sindhi.
"""

from __future__ import annotations

import logging
import os

from lingua.languages import LANGUAGES, Language, get as get_lang

from .tts_engines import EdgeEngine, GoogleEngine, MMSEngine, TTSEngine, VoiceResult

log = logging.getLogger(__name__)

_DEFAULT_ORDER = "edge,google,mms"


class TextToSpeech:
    def __init__(self, order: str | None = None):
        self._engines: dict[str, TTSEngine] = {
            "edge": EdgeEngine(),
            "google": GoogleEngine(),
            "mms": MMSEngine(),
        }
        raw = (order or os.getenv("LINGUA_TTS_ORDER") or _DEFAULT_ORDER)
        self._order = [n.strip() for n in raw.split(",")
                       if n.strip() in self._engines]
        if not self._order:
            log.warning("LINGUA_TTS_ORDER=%r matched no engines; using default", raw)
            self._order = _DEFAULT_ORDER.split(",")

    # --- engine selection --------------------------------------------------
    def _chain(self, offline_only: bool = False) -> list[TTSEngine]:
        engines = [self._engines[n] for n in self._order
                   if self._engines[n].available()]
        if offline_only:
            engines = [e for e in engines if e.offline]
        return engines

    def available(self) -> bool:
        return bool(self._chain())

    # --- voices ------------------------------------------------------------
    async def voices(self, refresh: bool = False) -> list[dict]:
        """Every voice we can offer, across engines."""
        out: list[dict] = []
        edge = self._engines["edge"]
        if edge.available():
            out.extend(await edge.all_voices(refresh=refresh))
        for language in LANGUAGES:
            for name in ("google", "mms"):
                engine = self._engines[name]
                if engine.available():
                    out.extend(await engine.voices_for(language))
        return out

    async def voices_for(self, language: str, offline_only: bool = False) -> list[dict]:
        lang = get_lang(language)
        if lang is None:
            return []
        out: list[dict] = []
        for engine in self._chain(offline_only):
            try:
                out.extend(await engine.voices_for(lang))
            except Exception as exc:
                log.debug("voices_for(%s) failed on %s: %s", lang.code, engine.name, exc)
        return out

    async def can_speak(self, language: str, offline_only: bool = False) -> bool:
        return bool(await self.voices_for(language, offline_only))

    async def pick_voice(self, language: str, prefer_gender: str | None = None,
                         explicit: str | None = None) -> str | None:
        if explicit:
            return explicit
        options = await self.voices_for(language)
        if not options:
            return None
        if prefer_gender:
            want = prefer_gender.strip().lower()
            for voice in options:
                if voice["gender"].lower() == want:
                    return voice["name"]
        return options[0]["name"]

    # --- synthesis ---------------------------------------------------------
    async def synthesize(self, text: str, language: str, *,
                         voice: str | None = None, gender: str | None = None,
                         rate: str | None = None,
                         offline_only: bool = False) -> VoiceResult:
        lang = get_lang(language)
        if lang is None:
            raise LookupError(f"unknown language {language!r}")
        text = text.strip()
        if not text:
            raise ValueError("nothing to speak")

        # An explicit voice name pins the engine it belongs to.
        forced = None
        if voice:
            for name, engine in self._engines.items():
                if voice.startswith(f"{name}-") or (
                        name == "edge" and "-" in voice and voice[0].isalpha()
                        and not voice.startswith(("google-", "mms-"))):
                    forced = engine
                    break

        chain = [forced] if forced else self._chain(offline_only)
        errors: list[str] = []

        for engine in chain:
            if not engine.available():
                continue
            try:
                if not forced and not await engine.supports(lang):
                    continue
                result = await engine.synthesize(
                    text, lang, voice=voice if forced or engine.name == "edge" else None,
                    gender=gender, rate=rate)
            except Exception as exc:
                log.info("tts engine %s declined %s: %s", engine.name, lang.code, exc)
                errors.append(f"{engine.name}: {exc}")
                continue
            if result.audio:
                return result
            errors.append(f"{engine.name}: empty audio")

        detail = "; ".join(errors) if errors else "no engine covers this language"
        raise LookupError(f"no voice available for {lang.code} ({detail})")

    # --- reporting ---------------------------------------------------------
    async def status(self) -> dict:
        engines = []
        for name in self._order:
            engine = self._engines[name]
            engines.append({"name": name, "available": engine.available(),
                            "offline": engine.offline})

        speakable, offline_speakable = [], []
        for language in LANGUAGES:
            if await self.can_speak(language.code):
                speakable.append(language.code)
            if await self.can_speak(language.code, offline_only=True):
                offline_speakable.append(language.code)

        indian = {l.code for l in LANGUAGES if l.indic or l.code == "en"}
        return {
            "available": self.available(),
            "engine": "+".join(self._order),
            "engines": engines,
            "order": self._order,
            "languages": len(speakable),
            "indian_languages": len(set(speakable) & indian),
            "offline_languages": len(offline_speakable),
            "silent": [l.name for l in LANGUAGES if l.code not in speakable],
            "note": "Edge and Google need a network; MMS is fully offline",
        }

    async def close(self) -> None:
        for engine in self._engines.values():
            try:
                await engine.close()
            except Exception:
                log.debug("error closing %s", engine.name, exc_info=True)
