"""Speech-recognition chain.

Picks between Whisper and IndicConformer per request, because they fail in
opposite ways: Whisper is the only one that can identify an unknown language,
IndicConformer is the only one that can transcribe eight of the scheduled
Indian languages at all.

The decision, in order:

1. Source language given and IndicConformer covers it -> IndicConformer.
   This is the accuracy path; whisper-small measurably mangles Indic speech
   (it turned Tamil "இன்று" into "இந்து" in testing).
2. Source is "auto" -> Whisper, which detects. If what it detected is an
   Indian language that IndicConformer covers, transcribe a second time with
   IndicConformer and keep that text. Costs an extra pass, so
   ``LINGUA_ASR_REFINE=0`` disables it.
3. Anything else -> Whisper.

If the preferred engine fails, the other one is tried before giving up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from lingua.languages import LANGUAGES, get as get_lang
from lingua.types import Transcript

from .asr_engines import (CONFORMER_LANGS, WHISPER_LANGS,
                          IndicConformerEngine, WhisperEngine)

log = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class SpeechRecognizer:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None):
        self.whisper = WhisperEngine(model_size, device, compute_type)
        self.conformer = IndicConformerEngine()
        self.refine = _truthy(os.getenv("LINGUA_ASR_REFINE"), True)

    # --- capability --------------------------------------------------------
    def available(self) -> bool:
        return self.whisper.available() or self.conformer.available()

    @property
    def model_size(self) -> str:
        return self.whisper.model_size

    def languages(self) -> set[str]:
        covered: set[str] = set()
        if self.whisper.available():
            covered |= WHISPER_LANGS
        if self.conformer.available():
            covered |= CONFORMER_LANGS
        return covered

    # --- transcription -----------------------------------------------------
    async def transcribe(self, audio_path: str | Path, language: str | None = None,
                         translate_to_english: bool = False) -> Transcript:
        if not self.available():
            raise RuntimeError(
                "no speech recognition engine installed; "
                "pip install -r requirements-local.txt")

        path = str(audio_path)
        lang = get_lang(language) if language and language != "auto" else None
        notes: list[str] = []

        # 1. Explicit Indian language -> IndicConformer, the accurate path.
        if lang is not None and self.conformer.available() \
                and self.conformer.supports(lang):
            try:
                return await self.conformer.transcribe(path, lang)
            except Exception as exc:
                log.info("IndicConformer declined %s: %s", lang.code, exc)
                notes.append(f"indic-conformer unavailable ({exc})")
                if lang.whisper is None:
                    # Whisper cannot do this language either; nothing left.
                    raise RuntimeError(
                        f"{lang.name} needs IndicConformer, which is not "
                        f"available: {exc}") from exc

        # 2/3. Whisper, which also handles detection.
        if not self.whisper.available():
            raise RuntimeError("Whisper is not installed and IndicConformer "
                               "could not serve this request")

        if lang is not None and lang.whisper is None:
            # Whisper has no model for this language. Handing it the audio
            # anyway makes it transcribe in some *other* language and label
            # the result as this one, which is worse than failing.
            detail = "; ".join(notes) if notes else "IndicConformer is not available"
            raise RuntimeError(
                f"{lang.name} can only be transcribed by IndicConformer, and "
                f"it could not be used ({detail}). Whisper has no model for "
                f"{lang.name}, so transcribing with it would silently return "
                f"text in the wrong language.")

        transcript = await self.whisper.transcribe(
            path, lang, translate_to_english=translate_to_english)

        # Refine: Whisper found an Indian language IndicConformer does better.
        if (self.refine and lang is None and not translate_to_english
                and self.conformer.available()
                and transcript.language in CONFORMER_LANGS):
            detected = get_lang(transcript.language)
            try:
                better = await self.conformer.transcribe(path, detected)
                if better.text.strip():
                    better.segments = transcript.segments
                    better.duration_s = transcript.duration_s
                    better.took_ms += transcript.took_ms
                    better.provider = (f"{transcript.provider}+"
                                       f"{better.provider}")
                    return better
            except Exception as exc:
                log.info("refine pass skipped: %s", exc)
                notes.append("refine skipped")

        return transcript

    # --- reporting ---------------------------------------------------------
    def status(self) -> dict:
        covered = self.languages()
        indian = {l.code for l in LANGUAGES if l.indic or l.code == "en"}
        missing = [l.name for l in LANGUAGES if l.code not in covered]
        return {
            "available": self.available(),
            "model": self.whisper.model_size,
            "device": self.whisper.resolve_device()[0]
            if self.whisper.available() else None,
            "engines": [self.whisper.status(), self.conformer.status()],
            "refine": self.refine,
            "languages": len(covered),
            "indian_languages": len(covered & indian),
            "deaf_to": missing,
        }

    async def close(self) -> None:
        await self.conformer.close()
