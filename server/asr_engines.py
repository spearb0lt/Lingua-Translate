"""Speech-recognition engines, chained by ``asr.py``.

``WhisperEngine``
    faster-whisper (CTranslate2). Detects the language by itself, covers 52 of
    our 61 languages, and runs int8 on CPU or GPU. It is the only engine that
    can answer "what language is this?", so it always goes first when the
    source is ``auto``.

``IndicConformerEngine``
    AI4Bharat IndicConformer-600M, MIT. Covers **all 22 scheduled Indian
    languages**, including the eight Whisper cannot transcribe at all -- Bodo,
    Dogri, Konkani, Kashmiri, Maithili, Manipuri, Odia, Santali -- and is more
    accurate than whisper-small on the rest. It cannot detect language, so it
    needs to be told which one, either explicitly or from Whisper's guess.

The chain therefore does something slightly unusual: for Indian-language audio
it can run Whisper first only to identify the language, then re-transcribe
with IndicConformer for the actual text. That costs a second pass, which is
why ``LINGUA_ASR_REFINE=0`` turns it off.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from lingua.languages import LANGUAGES, Language, get as get_lang
from lingua.types import Transcript

log = logging.getLogger(__name__)

DEFAULT_WHISPER = os.getenv("LINGUA_WHISPER_MODEL", "small")
INDIC_CONFORMER_REPO = os.getenv(
    "LINGUA_CONFORMER_REPO", "ai4bharat/indic-conformer-600m-multilingual")

# Whisper code -> our code. Several of our languages share one Whisper code
# (zh and zh-TW are both "zh"), so first registration wins and the canonical
# language keeps the mapping. Never build the coverage set from this dict --
# it is lossy by construction; use WHISPER_LANGS instead.
_WHISPER_CODES: dict[str, str] = {}
for _l in LANGUAGES:
    if _l.whisper and _l.whisper not in _WHISPER_CODES:
        _WHISPER_CODES[_l.whisper] = _l.code

#: Our language codes Whisper can transcribe.
WHISPER_LANGS = {l.code for l in LANGUAGES if l.whisper}

# The 22 scheduled languages IndicConformer handles, by our codes.
CONFORMER_LANGS = {
    "as", "bn", "brx", "doi", "gom", "gu", "hi", "kn", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}

CONFORMER_GATED_HELP = (
    "IndicConformer is gated on Hugging Face (MIT licensed, access is "
    "automatic). Accept the terms once at "
    "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual "
    "and set HF_TOKEN. The same token unlocks IndicTrans2."
)


class ASREngine:
    name = "abstract"
    detects = False

    def available(self) -> bool:
        return False

    def supports(self, language: Language | None) -> bool:
        return False

    async def transcribe(self, path: str, language: Language | None) -> Transcript:
        raise NotImplementedError

    async def close(self) -> None:
        pass


# --------------------------------------------------------------------------
class WhisperEngine(ASREngine):
    name = "faster-whisper"
    detects = True

    def __init__(self, model_size: str | None = None, device: str | None = None,
                 compute_type: str | None = None):
        self.model_size = model_size or DEFAULT_WHISPER
        self._device_pref = device
        self._compute_pref = compute_type
        self._model = None
        self._resolved: tuple[str, str] | None = None
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, language: Language | None) -> bool:
        return language is None or language.whisper is not None

    def resolve_device(self) -> tuple[str, str]:
        if self._resolved:
            return self._resolved
        device = (self._device_pref or os.getenv("LINGUA_DEVICE") or "auto").lower()
        if device == "auto":
            try:
                import torch
                free_gb = (torch.cuda.mem_get_info()[0] / (1024 ** 3)
                           if torch.cuda.is_available() else 0)
                device = "cuda" if free_gb >= 1.0 else "cpu"
            except Exception:
                device = "cpu"
        compute = self._compute_pref or os.getenv("LINGUA_WHISPER_COMPUTE")
        if not compute:
            compute = "int8_float16" if device == "cuda" else "int8"
        self._resolved = (device, compute)
        return self._resolved

    def _load(self):
        from faster_whisper import WhisperModel
        device, compute = self.resolve_device()
        log.info("loading Whisper %s on %s (%s)", self.model_size, device, compute)
        return WhisperModel(self.model_size, device=device, compute_type=compute,
                            download_root=os.getenv("LINGUA_MODEL_DIR") or None)

    async def _get_model(self):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._load)
        return self._model

    def _run(self, model, path: str, hint: str | None, task: str):
        segments, info = model.transcribe(
            path, language=hint, task=task,
            beam_size=int(os.getenv("LINGUA_WHISPER_BEAMS", "5")),
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
        return list(segments), info

    async def transcribe(self, path: str, language: Language | None,
                         translate_to_english: bool = False) -> Transcript:
        started = time.perf_counter()
        model = await self._get_model()
        hint = language.whisper if language and language.whisper else None
        task = "translate" if translate_to_english else "transcribe"
        segments, info = await asyncio.to_thread(self._run, model, path, hint, task)

        text = " ".join(s.text.strip() for s in segments).strip()
        detected = _WHISPER_CODES.get(info.language, info.language)
        return Transcript(
            text=text, language=detected,
            provider=f"faster-whisper:{self.model_size}",
            duration_s=float(getattr(info, "duration", 0.0) or 0.0),
            took_ms=int((time.perf_counter() - started) * 1000),
            segments=[{"start": round(s.start, 2), "end": round(s.end, 2),
                       "text": s.text.strip()} for s in segments])

    def status(self) -> dict:
        device, compute = self.resolve_device() if self.available() else (None, None)
        return {"name": self.name, "available": self.available(),
                "model": self.model_size, "device": device,
                "compute_type": compute, "loaded": self._model is not None,
                "detects": True, "languages": len(WHISPER_LANGS)}


# --------------------------------------------------------------------------
class IndicConformerEngine(ASREngine):
    """AI4Bharat IndicConformer-600M. All 22 scheduled languages, MIT, gated."""

    name = "indic-conformer"
    detects = False

    def __init__(self, repo: str = INDIC_CONFORMER_REPO,
                 decoding: str | None = None):
        self._repo = repo
        self._decoding = decoding or os.getenv("LINGUA_CONFORMER_DECODING", "ctc")
        self._model = None
        self._gated = False
        self._broken: str | None = None
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        if self._gated or self._broken:
            return False
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, language: Language | None) -> bool:
        # No detection: it must be told the language.
        return language is not None and language.code in CONFORMER_LANGS

    def _device(self) -> str:
        want = (os.getenv("LINGUA_DEVICE") or "auto").lower()
        if want == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
                if want == "cuda" or free_gb >= 1.5:
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load(self):
        from transformers import AutoModel
        log.info("loading %s (IndicConformer)", self._repo)
        model = AutoModel.from_pretrained(self._repo, trust_remote_code=True)
        device = self._device()
        try:
            model = model.to(device)
        except Exception:
            device = "cpu"
        model.eval()
        self._device_used = device
        return model

    def _run(self, model, path: str, code: str) -> str:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:                      # mixdown to mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            out = model(wav, code, self._decoding)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return str(out).strip()

    async def transcribe(self, path: str, language: Language | None) -> Transcript:
        if language is None or not self.supports(language):
            raise LookupError("IndicConformer needs an explicit Indian language")
        started = time.perf_counter()

        async with self._lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load)
                except Exception as exc:
                    text = f"{type(exc).__name__} {exc}".lower()
                    if "gated" in text or "401" in text or "restricted" in text:
                        self._gated = True
                        log.error("IndicConformer gated.\n%s", CONFORMER_GATED_HELP)
                        raise LookupError("gated; " + CONFORMER_GATED_HELP) from exc
                    self._broken = str(exc)
                    log.warning("IndicConformer unavailable: %s", exc)
                    raise

            text = await asyncio.to_thread(self._run, self._model, path,
                                           language.code)

        return Transcript(
            text=text, language=language.code,
            provider=f"indic-conformer:{self._decoding}",
            took_ms=int((time.perf_counter() - started) * 1000))

    def status(self) -> dict:
        note = None
        if self._gated:
            note = CONFORMER_GATED_HELP
        elif self._broken:
            note = self._broken
        return {"name": self.name, "available": self.available(),
                "model": self._repo, "decoding": self._decoding,
                "device": getattr(self, "_device_used", None),
                "loaded": self._model is not None, "detects": False,
                "gated": self._gated, "languages": len(CONFORMER_LANGS),
                "note": note}

    async def close(self) -> None:
        self._model = None
