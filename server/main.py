"""FastAPI app: the full edition, with local models and speech.

Run it:
    .venv/Scripts/python.exe -m uvicorn server.main:app --reload --port 8000

Endpoints are deliberately thin. All routing intelligence lives in
``lingua.router``; this module only handles HTTP, file uploads and streaming
audio back.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lingua.engine import build_router, max_tier_from_env
from lingua.languages import picker
from lingua.types import ProviderError, Tier, TranslationRequest

from .asr import SpeechRecognizer
from .tts import TextToSpeech

logging.basicConfig(
    level=os.getenv("LINGUA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("lingua.server")

STATIC_DIR = Path(__file__).parent / "static"
MAX_AUDIO_BYTES = int(os.getenv("LINGUA_MAX_AUDIO_MB", "25")) * 1024 * 1024
MAX_TEXT_CHARS = int(os.getenv("LINGUA_MAX_TEXT_CHARS", "20000"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["router"] = build_router()
    state["asr"] = SpeechRecognizer()
    state["tts"] = TextToSpeech()
    log.info("edition=%s providers=%s",
             os.getenv("LINGUA_EDITION", "full"),
             [p.name for p in state["router"].providers])

    if os.getenv("LINGUA_PRELOAD", "").lower() in {"1", "true", "yes"}:
        log.info("LINGUA_PRELOAD set: warming models now")
        try:
            await state["router"].translate(TranslationRequest(
                text="namaste", source="hi", target="en", max_tier=Tier.LOCAL))
        except Exception as exc:
            log.warning("preload failed: %s", exc)
    try:
        yield
    finally:
        await state["router"].close()


app = FastAPI(title="Lingua", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("LINGUA_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- models ---------------------------------------------------------------
class TranslateBody(BaseModel):
    text: str = Field(..., description="text to translate")
    target: str = Field(..., description="target language code")
    source: str = Field("auto", description="source code, or 'auto'")
    prefer: str | None = Field(None, description="force one provider by name")
    max_tier: str | None = Field(None, description="LOCAL|FREE_WEB|FREE_API|FREEMIUM")
    transliterate_input: bool = True


class TransliterateBody(BaseModel):
    text: str
    language: str


class SpeakBody(BaseModel):
    text: str
    language: str
    voice: str | None = None
    gender: str | None = None
    rate: str | None = None
    offline_only: bool = False   # force the on-device MMS voices


def _resolve_tier(name: str | None) -> Tier:
    if not name:
        return max_tier_from_env()
    try:
        return Tier[name.strip().upper()]
    except KeyError:
        raise HTTPException(400, f"unknown tier {name!r}; expected one of "
                                 f"{[t.name for t in Tier]}")


# --- metadata -------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "edition": os.getenv("LINGUA_EDITION", "full")}


@app.get("/api/languages")
async def languages():
    return {"groups": picker()}


@app.get("/api/providers")
async def providers():
    return {
        "translation": state["router"].describe(),
        "speech_to_text": state["asr"].status(),
        "text_to_speech": await state["tts"].status(),
        "max_tier": max_tier_from_env().name,
    }


# --- text -----------------------------------------------------------------
@app.post("/api/translate")
async def translate(body: TranslateBody):
    if len(body.text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text longer than {MAX_TEXT_CHARS} characters")
    req = TranslationRequest(
        text=body.text,
        target=body.target,
        source=body.source or "auto",
        prefer=body.prefer,
        max_tier=_resolve_tier(body.max_tier),
        transliterate_input=body.transliterate_input,
    )
    try:
        result = await state["router"].translate(req)
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ProviderError as exc:
        raise HTTPException(502, str(exc))
    return result.to_dict()


@app.post("/api/transliterate")
async def transliterate(body: TransliterateBody):
    """Romanized -> native script, exposed on its own for the live input hint."""
    from lingua.languages import get as get_lang

    lang = get_lang(body.language)
    if lang is None:
        raise HTTPException(400, f"unknown language {body.language!r}")
    native, provider = await state["router"].to_native_script(body.text, lang)
    if native is None:
        raise HTTPException(422, f"cannot transliterate {body.language}")
    return {"text": native, "provider": provider, "language": lang.code}


# --- speech ---------------------------------------------------------------
async def _save_upload(audio: UploadFile) -> Path:
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"audio larger than {MAX_AUDIO_BYTES // 1024 // 1024} MB")
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return Path(handle.name)


@app.post("/api/speech/transcribe")
async def speech_transcribe(audio: UploadFile = File(...),
                            language: str = Form("auto")):
    if not state["asr"].available():
        raise HTTPException(503, "speech recognition not installed "
                                 "(pip install -r requirements-local.txt)")
    path = await _save_upload(audio)
    try:
        transcript = await state["asr"].transcribe(path, language=language)
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(500, f"transcription failed: {exc}")
    finally:
        path.unlink(missing_ok=True)
    return transcript.to_dict()


@app.post("/api/speech/translate")
async def speech_translate(audio: UploadFile = File(...),
                           target: str = Form(...),
                           source: str = Form("auto"),
                           max_tier: str | None = Form(None)):
    """Speech in, translated text out. The speech-to-speech path adds /api/speak."""
    if not state["asr"].available():
        raise HTTPException(503, "speech recognition not installed")

    path = await _save_upload(audio)
    try:
        transcript = await state["asr"].transcribe(path, language=source)
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(500, f"transcription failed: {exc}")
    finally:
        path.unlink(missing_ok=True)

    if not transcript.text.strip():
        return {"transcript": transcript.to_dict(), "translation": None,
                "note": "no speech detected"}

    try:
        result = await state["router"].translate(TranslationRequest(
            text=transcript.text,
            target=target,
            source=transcript.language or "auto",
            max_tier=_resolve_tier(max_tier),
        ))
    except ProviderError as exc:
        raise HTTPException(502, str(exc))

    return {"transcript": transcript.to_dict(), "translation": result.to_dict()}


@app.get("/api/voices")
async def voices(language: str | None = Query(None),
                 offline_only: bool = Query(False)):
    engine = state["tts"]
    if not engine.available():
        raise HTTPException(503, "no text-to-speech engine is installed")
    items = await (engine.voices_for(language, offline_only) if language
                   else engine.voices())
    return {"voices": items, "language": language}


@app.post("/api/speak")
async def speak(body: SpeakBody):
    engine = state["tts"]
    if not engine.available():
        raise HTTPException(503, "no text-to-speech engine is installed")
    try:
        result = await engine.synthesize(
            body.text, body.language, voice=body.voice, gender=body.gender,
            rate=body.rate, offline_only=body.offline_only)
    except LookupError as exc:
        # 404 rather than 500: the language genuinely has no voice anywhere.
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("speech synthesis failed")
        raise HTTPException(502, f"speech synthesis failed: {exc}")
    # MP3 from the network engines, WAV from MMS, so the type varies.
    return Response(content=result.audio, media_type=result.mime,
                    headers={"X-Voice": result.voice,
                             "X-TTS-Engine": result.engine,
                             "X-TTS-Offline": str(result.offline).lower(),
                             "Cache-Control": "no-store"})


# --- static UI ------------------------------------------------------------
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    async def index():
        return JSONResponse({"error": "static directory missing"}, status_code=500)
