"""Lite edition: network providers only, no torch, no model weights.

Why this exists as a separate app: Render's free web service gives you 512 MB
of RAM and 0.1 CPU, and Vercel functions are similarly small and ephemeral.
IndicTrans2 alone wants more than that just to load, so the full edition
genuinely cannot run there. Rather than pretend, this edition drops the local
tier entirely and serves the same UI over the keyless and free-API providers.

It reuses ``lingua`` and the same static front end, so there is one codebase
and one set of behaviours -- only the provider list differs.

A caveat worth knowing before you demo from the cloud: Google's keyless
endpoints are much more likely to rate-limit a datacentre IP than your home
connection. The router falls through to MyMemory when that happens, so the
app keeps working, but quality drops. If it matters, point
``LIBRETRANSLATE_URL`` at an instance you control.

Run locally:
    .venv/Scripts/python.exe -m uvicorn lite.main:app --port 8001
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from lingua.engine import build_router, max_tier_from_env  # noqa: E402
from lingua.languages import get as get_lang, picker  # noqa: E402
from lingua.types import ProviderError, Tier, TranslationRequest  # noqa: E402

logging.basicConfig(level=os.getenv("LINGUA_LOG_LEVEL", "INFO"))
log = logging.getLogger("lingua.lite")

STATIC_DIR = ROOT / "server" / "static"
MAX_TEXT_CHARS = int(os.getenv("LINGUA_MAX_TEXT_CHARS", "8000"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["router"] = build_router("lite")
    log.info("lite edition up: %s", [p.name for p in state["router"].providers])
    try:
        yield
    finally:
        await state["router"].close()


app = FastAPI(title="Lingua (lite)", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("LINGUA_CORS_ORIGINS", "*").split(","),
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)


class TranslateBody(BaseModel):
    text: str
    target: str
    source: str = "auto"
    prefer: str | None = None
    max_tier: str | None = None
    transliterate_input: bool = True


class TransliterateBody(BaseModel):
    text: str
    language: str


def _tier(name: str | None) -> Tier:
    if not name:
        return max_tier_from_env()
    try:
        return Tier[name.strip().upper()]
    except KeyError:
        raise HTTPException(400, f"unknown tier {name!r}")


@app.get("/api/health")
async def health():
    return {"status": "ok", "edition": "lite"}


@app.get("/api/languages")
async def languages():
    return {"groups": picker()}


@app.get("/api/providers")
async def providers():
    return {
        "translation": state["router"].describe(),
        # The lite edition has no local speech; the UI hides the controls.
        "speech_to_text": {"available": False,
                           "note": "not available in the lite edition"},
        "text_to_speech": {"available": False,
                           "note": "not available in the lite edition"},
        "max_tier": max_tier_from_env().name,
    }


@app.post("/api/translate")
async def translate(body: TranslateBody):
    if len(body.text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text longer than {MAX_TEXT_CHARS} characters")
    try:
        result = await state["router"].translate(TranslationRequest(
            text=body.text, target=body.target, source=body.source or "auto",
            prefer=body.prefer, max_tier=_tier(body.max_tier),
            transliterate_input=body.transliterate_input))
    except KeyError as exc:
        raise HTTPException(400, str(exc))
    except ProviderError as exc:
        raise HTTPException(502, str(exc))
    return result.to_dict()


@app.post("/api/transliterate")
async def transliterate(body: TransliterateBody):
    lang = get_lang(body.language)
    if lang is None:
        raise HTTPException(400, f"unknown language {body.language!r}")
    native, provider = await state["router"].to_native_script(body.text, lang)
    if native is None:
        raise HTTPException(422, f"cannot transliterate {body.language}")
    return {"text": native, "provider": provider, "language": lang.code}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")
