"""Builds a configured Router.

Two editions, because the two deployment targets have very different budgets:

``full``
    Local models plus every network fallback. This is what runs on your laptop
    and on the Oracle ARM box.

``lite``
    Network providers only, no torch, no weights. Fits in Render's 512 MB /
    0.1 CPU free instance and in a Vercel serverless function, where a local
    model could never load.

Which providers exist is decided here; the order they are tried in is decided
by their tier in ``lingua.types.Tier``.
"""

from __future__ import annotations

import logging
import os

from .providers.base import Provider, Transliterator
from .providers.free_apis import LibreTranslateProvider, MyMemoryProvider
from .providers.google_free import GoogleDictProvider, GoogleFreeProvider
from .providers.inputtools import InputToolsTransliterator
from .providers.xlit_local import LocalTransliterator
from .router import Router
from .types import Tier

log = logging.getLogger(__name__)

EDITIONS = ("full", "lite")


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_router(edition: str | None = None, *, local: bool | None = None,
                 local_timeout: float | None = None) -> Router:
    edition = (edition or os.getenv("LINGUA_EDITION") or "full").lower()
    if edition not in EDITIONS:
        raise ValueError(f"edition must be one of {EDITIONS}, got {edition!r}")

    if local is None:
        local = edition == "full" and _truthy(os.getenv("LINGUA_LOCAL"), True)

    providers: list[Provider] = []

    if local:
        # Imported lazily so the lite edition never touches torch.
        from ._local_registry import local_providers
        providers.extend(local_providers())

    providers.append(GoogleFreeProvider())
    providers.append(GoogleDictProvider())
    providers.append(MyMemoryProvider())

    libre = LibreTranslateProvider()
    if libre.available():
        providers.append(libre)

    transliterators: list[Transliterator] = [InputToolsTransliterator()]
    offline_xlit = LocalTransliterator()
    if offline_xlit.available():
        transliterators.append(offline_xlit)

    if local_timeout is None:
        local_timeout = float(os.getenv("LINGUA_LOCAL_TIMEOUT", "120"))

    router = Router(providers, transliterators, local_timeout=local_timeout)
    log.info("edition=%s providers=%s", edition,
             [f"{p.name}:{p.tier.name}" for p in router.providers])
    return router


def max_tier_from_env() -> Tier:
    """Ceiling on which tiers may be used, from LINGUA_MAX_TIER."""
    raw = (os.getenv("LINGUA_MAX_TIER") or "FREEMIUM").upper()
    try:
        return Tier[raw]
    except KeyError:
        log.warning("unknown LINGUA_MAX_TIER=%r, defaulting to FREEMIUM", raw)
        return Tier.FREEMIUM
