"""Provider interface.

A provider translates text and knows nothing about routing, fallback or
scripts. The router in ``lingua.router`` owns all of that.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..languages import Language
from ..types import Tier


@dataclass
class ProviderOutput:
    text: str
    detected_source: str | None = None  # canonical code, if the provider detects
    romanized: bool = False             # provider says input was romanized Indic


class Provider:
    name: str = "abstract"
    tier: Tier = Tier.FREEMIUM
    #: Human-readable note shown in the UI next to the engine badge.
    blurb: str = ""
    #: True when this provider needs no network (matters for offline mode).
    offline: bool = False
    #: True when the provider can identify the source language itself. Local
    #: NMT models cannot, so the router must resolve the source before calling
    #: them; a detecting provider can be handed source=None instead.
    detects: bool = False

    def available(self) -> bool:
        """Can this provider run at all right now (deps installed, key set)?"""
        return True

    def supports(self, source: Language | None, target: Language) -> bool:
        """Can it handle this specific pair? ``source`` is None for auto-detect."""
        return True

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        raise NotImplementedError

    async def close(self) -> None:
        """Release sockets / unload models."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier.name,
            "blurb": self.blurb,
            "offline": self.offline,
            "available": self.available(),
        }


class Transliterator:
    """Converts romanized Indic text to its native script."""

    name: str = "abstract"
    offline: bool = False

    #: True when the conversion is a best-effort approximation rather than
    #: a faithful one. The router surfaces this to the user.
    approximate: bool = False

    def available(self) -> bool:
        return True

    def supports(self, language: Language) -> bool:
        return True

    async def to_native(self, text: str, language: Language) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        pass
