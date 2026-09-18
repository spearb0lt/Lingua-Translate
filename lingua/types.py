"""Shared datatypes for the translation engine."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum


class Tier(IntEnum):
    """Provider preference order. Lower is tried first.

    This encodes the project's stated priority: run it yourself if you can,
    then free-and-keyless, then free-with-limits, then paid.
    """
    LOCAL = 0        # open-source weights on our own hardware
    FREE_WEB = 1     # keyless public endpoints (Google's web translate)
    FREE_API = 2     # free tiers that need no card (MyMemory, LibreTranslate)
    FREEMIUM = 3     # requires an API key; only used if one is configured


class ProviderError(RuntimeError):
    """A provider could not serve this request; the router should move on."""

    def __init__(self, provider: str, message: str, *, permanent: bool = False):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.permanent = permanent  # True => don't retry this provider for this pair


@dataclass
class TranslationRequest:
    text: str
    target: str                      # canonical language code
    source: str = "auto"             # canonical code, or "auto"
    prefer: str | None = None        # force a specific provider by name
    max_tier: Tier = Tier.FREEMIUM   # refuse anything above this tier
    transliterate_input: bool = True # convert romanized Indic to native script
    use_cache: bool = True           # set False to force a fresh translation
    #: Score the result by translating it back with a different engine. Costs
    #: one extra call, so it is opt-in.
    score: bool = False
    #: Try this many providers and keep the best-scoring translation. Implies
    #: ``score``. 1 means "first provider that works", the normal behaviour.
    best_of: int = 1


@dataclass
class TranslationResult:
    text: str                        # the translation
    source: str                      # language actually detected/used
    target: str
    provider: str                    # which provider produced this
    tier: Tier
    # Filled in when the input was romanized Indic and we converted it.
    romanized_input: bool = False
    native_input: str | None = None  # e.g. "मेरा नाम राहुल" for "mera naam rahul"
    detected_by: str | None = None   # how the source language was determined
    confidence: float | None = None  # how sure we are of the SOURCE language
    took_ms: int = 0
    cached: bool = False             # served from the cache, no engine ran
    notes: list[str] = field(default_factory=list)
    # Filled in only when quality scoring was requested. ``quality`` is a 0..1
    # estimate of how well the meaning survived, from a round trip back
    # through a different engine; ``back_translation`` is that round trip, so
    # a human can judge it too.
    quality: float | None = None
    back_translation: str | None = None
    graded_by: str | None = None     # provider that did the back translation
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "target": self.target,
            "provider": self.provider,
            "tier": self.tier.name,
            "romanized_input": self.romanized_input,
            "native_input": self.native_input,
            "detected_by": self.detected_by,
            "confidence": self.confidence,
            "took_ms": self.took_ms,
            "cached": self.cached,
            "notes": self.notes,
            "quality": self.quality,
            "back_translation": self.back_translation,
            "graded_by": self.graded_by,
            "alternatives": self.alternatives,
        }


    @classmethod
    def from_dict(cls, data: dict) -> "TranslationResult":
        """Rebuild a result stored in the cache."""
        data = dict(data)
        data["tier"] = Tier[data["tier"]] if isinstance(data.get("tier"), str)             else data.get("tier", Tier.LOCAL)
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class Transcript:
    """Result of speech recognition."""
    text: str
    language: str            # canonical code Whisper detected
    provider: str
    duration_s: float = 0.0
    took_ms: int = 0
    segments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "provider": self.provider,
            "duration_s": round(self.duration_s, 2),
            "took_ms": self.took_ms,
            "segments": self.segments,
        }
