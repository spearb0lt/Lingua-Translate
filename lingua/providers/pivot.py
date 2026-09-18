"""Two-hop translation through English.

For a pair like Hindi -> Japanese, the only local model that covers it
directly is NLLB, and NLLB-600M is visibly strained on cross-script pairs --
it produced ``私の名前はラहुルです`` for "my name is Rahul", leaking a Devanagari
character into the katakana.

IndicTrans2 is much stronger than NLLB on Indic <-> English, so when exactly
one side of the pair is an Indian language we can do better by going through
English:

    Hindi -> Japanese   becomes   Hindi -> English (IndicTrans2)
                                  English -> Japanese (NLLB)

Two hops lose a little nuance, which is the classic pivot trade-off, but here
the first hop is the weak link being replaced and the gain outweighs it.

I expected this to pay off only when IndicTrans2 was available, and that was
wrong -- measured with NLLB on *both* legs it still beats NLLB direct, because
English is a far better-resourced hub than any Indic-to-CJK direction:

    hi->ja  direct  私の名前はラहुル   (Devanagari leaking into katakana)
            pivot   私の名前はラウル
    ja->hi  direct  मेरा नाम राफुल है।
            pivot   मेरा नाम राहुल है।

So the Indic leg uses IndicTrans2 when it can and falls back to the general
model otherwise, rather than the provider switching itself off.

Scope is deliberately narrow: exactly one side an Indian language. For pairs
where neither side is Indic the pivot showed no advantage (ta->fr came back
equally good both ways), and a second hop always costs some nuance, so it is
not applied speculatively.
"""

from __future__ import annotations

import logging

from ..languages import Language, require
from ..types import ProviderError, Tier
from .base import Provider, ProviderOutput

log = logging.getLogger(__name__)


class EnglishPivotProvider(Provider):
    name = "pivot_en"
    tier = Tier.LOCAL
    blurb = "Two hops via English (IndicTrans2 + NLLB) for Indic<->foreign pairs"
    offline = True
    detects = False

    def __init__(self, indic_provider: Provider | None, general_provider: Provider):
        #: preferred for the Indian-language leg, e.g. IndicTrans2. Optional --
        #: when it is missing or gated, the general model does both legs and
        #: the pivot still improves cross-script output.
        self.indic = indic_provider
        #: handles the non-Indian leg, e.g. NLLB
        self.general = general_provider
        self._english = require("en")

    def available(self) -> bool:
        return self.general.available()

    def _indic_leg(self) -> Provider:
        """IndicTrans2 if usable, otherwise the general model."""
        if self.indic is not None and self.indic.available():
            return self.indic
        return self.general

    def supports(self, source: Language | None, target: Language) -> bool:
        if source is None or not self.available():
            return False
        if source.code == "en" or target.code == "en":
            return False                      # no pivot needed, it is one hop
        if source.code == target.code:
            return False
        # Exactly one side Indian: the case where the Indic-specialist model
        # can replace NLLB's weakest leg.
        if source.indic == target.indic:
            return False
        indic_side, other_side = ((source, target) if source.indic
                                  else (target, source))
        # Use whichever provider will actually run the Indic leg: self.indic
        # may be None or gated, in which case the general model does both.
        leg = self._indic_leg()
        return (leg.supports(indic_side, self._english)
                or leg.supports(self._english, indic_side)) \
            and (self.general.supports(self._english, other_side)
                 or self.general.supports(other_side, self._english))

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        if source is None:
            raise ProviderError(self.name, "needs an explicit source language",
                                permanent=True)
        if not self.supports(source, target):
            raise ProviderError(self.name,
                                f"{source.code}->{target.code} is not a pivot case",
                                permanent=True)

        leg = self._indic_leg()
        first = leg if source.indic else self.general
        second = self.general if source.indic else leg

        middle = await self._hop(first, text, source, self._english)
        final = await self._hop(second, middle, self._english, target)

        log.debug("pivot %s->en->%s: %r -> %r", source.code, target.code,
                  middle[:60], final[:60])
        return ProviderOutput(text=final, detected_source=source.code)

    async def _hop(self, provider: Provider, text: str, source: Language,
                   target: Language) -> str:
        """Run one leg, retrying with the general model if the preferred one fails.

        The preferred provider for the Indic leg can fail on its very first
        call -- IndicTrans2 only discovers it is gated when it tries to
        download. Without this retry that first request would fall out of the
        pivot entirely and get the worse direct translation.
        """
        try:
            out = await provider.translate(text, source, target)
            if out.text.strip():
                return out.text
            reason = "returned nothing"
        except ProviderError as exc:
            reason = str(exc)

        if provider is self.general:
            raise ProviderError(
                self.name, f"hop {source.code}->{target.code} failed: {reason}")

        log.info("pivot leg %s->%s: %s failed (%s); retrying with %s",
                 source.code, target.code, provider.name, reason,
                 self.general.name)
        out = await self.general.translate(text, source, target)
        if not out.text.strip():
            raise ProviderError(
                self.name, f"hop {source.code}->{target.code} returned nothing")
        return out.text

    def status(self) -> dict:
        leg = self._indic_leg() if self.available() else None
        degraded = (self.indic is not None and not self.indic.available())
        return {
            "available": self.available(),
            "indic_leg": leg.name if leg else None,
            "other_leg": self.general.name,
            "note": ("IndicTrans2 unavailable, so both legs use the general "
                     "model; the pivot still helps cross-script pairs")
                    if degraded else None,
        }
