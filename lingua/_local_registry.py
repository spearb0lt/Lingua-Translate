"""Local providers, isolated behind a lazy import.

Kept in its own module so ``lingua.engine`` can build the lite edition
without importing anything that would drag in torch.
"""

from __future__ import annotations

import logging

from .providers.base import Provider

log = logging.getLogger(__name__)


def local_providers() -> list[Provider]:
    """IndicTrans2, then the English pivot, then NLLB.

    Order matters within the LOCAL tier, because the router keeps registration
    order inside a tier:

    1. IndicTrans2 for Indic <-> English and Indic <-> Indic, where it is the
       strongest thing available.
    2. The English pivot for Indic <-> foreign, which beats NLLB direct on
       cross-script pairs. It uses IndicTrans2 for the Indic leg when that is
       available and the general model otherwise.
    3. NLLB for everything else, and as the fallback for everything above.

    All three share one model cache so the resident footprint stays inside the
    VRAM budget instead of each keeping its own copy.
    """
    try:
        from .providers._local_util import ModelCache
        from .providers.indictrans2 import IndicTrans2Provider
        from .providers.nllb import NLLBProvider
        from .providers.pivot import EnglishPivotProvider
    except ImportError as exc:
        log.warning("local providers unavailable: %s", exc)
        return []

    cache = ModelCache()
    out: list[Provider] = []

    it2 = IndicTrans2Provider(cache=cache)
    nllb = NLLBProvider(cache=cache)

    if it2.available():
        out.append(it2)
    else:
        log.info("IndicTrans2 skipped: install requirements-local.txt to enable it")

    if nllb.available():
        # The pivot is registered even when IndicTrans2 is gated: measured
        # with NLLB on both legs, it still beats NLLB direct on cross-script
        # pairs. It must come before nllb so it wins for the pairs it claims.
        out.append(EnglishPivotProvider(it2, nllb))
        out.append(nllb)
    else:
        log.info("NLLB skipped: install requirements-local.txt to enable it")

    return out
