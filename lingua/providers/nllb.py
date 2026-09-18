"""NLLB-200 -- the local model for everything outside IndicTrans2's range.

Meta's No Language Left Behind, distilled to 600M parameters, covering 200
languages. This is what handles Hindi->French, Japanese->Tamil and the rest
of the non-Indic pairs without going to the network.

LICENCE, read this before shipping anything commercial: NLLB-200 is
CC-BY-NC-4.0, i.e. **non-commercial use only**. That is fine for a college or
hackathon project, and it is why the router labels which engine served each
translation. If this ever becomes a product, swap this provider for
permissively licensed weights -- MADLAD-400 (Apache-2.0) or the
Helsinki-NLP/opus-mt-* pairs (CC-BY / Apache) -- and the rest of the system
does not change.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..languages import Language
from ..types import ProviderError, Tier
from .base import Provider, ProviderOutput
from ._local_util import ModelCache, pick_device, split_sentences

log = logging.getLogger(__name__)

DEFAULT_REPO = os.getenv("LINGUA_NLLB_REPO", "facebook/nllb-200-distilled-600M")

_BATCH = int(os.getenv("LINGUA_NLLB_BATCH", "4"))
_BEAMS = int(os.getenv("LINGUA_NLLB_BEAMS", "4"))


class NLLBProvider(Provider):
    name = "nllb"
    tier = Tier.LOCAL
    blurb = "NLLB-200 distilled 600M on this machine (CC-BY-NC, non-commercial)"
    offline = True

    def __init__(self, cache: ModelCache | None = None, device: str | None = None,
                 repo: str = DEFAULT_REPO):
        self._cache = cache or ModelCache()
        self._device_pref = device
        self._device = None
        self._repo = repo
        self._load_lock = asyncio.Lock()

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, source: Language | None, target: Language) -> bool:
        if source is None:
            return False           # NLLB has no detection of its own
        return (source.nllb and target.nllb and source.flores and target.flores
                and source.code != target.code)

    @property
    def device(self):
        if self._device is None:
            self._device = pick_device(self._device_pref)
            log.info("NLLB device: %s (%s)", self._device.device, self._device.reason)
        return self._device

    def _load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dev = self.device
        log.info("loading %s onto %s", self._repo, dev.device)
        tokenizer = AutoTokenizer.from_pretrained(self._repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._repo, dtype=getattr(torch, dev.dtype), low_cpu_mem_usage=True)
        model.to(dev.device)
        model.eval()
        return tokenizer, model

    def _get_model(self):
        entry = self._cache.get(self._repo)
        if entry is None:
            entry = self._load()
            self._cache.put(self._repo, entry)
        return entry

    def _translate_sync(self, sentences: list[str], src_flores: str,
                        tgt_flores: str) -> list[str]:
        import torch

        tokenizer, model = self._get_model()
        dev = self.device
        tokenizer.src_lang = src_flores

        bos = tokenizer.convert_tokens_to_ids(tgt_flores)
        if bos is None or bos == tokenizer.unk_token_id:
            raise ProviderError(self.name, f"tokenizer has no tag {tgt_flores}",
                                permanent=True)

        out: list[str] = []
        for start in range(0, len(sentences), _BATCH):
            batch = sentences[start:start + _BATCH]
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=256)
            encoded = {k: v.to(dev.device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=bos,
                    num_beams=_BEAMS,
                    max_length=256,
                    early_stopping=True,
                )
            out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return out

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        if source is None:
            raise ProviderError(self.name, "needs an explicit source language",
                                permanent=True)
        if not self.supports(source, target):
            raise ProviderError(self.name,
                                f"{source.code}->{target.code} not in NLLB-200",
                                permanent=True)
        if not self.available():
            raise ProviderError(self.name, "torch/transformers not installed",
                                permanent=True)

        sentences = split_sentences(text)
        if not sentences:
            return ProviderOutput(text="", detected_source=source.code)

        async with self._load_lock:
            try:
                pieces = await asyncio.to_thread(
                    self._translate_sync, sentences, source.flores, target.flores)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(self.name, f"{type(exc).__name__}: {exc}") from exc

        return ProviderOutput(text=" ".join(p.strip() for p in pieces).strip(),
                              detected_source=source.code)

    def status(self) -> dict:
        dev = self.device if self.available() else None
        return {
            "available": self.available(),
            "device": dev.device if dev else None,
            "device_reason": dev.reason if dev else "dependencies missing",
            "loaded": self._cache.loaded(),
            "repos": [self._repo],
            "licence": "CC-BY-NC-4.0 (non-commercial)",
        }

    async def close(self) -> None:
        self._cache.clear()
