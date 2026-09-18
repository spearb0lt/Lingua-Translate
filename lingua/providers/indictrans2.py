"""IndicTrans2 -- the local model for Indian languages.

AI4Bharat's IndicTrans2, MIT licensed, covering English plus all 22 scheduled
Indian languages. We use the distilled checkpoints (200M for the
English-centric directions, 320M for Indic-to-Indic) because they run
comfortably on a 4 GB laptop GPU and on a CPU-only cloud box, which is the
whole point of preferring local models.

Two things it will not do, both handled by the router rather than here:

* It has no language detection -- the caller must say what the source is.
* It has no romanized input mode. "mera naam rahul" must be transliterated to
  "मेरा नाम राहुल" first; see ``lingua.providers.inputtools`` and
  ``lingua.providers.xlit_local``.

The checkpoints ship custom modelling code, so loading them needs
``trust_remote_code=True``. That executes code from the model repo, which is
worth knowing even though these are first-party AI4Bharat repos.

They are also **gated** on the Hub (``gated: auto``): still MIT licensed, and
approval is automatic, but you must be signed in and have accepted the terms
on each model page once. Without that, downloads fail with a 401 -- see
``GATED_HELP`` below for what to tell the user.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..languages import Language, indic_pair
from ..types import ProviderError, Tier
from .base import Provider, ProviderOutput
from ._local_util import ModelCache, pick_device, split_sentences

log = logging.getLogger(__name__)

EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-dist-320M"

ALL_REPOS = (EN_INDIC, INDIC_EN, INDIC_INDIC)

_BATCH = int(os.getenv("LINGUA_IT2_BATCH", "8"))
_BEAMS = int(os.getenv("LINGUA_IT2_BEAMS", "5"))

GATED_HELP = (
    "IndicTrans2 is gated on Hugging Face. It is still MIT licensed and access "
    "is granted automatically, but you have to accept the terms once while "
    "signed in:\n"
    "  1. create a free account at https://huggingface.co/join\n"
    "  2. open each model page and click 'Agree and access repository':\n"
    "     https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M\n"
    "     https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M\n"
    "     https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M\n"
    "  3. make a read token at https://huggingface.co/settings/tokens\n"
    "  4. set it: HF_TOKEN=hf_xxx  (or run: huggingface-cli login)\n"
    "Until then the router falls back to NLLB and the free network engines."
)


def _is_gated_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "gatedrepo" in text or ("401" in text and "gated" in text) \
        or "restricted" in text and "access" in text


def choose_repo(source: Language, target: Language) -> str:
    """Which of the three checkpoints handles this direction."""
    if source.code == "en":
        return EN_INDIC
    if target.code == "en":
        return INDIC_EN
    return INDIC_INDIC


class IndicTrans2Provider(Provider):
    name = "indictrans2"
    tier = Tier.LOCAL
    blurb = "IndicTrans2 (AI4Bharat, MIT) running on this machine"
    offline = True

    def __init__(self, cache: ModelCache | None = None, device: str | None = None):
        self._cache = cache or ModelCache()
        self._device_pref = device
        self._device = None
        self._processor = None
        self._gated = False
        self._load_lock = asyncio.Lock()

    # --- capability checks -------------------------------------------------
    def available(self) -> bool:
        if self._gated:
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import IndicTransToolkit  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, source: Language | None, target: Language) -> bool:
        if source is None:
            return False          # no detection of its own
        return indic_pair(source, target)

    @property
    def device(self):
        if self._device is None:
            self._device = pick_device(self._device_pref)
            log.info("IndicTrans2 device: %s (%s)", self._device.device,
                     self._device.reason)
        return self._device

    # --- model loading -----------------------------------------------------
    def _load(self, repo: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dev = self.device
        log.info("loading %s onto %s", repo, dev.device)
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            repo,
            trust_remote_code=True,
            dtype=getattr(torch, dev.dtype),
            low_cpu_mem_usage=True,
        )
        model.to(dev.device)
        model.eval()
        return tokenizer, model

    def _get_model(self, repo: str):
        entry = self._cache.get(repo)
        if entry is None:
            entry = self._load(repo)
            self._cache.put(repo, entry)
        return entry

    def _get_processor(self):
        if self._processor is None:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:  # older layout
                from IndicTransToolkit import IndicProcessor
            self._processor = IndicProcessor(inference=True)
        return self._processor

    # --- translation -------------------------------------------------------
    def _translate_sync(self, sentences: list[str], src_flores: str,
                        tgt_flores: str, repo: str) -> list[str]:
        import torch

        tokenizer, model = self._get_model(repo)
        processor = self._get_processor()
        dev = self.device
        out: list[str] = []

        for start in range(0, len(sentences), _BATCH):
            batch = sentences[start:start + _BATCH]
            prepared = processor.preprocess_batch(
                batch, src_lang=src_flores, tgt_lang=tgt_flores)
            encoded = tokenizer(prepared, truncation=True, padding="longest",
                                return_tensors="pt", max_length=256)
            encoded = {k: v.to(dev.device) for k, v in encoded.items()}

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    num_beams=_BEAMS,
                    num_return_sequences=1,
                    max_length=256,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            out.extend(processor.postprocess_batch(decoded, lang=tgt_flores))

        return out

    async def translate(self, text: str, source: Language | None,
                        target: Language) -> ProviderOutput:
        if source is None:
            raise ProviderError(self.name, "needs an explicit source language",
                                permanent=True)
        if not indic_pair(source, target):
            raise ProviderError(
                self.name, f"{source.code}->{target.code} is outside IndicTrans2",
                permanent=True)
        if not self.available():
            # Distinguish "you never installed it" from "the Hub gated it",
            # because the fixes are completely different.
            if self._gated:
                raise ProviderError(self.name, "gated repo; " + GATED_HELP,
                                    permanent=True)
            raise ProviderError(self.name, "local model dependencies not installed",
                                permanent=True)

        sentences = split_sentences(text)
        if not sentences:
            return ProviderOutput(text="", detected_source=source.code)

        repo = choose_repo(source, target)
        async with self._load_lock:
            try:
                pieces = await asyncio.to_thread(
                    self._translate_sync, sentences, source.flores,
                    target.flores, repo)
            except Exception as exc:
                if _is_gated_error(exc):
                    # Permanent: retrying will not help until the user accepts
                    # the terms, so stop asking this provider and fall through.
                    self._gated = True
                    log.error("IndicTrans2 is gated and no token is set.\n%s",
                              GATED_HELP)
                    raise ProviderError(self.name, "gated repo; " + GATED_HELP,
                                        permanent=True) from exc
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
            "repos": list(ALL_REPOS),
            "gated": self._gated,
            "note": GATED_HELP if self._gated else None,
        }

    async def close(self) -> None:
        self._cache.clear()
