"""The routing layer: decide what language this is, then who translates it.

Two decisions live here, and only here.

**Which language is the input?** Providers disagree about this. Local NMT
models cannot detect at all; Google detects well and is the only thing that
reliably reports romanized Indic as ``hi-Latn``. So we resolve the source
ourselves from the script (works offline), and let a detecting provider
override us when one actually serves the request.

**Who translates it?** Strict tier order -- local weights, then keyless web
endpoints, then free APIs, then anything needing a key. Each provider that
cannot or will not serve is skipped and the reason recorded, so a failure
degrades to the next tier instead of to an error.

Romanized input is forked deliberately: Google gets the original Latin text
because it handles Hinglish natively and better, while local models get a
transliterated native-script version because IndicTrans2 has no romanized
input mode at all.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .languages import Language, get as get_lang, require
from .providers.base import Provider, ProviderOutput, Transliterator
from .script import (SCRIPT_DEFAULT_LANG, ScriptInfo, analyse,
                     looks_like_wrong_script)
from .types import ProviderError, Tier, TranslationRequest, TranslationResult

log = logging.getLogger(__name__)

# Devanagari is shared by Hindi, Marathi, Nepali, Sanskrit, Konkani, Maithili
# and more, so seeing the script alone is weak evidence of the language.
_AMBIGUOUS_SCRIPTS = {"Devanagari", "Bengali", "Arabic", "Latin"}

#: Below this, we would rather a provider that can detect decides for itself.
#: This was once compared with a bare ``<`` against a value of exactly 0.6,
#: so Devanagari never qualified and every Marathi sentence was translated as
#: Hindi -- "माझे नाव राहुल आहे" came back as "My boat is Rahul", because नाव
#: means boat in Hindi and name in Marathi.
DETECT_THRESHOLD = 0.75


class SourceResolution:
    __slots__ = ("language", "method", "confidence", "info")

    def __init__(self, language: Language | None, method: str,
                 confidence: float, info: ScriptInfo):
        self.language = language
        self.method = method
        self.confidence = confidence
        self.info = info


class Router:
    def __init__(self, providers: list[Provider],
                 transliterators: list[Transliterator] | None = None,
                 local_timeout: float | None = 120.0):
        # Stable sort by tier keeps registration order within a tier, which is
        # how we express "gtx before clients5".
        self.providers = sorted(providers, key=lambda p: p.tier)
        self.transliterators = transliterators or []
        self.local_timeout = local_timeout

    # --- source language ---------------------------------------------------
    def resolve_source(self, text: str, declared: str | None) -> SourceResolution:
        info = analyse(text, declared)

        explicit = get_lang(declared)
        if explicit is not None:
            return SourceResolution(explicit, "explicit", 1.0, info)

        if info.romanized and info.guess:
            return SourceResolution(get_lang(info.guess), "romanized-heuristic",
                                    info.confidence, info)

        if info.script != "Latin":
            # analyse() already disambiguated within the script where it
            # could, and reported how sure it is; trust that rather than
            # flattening every script to one number.
            guess = info.guess or SCRIPT_DEFAULT_LANG.get(info.script)
            method = "script-lexicon" if info.confidence >= 0.5 else "script"
            return SourceResolution(get_lang(guess), method, info.confidence, info)

        return SourceResolution(get_lang("en"), "script", 0.5, info)

    # --- transliteration ---------------------------------------------------
    async def to_native_script(self, text: str,
                               language: Language) -> tuple[str | None, str | None]:
        """Romanized -> native script. Returns (native_text, provider_name)."""
        for tl in self.transliterators:
            if not tl.available():
                continue
            supports = getattr(tl, "supports", None)
            if supports is not None and not supports(language):
                continue
            try:
                native = await tl.to_native(text, language)
            except ProviderError as exc:
                log.info("transliterator %s declined: %s", tl.name, exc)
                continue
            except Exception as exc:
                log.warning("transliterator %s failed: %s", tl.name, exc)
                continue
            if native and native.strip():
                return native.strip(), tl.name
        return None, None

    # --- main entry point --------------------------------------------------
    async def translate(self, req: TranslationRequest) -> TranslationResult:
        started = time.perf_counter()
        text = req.text.strip()
        target = require(req.target)

        if not text:
            return TranslationResult(text="", source=req.source, target=target.code,
                                     provider="noop", tier=Tier.LOCAL)

        resolution = self.resolve_source(text, None if req.source == "auto" else req.source)
        source = resolution.language
        notes: list[str] = []

        # Nothing to do when the languages match and the script is already right.
        if source is not None and source.code == target.code and not resolution.info.romanized:
            return TranslationResult(
                text=text, source=source.code, target=target.code,
                provider="noop", tier=Tier.LOCAL,
                detected_by=resolution.method, confidence=resolution.confidence,
                notes=["source and target are the same language"],
                took_ms=int((time.perf_counter() - started) * 1000))

        # Romanized Indic: convert to native script before translating anything.
        #
        # This is not just for the local models. Measured against Google's own
        # endpoint, feeding it native script beats feeding it romanized text
        # for every language except Hindi, where the two tie:
        #   "ami tomake bhalobashi"     -> returned untranslated
        #   "আমি তোমাকে ভালোবাসি"        -> "i love you"
        #   "naan unnai kaadhalikkiren" -> "नान उन्नै कधलिक्किरें" (transliterated, not translated)
        #   "நான் உன்னை காதலிக்கிறேன்"   -> "मुझे तुमसे प्यार है"
        # So everyone gets the native text when we have a trustworthy one.
        native_text: str | None = None
        native_approx = False
        if resolution.info.romanized and req.transliterate_input and source is not None:
            native_text, xlit_name = await self.to_native_script(text, source)
            if native_text:
                native_approx = any(getattr(t, "approximate", False)
                                    for t in self.transliterators
                                    if t.name == xlit_name)
                notes.append(f"read as {source.name} via {xlit_name}"
                             + (" (approximate)" if native_approx else ""))
            else:
                notes.append("could not transliterate romanized input")

        candidates = self._candidates(req, source, target)
        if not candidates:
            raise ProviderError("router",
                                f"no provider can translate {source.code if source else 'auto'}"
                                f"->{target.code} at or below tier {req.max_tier.name}")

        errors: list[str] = []
        # Results we refused for wrong-script output, kept as a last resort:
        # a garbled translation still beats no translation at all.
        rejected: list[tuple[Provider, ProviderOutput, str]] = []

        for provider in candidates:
            payload = text
            src_arg = source

            if native_text and (provider.tier == Tier.LOCAL or not native_approx):
                # Native script plus a known source language is the safest
                # input for everyone. Local models have no other option, since
                # IndicTrans2 cannot read romanized text at all.
                payload = native_text
            elif provider.detects and req.source == "auto" \
                    and resolution.confidence < 0.6:
                # We are unsure what this is, and this provider has its own
                # detector, so let it decide rather than forcing our guess.
                # Providers without one (MyMemory) keep the explicit source,
                # since handing them None would only make them refuse.
                src_arg = None

            try:
                output = await self._invoke(provider, payload, src_arg, target)
            except ProviderError as exc:
                log.info("provider %s declined: %s", provider.name, exc)
                errors.append(str(exc))
                continue
            except asyncio.TimeoutError:
                msg = f"{provider.name}: timed out after {self.local_timeout}s"
                log.warning(msg)
                errors.append(msg)
                continue
            except Exception as exc:
                log.warning("provider %s failed: %s", provider.name, exc, exc_info=True)
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue

            if not output.text.strip():
                errors.append(f"{provider.name}: empty translation")
                continue

            # Small models garble proper nouns across scripts: NLLB-600M
            # rendered "Rahul" into Japanese as ラहुल, splicing Devanagari into
            # katakana, even when handed clean English. That output is
            # objectively broken, so treat it as a failure and let the next
            # provider try rather than serving it.
            wrong, why = looks_like_wrong_script(output.text, target.script)
            if wrong:
                log.info("%s produced wrong-script output (%s); falling through",
                         provider.name, why)
                errors.append(f"{provider.name}: {why}")
                rejected.append((provider, output, why))
                continue

            final_source = get_lang(output.detected_source) or source
            detected_by = resolution.method
            if output.detected_source and src_arg is None:
                detected_by = f"{provider.name}-detect"

            if errors:
                notes.append(f"fell back past {len(errors)} provider(s)")

            return TranslationResult(
                text=output.text,
                source=final_source.code if final_source else "auto",
                target=target.code,
                provider=provider.name,
                tier=provider.tier,
                romanized_input=resolution.info.romanized or output.romanized,
                native_input=native_text,
                detected_by=detected_by,
                confidence=resolution.confidence,
                took_ms=int((time.perf_counter() - started) * 1000),
                notes=notes,
            )

        if rejected:
            # Everyone produced wrong-script output. Serve the first one, but
            # say so, rather than failing the request outright.
            provider, output, why = rejected[0]
            final_source = get_lang(output.detected_source) or source
            notes.append(f"best available, but {why}")
            return TranslationResult(
                text=output.text,
                source=final_source.code if final_source else "auto",
                target=target.code,
                provider=provider.name,
                tier=provider.tier,
                romanized_input=resolution.info.romanized or output.romanized,
                native_input=native_text,
                detected_by=resolution.method,
                confidence=resolution.confidence,
                took_ms=int((time.perf_counter() - started) * 1000),
                notes=notes,
            )

        raise ProviderError("router", "every provider failed: " + "; ".join(errors))

    async def _invoke(self, provider: Provider, text: str,
                      source: Language | None, target: Language) -> ProviderOutput:
        coro = provider.translate(text, source, target)
        if provider.tier == Tier.LOCAL and self.local_timeout:
            return await asyncio.wait_for(coro, timeout=self.local_timeout)
        return await coro

    def _candidates(self, req: TranslationRequest, source: Language | None,
                    target: Language) -> list[Provider]:
        if req.prefer:
            chosen = [p for p in self.providers if p.name == req.prefer]
            if not chosen:
                raise ProviderError("router", f"unknown provider {req.prefer!r}")
            return chosen

        out: list[Provider] = []
        for provider in self.providers:
            if provider.tier > req.max_tier:
                continue
            if not provider.available():
                continue
            # Providers that cannot detect still get a shot when we resolved a
            # source ourselves, which is the usual case.
            if not provider.supports(source, target) and not provider.supports(None, target):
                continue
            out.append(provider)
        return out

    # --- introspection -----------------------------------------------------
    def describe(self) -> list[dict]:
        rows = []
        for provider in self.providers:
            row = provider.describe()
            status = getattr(provider, "status", None)
            if callable(status):
                try:
                    row["status"] = status()
                except Exception as exc:
                    row["status"] = {"error": str(exc)}
            rows.append(row)
        return rows

    async def close(self) -> None:
        for provider in self.providers:
            try:
                await provider.close()
            except Exception:
                log.warning("error closing %s", provider.name, exc_info=True)
        for tl in self.transliterators:
            try:
                await tl.close()
            except Exception:
                log.warning("error closing %s", tl.name, exc_info=True)
