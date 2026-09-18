"""End-to-end self test.

    .venv/Scripts/python.exe scripts/selftest.py            # everything available
    .venv/Scripts/python.exe scripts/selftest.py --offline  # no network providers
    .venv/Scripts/python.exe scripts/selftest.py --local    # force local models only

Exits non-zero if anything fails, so it doubles as a CI check.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lingua.engine import build_router  # noqa: E402
from lingua.languages import LANGUAGES, get  # noqa: E402
from lingua.script import analyse  # noqa: E402
from lingua.types import Tier, TranslationRequest  # noqa: E402

PASS, FAIL = "✓", "✗"


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  {PASS} {label}")
        else:
            self.failed.append(label)
            print(f"  {FAIL} {label}")
        if detail:
            print(f"      {detail}")
        return ok

    def section(self, title: str) -> None:
        print()
        print("=" * 72)
        print(title)
        print("=" * 72)


def test_registry(r: Report) -> None:
    r.section("1. Language registry (offline)")
    r.check(len(LANGUAGES) > 55, f"{len(LANGUAGES)} languages registered")
    indian = [l for l in LANGUAGES if l.indic]
    r.check(len(indian) == 22, f"all 22 scheduled Indian languages present "
                               f"(found {len(indian)})")
    r.check(get("hin_Deva") is not None and get("hin_Deva").code == "hi",
            "FLORES tag hin_Deva resolves to Hindi")
    r.check(get("hi-Latn") is not None and get("hi-Latn").code == "hi",
            "Google's hi-Latn resolves to Hindi")
    dupes = {l.flores for l in LANGUAGES if l.flores}
    r.check(len(dupes) == len([l for l in LANGUAGES if l.flores]),
            "no duplicate FLORES tags")


def test_shared_scripts(r: Report) -> None:
    """Languages that share a writing system must not collapse into one."""
    r.section("2b. Shared-script disambiguation (offline)")
    print("  Devanagari alone carries 8 of our languages; Bengali and Arabic")
    print("  carry several each. Guessing the most common one changes meaning:")
    print("  Marathi 'माझे नाव राहुल आहे' read as Hindi becomes 'My boat is Rahul'.")
    print()
    cases = [
        ("hi",  "मेरा नाम राहुल है और मैं दिल्ली से हूँ"),
        ("mr",  "माझे नाव राहुल आहे आणि मी मुंबईचा आहे"),
        ("ne",  "मेरो नाम राहुल हो र म काठमाडौंबाट हुँ"),
        ("sa",  "मम नाम राहुलः अस्ति"),
        ("mai", "हमर नाम राहुल अछि"),
        ("brx", "आंनि मुं राहुल"),
        ("bn",  "আমার নাম রাহুল এবং আমি দিল্লি থেকে এসেছি"),
        ("as",  "মোৰ নাম ৰাহুল আৰু মই দিল্লীৰ পৰা আহিছো"),
        ("ur",  "میرا نام راہول ہے اور میں دہلی سے ہوں"),
        ("sd",  "منهنجو نالو راهول آهي"),
    ]
    for truth, text in cases:
        info = analyse(text)
        r.check(info.guess == truth,
                f"{truth:<4} {text[:28]!r:<32} -> {info.guess} ({info.confidence})")


def test_script_detection(r: Report) -> None:
    r.section("2. Script and romanization detection (offline, no network)")
    cases = [
        ("mera naam rahul hai", "Latin", True, "hi"),
        ("मेरा नाम राहुल", "Devanagari", False, "hi"),
        ("The quick brown fox jumps over the lazy dog", "Latin", False, "en"),
        ("ami tomake bhalobashi ebong tumi amar bondhu", "Latin", True, "bn"),
        ("naan unnai kaadhalikkiren romba nalla iruku", "Latin", True, "ta"),
        ("Bonjour, comment allez-vous", "Latin", False, "en"),
        ("こんにちは世界", "Japanese", False, "ja"),
        ("السلام عليكم", "Arabic", False, "ur"),
    ]
    for text, script, romanized, guess in cases:
        info = analyse(text)
        ok = info.script == script and info.romanized == romanized and info.guess == guess
        r.check(ok, f"{text[:34]!r:<38} -> {info.script}/{info.romanized}/{info.guess}",
                "" if ok else f"expected {script}/{romanized}/{guess}")


async def test_translation(r: Report, router, offline: bool) -> None:
    r.section("3. Translation routing")
    print(f"  providers: {[(p.name, p.tier.name) for p in router.providers]}")
    print(f"  transliterators: {[t.name for t in router.transliterators]}")
    print()

    # (text, source, target, must_not_equal_input, note)
    cases = [
        ("mera naam rahul hai aur main dilli se hoon", "auto", "en", True,
         "romanized Hindi, auto-detected"),
        ("mera naam rahul hai", "hi", "ta", True, "romanized Hindi -> Tamil"),
        ("amar nam gobindo", "as", "en", True, "the Assamese example"),
        ("ami tomake bhalobashi", "auto", "en", True,
         "REGRESSION: romanized Bengali must not pass through untranslated"),
        ("naan unnai kaadhalikkiren", "ta", "hi", True,
         "REGRESSION: romanized Tamil must translate, not transliterate"),
        ("मेरा नाम राहुल है", "auto", "fr", True,
         "native Devanagari -> French"),
        ("Good morning, how are you today?", "en", "hi", True, "English -> Hindi"),
        ("Good morning, how are you today?", "en", "ja", True, "English -> Japanese"),
        ("मेरा नाम राहुल है और मैं दिल्ली से हूँ", "hi", "ja", True,
         "REGRESSION: no Devanagari may leak into Japanese output"),
        ("माझे नाव राहुल आहे आणि मी मुंबईचा आहे", "auto", "en", True,
         "REGRESSION: Marathi must not be read as Hindi ('My boat is Rahul')"),
    ]

    for text, sl, tl, must_change, note in cases:
        try:
            res = await router.translate(
                TranslationRequest(text=text, source=sl, target=tl))
        except Exception as exc:
            r.check(False, f"{sl:>4}->{tl}  {note}", f"{type(exc).__name__}: {exc}")
            continue

        changed = res.text.strip().lower() != text.strip().lower()
        ok = bool(res.text.strip()) and (changed or not must_change)
        detail = f"{text[:40]!r} -> {res.text[:56]!r}"
        extra = (f"via {res.provider}/{res.tier.name}, src={res.source}"
                 f" ({res.detected_by}), {res.took_ms}ms")
        # A translation must not contain script that cannot belong to it.
        from lingua.languages import require as _require
        from lingua.script import looks_like_wrong_script
        wrong, why = looks_like_wrong_script(res.text, _require(tl).script)
        if wrong and "best available" not in " ".join(res.notes or []):
            ok = False
            detail += f"  [{why}]"

        # The Marathi case has a specific wrong answer we must never see.
        if "boat" in res.text.lower():
            ok = False
            detail += "  [read as Hindi: नाव translated as 'boat', not 'name']"

        r.check(ok, f"{sl:>4}->{tl}  {note}", detail)
        print(f"      {extra}")
        if res.native_input:
            print(f"      read as: {res.native_input!r}")

    r.section("4. Fallback and tier control")
    try:
        res = await router.translate(TranslationRequest(
            text="mera naam rahul hai", source="hi", target="en",
            max_tier=Tier.LOCAL))
        r.check(res.tier == Tier.LOCAL,
                f"max_tier=LOCAL honoured (served by {res.provider})")
    except Exception as exc:
        r.check(offline is False, "max_tier=LOCAL correctly refuses when no local model",
                f"{type(exc).__name__}: {str(exc)[:100]}")

    # Each provider is probed with a pair it actually claims to handle. A
    # one-size-fits-all pair would unfairly fail pivot_en, which refuses
    # single-hop pairs like en->hi by design.
    probes = [("Good morning", "en", "hi"),
              ("मेरा नाम राहुल है", "hi", "ja"),
              ("मेरा नाम राहुल है", "hi", "en")]

    for provider in router.providers:
        name = provider.name
        chosen = next(((t, s, g) for t, s, g in probes
                       if provider.supports(get(s), get(g))), probes[0])
        text, source, target = chosen
        try:
            res = await router.translate(TranslationRequest(
                text=text, source=source, target=target, prefer=name))
            r.check(res.provider == name and bool(res.text.strip()),
                    f"prefer={name} works ({source}->{target})", f"-> {res.text!r}")
        except Exception as exc:
            message = str(exc)
            # A gated Hub repo is an account setup issue on this machine, not
            # a defect in the code, so report it as a skip with the fix.
            if "gated" in message.lower():
                print(f"  - prefer={name} skipped: model weights are gated on "
                      f"Hugging Face")
                print(f"      accept the terms and set HF_TOKEN; see the README")
                continue
            r.check(False, f"prefer={name} works ({source}->{target})",
                    f"{type(exc).__name__}: {message[:90]}")

    res = await router.translate(TranslationRequest(
        text="Hello world", source="en", target="en"))
    r.check(res.provider == "noop" and res.text == "Hello world",
            "same source and target short-circuits")


async def test_extras(r: Report, router) -> None:
    """Cache, quality scoring and document handling."""
    from lingua.quality import looks_untranslated, round_trip_score
    from server import documents

    r.section("5. Cache")
    router.cache.clear()
    req = lambda **kw: TranslationRequest(text="Good morning, how are you?",
                                          source="en", target="hi", **kw)
    first = await router.translate(req())
    second = await router.translate(req())
    r.check(not first.cached and second.cached,
            "second identical request is served from cache",
            f"{first.took_ms} ms -> {second.took_ms} ms")
    fresh = await router.translate(req(use_cache=False))
    r.check(not fresh.cached, "use_cache=False bypasses the cache")
    stats = router.cache.stats()
    r.check(stats["hits"] >= 1, f"cache reports hits ({stats['hits']})")

    r.section("5b. Approximate results must not poison the cache")
    from lingua.languages import get as _get
    exact = [t for t in router.transliterators
             if not getattr(t, "approximate", False)]
    rough = [t for t in router.transliterators
             if getattr(t, "approximate", False)]
    if exact and rough:
        router.cache.clear()
        # An offline-only run caches an approximate transliteration...
        router.transliterators = rough
        rough_text, rough_by = await router.to_native_script(
            "mera naam rahul hai", _get("hi"))
        # ...and must not be served once an exact transliterator is back.
        router.transliterators = exact + rough
        good_text, good_by = await router.to_native_script(
            "mera naam rahul hai", _get("hi"))
        r.check(good_by != rough_by and good_text != rough_text,
                "a cached approximate transliteration is re-done when an "
                "exact engine is available",
                f"{rough_text!r} via {rough_by} -> {good_text!r} via {good_by}")
        again_text, again_by = await router.to_native_script(
            "mera naam rahul hai", _get("hi"))
        r.check(again_text == good_text and again_by == good_by,
                "the exact transliteration is then cached and reused")
    else:
        print("  - skipped: needs both an exact and an approximate transliterator")

    r.section("6. Quality scoring")
    r.check(round_trip_score("hello world", "hello world") == 1.0,
            "an identical round trip scores 1.0")
    r.check(round_trip_score("My name is Rahul and I am from Delhi",
                             "The weather is nice today") < 0.2,
            "an unrelated round trip scores near zero")
    r.check(looks_untranslated("ami tomake bhalobashi", "ami tomake bhalobashi"),
            "an untranslated passthrough is detected")
    r.check(not looks_untranslated("मेरा नाम राहुल है और मैं दिल्ली से हूँ",
                                   "My name is Rahul and I am from Delhi"),
            "a real translation is not mistaken for a passthrough")

    scored = await router.translate(TranslationRequest(
        text="मेरा नाम राहुल है और मैं दिल्ली से हूँ", source="hi", target="en",
        score=True, use_cache=False))
    r.check(scored.quality is not None and scored.back_translation,
            f"a scored request returns a quality value ({scored.quality})",
            f"back: {scored.back_translation!r} by {scored.graded_by}")

    best = await router.translate(TranslationRequest(
        text="मेरा नाम राहुल है और मैं दिल्ली से हूँ", source="hi", target="en",
        best_of=3, use_cache=False))
    graders = {best.graded_by} | {None}
    r.check(len(best.alternatives) >= 1,
            f"best_of returns alternatives ({len(best.alternatives)})")
    identical = {best.text} | {a["text"] for a in best.alternatives}
    scores = {best.quality} | {a["quality"] for a in best.alternatives}
    r.check(len(identical) > 1 or len(scores) == 1,
            "candidates with identical text get identical scores (one grader)")

    r.section("7. Documents")
    txt = b"Hello world.\n\n    Indented line.\n"
    doc = documents.load(txt, "a.txt")
    r.check(doc.segments == ["Hello world.", "    Indented line."],
            "text: blank lines skipped, content kept")
    rebuilt = doc.rebuild(["Bonjour.", "Ligne."]).decode()
    r.check("\n\n" in rebuilt and rebuilt.splitlines()[2].startswith("    "),
            "text: blank line and indentation survive the rebuild")

    srt = (b"1\n00:00:01,000 --> 00:00:03,000\nGood morning.\n\n"
           b"2\n00:00:03,500 --> 00:00:06,000\nGood morning.\n")
    doc = documents.load(srt, "a.srt")
    unique, mapping = documents.dedupe(doc.segments)
    r.check(len(doc.segments) == 2 and len(unique) == 1,
            "subtitles: repeated cue is translated once")
    out = doc.rebuild(documents.expand(["Bonjour."], mapping)).decode()
    r.check("00:00:01,000 --> 00:00:03,000" in out and out.count("Bonjour.") == 2,
            "subtitles: timestamps untouched, translation reused")

    try:
        documents.load(b"x", "a.xyz")
        r.check(False, "unsupported format is refused")
    except documents.UnsupportedDocument:
        r.check(True, "unsupported format is refused")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="local providers only, no network")
    ap.add_argument("--local", action="store_true",
                    help="alias for --offline")
    args = ap.parse_args()
    offline = args.offline or args.local

    r = Report()
    test_registry(r)
    test_script_detection(r)
    test_shared_scripts(r)

    # Always build the full edition so local models are actually exercised;
    # --offline then strips everything that needs a network.
    router = build_router("full")
    if offline:
        router.providers = [p for p in router.providers if p.tier == Tier.LOCAL]
        router.transliterators = [t for t in router.transliterators if t.offline]
        if not router.providers:
            print("\nNo local providers available. Install requirements-local.txt,")
            print("or drop --offline to test the network engines.")
            return 1

    try:
        await test_translation(r, router, offline)
        if not offline:
            await test_extras(r, router)
    finally:
        await router.close()

    print()
    print("=" * 72)
    print(f"{r.passed} passed, {len(r.failed)} failed")
    for name in r.failed:
        print(f"  {FAIL} {name}")
    print("=" * 72)
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
