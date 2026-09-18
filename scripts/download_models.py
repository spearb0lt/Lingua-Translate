"""Pre-download model weights so the first request is fast, not a download.

    .venv/Scripts/python.exe scripts/download_models.py              # translation
    .venv/Scripts/python.exe scripts/download_models.py --whisper small
    .venv/Scripts/python.exe scripts/download_models.py --only indictrans2

Sizes, so you can decide before committing disk:

    indictrans2 en->indic dist 200M    ~0.9 GB   (gated)
    indictrans2 indic->en dist 200M    ~0.9 GB   (gated)
    indictrans2 indic->indic dist 320M ~1.3 GB   (gated)
    nllb-200-distilled-600M            ~2.5 GB
    indic-conformer-600m               ~2.4 GB   (gated) speech, 22 languages
    whisper base / small / medium      ~0.15 / 0.5 / 1.5 GB
    mms-tts voices                     ~0.15 GB each, only the ones you ask for
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lingua.providers.indictrans2 import ALL_REPOS  # noqa: E402
from lingua.providers.nllb import DEFAULT_REPO as NLLB_REPO  # noqa: E402


def human(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def fetch_seq2seq(repo: str, trust_remote_code: bool) -> bool:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"\n--- {repo}")
    started = time.perf_counter()
    try:
        AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        AutoModelForSeq2SeqLM.from_pretrained(repo, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"    ready in {human(time.perf_counter() - started)}")
    return True


def fetch_conformer(repo: str) -> bool:
    print(f"\n--- {repo}")
    started = time.perf_counter()
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained(repo, trust_remote_code=True)
    except Exception as exc:
        text = f"{type(exc).__name__} {exc}".lower()
        if "gated" in text or "401" in text or "restricted" in text:
            print("    GATED: accept the terms at "
                  f"https://huggingface.co/{repo} and set HF_TOKEN")
        else:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"    ready in {human(time.perf_counter() - started)}")
    return True


def fetch_mms(codes: list[str]) -> list[str]:
    """MMS-TTS voices. Small, so grabbing several is cheap."""
    from server.tts_engines import MMS_REPOS
    failed = []
    for code in codes:
        repo = f"facebook/mms-tts-{MMS_REPOS[code]}"
        print(f"\n--- {repo}")
        started = time.perf_counter()
        try:
            from transformers import AutoTokenizer, VitsModel
            AutoTokenizer.from_pretrained(repo)
            VitsModel.from_pretrained(repo)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append(repo)
            continue
        print(f"    ready in {human(time.perf_counter() - started)}")
    return failed


def fetch_whisper(size: str) -> bool:
    print(f"\n--- faster-whisper: {size}")
    started = time.perf_counter()
    try:
        from faster_whisper import WhisperModel
        WhisperModel(size, device="cpu", compute_type="int8")
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"    ready in {human(time.perf_counter() - started)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",
                    choices=["indictrans2", "nllb", "whisper", "conformer", "mms"],
                    help="download just one family")
    ap.add_argument("--whisper", default="small",
                    help="Whisper size: tiny, base, small, medium, large-v3")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--mms", default=None,
                    help="comma-separated MMS voice languages "
                         "(default: as,or,mai,pa -- the ones no other engine speaks)")
    args = ap.parse_args()

    want = args.only
    failures: list[str] = []

    if want in (None, "indictrans2"):
        print("=" * 64)
        print("IndicTrans2 (AI4Bharat, MIT) -- English + 22 Indian languages")
        print("=" * 64)
        for repo in ALL_REPOS:
            if not fetch_seq2seq(repo, trust_remote_code=True):
                failures.append(repo)

    if want in (None, "nllb"):
        print()
        print("=" * 64)
        print("NLLB-200 (Meta, CC-BY-NC -- non-commercial) -- everything else")
        print("=" * 64)
        if not fetch_seq2seq(NLLB_REPO, trust_remote_code=False):
            failures.append(NLLB_REPO)

    if want in (None, "conformer"):
        print()
        print("=" * 64)
        print("IndicConformer (AI4Bharat, MIT) -- speech for all 22 languages")
        print("=" * 64)
        if not fetch_conformer("ai4bharat/indic-conformer-600m-multilingual"):
            failures.append("indic-conformer")

    if want in (None, "mms"):
        print()
        print("=" * 64)
        print("MMS-TTS -- offline voices, and the only voice for some languages")
        print("=" * 64)
        from server.tts_engines import MMS_REPOS
        codes = ([c.strip() for c in args.mms.split(",") if c.strip() in MMS_REPOS]
                 if args.mms else ["as", "or", "mai", "pa"])
        failures.extend(fetch_mms(codes))

    if want in (None, "whisper") and not args.skip_whisper:
        print()
        print("=" * 64)
        print("Whisper via faster-whisper -- speech recognition")
        print("=" * 64)
        if not fetch_whisper(args.whisper):
            failures.append(f"whisper:{args.whisper}")

    print()
    if failures:
        print(f"{len(failures)} download(s) failed:")
        for name in failures:
            print(f"  - {name}")
        print("\nThe app still runs; it falls back to the network providers.")
        return 1
    print("All requested models are downloaded and cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
