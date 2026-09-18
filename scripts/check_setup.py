"""Check this machine can actually run everything, and say what to fix.

    .venv/Scripts/python.exe scripts/check_setup.py

Exits non-zero if anything required is missing. Gated models are reported as
warnings, not failures, because the app still runs without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OK, WARN, BAD = "\u2713", "!", "\u2717"
problems: list[str] = []
warnings: list[str] = []


def line(mark: str, label: str, detail: str = "") -> None:
    print(f"  {mark} {label}" + (f"  -- {detail}" if detail else ""))


def check_imports() -> None:
    print("\nPython packages")
    required = [("httpx", "network providers"), ("fastapi", "the server")]
    optional = [
        ("curl_cffi", "Google's keyless endpoint (blocked without it)"),
        ("edge_tts", "neural voices"),
        ("indic_transliteration", "offline romanized input"),
        ("torch", "local models"),
        ("transformers", "local models"),
        ("IndicTransToolkit", "IndicTrans2 preprocessing"),
        ("faster_whisper", "speech to text"),
        ("torchaudio", "IndicConformer audio loading"),
    ]
    for mod, why in required:
        try:
            __import__(mod)
            line(OK, mod)
        except ImportError:
            line(BAD, mod, f"required for {why}")
            problems.append(f"pip install {mod}")
    for mod, why in optional:
        try:
            __import__(mod)
            line(OK, mod)
        except ImportError as exc:
            line(WARN, mod, f"{why} unavailable ({exc.__class__.__name__})")
            warnings.append(f"{mod} missing: {why} will not work")


def check_versions() -> None:
    print("\nVersion compatibility")
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            line(BAD, f"transformers {transformers.__version__}",
                 "5.x breaks IndicTrans2; pin transformers<5")
            problems.append('pip install "transformers>=4.44,<5"')
        else:
            line(OK, f"transformers {transformers.__version__}")
    except ImportError:
        line(WARN, "transformers", "not installed")


def check_device() -> None:
    print("\nCompute")
    try:
        import torch
    except ImportError:
        line(WARN, "torch", "not installed; local models disabled")
        return
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        line(OK, torch.cuda.get_device_name(0),
             f"{free / 1024**3:.1f} GB free of {total / 1024**3:.1f} GB")
        if free / 1024**3 < 1.5:
            warnings.append("little free VRAM; models will fall back to CPU")
    else:
        line(WARN, "no CUDA device", "local models will run on CPU (slower)")


def check_gated() -> None:
    print("\nGated Hugging Face models")
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
             or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    if token:
        line(OK, "HF_TOKEN is set")
    else:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
        if token:
            line(OK, "logged in via huggingface-cli")
        else:
            line(WARN, "no Hugging Face token found")

    repos = [
        ("ai4bharat/indictrans2-en-indic-dist-200M", "IndicTrans2 (best Indic translation)"),
        ("ai4bharat/indictrans2-indic-en-dist-200M", "IndicTrans2"),
        ("ai4bharat/indictrans2-indic-indic-dist-320M", "IndicTrans2"),
        ("ai4bharat/indic-conformer-600m-multilingual", "IndicConformer (8 extra languages for speech)"),
    ]
    try:
        import httpx
    except ImportError:
        return
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    blocked = []
    for repo, why in repos:
        try:
            # Ask for an actual file, not the metadata API: the API answers
            # 200 for gated repos to anyone, and only file downloads 401.
            r = httpx.head(f"https://huggingface.co/{repo}/resolve/main/config.json",
                           headers=headers, timeout=20, follow_redirects=True)
            if r.status_code == 200:
                line(OK, repo, "downloadable")
            elif r.status_code in (401, 403):
                line(WARN, repo, f"gated, no access -- {why}")
                blocked.append(repo)
            else:
                line(WARN, repo, f"HTTP {r.status_code} -- {why}")
                blocked.append(repo)
        except Exception as exc:
            line(WARN, repo, str(exc)[:60])
    if blocked:
        warnings.append(
            "gated models not accessible. Accept the terms while signed in at:\n"
            + "".join(f"      https://huggingface.co/{r}\n" for r in blocked)
            + "      then create a read token at "
              "https://huggingface.co/settings/tokens and set HF_TOKEN.")


def main() -> int:
    print("=" * 68)
    print("Lingua setup check")
    print("=" * 68)
    check_imports()
    check_versions()
    check_device()
    check_gated()

    print()
    print("=" * 68)
    if problems:
        print("MUST FIX:")
        for item in problems:
            print(f"  - {item}")
    if warnings:
        print("OPTIONAL:")
        for item in warnings:
            print(f"  - {item}")
    if not problems and not warnings:
        print("Everything is ready.")
    elif not problems:
        print("\nThe app will run. The items above only limit optional features.")
    print("=" * 68)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
