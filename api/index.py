"""Vercel serverless entry point for the lite edition.

Vercel requires functions to live in /api at the repository root, so this is a
three-line shim; the actual app is in lite/main.py. Deploy with the project
root set to the repository root (not lite/), so that both this file and the
lingua package are included in the build.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite.main import app  # noqa: E402,F401  (Vercel looks for `app`)
