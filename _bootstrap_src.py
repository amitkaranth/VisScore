"""
Runtime helper to import the local `src/` package without installation.

Entry points (Streamlit / CLI) can do:
  import _bootstrap_src; _bootstrap_src.bootstrap()
before importing `viscore`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    repo_root = Path(__file__).resolve().parent
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

