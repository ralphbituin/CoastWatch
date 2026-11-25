"""Pytest configuration for the ML package.

Run from the repository root:

    ml/.venv/Scripts/python.exe -m pytest ml/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
