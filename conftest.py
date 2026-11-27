"""Pytest configuration for Adike Dhara.

Ensures the ``src/`` package is importable so the documented command
``python -m pytest tests/ -v`` works without setting PYTHONPATH.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
