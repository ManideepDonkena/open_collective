#!/usr/bin/env python3
"""Launch the interactive GUI:  python run_gui.py

Requires PySide6 (pip install PySide6) in addition to the core numpy/scipy stack.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
