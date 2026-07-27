"""
Interactive GUI for open-collective (BirdFlockLab modules 4 & 5).

PySide6 front end that drives the existing headless engine (core + models) live:
real-time parameter control, play/pause/step/reset, zoom/pan, and on-canvas
vision cones / neighbour links / trajectories, with the measurement module read
out live every frame.

    python run_gui.py         # from the repo root
    python -m gui             # equivalent

The GUI adds nothing to the physics -- every step goes through
`CollectiveModel.step` and every number through `core.metrics`, so what you see is
exactly what the batch experiments compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `core` / `models` importable however the GUI is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
