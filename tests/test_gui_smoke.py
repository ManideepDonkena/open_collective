"""
Headless smoke test for the GUI.

Runs under Qt's 'offscreen' platform so it needs no display. It builds the main
window, cycles through EVERY model in the GUI, steps each one, turns on the
trails / cones / links overlays, refreshes the measurement read-out, and finally
grabs a screenshot of the canvas to results/gui_smoke.png.

Skips cleanly (exit 0) if PySide6 is not installed.

    python tests/test_gui_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    from PySide6 import QtWidgets  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"SKIP: PySide6 not available ({e})")
    raise SystemExit(0)

from gui.main_window import GUI_MODELS, MainWindow


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    app.processEvents()

    # enable every overlay so their draw paths are exercised
    for cb in (win.cb_trails, win.cb_cones, win.cb_links, win.cb_groups):
        cb.setChecked(True)

    n_models = win.model_cb.count()
    for i in range(n_models):
        name = win.model_cb.itemText(i)
        win.model_cb.setCurrentIndex(i)          # triggers rebuild + reset
        x0 = win.state.positions.copy()
        for _ in range(30):
            win._tick()
        x1 = win.state.positions
        assert np.all(np.isfinite(x1)), f"{name}: non-finite positions"
        moved = not np.allclose(x0, x1)
        n_params = len(win.param_widgets)
        print(f"  PASS  {name:24s} stepped 30x, moved={moved}, "
              f"live params={n_params}")
        app.processEvents()

    # live parameter poke on the last model (must not raise)
    for pname, (w, kind) in list(win.param_widgets.items())[:3]:
        if kind == "float":
            w.set_value(w.value())     # re-emit
        win._apply_params()

    # screenshot the canvas
    out = Path("results"); out.mkdir(exist_ok=True)
    png = out / "gui_smoke.png"
    pix = win.canvas.grab()
    ok = pix.save(str(png))
    assert ok and png.exists() and png.stat().st_size > 0, "screenshot failed"
    print(f"\n  screenshot -> {png}  ({png.stat().st_size} bytes, "
          f"{pix.width()}x{pix.height()})")
    print(f"\n{n_models}/{n_models} GUI models stepped cleanly.")


if __name__ == "__main__":
    main()
