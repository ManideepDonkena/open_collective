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

    _check_experiment_manager(win, out)


def _check_experiment_manager(win, out):
    """Config round-trip + reproducibility, recording, exports, GIF."""
    from experiments import manager

    print("\n--- experiment manager integration ---")
    win.model_cb.setCurrentText("Vicsek")
    win.boundary_cb.setCurrentText("periodic")
    win.N_sb.setValue(80)

    # 1. config save -> load -> apply must reproduce the same config
    cfg = win.current_config()
    cpath = manager.save_config(cfg, out / "gui_config.json")
    win.model_cb.setCurrentText("Boids")            # perturb the GUI
    win.apply_config(manager.load_config(cpath))    # ...then restore from file
    assert win.current_config() == cfg, "config round-trip changed the config"
    print(f"  PASS  config round-trip identical -> {cpath}")

    # 2. the saved config is runnable headless (reproducibility guarantee)
    final, hist = manager.run_experiment(cfg)
    assert "polar_order" in hist and len(hist["t"]) > 0
    print(f"  PASS  saved config runs via manager.run_experiment "
          f"({len(hist['t'])} frames)")

    # 3. record a live run, then export metrics + trajectory
    win.rec_btn.setChecked(True)
    for _ in range(25):
        win._tick()
    win.rec_btn.setChecked(False)
    h = win._history_dict()
    assert h is not None and len(h["t"]) >= 25 and "trajectory" in h
    mpath = manager.export_measurements(h, out / "gui_metrics.csv")
    tpath = manager.export_trajectory(h, out / "gui_traj.csv")
    assert mpath.stat().st_size > 0 and tpath.stat().st_size > 0
    print(f"  PASS  recorded {len(h['t'])} frames -> {mpath.name}, {tpath.name}")

    # 4. GIF capture (skips gracefully if Pillow is absent)
    ok, msg = win._do_save_gif(out / "gui_anim.gif", frames=8, fps=20)
    gif = out / "gui_anim.gif"
    if ok:
        assert gif.exists() and gif.stat().st_size > 0
        print(f"  PASS  GIF written -> {gif.name} ({gif.stat().st_size} bytes)")
    else:
        print(f"  SKIP  GIF ({msg})")

    print("\nexperiment manager integration OK.")


if __name__ == "__main__":
    main()
