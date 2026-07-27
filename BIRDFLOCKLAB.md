# BirdFlockLab — Running Documentation & Update Log

Living document for turning the **open-collective** research codebase into the
**BirdFlockLab** interactive research-simulation platform. It tracks what exists,
how to run it, and every change as it lands.

- **Base project:** `open-collective` — a headless, research-grade collective-motion
  library whose central thesis is that *boundary conditions are a first-class,
  swappable object* (periodic vs open vs reflecting). See [README.md](README.md),
  [RESEARCH.md](RESEARCH.md).
- **Target:** BirdFlockLab v1.0 — modular, interactive platform for creating,
  visualizing, analyzing, and comparing collective-motion models.
- **Approach:** build the platform layer *on top of* the existing engine without
  rewriting the science core. The engine, model library, and measurements were
  already at or beyond spec; the work is the app layer (init, manager, GUI).

Last updated: **2026-07-27**.

---

## 1. Status against the BirdFlockLab v1.0 spec

Legend: ✅ done · 🟡 partial · ⬜ not started

| # | Spec module | State | Notes |
|---|---|---|---|
| 1 | Simulation Engine | ✅ | `State` + `CollectiveModel` + `run()`; periodic/open/reflecting boundary; metric/topological/vision-cone neighbours. Array-based (not per-`Bird` objects) — a perf choice, not a gap. |
| 2 | Model Library | ✅ | 10 free-running models incl. **Kuramoto** (added). Custom models = subclass `CollectiveModel`. Missing named model: **"Mind-Flock"** (dynamics TBD from user). |
| 3 | Initialization | ✅ | `core/init.py`: random / cluster / ring / grid / manual / CSV. Group-aware. |
| 4 | Parameter Control | ✅ | GUI exposes a **live slider per numeric constructor arg** of the selected model (introspected). Structural params (N, boundary, init) rebuild; others apply live. |
| 5 | Visualization | 🟡 | Interactive 2D canvas: play/pause/step/reset, zoom/pan, trails, vision cones, neighbour links, group colours, live metrics. **3D interactive** not yet (offline 3D exists in `core/viz3d.py`). |
| 6 | Measurement | ✅ | polarization, milling, angular momentum, cluster/fragment count, nn-distance, **density**, **heading_entropy**, **mean_speed**, + group-maintenance + consensus observables. |
| 7 | Experiment Manager | ✅ | `experiments/manager.py`: registry, JSON/YAML config save/load, CSV/HDF5 export, parameter sweep — **now wired into the GUI** (Save/Load config, Record, Export metrics/trajectory, Screenshot, Save GIF). A config saved from the GUI re-runs headless via `manager.run_experiment`. |

### Technology stack

| Piece | Spec | Here | Status |
|---|---|---|---|
| Language | Python | Python 3.12 | ✅ |
| Numerics | NumPy | NumPy | ✅ |
| GUI | PySide6 (Qt) | PySide6 6.11 | ✅ |
| Canvas | PyQtGraph/VisPy | QPainter (self-contained) | ✅ (deliberate substitution) |
| Data | Pandas + HDF5 | CSV (stdlib) + optional h5py | 🟡 (no Pandas dep; HDF5 optional) |
| Config | YAML/JSON | JSON (stdlib) + optional PyYAML | ✅ |
| Perf | Numba | pure NumPy | ⬜ (fine to a few hundred agents) |

---

## 2. Setup & run

> For a beginner-friendly, cross-platform walkthrough (Linux/macOS/Windows), see
> **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**. The notes below are the
> quick version and mention specifics seen on the original dev machine.

The system Python on this machine (Ubuntu 24.04) is PEP-668 "externally managed"
and lacks scipy/matplotlib — **use a virtualenv**.

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # numpy scipy matplotlib pillow
.venv/bin/pip install PySide6                  # for the GUI only
```

Headless engine, experiments, tests:

```bash
.venv/bin/python tests/test_theorems.py             # 13/13 theorem checks
.venv/bin/python experiments/exp1_boundary_collapse.py
.venv/bin/python experiments/demo_new_features.py   # init/metrics/Kuramoto/manager demo
```

Interactive GUI:

```bash
.venv/bin/python run_gui.py                         # or: python -m gui
.venv/bin/python tests/test_gui_smoke.py            # headless offscreen check + screenshot
```

> **Ubuntu 24.04 / Qt 6.5+ gotcha:** the live GUI needs a system library that is
> not installed by default. If `run_gui.py` fails with *"Could not load the Qt
> platform plugin xcb ... libxcb-cursor0 is needed"*, run:
> ```bash
> sudo apt update && sudo apt install -y libxcb-cursor0
> ```
> (If more xcb libs are still reported: `libxcb-xinerama0 libxkbcommon-x11-0`.)
> The GUI itself is fine — `tests/test_gui_smoke.py` renders it via the
> `offscreen` backend with no display and no extra system libs.

---

## 3. New modules — quick reference

### `core/init.py` — initialization (module 3)
```python
import core.init as cinit
from core import make_boundary
b = make_boundary("open", dim=2)
cinit.random_init(120, b, speed=0.5, n_groups=2)
cinit.cluster_init(120, b, n_clusters=3, aligned_within_cluster=True)
cinit.ring_init(120, b, radius=4.0, tangential=True)      # seeds a mill
cinit.grid_init(120, b, spacing=1.0)
cinit.manual_init(positions, velocities=None, groups=None) # GUI click-to-place hook
cinit.from_csv("init.csv"); cinit.state_to_csv(state, "snap.csv")
```
Groups live in `state.internal["groups"]` — the one convention shared with the
metrics, the 3D viewer, and `MultiGroupFlock`.

### `core/metrics.py` — new observables (module 6)
`mean_speed(velocities)`, `density(positions, boundary)`,
`heading_entropy(headings, bins=16)`. All three are in `summarize(...)` and in
the `run()` time-series history.

### `models` — Kuramoto (module 2)
`KuramotoModel(boundary, r_max=1.0, K=1.0, eta=0.0, v0=0.5, omega_spread=0.0,
topological=False, k=7)`. 2D phase oscillators on the interaction graph; heading
angle *is* the phase, so the sync order parameter equals `polar_order`.
Alignment-only (`cohesive=False`) → disperses in open space like Vicsek.

### `experiments/manager.py` — experiment manager (module 7)
```python
from experiments import manager
cfg = {
  "seed": 1,
  "boundary": {"kind": "periodic", "L": 10.0, "dim": 2},
  "init":  {"method": "random", "n": 120, "speed": 0.5},
  "model": {"name": "Vicsek", "params": {"r_max": 1.0, "eta": 0.3, "v0": 0.5}},
  "run":   {"steps": 300, "dt": 0.05, "r_link": 1.0, "record_every": 20,
            "record_traj": True},
}
final, hist = manager.run_experiment(cfg)
manager.save_config(cfg, "results/cfg.json")           # or .yaml (needs PyYAML)
manager.export_measurements(hist, "results/meas.csv")
manager.export_trajectory(hist, "results/traj.csv")     # or .h5 (needs h5py)
manager.parameter_sweep(cfg, "model.params.eta", [0.0, 0.3, 0.6, 1.0])
```
Registry keys: `Vicsek, Perception, SlowFast, Kuramoto, Boids, Couzin, DOrsogna,
CuckerSmale, OlfatiSaber, MultiGroupFlock` (+ formation models).

### `gui/` — interactive front end (modules 4 & 5)
`run_gui.py` → `gui.app.main()` → `MainWindow` (controls + QTimer loop) hosting
`SimCanvas` (QPainter renderer: agents, heading arrows, trails, vision cones,
neighbour links, group colours; wheel-zoom, drag-pan). 10 models selectable,
each with introspected live sliders. Overlays and live metrics toggle in-panel.

---

## 4. Update log

Newest first. Append an entry per increment.

### 2026-07-27 — Manual initial conditions in the GUI (spec modules 3 & 5)
**Added**
- `gui/canvas.py` — **place mode**: a left-click adds a bird at that world point;
  a left-drag still pans (distinguished by a 4px move threshold).
- `gui/main_window.py` — **Initial condition** panel: *Place mode*, *place as
  group* (0–7, colours the placed birds), *Clear birds*, *Load CSV…*. New birds
  are placed live; the model is rebuilt to drop stale per-N caches. Metrics guard
  against `N < 2` so mid-placement never crashes `summarize`.
- `core.init.from_csv` is now reachable from the GUI (Load CSV…).

**Verified** (`tests/test_gui_smoke.py`, extended)
- Clear → place 5 birds via the canvas callback → steps cleanly (N stays 5).
- Load 30 birds from a CSV written by `core.init` → steps cleanly.
- Visual check: 80 birds placed in two groups render blue/orange with headings
  (`results/gui_placed.png`).

### 2026-07-27 — Experiment Manager wired into the GUI (spec module 7)
**Added (in `gui/main_window.py`)**
- **Experiment** panel: Save config… / Load config… / Screenshot… / ● Record /
  Export metrics… / Export trajectory… / Save GIF….
- `current_config()` / `apply_config()` — the GUI setup ⇄ a manager-compatible
  config dict. A config saved from the GUI **re-runs headless** via
  `manager.run_experiment` (reproducibility), using `GUI_TO_KEY`/`KEY_TO_GUI`.
- Per-frame recording buffers (metrics + trajectory) consumed by
  `manager.export_measurements` (CSV) and `manager.export_trajectory` (CSV/HDF5).
- GIF capture via QPainter grab → Pillow (skips cleanly if Pillow absent).

**Verified** (`tests/test_gui_smoke.py`, extended)
- Config save → load → apply is **identical** to the original config.
- The saved config runs via `manager.run_experiment` (500 frames).
- Recording 26 frames then exporting metrics + trajectory writes non-empty CSVs.
- GIF written & re-read: 8 frames, 532×682 → `results/gui_anim.gif`.

### 2026-07-27 — Interactive GUI (spec modules 4 & 5)
**Added**
- `gui/canvas.py` — `SimCanvas`, a QPainter renderer (agents + heading arrows,
  fading trajectory trails, vision cones, neighbour links, group colours,
  dashed/solid boundary box; mouse-wheel zoom, drag pan).
- `gui/main_window.py` — `MainWindow`: setup controls (model/boundary/init/N/
  groups/speed/dt), transport (Play/Pause/Step/Reset), display toggles, live
  measurement read-out, and a **QTimer** run loop. Parameter panel is built by
  **introspecting the model's `__init__`** — one live control per numeric arg.
- `gui/app.py`, `gui/__main__.py`, `run_gui.py` — entry points.
- `tests/test_gui_smoke.py` — headless (`offscreen`) test.

**Verified**
- 10/10 GUI models step 30× cleanly with all overlays on; positions finite.
- Screenshots rendered & inspected: Multi-group flock (two segregating colours +
  trails), Perception (vision cones along heading + dashed walls), Vicsek
  (metric neighbour graph). → `results/gui_smoke.png`,
  `results/gui_perception_cones.png`, `results/gui_vicsek_links.png`.

**Known issue (environment, not code)**
- Live window needs `libxcb-cursor0` on Ubuntu 24.04 (see §2). Headless render
  works without it.

### 2026-07-27 — Science gaps (spec modules 2, 3, 6, 7)
**Added**
- `core/init.py` — random / cluster / ring / grid / manual / CSV initializers.
- `core/metrics.py` — `mean_speed`, `density`, `heading_entropy`; wired into
  `summarize()`.
- `models/alignment.py` — `KuramotoModel`; registered in `models/__init__.py`.
- `experiments/manager.py` — registry, JSON/YAML config I/O, CSV/HDF5 export,
  `parameter_sweep`.
- `experiments/demo_new_features.py` — end-to-end demo.

**Changed**
- `core/base.py::run()` now records `density`, `heading_entropy`, `mean_speed`
  time-series and passes `velocities` to `summarize` (backward compatible).
- `requirements.txt`, `README.md` — document optional deps and new modules.

**Verified**
- Demo runs: Kuramoto syncs (M 0.07→0.97), density exact (N/L²), config round-trip
  identical, exports written, eta-sweep reproduces the Vicsek order–disorder
  transition with `heading_entropy` tracking it inversely.
- **13/13 theorem tests still pass** — no regression.

### Baseline (prior work, before 2026-07-27)
- Engine (`core/`), model zoo (`models/`: alignment, cohesive, grouping,
  formation, consensus), group + cohesion diagnostics, offline 3D viewer
  (`core/viz3d.py`), UAV multi-team sim (`Uav/`), 13 theorem tests, and the
  boundary-collapse experiments/write-ups. ~5,200 LOC.

---

## 5. Roadmap / open items

- [ ] **"Mind-Flock" model** — needs the dynamics/definition from the user
      (closest existing model is `PerceptionQuantum`).
- [x] **Wire Experiment Manager into the GUI** — done 2026-07-27 (Save/Load
      config, Export metrics/trajectory CSV·HDF5, Screenshot, Save GIF).
      Remaining nice-to-have: MP4 export (needs ffmpeg/imageio).
- [ ] **3D interactive canvas** — the offline `viz3d.py` is 3D; the live canvas
      is 2D. Options: VisPy/OpenGL, or a 2.5D projection in the QPainter canvas.
- [ ] **Numba / vectorization** — for >~1000 agents at interactive FPS; several
      model `step()` loops are per-agent Python loops.
- [x] **CSV/manual init in the GUI** — done 2026-07-27 (Place mode, place-as-
      group, Clear, Load CSV…).
- [ ] **Pandas** for analysis convenience (currently CSV/NumPy only).

---

## 6. Verification snapshot (2026-07-27)

| Check | Command | Result |
|---|---|---|
| Theorem suite | `tests/test_theorems.py` | 13/13 pass |
| New features | `experiments/demo_new_features.py` | all sections pass |
| GUI (headless) | `tests/test_gui_smoke.py` | 10/10 models, screenshot, config round-trip, CSV/HDF5 export, GIF, click-to-place, CSV load |
| Live GUI | `run_gui.py` | pending `libxcb-cursor0` install on user's machine |
