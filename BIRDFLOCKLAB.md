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
| 2 | Model Library | ✅ | **17 free-running models in the GUI, 21 in the registry** — Vicsek (+vectorial noise), Kuramoto, Inertial-spin, Boids, Couzin, Cucker–Smale, Grégoire–Chaté, D'Orsogna, Olfati-Saber, Active-Brownian, Run-and-tumble, Szabó, Swarmalator, Perception, SlowFast, Multi-group (+ formation/consensus). Custom = subclass `CollectiveModel`. Only unresolved name: **"Mind-Flock"** (no equations given). |
| 3 | Initialization | ✅ | `core/init.py`: random / cluster / ring / grid / manual / CSV. Group-aware. |
| 4 | Parameter Control | ✅ | GUI exposes a **live slider per numeric constructor arg** of the selected model (introspected). Structural params (N, boundary, init) rebuild; others apply live. |
| 5 | Visualization | ✅ | Interactive **2D and 3D** canvas: play/pause/step/reset, zoom/pan, drag-to-rotate (3D), trails, neighbour links, group colours, live metrics, boundary box/cube. (Vision cones are drawn in 2D only.) |
| 6 | Measurement | ✅ | polarization, milling, angular momentum, cluster/fragment count, nn-distance, **density**, **heading_entropy**, **mean_speed**, + group-maintenance + consensus observables. |
| 7 | Experiment Manager | ✅ | `experiments/manager.py`: registry, JSON/YAML config save/load, CSV/HDF5 export, parameter sweep — **now wired into the GUI** (Save/Load config, Record, Export metrics/trajectory, Screenshot, Save GIF). A config saved from the GUI re-runs headless via `manager.run_experiment`. |

### Technology stack

| Piece | Spec | Here | Status |
|---|---|---|---|
| Language | Python | Python 3.12 | ✅ |
| Numerics | NumPy | NumPy | ✅ |
| GUI | PySide6 (Qt) | PySide6 6.11 | ✅ |
| Canvas | PyQtGraph/VisPy | QPainter (self-contained) | ✅ (deliberate substitution) |
| Data | Pandas + HDF5 | CSV (stdlib) + optional h5py + optional pandas | ✅ (both available as extras) |
| Config | YAML/JSON | JSON (stdlib) + optional PyYAML | ✅ |
| Perf | Numba | vectorised (scipy sparse) fast-path | ✅ (roadmap allowed vectorisation; Numba not needed) |
| Video | (implied) | GIF (pillow) + MP4 (imageio-ffmpeg) | ✅ |

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

### 2026-07-28 — In-app explanations (tooltips + model descriptions)
**Added (in `gui/main_window.py`)**
- Help dictionaries `MODEL_HELP`, `PARAM_HELP`, `METRIC_HELP`, `SETUP_HELP`,
  `DISPLAY_HELP`, `BUTTON_HELP`.
- **Hover tooltips** on every control: setup fields, the introspected model
  parameters (label + slider), display toggles, transport/experiment/initial-
  condition buttons, and each live measurement.
- A **plain-language model description** label under the Setup box that updates
  when the model changes (e.g. Couzin → "repel very close, align at mid-range,
  attract far…").

**Verified** — tooltips present and the description updates per model
(`results/gui_help_panel.png`); GUI still builds and all models step.

### 2026-07-27 — Seven new models from the literature (`models/active.py`)
**Added** (each with a paper reference, honest `cohesive` flag, boundary-aware):
- **Vicsek (vectorial noise)** — Chaté et al. 2008 (extrinsic noise).
- **Inertial spin** — Cavagna et al. 2015 (spin/inertia, turn waves). 2D.
- **Active Brownian particles** — Fily & Marchetti 2012 (rotational diffusion +
  soft repulsion → MIPS). 2D.
- **Run-and-tumble** — bacterial motility. 2D.
- **Grégoire–Chaté** — 2004 cohesive Vicsek (alignment + body force). `cohesive`.
- **Szabó** — 2006 self-propelled cells (repulsion/adhesion + polarity relaxation).
  2D, `cohesive`.
- **Swarmalator** — O'Keeffe–Hong–Strogatz 2017 (coupled position + phase). 2D,
  `cohesive`.
- Base gains a `two_d_only` flag (angle/phase models); the GUI 3D guard now uses
  it generically. Registered in the GUI (17 models), the manager registry (21),
  and `models/__init__`.

**Verified**
- All 17 GUI models step 30× cleanly; 2D-only models auto-kept at 2D.
- **Physics sanity**: in open space Vicsek's R_g grows **4.5×** while Grégoire–
  Chaté **0.18×** and Swarmalator **0.20×** (they cohere); inertial spin orders
  (M **0.10 → 1.00**); vectorial-noise Vicsek and Grégoire–Chaté also run in 3D.
- **13/13 theorem tests still pass.**

### 2026-07-27 — MP4 export, pandas layer, vectorised Vicsek
**Added**
- **MP4 recording** — `gui/main_window.py` `Save MP4…` (`_do_save_mp4` via
  imageio/imageio-ffmpeg). Optional; skips cleanly if imageio is absent.
- **pandas convenience** — `experiments/manager.py`: `to_dataframe(history)`,
  `load_measurements(path)`, `sweep_to_dataframe(rows)`. Optional import.
- **Vectorised Vicsek** — `models/alignment.py` `VicsekModel(fast=True)` replaces
  the per-agent averaging loop with one sparse matrix product. **Default off**, so
  published-table numerics are unchanged; exposed as a checkbox in the GUI.
- `start.py` now also installs pillow / imageio / imageio-ffmpeg / pandas, so
  every GUI button works out of the box.

**Verified**
- `tests/test_features.py`: fast == slow after one step (max diff **0.0**), stays
  finite over 30 steps, ~1.2× faster at N=1500 (KDTree query dominates); pandas
  `to_dataframe` → 12×10 frame.
- `tests/test_gui_smoke.py`: MP4 written and re-read (**10 frames, 532×682, 20 fps**).
- **13/13 theorem tests still pass** — no regression (fast defaults off).

### 2026-07-27 — 3D interactive canvas (spec module 5)
**Added**
- `gui/canvas.py` — orthographic 3D projection used automatically when the state
  is 3D: agents (depth-sorted) + heading tails, boundary **cube** wireframe
  (dashed periodic / solid reflecting), projected trails and neighbour links.
  **Left-drag rotates** (azim/elev), wheel zooms. Vision cones stay 2D-only.
- `gui/main_window.py` — a **2D / 3D** view selector. 3D builds the boundary and
  initializer at `dim=3` (the engine is already d-dimensional). **Kuramoto** (a
  2D phase model) is auto-kept at 2D; **place mode** is disabled in 3D.

**Verified** (`tests/test_gui_smoke.py`) — Vicsek / Boids / Cucker-Smale each step
cleanly in 3D (N=80); Kuramoto is forced to 2D; 3D scenes render to
`results/gui_3d.png` (open) and `results/gui_3d_box.png` (periodic cube), both
visually checked.

### 2026-07-27 — In-GUI parameter sweep (spec module 7)
**Added (in `gui/main_window.py`)**
- **Sweep…** button + dialog: pick a numeric model parameter, a from/to range,
  number of points, steps-per-run, and a metric; it runs one batch per value via
  `manager.parameter_sweep` (using the current GUI setup as the base config) and
  plots the chosen metric vs the parameter.
- Split for testability: `_run_sweep` / `_build_sweep_figure` / `_do_sweep`; a
  shared `_show_figure` helper now backs both the sweep and the metrics plots.

**Verified** (`tests/test_gui_smoke.py`) — a Vicsek `eta` sweep (0→1.5) reproduces
the order–disorder transition: polar order **0.73 (quiet) → 0.07 (noisy)**; plot
saved to `results/gui_sweep.png` and visually checked (clean transition curve).

### 2026-07-27 — In-GUI analysis plots (spec module 6)
**Added (in `gui/main_window.py`)**
- **Plot metrics…** button: opens a matplotlib window of the recorded run —
  a 2×3 panel of time-series (polar order, milling, R_g, fragments, heading
  entropy, mean speed). One measure per panel (no dual-axis), titled so no legend
  is needed, recessive axes, colour-blind-safe (Okabe-Ito) colours; a panel whose
  measure is undefined here (e.g. milling on a torus) is annotated "n/a here".
- Rendering is split so it's testable: `_build_metrics_figure()` returns the
  Figure, `_do_plot_metrics(path)` saves it (Agg), `_plot_metrics()` shows it in
  a Qt dialog (QtAgg).

**Verified** (`tests/test_gui_smoke.py`) — recorded run → `results/gui_plot.png`
(85 KB) rendered and visually checked (Vicsek ordering up, fragments down,
entropy down, mean speed flat, milling correctly "n/a" on the torus).

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
- [x] **3D interactive canvas** — done 2026-07-27 (orthographic projection in the
      QPainter canvas, drag-to-rotate; 2D/3D selector). Vision cones remain 2D-only.
- [x] **MP4 export with quality/length options** — done 2026-07-27.
- [x] **Vectorization** — done 2026-07-27 for Vicsek (`fast=True`, sparse matmul).
      Other models' loops could get the same treatment if a bottleneck appears.
      (Full Numba not pursued — vectorisation removes the Python hot loop already.)
- [x] **MP4 export** — done 2026-07-27 (imageio-ffmpeg).
- [x] **Pandas convenience** — done 2026-07-27 (`manager.to_dataframe` et al.).
- [x] **CSV/manual init in the GUI** — done 2026-07-27 (Place mode, place-as-
      group, Clear, Load CSV…).

---

## 6. Verification snapshot (2026-07-27)

| Check | Command | Result |
|---|---|---|
| Theorem suite | `tests/test_theorems.py` | 13/13 pass |
| New features | `experiments/demo_new_features.py` | all sections pass |
| GUI (headless) | `tests/test_gui_smoke.py` | 17/17 models, screenshot, config round-trip, CSV/HDF5 export, GIF, MP4, metrics plot, parameter sweep, click-to-place, CSV load, 3D render |
| New models | physics sanity (this session) | open-space cohesion + ordering checks pass |
| Live GUI | `run_gui.py` | pending `libxcb-cursor0` install on user's machine |
