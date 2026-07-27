# Getting Started

This guide shows you how to install and run the project. It works on **Linux,
macOS, and Windows**. Just follow the steps in order.

## ⭐ The easy way (no setup, no commands to learn)

If you are not technical, ignore everything else and do this:

1. Make sure **Python 3.10+** is installed (from [python.org](https://www.python.org/downloads/)).
2. In this folder, run one line:
   ```bash
   python3 start.py        # on Windows:  python start.py
   ```
   or **double-click** the file for your system:
   - **Windows** → `start.bat` (double-click runs it)
   - **Mac** → `start.command` (double-click runs it in Terminal)
   - **Linux (GNOME/most desktops)** → double-clicking `start.sh` usually just
     **opens it in a text editor, it does not run.** Instead, **right-click
     `start.sh` → "Run as a Program"**, or run `python3 start.py` in a terminal.
     For a real clickable icon, copy `BirdFlockLab.desktop` into
     `~/.local/share/applications/` (then find "BirdFlockLab" in your apps).

The first time, it makes its own private workspace and installs **everything the
app needs** — the visual app, plotting, and the file-saving formats (nothing
else on your computer is changed). This takes a few minutes. Then the app opens.
That's it. If a window doesn't appear on Linux, `start.py` prints the one line to
copy-paste that fixes it.

**Just want a picture?** After running `start.py` once, run `python3 example.py`
— it saves a picture called `flock.png` you can open. Change the three numbers
at the top of that file and run it again to experiment.

The rest of this page is the manual, step-by-step version.

---

- Want to know what the project does? See [../README.md](../README.md).
- Want the full research story? See [../RESEARCH.md](../RESEARCH.md).
- Want the build progress and change log? See [../BIRDFLOCKLAB.md](../BIRDFLOCKLAB.md).

---

## 1. What you need first

- **Python 3.10 or newer.** Check with `python --version` (or `python3 --version`).
- **pip.** It comes with Python.
- **git.** Only needed to download the code.
- **A normal desktop.** Only needed for the visual app, not for the rest.

> Tip: on some computers the command is `python3` instead of `python`. Once you
> turn on the virtual environment (Step 2), plain `python` always works. So this
> guide just uses `python`.

---

## 2. Install

### Step 1 — Get the code
```bash
git clone <repository-url> open_collective
cd open_collective
```
No git? Download the ZIP, unzip it, and open a terminal inside the folder.

### Step 2 — Make a safe space for the tools (a "virtual environment")

This keeps the project's tools in their own box, so they don't clash with
anything else on your computer.

Make the box (same command everywhere):
```bash
python -m venv .venv
```

Turn it on:

| Your system | Type this |
|---|---|
| Linux or macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

You will see `(.venv)` at the start of your prompt. That means it is on.
To turn it off later, type `deactivate`.

### Step 3 — Install the tools

The basics (needed by everything):
```bash
pip install -r requirements.txt
```

Extras — install only the ones you want:
```bash
pip install PySide6      # the visual app
pip install pillow       # to save animations as GIF files
pip install h5py pyyaml  # to save data as .h5 or configs as .yaml
```

Prefer one command? This installs the project itself so you can `import` it from
anywhere:
```bash
pip install -e .              # basics
pip install -e ".[gui]"       # basics + the visual app
```

---

## 3. Check that it works

With the box turned on (Step 2), run:
```bash
python tests/test_theorems.py           # you should see: 13/13 theorem tests passed
python experiments/demo_new_features.py  # runs a quick demo of the new features
```
If these finish with no errors, you are ready.

---

## 4. Run it

### Run the experiments (no screen needed)
```bash
python experiments/exp1_boundary_collapse.py   # the main result table
python experiments/exp2_grouping.py             # groups staying apart
python experiments/viz3d_demo.py                # 3D movies (needs pillow)
```

### Open the visual app (needs a desktop and PySide6)
```bash
python run_gui.py
```
Same thing, other ways: `python -m gui`, or `birdflocklab-gui` if you used
`pip install -e ".[gui]"`.

Window won't open? See Step 6.

### Try the app with no screen
This draws the app in the background and saves a picture. Handy on a server or
over SSH:
```bash
python tests/test_gui_smoke.py           # saves results/gui_smoke.png
```

---

## 5. Using the app

The picture of the birds is on the **left**. The buttons are on the **right**.

- **Setup** (top): pick the *model*, the *boundary* (periodic, open, or
  reflecting), how the birds *start* (random, cluster, ring, grid), how *many*
  birds, how many *groups*, their *speed*, and the *time step*. Changing any of
  these starts a fresh run.
- **Play / Pause / Step / Reset**: control the animation.
- **Display**: turn on or off the *trails*, *vision cones*, *neighbour lines*,
  and *group colours*.
- **Model parameters**: sliders for the picked model. Move a slider and the
  change happens right away, while it runs.
- **Initial condition (manual)**: turn on **Place mode** and click the canvas to
  add birds by hand (pick a colour with *place as group*). **Clear birds** empties
  it; **Load CSV** loads birds from a file. (While Place mode is on, click = add a
  bird, drag = move the view.)
- **Measurements**: live numbers (order, spread, groups, speed, and more) that
  update every frame.
- **Experiment**: save your current setup to a file (**Save config**) and load
  it back later (**Load config**) — the exact same run comes back. **● Record**
  starts saving each frame; then **Export metrics** writes the numbers to a CSV
  and **Export trajectory** writes every bird's path to a CSV or `.h5` file.
  **Plot metrics** opens charts of the recorded numbers over time.
  **Sweep** re-runs the experiment while one setting changes and plots the result
  (e.g. how order changes as noise goes up).
  **Screenshot** saves a picture. **Save GIF** and **Save MP4** save a short
  animation — for MP4 you pick length, frames-per-second, and quality (Low makes a
  small file, High makes a sharper, larger one).
- **View (2D / 3D)**: switch the canvas between flat 2D and a rotatable 3D view.
- **Move around**: **scroll** to zoom; **drag** moves the view in 2D and **rotates**
  it in 3D.

---

## 6. If the app won't open

The experiments always work. Only the visual app needs a bit of help sometimes,
because it uses a graphics tool called Qt.

### Linux — error says "Could not load the Qt platform plugin xcb"
Your system is missing one small library. Install it:

| Your Linux | Type this |
|---|---|
| Ubuntu, Debian, Mint | `sudo apt update && sudo apt install -y libxcb-cursor0` |
| Fedora, RHEL, CentOS | `sudo dnf install -y xcb-util-cursor` |
| Arch, Manjaro | `sudo pacman -S xcb-util-cursor` |
| openSUSE | `sudo zypper install -y libxcb-cursor0` |

Still asking for more? On Ubuntu/Debian also try
`sudo apt install -y libxcb-xinerama0 libxkbcommon-x11-0`.

### macOS
Usually just works after `pip install PySide6`.

### Windows
Usually just works after `pip install PySide6`. Nothing else to install.

### No screen (server, SSH, or WSL)
You can still run it in the background and save a picture:

Linux/macOS:
```bash
QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py
```
Windows (PowerShell):
```powershell
$env:QT_QPA_PLATFORM="offscreen"; python tests/test_gui_smoke.py
```

---

## 7. Use it in your own code

Run a simulation and read the numbers:
```python
from core import make_boundary, run
from core.metrics import summarize
import core.init as cinit
from models import VicsekModel

boundary = make_boundary("open", dim=2)               # open / periodic / reflecting
state = cinit.random_init(200, boundary, speed=0.5)   # or cluster/ring/grid
model = VicsekModel(boundary, r_max=1.0, eta=0.2)
final, history = run(model, state, steps=500, dt=0.05, r_link=1.0)
print(summarize(final.positions, final.headings, boundary, 1.0,
                velocities=final.velocities))
```

Run from a config, then save results and try many settings:
```python
from experiments import manager

cfg = {
  "boundary": {"kind": "periodic", "L": 10.0, "dim": 2},
  "init":  {"method": "random", "n": 120, "speed": 0.5},
  "model": {"name": "Vicsek", "params": {"r_max": 1.0, "eta": 0.3}},
  "run":   {"steps": 300, "dt": 0.05, "r_link": 1.0, "record_traj": True},
}
final, hist = manager.run_experiment(cfg)
manager.save_config(cfg, "results/my_config.json")      # or .yaml
manager.export_measurements(hist, "results/my_metrics.csv")
manager.export_trajectory(hist, "results/my_traj.csv")  # or .h5
manager.parameter_sweep(cfg, "model.params.eta", [0.0, 0.3, 0.6, 1.0])
```

Model names you can use: `Vicsek, Kuramoto, Boids, Couzin, CuckerSmale,
DOrsogna, OlfatiSaber, Perception, SlowFast, MultiGroupFlock`.

---

## 8. Common problems and fixes

| You see | Why | Do this |
|---|---|---|
| `No module named 'core'` | not run from the main folder | run from the project's top folder, or `pip install -e .` |
| `No module named 'scipy'` (or numpy/matplotlib) | tools not installed, or box is off | turn on the box (Step 2), then `pip install -r requirements.txt` |
| `externally-managed-environment` when installing | installing outside the box | use the virtual environment (Step 2) |
| `command not found: python` | Python missing, or it's `python3` | install Python 3.10+, or type `python3` |
| `Could not load the Qt platform plugin "xcb"` | missing Linux library | see Step 6 |
| `No module named 'PySide6'` | the app tool isn't installed | `pip install PySide6` |
| animations don't save | Pillow missing | `pip install pillow` |

---

## 9. What is in the folders

```
core/          the engine: boundaries, neighbours, measurements, the run loop, setup
models/        the models: Vicsek, Boids, Kuramoto, Cucker-Smale, and more
experiments/   ready-to-run studies + the manager (save, load, export, sweeps)
gui/           the visual app (PySide6)
tests/         checks that everything still works
Uav/           drone / multi-team simulations
results/       pictures and data that get produced
```

---

## 10. Where to go next

- What it does and the main result: [../README.md](../README.md)
- The full research write-up: [../RESEARCH.md](../RESEARCH.md)
- Build progress and change log: [../BIRDFLOCKLAB.md](../BIRDFLOCKLAB.md)
- Add your own model: make a new class in `models/` that extends
  `CollectiveModel` and write its `step(state, dt)`. Get distances between birds
  with `self.boundary.displacement(...)` so it works in every boundary type.
