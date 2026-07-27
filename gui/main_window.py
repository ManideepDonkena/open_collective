"""
MainWindow -- the interactive front end.

Left: the SimCanvas. Right: a scrollable control panel that
  * picks model / boundary / initializer and their structural parameters,
  * exposes every NUMERIC constructor argument of the chosen model as a live
    slider (introspected from its __init__ signature), applied to the running
    model without a reset, and
  * reads the measurement module out live every frame.

A QTimer advances the simulation while "Play" is on; each tick calls
`model.step`, feeds the new State to the canvas, and refreshes the metrics.
"""

from __future__ import annotations

import inspect
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import core.init as cinit
from core.boundary import make_boundary
from core.metrics import milling_order, summarize
from core.neighbors import (metric_neighbors, topological_neighbors,
                            vision_cone_neighbors)
from experiments import manager
from models import (BoidsModel, CouzinModel, CuckerSmaleModel, DOrsognaModel,
                    KuramotoModel, MultiGroupFlock, OlfatiSaberModel,
                    PerceptionQuantum, SlowFastPerception, VicsekModel)

from .canvas import SimCanvas

# Curated to the free-running self-propelled models (formation/consensus models
# need adjacency/target matrices and don't belong on a free canvas).
GUI_MODELS = {
    "Vicsek": VicsekModel,
    "Kuramoto": KuramotoModel,
    "Boids": BoidsModel,
    "Couzin": CouzinModel,
    "Cucker-Smale": CuckerSmaleModel,
    "D'Orsogna": DOrsognaModel,
    "Olfati-Saber": OlfatiSaberModel,
    "Perception (PAPER_1)": PerceptionQuantum,
    "SlowFast (PAPER_2)": SlowFastPerception,
    "Multi-group flock": MultiGroupFlock,
}
# GUI display label <-> experiments.manager registry key, so a config saved from
# the GUI is directly runnable by manager.run_experiment (reproducibility).
GUI_TO_KEY = {
    "Vicsek": "Vicsek", "Kuramoto": "Kuramoto", "Boids": "Boids",
    "Couzin": "Couzin", "Cucker-Smale": "CuckerSmale", "D'Orsogna": "DOrsogna",
    "Olfati-Saber": "OlfatiSaber", "Perception (PAPER_1)": "Perception",
    "SlowFast (PAPER_2)": "SlowFast", "Multi-group flock": "MultiGroupFlock",
}
KEY_TO_GUI = {v: k for k, v in GUI_TO_KEY.items()}

# Numeric observables recorded per frame while "Record" is on (for CSV/HDF5 export).
_RECORD_KEYS = ["polar_order", "radius_of_gyration", "nn_distance", "n_fragments",
                "largest_cluster_frac", "mean_neighbors", "density",
                "heading_entropy", "mean_speed", "milling"]

_SKIP_PARAMS = {"self", "boundary", "rng", "groups"}
_METRIC_ROWS = [
    ("t", "t"), ("polar_order", "polar order M"), ("milling", "milling"),
    ("radius_of_gyration", "R_g"), ("n_fragments", "fragments"),
    ("largest_cluster_frac", "largest frac"), ("mean_neighbors", "mean neigh"),
    ("density", "density"), ("heading_entropy", "heading entropy"),
    ("mean_speed", "mean speed"), ("fps", "fps"),
]


def _numeric_params(cls):
    """(name, kind, default) for each bool/int/float constructor arg."""
    out = []
    for name, prm in inspect.signature(cls.__init__).parameters.items():
        if name in _SKIP_PARAMS:
            continue
        d = prm.default
        if isinstance(d, bool):
            out.append((name, "bool", d))
        elif isinstance(d, int):
            out.append((name, "int", d))
        elif isinstance(d, float):
            out.append((name, "float", d))
    return out


def _range_for(d):
    if d == 0:
        return 0.0, 1.0
    if d > 0:
        return 0.0, max(1.0, 4.0 * d)
    return 2.0 * d, -2.0 * d


class FloatSlider(QtWidgets.QWidget):
    """A QSlider (int 0..1000) mapped onto a float range, with a value label."""
    valueChanged = Signal(float)

    def __init__(self, lo, hi, val):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.readout = QtWidgets.QLabel()
        self.readout.setMinimumWidth(52)
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.readout)
        self.set_value(val)
        self.slider.valueChanged.connect(self._on)

    def _to_slider(self, v):
        frac = (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.0
        return int(round(np.clip(frac, 0.0, 1.0) * 1000))

    def value(self):
        return self.lo + self.slider.value() / 1000.0 * (self.hi - self.lo)

    def set_value(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(v))
        self.slider.blockSignals(False)
        self.readout.setText(f"{v:.3g}")

    def _on(self, _):
        v = self.value()
        self.readout.setText(f"{v:.3g}")
        self.valueChanged.emit(v)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("open-collective — BirdFlockLab")
        self.resize(1120, 720)

        self.canvas = SimCanvas()
        self.model = None
        self.state = None
        self.boundary = None
        self.dt = 0.05
        self.r_link = 1.0
        self._steps = 0
        self._fps = 0.0
        self._last_tick = time.perf_counter()
        self.param_widgets = {}      # name -> (widget, kind)
        self._loading = False        # suppress resets while a config loads

        # recording buffers (filled while "Record" is on; consumed by exports)
        self.recording = False
        self.rec_metrics = None      # dict[str, list] incl. "t"
        self.rec_traj, self.rec_vel, self.rec_grp = [], [], []

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)   # ~30 fps
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._on_model_changed()     # builds params + first reset

    # -- UI ---------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.addWidget(self.canvas, 1)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        pl = QtWidgets.QVBoxLayout(panel)

        # --- setup group (structural: change => reset) ---
        setup = QtWidgets.QGroupBox("Setup")
        form = QtWidgets.QFormLayout(setup)
        self.model_cb = QtWidgets.QComboBox(); self.model_cb.addItems(GUI_MODELS)
        self.boundary_cb = QtWidgets.QComboBox()
        self.boundary_cb.addItems(["periodic", "open", "reflecting"])
        self.boundary_cb.setCurrentText("open")
        self.L_sb = QtWidgets.QDoubleSpinBox(); self.L_sb.setRange(1, 200); self.L_sb.setValue(10.0)
        self.init_cb = QtWidgets.QComboBox()
        self.init_cb.addItems(["random", "cluster", "ring", "grid"])
        self.N_sb = QtWidgets.QSpinBox(); self.N_sb.setRange(2, 3000); self.N_sb.setValue(150)
        self.groups_sb = QtWidgets.QSpinBox(); self.groups_sb.setRange(1, 8); self.groups_sb.setValue(1)
        self.speed_sb = QtWidgets.QDoubleSpinBox(); self.speed_sb.setRange(0, 10); self.speed_sb.setSingleStep(0.1); self.speed_sb.setValue(0.5)
        self.dt_sb = QtWidgets.QDoubleSpinBox(); self.dt_sb.setRange(0.001, 1.0); self.dt_sb.setSingleStep(0.01); self.dt_sb.setDecimals(3); self.dt_sb.setValue(0.05)
        for lab, w in [("model", self.model_cb), ("boundary", self.boundary_cb),
                       ("box L", self.L_sb), ("init", self.init_cb),
                       ("N", self.N_sb), ("groups", self.groups_sb),
                       ("speed", self.speed_sb), ("dt", self.dt_sb)]:
            form.addRow(lab, w)
        pl.addWidget(setup)

        self.model_cb.currentIndexChanged.connect(self._on_model_changed)
        for w in (self.boundary_cb, self.init_cb):
            w.currentIndexChanged.connect(self._reset)
        for w in (self.L_sb, self.speed_sb):
            w.valueChanged.connect(self._reset)
        self.N_sb.valueChanged.connect(self._reset)
        self.groups_sb.valueChanged.connect(self._reset)
        self.dt_sb.valueChanged.connect(lambda v: setattr(self, "dt", float(v)))

        # --- transport ---
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play"); self.play_btn.setCheckable(True)
        self.step_btn = QtWidgets.QPushButton("Step")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        row.addWidget(self.play_btn); row.addWidget(self.step_btn); row.addWidget(self.reset_btn)
        pl.addLayout(row)
        self.play_btn.toggled.connect(self._on_play)
        self.step_btn.clicked.connect(self._tick)
        self.reset_btn.clicked.connect(self._reset)

        # --- display toggles ---
        disp = QtWidgets.QGroupBox("Display")
        dl = QtWidgets.QGridLayout(disp)
        self.cb_trails = QtWidgets.QCheckBox("trajectories"); self.cb_trails.setChecked(True)
        self.cb_cones = QtWidgets.QCheckBox("vision cones")
        self.cb_links = QtWidgets.QCheckBox("neighbour links")
        self.cb_groups = QtWidgets.QCheckBox("group colours"); self.cb_groups.setChecked(True)
        for i, cb in enumerate((self.cb_trails, self.cb_cones, self.cb_links, self.cb_groups)):
            dl.addWidget(cb, i // 2, i % 2)
            cb.toggled.connect(self._apply_display)
        pl.addWidget(disp)

        # --- model parameters (dynamic) ---
        self.param_box = QtWidgets.QGroupBox("Model parameters (live)")
        self.param_form = QtWidgets.QFormLayout(self.param_box)
        pl.addWidget(self.param_box)

        # --- experiment manager (config / export / capture) ---
        exp = QtWidgets.QGroupBox("Experiment")
        eg = QtWidgets.QGridLayout(exp)
        self.rec_btn = QtWidgets.QPushButton("● Record"); self.rec_btn.setCheckable(True)
        buttons = [
            ("Save config…", self._save_config), ("Load config…", self._load_config),
            ("Screenshot…", self._screenshot), (self.rec_btn, self._toggle_record),
            ("Export metrics…", self._export_metrics), ("Export trajectory…", self._export_traj),
            ("Save GIF…", self._save_gif),
        ]
        for i, (item, slot) in enumerate(buttons):
            btn = item if isinstance(item, QtWidgets.QPushButton) else QtWidgets.QPushButton(item)
            (btn.toggled if btn.isCheckable() else btn.clicked).connect(slot)
            eg.addWidget(btn, i // 2, i % 2)
        self.rec_status = QtWidgets.QLabel("not recording")
        eg.addWidget(self.rec_status, (len(buttons) + 1) // 2, 0, 1, 2)
        pl.addWidget(exp)

        # --- metrics ---
        met = QtWidgets.QGroupBox("Measurements")
        ml = QtWidgets.QFormLayout(met)
        self.metric_labels = {}
        for key, lab in _METRIC_ROWS:
            v = QtWidgets.QLabel("—")
            self.metric_labels[key] = v
            ml.addRow(lab, v)
        pl.addWidget(met)
        pl.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setWidget(panel)
        scroll.setFixedWidth(344)
        root.addWidget(scroll)
        self.setCentralWidget(central)

    # -- model / params ---------------------------------------------------
    def _on_model_changed(self, *_):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.param_widgets = {}
        cls = GUI_MODELS[self.model_cb.currentText()]
        for name, kind, default in _numeric_params(cls):
            if kind == "bool":
                w = QtWidgets.QCheckBox(); w.setChecked(default)
                w.toggled.connect(self._apply_params)
            elif kind == "int":
                w = QtWidgets.QSpinBox(); lo, hi = _range_for(default)
                w.setRange(int(lo), max(1, int(hi))); w.setValue(default)
                w.valueChanged.connect(self._apply_params)
            else:
                lo, hi = _range_for(default)
                w = FloatSlider(lo, hi, default)
                w.valueChanged.connect(self._apply_params)
            self.param_widgets[name] = (w, kind)
            self.param_form.addRow(name, w)
        self._reset()

    def _param_values(self):
        out = {}
        for name, (w, kind) in self.param_widgets.items():
            out[name] = w.isChecked() if kind == "bool" else (
                w.value() if kind == "int" else w.value())
        return out

    def _apply_params(self, *_):
        """Push live parameter values onto the running model object."""
        if self.model is None:
            return
        for name, val in self._param_values().items():
            if hasattr(self.model, name):
                setattr(self.model, name, val)

    def _apply_display(self, *_):
        self.canvas.show_trails = self.cb_trails.isChecked()
        self.canvas.show_cones = self.cb_cones.isChecked()
        self.canvas.show_links = self.cb_links.isChecked()
        self.canvas.show_groups = self.cb_groups.isChecked()
        if not self.canvas.show_trails:
            self.canvas.clear_trails()
        self._refresh_canvas()

    # -- build / reset ----------------------------------------------------
    def _reset(self, *_):
        if self._loading:            # a config is being applied; rebuild once at the end
            return
        if self.recording:          # a fresh run invalidates the current recording
            self.rec_btn.setChecked(False)
        kind = self.boundary_cb.currentText()
        L = float(self.L_sb.value())
        self.boundary = make_boundary(kind, L, 2)
        self.r_link = 1.0
        n = int(self.N_sb.value())
        ng = int(self.groups_sb.value())
        speed = float(self.speed_sb.value())
        rng = np.random.default_rng(0)

        method = self.init_cb.currentText()
        if method == "random":
            st = cinit.random_init(n, self.boundary, speed=speed, n_groups=ng, rng=rng)
        elif method == "cluster":
            st = cinit.cluster_init(n, self.boundary, speed=speed,
                                    n_clusters=max(1, ng),
                                    aligned_within_cluster=True, rng=rng)
        elif method == "ring":
            st = cinit.ring_init(n, self.boundary, speed=speed, tangential=True,
                                 n_groups=ng, rng=rng)
        else:
            st = cinit.grid_init(n, self.boundary, speed=speed, n_groups=ng, rng=rng)

        name = self.model_cb.currentText()
        cls = GUI_MODELS[name]
        # Multi-group flock needs an allegiance vector to exist.
        if cls is MultiGroupFlock and "groups" not in st.internal:
            st.internal["groups"] = cinit.assign_groups(n, max(2, ng))
        self.model = cls(self.boundary, rng=np.random.default_rng(0),
                         **self._param_values())
        self.state = st
        self._steps = 0
        self.dt = float(self.dt_sb.value())
        self.canvas.clear_trails()
        self.canvas.fit(st.positions, self.boundary)
        self._apply_display()
        self._update_metrics()

    # -- run loop ---------------------------------------------------------
    def _on_play(self, on):
        self.play_btn.setText("Pause" if on else "Play")
        self._last_tick = time.perf_counter()
        (self.timer.start if on else self.timer.stop)()

    def _tick(self):
        self.state = self.model.step(self.state, self.dt)
        self._steps += 1
        if self.canvas.show_trails and self._steps % 2 == 0:
            self.canvas.push_trail(self.state.positions)
        now = time.perf_counter()
        dt_wall = now - self._last_tick
        if dt_wall > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt_wall)
        self._last_tick = now
        self._refresh_canvas()
        # compute metrics every 3rd frame for display, but every frame while recording
        if self.recording or self._steps % 3 == 0:
            m = self._metrics_now()
            self._set_metric_labels(m)
            if self.recording:
                self._record_step(m)

    def _neighbor_info(self):
        if not (self.canvas.show_links or self.canvas.show_cones):
            return None, None
        m, b, st = self.model, self.boundary, self.state
        alpha = getattr(m, "alpha", None)
        r_min = getattr(m, "r_min", 0.0)
        r_max = (getattr(m, "r_max", None) or getattr(m, "r_coh", None)
                 or getattr(m, "zoa", None) or getattr(m, "cutoff", None))
        if alpha is not None and r_max is not None:
            neigh = vision_cone_neighbors(st.positions, st.headings, b, alpha,
                                          r_max, r_min)
            return neigh, (alpha, r_max)
        if getattr(m, "topological", False):
            return topological_neighbors(st.positions, b, getattr(m, "k", 7)), None
        if r_max is not None:
            return metric_neighbors(st.positions, b, r_max, r_min), (None, r_max)
        return None, None

    def _refresh_canvas(self):
        neigh, vision = self._neighbor_info()
        self.canvas.set_frame(self.state.positions, self.state.headings,
                              self.boundary, self.state.internal.get("groups"),
                              neigh, vision)

    def _metrics_now(self):
        st, b = self.state, self.boundary
        s = summarize(st.positions, st.headings, b, self.r_link,
                      velocities=st.velocities)
        s["t"] = st.t
        s["milling"] = milling_order(st.positions, st.velocities, b)
        s["fps"] = self._fps
        return s

    def _set_metric_labels(self, s):
        for key, _ in _METRIC_ROWS:
            v = s.get(key)
            if isinstance(v, float) and not np.isfinite(v):
                txt = "n/a"
            elif key in ("n_fragments",):
                txt = str(int(v))
            elif isinstance(v, float):
                txt = f"{v:.3f}"
            else:
                txt = str(v)
            self.metric_labels[key].setText(txt)

    def _update_metrics(self):
        self._set_metric_labels(self._metrics_now())

    # -- recording --------------------------------------------------------
    def _toggle_record(self, on):
        self.recording = on
        if on:
            self.rec_metrics = {k: [] for k in ["t"] + _RECORD_KEYS}
            self.rec_traj, self.rec_vel, self.rec_grp = [], [], []
            self._record_step(self._metrics_now())      # capture frame 0
        self.rec_btn.setText("● Recording" if on else "● Record")
        self._update_rec_status()

    def _record_step(self, m):
        for k in ["t"] + _RECORD_KEYS:
            self.rec_metrics[k].append(float(m.get(k, np.nan)))
        self.rec_traj.append(self.state.positions.copy())
        self.rec_vel.append(self.state.velocities.copy())
        g = self.state.internal.get("groups")
        self.rec_grp.append(None if g is None else np.asarray(g).copy())
        self._update_rec_status()

    def _update_rec_status(self):
        n = len(self.rec_metrics["t"]) if self.rec_metrics else 0
        self.rec_status.setText(f"recording… {n} frames" if self.recording
                                else (f"recorded {n} frames" if n else "not recording"))

    def _history_dict(self):
        if not self.rec_metrics or not self.rec_metrics["t"]:
            return None
        h = {k: np.asarray(v) for k, v in self.rec_metrics.items()}
        if self.rec_traj:
            h["trajectory"] = np.asarray(self.rec_traj)
            h["velocity_trajectory"] = np.asarray(self.rec_vel)
            if self.rec_grp and self.rec_grp[0] is not None:
                h["group_trajectory"] = np.asarray(self.rec_grp)
        return h

    # -- config round-trip ------------------------------------------------
    def current_config(self):
        """The current GUI setup as a manager-compatible config dict.

        The result is directly runnable headless via
        `experiments.manager.run_experiment(cfg)` -- that is the reproducibility
        guarantee: what you set up in the GUI is exactly what re-runs.
        """
        method = self.init_cb.currentText()
        n = int(self.N_sb.value()); speed = float(self.speed_sb.value())
        ng = int(self.groups_sb.value())
        if method == "cluster":
            init = {"method": "cluster", "n": n, "speed": speed,
                    "n_clusters": max(1, ng), "aligned_within_cluster": True}
        elif method == "ring":
            init = {"method": "ring", "n": n, "speed": speed,
                    "tangential": True, "n_groups": ng}
        else:  # random / grid
            init = {"method": method, "n": n, "speed": speed, "n_groups": ng}
        return {
            "seed": 0,
            "boundary": {"kind": self.boundary_cb.currentText(),
                         "L": float(self.L_sb.value()), "dim": 2},
            "init": init,
            "model": {"name": GUI_TO_KEY[self.model_cb.currentText()],
                      "params": self._param_values()},
            "run": {"steps": 500, "dt": float(self.dt_sb.value()),
                    "r_link": self.r_link, "record_every": 1, "record_traj": True},
        }

    def apply_config(self, cfg):
        """Set every control from a config dict, then rebuild once."""
        self._loading = True
        try:
            b = cfg.get("boundary", {})
            self.boundary_cb.setCurrentText(b.get("kind", "open"))
            self.L_sb.setValue(float(b.get("L", 10.0)))
            ic = dict(cfg.get("init", {}))
            self.init_cb.setCurrentText(ic.get("method", "random"))
            self.N_sb.setValue(int(ic.get("n", 150)))
            self.groups_sb.setValue(int(ic.get("n_groups", ic.get("n_clusters", 1))))
            self.speed_sb.setValue(float(ic.get("speed", 0.5)))
            self.dt_sb.setValue(float(cfg.get("run", {}).get("dt", 0.05)))
            key = cfg.get("model", {}).get("name", "Vicsek")
            self.model_cb.setCurrentText(KEY_TO_GUI.get(key, self.model_cb.currentText()))
            self._on_model_changed()   # rebuild sliders for this model (reset suppressed)
        finally:
            self._loading = False
        for name, val in cfg.get("model", {}).get("params", {}).items():
            if name in self.param_widgets:
                w, kind = self.param_widgets[name]
                if kind == "bool":
                    w.setChecked(bool(val))
                elif kind == "int":
                    w.setValue(int(val))
                else:
                    w.set_value(float(val))
        self._reset()

    # -- button slots (thin wrappers around testable _do_* methods) -------
    def _save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config", "results/config.json",
            "Config (*.json *.yaml *.yml)")
        if path:
            manager.save_config(self.current_config(), path)
            self.statusBar().showMessage(f"Saved config -> {path}", 4000)

    def _load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load config", "results", "Config (*.json *.yaml *.yml)")
        if path:
            self.apply_config(manager.load_config(path))
            self.statusBar().showMessage(f"Loaded config <- {path}", 4000)

    def _screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", "results/screenshot.png", "PNG (*.png)")
        if path:
            self.canvas.grab().save(path)
            self.statusBar().showMessage(f"Saved screenshot -> {path}", 4000)

    def _export_metrics(self):
        h = self._history_dict()
        if h is None:
            return self._warn("Nothing recorded. Press ● Record, then run.")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export metrics", "results/metrics.csv", "CSV (*.csv)")
        if path:
            manager.export_measurements(h, path)
            self.statusBar().showMessage(f"Exported metrics -> {path}", 4000)

    def _export_traj(self):
        h = self._history_dict()
        if h is None or "trajectory" not in h:
            return self._warn("Nothing recorded. Press ● Record, then run.")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export trajectory", "results/trajectory.csv",
            "CSV (*.csv);;HDF5 (*.h5 *.hdf5)")
        if path:
            manager.export_trajectory(h, path)
            self.statusBar().showMessage(f"Exported trajectory -> {path}", 4000)

    def _save_gif(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GIF", "results/animation.gif", "GIF (*.gif)")
        if path:
            ok, msg = self._do_save_gif(path)
            (self.statusBar().showMessage if ok else self._warn)(
                f"Saved GIF -> {path}" if ok else msg)

    def _do_save_gif(self, path, frames=60, fps=25):
        """Step the sim `frames` times, grabbing the canvas into an animated GIF."""
        try:
            from PIL import Image
        except Exception:
            return False, "Pillow not installed (pip install pillow)."
        imgs = []
        for _ in range(frames):
            self._tick()
            imgs.append(self._grab_pil(Image))
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        return True, path

    def _grab_pil(self, Image):
        img = self.canvas.grab().toImage().convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        buf = np.frombuffer(bytes(img.constBits()), np.uint8)
        buf = buf.reshape((h, img.bytesPerLine() // 4, 4))[:, :w, :]
        return Image.fromarray(buf, "RGBA").convert("RGB")

    def _warn(self, msg):
        QtWidgets.QMessageBox.warning(self, "open-collective", msg)
