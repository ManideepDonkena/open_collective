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
from core.base import State
from core.boundary import make_boundary
from core.metrics import milling_order, summarize
from core.neighbors import (metric_neighbors, topological_neighbors,
                            vision_cone_neighbors)
from experiments import manager
from models import (ActiveBrownianParticles, BoidsModel, CouzinModel,
                    CuckerSmaleModel, DOrsognaModel, GregoireChateModel,
                    InertialSpinModel, KuramotoModel, MultiGroupFlock,
                    OlfatiSaberModel, PerceptionQuantum, RunAndTumbleModel,
                    SlowFastPerception, SwarmalatorModel, SzaboModel,
                    VicsekModel, VicsekVectorialNoise)

from .canvas import SimCanvas

# Curated to the free-running self-propelled models (formation/consensus models
# need adjacency/target matrices and don't belong on a free canvas).
GUI_MODELS = {
    "Vicsek": VicsekModel,
    "Vicsek (vectorial noise)": VicsekVectorialNoise,
    "Kuramoto": KuramotoModel,
    "Inertial spin": InertialSpinModel,
    "Boids": BoidsModel,
    "Couzin": CouzinModel,
    "Cucker-Smale": CuckerSmaleModel,
    "Grégoire–Chaté": GregoireChateModel,
    "D'Orsogna": DOrsognaModel,
    "Olfati-Saber": OlfatiSaberModel,
    "Active Brownian": ActiveBrownianParticles,
    "Run-and-tumble": RunAndTumbleModel,
    "Szabó (cells)": SzaboModel,
    "Swarmalator": SwarmalatorModel,
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
    "Vicsek (vectorial noise)": "VicsekVectorialNoise",
    "Inertial spin": "InertialSpin", "Grégoire–Chaté": "GregoireChate",
    "Active Brownian": "ActiveBrownian", "Run-and-tumble": "RunAndTumble",
    "Szabó (cells)": "Szabo", "Swarmalator": "Swarmalator",
}
KEY_TO_GUI = {v: k for k, v in GUI_TO_KEY.items()}

# --------------------------------------------------------------------------
# Explanatory help text (shown as tooltips / a model-description label)
# --------------------------------------------------------------------------
MODEL_HELP = {
    "Vicsek": "Classic flocking: each agent turns toward the average heading of "
              "neighbours within a radius, plus noise. Alignment-only, so it "
              "disperses in open space.",
    "Vicsek (vectorial noise)": "Vicsek where the noise is added to the neighbour "
              "vector-sum (measurement error). Gives a sharper, first-order "
              "order–disorder transition.",
    "Kuramoto": "Coupled phase oscillators: headings synchronise like coupled "
              "clocks. The sync order parameter equals the flocking order.",
    "Inertial spin": "Vicsek plus turning inertia (a 'spin'): flocks turn together "
              "and turn information propagates as spin waves (Cavagna 2015).",
    "Boids": "Reynolds' three rules — separation (avoid crowding), alignment "
              "(match heading), and cohesion (steer to the group centre).",
    "Couzin": "Zonal model: repel very close, align at mid-range, attract far. "
              "Produces swarms, tori (mills), and moving flocks.",
    "Cucker-Smale": "Everyone follows a weighted average of all others; a falloff "
              "exponent β ≤ 0.5 guarantees the flock never breaks apart.",
    "Grégoire–Chaté": "Vicsek plus an attraction/repulsion body force — a cohesive "
              "flock that stays together even in open space (2004).",
    "D'Orsogna": "Self-propelled particles with a Morse potential; shows mills, "
              "rings, and collapsed clumps depending on the parameters.",
    "Olfati-Saber": "Flocking control law with a smooth potential and navigation "
              "feedback (used in robotics/UAV swarms).",
    "Active Brownian": "No alignment at all — self-propelled particles that only "
              "repel. They still clump via motility-induced phase separation.",
    "Run-and-tumble": "Bacterial motion: run straight for a while, then randomly "
              "reorient ('tumble'). Optional soft repulsion.",
    "Szabó (cells)": "Self-propelled biological cells: repel/adhere to neighbours, "
              "and slowly steer toward their actual direction of motion.",
    "Swarmalator": "Agents whose position and internal phase (shown as colour) "
              "affect each other — clustering coupled with synchronisation.",
    "Perception (PAPER_1)": "Quantum-inspired perception operator over a forward "
              "vision cone (Beuria, Chaurasiya & Behera).",
    "SlowFast (PAPER_2)": "Two-timescale perception with a slow memory register "
              "that makes the flock's response history-dependent (Beuria).",
    "Multi-group flock": "K groups that stay distinct: in-group Cucker–Smale "
              "cohesion plus out-group repulsion (segregation).",
}

# Parameter name -> one-line explanation. Falls back to the raw name if absent.
PARAM_HELP = {
    "r_max": "Interaction radius — how far an agent senses neighbours.",
    "r_min": "Inner radius — neighbours closer than this are ignored.",
    "eta": "Noise strength — higher means more random, less ordered motion.",
    "v0": "Cruising speed of each agent.",
    "k": "Number of nearest neighbours used (topological interaction).",
    "topological": "Interact with the k nearest neighbours instead of all within a radius.",
    "fast": "Faster vectorised update — same result, quicker for many agents.",
    "K": "Coupling / interaction strength.",
    "sigma": "Interaction length scale (also the particle size for repulsion).",
    "beta": "Interaction falloff exponent (Cucker–Smale: ≤ 0.5 guarantees flocking).",
    "lam": "Interaction gain.",
    "lam_in": "In-group interaction gain.",
    "kappa": "Coupling gain.",
    "alpha": "Vision-cone angle (radians) the agent can see.",
    "chi": "Turning inertia — resistance to changing heading (inertial spin).",
    "J": "Alignment / coupling strength.",
    "Dr": "Rotational diffusion — how fast headings drift randomly.",
    "k_rep": "Repulsion stiffness — how hard agents push apart when overlapping.",
    "mu": "Mobility — how strongly forces translate into motion.",
    "tumble_rate": "How often an agent randomly reorients (tumbles).",
    "tau": "Relaxation time — how quickly heading follows actual motion.",
    "F_rep": "Repulsion force strength.",
    "F_adh": "Adhesion (stickiness) force strength.",
    "r_eq": "Preferred spacing — the force is zero at this distance.",
    "r_0": "Interaction cutoff distance.",
    "r_c": "Hard-core radius — strong repulsion below this.",
    "r_e": "Equilibrium distance where attraction and repulsion balance.",
    "r_a": "Distance beyond which attraction saturates.",
    "f_rep": "Hard-core repulsion strength.",
    "A": "Spatial attraction strength.",
    "B": "Spatial repulsion strength.",
    "Ca": "Morse attraction strength.", "la": "Morse attraction length.",
    "Cr": "Morse repulsion strength.", "lr": "Morse repulsion length.",
    "cutoff": "Maximum interaction distance (blank = all pairs).",
    "zor": "Zone-of-repulsion radius (Couzin).",
    "zoo": "Zone-of-orientation radius (Couzin).",
    "zoa": "Zone-of-attraction radius (Couzin).",
    "blind": "Rear blind-angle the agent cannot see (radians).",
    "theta_dot_max": "Maximum turning rate.",
    "w_sep": "Weight of the separation rule.",
    "w_ali": "Weight of the alignment rule.",
    "w_coh": "Weight of the cohesion rule.",
    "r_sep": "Separation radius.", "r_ali": "Alignment radius.", "r_coh": "Cohesion radius.",
    "vmax": "Maximum speed.", "fmax": "Maximum acceleration (force clip).",
    "w_sep2": "Short-range separation weight.",
    "w_seg": "Out-group repulsion (segregation) weight.",
    "r_sep2": "Short-range separation radius.", "r_seg": "Out-group repulsion radius.",
    "w_global": "Weak pull toward the global centre (keeps groups in one arena).",
    "k_speed": "How strongly the cruising speed is enforced.",
    "switching": "Let agents change group allegiance dynamically.",
    "chi_": "Turning inertia.",
    "Gamma": "Damping of the perceptual field.",
    "T1": "Longitudinal relaxation time (memory).",
    "T2": "Transverse relaxation time.",
    "gamma_s": "Slow-register update rate.",
    "lam_fb": "Feedback strength of the slow register.",
    "rho": "Self-weight in the heading update.",
    "tau_p": "Perceptual time constant.",
    "s_eq": "Slow-register set-point.",
    "h_ext": "External bias field.",
    "d": "Desired inter-agent spacing (Olfati-Saber).",
    "kappa_ratio": "Interaction range / spacing ratio.",
    "eps": "Bump-function smoothing.",
    "h": "Bump-function shape parameter.",
    "a": "Attraction gain of the action function.",
    "b_": "Repulsion gain of the action function.",
    "c1_a": "Position feedback (flocking).", "c2_a": "Velocity feedback (flocking).",
    "c1_g": "Position feedback (navigation).", "c2_g": "Velocity feedback (navigation).",
    "omega_spread": "Spread of intrinsic turning rates (Kuramoto).",
}

# Measurement key -> explanation.
METRIC_HELP = {
    "t": "Simulation time elapsed.",
    "polar_order": "Alignment, 0–1: 1 means every agent heads the same way.",
    "milling": "Rotation, 0–1: 1 is a spinning mill (open/reflecting boundary only).",
    "radius_of_gyration": "How spread out the group is around its centre.",
    "n_fragments": "How many separate clusters exist (1 = a single flock).",
    "largest_cluster_frac": "Fraction of agents in the biggest cluster.",
    "mean_neighbors": "Average number of neighbours per agent.",
    "density": "Agents per unit area (2D) or volume (3D).",
    "heading_entropy": "Heading disorder, 0–1: 0 = aligned, 1 = fully random.",
    "mean_speed": "Average agent speed.",
    "fps": "Animation speed (frames per second).",
}

SETUP_HELP = {
    "model": "The collective-motion model to simulate (hover the description below).",
    "boundary": "periodic = wrap-around torus; open = infinite space; reflecting = walls.",
    "view": "Flat 2D view, or a rotatable 3D view (drag to rotate).",
    "box L": "Side length of the box (periodic/reflecting boundaries).",
    "init": "How agents start: random, clustered, on a ring, or on a grid.",
    "N": "Number of agents.",
    "groups": "Number of coloured groups / species.",
    "speed": "Initial speed of the agents.",
    "dt": "Time step per frame — smaller is more accurate but slower.",
}
DISPLAY_HELP = {
    "trajectories": "Draw fading trails behind each agent.",
    "vision cones": "Show each agent's field of view (2D; models with a vision cone).",
    "neighbour links": "Draw a line between every pair of interacting agents.",
    "group colours": "Colour agents by their group.",
}
BUTTON_HELP = {
    "Save config…": "Save all current settings to a file to reload or re-run later.",
    "Load config…": "Load settings from a saved config file.",
    "Screenshot…": "Save a picture of the canvas as a PNG.",
    "Export metrics…": "Save the recorded measurements as a CSV (needs a recording).",
    "Export trajectory…": "Save every agent's path as CSV or HDF5 (needs a recording).",
    "Plot metrics…": "Show charts of the recorded measurements over time.",
    "Sweep…": "Re-run the experiment while one parameter changes, and plot the result.",
    "Save GIF…": "Record a short animated GIF (choose length and speed).",
    "Save MP4…": "Record a video (choose length, frames-per-second, and quality).",
    "Place mode": "Turn on, then click the canvas to add agents by hand (2D only).",
    "Clear birds": "Remove all agents from the canvas.",
    "Load CSV…": "Load a starting arrangement of agents from a CSV file.",
    "● Record": "Start/stop capturing each frame (its measurements and positions) "
                "so you can export or plot them.",
}

# Numeric observables recorded per frame while "Record" is on (for CSV/HDF5 export).
_RECORD_KEYS = ["polar_order", "radius_of_gyration", "nn_distance", "n_fragments",
                "largest_cluster_frac", "mean_neighbors", "density",
                "heading_entropy", "mean_speed", "milling"]

# Time-series panels for the "Plot metrics" window: one measure per panel (never a
# dual-axis chart), each panel titled so it needs no legend. Colours are from the
# colour-blind-safe Okabe-Ito set, one per panel.
_PLOT_PANELS = [
    ("polar_order", "polar order M", "#0072B2"),
    ("milling", "milling", "#E69F00"),
    ("radius_of_gyration", "radius of gyration", "#009E73"),
    ("n_fragments", "fragments", "#D55E00"),
    ("heading_entropy", "heading entropy", "#CC79A7"),
    ("mean_speed", "mean speed", "#56B4E9"),
]

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
        self._place_rng = np.random.default_rng(12345)   # headings for placed birds

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

        # --- help ---
        help_btn = QtWidgets.QPushButton("❓  Help && guide")
        help_btn.setToolTip("Open a full explanation of the app, the models, and every control.")
        help_btn.clicked.connect(self._show_help)
        pl.addWidget(help_btn)

        # --- setup group (structural: change => reset) ---
        setup = QtWidgets.QGroupBox("Setup")
        form = QtWidgets.QFormLayout(setup)
        self.model_cb = QtWidgets.QComboBox(); self.model_cb.addItems(GUI_MODELS)
        self.boundary_cb = QtWidgets.QComboBox()
        self.boundary_cb.addItems(["periodic", "open", "reflecting"])
        self.boundary_cb.setCurrentText("open")
        self.dim_cb = QtWidgets.QComboBox(); self.dim_cb.addItems(["2D", "3D"])
        self.L_sb = QtWidgets.QDoubleSpinBox(); self.L_sb.setRange(1, 200); self.L_sb.setValue(10.0)
        self.init_cb = QtWidgets.QComboBox()
        self.init_cb.addItems(["random", "cluster", "ring", "grid"])
        self.N_sb = QtWidgets.QSpinBox(); self.N_sb.setRange(2, 3000); self.N_sb.setValue(150)
        self.groups_sb = QtWidgets.QSpinBox(); self.groups_sb.setRange(1, 8); self.groups_sb.setValue(1)
        self.speed_sb = QtWidgets.QDoubleSpinBox(); self.speed_sb.setRange(0, 10); self.speed_sb.setSingleStep(0.1); self.speed_sb.setValue(0.5)
        self.dt_sb = QtWidgets.QDoubleSpinBox(); self.dt_sb.setRange(0.001, 1.0); self.dt_sb.setSingleStep(0.01); self.dt_sb.setDecimals(3); self.dt_sb.setValue(0.05)
        for lab, w in [("model", self.model_cb), ("boundary", self.boundary_cb),
                       ("view", self.dim_cb), ("box L", self.L_sb),
                       ("init", self.init_cb), ("N", self.N_sb),
                       ("groups", self.groups_sb), ("speed", self.speed_sb),
                       ("dt", self.dt_sb)]:
            tip = SETUP_HELP.get(lab, "")
            w.setToolTip(tip)
            lw = QtWidgets.QLabel(lab); lw.setToolTip(tip)
            form.addRow(lw, w)
        pl.addWidget(setup)

        # a plain-language description of the selected model (updates on change)
        self.model_desc = QtWidgets.QLabel()
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color: palette(mid); font-size: 11px;"
                                      "padding: 2px 2px 6px 2px;")
        pl.addWidget(self.model_desc)

        self.model_cb.currentIndexChanged.connect(self._on_model_changed)
        for w in (self.boundary_cb, self.init_cb, self.dim_cb):
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
        self.play_btn.setToolTip("Run or pause the simulation.")
        self.step_btn.setToolTip("Advance the simulation by a single frame.")
        self.reset_btn.setToolTip("Rebuild the simulation with the current settings.")

        # --- display toggles ---
        disp = QtWidgets.QGroupBox("Display")
        dl = QtWidgets.QGridLayout(disp)
        self.cb_trails = QtWidgets.QCheckBox("trajectories"); self.cb_trails.setChecked(True)
        self.cb_cones = QtWidgets.QCheckBox("vision cones")
        self.cb_links = QtWidgets.QCheckBox("neighbour links")
        self.cb_groups = QtWidgets.QCheckBox("group colours"); self.cb_groups.setChecked(True)
        for i, cb in enumerate((self.cb_trails, self.cb_cones, self.cb_links, self.cb_groups)):
            dl.addWidget(cb, i // 2, i % 2)
            cb.setToolTip(DISPLAY_HELP.get(cb.text(), ""))
            cb.toggled.connect(self._apply_display)
        pl.addWidget(disp)

        # --- model parameters (dynamic) ---
        self.param_box = QtWidgets.QGroupBox("Model parameters (live)")
        self.param_form = QtWidgets.QFormLayout(self.param_box)
        pl.addWidget(self.param_box)

        # --- manual initial condition (click-to-place / load CSV) ---
        ic = QtWidgets.QGroupBox("Initial condition (manual)")
        il = QtWidgets.QGridLayout(ic)
        self.place_btn = QtWidgets.QPushButton("Place mode"); self.place_btn.setCheckable(True)
        self.clear_btn = QtWidgets.QPushButton("Clear birds")
        self.loadcsv_btn = QtWidgets.QPushButton("Load CSV…")
        self.place_group_sb = QtWidgets.QSpinBox(); self.place_group_sb.setRange(0, 7)
        il.addWidget(self.place_btn, 0, 0); il.addWidget(self.clear_btn, 0, 1)
        il.addWidget(self.loadcsv_btn, 1, 0)
        il.addWidget(QtWidgets.QLabel("place as group"), 2, 0)
        il.addWidget(self.place_group_sb, 2, 1)
        pl.addWidget(ic)
        self.place_btn.toggled.connect(self._toggle_place)
        self.clear_btn.clicked.connect(self._clear_birds)
        self.loadcsv_btn.clicked.connect(self._load_csv)
        self.place_btn.setToolTip(BUTTON_HELP["Place mode"])
        self.clear_btn.setToolTip(BUTTON_HELP["Clear birds"])
        self.loadcsv_btn.setToolTip(BUTTON_HELP["Load CSV…"])
        self.place_group_sb.setToolTip("Which group / colour newly placed agents get.")

        # --- experiment manager (config / export / capture) ---
        exp = QtWidgets.QGroupBox("Experiment")
        eg = QtWidgets.QGridLayout(exp)
        self.rec_btn = QtWidgets.QPushButton("● Record"); self.rec_btn.setCheckable(True)
        buttons = [
            ("Save config…", self._save_config), ("Load config…", self._load_config),
            ("Screenshot…", self._screenshot), (self.rec_btn, self._toggle_record),
            ("Export metrics…", self._export_metrics), ("Export trajectory…", self._export_traj),
            ("Plot metrics…", self._plot_metrics), ("Sweep…", self._sweep_dialog),
            ("Save GIF…", self._save_gif), ("Save MP4…", self._save_mp4),
        ]
        for i, (item, slot) in enumerate(buttons):
            btn = item if isinstance(item, QtWidgets.QPushButton) else QtWidgets.QPushButton(item)
            (btn.toggled if btn.isCheckable() else btn.clicked).connect(slot)
            btn.setToolTip(BUTTON_HELP.get(btn.text(), ""))
            eg.addWidget(btn, i // 2, i % 2)
        self.rec_status = QtWidgets.QLabel("not recording")
        eg.addWidget(self.rec_status, (len(buttons) + 1) // 2, 0, 1, 2)
        pl.addWidget(exp)

        # --- metrics ---
        met = QtWidgets.QGroupBox("Measurements")
        ml = QtWidgets.QFormLayout(met)
        self.metric_labels = {}
        for key, lab in _METRIC_ROWS:
            tip = METRIC_HELP.get(key, "")
            v = QtWidgets.QLabel("—"); v.setToolTip(tip)
            self.metric_labels[key] = v
            ll = QtWidgets.QLabel(lab); ll.setToolTip(tip)
            ml.addRow(ll, v)
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
            tip = PARAM_HELP.get(name, f"Model parameter '{name}'.")
            w.setToolTip(tip)
            if isinstance(w, FloatSlider):
                w.slider.setToolTip(tip); w.readout.setToolTip(tip)
            self.param_widgets[name] = (w, kind)
            lw = QtWidgets.QLabel(name); lw.setToolTip(tip)
            self.param_form.addRow(lw, w)
        self.model_desc.setText(MODEL_HELP.get(self.model_cb.currentText(), ""))
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
        name = self.model_cb.currentText()
        cls = GUI_MODELS[name]
        dim = 3 if self.dim_cb.currentText() == "3D" else 2
        if getattr(cls, "two_d_only", False) and dim == 3:   # angle/phase models are 2D
            dim = 2
            self.dim_cb.blockSignals(True); self.dim_cb.setCurrentText("2D")
            self.dim_cb.blockSignals(False)
            self.statusBar().showMessage(f"{name} is 2D-only — using 2D.", 4000)
        self.boundary = make_boundary(kind, L, dim)
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
            if self.recording and m is not None:
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
        if st.n < 2:                 # too few birds to measure (e.g. mid-placement)
            return None
        s = summarize(st.positions, st.headings, b, self.r_link,
                      velocities=st.velocities)
        s["t"] = st.t
        s["milling"] = milling_order(st.positions, st.velocities, b)
        s["fps"] = self._fps
        return s

    def _set_metric_labels(self, s):
        if s is None:
            for key, _ in _METRIC_ROWS:
                self.metric_labels[key].setText("—")
            return
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
        opts = self._ask_recording_opts("GIF")
        if not opts:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save GIF", "results/animation.gif", "GIF (*.gif)")
        if path:
            ok, msg = self._do_save_gif(path, opts["frames"], opts["fps"])
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

    #: named MP4 quality presets -> imageio ffmpeg quality (0..10, higher = better).
    _MP4_QUALITY = {"Low (small file)": 3, "Medium": 5, "High": 8}

    def _ask_recording_opts(self, kind):
        """Small modal dialog for length / fps / (MP4) quality. Returns dict or None."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{kind} settings")
        form = QtWidgets.QFormLayout(dlg)
        frames = QtWidgets.QSpinBox(); frames.setRange(5, 3000); frames.setValue(120)
        fps = QtWidgets.QSpinBox(); fps.setRange(1, 60)
        fps.setValue(25 if kind == "GIF" else 30)
        q_cb = None
        form.addRow("length (frames)", frames)
        form.addRow("frames per second", fps)
        if kind == "MP4":
            q_cb = QtWidgets.QComboBox(); q_cb.addItems(list(self._MP4_QUALITY))
            q_cb.setCurrentText("Low (small file)")     # default to a smaller file
            form.addRow("quality", q_cb)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        out = {"frames": frames.value(), "fps": fps.value()}
        if q_cb is not None:
            out["quality"] = self._MP4_QUALITY[q_cb.currentText()]
        return out

    def _save_mp4(self):
        opts = self._ask_recording_opts("MP4")
        if not opts:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save MP4", "results/animation.mp4", "MP4 (*.mp4)")
        if path:
            ok, msg = self._do_save_mp4(path, opts["frames"], opts["fps"],
                                        opts["quality"])
            (self.statusBar().showMessage if ok else self._warn)(
                f"Saved MP4 -> {path}" if ok else msg)

    def _do_save_mp4(self, path, frames=120, fps=30, quality=3):
        """Step the sim `frames` times into an MP4. `quality` is 0..10 (lower =
        smaller file / lower quality)."""
        try:
            import imageio.v2 as imageio
            from PIL import Image
        except Exception:
            return False, "imageio not installed (pip install imageio imageio-ffmpeg)."
        try:
            writer = imageio.get_writer(str(path), fps=fps, quality=quality,
                                        macro_block_size=None)
        except Exception as e:                       # usually a missing ffmpeg backend
            return False, f"Could not open the video writer: {e}"
        for _ in range(frames):
            self._tick()
            writer.append_data(np.asarray(self._grab_pil(Image)))
        writer.close()
        return True, path

    def _grab_pil(self, Image):
        img = self.canvas.grab().toImage().convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        buf = np.frombuffer(bytes(img.constBits()), np.uint8)
        buf = buf.reshape((h, img.bytesPerLine() // 4, 4))[:, :w, :]
        return Image.fromarray(buf, "RGBA").convert("RGB")

    # -- analysis plots ---------------------------------------------------
    def _build_metrics_figure(self):
        """A matplotlib Figure of the recorded time-series, or None if nothing recorded."""
        h = self._history_dict()
        if h is None or len(h["t"]) < 2:
            return None
        from matplotlib.figure import Figure
        t = np.asarray(h["t"])
        fig = Figure(figsize=(8.2, 5.2))
        fig.suptitle(f"{self.model_cb.currentText()}  ·  "
                     f"{self.boundary_cb.currentText()} boundary  ·  "
                     f"{len(t)} recorded frames", fontsize=11)
        for i, (key, label, color) in enumerate(_PLOT_PANELS):
            ax = fig.add_subplot(2, 3, i + 1)
            data = np.asarray(h[key], dtype=float) if key in h else None
            if data is not None and np.any(np.isfinite(data)):
                ax.plot(t, data, color=color, lw=1.8)
            else:                                # e.g. milling is undefined on a torus
                ax.text(0.5, 0.5, "n/a here", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.6")
            ax.set_title(label, fontsize=9)      # single series -> title names it, no legend
            ax.set_xlabel("time", fontsize=8)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=7)
            for side in ("top", "right"):        # recessive axes
                ax.spines[side].set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def _do_plot_metrics(self, save_path):
        """Render the recorded-metrics figure to an image file. Returns True on success."""
        fig = self._build_metrics_figure()
        if fig is None:
            return False
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)                     # attach an offscreen canvas
        fig.savefig(str(save_path), dpi=120)
        return True

    # -- help / guide -----------------------------------------------------
    @staticmethod
    def _help_rows(mapping):
        return "".join(
            f"<tr><td valign='top'><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in mapping.items())

    def _build_help_html(self):
        t = "<table cellspacing='0' cellpadding='4'>{}</table>".format
        metric_labeled = {lab: METRIC_HELP[key] for key, lab in _METRIC_ROWS
                          if key in METRIC_HELP}
        return f"""
        <h2>open-collective — BirdFlockLab</h2>
        <p>An interactive lab for <i>collective motion</i>: pick a model, watch a
        flock evolve, measure it, and export the results. Nothing here changes the
        physics — every number you see is what the underlying engine computes.</p>

        <h3>Get a result in four steps</h3>
        <ol>
          <li><b>Choose a model</b> and a <b>boundary</b> in the Setup box (its
              description appears just below).</li>
          <li>Press <b>Play</b> (or <b>Step</b>) to run it. Drag the sliders to
              change parameters live.</li>
          <li>Press <b>● Record</b>, let it run, then <b>Plot metrics</b> or
              <b>Export</b> to save the data.</li>
          <li><b>Sweep</b> re-runs it while one parameter changes and plots how a
              measurement responds.</li>
        </ol>

        <h3>Canvas</h3>
        <p>Each agent is a dot with a short line for its heading. <b>Scroll</b> to
        zoom; <b>drag</b> to move the view in 2D or to rotate it in 3D. In
        <i>Place mode</i> a click adds an agent (2D).</p>

        <h3>Models</h3>
        {t(self._help_rows(MODEL_HELP))}

        <h3>Setup</h3>
        {t(self._help_rows(SETUP_HELP))}

        <h3>Display overlays</h3>
        {t(self._help_rows(DISPLAY_HELP))}

        <h3>Experiment — record, export, analyse</h3>
        {t(self._help_rows(BUTTON_HELP))}

        <h3>Measurements (live, top-right)</h3>
        {t(self._help_rows(metric_labeled))}

        <h3>Model parameters</h3>
        <p>Each model shows sliders for its own parameters; the panel builds them
        automatically. Common ones:</p>
        {t(self._help_rows(PARAM_HELP))}

        <p style='color:gray'>Tip: hover any control in the app for the same help
        as a pop-up tooltip.</p>
        """

    def _show_help(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Help & guide")
        dlg.resize(640, 720)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(self._build_help_html())
        QtWidgets.QVBoxLayout(dlg).addWidget(view)
        dlg.show()
        self._open_dialogs = getattr(self, "_open_dialogs", [])
        self._open_dialogs.append(dlg)

    def _show_figure(self, fig, title, size=(860, 560)):
        """Show a matplotlib Figure in a non-modal Qt dialog."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(*size)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        QtWidgets.QVBoxLayout(dlg).addWidget(FigureCanvas(fig))
        dlg.show()
        self._open_dialogs = getattr(self, "_open_dialogs", [])
        self._open_dialogs.append(dlg)           # keep a ref so it isn't garbage-collected

    def _plot_metrics(self):
        fig = self._build_metrics_figure()
        if fig is None:
            return self._warn("Nothing recorded yet. Press ● Record, run it, then plot.")
        self._show_figure(fig, "Recorded measurements")

    # -- parameter sweep --------------------------------------------------
    def _run_sweep(self, param, values, steps, metric_key):
        """Run one batch per value of `param`; return manager.parameter_sweep rows.

        Uses the current GUI setup as the base config, so a sweep is just 'this
        experiment, repeated while one number changes'.
        """
        base = self.current_config()
        base["run"] = dict(base["run"])
        base["run"]["steps"] = int(steps)
        base["run"]["record_traj"] = False
        base["run"]["record_every"] = max(1, int(steps) // 25)
        return manager.parameter_sweep(base, f"model.params.{param}",
                                       [float(v) for v in values],
                                       metric_keys=[metric_key])

    def _build_sweep_figure(self, param, rows, metric_key, metric_label):
        from matplotlib.figure import Figure
        key = f"model.params.{param}"
        xs = [r[key] for r in rows]
        ys = [r.get(metric_key, float("nan")) for r in rows]
        fig = Figure(figsize=(6.4, 4.4))
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, "o-", color="#0072B2", lw=1.8, ms=6)
        ax.set_xlabel(param)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{self.model_cb.currentText()}: {metric_label} vs {param}",
                     fontsize=11)
        ax.grid(True, alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        return fig

    def _do_sweep(self, param, lo, hi, points, steps, metric_key, save_path=None):
        """Headless-friendly: run a sweep, build the figure, optionally save it."""
        values = np.linspace(lo, hi, int(points))
        rows = self._run_sweep(param, values, steps, metric_key)
        fig = self._build_sweep_figure(param, rows, metric_key, metric_key)
        if save_path is not None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(fig)
            fig.savefig(str(save_path), dpi=120)
        return rows

    def _sweep_dialog(self):
        params = [n for n, (w, k) in self.param_widgets.items() if k in ("float", "int")]
        if not params:
            return self._warn("This model has no numeric parameters to sweep.")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Parameter sweep")
        form = QtWidgets.QFormLayout(dlg)
        p_cb = QtWidgets.QComboBox(); p_cb.addItems(params)
        lo = QtWidgets.QDoubleSpinBox(); lo.setRange(-1e6, 1e6); lo.setDecimals(3)
        hi = QtWidgets.QDoubleSpinBox(); hi.setRange(-1e6, 1e6); hi.setDecimals(3)
        pts = QtWidgets.QSpinBox(); pts.setRange(2, 20); pts.setValue(6)
        steps = QtWidgets.QSpinBox(); steps.setRange(20, 5000); steps.setValue(200)
        m_cb = QtWidgets.QComboBox()
        for _, label, _c in _PLOT_PANELS:
            m_cb.addItem(label)
        run = QtWidgets.QPushButton("Run sweep")
        for lab, w in [("parameter", p_cb), ("from", lo), ("to", hi),
                       ("points", pts), ("steps / run", steps), ("plot", m_cb)]:
            form.addRow(lab, w)
        form.addRow(run)

        def sync_range():
            w, _k = self.param_widgets[p_cb.currentText()]
            cur = float(w.value())
            lo.setValue(0.0)
            hi.setValue(max(1.0, 2.0 * cur) if cur else 1.0)
        p_cb.currentTextChanged.connect(lambda *_: sync_range())
        sync_range()

        def do_run():
            key = next(k for k, lab, _c in _PLOT_PANELS if lab == m_cb.currentText())
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                rows = self._do_sweep(p_cb.currentText(), lo.value(), hi.value(),
                                      pts.value(), steps.value(), key)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            fig = self._build_sweep_figure(p_cb.currentText(), rows, key,
                                           m_cb.currentText())
            dlg.accept()
            self._show_figure(fig, f"Sweep: {m_cb.currentText()} vs {p_cb.currentText()}",
                              size=(660, 500))
        run.clicked.connect(do_run)
        dlg.show()
        self._open_dialogs = getattr(self, "_open_dialogs", [])
        self._open_dialogs.append(dlg)

    def _warn(self, msg):
        QtWidgets.QMessageBox.warning(self, "open-collective", msg)

    # -- manual initial condition -----------------------------------------
    def _rebuild_model_keep_state(self):
        """Reconstruct the model for the current state (drops stale internal caches)."""
        cls = GUI_MODELS[self.model_cb.currentText()]
        if cls is MultiGroupFlock and "groups" not in self.state.internal:
            self.state.internal["groups"] = cinit.assign_groups(
                self.state.n, max(2, int(self.groups_sb.value())))
        self.model = cls(self.boundary, rng=np.random.default_rng(0),
                         **self._param_values())

    def _toggle_place(self, on):
        if on and self.state is not None and self.state.dim == 3:
            self.place_btn.setChecked(False)         # click-to-place is 2D only
            return self._warn("Place mode works in the 2D view only.")
        if on and self.play_btn.isChecked():
            self.play_btn.setChecked(False)          # pause while placing
        self.canvas.place_mode = on
        self.canvas.on_place = self._place_bird if on else None
        self.statusBar().showMessage(
            "Place mode ON — click the canvas to add birds." if on else "", 4000)

    def _place_bird(self, wx, wy):
        ang = float(self._place_rng.uniform(0, 2 * np.pi))
        speed = float(self.speed_sb.value())
        v = speed * np.array([[np.cos(ang), np.sin(ang)]])
        self.state.positions = np.vstack([self.state.positions, [[wx, wy]]])
        self.state.velocities = np.vstack([self.state.velocities, v])
        g = np.asarray(self.state.internal.get("groups", np.zeros(0, dtype=int)),
                       dtype=int)
        g = np.append(g, int(self.place_group_sb.value()))
        self.state.internal = {"groups": g}          # keep groups, drop model caches
        self._rebuild_model_keep_state()
        self._refresh_canvas()
        self._update_metrics()

    def _clear_birds(self, *_):
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        self.state = State(positions=np.zeros((0, 2)), velocities=np.zeros((0, 2)))
        self.state.internal["groups"] = np.zeros(0, dtype=int)
        self._rebuild_model_keep_state()
        self.canvas.clear_trails()
        self._refresh_canvas()
        self._update_metrics()

    def _load_csv(self, *_):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load initial condition (CSV)", "results", "CSV (*.csv)")
        if path:
            self._do_load_csv(path)

    def _do_load_csv(self, path):
        st = cinit.from_csv(path)
        if st.dim != 2:
            return self._warn("The CSV must be 2D (needs x and y columns).")
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        self.state = st
        self._steps = 0
        self._rebuild_model_keep_state()
        self.canvas.clear_trails()
        self.canvas.fit(st.positions, self.boundary)
        self._refresh_canvas()
        self._update_metrics()
        self.statusBar().showMessage(f"Loaded {st.n} birds from {path}", 4000)
