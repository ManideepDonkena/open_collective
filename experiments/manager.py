"""
Experiment Manager -- BirdFlockLab module 7.

One place to turn a plain dict / JSON / YAML config into a runnable experiment, and
to get the results back out as data:

    config (dict / .json / .yaml)
        -> build_experiment()  -> (model, state, run_kwargs)
        -> core.run()          -> (final_state, history)
        -> export_measurements()  (scalar time series -> CSV)
        -> export_trajectory()    (full trajectory   -> CSV or HDF5)

Plus `parameter_sweep()` for batch runs over one config field.

Config schema
-------------
    {
      "seed": 0,
      "boundary": {"kind": "open", "L": 10.0, "dim": 2},
      "init":  {"method": "random", "n": 120, "speed": 0.5, "spread": 10.0},
      "model": {"name": "Vicsek", "params": {"r_max": 1.0, "eta": 0.2, "v0": 0.5}},
      "run":   {"steps": 400, "dt": 0.05, "r_link": 1.0, "record_every": 10,
                "record_traj": false}
    }

`init.method` is a key of `core.init.INIT_REGISTRY`; its other fields are passed
straight to that initializer (for method "csv", pass "path" instead of "n").
`model.name` is a key of `MODEL_REGISTRY` below; `model.params` are its kwargs.

Optional dependencies: PyYAML (only for .yaml configs) and h5py (only for .h5
trajectory export). JSON and CSV need nothing beyond the standard library.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from core import run
from core.boundary import make_boundary
from core.init import INIT_REGISTRY, _POS_COLS, _VEL_COLS
from core.metrics import summarize
from models import (VicsekModel, PerceptionQuantum, SlowFastPerception,
                    KuramotoModel, BoidsModel, CouzinModel, DOrsognaModel,
                    CuckerSmaleModel, OlfatiSaberModel, MultiGroupFlock,
                    DisplacementFormation, DistanceFormation, LeaderFollower,
                    CyclicPursuit)


#: config name -> model class. Every entry is a CollectiveModel whose constructor
#: is (boundary, ..., rng=None) and which drives through core.run().
MODEL_REGISTRY = {
    "Vicsek": VicsekModel,
    "Perception": PerceptionQuantum,
    "SlowFast": SlowFastPerception,
    "Kuramoto": KuramotoModel,
    "Boids": BoidsModel,
    "Couzin": CouzinModel,
    "DOrsogna": DOrsognaModel,
    "CuckerSmale": CuckerSmaleModel,
    "OlfatiSaber": OlfatiSaberModel,
    "MultiGroupFlock": MultiGroupFlock,
    "DisplacementFormation": DisplacementFormation,
    "DistanceFormation": DistanceFormation,
    "LeaderFollower": LeaderFollower,
    "CyclicPursuit": CyclicPursuit,
}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_experiment(config: dict):
    """Config dict -> (model, state, run_kwargs). Pure; runs nothing."""
    cfg = copy.deepcopy(config)
    seed = int(cfg.get("seed", 0))

    bcfg = dict(cfg["boundary"])
    boundary = make_boundary(bcfg.get("kind", "open"),
                             bcfg.get("L", 10.0), bcfg.get("dim", 2))

    icfg = dict(cfg["init"])
    method = icfg.pop("method", "random")
    if method not in INIT_REGISTRY:
        raise ValueError(f"Unknown init method {method!r}. "
                         f"Choose from {list(INIT_REGISTRY)}")
    init_fn = INIT_REGISTRY[method]
    init_rng = np.random.default_rng(int(icfg.pop("seed", seed)))
    if method == "csv":
        state = init_fn(icfg.pop("path"), rng=init_rng, **icfg)
    else:
        n = int(icfg.pop("n"))
        state = init_fn(n, boundary, rng=init_rng, **icfg)

    mcfg = dict(cfg["model"])
    if mcfg["name"] not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {mcfg['name']!r}. "
                         f"Choose from {list(MODEL_REGISTRY)}")
    ModelCls = MODEL_REGISTRY[mcfg["name"]]
    model = ModelCls(boundary, rng=np.random.default_rng(seed),
                     **dict(mcfg.get("params", {})))

    run_kwargs = dict(cfg.get("run", {}))
    return model, state, run_kwargs


def run_experiment(config: dict):
    """build_experiment() + core.run(). Returns (final_state, history)."""
    model, state, run_kwargs = build_experiment(config)
    return run(model, state, **run_kwargs)


# --------------------------------------------------------------------------
# config I/O
# --------------------------------------------------------------------------

def save_config(config: dict, path) -> Path:
    """Write config as JSON (default) or YAML (.yaml/.yml, needs PyYAML)."""
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        path.write_text(json.dumps(config, indent=2))
    return path


def load_config(path) -> dict:
    """Read a JSON or YAML config."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency
        return yaml.safe_load(text)
    return json.loads(text)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_measurements(history: dict, path) -> Path:
    """Write every scalar time series in `history` to a wide CSV (one row / frame)."""
    path = Path(path)
    keys = ["t"] + [k for k, v in history.items()
                    if k != "t" and np.asarray(v).ndim == 1]
    T = len(history["t"])
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(T):
            w.writerow([history[k][i] for k in keys])
    return path


def export_trajectory(history: dict, path) -> Path:
    """Write the full trajectory to CSV (long format) or HDF5 (.h5/.hdf5, needs h5py).

    Requires the history to have been produced with record_traj=True.
    """
    path = Path(path)
    if "trajectory" not in history:
        raise ValueError("history has no 'trajectory' -- run with record_traj=True")
    traj = np.asarray(history["trajectory"])           # (T, N, dim)
    vtraj = history.get("velocity_trajectory")
    gtraj = history.get("group_trajectory")
    tvals = np.asarray(history.get("t"))
    T, N, dim = traj.shape

    if path.suffix.lower() in (".h5", ".hdf5"):
        import h5py  # optional dependency
        with h5py.File(path, "w") as f:
            f.create_dataset("trajectory", data=traj, compression="gzip")
            if vtraj is not None:
                f.create_dataset("velocity", data=np.asarray(vtraj),
                                 compression="gzip")
            if tvals is not None and tvals.size:
                f.create_dataset("t", data=tvals)
            if gtraj is not None:
                f.create_dataset("groups", data=np.asarray(gtraj))
        return path

    header = ["frame", "t", "agent"] + _POS_COLS[dim]
    if vtraj is not None:
        header += _VEL_COLS[dim]
    if gtraj is not None:
        header += ["group"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for fr in range(T):
            for a in range(N):
                row = [fr, float(tvals[fr]) if tvals.size else fr, a]
                row += list(traj[fr, a])
                if vtraj is not None:
                    row += list(np.asarray(vtraj)[fr, a])
                if gtraj is not None:
                    row.append(int(np.asarray(gtraj)[fr, a]))
                w.writerow(row)
    return path


# --------------------------------------------------------------------------
# parameter sweep / batch
# --------------------------------------------------------------------------

def _set_path(cfg: dict, dotted: str, value):
    """Set cfg["a"]["b"]["c"] = value for dotted = 'a.b.c', creating dicts."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parameter_sweep(base_config: dict, dotted_param: str, values,
                    metric_keys=None):
    """Run `base_config` once per value of one dotted field; return final metrics.

    Example
    -------
        parameter_sweep(cfg, "model.params.eta", [0.0, 0.2, 0.5, 1.0])

    Returns a list of dicts, one per value: {dotted_param: value, **final_metrics},
    where final_metrics is the last recorded frame of the scalar time series.
    """
    default_keys = ["polar_order", "radius_of_gyration", "n_fragments",
                    "largest_cluster_frac", "mean_neighbors", "density",
                    "heading_entropy", "mean_speed"]
    metric_keys = metric_keys or default_keys
    rows = []
    for val in values:
        cfg = copy.deepcopy(base_config)
        _set_path(cfg, dotted_param, val)
        final, hist = run_experiment(cfg)
        row = {dotted_param: val}
        for k in metric_keys:
            row[k] = float(hist[k][-1]) if k in hist and len(hist[k]) else float("nan")
        rows.append(row)
    return rows
