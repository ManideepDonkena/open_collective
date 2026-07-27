"""
Smoke test / demo for the BirdFlockLab increment (option b):

  * core.init          -- random / cluster / ring / grid / manual / CSV initializers
  * core.metrics       -- mean_speed, density, heading_entropy
  * models.KuramotoModel
  * experiments.manager-- config save/load, CSV/HDF5 export, parameter sweep

Run from the repo root:  python experiments/demo_new_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import run, make_boundary
from core.metrics import summarize
import core.init as cinit
from models import KuramotoModel, VicsekModel
from experiments import manager

OUT = Path("results"); OUT.mkdir(exist_ok=True)


def hr(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def demo_init():
    hr("1. INITIALIZATION MODULE  (core/init.py)")
    periodic = make_boundary("periodic", L=10.0, dim=2)
    open_b = make_boundary("open", dim=2)
    states = {
        "random (2 groups)": cinit.random_init(60, periodic, n_groups=2),
        "cluster (3 blobs)": cinit.cluster_init(60, open_b, n_clusters=3,
                                                aligned_within_cluster=True),
        "ring (tangential)": cinit.ring_init(60, open_b, radius=4.0, tangential=True),
        "grid": cinit.grid_init(60, periodic, spacing=1.0),
        "manual": cinit.manual_init(np.random.default_rng(0).uniform(0, 5, (10, 2))),
    }
    for name, st in states.items():
        g = st.internal.get("groups")
        ng = len(np.unique(g)) if g is not None else 1
        print(f"  {name:22s} N={st.n:3d} dim={st.dim} groups={ng} "
              f"speed~{st.speeds.mean():.3f}")

    # CSV round trip
    csv_path = OUT / "init_snapshot.csv"
    cinit.state_to_csv(states["random (2 groups)"], csv_path)
    reloaded = cinit.from_csv(csv_path)
    match = np.allclose(reloaded.positions, states["random (2 groups)"].positions)
    print(f"  CSV round-trip -> {csv_path}  positions match: {match}")


def demo_metrics_and_kuramoto():
    hr("2. NEW METRICS + KURAMOTO MODEL")
    b = make_boundary("periodic", L=10.0, dim=2)
    st = cinit.random_init(150, b, speed=0.5)
    model = KuramotoModel(b, r_max=1.5, K=3.0, eta=0.05, v0=0.5)
    final, hist = run(model, st, steps=400, dt=0.05, r_link=1.5, record_every=20)
    s0 = summarize(st.positions, st.headings, b, 1.5, velocities=st.velocities)
    sT = summarize(final.positions, final.headings, b, 1.5, velocities=final.velocities)
    print(f"  Kuramoto sync:  polar_order {s0['polar_order']:.3f} -> {sT['polar_order']:.3f}")
    print(f"                  heading_entropy {s0['heading_entropy']:.3f} -> "
          f"{sT['heading_entropy']:.3f}  (should fall as it syncs)")
    print(f"                  density {sT['density']:.3f}   mean_speed {sT['mean_speed']:.3f}")


def demo_manager():
    hr("3. EXPERIMENT MANAGER  (config / export / sweep)")
    cfg = {
        "seed": 1,
        "boundary": {"kind": "periodic", "L": 10.0, "dim": 2},
        "init": {"method": "random", "n": 120, "speed": 0.5},
        "model": {"name": "Vicsek", "params": {"r_max": 1.0, "eta": 0.3, "v0": 0.5}},
        "run": {"steps": 300, "dt": 0.05, "r_link": 1.0, "record_every": 20,
                "record_traj": True},
    }
    # save / load config round-trip
    cpath = manager.save_config(cfg, OUT / "demo_config.json")
    cfg2 = manager.load_config(cpath)
    print(f"  config saved+loaded: {cpath}  identical: {cfg == cfg2}")

    final, hist = manager.run_experiment(cfg2)
    mpath = manager.export_measurements(hist, OUT / "demo_measurements.csv")
    tpath = manager.export_trajectory(hist, OUT / "demo_trajectory.csv")
    print(f"  measurements -> {mpath}  ({len(hist['t'])} frames)")
    print(f"  trajectory   -> {tpath}  "
          f"({hist['trajectory'].shape[0]}x{hist['trajectory'].shape[1]} = "
          f"{hist['trajectory'].shape[0]*hist['trajectory'].shape[1]} rows)")

    hr("4. PARAMETER SWEEP  (Vicsek noise eta)")
    rows = manager.parameter_sweep(cfg, "model.params.eta",
                                   [0.0, 0.3, 0.6, 1.0, 1.5])
    print(f"  {'eta':>6} {'polar_order':>12} {'heading_entropy':>16} {'n_frag':>7}")
    for r in rows:
        print(f"  {r['model.params.eta']:>6.2f} {r['polar_order']:>12.3f} "
              f"{r['heading_entropy']:>16.3f} {int(r['n_fragments']):>7d}")


if __name__ == "__main__":
    demo_init()
    demo_metrics_and_kuramoto()
    demo_manager()
    print("\nAll demos completed.\n")
