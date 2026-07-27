"""
3D VISUALISATION OF THE MODELS -- see the collapse instead of reading about it.
==============================================================================

Every claim in this repo is a claim about a picture, and until now the pictures were
bar charts of the summary statistics. This script renders the runs themselves, in
3D, with the numbers burned into each frame so that the movie and the metrics cannot
drift apart.

    python experiments/viz3d_demo.py            # the four headline scenes
    python experiments/viz3d_demo.py --scene groups
    python experiments/viz3d_demo.py --list
    python experiments/viz3d_demo.py --quick    # short, coarse, for a fast look

Scenes
------
    collapse : Vicsek in a periodic box, then the SAME model in open space. This is
               the repo's headline result, as a movie. On the torus it is a textbook
               ordered flock; delete the walls and it tears itself into fragments
               that each order independently -- and the global M stays high while it
               happens, which is the whole point.
    survives : Cucker-Smale beta=0.4 in open space. The control. Same boundary as
               the second half of `collapse`, and it simply does not spread.
    groups   : MultiGroupFlock, K=3, open space. Group maintenance: three flocks in
               one airspace, each cohesive, none merging.
    fusion   : the same model with segregation off. THE ONE TO WATCH. The groups
               merge into a single blob -- and the Experiment-1 cohesion metrics in
               the HUD read a perfect 1.00 the entire time they do it.

Output: results/viz3d_<scene>.gif plus a _panels.png still strip for each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import init_state, make_boundary
from core.metrics import (fragmentation, group_summary, polar_order,
                          radius_of_gyration)
from core.viz3d import render
from models import CuckerSmaleModel, MultiGroupFlock, VicsekModel, init_grouped_state

RESULTS = Path(__file__).resolve().parents[1] / "results"

N = 120
DIM = 3
DT = 0.05
R_LINK = 1.5

#: Box length for the 3D scenes. NOT exp1's L=10, and the difference matters.
#: exp1 is 2D, where N=120 in L=10 gives rho=1.2 and ~8.4 neighbours inside r_link.
#: Re-using L=10 in 3D gives rho=0.12 and ~1.7 neighbours: the flock starts as 43
#: disconnected fragments, before a single step is taken. That scene would look like
#: a dramatic collapse and would be measuring DILUTION, not the boundary condition.
#: L=6 restores rho=0.56 and ~8.0 neighbours -- matched to the 2D experiment on the
#: quantity the interaction actually depends on -- so the only thing left varying
#: between the periodic and open runs is the topology, which is the entire point.
L = 6.0

#: Default run length. 4000 steps at dt=0.05 is t = 200 -- long enough that the
#: outcomes have actually settled rather than merely started: Vicsek's open-space
#: fragments separate and stop interacting, and the no-segregation groups reach the
#: S ~ 0.50 plateau instead of being caught mid-merge. Override with --time/--steps.
STEPS = 4000

#: Rendered frames, held constant as --time grows (see main()). A longer simulation
#: should buy more simulated time, not a larger GIF. 150 frames at 8 fps is ~19 s of
#: playback and lands around 10 MB; 220 frames at dpi 110 produced a 33 MB file for
#: the periodic scene, which is a file nobody opens twice.
TARGET_FRAMES = 150
DPI = 95


def _flock_hud(hist, boundary):
    """HUD showing exactly the Experiment-1 metrics, live, frame by frame."""
    traj, vtraj = hist["trajectory"], hist["velocity_trajectory"]

    def fn(k):
        x, v = traj[k], vtraj[k]
        h = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        nf, lg = fragmentation(x, boundary, R_LINK)
        return (f"M       = {polar_order(h):.3f}\n"
                f"R_g     = {radius_of_gyration(x, boundary):5.2f}\n"
                f"frag    = {nf}\n"
                f"largest = {lg:.2f}")
    return fn


def _group_hud(hist, boundary):
    """HUD for the multi-group scenes: group metrics AND the cohesion metrics that
    miss fusion, side by side, so the discrepancy is visible as it opens up."""
    traj, vtraj = hist["trajectory"], hist["velocity_trajectory"]
    gtraj = hist["group_trajectory"]

    def fn(k):
        x, v, g = traj[k], vtraj[k], gtraj[k]
        h = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        gs = group_summary(x, h, g, boundary, R_LINK)
        nf, lg = fragmentation(x, boundary, R_LINK)
        return (f"segregation S = {gs['segregation_index']:.2f}   <- groups distinct?\n"
                f"integrity   I = {gs['group_integrity']:.2f}\n"
                f"largest       = {lg:.2f}   <- exp-1 cohesion:\n"
                f"fragments     = {nf}      blind to fusion")
    return fn


def _render(model, state, steps, out, title, hud_kind, **kw):
    """Render, then attach the HUD. The HUD needs the recorded run, so the run is
    done first (cheap) and the movie is written from it."""
    from core import run as _run
    from core.viz3d import animate3d, panels3d

    _, hist = _run(model, state, steps=steps, dt=DT, r_link=R_LINK,
                   record_every=kw.pop("record_every", 10), record_traj=True)
    hud = (_group_hud(hist, model.boundary) if hud_kind == "group"
           else _flock_hud(hist, model.boundary))
    kw.setdefault("dpi", DPI)
    kw.setdefault("fps", 8)
    animate3d(hist, model.boundary, out=str(out), title=title, hud_fn=hud, **kw)
    panels3d(hist, model.boundary, out=str(Path(out).with_suffix("")) + "_panels.png",
             title=title)
    return hist


def scene_collapse(steps, record_every):
    """Vicsek: torus vs open space. The headline result, as a movie."""
    for bc in ("periodic", "open"):
        b = make_boundary(bc, L=L, dim=DIM)
        st = init_state(N, b, dim=DIM, speed=0.5, spread=L,
                        rng=np.random.default_rng(0))
        m = VicsekModel(b, r_max=R_LINK, eta=0.2, v0=0.5,
                        rng=np.random.default_rng(1))
        tag = ("PERIODIC: walls guarantee neighbours -> textbook flock"
               if bc == "periodic"
               else "OPEN: no walls, no attraction -> it shatters (watch M lie)")
        _render(m, st, steps, RESULTS / f"viz3d_collapse_{bc}.gif",
                f"Vicsek -- {tag}", "flock", record_every=record_every)


def scene_survives(steps, record_every):
    """Cucker-Smale beta=0.4, open space. The control that does not evaporate."""
    b = make_boundary("open", L=L, dim=DIM)
    st = init_state(N, b, dim=DIM, speed=0.5, spread=L,
                    rng=np.random.default_rng(0))
    m = CuckerSmaleModel(b, K=2.0, sigma=1.0, beta=0.4,
                         rng=np.random.default_rng(1))
    _render(m, st, steps, RESULTS / "viz3d_survives.gif",
            "Cucker-Smale beta=0.4, OPEN -- psi(r)>0 forever, so nobody is ever lost",
            "flock", record_every=record_every)


def _group_scene(w_seg, name, title, steps, record_every):
    b = make_boundary("open", L=12.0, dim=DIM)
    st = init_grouped_state(150, 3, b, dim=DIM, speed=0.5, blob_radius=0.9,
                            arena=8.0, rng=np.random.default_rng(0))
    m = MultiGroupFlock(b, beta=0.4, w_seg=w_seg, w_global=0.05,
                        rng=np.random.default_rng(1))
    _render(m, st, steps, RESULTS / f"viz3d_{name}.gif", title, "group",
            record_every=record_every)


def scene_groups(steps, record_every):
    """Three groups, one airspace, no walls. Maintained."""
    _group_scene(2.5, "groups",
                 "Multi-group flock, OPEN -- 3 groups coexist, none merge",
                 steps, record_every)


def scene_fusion(steps, record_every):
    """The same model with the anti-fusion term deleted."""
    _group_scene(0.0, "fusion",
                 "Same model, segregation OFF -- groups FUSE while 'largest'=1.00",
                 steps, record_every)


SCENES = {
    "collapse": scene_collapse,
    "survives": scene_survives,
    "groups": scene_groups,
    "fusion": scene_fusion,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES) + ["all"], default="all")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help=f"integration steps (default {STEPS} = t {STEPS * DT:.0f})")
    ap.add_argument("--time", type=float, default=None,
                    help="simulated duration; overrides --steps (steps = time/dt)")
    ap.add_argument("--record-every", type=int, default=None,
                    help="steps per rendered frame (default: auto, ~%d frames)"
                         % TARGET_FRAMES)
    ap.add_argument("--quick", action="store_true",
                    help="short coarse render, for a fast look")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    a = ap.parse_args()

    if a.list:
        for k, f in SCENES.items():
            print(f"  {k:<10} {f.__doc__.splitlines()[0]}")
        return

    if a.quick:
        steps = 600
    elif a.time is not None:
        steps = max(1, int(round(a.time / DT)))
    else:
        steps = a.steps
    # Frame count is held ~constant however long the run is, so that asking for a
    # longer simulation buys more SIMULATED TIME rather than a bigger, slower file.
    rec = a.record_every or max(1, round(steps / TARGET_FRAMES))
    print(f"steps={steps}  dt={DT}  simulated time={steps * DT:.1f}  "
          f"frames={steps // rec}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = list(SCENES) if a.scene == "all" else [a.scene]
    for name in todo:
        print(f"\n--- scene: {name} ---")
        SCENES[name](steps, rec)


if __name__ == "__main__":
    main()
