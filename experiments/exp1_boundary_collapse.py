"""
EXPERIMENT 1 -- THE HEADLINE RESULT
===================================

Claim under test
----------------
"Current flocking/formation models are simulated on a periodic torus. They will not
 work in open space."

Design
------
Every model is run TWICE from a *statistically identical* initial condition:
  * PERIODIC : box of side L, torus topology
  * OPEN     : same initial blob of side L, but in unbounded R^2

At t = 0 the two runs are indistinguishable: same N, same density N/L^2, same
random headings. The ONLY difference is the topology. Every divergence afterwards
is therefore *caused by the walls* and by nothing else.

What we measure
---------------
  M            polar order            (what the literature reports)
  R_g          radius of gyration     (what the literature omits)
  frag         number of components   (is it even one flock any more?)
  largest      largest cluster frac
  <n_i>        mean neighbour count   (has the interaction switched off?)

The prediction
--------------
Alignment-only models (Vicsek, and BOTH Beuria papers) will show high M in the
periodic box and a *fraudulently* high or meaningless M in open space while R_g
diverges and frag >> 1. Cohesive models will hold R_g bounded and frag == 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import init_state, make_boundary, run
from models import (BoidsModel, CouzinModel, CuckerSmaleModel, DOrsognaModel,
                    OlfatiSaberModel, PerceptionQuantum, SlowFastPerception,
                    VicsekModel)

# --------------------------------------------------------------------------
N = 120
L = 10.0
DT = 0.05
STEPS = 1500
R_LINK = 1.5          # interaction range used for fragmentation analysis
SEEDS = [0, 1]
# --------------------------------------------------------------------------


def build(name, boundary, rng):
    """Model factory. Parameters chosen so all models see the same r_link scale."""
    if name == "Vicsek":
        return VicsekModel(boundary, r_max=R_LINK, eta=0.2, v0=0.5, rng=rng)
    if name == "Vicsek(topological k=7)":
        return VicsekModel(boundary, topological=True, k=7, eta=0.2, v0=0.5, rng=rng)
    if name == "Perception[Phi+] (PAPER_1)":
        return PerceptionQuantum(boundary, "Phi+", alpha=np.pi / 2, r_max=R_LINK,
                                 r_min=0.1, eta=0.2, v0=0.5, rng=rng)
    if name == "Perception[GHZ3] (PAPER_1)":
        return PerceptionQuantum(boundary, "GHZ3", alpha=np.pi / 2, r_max=R_LINK,
                                 r_min=0.1, eta=0.2, v0=0.5, rng=rng)
    if name == "SlowFast fixed-graph (PAPER_2)":
        return SlowFastPerception(boundary, k=6, fixed_graph=True, v0=0.5, rng=rng)
    if name == "SlowFast live-graph (PAPER_2)":
        return SlowFastPerception(boundary, metric=True, r_max=R_LINK,
                                  fixed_graph=False, v0=0.5, rng=rng)
    if name == "Boids":
        return BoidsModel(boundary, r_sep=0.4, r_ali=1.0, r_coh=R_LINK,
                          v0=0.5, vmax=0.6, rng=rng)
    if name == "Couzin":
        return CouzinModel(boundary, zor=0.3, zoo=0.8, zoa=R_LINK, v0=0.5, rng=rng)
    if name == "D'Orsogna":
        return DOrsognaModel(boundary, alpha=1.0, beta=2.0, Ca=0.5, la=2.0,
                             Cr=1.0, lr=0.5, cutoff=8.0, rng=rng)
    if name == "Cucker-Smale(b=0.4)":
        return CuckerSmaleModel(boundary, K=2.0, sigma=1.0, beta=0.4, rng=rng)
    if name == "Cucker-Smale(b=0.9)":
        return CuckerSmaleModel(boundary, K=2.0, sigma=1.0, beta=0.9, rng=rng)
    if name == "Olfati-Saber":
        return OlfatiSaberModel(boundary, d=0.7, kappa_ratio=1.2, rng=rng)
    raise ValueError(name)


MODELS = [
    ("Vicsek", False),
    ("Vicsek(topological k=7)", False),
    ("Perception[Phi+] (PAPER_1)", False),
    ("Perception[GHZ3] (PAPER_1)", False),
    ("SlowFast fixed-graph (PAPER_2)", False),
    ("SlowFast live-graph (PAPER_2)", False),
    ("Boids", True),
    ("Couzin", True),
    ("D'Orsogna", True),
    ("Cucker-Smale(b=0.4)", True),
    ("Cucker-Smale(b=0.9)", True),
    ("Olfati-Saber", True),
]


def one(name, bc, seed):
    boundary = make_boundary(bc, L=L)
    rng = np.random.default_rng(1000 + seed)
    st = init_state(N, boundary, speed=0.5, spread=L,
                    rng=np.random.default_rng(seed))
    model = build(name, boundary, rng)
    _, h = run(model, st, steps=STEPS, dt=DT, r_link=R_LINK, record_every=50)
    return {
        "M": h["polar_order"][-1],
        "Rg": h["radius_of_gyration"][-1],
        "Rg0": h["radius_of_gyration"][0],
        "frag": h["n_fragments"][-1],
        "largest": h["largest_cluster_frac"][-1],
        "nbrs": h["mean_neighbors"][-1],
        "hist": h,
    }


def main():
    print("=" * 104)
    print("EXPERIMENT 1: does the model survive when you delete the periodic walls?")
    print(f"N={N}  L={L}  steps={STEPS}  dt={DT}  r_link={R_LINK}  seeds={SEEDS}")
    print("=" * 104)
    hdr = (f"{'model':<32}{'BC':<10}{'M':>7}{'R_g':>9}{'R_g/R_g0':>10}"
           f"{'frag':>6}{'largest':>9}{'<n_i>':>7}  verdict")
    print(hdr)
    print("-" * 104)

    rows = []
    for name, is_coh in MODELS:
        for bc in ("periodic", "open"):
            acc = [one(name, bc, s) for s in SEEDS]
            M = np.mean([a["M"] for a in acc])
            Rg = np.mean([a["Rg"] for a in acc])
            Rg0 = np.mean([a["Rg0"] for a in acc])
            frag = np.mean([a["frag"] for a in acc])
            largest = np.mean([a["largest"] for a in acc])
            nbrs = np.mean([a["nbrs"] for a in acc])
            growth = Rg / max(Rg0, 1e-9)

            if bc == "periodic":
                verdict = "(walls active)"
            elif frag <= 1.2 and growth < 3.0:
                verdict = "SURVIVES"
            elif largest >= 0.5:
                verdict = "PARTIAL"
            else:
                verdict = "*** EVAPORATED ***"

            print(f"{name:<32}{bc:<10}{M:>7.3f}{Rg:>9.2f}{growth:>10.2f}"
                  f"{frag:>6.1f}{largest:>9.2f}{nbrs:>7.2f}  {verdict}")
            rows.append(dict(model=name, bc=bc, M=M, Rg=Rg, growth=growth,
                             frag=frag, largest=largest, nbrs=nbrs,
                             cohesive=is_coh, hist=acc[0]["hist"]))
        print("-" * 104)

    np.save(Path(__file__).parent.parent / "results" / "exp1.npy",
            np.array(rows, dtype=object), allow_pickle=True)
    return rows


if __name__ == "__main__":
    main()
