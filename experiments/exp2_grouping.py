"""
EXPERIMENT 2 -- CAN A MODEL MAINTAIN SEPARATE GROUPS IN OPEN SPACE?
===================================================================

Claim under test
----------------
"Group maintenance needs TWO mechanisms, and Experiment 1's metrics can only see one
 of them."

A multi-group flock fails in two independent directions:

    FISSION : a group loses its own members.
    FUSION  : the groups merge into one blob. Every agent is still perfectly
              cohesive; the GROUPS are gone.

Fusion is invisible to `largest_cluster_frac` and `n_fragments` -- a fused ensemble
scores a perfect 1.0 on both. So a model can be certified an unqualified success by
every metric in Experiment 1 while having destroyed the structure it exists to
represent. `MultiGroupFlock` has one knob per failure mode, so each can be disabled
alone:

    beta <= 1/2 : in-group Cucker-Smale weight in its proved regime  -> anti-FISSION
    w_seg  > 0  : out-group repulsion                                -> anti-FUSION

PART A -- maintenance, in a shared arena (w_global > 0)
    The groups are held in one arena so that maintenance is TESTED against
    interference rather than achieved by everyone flying away from everyone else.
    Arms: full / no-segregation / switching / no-segregation + switching.

PART B -- which mechanism is the anti-fission one? (w_global = 0)
    The arena term is a cohesive force that is not Cucker-Smale, and it props the
    groups up: with it on, beta=0.9 maintains groups just as well as beta=0.4 and
    the threshold is invisible. Part B therefore switches it OFF, so that NOTHING
    but the in-group weight holds a group together, and starts the groups both tight
    and spread -- because "unconditional" is a claim about ALL initial conditions,
    and can only be tested by varying them.

    Measured by per-group R_g growth, not integrity. CS promises bounded relative
    positions, not connectivity at r_link; scoring it by integrity would be marking
    it down for an initial condition it never claimed to fix. See
    `core.metrics.group_expansion`.

Findings this experiment produced that contradicted the prediction written before it
-- both left in, because they are the interesting part:

  * no-segregation does not fuse to S = 0, it PLATEAUS at S ~ 0.50 (measured stable
    from t ~ 40 to t ~ 240, and the same with walls as without). In-group velocity
    consensus keeps group-mates weakly co-located even inside a merged blob, so
    alignment alone buys a little sorting for free -- just never enough to call them
    groups.
  * no-segregation + switching scores HIGHER purity (0.72-0.85) than no-segregation
    with frozen allegiance (~0.50), while most of the swarm changes allegiance
    (78-118 of 150). Defection re-sorts the labels onto whatever clumps exist, so the
    labels track the geometry instead of organising it. S alone calls that run a
    success; S with the defection count calls it what it is.
  * beta > 1/2 does NOT cause fission here when the arena term is on, or from a tight
    start. The theorem is about what is GUARANTEED, and a favourable initial
    condition satisfies conditional flocking's condition. Part B is built to remove
    that help; without it, the threshold is plain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import make_boundary, run
from core.metrics import fragmentation, group_expansion, group_summary
from models import MultiGroupFlock, init_grouped_state

# --------------------------------------------------------------------------
N = 150
K = 3                  # number of groups
L = 12.0
DIM = 3
DT = 0.05
#: t = 150. Long enough for the outcomes to SETTLE rather than merely start: the
#: no-segregation arm reaches its S ~ 0.50 plateau (measured stable from t ~ 40 out
#: to t ~ 240) instead of being scored mid-merge, and the defection counts saturate.
STEPS = 3000
R_LINK = 1.5           # interaction range for fragmentation / integrity analysis
SEEDS = [0, 1]
# --------------------------------------------------------------------------

ARMS_A = [
    ("full (b=0.4, seg on)",     dict(beta=0.4, w_seg=2.5)),
    ("no segregation",           dict(beta=0.4, w_seg=0.0)),
    ("full + switching",         dict(beta=0.4, w_seg=2.5, switching=True)),
    ("no seg + switching",       dict(beta=0.4, w_seg=0.0, switching=True)),
]

#  beta, initial blob radius. arena term OFF for all of these.
ARMS_B = [(0.4, 0.9), (0.9, 0.9), (0.4, 3.5), (0.9, 3.5)]


def _setup(bc, seed, blob=0.9, arena=8.0):
    boundary = make_boundary(bc, L=L, dim=DIM)
    st = init_grouped_state(N, K, boundary, dim=DIM, speed=0.5, blob_radius=blob,
                            arena=arena, rng=np.random.default_rng(seed))
    return boundary, st


def one_a(kwargs, bc, seed):
    b, st = _setup(bc, seed)
    model = MultiGroupFlock(b, rng=np.random.default_rng(1000 + seed), v0=0.5,
                            r_sep=0.5, r_seg=1.6, w_global=0.05, **kwargs)
    final, _ = run(model, st, steps=STEPS, dt=DT, r_link=R_LINK, record_every=200)
    g = final.internal["groups"]
    gs = group_summary(final.positions, final.headings, g, b, R_LINK)
    n_frag, largest = fragmentation(final.positions, b, R_LINK)
    return {
        "S": gs["segregation_index"], "I": gs["group_integrity"],
        "sep": gs["group_separation"], "M_g": gs["mean_group_polar_order"],
        "defect": final.internal.get("defections", 0),
        "largest": largest, "frag": float(n_frag),
        "verdict": gs["group_verdict"],
    }


def one_b(beta, blob, bc, seed):
    # arena=24 so the groups do not overlap at t=0 even when they start spread;
    # w_global=0 so the in-group CS weight is the ONLY thing holding a group.
    b, st = _setup(bc, seed, blob=blob, arena=24.0)
    model = MultiGroupFlock(b, rng=np.random.default_rng(1000 + seed), v0=0.5,
                            beta=beta, w_seg=2.5, r_sep=0.5, r_seg=1.6, w_global=0.0)
    Rg0 = group_expansion(st.positions, st.internal["groups"], b)
    # 2x the Part-A length: expansion is a RATE, and a slow leak needs time on the
    # clock to separate itself from no leak at all.
    final, _ = run(model, st, steps=2 * STEPS, dt=DT, r_link=R_LINK,
                   record_every=400)
    g = final.internal["groups"]
    RgT = group_expansion(final.positions, g, b)
    gs = group_summary(final.positions, final.headings, g, b, R_LINK)
    return {"Rg0": Rg0, "RgT": RgT, "growth": RgT / max(Rg0, 1e-9),
            "S": gs["segregation_index"], "M_g": gs["mean_group_polar_order"]}


def part_a():
    print("=" * 104)
    print("PART A -- can K groups be MAINTAINED in a shared arena, without walls?")
    print(f"N={N}  K={K}  dim={DIM}  L={L}  steps={STEPS}  dt={DT}  "
          f"r_link={R_LINK}  seeds={SEEDS}")
    print("=" * 104)
    print(f"{'arm':<24}{'BC':<10}{'S':>6}{'I':>6}{'sep':>6}{'M_g':>6}"
          f"{'defect':>8}{'largest':>9}{'frag':>6}   verdict")
    print("-" * 104)
    rows = []
    for label, kwargs in ARMS_A:
        for bc in ("periodic", "open"):
            acc = [one_a(kwargs, bc, s) for s in SEEDS]
            m = {k: float(np.mean([a[k] for a in acc]))
                 for k in ("S", "I", "sep", "M_g", "largest", "frag", "defect")}
            vs = [a["verdict"] for a in acc]
            v = max(set(vs), key=vs.count)
            print(f"{label:<24}{bc:<10}{m['S']:>6.2f}{m['I']:>6.2f}{m['sep']:>6.2f}"
                  f"{m['M_g']:>6.2f}{m['defect']:>8.0f}{m['largest']:>9.2f}"
                  f"{m['frag']:>6.1f}   {v}")
            rows.append(dict(part="A", arm=label, bc=bc, **m, verdict=v))
        print("-" * 104)
    print("S = segregation index (1 = groups distinct, 0 = fused).  I = group integrity.")
    print("largest/frag = Experiment 1's cohesion metrics, shown for comparison.")
    print()
    print("READ THE 'no segregation' ROW. largest = 1.00 and frag = 1: a flawless")
    print("score by Experiment 1's standards -- for a run in which the groups have")
    print("merged (S: 1.00 -> ~0.50, the same with walls and without). Cohesion")
    print("metrics cannot see fusion at all. Note also that I stays at 1.00: the")
    print("groups did not lose a single member. They lost only the thing that made")
    print("them groups -- which is invisible to every metric in Experiment 1.")
    print()
    print("READ THE TWO SWITCHING ROWS TOGETHER. Segregation on: 0-1 defections out")
    print("of 150. Segregation off: 78-118 -- most of the swarm changes allegiance.")
    print("Yet S goes UP (0.72-0.85, against ~0.50 frozen), because defection")
    print("re-sorts the labels onto whatever clumps exist. Purity alone calls that")
    print("run BETTER. It is the 'M is a liar' trap, one level up: the number")
    print("improves precisely because the structure it measures has given up and")
    print("started tracking the geometry instead of organising it.")
    return rows


def part_b():
    print()
    print("=" * 104)
    print("PART B -- the Cucker-Smale beta threshold, per group. NO arena term")
    print("          (w_global=0), so ONLY the in-group weight holds a group together.")
    print("          Metric: per-group R_g growth -- what the theorem actually promises.")
    print("=" * 104)
    print(f"{'beta':<8}{'start':<12}{'BC':<10}{'R_g(0)':>9}{'R_g(T)':>9}"
          f"{'growth':>9}{'S':>6}{'M_g':>6}   regime")
    print("-" * 104)
    rows = []
    for beta, blob in ARMS_B:
        for bc in ("open",):
            acc = [one_b(beta, blob, bc, s) for s in SEEDS]
            m = {k: float(np.mean([a[k] for a in acc]))
                 for k in ("Rg0", "RgT", "growth", "S", "M_g")}
            start = "tight" if blob < 2.0 else "spread"
            regime = ("unconditional (proved)" if beta <= 0.5
                      else "conditional only")
            print(f"{beta:<8.1f}{start:<12}{bc:<10}{m['Rg0']:>9.2f}{m['RgT']:>9.2f}"
                  f"{m['growth']:>9.2f}{m['S']:>6.2f}{m['M_g']:>6.2f}   {regime}")
            rows.append(dict(part="B", beta=beta, start=start, bc=bc, **m))
    print("-" * 104)
    print("beta=0.4 does not expand from EITHER initial condition. beta=0.9 expands")
    print("from both, and worse from the spread one -- its flocking is CONDITIONAL,")
    print("so the initial condition is allowed to matter. That difference IS the")
    print("beta = 1/2 threshold of the theorem, measured, per group.")
    print()
    print("Note M_g ~ 1.0 everywhere: every group is perfectly aligned in every arm.")
    print("Alignment was never the thing at stake, and reporting it would hide this.")
    return rows


def main():
    rows = part_a() + part_b()
    out = Path(__file__).parent.parent / "results" / "exp2.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(rows, dtype=object), allow_pickle=True)
    return rows


if __name__ == "__main__":
    main()
