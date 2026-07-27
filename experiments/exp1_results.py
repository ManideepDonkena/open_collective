"""Compiled results of Experiment 1, plus the summary figure.

Numbers below are the measured output of exp1_boundary_collapse.py at
N=120, L=10, steps=1500, dt=0.05, r_link=1.5, averaged over 2 seeds.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# model, cohesive?, (M, Rg, growth, frag, largest) periodic, then open
RESULTS = [
    # name,                       coh,   M_p,   Rg_p, g_p,  fr_p, lg_p,   M_o,   Rg_o,  g_o,  fr_o, lg_o
    ("Vicsek",                     0,  0.989,  3.72, 0.94,  1.0, 1.00,  0.843, 18.18, 4.41,  3.0, 0.76),
    ("Vicsek (topological k=7)",   0,  0.990,  3.74, 0.94,  1.0, 1.00,  0.768, 12.36, 3.00,  6.0, 0.65),
    ("Perception[Phi+] PAPER_1",   0,  0.987,  2.71, 0.69,  2.0, 0.93,  0.434, 33.37, 8.10, 39.0, 0.17),
    ("Perception[GHZ3] PAPER_1",   0,  0.987,  1.59, 0.40,  3.5, 0.93,  0.397, 33.91, 8.23, 42.5, 0.23),
    ("SlowFast PAPER_2",           0,  1.000,  3.55, 0.90,  1.5, 1.00,  0.695, 16.01, 3.89, 18.5, 0.60),
    ("Boids",                      1,  0.602,  1.73, 0.44,  1.0, 1.00,  0.508,  8.43, 2.05,  3.0, 0.77),
    ("Couzin",                     1,  0.723,  2.20, 0.55,  1.5, 0.88,  0.293, 17.43, 4.23,  5.0, 0.47),
    ("Olfati-Saber",               1,  0.999,  3.76, 0.95,  1.0, 1.00,  0.393, 14.67, 3.56, 17.0, 0.31),
    ("Cucker-Smale beta=0.9",      1,  1.000,  3.88, 0.98,  1.0, 1.00,  0.927,  7.74, 1.88, 40.5, 0.38),
    ("Cucker-Smale beta=0.4",      1,  1.000,  3.95, 1.00,  1.0, 1.00,  1.000,  4.24, 1.03,  3.0, 0.97),
    ("D'Orsogna (Morse)",          1,  0.534,  1.28, 0.32,  1.0, 1.00,  0.119,  1.29, 0.31,  1.0, 1.00),
]


def verdict(growth, largest):
    if growth < 1.5 and largest > 0.9:
        return "SURVIVES"
    if largest >= 0.5:
        return "PARTIAL"
    return "EVAPORATED"


def table():
    print("=" * 100)
    print("EXPERIMENT 1 -- PERIODIC vs OPEN.  N=120, L=10, 1500 steps, 2 seeds.")
    print("=" * 100)
    print(f"{'model':<28}{'M_per':>7}{'M_open':>8}{'Rg growth':>11}{'frag':>7}"
          f"{'largest':>9}   verdict")
    print("-" * 100)
    for r in RESULTS:
        name, coh = r[0], r[1]
        M_p, M_o = r[2], r[7]
        g_o, fr_o, lg_o = r[9], r[10], r[11]
        v = verdict(g_o, lg_o)
        tag = "  " if coh else " *"
        print(f"{name+tag:<28}{M_p:>7.3f}{M_o:>8.3f}{g_o:>11.2f}x{fr_o:>6.1f}"
              f"{lg_o:>9.2f}   {v}")
    print("-" * 100)
    print("* = alignment-only (no attraction term).  Rg growth = R_g(final)/R_g(0) in OPEN space.")
    print("A growth of 1.0 means the group never expanded. 8.2x means it evaporated.")


def figure(out="results/fig1_boundary_collapse.png"):
    names = [r[0] for r in RESULTS]
    coh = np.array([r[1] for r in RESULTS])
    M_p = np.array([r[2] for r in RESULTS])
    M_o = np.array([r[7] for r in RESULTS])
    growth = np.array([r[9] for r in RESULTS])
    largest = np.array([r[11] for r in RESULTS])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    y = np.arange(len(names))
    colors = ["#c0392b" if c == 0 else "#1a7a4c" for c in coh]

    # --- panel 1: the order parameter lies ---------------------------------
    ax = axes[0]
    ax.barh(y - 0.2, M_p, height=0.38, color="#9aa4b0", label="periodic (torus)")
    ax.barh(y + 0.2, M_o, height=0.38, color=colors, label="open (free space)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("polar order  M")
    ax.set_title("(a) What the literature reports\nM looks fine in both. It is not enough.",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)

    # --- panel 2: the cohesion truth ---------------------------------------
    ax = axes[1]
    ax.barh(y, growth, color=colors)
    ax.axvline(1.0, color="k", ls="--", lw=1.2, label="no expansion")
    ax.axvline(3.0, color="#c0392b", ls=":", lw=1.4, label="dispersal threshold")
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel(r"$R_g(T)\,/\,R_g(0)$  in open space")
    ax.set_title("(b) What periodic BC hides\nGroup expansion. >3x = evaporated.",
                 fontsize=10)
    for i, g in enumerate(growth):
        ax.text(g + 0.12, i, f"{g:.2f}x", va="center", fontsize=8)
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 9.6)
    ax.grid(axis="x", alpha=0.25)

    # --- panel 3: is it still one flock? -----------------------------------
    ax = axes[2]
    ax.barh(y, largest, color=colors)
    ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.axvline(0.5, color="#c0392b", ls=":", lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("largest cluster fraction (open)")
    ax.set_title("(c) Is it even one flock?\n1.0 = intact. 0.17 = shattered.",
                 fontsize=10)
    for i, l in enumerate(largest):
        ax.text(l + 0.015, i, f"{l:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.12); ax.grid(axis="x", alpha=0.25)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#c0392b", label="alignment-only (no attraction)"),
                        Patch(color="#1a7a4c", label="cohesive mechanism")],
               loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Remove the periodic walls and most flocking models evaporate",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=145)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    table()
    figure()
