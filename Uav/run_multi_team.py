"""
MULTI-TEAM UAV EXPERIMENT -- what the flocking theory is worth to a real swarm.
==============================================================================

    python Uav/run_multi_team.py            # both tables
    python Uav/run_multi_team.py --viz      # + 3D movies

Scenario. K teams of drones, 3D open airspace, crossing waypoints so the teams must
fly through each other and deconflict. At t = `LEADER_LOSS_AT` every leader stops
transmitting, the formation layer loses its anchor, and the alignment layer is all
that is left holding each team together.

The rule under test is one line of `swarm_sim.py`:

    cs_weight = 1.0 / (1.0 + d**2)
    align = (cs_weight[:, None] * v_n).sum(axis=0) / cs_weight.sum()

Two separate things are true about it, and they pull in opposite directions:

  * `1/(1+d^2)` is psi(r) = K/(sigma^2+r^2)^beta at BETA = 1. Cucker-Smale's
    unconditional-flocking theorem needs beta <= 1/2. At beta = 1 flocking is
    conditional -- guaranteed only from a good enough initial condition.
  * dividing by `cs_weight.sum()` makes it a weighted AVERAGE, so the distance decay
    only sets RELATIVE weights between neighbours and never weakens the coupling in
    absolute terms. That is a DeGroot/Vicsek consensus step, and the CS theorem does
    not describe it at any beta.

TEST 1 -- leader loss under turbulence, 400 s. Does the team disperse?
TEST 2 -- the gust/straggler event. One drone is blown D metres out of formation
          with a velocity error. Does its velocity re-converge (gap BOUNDED: the
          drone is trackable) or not (gap grows without bound: lost airframe)?

Both tests agree, and neither is redundant: Test 1 asks whether the team survives
being left alone, Test 2 asks whether it can afford to lose one drone's station.

A NOTE ON RUN LENGTH, because it changed the answer. At t = 120 s every arm in
Test 1 scored an identical R_g growth of 1.01x and the table read "all safe". At
t = 400 s -- a realistic sortie -- the beta=1 CS sum reads 31x and integrity 0.17.
Dispersal is a RATE. The leak was always there; 120 s was simply too short to see
it, and a short run would have signed off on a controller that loses the swarm.

THE RESULT, AND IT IS NOT WHAT THIS FILE ORIGINALLY PREDICTED
-------------------------------------------------------------
The normalisation is what saves `swarm_sim.py`. A lone straggler's only neighbours
are its own team, so a weighted average gives them full weight however far away they
are, and its velocity re-converges: gap bounded at 22 m / 63 m / 123 m. The current
code is operationally sound -- for a reason its own comment does not name, and
without any theorem behind it.

The trap is the obvious "fix". Remove the normalisation to make it a genuine CS sum
and LEAVE beta = 1, and the straggler is gone: the gap grows to 812 m / 1430 m /
1600 m and keeps going. Half-adopting the theory is far worse than ignoring it.
Unnormalise AND set beta <= 1/2 and the gap is bounded again (22 m / 63 m / 125 m),
now with Cucker-Smale's guarantee under it rather than luck.

AND THE OPERATIONAL CAVEAT THAT MATTERS MOST
--------------------------------------------
No alignment rule RECOVERS the straggler. 22.4 m at t=100 s is still 22.5 m at
t=600 s. Cucker-Smale gives velocity consensus and bounded relative positions -- it
never promised to close a gap, and it does not. The drone cruises alongside forever
at a fixed offset. Rejoining requires the formation layer (a live leader) or a
genuine attraction term. If your CONOPS assumes a gust-displaced drone will slide
back into its slot on its own, no choice of beta will deliver that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import make_boundary, run
from core.metrics import (group_expansion, group_integrity,
                          per_group_polar_order, segregation_index)
from Uav.multi_team_3d import MultiTeamUAV, init_teams

N_PER_TEAM = 8
K_TEAMS = 3
DT = 0.05
#: t = 400 s, ~6.5 min: a realistic multirotor sortie, and long enough for a slow
#: leak to separate itself from no leak. Dispersal is a RATE.
STEPS = 8000
LEADER_LOSS_AT = 30.0   # s
R_LINK = 8.0            # radio / sensing range, used for the connectivity check
MIN_SEP = 1.5           # hard safety floor, m
TURBULENCE = 0.35       # per-drone UNCORRELATED gust. A uniform wind tests nothing.
SEEDS = [0, 1]

#: Test 2 runs to 600 s: the beta=1 CS-sum gap grows LINEARLY and for ever, so the
#: only honest way to report it is to show it still growing at the end.
STRAGGLER_T = 600.0
STRAGGLER_D = [20.0, 60.0, 120.0]

ARMS = [
    ("swarm_sim.py rule (b=1, norm)", dict(beta=1.0, normalize_align=True)),
    ("b=1.0, CS sum (unnorm)",        dict(beta=1.0, normalize_align=False)),
    ("b=0.5, CS sum (threshold)",     dict(beta=0.5, normalize_align=False)),
    ("b=0.4, CS sum (inside)",        dict(beta=0.4, normalize_align=False)),
]


def _waypoints(k, reach=2000.0):
    """Crossing waypoints: each team aims at the far side, so the teams meet in the
    middle. A parallel cruise would never test deconfliction.

    `reach` is deliberately far beyond the run: at 3 m/s over 400 s a team covers
    ~1200 m, so the leader NEVER arrives and the mission stays a continuous cruise.
    An earlier version put the waypoint at 60 m; the leader reached it at t ~ 20 s,
    `to_wp` went to zero, the swarm parked itself before the leader-loss event at
    t = 30 s ever fired, and all four arms scored identically for a reason that had
    nothing to do with any of them.
    """
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([-reach * np.cos(ang), -reach * np.sin(ang),
                     np.zeros(k)], axis=1)


def _min_separation(x):
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _model(b, teams, kwargs, seed, turbulence=TURBULENCE):
    return MultiTeamUAV(b, teams, _waypoints(K_TEAMS), min_separation=MIN_SEP,
                        turbulence=turbulence, leader_loss_at=LEADER_LOSS_AT,
                        rng=np.random.default_rng(100 + seed), **kwargs)


# --------------------------------------------------------------------------
# TEST 1 -- leader loss under turbulence
# --------------------------------------------------------------------------

def one_dispersal(kwargs, seed, record=False):
    b = make_boundary("open", dim=3)         # airspace is open. Not a choice.
    st, teams = init_teams(N_PER_TEAM, K_TEAMS, rng=np.random.default_rng(seed))
    m = _model(b, teams, kwargs, seed)

    # R_g is measured from leader loss onward: before that the formation term is
    # holding the team, and including it would average the failure away.
    pre, _ = run(m, st, steps=int(LEADER_LOSS_AT / DT), dt=DT, r_link=R_LINK,
                 record_every=100)
    Rg0 = group_expansion(pre.positions, teams, b)
    final, h = run(m, pre, steps=STEPS - int(LEADER_LOSS_AT / DT), dt=DT,
                   r_link=R_LINK, record_every=10, record_traj=True)

    out = {
        "growth": group_expansion(final.positions, teams, b) / max(Rg0, 1e-9),
        # PER TEAM. The ensemble's largest_cluster_frac is 0.33 here by construction
        # -- three deconflicted teams -- and would be pure noise in this table.
        "integrity": group_integrity(final.positions, teams, b, R_LINK),
        "M_team": float(np.mean(per_group_polar_order(final.headings, teams))),
        "S": segregation_index(final.positions, teams, b),
        # A safety floor is breached at an INSTANT: check the whole trajectory,
        # never just the final frame.
        "min_sep": min(_min_separation(x) for x in h["trajectory"]),
    }
    return (out, h, m, b, teams) if record else (out, None, None, None, None)


def table_dispersal():
    print("=" * 104)
    print("TEST 1 -- LEADER LOSS under turbulence. Does each team hold together?")
    print(f"{K_TEAMS} teams x {N_PER_TEAM} drones, 3D open airspace, dt={DT}, "
          f"t={STEPS * DT:.0f}s, leader lost at t={LEADER_LOSS_AT:.0f}s,")
    print(f"turbulence={TURBULENCE} (per-drone, uncorrelated), seeds={SEEDS}")
    print("=" * 104)
    print(f"{'alignment rule':<32}{'theorem?':<10}{'R_g growth':>11}{'integrity':>11}"
          f"{'M_team':>8}{'S':>6}{'min_sep':>9}   verdict")
    print("-" * 104)
    for label, kw in ARMS:
        acc = [one_dispersal(kw, s)[0] for s in SEEDS]
        m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        m["min_sep"] = min(a["min_sep"] for a in acc)      # worst case, not the mean
        proved = kw["beta"] <= 0.5 and not kw["normalize_align"]
        if m["min_sep"] < MIN_SEP:
            v = "*** SAFETY BREACH ***"
        elif m["growth"] < 1.5 and m["integrity"] > 0.9:
            v = "HOLDS"
        elif m["integrity"] >= 0.5:
            v = "PARTIAL"
        else:
            v = "*** SCATTERED ***"
        print(f"{label:<32}{('yes' if proved else 'no'):<10}{m['growth']:>10.2f}x"
              f"{m['integrity']:>11.2f}{m['M_team']:>8.2f}{m['S']:>6.2f}"
              f"{m['min_sep']:>9.2f}   {v}")
    print("-" * 104)
    print(f"min_sep = WORST separation over the whole trajectory and both seeds, vs")
    print(f"a {MIN_SEP} m floor. A constraint, not a score: breach it and the arm has")
    print("failed however good the rest of its row looks.")
    print()
    print("The beta=1 CS SUM scatters: R_g growth ~31x, integrity 0.17 -- the team is")
    print("gone. The other three hold at ~1.01x. Note WHICH two survive and why they")
    print("are not the same reason: beta<=0.5 survives by theorem; the current")
    print("swarm_sim.py rule survives because normalising keeps the coupling strong")
    print("at any range. Same column, different guarantees.")
    print()
    print("RUN LENGTH IS LOAD-BEARING HERE. At t=120s every arm scored 1.01x and this")
    print("table said 'all safe'. Dispersal is a RATE: the beta=1 leak needed ~400s of")
    print("flight to show itself. A short sortie in sim is not evidence of a safe one.")


# --------------------------------------------------------------------------
# TEST 2 -- the gust / straggler event
# --------------------------------------------------------------------------

def one_straggler(kwargs, seed, D):
    """Blow drone 1 of team 0 out of formation by D m, with a velocity error.

    Returns the gap between the straggler and its team's centroid, early and late.
    Growing gap = the drone is leaving and will not stop. Bounded gap = it holds
    station at an offset: still out of formation, but trackable and recoverable.
    """
    b = make_boundary("open", dim=3)
    st, teams = init_teams(N_PER_TEAM, K_TEAMS, rng=np.random.default_rng(seed))
    # Turbulence off HERE: this test is about one deterministic gust event, and
    # noise on top would only blur the comparison it exists to make.
    m = _model(b, teams, kwargs, seed, turbulence=0.0)
    s, _ = run(m, st, steps=int((LEADER_LOSS_AT + 1.0) / DT), dt=DT, r_link=R_LINK,
               record_every=1000)

    s.positions[1] = s.positions[1] + np.array([0.0, 1.0, 0.0]) * D
    s.velocities[1] = np.array([0.0, 2.5, 0.0])

    team0 = np.flatnonzero(teams == 0)
    rest = [i for i in team0 if i != 1]
    gaps = []
    for t_end in (100.0, STRAGGLER_T):
        s, _ = run(m, s, steps=int((t_end - s.t) / DT), dt=DT, r_link=R_LINK,
                   record_every=4000)
        gaps.append(float(np.linalg.norm(s.positions[1] -
                                         s.positions[rest].mean(axis=0))))
    return gaps


def table_straggler():
    print()
    print("=" * 104)
    print("TEST 2 -- GUST / STRAGGLER. One drone blown D m out of formation, after")
    print("          leader loss. Does its velocity re-converge, or is it gone?")
    print(f"          (no turbulence: one deterministic gust, t -> {STRAGGLER_T:.0f}s)")
    print("=" * 104)
    print(f"{'alignment rule':<32}{'theorem?':<10}{'D (m)':>7}{'gap t=100s':>12}"
          f"{'gap t=600s':>12}   outcome")
    print("-" * 104)
    for label, kw in ARMS:
        proved = kw["beta"] <= 0.5 and not kw["normalize_align"]
        for D in STRAGGLER_D:
            g = np.mean([one_straggler(kw, s, D) for s in SEEDS], axis=0)
            grew = g[1] > 1.5 * g[0]
            out = "*** LOST (gap still growing)" if grew else "bounded (holds station)"
            print(f"{label:<32}{('yes' if proved else 'no'):<10}{D:>7.0f}"
                  f"{g[0]:>12.1f}{g[1]:>12.1f}   {out}")
        print("-" * 104)
    print("The normalisation is what saves the CURRENT code: a lone straggler's only")
    print("neighbours are its own team, and a weighted average gives them full weight")
    print("at any distance, so its velocity re-converges. Sound -- but by luck, not")
    print("by theorem.")
    print()
    print("THE TRAP: unnormalising to get a 'real' Cucker-Smale sum while LEAVING")
    print("beta=1 loses the drone outright. Half-adopting the theory is much worse")
    print("than ignoring it. Unnormalise AND set beta <= 1/2 -- then the gap is")
    print("bounded and the theorem, not luck, is why.")
    print()
    print("CAVEAT, AND IT IS THE ONE THAT MATTERS OPERATIONALLY: no alignment rule")
    print("RECOVERS the straggler. 22 m at t=100 is 22 m at t=600. Cucker-Smale gives")
    print("velocity consensus and a bounded gap; it never promised to close one. If")
    print("your CONOPS assumes a displaced drone rejoins its slot unaided, no beta")
    print("delivers that -- you need the formation layer (a live leader) or a real")
    print("attraction term.")


def viz():
    """Movies of TEAM 0 through the leader-loss event.

    Camera locked to team 0 at a fixed 30 m half-width. The auto-fit is wrong here
    and the reason is worth stating: the teams cruise ~1.2 km while each stays ~6 m
    across, so an auto-fitted view spans the JOURNEY and renders each team as a
    single pixel (it did exactly that on the first attempt). A hand-set zoom can
    hide expansion, though -- so R_g stays in the HUD, and it is the check on the
    picture: watch it against the 30 m frame, not the frame alone.
    """
    from core.viz3d import animate3d
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    # The b=1 CS sum (ARMS[1]) is the one that scatters -- render it against the
    # rule the code actually uses today (ARMS[0]) and the proved one (ARMS[3]).
    for label, kw in (ARMS[0], ARMS[1], ARMS[3]):
        _, h, m, b, teams = one_dispersal(kw, 0, record=True)
        tag = ("b1_normalised" if kw["normalize_align"]
               else f"b{kw['beta']:.1f}_cs_sum".replace(".", ""))
        team0 = np.flatnonzero(teams == 0)

        def hud(k, h=h, b=b, teams=teams):
            x = h["trajectory"][k]
            return (f"R_g/team = {group_expansion(x, teams, b):6.2f} m\n"
                    f"min sep  = {_min_separation(x):6.2f} m  (floor {MIN_SEP})\n"
                    f"S        = {segregation_index(x, teams, b):6.2f}")

        animate3d(h, b, out=str(out_dir / f"uav_{tag}.gif"),
                  title=f"UAV team after leader loss -- {label}",
                  hud_fn=hud, trail=15, arrow=2.0, spin=0.05,
                  focus=team0, half_width=30.0, stride=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true", help="also write 3D movies")
    a = ap.parse_args()
    table_dispersal()
    table_straggler()
    if a.viz:
        viz()
