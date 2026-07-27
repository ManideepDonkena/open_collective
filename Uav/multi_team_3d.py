"""
MULTI-TEAM UAV SWARM IN 3D -- the grouping model applied to the real scenario.
=============================================================================

What this adds to `swarm_sim.py`
--------------------------------
`swarm_sim.py` flies ONE 12-drone V-formation in 2D behind ONE leader. This flies
K TEAMS in 3D, each with its own leader, its own V-slot geometry and its own
waypoint, sharing one airspace and required to stay out of each other's way. That
is the multi-group maintenance problem from `models/grouping.py`, with the two extra
constraints that make it a real UAV problem rather than a bird problem:

  * a HARD safety floor -- birds tolerate near-misses, airframes do not;
  * every control output is a velocity setpoint an autopilot will actually accept.

Three findings from the flocking side that apply directly to `swarm_sim.py`
--------------------------------------------------------------------------
1. THE ALIGNMENT WEIGHT IS OUTSIDE THE THEOREM IT CITES.
   `swarm_sim.py` computes `cs_weight = 1.0 / (1.0 + d**2)` and calls it
   "Cucker-Smale-style". Cucker-Smale is psi(r) = K/(sigma^2 + r^2)^beta, so that
   line is exactly beta = 1. The theorem's unconditional-flocking regime is
   beta <= 1/2. At beta = 1 flocking is CONDITIONAL: guaranteed only if the swarm
   starts tight enough relative to its velocity spread. Changing the exponent to 0.5
   costs one `np.sqrt` and buys a theorem that holds for ANY initial condition.

2. IT IS NOT ACTUALLY THE CUCKER-SMALE SYSTEM, BECAUSE IT NORMALISES.
   `align = (cs_weight * v_n).sum() / cs_weight.sum()` is a weighted AVERAGE -- a
   DeGroot/Vicsek consensus step. The distance decay only sets relative weights
   between neighbours; it never weakens the coupling in absolute terms. The CS
   theorem is a statement about the UNNORMALISED sum, so it does not apply to that
   line in any regime. Set `normalize_align=False` here for the real thing.

3. THE SWARM'S COHESION IS CURRENTLY THE LEADER'S JOB, AND THAT IS THE RISK.
   In `swarm_sim.py` what actually holds the group together is the formation term
   pulling every follower toward `leader_pos + offset`. That is a single point of
   failure: it is one drone, and Experiment 1 in this repo measured what a
   topological-k alignment swarm does in open space once nothing pulls it together
   (Vicsek k=7: 6 fragments, largest cluster 0.65). `leader_loss_at` here makes that
   testable rather than hypothetical -- and an unnormalised beta<=1/2 alignment layer
   is what lets a team survive it, because psi(r) > 0 at every range means a drone
   that falls behind is still heard.

The boundary condition is OPEN, and that is not a detail
--------------------------------------------------------
Drones fly in unbounded airspace. There is no periodic box, so every result in this
repo's Experiment 1 about models that only work on a torus applies here directly and
with no translation: a UAV controller whose cohesion depends on walls has no walls.
This is the one application where the open boundary is not the hard test -- it is
simply the truth.

Run: python Uav/run_multi_team.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.base import CollectiveModel, State


def v_formation_3d(n_team: int, spacing: float = 3.0, dihedral: float = 0.35):
    """Offsets of each team member from its leader, in the leader's frame.

    Slot 0 is the leader at the tip. Followers alternate left/right and step BACK
    and slightly UP with rank -- the vertical stagger (`dihedral`) is the part that
    only exists in 3D, and it is what buys altitude separation between wingtips
    instead of relying on lateral spacing alone.
    """
    off = np.zeros((n_team, 3))
    for i in range(1, n_team):
        side = 1 if i % 2 == 1 else -1
        rank = (i + 1) // 2
        off[i] = (-rank * spacing,
                  side * rank * spacing * 0.8,
                  rank * spacing * dihedral)
    return off


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


class MultiTeamUAV(CollectiveModel):
    """K UAV teams, 3D, open airspace, with inter-team deconfliction.

    Control stack per drone (all layers output into one velocity setpoint):

        L1  hard separation      -- everyone, inside `min_separation`. Safety.
        L2a formation keeping    -- toward leader_pos + slot offset (own team only)
        L2b alignment            -- psi(r)-weighted velocity matching (own team only)
        L3  mission              -- leader cruises at constant speed to its waypoint
        L4  deconfliction        -- repulsion from OTHER teams inside `deconflict_radius`

    Parameters that carry the argument
    ----------------------------------
    beta           : the psi exponent. <= 0.5 is Cucker-Smale's unconditional regime.
                     `swarm_sim.py` uses the equivalent of beta = 1.0.
    normalize_align: True reproduces `swarm_sim.py` (a weighted average, no theorem).
                     False gives the genuine CS sum, to which the theorem applies.
    leader_loss_at : simulated time at which every team's leader stops transmitting.
                     The formation layer dies with it and the alignment layer is all
                     that is left. This is the test that separates a swarm that is
                     cohesive from a swarm whose leader was doing the cohering.
    teams          : (N,) int team id. `deconflict` is the anti-fusion mechanism from
                     models/grouping.py, in metres.

    Notes
    -----
    * `finalize` is not called: open airspace has nothing to wrap or reflect against.
      Positions integrate freely, which is what a real flight does.
    * Accel and speed are saturated to `max_accel` / `max_speed`, so the setpoints
      stay inside an airframe's envelope rather than being physically impossible.
    """

    name = "Multi-team UAV (3D, open airspace)"
    cohesive = True

    def __init__(self, boundary, teams, waypoints, K=2.0, sigma=1.0, beta=0.5,
                 normalize_align=False, spacing=3.0, min_separation=1.5,
                 deconflict_radius=6.0, w_align=1.6, w_form=1.2, w_sep=2.0,
                 w_deconf=2.5, max_speed=3.0, max_accel=4.0, noise_std=0.02,
                 cruise_speed=2.5, w_cruise=1.0, turbulence=0.0,
                 leader_loss_at=None, rng=None):
        super().__init__(boundary, rng)
        self.teams = np.asarray(teams, dtype=int)
        self.waypoints = np.asarray(waypoints, dtype=float)   # (K, 3)
        self.K, self.sigma, self.beta = K, sigma, beta
        self.normalize_align = normalize_align
        self.min_separation = min_separation
        self.deconflict_radius = deconflict_radius
        self.w_align, self.w_form = w_align, w_form
        self.w_sep, self.w_deconf = w_sep, w_deconf
        self.max_speed, self.max_accel = max_speed, max_accel
        self.noise_std = noise_std
        self.cruise_speed, self.w_cruise = cruise_speed, w_cruise
        self.turbulence = turbulence
        self.leader_loss_at = leader_loss_at

        self.team_ids = np.unique(self.teams)
        # Slot 0 of each team is its leader.
        self.leaders = np.array([np.flatnonzero(self.teams == t)[0]
                                 for t in self.team_ids])
        self.offsets = np.zeros((len(self.teams), 3))
        for t in self.team_ids:
            idx = np.flatnonzero(self.teams == t)
            self.offsets[idx] = v_formation_3d(len(idx), spacing)

    @property
    def unconditional(self) -> bool:
        """True iff this configuration is inside Cucker-Smale's proved regime.

        Requires BOTH beta <= 1/2 and an unnormalised sum -- normalising turns the
        law into a DeGroot average, which the theorem says nothing about.
        """
        return self.beta <= 0.5 and not self.normalize_align

    def psi(self, r):
        return self.K / (self.sigma ** 2 + r ** 2) ** self.beta

    def leader_alive(self, t) -> bool:
        return self.leader_loss_at is None or t < self.leader_loss_at

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        n = s.n
        x, v = s.positions, s.velocities

        d = b.displacement(x[:, None, :], x[None, :, :])      # i -> j
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)
        same = self.teams[:, None] == self.teams[None, :]
        alive = self.leader_alive(s.t)

        vdes = np.zeros((n, 3))

        # --- L2b alignment, own team only -----------------------------------
        w = np.where(same, self.psi(r), 0.0)
        np.fill_diagonal(w, 0.0)
        if self.normalize_align:
            # swarm_sim.py's rule: a weighted AVERAGE of neighbour velocities.
            denom = w.sum(axis=1, keepdims=True)
            avg = (w[:, :, None] * v[None, :, :]).sum(axis=1) / np.maximum(denom, 1e-9)
            vdes += self.w_align * np.where(denom > 1e-9, avg - v, 0.0)
        else:
            # The genuine Cucker-Smale sum, normalised only by team size.
            n_t = np.array([np.sum(self.teams == self.teams[i]) for i in range(n)])
            dv = v[None, :, :] - v[:, None, :]
            vdes += self.w_align * (w[:, :, None] * dv).sum(axis=1) / n_t[:, None]

        # --- L2a formation keeping, and L3 mission --------------------------
        is_alive_leader = np.zeros(n, dtype=bool)
        for t, lead in zip(self.team_ids, self.leaders):
            idx = np.flatnonzero(self.teams == t)
            followers = idx[1:]
            if alive:
                is_alive_leader[lead] = True
                target = x[lead][None, :] + self.offsets[followers]
                vdes[followers] += self.w_form * b.displacement(x[followers], target)
                # Leader pursues the waypoint at CONSTANT cruise speed: a decaying
                # position error would let its heading go noise-dominated on arrival
                # and drag the team's order parameter down with it.
                # ADDED, not assigned -- overwriting would make the leader the one
                # drone in the sky ignoring its own collision avoidance.
                to_wp = b.displacement(x[lead], self.waypoints[t])
                vdes[lead] += self.max_speed * _unit(to_wp[None, :])[0]
            # else: leader lost. The formation layer has no anchor and contributes
            # nothing. Alignment, cruise and deconfliction are all that remain --
            # THIS is the regime the beta exponent is finally allowed to matter in.

        # --- airspeed maintenance -------------------------------------------
        # Every drone that is not an actively-navigating leader holds cruise speed
        # along its own heading. Without this the followers have NO velocity source
        # once the leader is gone: vdes collapses to a small alignment correction,
        # acc = (vdes - v)/dt brakes them, and the swarm coasts to a halt (measured:
        # 3.0 -> 0.13 m/s). A halted swarm cannot disperse, so every alignment rule
        # scores a perfect R_g growth of 1.00 and the experiment says nothing.
        # It is also just wrong: a cruising aircraft holds airspeed.
        hold = ~is_alive_leader
        if hold.any():
            vdes[hold] += self.w_cruise * self.cruise_speed * _unit(v[hold])

        # --- L1 hard separation, everyone -----------------------------------
        m = r < self.min_separation * 1.8
        if m.any():
            mag = np.clip(self.min_separation * 1.8 - r, 0.0, None)
            rep = -(d / np.maximum(r, 1e-9)[:, :, None]) * mag[:, :, None]
            vdes += self.w_sep * np.where(m[:, :, None], rep, 0.0).sum(axis=1)

        # --- L4 inter-team deconfliction (the anti-fusion term) -------------
        m = (r < self.deconflict_radius) & ~same
        if m.any():
            rr = np.where(m, r, np.inf)
            rep = -d / (rr ** 2 + 1e-9)[:, :, None]
            vdes += self.w_deconf * np.where(m[:, :, None], rep, 0.0).sum(axis=1)

        vdes += self.rng.normal(0.0, self.noise_std, size=(n, 3))

        # --- saturate to the airframe envelope ------------------------------
        acc = (vdes - v) / dt
        an = np.linalg.norm(acc, axis=1, keepdims=True)
        acc *= np.minimum(1.0, self.max_accel / np.maximum(an, 1e-9))
        v = v + acc * dt

        # --- turbulence, applied AFTER saturation ---------------------------
        # Per-drone INDEPENDENT gusts, which is the point: a uniform wind field
        # displaces a whole team equally and never tests cohesion at all. Only the
        # uncorrelated component pushes drones apart, and it is what the alignment
        # layer has to fight. `swarm_sim.py`'s own README lists adding wind as a
        # prerequisite to trusting its numbers; this is that term.
        # It is a disturbance, not a command, so no autopilot saturates it away.
        if self.turbulence > 0.0:
            v = v + self.rng.normal(0.0, self.turbulence, size=(n, 3)) * np.sqrt(dt)

        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v *= np.minimum(1.0, self.max_speed / np.maximum(sp, 1e-9))

        s.velocities = v
        s.positions = x + v * dt
        s.internal["groups"] = self.teams        # lets viz3d colour by team
        s.t += dt
        return s      # open airspace: nothing to wrap, nothing to reflect off


def init_teams(n_per_team, k_teams, spacing=3.0, separation=25.0, speed=1.0,
               rng=None):
    """K teams, each a loose cluster around its own start point, all at cruise.

    Returns (State, teams). Teams start well apart (`separation` m) and are given
    crossing waypoints by the runner, so deconfliction is actually exercised: the
    interesting moment is the crossing, not the cruise.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n = n_per_team * k_teams
    teams = np.repeat(np.arange(k_teams), n_per_team)
    x = np.zeros((n, 3))
    v = np.zeros((n, 3))
    angles = np.linspace(0, 2 * np.pi, k_teams, endpoint=False)
    for t in range(k_teams):
        idx = np.flatnonzero(teams == t)
        hub = np.array([0.5 * separation * np.cos(angles[t]),
                        0.5 * separation * np.sin(angles[t]),
                        0.0])
        x[idx] = hub + rng.normal(scale=spacing, size=(len(idx), 3))
        heading = -_unit(hub[None, :])[0]          # aimed across the airspace
        v[idx] = speed * heading
    st = State(positions=x, velocities=v)
    st.internal["groups"] = teams
    return st, teams
