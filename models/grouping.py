"""
MULTI-GROUP FLOCKING.  cohesive = True.

The question every other model in this repo dodges
--------------------------------------------------
Every model in `alignment.py` and `cohesive.py` asks a ONE-group question: does the
flock hold together, yes or no? Real birds do something harder. Several groups
occupy the same airspace, each stays internally coherent, and each stays
DISTINGUISHABLE from the others -- no merging, no dissolving, no wholesale
defection. That is a strictly stronger requirement than cohesion, and it fails in
two directions rather than one:

    fission   -- a group loses its own members (the failure `cohesive.py` studies)
    fusion    -- groups merge into one undifferentiated blob, so the *labels* die
                 even though the agents are perfectly cohesive

A model that only prevents fission scores perfectly on `largest_cluster_frac` and
still gets the biology wrong: it predicts one super-flock. `core.metrics.group_summary`
measures both directions, because a grouping model can only be judged by both.

The model
---------
`MultiGroupFlock`. Each agent carries an allegiance g_i in {0..k-1}. Per unit time:

    dv_i/dt =  (lam_in / n_{g_i}) * sum_{j: g_j = g_i}  psi(r_ij) (v_j - v_i)    [1]
             -  w_sep * sum_{j: r_ij < r_sep}           d_ij / r_ij^2            [2]
             -  w_seg * sum_{j: g_j != g_i, r_ij < r_seg}  d_ij / r_ij^2         [3]
             +  w_global * (x_com - x_i)                                         [4]
             +  k_speed * (v0 - |v_i|) * v_hat_i                                 [5]

    psi(r) = K / (sigma^2 + r^2)^beta          (Cucker-Smale communication weight)

[1] IN-GROUP COHESION, and the reason this model works in open space. Restricted to
    one group, [1] IS the Cucker-Smale system with N -> n_g. Cucker & Smale's
    theorem therefore applies verbatim to each group in isolation: for beta <= 1/2
    the group flocks UNCONDITIONALLY -- velocity dispersion -> 0 and relative
    positions stay bounded, in unbounded R^d, from ANY initial condition. The
    theorem is imported per group; it is not re-proved here and does not need to be.
    Measured per-group R_g growth (exp2 Part B, open BC, no arena term): 1.09 at
    beta=0.4 against 1.38 at beta=0.9 from a tight start, and 1.00 against 1.50 from
    a spread one. The threshold is visible, and note the SHAPE of it -- beta=0.4 does
    not expand from EITHER initial condition, while beta=0.9 does worse the worse its
    start. That asymmetry is what "unconditional" versus "conditional" means, and it
    is why beta is not just a tuning knob.
[2] Short-range separation. Applies to EVERYONE, group-mates included: birds do not
    overlap.
[3] SEGREGATION, and the reason the groups stay distinguishable. Out-group agents
    repel at medium range, so groups cannot interpenetrate. This is the anti-fusion
    term, and `w_seg=0` is the honest control that isolates it: measured S falls from
    1.00 to ~0.50 -- the same with walls and without -- while `largest_cluster_frac`
    stays at 1.00, `n_fragments` at 1 and group_integrity at 1.00: a perfect score by
    every Experiment-1 metric, for a run in which the groups have merged. The groups
    did not lose one member; they lost only the thing that made them groups.
    Note S PLATEAUS there rather than reaching 0. The residue is real and worth
    understanding: [1] is velocity consensus, so group-mates end up co-moving and
    therefore weakly co-located even inside a merged blob. Alignment alone buys some
    sorting for free -- just not enough to call them groups.
[4] Optional weak attraction to the global centre of mass. Pure segregation would
    push the groups apart forever; [4] holds them in a shared arena so that
    maintenance is actually being TESTED against interference rather than trivially
    achieved by everyone flying away from everyone else. w_global=0 is legitimate --
    it just answers a less interesting question.
[5] Speed relaxation to a cruising speed v0. Birds do not coast to a halt.

HONEST STATEMENT OF WHAT IS AND IS NOT PROVED. Cucker-Smale is a theorem about an
isolated group. Terms [2]-[4] are perturbations to it, and the composite system has
no proof. What [1] buys is that the in-group mechanism cannot be the thing that
fails: psi(r) > 0 everywhere, so a straggler is always still heard by its group,
which is precisely the failure mode that kills every finite-range model in
`cohesive.py`. Whether the groups survive each other is an empirical question, and
`experiments/exp2_grouping.py` measures it rather than asserting it.

ALLEGIANCE. With `switching=False` (default) g is a constant of motion and the model
answers: can fixed groups coexist? With `switching=True` allegiance becomes dynamic
-- an agent whose own group falls below `defect_threshold` of its k_social nearest
neighbours defects to the local majority at rate `switch_rate`. Now the labels
themselves must survive, and maintenance becomes an ACTIVE balance: segregation keeps
each agent's local majority in-group, which is what removes the pressure to defect,
which preserves the groups that do the segregating.

MEASURED (exp2 Part A, N=150, K=3): 0-1 defections of 150 with segregation on,
78-118 with it off. Segregation is what makes allegiance stable, by two orders of
magnitude.

What is NOT true, and was worth being wrong about: turning segregation off does not
destroy the label structure. It scores S = 0.72-0.85, HIGHER than the same run with
allegiance frozen (S ~ 0.50). The reason is the opposite of the obvious reading
-- defection re-sorts the labels to match whatever clumps the geometry has produced,
so purity goes UP while more than half the population changes allegiance. The labels
end up slaved to the geometry rather than organising it. Read S alone and that run
looks BETTER than the frozen one; read S with the defection count and it is
obviously worse. One number cannot tell them apart, which is the same lesson as `M`
in Experiment 1, one level up.
"""

from __future__ import annotations

import numpy as np

from core.base import CollectiveModel, State
from core.metrics import centroid


def assign_groups(n: int, k: int, rng=None, sizes=None) -> np.ndarray:
    """Allegiance vector g in {0..k-1}^n.

    `sizes` gives explicit group sizes (must sum to n); otherwise groups are as
    equal as n // k allows, with the remainder spread over the first groups.
    """
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=int)
        if sizes.sum() != n:
            raise ValueError(f"sizes sum to {sizes.sum()}, expected n={n}")
        return np.repeat(np.arange(len(sizes)), sizes)
    base, extra = divmod(n, k)
    sizes = np.full(k, base)
    sizes[:extra] += 1
    return np.repeat(np.arange(k), sizes)


def init_grouped_state(
    n: int,
    k: int,
    boundary,
    dim: int = 3,
    speed: float = 0.5,
    blob_radius: float = 1.2,
    arena: float = 8.0,
    aligned_within_group: bool = True,
    rng=None,
    sizes=None,
) -> State:
    """k separated blobs, one per group, each with its own initial heading.

    This is the initial condition the grouping question deserves. Starting from a
    well-mixed cloud would conflate two questions -- can the model FORM groups, and
    can it MAINTAIN them -- and only the second one has a clean answer. Groups here
    start separated and internally aligned; everything afterwards is maintenance.

    Blob centres are placed on a circle of radius `arena`/2 (in the first two
    dimensions), inside the box when the boundary is finite.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    groups = assign_groups(n, k, rng, sizes=sizes)
    L = boundary.box_size
    arena = min(arena, 0.7 * L) if L is not None else arena

    centre = np.full(dim, 0.5 * L if L is not None else 0.0)
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)

    x = np.empty((n, dim))
    v = np.empty((n, dim))
    for g in range(k):
        idx = np.flatnonzero(groups == g)
        hub = centre.copy()
        hub[0] += 0.5 * arena * np.cos(angles[g])
        hub[1] += 0.5 * arena * np.sin(angles[g])
        x[idx] = hub + rng.normal(scale=blob_radius, size=(len(idx), dim))

        if aligned_within_group:
            h = rng.normal(size=dim)
            h /= np.linalg.norm(h)
            jitter = rng.normal(scale=0.15, size=(len(idx), dim))
            hv = h[None, :] + jitter
        else:
            hv = rng.normal(size=(len(idx), dim))
        hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        v[idx] = speed * hv

    st = State(positions=boundary.wrap(x), velocities=v)
    st.internal["groups"] = groups
    st.internal["defections"] = 0
    return st


class MultiGroupFlock(CollectiveModel):
    """k coexisting flocks: in-group Cucker-Smale, out-group segregation.

    Parameters
    ----------
    groups     : (N,) int allegiance vector. Ignored if the State already carries
                 `internal["groups"]` (which `init_grouped_state` sets), so the two
                 always agree and switching has one owner.
    K, sigma, beta, lam_in
                 Cucker-Smale weight psi(r) = K/(sigma^2+r^2)^beta and its gain.
                 beta <= 1/2 puts each group in the unconditional-flocking regime.
    w_sep, r_sep : separation from everyone at short range.
    w_seg, r_seg : repulsion from OUT-GROUP agents at medium range. The anti-fusion
                 term. r_seg > r_sep, or segregation is indistinguishable from
                 ordinary collision avoidance.
    w_global   : weak attraction to the global centre of mass (holds the groups in
                 one arena so maintenance is tested rather than assumed).
    v0, k_speed: cruising speed and how hard it is enforced.
    fmax       : acceleration clip. Numerical hygiene -- r^-2 repulsion is singular
                 and dt is finite.
    switching  : if True, allegiance is dynamic (see module docstring).
    defect_threshold, switch_rate, k_social : the defection rule.

    Notes
    -----
    Forces are computed from a dense (N, N, dim) displacement array via
    `boundary.displacement`, so the periodic/open swap is valid here exactly as it
    is everywhere else in this repo. Cost is O(N^2 * dim) memory; fine to a few
    hundred agents, which is the regime this model is for.
    """

    name = "Multi-group flock (in-group CS + segregation)"
    cohesive = True

    def __init__(self, boundary, groups=None, K=2.0, sigma=1.0, beta=0.4,
                 lam_in=1.0, w_sep=1.0, r_sep=0.5, w_seg=2.5, r_seg=1.6,
                 w_global=0.05, v0=0.5, k_speed=1.0, fmax=6.0,
                 switching=False, defect_threshold=0.4, switch_rate=0.5,
                 k_social=7, rng=None):
        super().__init__(boundary, rng)
        self.groups = None if groups is None else np.asarray(groups, dtype=int)
        self.K, self.sigma, self.beta, self.lam_in = K, sigma, beta, lam_in
        self.w_sep, self.r_sep = w_sep, r_sep
        self.w_seg, self.r_seg = w_seg, r_seg
        self.w_global = w_global
        self.v0, self.k_speed, self.fmax = v0, k_speed, fmax
        self.switching = switching
        self.defect_threshold = defect_threshold
        self.switch_rate = switch_rate
        self.k_social = k_social

    @property
    def unconditional(self) -> bool:
        """True iff beta <= 1/2: each group is in Cucker-Smale's proved regime."""
        return self.beta <= 0.5

    @property
    def segregating(self) -> bool:
        """True iff an anti-fusion mechanism is present at all."""
        return self.w_seg > 0.0 and self.r_seg > 0.0

    def psi(self, r):
        """Cucker-Smale communication weight. Strictly positive for all r."""
        return self.K / (self.sigma ** 2 + r ** 2) ** self.beta

    def _groups_of(self, state: State) -> np.ndarray:
        g = state.internal.get("groups")
        if g is None:
            if self.groups is None:
                raise ValueError(
                    "No allegiance vector. Pass groups= to the model, or build the "
                    "state with init_grouped_state()."
                )
            g = self.groups.copy()
            state.internal["groups"] = g
        return g

    def _switch(self, state, g, r, dt):
        """Defection to the local majority. Mutates and returns g."""
        n = len(g)
        k_eff = min(self.k_social, n - 1)
        if k_eff <= 0:
            return g
        # k nearest by the boundary-consistent metric (r has inf on the diagonal).
        nearest = np.argpartition(r, k_eff - 1, axis=1)[:, :k_eff]
        labels = g[nearest]                                  # (n, k_eff)
        n_groups = int(g.max()) + 1
        # counts[i, c] = how many of i's k nearest neighbours are in group c
        counts = np.zeros((n, n_groups), dtype=int)
        np.add.at(counts, (np.repeat(np.arange(n), k_eff), labels.ravel()), 1)

        own = counts[np.arange(n), g] / k_eff
        majority = counts.argmax(axis=1)
        p = 1.0 - np.exp(-self.switch_rate * dt)
        flip = ((own < self.defect_threshold)
                & (majority != g)
                & (self.rng.random(n) < p))
        if flip.any():
            g = g.copy()
            g[flip] = majority[flip]
            state.internal["defections"] = (
                state.internal.get("defections", 0) + int(flip.sum()))
        return g

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        n, dim = s.n, s.dim
        g = self._groups_of(s)

        # d[i, j] = vector from i to j, under THIS topology. Everything below is
        # built from this one array, which is what makes the BC swap honest.
        d = b.displacement(s.positions[:, None, :], s.positions[None, :, :])
        r = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(r, np.inf)                # self never interacts

        if self.switching:
            g = self._switch(s, g, r, dt)
        same = g[:, None] == g[None, :]

        acc = np.zeros((n, dim))

        # --- [1] in-group Cucker-Smale alignment ------------------------------
        # Per-group normalisation (not /N) so the theorem transfers with n_g.
        w = np.where(same, self.psi(r), 0.0)
        np.fill_diagonal(w, 0.0)
        dv = s.velocities[None, :, :] - s.velocities[:, None, :]   # v_j - v_i
        n_g = np.bincount(g, minlength=int(g.max()) + 1)[g].astype(float)
        acc += self.lam_in * (w[:, :, None] * dv).sum(axis=1) / n_g[:, None]

        # --- [2] separation from everyone at short range ----------------------
        if self.w_sep > 0.0:
            m = r < self.r_sep
            if m.any():
                acc -= self.w_sep * self._inverse_square(d, r, m)

        # --- [3] segregation from the out-group -------------------------------
        if self.segregating:
            m = (r < self.r_seg) & ~same
            if m.any():
                acc -= self.w_seg * self._inverse_square(d, r, m)

        # --- [4] weak global cohesion (shared arena) --------------------------
        if self.w_global > 0.0:
            # centroid(), not positions.mean(): on a torus the naive mean of
            # coordinates is not a point on the flock at all (agents at x=0.1 and
            # x=L-0.1 are neighbours, and average to the far side of the box), so
            # attracting to it would be an artifact of the wrapping, not a force.
            # centroid() uses the circular mean and is correct under every boundary.
            com = centroid(s.positions, b)
            acc += self.w_global * b.displacement(s.positions, com[None, :])

        # --- [5] cruising-speed relaxation ------------------------------------
        if self.k_speed > 0.0:
            sp = s.speeds[:, None]
            acc += self.k_speed * (self.v0 - sp) * (s.velocities /
                                                    np.where(sp > 1e-12, sp, 1.0))

        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        acc *= np.minimum(1.0, self.fmax / np.where(nrm > 1e-12, nrm, 1.0))

        s.velocities = s.velocities + acc * dt
        s.positions = s.positions + s.velocities * dt
        s.internal["groups"] = g
        s.t += dt
        return self.finalize(s)

    @staticmethod
    def _inverse_square(d, r, mask):
        """sum_j (d_ij / r_ij^2) over the masked pairs. Points i -> j, so the
        caller SUBTRACTS it to push i away from j."""
        rr = np.where(mask, r, np.inf)
        contrib = d / (rr ** 2 + 1e-9)[:, :, None]
        return np.where(mask[:, :, None], contrib, 0.0).sum(axis=1)
