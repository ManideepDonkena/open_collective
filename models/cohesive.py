"""
COHESIVE MODELS.  cohesive = True.

These contain a genuine mechanism that holds the group together in unbounded
space, with no help from periodic walls. This is what you need if you want to
model a real flock rather than a torus artefact.

Included, roughly in order of increasing rigour:

  * BoidsModel      -- Reynolds, SIGGRAPH 1987. Separation / alignment / cohesion.
                       Heuristic but works.
  * CouzinModel     -- Couzin, Krause, James, Ruxton & Franks, J. Theor. Biol. 218,
                       1 (2002). Zonal: repulsion < orientation < attraction, with
                       a rear blind angle. The standard biological model. Cohesion
                       is unconditional because the attraction zone is unbounded in
                       practice (zoa is large) and repulsion prevents collapse.
  * DOrsognaModel   -- D'Orsogna, Chuang, Bertozzi & Chayes, PRL 96, 104302 (2006).
                       Self-propulsion (alpha - beta|v|^2)v plus a Morse potential.
                       Produces the mill / flock / ring phases in FREE SPACE. The
                       H-stability condition C*l^2 = (Cr/Ca)(lr/la)^2 decides
                       whether the group stays a bounded blob.
  * CuckerSmaleModel-- Cucker & Smale, IEEE TAC 52, 852 (2007). THE rigorous one.
                       psi(r) = K/(sigma^2 + r^2)^beta. THEOREM: if beta <= 1/2,
                       flocking is UNCONDITIONAL -- velocities converge and the
                       relative positions stay bounded, for ANY initial condition,
                       in unbounded R^d. If beta > 1/2 flocking is conditional.
                       This is the direct mathematical answer to "will it hold
                       together without a box?".
  * OlfatiSaberModel-- Olfati-Saber, IEEE TAC 51, 401 (2006). Gradient-based
                       flocking with sigma-norms and bump functions, plus a
                       navigational feedback (gamma-agent) term. Provably produces
                       a lattice-like cohesive formation in free space.
"""

from __future__ import annotations

import numpy as np

from core.base import CollectiveModel, State
from core.neighbors import metric_neighbors, topological_neighbors


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, 1.0)


def _clip_norm(v, vmax):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, vmax / np.where(n > 1e-12, n, 1.0))
    return v * scale


class BoidsModel(CollectiveModel):
    """Reynolds 1987 boids: separation + alignment + cohesion.

    The COHESION term (steer toward the local centre of mass) is the thing Vicsek
    lacks. It is what lets this model survive open boundaries.
    """

    name = "Boids (Reynolds 1987)"
    cohesive = True

    def __init__(self, boundary, r_sep=0.5, r_ali=2.0, r_coh=4.0,
                 w_sep=1.5, w_ali=1.0, w_coh=1.0, v0=0.5, vmax=1.0,
                 fmax=2.0, rng=None):
        super().__init__(boundary, rng)
        self.r_sep, self.r_ali, self.r_coh = r_sep, r_ali, r_coh
        self.w_sep, self.w_ali, self.w_coh = w_sep, w_ali, w_coh
        self.v0, self.vmax, self.fmax = v0, vmax, fmax

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        neigh = metric_neighbors(s.positions, b, self.r_coh)
        acc = np.zeros_like(s.velocities)

        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            d = b.displacement(s.positions[i], s.positions[cand])   # i -> j
            dist = np.linalg.norm(d, axis=1)

            # --- separation: push away from anyone too close ------------------
            close = dist < self.r_sep
            if close.any():
                dd = d[close]
                rr = dist[close][:, None]
                acc[i] -= self.w_sep * (dd / (rr ** 2 + 1e-9)).sum(axis=0)

            # --- alignment: match neighbour headings --------------------------
            ali = dist < self.r_ali
            if ali.any():
                acc[i] += self.w_ali * (s.velocities[cand[ali]].mean(axis=0)
                                        - s.velocities[i])

            # --- cohesion: steer toward the local centre of mass --------------
            # THIS is the term Vicsek and both Beuria models are missing.
            acc[i] += self.w_coh * d.mean(axis=0)

        acc = _clip_norm(acc, self.fmax)
        v = _clip_norm(s.velocities + acc * dt, self.vmax)
        s.velocities = v
        s.positions = s.positions + v * dt
        s.t += dt
        return self.finalize(s)


class CouzinModel(CollectiveModel):
    """Couzin et al. 2002 zonal model. The standard biological flocking model.

    Three concentric zones about each agent:
        zor : zone of repulsion    (r < zor)          -> move directly away
        zoo : zone of orientation  (zor <= r < zoo)   -> align headings
        zoa : zone of attraction   (zoo <= r < zoa)   -> move toward
    plus a rear blind angle of width `blind`.

    Rule (important): if ANY neighbour is in the repulsion zone, repulsion
    OVERRIDES everything else. Otherwise orientation and attraction are averaged.
    Turning is rate-limited by theta_dot_max, which is what produces the
    characteristic swarm / torus / dynamic-parallel phases.
    """

    name = "Couzin et al. 2002 (zonal)"
    cohesive = True

    def __init__(self, boundary, zor=0.5, zoo=2.0, zoa=6.0, blind=np.deg2rad(60),
                 v0=0.5, theta_dot_max=np.deg2rad(100), sigma=0.05, rng=None):
        super().__init__(boundary, rng)
        self.zor, self.zoo, self.zoa = zor, zoo, zoa
        self.blind, self.v0 = blind, v0
        self.theta_dot_max, self.sigma = theta_dot_max, sigma

    def _visible(self, d, dist, h_i):
        """Mask out neighbours in the rear blind cone."""
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = d / dist[:, None]
        cosang = unit @ h_i
        # blind spot of full width `blind` centred directly behind
        return cosang > np.cos(np.pi - 0.5 * self.blind)

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        neigh = metric_neighbors(s.positions, b, self.zoa)
        h = s.headings
        desired = h.copy()

        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            d = b.displacement(s.positions[i], s.positions[cand])
            dist = np.linalg.norm(d, axis=1)
            ok = dist > 1e-9
            cand, d, dist = cand[ok], d[ok], dist[ok]
            if len(cand) == 0:
                continue

            vis = self._visible(d, dist, h[i])
            cand, d, dist = cand[vis], d[vis], dist[vis]
            if len(cand) == 0:
                continue

            rep = dist < self.zor
            if rep.any():
                # Repulsion overrides. Sum of unit vectors AWAY from neighbours.
                dr = -(d[rep] / dist[rep][:, None]).sum(axis=0)
                desired[i] = dr
            else:
                ori = (dist >= self.zor) & (dist < self.zoo)
                att = (dist >= self.zoo) & (dist < self.zoa)
                do = h[cand[ori]].sum(axis=0) if ori.any() else np.zeros(s.dim)
                da = (d[att] / dist[att][:, None]).sum(axis=0) if att.any() else np.zeros(s.dim)
                if ori.any() and att.any():
                    desired[i] = 0.5 * (_normalize(do) + _normalize(da))
                elif ori.any():
                    desired[i] = do
                elif att.any():
                    desired[i] = da

        desired = _normalize(desired + self.sigma * self.rng.normal(size=h.shape))

        # rate-limited turning
        if s.dim == 2:
            cur = np.arctan2(h[:, 1], h[:, 0])
            tgt = np.arctan2(desired[:, 1], desired[:, 0])
            dth = np.arctan2(np.sin(tgt - cur), np.cos(tgt - cur))
            dth = np.clip(dth, -self.theta_dot_max * dt, self.theta_dot_max * dt)
            new = cur + dth
            h_new = np.stack([np.cos(new), np.sin(new)], axis=1)
        else:
            h_new = _normalize(h + np.clip(desired - h, -self.theta_dot_max * dt,
                                           self.theta_dot_max * dt))

        s.velocities = self.v0 * h_new
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class DOrsognaModel(CollectiveModel):
    """D'Orsogna et al., PRL 96, 104302 (2006). Self-propelled Morse interaction.

        dv_i/dt = (alpha - beta |v_i|^2) v_i - grad_i sum_{j != i} U(|x_i - x_j|)
        U(r)    = -Ca exp(-r/la) + Cr exp(-r/lr)

    Free-space model: produces coherent flocks, single mills, double mills and
    rings with NO boundaries at all. Terminal speed is sqrt(alpha/beta).

    H-stability: with C = Cr/Ca and l = lr/la, the group stays a bounded,
    finite-density blob when C*l^2 > 1 (H-stable) and collapses into a
    catastrophic dense core when C*l^2 < 1. `hstability_ratio` reports it.
    """

    name = "D'Orsogna et al. 2006 (Morse)"
    cohesive = True

    def __init__(self, boundary, alpha=1.0, beta=0.5, Ca=0.5, la=2.0,
                 Cr=1.0, lr=0.5, cutoff=12.0, rng=None):
        super().__init__(boundary, rng)
        self.alpha, self.beta = alpha, beta
        self.Ca, self.la, self.Cr, self.lr = Ca, la, Cr, lr
        self.cutoff = cutoff

    @property
    def hstability_ratio(self) -> float:
        """C * l^2. > 1 => H-stable (bounded blob). < 1 => collapse."""
        C = self.Cr / self.Ca
        l = self.lr / self.la
        return C * l * l

    @property
    def terminal_speed(self) -> float:
        return float(np.sqrt(self.alpha / self.beta))

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        neigh = metric_neighbors(s.positions, b, self.cutoff)
        v = s.velocities
        speed2 = (v ** 2).sum(axis=1, keepdims=True)
        acc = (self.alpha - self.beta * speed2) * v          # self-propulsion

        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            d = b.displacement(s.positions[i], s.positions[cand])  # i -> j
            r = np.linalg.norm(d, axis=1)
            ok = r > 1e-9
            d, r = d[ok], r[ok]
            if len(r) == 0:
                continue
            u = d / r[:, None]
            # -dU/dr, projected. Force on i from j is -grad_i U = (dU/dr) * u_hat
            dUdr = (self.Ca / self.la) * np.exp(-r / self.la) \
                   - (self.Cr / self.lr) * np.exp(-r / self.lr)
            acc[i] += (dUdr[:, None] * u).sum(axis=0)

        s.velocities = v + acc * dt
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class CuckerSmaleModel(CollectiveModel):
    """Cucker & Smale, IEEE TAC 52, 852 (2007). THE rigorous open-space model.

        dv_i/dt = (lambda/N) sum_j psi(|x_i - x_j|) (v_j - v_i)
        psi(r)  = K / (sigma^2 + r^2)^beta

    THEOREM (Cucker-Smale). If beta <= 1/2 the system flocks UNCONDITIONALLY:
    velocity dispersion -> 0 and the relative positions remain bounded, for ANY
    initial condition, in unbounded R^d with no boundary whatsoever. If beta > 1/2
    flocking is only conditional -- it holds when the initial dispersion is small
    relative to the initial spread.

    This is precisely the result that the Vicsek family lacks. The mechanism is not
    an attractive force at all: it is a communication weight that decays SLOWLY
    ENOUGH that agents can never fully lose contact. `unconditional` reports
    whether your beta is in the safe regime.

    Note: CS gives velocity consensus and bounded relative positions, but does NOT
    prevent the group's centre of mass from translating -- momentum is conserved.
    That is correct physics, not a defect.
    """

    name = "Cucker-Smale 2007"
    cohesive = True

    def __init__(self, boundary, K=1.0, sigma=1.0, beta=0.4, lam=1.0,
                 cutoff=None, rng=None):
        super().__init__(boundary, rng)
        self.K, self.sigma, self.beta, self.lam = K, sigma, beta, lam
        self.cutoff = cutoff       # None = all-to-all, as in the theorem

    @property
    def unconditional(self) -> bool:
        """True iff beta <= 1/2, the unconditional-flocking regime."""
        return self.beta <= 0.5

    def psi(self, r):
        return self.K / (self.sigma ** 2 + r ** 2) ** self.beta

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        n = s.n
        acc = np.zeros_like(s.velocities)

        if self.cutoff is None:
            for i in range(n):
                d = b.displacement(s.positions[i], s.positions)     # i -> all
                r = np.linalg.norm(d, axis=1)
                w = self.psi(r)
                w[i] = 0.0
                acc[i] = (self.lam / n) * (w[:, None] *
                                           (s.velocities - s.velocities[i])).sum(axis=0)
        else:
            neigh = metric_neighbors(s.positions, b, self.cutoff)
            for i, cand in enumerate(neigh):
                if len(cand) == 0:
                    continue
                r = b.distance(s.positions[i], s.positions[cand])
                w = self.psi(r)
                acc[i] = (self.lam / n) * (w[:, None] *
                                           (s.velocities[cand] - s.velocities[i])).sum(axis=0)

        s.velocities = s.velocities + acc * dt
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class OlfatiSaberModel(CollectiveModel):
    """Olfati-Saber, IEEE TAC 51, 401 (2006). Algorithm 2 (with gamma-agent).

        u_i = c1_a * sum_j phi_a(||q_j-q_i||_s) n_ij
            + c2_a * sum_j a_ij(q) (p_j - p_i)          [gradient + consensus]
            - c1_g (q_i - q_g) - c2_g (p_i - p_g)       [navigational feedback]

    Uses the sigma-norm  ||z||_s = (1/eps)(sqrt(1 + eps|z|^2) - 1), which is
    differentiable everywhere, and a bump function rho_h for smooth cutoff.

    Provably converges to an alpha-lattice: a cohesive, collision-free formation
    with uniform spacing d, in free space. This is the control-theoretic gold
    standard for open-boundary flocking -- and it is the natural meeting point
    with Tripathy's formation-control work.
    """

    name = "Olfati-Saber 2006"
    cohesive = True

    def __init__(self, boundary, d=1.0, kappa_ratio=1.2, eps=0.1, h=0.2,
                 a=5.0, b_=5.0, c1_a=1.0, c2_a=2.0, c1_g=0.3, c2_g=0.6,
                 q_gamma=None, p_gamma=None, rng=None):
        super().__init__(boundary, rng)
        self.d, self.r = d, kappa_ratio * d
        self.eps, self.h = eps, h
        self.a, self.b = a, b_
        self.c = abs(a - b_) / np.sqrt(4.0 * a * b_)
        self.c1_a, self.c2_a = c1_a, c2_a
        self.c1_g, self.c2_g = c1_g, c2_g
        self.q_gamma = q_gamma
        self.p_gamma = p_gamma
        self.d_s = self._sigma_norm_scalar(d)
        self.r_s = self._sigma_norm_scalar(self.r)

    # --- sigma-norm machinery -------------------------------------------------
    def _sigma_norm_scalar(self, z):
        return (np.sqrt(1.0 + self.eps * z ** 2) - 1.0) / self.eps

    def _sigma_norm(self, z):
        return (np.sqrt(1.0 + self.eps * (z ** 2).sum(axis=-1)) - 1.0) / self.eps

    def _sigma_grad(self, z):
        denom = 1.0 + self.eps * self._sigma_norm(z)
        return z / denom[:, None]

    def _rho_h(self, z):
        out = np.zeros_like(z)
        m1 = z < self.h
        m2 = (z >= self.h) & (z <= 1.0)
        out[m1] = 1.0
        out[m2] = 0.5 * (1.0 + np.cos(np.pi * (z[m2] - self.h) / (1.0 - self.h)))
        return out

    def _phi(self, z):
        sig = (z + self.c) / np.sqrt(1.0 + (z + self.c) ** 2)
        return 0.5 * ((self.a + self.b) * sig + (self.a - self.b))

    def _phi_alpha(self, z):
        return self._rho_h(z / self.r_s) * self._phi(z - self.d_s)

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        neigh = metric_neighbors(s.positions, b, self.r)
        u = np.zeros_like(s.velocities)

        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            z = b.displacement(s.positions[i], s.positions[cand])   # q_j - q_i
            z_s = self._sigma_norm(z)
            n_ij = self._sigma_grad(z)
            # gradient term (spacing -> d)
            u[i] += self.c1_a * (self._phi_alpha(z_s)[:, None] * n_ij).sum(axis=0)
            # velocity consensus term, weighted by the spatial adjacency
            a_ij = self._rho_h(z_s / self.r_s)
            u[i] += self.c2_a * (a_ij[:, None] *
                                 (s.velocities[cand] - s.velocities[i])).sum(axis=0)

        # navigational feedback toward a gamma-agent (optional)
        if self.q_gamma is not None:
            qg = np.asarray(self.q_gamma, dtype=float)
            pg = np.asarray(self.p_gamma if self.p_gamma is not None
                            else np.zeros(s.dim), dtype=float)
            u += -self.c1_g * (-b.displacement(s.positions, qg[None, :])) \
                 - self.c2_g * (s.velocities - pg[None, :])

        s.velocities = s.velocities + u * dt
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)
