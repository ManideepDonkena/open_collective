"""
Additional collective-motion models, surveyed from the active-matter / flocking
literature and implemented on the common CollectiveModel interface.

Every model carries the honest `cohesive` flag used throughout this repo (does it
contain a mechanism that holds the group together WITHOUT periodic walls?), and
obtains inter-agent separations via `self.boundary.displacement`, so the
periodic / open / reflecting swap stays valid. Angle-based models set
`two_d_only = True`.

References
  Grégoire & Chaté, PRL 92, 025702 (2004)              -- cohesive Vicsek
  Chaté, Ginelli, Grégoire, Raynaud, PRE 77, 046113 (2008) -- vectorial noise
  Cavagna et al., J. Stat. Phys. 158, 601 (2015)        -- inertial spin model
  Fily & Marchetti, PRL 108, 235702 (2012)              -- active Brownian particles
  Szabó et al., PRE 74, 061908 (2006)                   -- self-propelled cells
  O'Keeffe, Hong & Strogatz, Nat. Commun. 8, 1504 (2017)-- swarmalators
  Berg, Random Walks in Biology (1993)                  -- run-and-tumble
"""

from __future__ import annotations

import numpy as np

from core.base import CollectiveModel, State
from core.neighbors import metric_neighbors, topological_neighbors


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, 1.0)


def _random_unit(rng, n, dim):
    if dim == 2:
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    return _normalize(rng.normal(size=(n, dim)))


def _angles(v):
    return np.arctan2(v[:, 1], v[:, 0])


def _dense_disp(boundary, x):
    """(D, R): D[i,j] = x_j - x_i (boundary-aware); R[i,j] = |D|, diagonal = inf."""
    D = boundary.displacement(x[:, None, :], x[None, :, :])
    R = np.linalg.norm(D, axis=2)
    np.fill_diagonal(R, np.inf)
    return D, R


# ==========================================================================
# Alignment-only variants (cohesive = False)
# ==========================================================================

class VicsekVectorialNoise(CollectiveModel):
    """Vicsek with EXTRINSIC (vectorial) noise -- Chaté et al. PRE 77 (2008).

    Standard Vicsek adds angular noise to the mean heading. Here the noise is a
    random vector added to the neighbour-vector SUM before normalising, scaled by
    the neighbourhood size -- modelling error in measuring neighbours rather than
    in acting. This changes the order-disorder transition from continuous to
    discontinuous. Still alignment-only, so it disperses in open space.
    """

    name = "Vicsek (vectorial noise)"
    cohesive = False

    def __init__(self, boundary, r_max=1.0, eta=0.4, v0=0.5, r_min=0.0, rng=None):
        super().__init__(boundary, rng)
        self.r_max, self.r_min, self.eta, self.v0 = r_max, r_min, eta, v0

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        neigh = metric_neighbors(s.positions, self.boundary, self.r_max, self.r_min)
        h = s.headings
        new_h = np.empty_like(h)
        for i, cand in enumerate(neigh):
            vsum = h[i] + (h[cand].sum(axis=0) if len(cand) else 0.0)
            ncount = len(cand) + 1
            noise = self.eta * ncount * _random_unit(self.rng, 1, s.dim)[0]
            new_h[i] = _normalize((vsum + noise)[None, :])[0]
        s.velocities = self.v0 * new_h
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class InertialSpinModel(CollectiveModel):
    """Inertial spin model -- Cavagna et al., J. Stat. Phys. 158 (2015). 2D.

    Adds an internal spin s_i (angular momentum conjugate to the heading angle).
    The social force acts on the SPIN, and the spin rotates the velocity, giving
    the flock orientational inertia and propagating turn ("spin") waves:

        dphi_i/dt = s_i / chi
        ds_i/dt   = -(eta/chi) s_i + (J/|N_i|) sum_j sin(phi_j - phi_i) + noise

    chi is the generalized inertia, eta the friction. chi -> 0 recovers Vicsek.
    Alignment-only: disperses in open space.
    """

    name = "Inertial spin (Cavagna)"
    cohesive = False
    two_d_only = True

    def __init__(self, boundary, r_max=1.0, J=1.0, chi=1.0, eta=0.3, v0=0.5,
                 sigma=0.1, topological=False, k=7, rng=None):
        super().__init__(boundary, rng)
        self.r_max, self.J, self.chi, self.eta = r_max, J, chi, eta
        self.v0, self.sigma = v0, sigma
        self.topological, self.k = topological, k

    def _neighbors(self, state):
        if self.topological:
            return topological_neighbors(state.positions, self.boundary, self.k)
        return metric_neighbors(state.positions, self.boundary, self.r_max)

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("InertialSpinModel is 2D.")
        s = state.copy()
        if "spin" not in s.internal:
            s.internal["spin"] = np.zeros(s.n)
        spin = s.internal["spin"]
        phi = _angles(s.velocities)
        neigh = self._neighbors(s)
        torque = np.zeros(s.n)
        for i, cand in enumerate(neigh):
            if len(cand):
                torque[i] = self.J * np.mean(np.sin(phi[cand] - phi[i]))
        spin = spin + (-(self.eta / self.chi) * spin + torque) * dt \
            + self.sigma * self.rng.standard_normal(s.n) * np.sqrt(dt)
        phi = phi + (spin / self.chi) * dt
        s.velocities = self.v0 * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        s.positions = s.positions + s.velocities * dt
        s.internal["spin"] = spin
        s.t += dt
        return self.finalize(s)


class ActiveBrownianParticles(CollectiveModel):
    """Active Brownian particles -- Fily & Marchetti, PRL 108 (2012). 2D.

    NO alignment. Each particle self-propels at v0 along theta_i, which does pure
    rotational diffusion; particles repel on contact (soft harmonic core). Purely
    repulsive active particles still cluster via Motility-Induced Phase Separation
    (MIPS) -- collective structure with no aligning interaction at all.

        theta_i    += sqrt(2 D_r dt) * xi
        x_i        += (v0 e(theta_i) + mu * F_rep_i) dt
    """

    name = "Active Brownian particles"
    cohesive = False
    two_d_only = True

    def __init__(self, boundary, v0=0.5, Dr=0.5, sigma=1.0, k_rep=10.0, mu=1.0,
                 rng=None):
        super().__init__(boundary, rng)
        self.v0, self.Dr, self.sigma = v0, Dr, sigma
        self.k_rep, self.mu = k_rep, mu

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("ActiveBrownianParticles is 2D.")
        s = state.copy()
        if "theta" not in s.internal:
            s.internal["theta"] = _angles(s.velocities)
        theta = s.internal["theta"] + np.sqrt(2 * self.Dr * dt) * self.rng.standard_normal(s.n)

        F = np.zeros((s.n, 2))
        neigh = metric_neighbors(s.positions, self.boundary, self.sigma)
        for i, cand in enumerate(neigh):
            if not len(cand):
                continue
            d = self.boundary.displacement(s.positions[i], s.positions[cand])  # to j
            r = np.linalg.norm(d, axis=1)
            m = r > 1e-9
            overlap = self.k_rep * (self.sigma - r[m])
            F[i] -= (overlap[:, None] * d[m] / r[m, None]).sum(axis=0)  # push away

        prop = self.v0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        s.velocities = prop
        s.positions = s.positions + (prop + self.mu * F) * dt
        s.internal["theta"] = theta
        s.t += dt
        return self.finalize(s)


class RunAndTumbleModel(CollectiveModel):
    """Run-and-tumble motility (E. coli) -- Berg (1993). 2D.

    Each agent runs straight at v0, and 'tumbles' to a random new direction at
    rate `tumble_rate` (probability tumble_rate*dt per step). Optional soft
    repulsion prevents overlap. Alignment-free.
    """

    name = "Run-and-tumble"
    cohesive = False
    two_d_only = True

    def __init__(self, boundary, v0=0.5, tumble_rate=1.0, sigma=1.0, k_rep=5.0,
                 mu=1.0, rng=None):
        super().__init__(boundary, rng)
        self.v0, self.tumble_rate = v0, tumble_rate
        self.sigma, self.k_rep, self.mu = sigma, k_rep, mu

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("RunAndTumbleModel is 2D.")
        s = state.copy()
        if "theta" not in s.internal:
            s.internal["theta"] = _angles(s.velocities)
        theta = s.internal["theta"].copy()
        tumble = self.rng.random(s.n) < self.tumble_rate * dt
        theta[tumble] = self.rng.uniform(0, 2 * np.pi, size=int(tumble.sum()))

        F = np.zeros((s.n, 2))
        if self.k_rep > 0:
            neigh = metric_neighbors(s.positions, self.boundary, self.sigma)
            for i, cand in enumerate(neigh):
                if not len(cand):
                    continue
                d = self.boundary.displacement(s.positions[i], s.positions[cand])
                r = np.linalg.norm(d, axis=1)
                m = r > 1e-9
                F[i] -= ((self.k_rep * (self.sigma - r[m]))[:, None]
                         * d[m] / r[m, None]).sum(axis=0)

        prop = self.v0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        s.velocities = prop
        s.positions = s.positions + (prop + self.mu * F) * dt
        s.internal["theta"] = theta
        s.t += dt
        return self.finalize(s)


# ==========================================================================
# Cohesive models (cohesive = True)
# ==========================================================================

class GregoireChateModel(CollectiveModel):
    """Cohesive Vicsek -- Grégoire & Chaté, PRL 92 (2004).

    Vicsek alignment PLUS a pairwise body force (hard-core repulsion below r_c,
    linear attraction that vanishes at r_e and saturates beyond r_a). The
    attraction is a genuine cohesive mechanism, so -- unlike plain Vicsek -- the
    flock can stay together in open space (given initial connectivity within
    r_max, like Boids).

        u_i = sum_j vhat_j + beta * sum_j f(r_ij) ehat_ij       (+ noise)
        f(r) = -f_rep (r<r_c);  (r-r_e)/(r_a-r_e) (r<r_a);  1 (r>=r_a)
    """

    name = "Grégoire–Chaté (cohesive Vicsek)"
    cohesive = True

    def __init__(self, boundary, r_max=3.0, r_c=0.2, r_e=0.5, r_a=1.0,
                 beta=0.8, eta=0.2, v0=0.5, f_rep=8.0, rng=None):
        super().__init__(boundary, rng)
        self.r_max, self.r_c, self.r_e, self.r_a = r_max, r_c, r_e, r_a
        self.beta, self.eta, self.v0, self.f_rep = beta, eta, v0, f_rep

    def _fmag(self, r):
        out = np.where(r < self.r_a, (r - self.r_e) / (self.r_a - self.r_e), 1.0)
        out = np.where(r < self.r_c, -self.f_rep, out)
        return out

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        neigh = metric_neighbors(s.positions, self.boundary, self.r_max)
        h = s.headings
        u = np.empty_like(h)
        for i, cand in enumerate(neigh):
            align = h[i] + (h[cand].sum(axis=0) if len(cand) else 0.0)
            force = np.zeros(s.dim)
            if len(cand):
                d = self.boundary.displacement(s.positions[i], s.positions[cand])
                r = np.linalg.norm(d, axis=1)
                m = r > 1e-9
                force = (self._fmag(r[m])[:, None] * d[m] / r[m, None]).sum(axis=0)
            noise = self.eta * (len(cand) + 1) * _random_unit(self.rng, 1, s.dim)[0]
            u[i] = align + self.beta * force + noise
        s.velocities = self.v0 * _normalize(u)
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class SzaboModel(CollectiveModel):
    """Self-propelled biological cells -- Szabó et al., PRE 74 (2006). 2D.

    Each cell moves at v0 along its polarity n_i, plus intercellular forces
    (short-range repulsion, mid-range adhesion). The polarity then relaxes toward
    the cell's ACTUAL velocity direction with time constant tau, plus angular
    noise -- an effective alignment that emerges from motion, not direct heading
    copying. Adhesion makes it cohesive.
    """

    name = "Szabó (self-propelled cells)"
    cohesive = True
    two_d_only = True

    def __init__(self, boundary, v0=0.5, r_eq=0.6, r_0=1.2, F_rep=15.0, F_adh=2.0,
                 tau=1.0, eta=0.2, mu=1.0, rng=None):
        super().__init__(boundary, rng)
        self.v0, self.r_eq, self.r_0 = v0, r_eq, r_0
        self.F_rep, self.F_adh, self.tau, self.eta, self.mu = F_rep, F_adh, tau, eta, mu

    def _fmag(self, r):
        rep = self.F_rep * (r - self.r_eq) / self.r_eq              # <0 for r<r_eq
        adh = self.F_adh * (r - self.r_eq) / (self.r_0 - self.r_eq)  # >0 for r>r_eq
        out = np.where(r < self.r_eq, rep, adh)
        return np.where(r < self.r_0, out, 0.0)

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("SzaboModel is 2D.")
        s = state.copy()
        if "theta" not in s.internal:
            s.internal["theta"] = _angles(s.velocities)
        theta = s.internal["theta"]
        nhat = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        F = np.zeros((s.n, 2))
        neigh = metric_neighbors(s.positions, self.boundary, self.r_0)
        for i, cand in enumerate(neigh):
            if not len(cand):
                continue
            d = self.boundary.displacement(s.positions[i], s.positions[cand])
            r = np.linalg.norm(d, axis=1)
            m = r > 1e-9
            F[i] += (self._fmag(r[m])[:, None] * d[m] / r[m, None]).sum(axis=0)

        vel = self.v0 * nhat + self.mu * F
        target = _angles(vel)
        dtheta = np.arctan2(np.sin(target - theta), np.cos(target - theta))  # wrapped
        theta = theta + (dtheta / self.tau) * dt \
            + self.eta * self.rng.standard_normal(s.n) * np.sqrt(dt)

        s.velocities = self.v0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        s.positions = s.positions + vel * dt
        s.internal["theta"] = theta
        s.t += dt
        return self.finalize(s)


class SwarmalatorModel(CollectiveModel):
    """Swarmalators -- O'Keeffe, Hong & Strogatz, Nat. Commun. 8 (2017). 2D.

    Agents that both swarm (in space) and oscillate (an internal phase), with the
    two coupled: spatial attraction is modulated by phase similarity, and phase
    coupling weakens with distance. Produces static sync, phase waves, and
    'splintered' states.

        dx_i/dt = <(x_j-x_i)/|x_j-x_i| (A + J cos(theta_j-theta_i))
                   - B (x_j-x_i)/|x_j-x_i|^2 >_j
        dtheta_i/dt = (K/N) sum_j sin(theta_j - theta_i) / |x_j - x_i|
    """

    name = "Swarmalator (O'Keeffe–Hong–Strogatz)"
    cohesive = True
    two_d_only = True

    def __init__(self, boundary, J=0.8, K=1.0, A=1.0, B=1.0, rng=None):
        super().__init__(boundary, rng)
        self.J, self.K, self.A, self.B = J, K, A, B

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("SwarmalatorModel is 2D.")
        s = state.copy()
        n = s.n
        if "phase" not in s.internal:
            s.internal["phase"] = self.rng.uniform(0, 2 * np.pi, size=n)
        th = s.internal["phase"]

        D, R = _dense_disp(self.boundary, s.positions)      # D[i,j]=x_j-x_i, R diag inf
        U = D / R[:, :, None]                                # unit vectors, 0 on diag
        dphase = th[None, :] - th[:, None]                   # theta_j - theta_i
        attract = (self.A + self.J * np.cos(dphase))[:, :, None]
        vel = np.nansum(U * attract - self.B * D / (R ** 2)[:, :, None], axis=1) / n
        dth = self.K * np.nansum(np.sin(dphase) / R, axis=1) / n

        s.internal["phase"] = th + dth * dt
        s.velocities = vel                                   # motion direction (for metrics)
        s.positions = s.positions + vel * dt
        s.t += dt
        return self.finalize(s)
