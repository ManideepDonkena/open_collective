"""
ALIGNMENT-ONLY MODELS.  cohesive = False.

These models interact ONLY through heading alignment. They contain no attraction
and no distance-dependent cohesion. Under periodic BC they reproduce the famous
order-disorder transition. Under open BC they evaporate.

Included:
  * VicsekModel          -- Vicsek et al., PRL 75, 1226 (1995)
  * PerceptionQuantum    -- Beuria, Chaurasiya & Behera (PAPER_1), the quantum-
                            inspired perception-operator model
  * SlowFastPerception   -- Beuria (PAPER_2), the two-timescale GKSL/Bloch model

The last two are included precisely so you can test them in open space. Both papers
simulate only in a periodic box (PAPER_1: L=10, N=200; PAPER_2: "two-dimensional
periodic domain"). Neither contains an attractive term. The prediction is therefore
that both disperse under OpenBoundary -- i.e. their reported order parameters are
contingent on the torus.
"""

from __future__ import annotations

import numpy as np

from core.base import CollectiveModel, State
from core.neighbors import (metric_neighbors, subsample_neighbors,
                            topological_neighbors, vision_cone_neighbors)


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, 1.0)


def _random_unit(rng, n, dim):
    if dim == 2:
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([np.cos(th), np.sin(th)], axis=1)
    v = rng.normal(size=(n, dim))
    return _normalize(v)


class VicsekModel(CollectiveModel):
    """Vicsek et al. 1995. The canonical alignment-only model.

        p_i(t+1) = p_i(t) + (1/n) sum_{j in N_i} p_j(t) + eta * v0 * e_i(t)

    (Beuria PAPER_1 Eq. 26.) Interaction: metric ball of radius r_max, or the k
    nearest neighbours if `topological=True`.

    NO ATTRACTION. Under OpenBoundary this model disperses -- the group spreads,
    neighbour lists empty, and the flock shatters into independent fragments.
    """

    name = "Vicsek"
    cohesive = False

    def __init__(self, boundary, r_max=1.0, eta=0.2, v0=0.5, r_min=0.0,
                 topological=False, k=7, fast=False, rng=None):
        super().__init__(boundary, rng)
        self.r_max, self.r_min, self.eta, self.v0 = r_max, r_min, eta, v0
        self.topological, self.k = topological, k
        #: fast=True replaces the per-agent Python averaging loop with a single
        #: sparse matrix product (same result up to float round-off, much faster
        #: for large N). Default False so published-table numerics are unchanged.
        self.fast = fast

    def _neighbors(self, state):
        if self.topological:
            return topological_neighbors(state.positions, self.boundary, self.k)
        return metric_neighbors(state.positions, self.boundary, self.r_max, self.r_min)

    @staticmethod
    def _mean_headings_sparse(neigh, h):
        """Mean neighbour heading per agent via one sparse matmul; keeps h[i]
        where an agent has no neighbours. Vectorised equivalent of the loop."""
        from scipy.sparse import csr_matrix
        n = len(h)
        lens = np.fromiter((len(c) for c in neigh), dtype=int, count=n)
        total = int(lens.sum())
        new_h = h.copy()
        if total:
            rows = np.repeat(np.arange(n), lens)
            cols = np.concatenate([c for c in neigh if len(c)])
            A = csr_matrix((np.ones(total), (rows, cols)), shape=(n, n))
            S = np.asarray(A @ h)                    # (n, dim) sum of neighbour headings
            mask = lens > 0
            new_h[mask] = S[mask] / lens[mask, None]
        return new_h

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        neigh = self._neighbors(s)
        h = s.headings
        if self.fast:
            new_h = self._mean_headings_sparse(neigh, h)
        else:
            new_h = np.empty_like(h)
            for i, cand in enumerate(neigh):
                if len(cand) == 0:
                    new_h[i] = h[i]         # isolated -> no signal at all
                else:
                    new_h[i] = h[cand].mean(axis=0)
        noise = self.eta * _random_unit(self.rng, s.n, s.dim)
        new_h = _normalize(new_h + noise)
        s.velocities = self.v0 * new_h
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class PerceptionQuantum(CollectiveModel):
    """Beuria, Chaurasiya & Behera (PAPER_1): quantum-inspired perception operator.

    Each agent builds a perceptual decision state over n neighbours in its vision
    cone, and the driving force is the quantum expectation <O_i^k> of a Hermitian
    perception operator. The paper's key analytic result is that for entangled
    states with POSITIVE coefficients the dynamics reduce to Vicsek:

        Phi+  (Eq. 30) :  p_i(t+1) = p_i + kappa[ (1/2)(p_i1 + p_i2) + eta*e_i ]
        uniform (Eq. 31):  p_i(t+1) = p_i + kappa[ (3/2)(p_i1 + p_i2) + eta*e_i ]
        GHZ3  (Eq. A.6) :  coefficient 1/3
        W3    (Eq. A.6) :  coefficient 9/4

    and that for NEGATIVE coefficients (Phi-, Eq. 32) the neighbour momenta enter
    DESTRUCTIVELY:

        Phi-  (Eq. 32) :  p_i(t+1) = p_i + kappa[ -(1/2)(p_i1 + p_i2) + eta*e_i ]

    which produces no order. We implement the analytic reduction directly (the
    trace has been evaluated in the paper), which is both exact and fast.

    NOTE ON COHESION: every one of these reductions is an average of neighbour
    MOMENTA plus noise. There is no positional attraction anywhere. So this model
    is alignment-only and must disperse in open space, exactly like Vicsek.
    """

    name = "Perception (quantum-inspired, PAPER_1)"
    cohesive = False

    #: Analytic prefactors from the paper's trace evaluations.
    COEFFS = {
        "Phi+": (0.5, +1), "Psi+": (0.5, +1),
        "Phi-": (0.5, -1), "Psi-": (0.5, -1),
        "uniform2": (1.5, +1),
        "GHZ3": (1.0 / 3.0, +1),
        "W3": (9.0 / 4.0, +1),
        "uniform3": (1.5, +1),
    }
    N_NEIGH = {"Phi+": 2, "Psi+": 2, "Phi-": 2, "Psi-": 2, "uniform2": 2,
               "GHZ3": 3, "W3": 3, "uniform3": 3}

    def __init__(self, boundary, state_name="Phi+", alpha=np.pi / 2, r_max=5.0,
                 r_min=0.1, eta=0.2, kappa=1.0, v0=0.5, rng=None):
        super().__init__(boundary, rng)
        if state_name not in self.COEFFS:
            raise ValueError(f"Unknown decision state {state_name!r}. "
                             f"Choose from {list(self.COEFFS)}")
        self.state_name = state_name
        self.coeff, self.sign = self.COEFFS[state_name]
        self.n_neigh = self.N_NEIGH[state_name]
        self.alpha, self.r_max, self.r_min = alpha, r_max, r_min
        self.eta, self.kappa, self.v0 = eta, kappa, v0
        self.name = f"Perception[{state_name}]"

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        neigh = vision_cone_neighbors(s.positions, s.headings, self.boundary,
                                      self.alpha, self.r_max, self.r_min)
        neigh = subsample_neighbors(neigh, self.n_neigh, self.rng)

        p = s.velocities
        drive = np.zeros_like(p)
        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            # sum of neighbour momenta, times the state-dependent prefactor & sign
            drive[i] = self.sign * self.coeff * p[cand].sum(axis=0)

        noise = self.eta * _random_unit(self.rng, s.n, s.dim)
        new_p = p + self.kappa * (drive + noise) * dt
        s.velocities = self.v0 * _normalize(new_p)
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class SlowFastPerception(CollectiveModel):
    """Beuria (PAPER_2): two-timescale perceptual model, GKSL-derived Bloch form.

    Each perceptual channel (i,j) carries a Bloch vector m^{ij} = (mx, my, mz).
    Each agent carries a SLOW regulatory scalar s_i that integrates recent
    alignment and feeds back into the fast field.

    Fast register (Appendix A):
        B_ij      = (-Gamma, 0, -h_eff_ij)
        m_dot|unitary = 2 B_ij x m
        plus GKSL dissipation: T1 longitudinal, T2 transverse relaxation.

    Effective field (main text):
        h_eff_ij = kappa * (p_hat_i . p_hat_j) + beta * s_i + h_ext

    Slow register:
        s_dot_i = gamma_s * ( -s_i + tanh(lambda_fb * <m_z>_i / tau_p) + s_eq )

    Heading map (Eq. 25-26):
        u_i = rho * p_hat_i + sum_j w_ij * (1 + m^{ij}_z)/2 * p_hat_j

    THE MEMORY MECHANISM: the full state (m, s, p_hat, x) is Markovian, but when
    s_i is not observed the reduced dynamics of m and of M(t) become history
    dependent. That is the paper's central claim, and it is real.

    THE COHESION GAP: notice that the heading map mixes only HEADINGS. Positions
    enter solely through the neighbour rule. There is no attraction. The paper uses
    a FIXED topological neighbourhood (w_ij = 1/n, Eq. 24), which keeps the
    interaction graph connected forever -- and that is what conceals the problem.
    Set `fixed_graph=False` to rebuild the topological graph from live positions,
    or use a metric rule, and the model behaves like Vicsek in open space.
    """

    name = "SlowFast Perception (PAPER_2)"
    cohesive = False

    def __init__(self, boundary, k=6, Gamma=0.1, T1=20.0, T2=8.0, kappa=1.0,
                 beta=6.0, gamma_s=1e-3, lam_fb=1e-2, rho=0.7, tau_p=0.25,
                 s_eq=0.0, h_ext=0.0, v0=1.0, fixed_graph=True,
                 metric=False, r_max=None, rng=None):
        super().__init__(boundary, rng)
        self.k, self.Gamma, self.T1, self.T2 = k, Gamma, T1, T2
        self.kappa, self.beta, self.gamma_s = kappa, beta, gamma_s
        self.lam_fb, self.rho, self.tau_p = lam_fb, rho, tau_p
        self.s_eq, self.h_ext, self.v0 = s_eq, h_ext, v0
        self.fixed_graph, self.metric, self.r_max = fixed_graph, metric, r_max
        self._graph = None

    def _neighbors(self, state):
        if self.metric:
            return metric_neighbors(state.positions, self.boundary, self.r_max)
        if self.fixed_graph:
            if self._graph is None:
                self._graph = topological_neighbors(state.positions,
                                                    self.boundary, self.k)
            return self._graph
        return topological_neighbors(state.positions, self.boundary, self.k)

    def _init_internal(self, state, neigh):
        n = state.n
        # Bloch vectors per channel, stored as a padded (N, k, 3) array.
        kmax = max((len(c) for c in neigh), default=1) or 1
        m = np.zeros((n, kmax, 3))
        m[..., 2] = 0.0     # start unbiased
        m[..., 0] = 0.0
        state.internal["m"] = m
        state.internal["s"] = np.zeros(n)
        return state

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        neigh = self._neighbors(s)
        if "m" not in s.internal:
            s = self._init_internal(s, neigh)

        m, sl = s.internal["m"], s.internal["s"]
        h = s.headings
        n, kmax = m.shape[0], m.shape[1]

        mz_mean = np.zeros(n)
        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            nk = min(len(cand), kmax)
            # ---- fast register: effective longitudinal field -----------------
            align = h[cand[:nk]] @ h[i]                      # cos of heading angle
            h_eff = self.kappa * align + self.beta * sl[i] + self.h_ext

            mi = m[i, :nk, :]
            Bx, Bz = -self.Gamma, -h_eff
            # unitary part: m_dot = 2 B x m
            mdot = np.empty_like(mi)
            mdot[:, 0] = 2.0 * (0.0 * mi[:, 2] - Bz * mi[:, 1])
            mdot[:, 1] = 2.0 * (Bz * mi[:, 0] - Bx * mi[:, 2])
            mdot[:, 2] = 2.0 * (Bx * mi[:, 1] - 0.0 * mi[:, 0])
            # GKSL dissipation
            mdot[:, 0] -= mi[:, 0] / self.T2
            mdot[:, 1] -= mi[:, 1] / self.T2
            mdot[:, 2] -= (mi[:, 2] - np.tanh(h_eff)) / self.T1

            m[i, :nk, :] = np.clip(mi + mdot * dt, -1.0, 1.0)
            mz_mean[i] = m[i, :nk, 2].mean()

        # ---- slow register ---------------------------------------------------
        target = np.tanh(self.lam_fb * mz_mean / self.tau_p)
        sl += self.gamma_s * (-sl + target + self.s_eq) * dt
        sl = np.clip(sl, -1.0, 1.0)

        # ---- heading map (Eq. 25-26) ----------------------------------------
        u = self.rho * h.copy()
        for i, cand in enumerate(neigh):
            if len(cand) == 0:
                continue
            nk = min(len(cand), kmax)
            w = 1.0 / nk
            weight = (1.0 + m[i, :nk, 2]) / 2.0
            u[i] += w * (weight[:, None] * h[cand[:nk]]).sum(axis=0)

        s.internal["m"], s.internal["s"] = m, sl
        s.velocities = self.v0 * _normalize(u)
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)


class KuramotoModel(CollectiveModel):
    """Kuramoto phase oscillators on the interaction graph (2D).

        theta_i(t+dt) = theta_i + [ omega_i
                                    + (K/|N_i|) sum_{j in N_i} sin(theta_j - theta_i)
                                  ] dt + eta * xi_i

    Each agent's heading angle theta_i IS its Kuramoto phase; it self-propels at
    fixed speed v0 along that heading. omega_i is an intrinsic turning rate drawn
    once from N(0, omega_spread) and stored in internal["omega"]. The Kuramoto
    order parameter r = |<e^{i theta}>| is exactly `metrics.polar_order` here, so
    the phase-synchronisation transition and the flocking order-disorder
    transition are the SAME transition measured two ways.

    NO ATTRACTION: coupling is through phase only. Like Vicsek this is alignment-
    only, so under OpenBoundary the neighbour lists empty, the sine coupling
    switches off, and the group desynchronises as it disperses -- the boundary
    thesis of this repo applies to Kuramoto exactly as it does to Vicsek.
    """

    name = "Kuramoto"
    cohesive = False
    two_d_only = True

    def __init__(self, boundary, r_max=1.0, K=1.0, eta=0.0, v0=0.5, r_min=0.0,
                 omega_spread=0.0, topological=False, k=7, rng=None):
        super().__init__(boundary, rng)
        self.r_max, self.r_min, self.K, self.eta, self.v0 = r_max, r_min, K, eta, v0
        self.omega_spread = omega_spread
        self.topological, self.k = topological, k

    def _neighbors(self, state):
        if self.topological:
            return topological_neighbors(state.positions, self.boundary, self.k)
        return metric_neighbors(state.positions, self.boundary, self.r_max, self.r_min)

    def step(self, state: State, dt: float) -> State:
        if state.dim != 2:
            raise ValueError("KuramotoModel is a 2D phase model (state.dim must be 2).")
        s = state.copy()
        if "omega" not in s.internal:
            s.internal["omega"] = self.omega_spread * self.rng.standard_normal(s.n)
        omega = s.internal["omega"]

        v = s.velocities
        theta = np.arctan2(v[:, 1], v[:, 0])
        neigh = self._neighbors(s)

        dtheta = omega.copy()
        for i, cand in enumerate(neigh):
            if len(cand):
                dtheta[i] += (self.K / len(cand)) * np.sin(theta[cand] - theta[i]).sum()
        theta = theta + dtheta * dt + self.eta * self.rng.standard_normal(s.n)

        s.velocities = self.v0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        s.positions = s.positions + s.velocities * dt
        s.t += dt
        return self.finalize(s)
