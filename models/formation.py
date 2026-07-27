"""
FORMATION CONTROL MODELS. All are cohesive = True and all work in free space.

Formation control is where the open-boundary question is already SOLVED, because
control theorists never had the luxury of a periodic box -- a real quadrotor swarm
flies in R^3. The convergence proofs here are unconditional and constructive.

Included:
  * DisplacementFormation -- consensus on relative positions: q_i - q_j -> c_i - c_j.
                             Requires a common orientation reference. Linear;
                             converges exponentially at rate lambda_2(L) (Fiedler).
  * DistanceFormation     -- gradient descent on a rigid-graph distance potential:
                             |q_i - q_j| -> d_ij. No shared reference frame needed.
                             Needs INFINITESIMAL RIGIDITY of the framework, and only
                             converges locally (there are spurious equilibria --
                             flip and flex ambiguities).
  * LeaderFollower        -- a designated leader drives; followers hold offsets.
                             Convergence requires a spanning tree rooted at the leader.
  * CyclicPursuit         -- agent i pursues agent i+1 (mod n). Bruckstein; Marshall,
                             Broucke & Francis (IEEE TAC 2004). With a bearing offset
                             alpha it produces polygonal / spiral / logarithmic
                             equilibria. This is Twinkle Tripathy's home territory --
                             she has a 2024 Automatica paper on nonlinear cyclic
                             pursuit convergence.
"""

from __future__ import annotations

import numpy as np

from core.base import CollectiveModel, State


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > eps, n, 1.0)


def laplacian(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian L = D - A (out-degree convention)."""
    return np.diag(A.sum(axis=1)) - A


def algebraic_connectivity(A: np.ndarray) -> float:
    """Fiedler value lambda_2 of the symmetrised Laplacian.

    > 0 iff the graph is connected. Sets the exponential convergence rate of
    displacement-based formation and of Laplacian consensus. This is the single
    number that decides whether a formation law works at all.
    """
    As = np.maximum(A, A.T)
    L = laplacian(As)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (L + L.T)))
    return float(ev[1]) if len(ev) > 1 else 0.0


def is_infinitesimally_rigid(positions: np.ndarray, A: np.ndarray) -> bool:
    """Check infinitesimal rigidity in 2D via the rigidity matrix rank.

    A framework in R^d with n >= d+1 vertices is infinitesimally rigid iff
    rank(R) = d*n - d(d+1)/2. For 2D: rank = 2n - 3.

    THIS MATTERS: distance-based formation control converges ONLY if the framework
    is rigid. A non-rigid framework has continuous flexes and the formation will
    deform forever without ever converging.
    """
    x = np.asarray(positions, dtype=float)
    n, d = x.shape
    if n < d + 1:
        return False
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if A[i, j] or A[j, i]]
    if not edges:
        return False
    R = np.zeros((len(edges), d * n))
    for e, (i, j) in enumerate(edges):
        diff = x[i] - x[j]
        R[e, d * i:d * i + d] = diff
        R[e, d * j:d * j + d] = -diff
    target = d * n - d * (d + 1) // 2
    return bool(np.linalg.matrix_rank(R, tol=1e-8) == target)


class DisplacementFormation(CollectiveModel):
    """Displacement-based formation. Linear, globally convergent.

        u_i = -k * sum_{j in N_i} a_ij * [ (q_i - q_j) - (c_i - c_j) ]

    where c is the desired shape. Converges exponentially to the target shape (up
    to a global translation) iff the graph is connected, i.e. lambda_2 > 0.
    Requires all agents to share a common orientation reference.
    """

    name = "Displacement formation"
    cohesive = True

    def __init__(self, boundary, A: np.ndarray, target: np.ndarray,
                 k: float = 1.0, damping: float = 0.0, rng=None):
        super().__init__(boundary, rng)
        self.A = np.asarray(A, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.k, self.damping = k, damping

    @property
    def convergence_rate(self) -> float:
        return algebraic_connectivity(self.A)

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        u = np.zeros_like(s.positions)
        for i in range(s.n):
            nb = np.flatnonzero(self.A[i])
            if len(nb) == 0:
                continue
            rel = -b.displacement(s.positions[i], s.positions[nb])   # q_i - q_j
            des = self.target[i] - self.target[nb]
            u[i] = -self.k * (self.A[i, nb][:, None] * (rel - des)).sum(axis=0)
        u -= self.damping * s.velocities
        s.velocities = u
        s.positions = s.positions + u * dt
        s.t += dt
        return self.finalize(s)


class DistanceFormation(CollectiveModel):
    """Distance-based (rigid) formation. Gradient flow, LOCALLY convergent only.

        u_i = -k * sum_j ( |q_i-q_j|^2 - d_ij^2 ) * (q_i - q_j)

    No common reference frame needed -- each agent only measures ranges. But the
    potential is non-convex: there are spurious equilibria (reflections/flips), so
    convergence is only local, and the framework MUST be infinitesimally rigid.
    Use `is_infinitesimally_rigid` to check before you trust the result.
    """

    name = "Distance formation (rigid)"
    cohesive = True

    def __init__(self, boundary, A: np.ndarray, D: np.ndarray,
                 k: float = 0.5, rng=None):
        super().__init__(boundary, rng)
        self.A = np.asarray(A, dtype=float)
        self.D = np.asarray(D, dtype=float)   # desired pairwise distances
        self.k = k

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        u = np.zeros_like(s.positions)
        for i in range(s.n):
            nb = np.flatnonzero(self.A[i])
            if len(nb) == 0:
                continue
            rel = -b.displacement(s.positions[i], s.positions[nb])   # q_i - q_j
            dist2 = (rel ** 2).sum(axis=1)
            err = dist2 - self.D[i, nb] ** 2
            u[i] = -self.k * (err[:, None] * rel).sum(axis=0)
        s.velocities = u
        s.positions = s.positions + u * dt
        s.t += dt
        return self.finalize(s)


class LeaderFollower(CollectiveModel):
    """Leader-follower formation.

    Leaders follow a prescribed trajectory; followers run displacement consensus
    on their offsets. Converges iff the graph contains a spanning tree rooted at
    the leader set -- the standard directed-graph condition.
    """

    name = "Leader-follower formation"
    cohesive = True

    def __init__(self, boundary, A: np.ndarray, target: np.ndarray,
                 leaders, leader_vel=None, k: float = 1.5, rng=None):
        super().__init__(boundary, rng)
        self.A = np.asarray(A, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.leaders = np.asarray(leaders, dtype=int)
        self.leader_vel = (np.asarray(leader_vel, dtype=float)
                           if leader_vel is not None else np.array([0.3, 0.0]))
        self.k = k

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        u = np.zeros_like(s.positions)
        for i in range(s.n):
            if i in self.leaders:
                u[i] = self.leader_vel
                continue
            nb = np.flatnonzero(self.A[i])
            if len(nb) == 0:
                continue
            rel = -b.displacement(s.positions[i], s.positions[nb])
            des = self.target[i] - self.target[nb]
            u[i] = -self.k * (self.A[i, nb][:, None] * (rel - des)).sum(axis=0)
        s.velocities = u
        s.positions = s.positions + u * dt
        s.t += dt
        return self.finalize(s)


class CyclicPursuit(CollectiveModel):
    """Cyclic pursuit: agent i pursues agent i+1 (mod n).

    Marshall, Broucke & Francis, IEEE TAC 49, 1963 (2004); Bruckstein et al.
    Twinkle Tripathy & Tal Shima, Automatica 159, 111315 (2024) treat the
    nonlinear case.

        u_i = k * R(alpha) * ( q_{i+1} - q_i )

    where R(alpha) rotates the pursuit bearing by a fixed offset alpha.
      alpha = 0            -> rendezvous at the centroid
      alpha in (0, pi/2)   -> logarithmic spiral, converging
      alpha = pi/n         -> a stable regular n-gon (the classic result)
      alpha > pi/2         -> diverging spiral

    Purely local, purely relative, and works in unbounded space with no global
    information at all. This is the cleanest illustration that "open boundary"
    is a solved problem in control theory.
    """

    name = "Cyclic pursuit"
    cohesive = True

    def __init__(self, boundary, alpha: float = 0.0, k: float = 1.0,
                 speed_limit=None, rng=None):
        super().__init__(boundary, rng)
        self.alpha, self.k, self.speed_limit = alpha, k, speed_limit

    @staticmethod
    def polygon_alpha(n: int) -> float:
        """Bearing offset that yields a stable regular n-gon."""
        return np.pi / n

    def step(self, state: State, dt: float) -> State:
        s = state.copy()
        b = self.boundary
        nxt = np.roll(np.arange(s.n), -1)
        d = b.displacement(s.positions, s.positions[nxt])   # i -> i+1
        c, sn = np.cos(self.alpha), np.sin(self.alpha)
        R = np.array([[c, -sn], [sn, c]])
        u = self.k * (d @ R.T)
        if self.speed_limit is not None:
            nrm = np.linalg.norm(u, axis=1, keepdims=True)
            u = u * np.minimum(1.0, self.speed_limit / np.where(nrm > 1e-12, nrm, 1.0))
        s.velocities = u
        s.positions = s.positions + u * dt
        s.t += dt
        return self.finalize(s)


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def regular_polygon(n: int, radius: float = 3.0, center=(0.0, 0.0)) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + radius * np.cos(th),
                     center[1] + radius * np.sin(th)], axis=1)


def grid_shape(n: int, spacing: float = 1.0) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n)))
    pts = [(i * spacing, j * spacing) for i in range(side) for j in range(side)]
    return np.asarray(pts[:n], dtype=float)


def v_shape(n: int, spacing: float = 1.0, angle: float = np.pi / 4) -> np.ndarray:
    """Echelon / V formation, as in migrating geese."""
    pts = [(0.0, 0.0)]
    for i in range(1, n):
        arm = (i + 1) // 2
        side = 1 if i % 2 else -1
        pts.append((-arm * spacing * np.cos(angle),
                    side * arm * spacing * np.sin(angle)))
    return np.asarray(pts[:n], dtype=float)


def complete_graph(n: int) -> np.ndarray:
    return np.ones((n, n)) - np.eye(n)


def cycle_graph(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def distance_matrix_from_shape(shape: np.ndarray) -> np.ndarray:
    d = shape[:, None, :] - shape[None, :, :]
    return np.linalg.norm(d, axis=2)
