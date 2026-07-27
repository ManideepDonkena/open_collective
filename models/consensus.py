"""
GROUP CONSENSUS / OPINION DYNAMICS MODELS.

These live in an abstract opinion space, not physical space, so boundary conditions
are irrelevant to them -- which is exactly why this family is worth having next to
the flocking models. It isolates the CONSENSUS question from the COHESION question.

Included:
  * DeGroot           -- DeGroot, JASA 69, 118 (1974). x(t+1) = W x(t).
                         Converges to consensus iff W is row-stochastic, primitive
                         (irreducible + aperiodic). Vicsek IS this rule applied to
                         headings, which is the deep link between the two families.
  * FriedkinJohnsen   -- Friedkin & Johnsen (1990/1999).
                         x(t+1) = Lambda W x(t) + (I - Lambda) u
                         Stubbornness Lambda anchors agents to their prejudice u.
                         Explains PERSISTENT DISAGREEMENT: the model's whole point.
  * SignedFJ          -- Shrinate & Tripathy (2025), 'A Signed Friedkin-Johnsen
                         Model for Arbitrary Network Topologies', under the
                         opposing rule. Negative edges = antagonism.
  * AltafiniBipartite -- Altafini, IEEE TAC 58, 935 (2013). Signed Laplacian on a
                         STRUCTURALLY BALANCED graph -> bipartite consensus: two
                         camps at +c and -c.
  * GroupConsensus    -- multiconsensus / cluster consensus: agents partitioned into
                         subgroups that agree internally and disagree across.
                         This is the echo-chamber setting of Shrinate, Tripathy &
                         Behera (2026).

WHY THIS MATTERS FOR THE FLOCKING SIDE
--------------------------------------
Beuria PAPER_1 finds numerically that decision states with NEGATIVE coefficients
(Phi-, Psi-) destroy flock cohesion -- neighbour momenta enter destructively
(his Eq. 32). That is *exactly* a signed network. Structural balance theory gives
the rigorous necessary-and-sufficient condition for when negative interactions
produce bipartite consensus instead of disorder. `structural_balance` below is that
test. It is the bridge nobody has built yet.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------------------
# Graph theory: the tools Tripathy's group uses
# --------------------------------------------------------------------------

def structural_balance(A: np.ndarray):
    """Test structural balance of a signed graph. Returns (is_balanced, partition).

    A signed graph is STRUCTURALLY BALANCED iff its vertices can be split into two
    sets V1, V2 such that every edge inside a set is positive and every edge across
    is negative. Equivalently: every cycle has an even number of negative edges.
    Equivalently: there is a gauge transformation D = diag(+-1) with D A D >= 0.

    ALTAFINI'S THEOREM: on a connected, structurally balanced signed graph, the
    signed Laplacian flow converges to BIPARTITE CONSENSUS -- two camps of equal
    magnitude and opposite sign. If UNBALANCED, it converges to zero (trivial
    consensus / total disagreement collapse).

    THIS IS THE RIGOROUS VERSION OF BEURIA'S Phi- OBSERVATION.
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    As = np.where(np.abs(A) + np.abs(A.T) > 0, 1, 0)
    n_comp, labels = connected_components(csr_matrix(As), directed=False)

    sign = np.zeros(n, dtype=int)
    for comp in range(n_comp):
        idx = np.flatnonzero(labels == comp)
        root = idx[0]
        sign[root] = 1
        stack = [root]
        seen = {root}
        while stack:
            i = stack.pop()
            for j in range(n):
                if i == j:
                    continue
                w = A[i, j] if A[i, j] != 0 else A[j, i]
                if w == 0:
                    continue
                proposed = sign[i] * (1 if w > 0 else -1)
                if j not in seen:
                    sign[j] = proposed
                    seen.add(j)
                    stack.append(j)
                elif sign[j] != proposed:
                    return False, None
    return True, sign


def gauge_transform(A: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """D A D with D = diag(sign). Makes a balanced signed graph non-negative."""
    D = np.diag(sign)
    return D @ A @ D


def signed_laplacian(A: np.ndarray) -> np.ndarray:
    """Altafini's signed Laplacian: L = D_|A| - A, with D_|A| = diag(sum_j |a_ij|)."""
    A = np.asarray(A, dtype=float)
    return np.diag(np.abs(A).sum(axis=1)) - A


def is_primitive(W: np.ndarray, tol: float = 1e-12) -> bool:
    """Primitive (irreducible + aperiodic) => DeGroot converges to consensus."""
    n = len(W)
    M = np.linalg.matrix_power((np.abs(W) > tol).astype(float), (n - 1) ** 2 + 1)
    return bool(np.all(M > 0))


def row_stochastic(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    rs = A.sum(axis=1, keepdims=True)
    return np.divide(A, np.where(rs > 1e-12, rs, 1.0))


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class DeGroot:
    """x(t+1) = W x(t). Plain averaging. Converges to consensus iff W primitive.

    The consensus value is pi^T x(0), where pi is the left Perron eigenvector of W
    -- the 'social power' distribution.

    NOTE: Vicsek's heading update IS this rule, with W rebuilt from positions at
    every step. That is the entire relationship between opinion dynamics and
    flocking: same operator, but the flocking version has a STATE-DEPENDENT,
    TIME-VARYING W. Everything hard about flocking comes from that dependence.
    """

    name = "DeGroot"

    def __init__(self, W: np.ndarray):
        self.W = row_stochastic(W)

    @property
    def converges(self) -> bool:
        return is_primitive(self.W)

    def social_power(self) -> np.ndarray:
        ev, evec = np.linalg.eig(self.W.T)
        i = np.argmin(np.abs(ev - 1.0))
        pi = np.real(evec[:, i])
        return pi / pi.sum()

    def run(self, x0: np.ndarray, steps: int = 200):
        x = np.asarray(x0, dtype=float).copy()
        hist = [x.copy()]
        for _ in range(steps):
            x = self.W @ x
            hist.append(x.copy())
        return x, np.asarray(hist)


class FriedkinJohnsen:
    """x(t+1) = Lambda W x(t) + (I - Lambda) u.

    Lambda = diag(lambda_i) is the susceptibility; (1 - lambda_i) = beta_i is the
    STUBBORNNESS of agent i, anchoring it to its prejudice u_i.

    Equilibrium (when it exists): x* = (I - Lambda W)^{-1} (I - Lambda) u.

    KEY POINT: the FJ model is built to explain why disagreement PERSISTS even
    among closely interacting people. Contrast with DeGroot, which always agrees.

    RELATION TO BEURIA PAPER_2: the slow regulatory variable s_i is a DYNAMIC,
    self-generated version of beta_i. PAPER_2 is, structurally, an FJ model whose
    stubbornness evolves in time and integrates the agent's own alignment history.
    Nobody has written that down. That is the paper worth writing.
    """

    name = "Friedkin-Johnsen"

    def __init__(self, W: np.ndarray, lam: np.ndarray, u: np.ndarray):
        self.W = row_stochastic(W)
        self.lam = np.asarray(lam, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.Lam = np.diag(self.lam)

    def equilibrium(self):
        n = len(self.W)
        M = np.eye(n) - self.Lam @ self.W
        try:
            return np.linalg.solve(M, (np.eye(n) - self.Lam) @ self.u)
        except np.linalg.LinAlgError:
            return None

    def run(self, x0=None, steps: int = 300):
        x = np.asarray(x0 if x0 is not None else self.u, dtype=float).copy()
        hist = [x.copy()]
        for _ in range(steps):
            x = self.Lam @ self.W @ x + (np.eye(len(x)) - self.Lam) @ self.u
            hist.append(x.copy())
        return x, np.asarray(hist)


class SignedFJ:
    """Signed Friedkin-Johnsen under the OPPOSING RULE.

    Shrinate & Tripathy, 'A Signed Friedkin-Johnsen Model for Arbitrary Network
    Topologies' (arXiv 2509.11038). Under the opposing rule an agent connected by a
    negative edge moves toward the NEGATIVE of its neighbour's opinion:

        x_i(t+1) = lambda_i * sum_j |w_ij| * sgn(w_ij) * x_j(t) + (1-lambda_i) u_i

    Their finding: adding signed interactions produces behaviours impossible in the
    cooperative FJ model -- opinion leaders (sinks of the condensation graph) can
    become NON-INFLUENTIAL, and non-influential agents can become influential.
    """

    name = "Signed Friedkin-Johnsen"

    def __init__(self, A: np.ndarray, lam: np.ndarray, u: np.ndarray):
        A = np.asarray(A, dtype=float)
        rs = np.abs(A).sum(axis=1, keepdims=True)
        self.W = np.divide(A, np.where(rs > 1e-12, rs, 1.0))  # signed, |rows| = 1
        self.lam = np.asarray(lam, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.Lam = np.diag(self.lam)
        self.A = A

    @property
    def balanced(self):
        return structural_balance(self.A)[0]

    def run(self, x0=None, steps: int = 400):
        x = np.asarray(x0 if x0 is not None else self.u, dtype=float).copy()
        hist = [x.copy()]
        for _ in range(steps):
            x = self.Lam @ self.W @ x + (np.eye(len(x)) - self.Lam) @ self.u
            hist.append(x.copy())
        return x, np.asarray(hist)


class AltafiniBipartite:
    """Signed Laplacian flow: x_dot = -L_signed x.  Altafini, IEEE TAC 58, 935 (2013).

    THEOREM:
      * structurally balanced + connected -> BIPARTITE CONSENSUS. Opinions split
        into two camps at +c and -c. Order survives, but as two opposed groups.
      * unbalanced + connected            -> x -> 0. Everything collapses to zero;
        no order at all.

    This is the exact rigorous analogue of Beuria PAPER_1's Phi- result. He found
    numerically that a negative coefficient kills flock cohesion; Altafini's theorem
    tells you *when* it kills it (unbalanced) and when it instead produces two
    counter-propagating sub-flocks (balanced). PAPER_1 only ever tests the
    all-negative case, which is trivially balanced with a single camp -- so the
    interesting regime was never explored.
    """

    name = "Altafini bipartite consensus"

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.L = signed_laplacian(self.A)

    @property
    def balanced(self):
        return structural_balance(self.A)

    def run(self, x0: np.ndarray, steps: int = 2000, dt: float = 0.01):
        x = np.asarray(x0, dtype=float).copy()
        hist = [x.copy()]
        for _ in range(steps):
            x = x - dt * (self.L @ x)
            hist.append(x.copy())
        return x, np.asarray(hist)


class GroupConsensus:
    """Multiconsensus / cluster consensus: k subgroups, agree within, disagree across.

    Shrinate, Siddharth & Tripathy, 'Topology-based Conditions for Multiconsensus
    under the Signed Friedkin-Johnsen Model' (2025); and Shrinate, Tripathy &
    Behera, 'Topological Conditions for Echo Chamber Formation under the FJ model'
    (2026).

    Implemented here as an in-group/out-group weighting: cooperative inside a group,
    antagonistic (or absent) across groups. The resulting steady state is the
    echo-chamber configuration.
    """

    name = "Group (cluster) consensus"

    def __init__(self, groups, w_in: float = 1.0, w_out: float = -0.4,
                 lam: float = 0.9, u=None, rng=None):
        self.groups = np.asarray(groups, dtype=int)
        n = len(self.groups)
        rng = rng if rng is not None else np.random.default_rng(0)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                A[i, j] = w_in if self.groups[i] == self.groups[j] else w_out
        self.A = A
        self.lam = np.full(n, lam)
        self.u = np.asarray(u) if u is not None else rng.uniform(-1, 1, size=n)
        self.model = SignedFJ(A, self.lam, self.u)

    @property
    def balanced(self):
        return structural_balance(self.A)[0]

    def run(self, x0=None, steps: int = 400):
        return self.model.run(x0, steps)


def condensation_leaders(A: np.ndarray):
    """Opinion leaders = sinks of the condensation graph (Shrinate & Tripathy).

    Returns (labels, leader_components, leader_agents). An agent is an opinion
    LEADER if its strongly connected component has no outgoing edges to another
    component; everyone else is a FOLLOWER. In the cooperative FJ model leaders
    drive the final opinions -- but Shrinate & Tripathy show that adding signed
    edges can strip a leader of its influence entirely.
    """
    A = np.asarray(A, dtype=float)
    B = (np.abs(A) > 0).astype(int)
    n_comp, labels = connected_components(csr_matrix(B), directed=True,
                                          connection="strong")
    leaders = []
    for c in range(n_comp):
        idx = np.flatnonzero(labels == c)
        out = False
        for i in idx:
            for j in np.flatnonzero(B[i]):
                if labels[j] != c:
                    out = True
                    break
            if out:
                break
        if not out:
            leaders.append(c)
    leader_agents = np.flatnonzero(np.isin(labels, leaders))
    return labels, leaders, leader_agents
