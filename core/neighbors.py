"""
Neighbour selection rules.

Three families, all boundary-aware:

  * metric        : all agents within [r_min, r_max]. Density-dependent -> this is
                    the rule that DIES in open space, because as the group spreads
                    the neighbour count collapses to zero.
  * topological   : the k nearest agents, regardless of distance. Density-
                    independent (Ballerini et al. found k ~ 7 for starlings), so it
                    survives dispersal in the sense that the graph stays connected
                    -- but note it does NOT prevent the group from spreading, it
                    only keeps the interaction switched on while it spreads.
  * vision cone   : metric + angular gate, as in Beuria PAPER_1 (angular width
                    alpha, radial range [r_min, r_max]).

The returned structure is a list of integer arrays (ragged), plus helpers to build
dense weight matrices for the consensus models.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .boundary import Boundary, PeriodicBoundary


def _build_tree(positions: np.ndarray, boundary: Boundary) -> cKDTree:
    """cKDTree that respects periodicity when the boundary is a torus."""
    if isinstance(boundary, PeriodicBoundary):
        # cKDTree requires coordinates inside [0, L) for boxsize to be valid.
        pos = np.mod(positions, boundary.L)
        return cKDTree(pos, boxsize=boundary.L)
    return cKDTree(np.asarray(positions))


def metric_neighbors(
    positions: np.ndarray,
    boundary: Boundary,
    r_max: float,
    r_min: float = 0.0,
):
    """Agents within an annulus [r_min, r_max]. Excludes self.

    r_min > 0 gives the 'non-local' model referenced in Beuria PAPER_1, where very
    close agents do not explicitly influence the focal agent.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    tree = _build_tree(positions, boundary)
    raw = tree.query_ball_point(
        np.mod(positions, boundary.L) if isinstance(boundary, PeriodicBoundary) else positions,
        r=r_max,
    )

    out = []
    for i in range(n):
        cand = np.asarray([j for j in raw[i] if j != i], dtype=int)
        if len(cand) == 0:
            out.append(cand)
            continue
        if r_min > 0.0:
            d = boundary.distance(positions[i], positions[cand])
            cand = cand[d >= r_min]
        out.append(cand)
    return out


def topological_neighbors(positions: np.ndarray, boundary: Boundary, k: int):
    """The k nearest agents. Density-independent (Ballerini et al. 2008)."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return [np.array([], dtype=int) for _ in range(n)]
    tree = _build_tree(positions, boundary)
    query_pos = (
        np.mod(positions, boundary.L)
        if isinstance(boundary, PeriodicBoundary)
        else positions
    )
    _, idx = tree.query(query_pos, k=k_eff + 1)  # +1 because self is included
    idx = np.atleast_2d(idx)
    return [np.asarray([j for j in idx[i] if j != i][:k_eff], dtype=int) for i in range(n)]


def vision_cone_neighbors(
    positions: np.ndarray,
    headings: np.ndarray,
    boundary: Boundary,
    alpha: float,
    r_max: float,
    r_min: float = 0.0,
):
    """Metric annulus gated by an angular half-width alpha/2 about the heading.

    This reproduces the perception geometry of Beuria PAPER_1 Figure 1: an agent
    A_i perceives neighbours inside a cone of angular width alpha centred on its
    direction of motion, restricted to radii in [r_min, r_max].

    Parameters
    ----------
    headings : (N, dim) unit vectors.
    alpha    : FULL angular width of the cone in radians (pi/2 in PAPER_1).
    """
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    base = metric_neighbors(positions, boundary, r_max, r_min)
    half = 0.5 * alpha
    cos_half = np.cos(half)

    out = []
    for i, cand in enumerate(base):
        if len(cand) == 0:
            out.append(cand)
            continue
        d = boundary.displacement(positions[i], positions[cand])
        norms = np.linalg.norm(d, axis=-1)
        good = norms > 1e-12
        cand, d, norms = cand[good], d[good], norms[good]
        if len(cand) == 0:
            out.append(np.array([], dtype=int))
            continue
        unit = d / norms[:, None]
        h = headings[i]
        h = h / (np.linalg.norm(h) + 1e-12)
        cosang = unit @ h
        out.append(cand[cosang >= cos_half])
    return out


def subsample_neighbors(neigh, n_max: int, rng: np.random.Generator):
    """Randomly keep at most n_max neighbours per agent.

    Beuria PAPER_1 builds a 2^n dimensional perceptual Hilbert space for n
    neighbours, so n must stay tiny (n = 2 or 3 in that paper). This helper
    enforces that ceiling by random selection, matching the paper's scheme of
    'n randomly selected neighbours'.
    """
    out = []
    for cand in neigh:
        if len(cand) > n_max:
            out.append(rng.choice(cand, size=n_max, replace=False))
        else:
            out.append(cand)
    return out


def neighbors_to_weight_matrix(neigh, n: int, mode: str = "uniform") -> np.ndarray:
    """Row-stochastic weight matrix W from a ragged neighbour list.

    mode='uniform' gives w_ij = 1/|N_i|, the convention used in Beuria PAPER_2
    (Eq. 24) and in the classical DeGroot / Vicsek averaging rule.
    """
    W = np.zeros((n, n))
    for i, cand in enumerate(neigh):
        if len(cand) == 0:
            continue
        if mode == "uniform":
            W[i, cand] = 1.0 / len(cand)
        else:
            raise ValueError(f"Unknown mode {mode!r}")
    return W


def adjacency_from_neighbors(neigh, n: int, symmetrize: bool = False) -> np.ndarray:
    """Binary adjacency matrix. Topological rules give asymmetric A by default."""
    A = np.zeros((n, n), dtype=int)
    for i, cand in enumerate(neigh):
        A[i, cand] = 1
    if symmetrize:
        A = np.maximum(A, A.T)
    return A
