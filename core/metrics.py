"""
Observables and diagnostics.

Two classes of observable:

ORDER observables (what the literature reports)
  * polar_order          : |<p_hat>|, the Vicsek order parameter. Beuria PAPER_2
                           Eq. (30). Reported by essentially every flocking paper.
  * milling_order        : normalised angular momentum about the centroid; detects
                           the rotating-mill states of D'Orsogna et al.

COHESION observables (what the literature usually omits, and what periodic BC hides)
  * radius_of_gyration   : RMS spread about the centroid. Under periodic BC this is
                           bounded above by ~L/2 BY CONSTRUCTION. In open space it
                           is unbounded and grows ~ballistically for an alignment-
                           only model. This single number is the whole argument.
  * nn_distance          : mean nearest-neighbour distance; grows as the group
                           dilutes.
  * n_fragments          : number of connected components of the r_max interaction
                           graph. 1 = one flock. >1 = the group has shattered and
                           the global order parameter is meaningless.
  * largest_cluster_frac : fraction of agents in the biggest component. The single
                           most honest scalar for "is this still one group?".
  * cohesion_verdict     : a convenience classifier.

GROUP-MAINTENANCE observables (for the multi-group models in models/grouping.py)
  * segregation_index    : do the groups stay DISTINCT, or fuse into one blob?
  * group_integrity      : is each group internally connected at range r_link?
  * group_expansion      : per-group R_g -- the anti-fission diagnostic that matches
                           what Cucker-Smale actually promises (bounded relative
                           positions, NOT connectivity at any particular range).
  * group_separation     : are the groups in different places?
  * per_group_polar_order: M within each group -- the global M is meaningless once
                           more than one group is present.

A grouping model fails in two independent directions, fission and fusion, and the
cohesion observables above can only see the first. A fully fused ensemble scores a
PERFECT cohesion score while having destroyed every group in it. See the
GROUP-MAINTENANCE section below.

THE KEY POINT
-------------
A periodic box makes radius_of_gyration and n_fragments UNINFORMATIVE, because the
torus caps the spread and re-supplies neighbours through the walls. Reporting only
polar_order in a periodic box therefore cannot distinguish a genuinely cohesive
model from one that would evaporate the moment you removed the walls.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .boundary import Boundary, PeriodicBoundary
from .neighbors import (adjacency_from_neighbors, metric_neighbors,
                        topological_neighbors)


# --------------------------------------------------------------------------
# Order observables
# --------------------------------------------------------------------------

def polar_order(headings: np.ndarray) -> float:
    """M = |(1/N) sum_i p_hat_i|. Beuria PAPER_2 Eq. (30). In [0, 1]."""
    h = np.asarray(headings, dtype=float)
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h = h / np.where(norms > 1e-12, norms, 1.0)
    return float(np.linalg.norm(h.mean(axis=0)))


def milling_order(positions: np.ndarray, velocities: np.ndarray,
                  boundary: Boundary) -> float:
    """Normalised angular momentum about the centroid. In [0, 1].

    ~1 for a coherent rotating mill, ~0 for a translating flock. Only meaningful
    under open/reflecting BC -- a torus has no well-defined global centroid.

    In 2D the angular momentum is the scalar cross product; in 3D it is the norm of
    the summed cross product, so a mill is detected whatever plane it rotates in.
    """
    x = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    if isinstance(boundary, PeriodicBoundary):
        return float("nan")  # centroid is ill-defined on a torus
    c = x.mean(axis=0)
    r = x - c
    rn = np.linalg.norm(r, axis=1)
    vn = np.linalg.norm(v, axis=1)
    denom = (rn * vn).sum()
    if denom < 1e-12:
        return 0.0
    dim = x.shape[1]
    if dim == 2:
        cross = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]  # 2D scalar cross product
        return float(abs(cross.sum()) / denom)
    if dim == 3:
        return float(np.linalg.norm(np.cross(r, v).sum(axis=0)) / denom)
    return float("nan")  # angular momentum is a bivector for d > 3


# --------------------------------------------------------------------------
# Cohesion observables -- the diagnostics periodic BC suppresses
# --------------------------------------------------------------------------

def centroid(positions: np.ndarray, boundary: Boundary) -> np.ndarray:
    """Centroid. On a torus this uses the circular-mean trick."""
    x = np.asarray(positions, dtype=float)
    if isinstance(boundary, PeriodicBoundary):
        theta = 2.0 * np.pi * np.mod(x, boundary.L) / boundary.L
        xi = np.cos(theta).mean(axis=0)
        zeta = np.sin(theta).mean(axis=0)
        theta_bar = np.arctan2(-zeta, -xi) + np.pi
        return boundary.L * theta_bar / (2.0 * np.pi)
    return x.mean(axis=0)


def radius_of_gyration(positions: np.ndarray, boundary: Boundary) -> float:
    """R_g = sqrt(<|x_i - centroid|^2>).

    THE headline diagnostic. Under PeriodicBoundary this saturates near L/sqrt(6)
    for a uniform gas and can never exceed ~L/2 -- the torus caps it. Under
    OpenBoundary it is unbounded, and for an alignment-only model it grows roughly
    linearly in time as the group evaporates.
    """
    x = np.asarray(positions, dtype=float)
    c = centroid(x, boundary)
    d = boundary.displacement(c[None, :], x)
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def nn_distance(positions: np.ndarray, boundary: Boundary) -> float:
    """Mean nearest-neighbour distance. Grows as the group dilutes."""
    x = np.asarray(positions, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    dists = np.empty(n)
    for i in range(n):
        d = boundary.distance(x[i], x)
        d[i] = np.inf
        dists[i] = d.min()
    return float(dists.mean())


def fragmentation(positions: np.ndarray, boundary: Boundary, r_link: float):
    """Connected components of the interaction graph at range r_link.

    Returns (n_fragments, largest_cluster_fraction).

    n_fragments == 1 and largest_frac == 1.0 means the group is still one flock.
    Anything else means the global polar order parameter is averaging over
    sub-groups that cannot communicate, and is therefore not measuring collective
    behaviour at all.
    """
    x = np.asarray(positions, dtype=float)
    n = len(x)
    neigh = metric_neighbors(x, boundary, r_max=r_link)
    A = adjacency_from_neighbors(neigh, n, symmetrize=True)
    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    _, counts = np.unique(labels, return_counts=True)
    return int(n_comp), float(counts.max() / n)


def mean_neighbor_count(positions: np.ndarray, boundary: Boundary,
                        r_max: float, r_min: float = 0.0) -> float:
    """Average |N_i|. Decays toward zero in open space -> interaction switches off."""
    neigh = metric_neighbors(positions, boundary, r_max=r_max, r_min=r_min)
    return float(np.mean([len(c) for c in neigh]))


def mean_speed(velocities: np.ndarray) -> float:
    """Mean |v_i|. Constant for the fixed-speed self-propelled models, but a real
    diagnostic for the ones that let speed vary (Cucker-Smale, D'Orsogna, Boids)."""
    v = np.asarray(velocities, dtype=float)
    if len(v) == 0:
        return float("nan")
    return float(np.linalg.norm(v, axis=1).mean())


def density(positions: np.ndarray, boundary: Boundary) -> float:
    """Number density N / volume, in [length]^-dim.

    Under a finite boundary the volume is L^dim BY CONSTRUCTION -- density is
    pinned and cannot decay, which is precisely the effect a torus hides. Under an
    open boundary there is no fixed volume, so we use the axis-aligned bounding box
    of the current positions as the occupied volume; this GROWS as an alignment-
    only group evaporates, so open-space density falls toward zero -- the same
    story `radius_of_gyration` tells, expressed as a density.
    """
    x = np.asarray(positions, dtype=float)
    n, dim = x.shape
    L = boundary.box_size
    if L is not None:
        vol = float(L) ** dim
    else:
        ranges = x.max(axis=0) - x.min(axis=0)
        vol = float(np.prod(np.where(ranges > 1e-9, ranges, np.nan)))
    if not np.isfinite(vol) or vol < 1e-12:
        return float("nan")
    return float(n / vol)


def heading_entropy(headings: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy of the heading-angle distribution, normalised to [0, 1].

    Complements `polar_order`: where M measures the length of the mean heading,
    this measures how spread the headings are across direction bins.

      ~0 : every agent points the same way (ordered).
      ~1 : headings uniformly spread over all directions (disordered gas).

    Angles are taken in the xy-plane (atan2(v_y, v_x)); for dim > 2 this is the
    azimuthal projection, which is enough for an order/disorder scalar.
    """
    h = np.asarray(headings, dtype=float)
    if len(h) == 0 or h.shape[1] < 2:
        return float("nan")
    ang = np.arctan2(h[:, 1], h[:, 0])
    counts, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    p = counts[counts > 0] / counts.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(bins))


def cohesion_verdict(largest_frac: float, n_frag: int) -> str:
    """Blunt classifier for reporting tables."""
    if n_frag == 1:
        return "COHESIVE"
    if largest_frac >= 0.5:
        return "PARTIAL"
    return "DISPERSED"


# --------------------------------------------------------------------------
# GROUP-MAINTENANCE observables
# --------------------------------------------------------------------------
# The diagnostics above ask a one-group question: did the flock hold together?
# A multi-group model can fail in a second, INDEPENDENT direction that none of them
# can see:
#
#   fission -- a group loses its own members.        Caught by group_integrity.
#   fusion  -- the groups merge into one blob.       Caught by segregation_index.
#
# Fusion is invisible to every metric above. A model whose groups have all merged
# scores n_fragments = 1 and largest_cluster_frac = 1.0 -- a PERFECT cohesion
# score -- while having destroyed exactly the structure it was built to maintain.
# So: never report cohesion alone for a grouping model. Report both directions.

def _group_sizes(groups: np.ndarray) -> np.ndarray:
    g = np.asarray(groups, dtype=int)
    return np.bincount(g, minlength=int(g.max()) + 1 if len(g) else 0)


def mixing_baseline(groups: np.ndarray) -> float:
    """Expected same-group neighbour fraction if the groups were WELL MIXED.

    sum_g (n_g/N) * ((n_g-1)/(N-1)): the chance that a randomly chosen neighbour of
    a randomly chosen agent shares its group, under total mixing.

    This is the number `group_purity` must be compared against. Raw purity is not
    interpretable on its own -- with k=2 equal groups, a totally fused, structureless
    cloud still scores ~0.5, which looks like "half the neighbours are group-mates"
    and means nothing at all.
    """
    g = np.asarray(groups, dtype=int)
    n = len(g)
    if n < 2:
        return float("nan")
    sizes = _group_sizes(g)
    return float(np.sum((sizes / n) * ((sizes - 1) / (n - 1))))


def group_purity(positions: np.ndarray, groups: np.ndarray, boundary: Boundary,
                 k: int = 7) -> float:
    """Mean fraction of an agent's k nearest neighbours that share its group.

    Topological (k-NN) rather than metric, deliberately: purity must stay meaningful
    while the group spreads. A metric ball empties out as agents disperse and the
    fraction becomes an average over nothing.

    Interpretation requires `mixing_baseline` -- use `segregation_index` instead.
    """
    x = np.asarray(positions, dtype=float)
    g = np.asarray(groups, dtype=int)
    n = len(x)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return float("nan")
    neigh = topological_neighbors(x, boundary, k_eff)
    fracs = [np.mean(g[c] == g[i]) for i, c in enumerate(neigh) if len(c)]
    return float(np.mean(fracs)) if fracs else float("nan")


def segregation_index(positions: np.ndarray, groups: np.ndarray,
                      boundary: Boundary, k: int = 7) -> float:
    """Purity, rescaled so that well-mixed = 0 and perfectly segregated = 1.

        S = (purity - baseline) / (1 - baseline)

    THE ANTI-FUSION DIAGNOSTIC, and the one number to report for a grouping model.

      S ~ 1  : every agent is surrounded by its own group. Groups are real.
      S ~ 0  : the groups have fused. Labels survive in the array and nowhere else.
      S < 0  : agents preferentially neighbour the OUT-group (anti-segregation).

    Note that S is a LOCAL measure -- it asks who your neighbours are, not where the
    groups are. Two groups can be perfectly segregated (S = 1) while flying side by
    side in contact. That is the correct reading: they are touching but distinct.
    """
    p = group_purity(positions, groups, boundary, k)
    b = mixing_baseline(groups)
    if not np.isfinite(p) or not np.isfinite(b) or b >= 1.0 - 1e-9:
        return float("nan")
    return float((p - b) / (1.0 - b))


def group_integrity(positions: np.ndarray, groups: np.ndarray,
                    boundary: Boundary, r_link: float) -> float:
    """Mean over groups of (largest connected component of the group / group size).

    THE ANTI-FISSION DIAGNOSTIC. `fragmentation` above asks whether the WHOLE
    ensemble is connected, which for a segregating multi-group model is the wrong
    question -- groups that have correctly pushed each other apart report many
    fragments and that is a success, not a failure. This asks the right one: is each
    group internally whole?

      1.0 : every group is a single connected blob.
      0.5 : the average group has shed half its members.
    """
    x = np.asarray(positions, dtype=float)
    g = np.asarray(groups, dtype=int)
    out = []
    for gid in np.unique(g):
        idx = np.flatnonzero(g == gid)
        if len(idx) < 2:
            out.append(1.0)
            continue
        n_frag, frac = fragmentation(x[idx], boundary, r_link)
        out.append(frac)
    return float(np.mean(out)) if out else float("nan")


def group_expansion(positions: np.ndarray, groups: np.ndarray,
                    boundary: Boundary) -> float:
    """Mean over groups of the WITHIN-group radius of gyration.

    Report it as a ratio to its t=0 value: this is `R_g growth` from Experiment 1,
    applied per group, and it is the honest anti-fission diagnostic for a
    Cucker-Smale in-group weight.

    WHY NOT JUST USE group_integrity? Because CS does not promise what integrity
    measures. The theorem gives velocity consensus and BOUNDED relative positions --
    bounded, not small, and certainly not "within r_link". A group that starts
    spread over 5 units stays spread over 5 units: R_g growth 1.00, a flawless
    result for the theorem, while integrity at r_link=1.5 reads ~0.07 because the
    group was never connected at that range to begin with. Integrity would be
    scoring the initial condition and calling it a model failure.

    Growth is the measure that matches the promise, and the beta threshold is
    visible in it: measured 1.09 (beta=0.4) vs 1.38 (beta=0.9) from a tight start,
    and 1.00 vs 1.50 from a spread one -- i.e. beta=0.4 does not expand from ANY
    initial condition, which is exactly what "unconditional" means.
    """
    x = np.asarray(positions, dtype=float)
    g = np.asarray(groups, dtype=int)
    out = [radius_of_gyration(x[g == gid], boundary)
           for gid in np.unique(g) if (g == gid).sum() > 1]
    return float(np.mean(out)) if out else float("nan")


def group_separation(positions: np.ndarray, groups: np.ndarray,
                     boundary: Boundary) -> float:
    """(min distance between group centroids) / (mean within-group R_g).

    A GLOBAL companion to segregation_index: are the groups in different PLACES?

      >> 2 : groups are far apart relative to their own size. Distinct blobs.
      ~ 0  : centroids coincide -- either fused, or one group has enveloped another.

    Returns nan for a single group (nothing to separate from).
    """
    x = np.asarray(positions, dtype=float)
    g = np.asarray(groups, dtype=int)
    gids = np.unique(g)
    if len(gids) < 2:
        return float("nan")
    cents = np.stack([centroid(x[g == gid], boundary) for gid in gids])
    spread = np.mean([radius_of_gyration(x[g == gid], boundary)
                      if (g == gid).sum() > 1 else 0.0 for gid in gids])
    dmin = np.inf
    for a in range(len(gids)):
        for b in range(a + 1, len(gids)):
            dmin = min(dmin, float(boundary.distance(cents[a], cents[b])))
    if spread < 1e-9:
        return float("inf") if dmin > 1e-9 else 0.0
    return float(dmin / spread)


def per_group_polar_order(headings: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """M computed WITHIN each group. Shape (n_groups,).

    The global M is meaningless for a multi-group flock -- it averages headings over
    groups that are deliberately not aligned with each other, so a set of perfectly
    ordered groups flying in different directions reports M ~ 0. That is the same
    'M is a liar' failure the fragmentation analysis exposes, one level up.
    """
    h = np.asarray(headings, dtype=float)
    g = np.asarray(groups, dtype=int)
    return np.array([polar_order(h[g == gid]) for gid in np.unique(g)])


def group_verdict(seg: float, integ: float) -> str:
    """Blunt classifier over BOTH failure directions."""
    if not np.isfinite(seg) or not np.isfinite(integ):
        return "UNDEFINED"
    if seg >= 0.75 and integ >= 0.9:
        return "MAINTAINED"
    if seg < 0.25:
        return "FUSED"          # groups merged: the labels are gone
    if integ < 0.5:
        return "FRAGMENTED"     # groups lost their own members
    return "PARTIAL"


def group_summary(positions, headings, groups, boundary, r_link, k: int = 7):
    """One-shot dict of every group diagnostic. Used by exp2 and the 3D viewer."""
    seg = segregation_index(positions, groups, boundary, k)
    integ = group_integrity(positions, groups, boundary, r_link)
    M_g = per_group_polar_order(headings, groups)
    return {
        "purity": group_purity(positions, groups, boundary, k),
        "mixing_baseline": mixing_baseline(groups),
        "segregation_index": seg,
        "group_integrity": integ,
        "group_expansion": group_expansion(positions, groups, boundary),
        "group_separation": group_separation(positions, groups, boundary),
        "mean_group_polar_order": float(np.mean(M_g)),
        "n_groups_alive": int(len(np.unique(groups))),
        "group_verdict": group_verdict(seg, integ),
    }


# --------------------------------------------------------------------------
# Consensus-model observables
# --------------------------------------------------------------------------

def disagreement(opinions: np.ndarray) -> float:
    """Total disagreement = std of the opinion vector. -> 0 at consensus."""
    return float(np.std(np.asarray(opinions, dtype=float)))


def cluster_count(opinions: np.ndarray, tol: float = 1e-2) -> int:
    """Number of distinct opinion clusters, by simple 1D agglomeration.

    Detects the multiconsensus / echo-chamber outcomes that Shrinate & Tripathy
    characterise topologically under the signed Friedkin-Johnsen model.
    """
    o = np.sort(np.asarray(opinions, dtype=float).ravel())
    if len(o) == 0:
        return 0
    return int(1 + np.sum(np.diff(o) > tol))


def bipartite_order(opinions: np.ndarray) -> float:
    """|<sign-folded opinion>| / <|opinion|>. ~1 for Altafini bipartite consensus."""
    o = np.asarray(opinions, dtype=float).ravel()
    denom = np.mean(np.abs(o))
    if denom < 1e-12:
        return 0.0
    return float(np.abs(np.mean(np.abs(o))) / denom)


def summarize(positions, headings, boundary, r_link, velocities=None):
    """One-shot dict of every diagnostic. Used by the experiment runners.

    `velocities` is optional; pass it to get a true `mean_speed` (from headings
    alone every speed is 1 by construction).
    """
    n_frag, frac = fragmentation(positions, boundary, r_link)
    return {
        "polar_order": polar_order(headings),
        "radius_of_gyration": radius_of_gyration(positions, boundary),
        "nn_distance": nn_distance(positions, boundary),
        "n_fragments": n_frag,
        "largest_cluster_frac": frac,
        "mean_neighbors": mean_neighbor_count(positions, boundary, r_link),
        "density": density(positions, boundary),
        "heading_entropy": heading_entropy(headings),
        "mean_speed": mean_speed(velocities) if velocities is not None else 1.0,
        "verdict": cohesion_verdict(frac, n_frag),
    }
