"""
Initialization module -- the many ways to lay down the first `State`.

BirdFlockLab module 3. Every initializer returns a `core.base.State` and is
boundary-aware in exactly the sense the rest of the repo requires:

  * a finite boundary (periodic / reflecting) has a `box_size` L, so points are
    laid down inside [0, L]^dim and centred on (L/2, ...);
  * an unbounded boundary (open) has `box_size is None`, so points are laid down
    around the origin with an explicit `spread` / `arena` scale.

Group identity is optional. When more than one group is requested (or an explicit
allegiance vector is passed) it is stored in `state.internal["groups"]` as an int
array -- the SAME convention `models.grouping.init_grouped_state`, the metrics, and
the 3D viewer already use, so colour, segregation, and switching all read it
through one owner. An explicit per-agent colour list, when supplied to
`manual_init`, is stored in `state.internal["colors"]`; otherwise colour is derived
from the group id downstream (see `core.viz3d`).

This module deliberately does NOT import `models` -- it sits in `core`, below the
model zoo, so there is no import cycle.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .base import State
from .boundary import Boundary


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _rng(rng):
    return rng if rng is not None else np.random.default_rng(0)


def _centre_and_scale(boundary: Boundary, dim: int, fallback: float):
    """(centre, L) for placement. Finite boundary -> centred in the box."""
    L = boundary.box_size
    if L is not None:
        return np.full(dim, 0.5 * L), float(L)
    return np.zeros(dim), float(fallback)


def _random_headings(rng, n: int, dim: int, speed: float) -> np.ndarray:
    """n random unit velocities at the given speed."""
    if dim == 2:
        th = rng.uniform(0.0, 2.0 * np.pi, size=n)
        v = np.stack([np.cos(th), np.sin(th)], axis=1)
    else:
        v = rng.normal(size=(n, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return speed * v


def assign_groups(n: int, k: int, sizes=None) -> np.ndarray:
    """Allegiance vector g in {0..k-1}^n, as equal as n // k allows.

    Mirrors `models.grouping.assign_groups` but lives here so `core` stays
    independent of `models`. `sizes` gives explicit group sizes (must sum to n).
    """
    if k <= 1:
        return np.zeros(n, dtype=int)
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=int)
        if sizes.sum() != n:
            raise ValueError(f"sizes sum to {sizes.sum()}, expected n={n}")
        return np.repeat(np.arange(len(sizes)), sizes)
    base, extra = divmod(n, k)
    sizes = np.full(k, base)
    sizes[:extra] += 1
    return np.repeat(np.arange(k), sizes)


def _attach_groups(state: State, groups) -> State:
    if groups is not None:
        state.internal["groups"] = np.asarray(groups, dtype=int)
    return state


# --------------------------------------------------------------------------
# initializers
# --------------------------------------------------------------------------

def random_init(n: int, boundary: Boundary, dim: int | None = None,
                speed: float = 0.5, spread: float = 10.0, n_groups: int = 1,
                rng=None) -> State:
    """Uniformly random positions, random headings.

    Under a finite boundary points fill the box [0, L]^dim; under an open
    boundary they fill a `spread`-sized box anchored at the origin. Choosing
    `spread == L` makes an open run and a periodic run identical at t=0 -- the
    comparison this whole repo is built around.
    """
    rng = _rng(rng)
    dim = boundary.dim if dim is None else dim
    L = boundary.box_size
    if L is not None:
        x = rng.uniform(0.0, L, size=(n, dim))
    else:
        x = rng.uniform(0.0, spread, size=(n, dim))
    v = _random_headings(rng, n, dim, speed)
    st = State(positions=x, velocities=v)
    if n_groups > 1:
        _attach_groups(st, assign_groups(n, n_groups))
    return st


def cluster_init(n: int, boundary: Boundary, dim: int | None = None,
                 speed: float = 0.5, n_clusters: int = 3, cluster_std: float = 1.0,
                 arena: float = 8.0, aligned_within_cluster: bool = False,
                 as_groups: bool = True, rng=None) -> State:
    """`n_clusters` Gaussian blobs whose centres sit on a ring of radius arena/2.

    With `aligned_within_cluster` each blob gets one shared heading (+jitter), so
    the clusters start as coherent sub-flocks; otherwise headings are random.
    With `as_groups` each cluster becomes a group id -- this is the clean initial
    condition for the multi-group maintenance question.
    """
    rng = _rng(rng)
    dim = boundary.dim if dim is None else dim
    centre, L = _centre_and_scale(boundary, dim, arena)
    arena = min(arena, 0.7 * L) if boundary.box_size is not None else arena
    groups = assign_groups(n, n_clusters)
    angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)

    x = np.empty((n, dim))
    v = np.empty((n, dim))
    for g in range(n_clusters):
        idx = np.flatnonzero(groups == g)
        hub = centre.copy()
        hub[0] += 0.5 * arena * np.cos(angles[g])
        if dim >= 2:
            hub[1] += 0.5 * arena * np.sin(angles[g])
        x[idx] = hub + rng.normal(scale=cluster_std, size=(len(idx), dim))
        if aligned_within_cluster:
            h = rng.normal(size=dim)
            h /= np.linalg.norm(h)
            hv = h[None, :] + rng.normal(scale=0.15, size=(len(idx), dim))
            hv /= np.linalg.norm(hv, axis=1, keepdims=True)
            v[idx] = speed * hv
        else:
            v[idx] = _random_headings(rng, len(idx), dim, speed)

    st = State(positions=boundary.wrap(x), velocities=v)
    if as_groups and n_clusters > 1:
        _attach_groups(st, groups)
    return st


def ring_init(n: int, boundary: Boundary, dim: int | None = None,
              speed: float = 0.5, radius: float = 4.0, jitter: float = 0.0,
              tangential: bool = True, n_groups: int = 1, rng=None) -> State:
    """Agents on a circle (xy-plane), optionally with tangential headings.

    `tangential=True` seeds a rotating configuration -- the natural initial
    condition for milling states (see `metrics.milling_order`). `tangential=False`
    gives random headings on the ring.
    """
    rng = _rng(rng)
    dim = boundary.dim if dim is None else dim
    centre, L = _centre_and_scale(boundary, dim, 2.0 * radius + 2.0)
    if boundary.box_size is not None:
        radius = min(radius, 0.45 * L)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.tile(centre, (n, 1)).astype(float)
    x[:, 0] += radius * np.cos(theta)
    if dim >= 2:
        x[:, 1] += radius * np.sin(theta)
    if jitter > 0.0:
        x += rng.normal(scale=jitter, size=(n, dim))
    if tangential and dim >= 2:
        v = np.zeros((n, dim))
        v[:, 0] = -np.sin(theta)
        v[:, 1] = np.cos(theta)
        v *= speed
    else:
        v = _random_headings(rng, n, dim, speed)
    st = State(positions=boundary.wrap(x), velocities=v)
    if n_groups > 1:
        _attach_groups(st, assign_groups(n, n_groups))
    return st


def grid_init(n: int, boundary: Boundary, dim: int | None = None,
              speed: float = 0.5, spacing: float = 1.0, n_groups: int = 1,
              rng=None) -> State:
    """A regular lattice of (up to) n sites, centred in the domain.

    The lattice side is ceil(n ** (1/dim)); the first n sites are used, so n need
    not be a perfect power. Headings are random.
    """
    rng = _rng(rng)
    dim = boundary.dim if dim is None else dim
    side = int(np.ceil(n ** (1.0 / dim)))
    axes = [np.arange(side) * spacing for _ in range(dim)]
    mesh = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
    x = mesh[:n].astype(float)
    x -= x.mean(axis=0)                       # centre on origin
    centre, L = _centre_and_scale(boundary, dim, side * spacing + 2.0)
    x += centre
    v = _random_headings(rng, n, dim, speed)
    st = State(positions=boundary.wrap(x), velocities=v)
    if n_groups > 1:
        _attach_groups(st, assign_groups(n, n_groups))
    return st


def manual_init(positions, velocities=None, groups=None, colors=None,
                speed: float = 0.5, rng=None) -> State:
    """Build a State from explicit arrays -- the hook the GUI's click-to-place uses.

    `positions` : (N, dim) array-like (required).
    `velocities`: (N, dim); if None, random unit headings at `speed`.
    `groups`    : (N,) int allegiance, stored in internal["groups"].
    `colors`    : (N,) anything; stored verbatim in internal["colors"].
    """
    rng = _rng(rng)
    x = np.asarray(positions, dtype=float)
    if x.ndim != 2:
        raise ValueError("positions must be (N, dim)")
    n, dim = x.shape
    if velocities is None:
        v = _random_headings(rng, n, dim, speed)
    else:
        v = np.asarray(velocities, dtype=float)
        if v.shape != x.shape:
            raise ValueError(f"velocities shape {v.shape} != positions {x.shape}")
    st = State(positions=x, velocities=v)
    _attach_groups(st, groups)
    if colors is not None:
        st.internal["colors"] = list(colors)
    return st


# --------------------------------------------------------------------------
# CSV round-trip
# --------------------------------------------------------------------------

_POS_COLS = {2: ["x", "y"], 3: ["x", "y", "z"]}
_VEL_COLS = {2: ["vx", "vy"], 3: ["vx", "vy", "vz"]}


def state_to_csv(state: State, path) -> Path:
    """Write one State snapshot to CSV: x,y[,z],vx,vy[,vz][,group]."""
    path = Path(path)
    dim = state.dim
    header = _POS_COLS[dim] + _VEL_COLS[dim]
    groups = state.internal.get("groups")
    if groups is not None:
        header = header + ["group"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(state.n):
            row = list(state.positions[i]) + list(state.velocities[i])
            if groups is not None:
                row.append(int(groups[i]))
            w.writerow(row)
    return path


def from_csv(path, speed: float = 0.5, rng=None) -> State:
    """Read an initial State from CSV.

    Recognised columns (header row required, case-insensitive): x, y[, z] for
    position; vx, vy[, vz] for velocity (optional -- random headings if absent);
    group for allegiance (optional). Dimension is inferred from whether a `z`
    column is present.
    """
    path = Path(path)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    cols = {c.lower(): c for c in rows[0].keys()}
    dim = 3 if "z" in cols else 2
    pcols = _POS_COLS[dim]
    x = np.array([[float(r[cols[c]]) for c in pcols] for r in rows])
    if all(c in cols for c in _VEL_COLS[dim]):
        v = np.array([[float(r[cols[c]]) for c in _VEL_COLS[dim]] for r in rows])
    else:
        v = _random_headings(_rng(rng), len(rows), dim, speed)
    st = State(positions=x, velocities=v)
    if "group" in cols:
        _attach_groups(st, [int(float(r[cols["group"]])) for r in rows])
    return st


#: name -> initializer, for the experiment-manager registry.
INIT_REGISTRY = {
    "random": random_init,
    "cluster": cluster_init,
    "ring": ring_init,
    "grid": grid_init,
    "csv": from_csv,
}
