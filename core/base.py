"""State container, model interface, and the simulation driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .boundary import Boundary, ReflectingBoundary
from .metrics import summarize


@dataclass
class State:
    """Full state of a self-propelled collective.

    positions  : (N, dim)
    velocities : (N, dim)
    internal   : model-specific dict (slow registers, opinions, Bloch vectors...)
    t          : time
    """

    positions: np.ndarray
    velocities: np.ndarray
    internal: dict = field(default_factory=dict)
    t: float = 0.0

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    @property
    def headings(self) -> np.ndarray:
        """Unit heading vectors."""
        v = self.velocities
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms > 1e-12, norms, 1.0)

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    def copy(self) -> "State":
        return State(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            internal={k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in self.internal.items()},
            t=self.t,
        )


class CollectiveModel(ABC):
    """Interface every model implements.

    Contract
    --------
    * `step` must obtain ALL inter-agent separations via `self.boundary.displacement`
      so that the same code is valid on a torus and in free space.
    * `step` must not assume a finite domain.
    * `cohesive` declares whether the model contains a mechanism that can hold the
      group together WITHOUT help from periodic walls. This is the honest label.
    """

    name: str = "unnamed"
    #: Does the model contain a genuine cohesive mechanism (attraction, or a
    #: communication weight satisfying a flocking theorem)? If False, the model
    #: will disperse under OpenBoundary -- and its periodic-box results are
    #: therefore contingent on the walls.
    cohesive: bool = False

    def __init__(self, boundary: Boundary, rng: np.random.Generator | None = None):
        self.boundary = boundary
        self.rng = rng if rng is not None else np.random.default_rng(0)

    @abstractmethod
    def step(self, state: State, dt: float) -> State:
        """Advance the state by dt. Return a NEW state."""

    def finalize(self, state: State) -> State:
        """Apply boundary handling after a step. Models should call this."""
        if isinstance(self.boundary, ReflectingBoundary):
            x, v = self.boundary.reflect(state.positions, state.velocities)
            state.positions, state.velocities = x, v
        else:
            state.positions = self.boundary.wrap(state.positions)
        return state

    def __repr__(self) -> str:
        tag = "cohesive" if self.cohesive else "alignment-only"
        return f"{self.__class__.__name__}(name={self.name!r}, {tag}, {self.boundary})"


def init_state(
    n: int,
    boundary: Boundary,
    dim: int = 2,
    speed: float = 0.5,
    spread: float = 10.0,
    rng: np.random.Generator | None = None,
) -> State:
    """Random initial condition.

    Under OpenBoundary, `spread` sets the initial blob size. Choosing the same
    spread as the periodic box length L makes the two runs directly comparable at
    t = 0: identical density, identical order, identical everything -- the ONLY
    difference is the topology. Any divergence afterwards is caused by the walls.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    L = spread if boundary.box_size is None else boundary.box_size
    x = rng.uniform(0.0, L, size=(n, dim))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    if dim == 2:
        v = speed * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    else:
        v = rng.normal(size=(n, dim))
        v = speed * v / np.linalg.norm(v, axis=1, keepdims=True)
    return State(positions=x, velocities=v)


def run(
    model: CollectiveModel,
    state: State,
    steps: int,
    dt: float,
    r_link: float,
    record_every: int = 10,
    record_traj: bool = False,
):
    """Drive a simulation and record diagnostics.

    Returns (final_state, history_dict).

    With record_traj=True the history also carries `trajectory` (T, N, dim) and
    `velocity_trajectory` (T, N, dim) -- the second is what the 3D viewer needs to
    draw headings -- plus `group_trajectory` (T, N) if the model maintains an
    allegiance vector in `state.internal["groups"]`, so that group switching can be
    animated as it happens.
    """
    hist = {k: [] for k in
            ["t", "polar_order", "radius_of_gyration", "nn_distance",
             "n_fragments", "largest_cluster_frac", "mean_neighbors",
             "density", "heading_entropy", "mean_speed"]}
    traj, vtraj, gtraj = [], [], []

    for k in range(steps):
        if k % record_every == 0:
            s = summarize(state.positions, state.headings, model.boundary, r_link,
                          velocities=state.velocities)
            hist["t"].append(state.t)
            for key in hist:
                if key != "t":
                    hist[key].append(s[key])
            if record_traj:
                traj.append(state.positions.copy())
                vtraj.append(state.velocities.copy())
                if "groups" in state.internal:
                    gtraj.append(np.asarray(state.internal["groups"]).copy())
        state = model.step(state, dt)

    out = {k: np.asarray(v) for k, v in hist.items()}
    if record_traj:
        out["trajectory"] = np.asarray(traj)
        out["velocity_trajectory"] = np.asarray(vtraj)
        if gtraj:
            out["group_trajectory"] = np.asarray(gtraj)
    return state, out
