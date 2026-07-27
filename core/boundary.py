"""
Boundary conditions as a first-class, swappable abstraction.

THE CENTRAL POINT OF THIS CODEBASE
----------------------------------
Almost all of the flocking / active-matter literature (Vicsek 1995, Gregoire-Chate,
and both Beuria papers) simulates in a PERIODIC BOX. Periodic boundaries do two
things that quietly rescue a model:

  1. They pin the global density at rho = N / L^d forever.
  2. They guarantee that an agent always has neighbours within r_max, because
     the domain has no "outside" to escape into.

An alignment-only model (Vicsek and its descendants) has NO attractive term. Its
only interaction is "rotate my heading toward my neighbours' mean heading". Under
periodic BC this is enough to produce a beautiful order-disorder transition.
Under OPEN BC the same model disperses: agents drift apart, neighbour lists empty,
the interaction switches off, and the group fragments into non-interacting
sub-clusters that each order independently. The global polar order parameter then
reports a number that means nothing.

This module makes the boundary an explicit object so that the SAME model code can
be run under periodic, open, or reflecting conditions and compared honestly.

Every model must obtain inter-agent separations through `boundary.displacement()`
rather than by naive subtraction. That is the only discipline required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Boundary(ABC):
    """Abstract boundary condition.

    Responsibilities
    ----------------
    * ``displacement(xi, xj)`` : the vector from xi to xj under this topology.
    * ``wrap(x)``              : map positions back into the fundamental domain.
    * ``box_size``             : side length for KD-tree ``boxsize``, or None.
    """

    dim: int

    @abstractmethod
    def displacement(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Vector(s) pointing from xi to xj. Broadcasting-friendly."""

    @abstractmethod
    def wrap(self, x: np.ndarray) -> np.ndarray:
        """Return positions mapped into the fundamental domain."""

    @property
    @abstractmethod
    def box_size(self):
        """Side length for scipy cKDTree boxsize, or None if unbounded."""

    def distance(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Scalar distance(s) consistent with this topology."""
        return np.linalg.norm(self.displacement(xi, xj), axis=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


class PeriodicBoundary(Boundary):
    """Torus topology. The standard (and misleading) choice in the literature.

    Uses the minimum-image convention: the separation between two agents is the
    shortest of the 3^d periodic images. This is what silently guarantees that
    density never decays and neighbours are never lost.
    """

    def __init__(self, L: float, dim: int = 2):
        if L <= 0:
            raise ValueError("Box length L must be positive.")
        self.L = float(L)
        self.dim = int(dim)

    def displacement(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        d = np.asarray(xj) - np.asarray(xi)
        # Minimum image convention.
        return d - self.L * np.round(d / self.L)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        return np.mod(np.asarray(x), self.L)

    @property
    def box_size(self):
        return self.L

    def __repr__(self) -> str:
        return f"PeriodicBoundary(L={self.L}, dim={self.dim})"


class OpenBoundary(Boundary):
    """Unbounded free space R^d. The honest test.

    No wrapping, no minimum image, no density pinning. A model survives here only
    if it contains a genuine cohesive mechanism (attraction, or a communication
    weight that decays slowly enough to satisfy a flocking theorem such as
    Cucker-Smale's beta <= 1/2 condition).

    Alignment-only models FAIL here. That failure is the physics, not a bug.
    """

    def __init__(self, dim: int = 2):
        self.dim = int(dim)

    def displacement(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return np.asarray(xj) - np.asarray(xi)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)  # nothing to wrap; space is infinite

    @property
    def box_size(self):
        return None

    def __repr__(self) -> str:
        return f"OpenBoundary(dim={self.dim})"


class ReflectingBoundary(Boundary):
    """Hard walls in a box [0, L]^d.

    An intermediate case: the domain is finite (so density cannot decay to zero)
    but the topology is NOT a torus, so there is a real boundary that agents
    interact with. Useful as a control: it separates "does the model need a
    torus?" from "does the model need a finite domain?".

    Note that displacement here is plain Euclidean (no minimum image); the walls
    are enforced by `reflect_velocities` during integration, not by the metric.
    """

    def __init__(self, L: float, dim: int = 2):
        self.L = float(L)
        self.dim = int(dim)

    def displacement(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return np.asarray(xj) - np.asarray(xi)

    def wrap(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x), 0.0, self.L)

    def reflect(self, x: np.ndarray, v: np.ndarray):
        """Reflect positions and velocities off the walls. Returns (x, v)."""
        x = np.asarray(x, dtype=float).copy()
        v = np.asarray(v, dtype=float).copy()
        below = x < 0.0
        above = x > self.L
        x[below] = -x[below]
        v[below] = -v[below]
        x[above] = 2.0 * self.L - x[above]
        v[above] = -v[above]
        return np.clip(x, 0.0, self.L), v

    @property
    def box_size(self):
        return None  # not a torus; cKDTree boxsize would be wrong here

    def __repr__(self) -> str:
        return f"ReflectingBoundary(L={self.L}, dim={self.dim})"


def make_boundary(kind: str, L: float = 10.0, dim: int = 2) -> Boundary:
    """Factory. kind in {'periodic', 'open', 'reflecting'}."""
    kind = kind.lower()
    if kind == "periodic":
        return PeriodicBoundary(L, dim)
    if kind == "open":
        return OpenBoundary(dim)
    if kind == "reflecting":
        return ReflectingBoundary(L, dim)
    raise ValueError(f"Unknown boundary kind: {kind!r}")
