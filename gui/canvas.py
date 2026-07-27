"""
SimCanvas -- a QWidget that renders a `core.base.State` with QPainter.

Pure renderer: it holds a State plus display options and draws them. It knows
nothing about models or stepping; the main window feeds it a new State (and, when
the relevant toggles are on, a neighbour list and vision geometry) each frame.

World <-> screen
----------------
`scale` is pixels per world unit; `centre` is the world coordinate shown at the
widget centre; the y axis is flipped so +y points up. Wheel zooms about the
cursor, left-drag pans. `fit()` frames a set of positions (or the boundary box).
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, Qt

from core.boundary import Boundary, PeriodicBoundary, ReflectingBoundary

# A colour-blind-friendly categorical palette for groups (Okabe-Ito).
GROUP_QCOLORS = [
    QtGui.QColor(0, 114, 178),    # blue
    QtGui.QColor(230, 159, 0),    # orange
    QtGui.QColor(0, 158, 115),    # green
    QtGui.QColor(204, 121, 167),  # pink
    QtGui.QColor(213, 94, 0),     # vermillion
    QtGui.QColor(86, 180, 233),   # sky
    QtGui.QColor(240, 228, 66),   # yellow
    QtGui.QColor(120, 120, 120),  # grey
]
_BG = QtGui.QColor(18, 18, 22)
_FG = QtGui.QColor(235, 235, 235)


class SimCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # simulation payload (set via set_frame)
        self.positions = np.zeros((0, 2))
        self.headings = np.zeros((0, 2))
        self.groups = None
        self.boundary: Boundary | None = None
        self.neighbors = None                 # ragged list[np.ndarray] or None
        self.vision = None                    # (alpha_or_None, r_max) or None
        self.trails = []                       # list of (N,2) past positions

        # display options
        self.show_trails = True
        self.show_cones = False
        self.show_links = False
        self.show_groups = True
        self.agent_px = 4.0

        # view transform
        self.scale = 40.0
        self.centre = QPointF(5.0, 5.0)
        self._last_mouse = None

        # click-to-place ("place mode"): a click adds a bird, a drag still pans
        self.place_mode = False
        self.on_place = None          # callback(world_x, world_y)
        self._press_pos = None
        self._moved = False

    # -- payload ----------------------------------------------------------
    def set_frame(self, positions, headings, boundary, groups=None,
                  neighbors=None, vision=None):
        self.positions = np.asarray(positions, dtype=float)
        self.headings = np.asarray(headings, dtype=float)
        self.boundary = boundary
        self.groups = None if groups is None else np.asarray(groups, dtype=int)
        self.neighbors = neighbors
        self.vision = vision
        self.update()

    def push_trail(self, positions, max_len=40):
        self.trails.append(np.asarray(positions, dtype=float).copy())
        if len(self.trails) > max_len:
            self.trails.pop(0)

    def clear_trails(self):
        self.trails = []

    # -- view -------------------------------------------------------------
    def fit(self, positions, boundary):
        """Frame either the finite domain or the current point cloud."""
        L = boundary.box_size if boundary is not None else None
        if L is not None:
            lo = np.array([0.0, 0.0]); hi = np.array([L, L])
        elif len(positions):
            p = np.asarray(positions, dtype=float)
            lo, hi = p.min(axis=0), p.max(axis=0)
            pad = 0.1 * np.maximum(hi - lo, 1.0)
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = np.array([0.0, 0.0]), np.array([10.0, 10.0])
        span = np.maximum(hi - lo, 1e-6)
        self.centre = QPointF(float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
        w = max(self.width(), 1); h = max(self.height(), 1)
        self.scale = 0.9 * min(w / span[0], h / span[1])
        self.update()

    def _w2s(self, x, y):
        cx, cy = self.width() / 2, self.height() / 2
        return QPointF(cx + (x - self.centre.x()) * self.scale,
                       cy - (y - self.centre.y()) * self.scale)

    def _s2w(self, px, py):
        cx, cy = self.width() / 2, self.height() / 2
        return (self.centre.x() + (px - cx) / self.scale,
                self.centre.y() - (py - cy) / self.scale)

    def wheelEvent(self, e):
        wx, wy = self._s2w(e.position().x(), e.position().y())
        factor = 1.0015 ** e.angleDelta().y()
        self.scale = float(np.clip(self.scale * factor, 1.0, 5000.0))
        # keep the world point under the cursor fixed
        nx, ny = self._s2w(e.position().x(), e.position().y())
        self.centre = QPointF(self.centre.x() + (wx - nx),
                              self.centre.y() + (wy - ny))
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._last_mouse = e.position()
            self._press_pos = e.position()
            self._moved = False

    def mouseMoveEvent(self, e):
        if self._last_mouse is not None:
            d = e.position() - self._last_mouse
            self.centre = QPointF(self.centre.x() - d.x() / self.scale,
                                  self.centre.y() + d.y() / self.scale)
            self._last_mouse = e.position()
            if self._press_pos is not None:
                dd = e.position() - self._press_pos
                if (dd.x() ** 2 + dd.y() ** 2) ** 0.5 > 4:   # a real drag, not a click
                    self._moved = True
            self.update()

    def mouseReleaseEvent(self, e):
        if (e.button() == Qt.LeftButton and self.place_mode
                and not self._moved and self.on_place is not None):
            wx, wy = self._s2w(e.position().x(), e.position().y())
            self.on_place(wx, wy)
        self._last_mouse = None
        self._press_pos = None

    # -- colours ----------------------------------------------------------
    def _color(self, i):
        if self.show_groups and self.groups is not None:
            return GROUP_QCOLORS[int(self.groups[i]) % len(GROUP_QCOLORS)]
        return QtGui.QColor(120, 190, 255)

    # -- paint ------------------------------------------------------------
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.fillRect(self.rect(), _BG)
        if self.boundary is not None:
            self._draw_domain(p)
        if self.show_trails and self.trails:
            self._draw_trails(p)
        if self.show_links and self.neighbors is not None:
            self._draw_links(p)
        if self.show_cones and self.vision is not None:
            self._draw_cones(p)
        self._draw_agents(p)
        p.end()

    def _draw_domain(self, p):
        b = self.boundary
        if not isinstance(b, (PeriodicBoundary, ReflectingBoundary)):
            return
        L = b.L
        pen = QtGui.QPen(QtGui.QColor(120, 120, 130))
        pen.setCosmetic(True)
        if isinstance(b, PeriodicBoundary):
            pen.setStyle(Qt.DashLine)      # the walls are not real
        p.setPen(pen)
        tl = self._w2s(0, L); br = self._w2s(L, 0)
        p.drawRect(QtCore.QRectF(tl, br))

    def _draw_trails(self, p):
        L = self.boundary.box_size if self.boundary else None
        n = self.trails[-1].shape[0]
        for i in range(n):
            col = QtGui.QColor(self._color(i)); col.setAlpha(70)
            pen = QtGui.QPen(col); pen.setCosmetic(True); p.setPen(pen)
            prev = None
            for t, frame in enumerate(self.trails):
                if i >= frame.shape[0]:
                    prev = None; continue
                cur = frame[i]
                if prev is not None:
                    # skip segments that jump across a periodic wall
                    if L is None or np.max(np.abs(cur - prev)) < 0.5 * L:
                        p.drawLine(self._w2s(*prev), self._w2s(*cur))
                prev = cur

    def _draw_links(self, p):
        pen = QtGui.QPen(QtGui.QColor(150, 150, 160, 90))
        pen.setCosmetic(True); p.setPen(pen)
        x = self.positions
        b = self.boundary
        seen = set()
        for i, cand in enumerate(self.neighbors):
            for j in np.asarray(cand).ravel():
                key = (i, int(j)) if i < j else (int(j), i)
                if key in seen:
                    continue
                seen.add(key)
                # draw to the minimum-image position so periodic links look right
                d = b.displacement(x[i], x[int(j)]) if b is not None else x[int(j)] - x[i]
                p.drawLine(self._w2s(x[i, 0], x[i, 1]),
                           self._w2s(x[i, 0] + d[0], x[i, 1] + d[1]))

    def _draw_cones(self, p):
        alpha, r_max = self.vision
        x, h = self.positions, self.headings
        for i in range(len(x)):
            col = QtGui.QColor(self._color(i)); col.setAlpha(45)
            p.setBrush(col); p.setPen(Qt.NoPen)
            cx, cy = x[i, 0], x[i, 1]
            if alpha is None:
                c = self._w2s(cx, cy); r = r_max * self.scale
                p.drawEllipse(c, r, r)      # omnidirectional perception radius
                continue
            # Build the wedge from sampled points through our own transform, so
            # orientation is correct regardless of Qt's arc-angle convention.
            base = np.arctan2(h[i, 1], h[i, 0])
            angs = np.linspace(base - alpha / 2, base + alpha / 2, 16)
            path = QtGui.QPainterPath(self._w2s(cx, cy))
            for a in angs:
                path.lineTo(self._w2s(cx + r_max * np.cos(a),
                                      cy + r_max * np.sin(a)))
            path.closeSubpath()
            p.drawPath(path)

    def _draw_agents(self, p):
        x, h = self.positions, self.headings
        r = self.agent_px
        arrow = 2.2 * r / self.scale       # arrow length in world units
        for i in range(len(x)):
            col = self._color(i)
            c = self._w2s(x[i, 0], x[i, 1])
            p.setBrush(col); p.setPen(Qt.NoPen)
            p.drawEllipse(c, r, r)
            pen = QtGui.QPen(col); pen.setWidthF(1.6); pen.setCosmetic(True)
            p.setPen(pen)
            tip = self._w2s(x[i, 0] + h[i, 0] * arrow, x[i, 1] + h[i, 1] * arrow)
            p.drawLine(c, tip)
