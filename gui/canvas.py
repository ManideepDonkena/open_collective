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

        # 3D view (used automatically when positions are 3D): orthographic,
        # rotate by dragging. azim/elev are the view angles; _center3 is the
        # world point the projection is centred on.
        self.azim = 0.6
        self.elev = 0.4
        self._center3 = np.zeros(3)

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
        pos = np.asarray(positions, dtype=float)
        if pos.ndim == 2 and pos.shape[1] >= 3:      # 3D view
            if L is not None:
                self._center3 = np.array([L / 2, L / 2, L / 2]); extent = L
            elif len(pos):
                lo3, hi3 = pos.min(axis=0), pos.max(axis=0)
                self._center3 = (lo3 + hi3) / 2
                extent = float(max((hi3 - lo3).max(), 1.0)) * 1.3
            else:
                self._center3 = np.zeros(3); extent = 10.0
            w = max(self.width(), 1); h = max(self.height(), 1)
            self.scale = 0.8 * min(w, h) / extent
            self.update()
            return
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
            if self.positions.ndim == 2 and self.positions.shape[1] >= 3:
                self.azim += d.x() * 0.01                     # 3D: drag rotates
                self.elev = float(np.clip(self.elev + d.y() * 0.01, -1.55, 1.55))
            else:
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
        if self.positions.ndim == 2 and self.positions.shape[1] >= 3:
            self._paint3d(p)
        else:
            self._paint2d(p)
        p.end()

    def _paint2d(self, p):
        if self.boundary is not None:
            self._draw_domain(p)
        if self.show_trails and self.trails:
            self._draw_trails(p)
        if self.show_links and self.neighbors is not None:
            self._draw_links(p)
        if self.show_cones and self.vision is not None:
            self._draw_cones(p)
        self._draw_agents(p)

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

    # -- 3D rendering (orthographic, rotatable) ---------------------------
    def _project3(self, pts):
        """World (N,3) -> (screen_x, screen_y, depth). Depth sorts near/far."""
        q = np.asarray(pts, dtype=float).reshape(-1, 3) - self._center3
        ca, sa = np.cos(self.azim), np.sin(self.azim)
        ce, se = np.cos(self.elev), np.sin(self.elev)
        x = ca * q[:, 0] - sa * q[:, 1]                 # yaw about z
        y = sa * q[:, 0] + ca * q[:, 1]
        z = q[:, 2]
        yv = ce * y - se * z                            # pitch about x
        depth = se * y + ce * z
        sx = self.width() / 2 + x * self.scale
        sy = self.height() / 2 - yv * self.scale
        return sx, sy, depth

    def _paint3d(self, p):
        if self.boundary is not None:
            self._draw_box3d(p)
        if self.show_trails and self.trails:
            self._draw_trails3d(p)
        if self.show_links and self.neighbors is not None:
            self._draw_links3d(p)
        self._draw_agents3d(p)

    def _draw_box3d(self, p):
        b = self.boundary
        if not isinstance(b, (PeriodicBoundary, ReflectingBoundary)):
            return
        L = b.L
        corners = np.array([[a, c, d] for a in (0, L) for c in (0, L)
                            for d in (0, L)], dtype=float)
        sx, sy, _ = self._project3(corners)
        pen = QtGui.QPen(QtGui.QColor(120, 120, 130)); pen.setCosmetic(True)
        if isinstance(b, PeriodicBoundary):
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count("1") == 1:          # cube edge: differ in one axis
                    p.drawLine(QPointF(sx[i], sy[i]), QPointF(sx[j], sy[j]))

    def _draw_agents3d(self, p):
        x, h = self.positions, self.headings
        sx, sy, depth = self._project3(x)
        hx, hy, _ = self._project3(x + h * 0.7)         # heading tip in world space
        r = self.agent_px
        for i in np.argsort(depth):                     # far first, near last
            col = self._color(int(i))
            c = QPointF(sx[i], sy[i])
            p.setBrush(col); p.setPen(Qt.NoPen)
            p.drawEllipse(c, r, r)
            pen = QtGui.QPen(col); pen.setWidthF(1.4); pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(c, QPointF(hx[i], hy[i]))

    def _draw_trails3d(self, p):
        L = self.boundary.box_size if self.boundary else None
        n = self.trails[-1].shape[0]
        for i in range(n):
            pts = [f[i] for f in self.trails if i < f.shape[0]]
            if len(pts) < 2:
                continue
            arr = np.asarray(pts)
            sx, sy, _ = self._project3(arr)
            col = QtGui.QColor(self._color(i)); col.setAlpha(70)
            pen = QtGui.QPen(col); pen.setCosmetic(True); p.setPen(pen)
            for k in range(1, len(arr)):
                if L is None or np.max(np.abs(arr[k] - arr[k - 1])) < 0.5 * L:
                    p.drawLine(QPointF(sx[k - 1], sy[k - 1]),
                               QPointF(sx[k], sy[k]))

    def _draw_links3d(self, p):
        x, b = self.positions, self.boundary
        sx, sy, _ = self._project3(x)
        pen = QtGui.QPen(QtGui.QColor(150, 150, 160, 90)); pen.setCosmetic(True)
        p.setPen(pen)
        seen = set()
        for i, cand in enumerate(self.neighbors):
            for j in np.asarray(cand).ravel():
                j = int(j)
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    continue
                seen.add(key)
                d = b.displacement(x[i], x[j]) if b is not None else x[j] - x[i]
                ex, ey, _ = self._project3(x[i] + d)
                p.drawLine(QPointF(sx[i], sy[i]), QPointF(ex[0], ey[0]))
