"""
3D VISUALISATION.  Watch the models instead of reading their summary statistics.

Why this is in `core/` and not in a notebook
--------------------------------------------
The whole argument of this repo is that a scalar order parameter can be high while
the flock it describes has evaporated. `M = 0.843` and "it split into three pieces"
are the same run. Numbers alone let that hide; a picture does not. The 3D viewer is
the visual counterpart of `metrics.py`: watch Vicsek in open space and you SEE the
fragments drift apart, watch Cucker-Smale at beta=0.4 next to it and you see a group
that simply refuses to spread.

Everything here is boundary-aware, because a picture can lie in exactly the way the
metrics can:

  * PERIODIC : the box is drawn as a wireframe, and trails are CUT at the wrap.
               Without that cut, a bird crossing the wall draws a stripe straight
               across the box -- a line through space it never travelled.
  * OPEN     : the camera follows the centroid at a FIXED zoom (`follow=True`).
               This matters. Autoscaling each frame would silently renormalise away
               the group's expansion, which is the one thing the run is about: an
               evaporating flock would look like a stable one that merely drifts.
               Fixed zoom means dispersal looks like dispersal.

2D states are embedded in the z=0 plane, so every model in the repo can be viewed
here, not just the 3D ones.

Entry points
------------
    render(model, state, steps, dt, r_link, out=...)   simulate + write the movie
    animate3d(hist, boundary, out=...)                 movie from a recorded run
    panels3d(hist, boundary, out=...)                  static t-snapshot strip
    snapshot3d(positions, velocities, ...)             one frame onto an Axes3D

Movies are written as GIF via matplotlib's Pillow writer (no ffmpeg needed); pass
an `.mp4` path and ffmpeg is used if present, with a GIF fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

# Force a headless backend only when there is genuinely no display. This module is
# importable from a notebook or an IDE, and hard-coding Agg there would silently
# kill the caller's inline figures. Everything here writes to a file, so any
# backend works.
if not (os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .boundary import Boundary, OpenBoundary, PeriodicBoundary, ReflectingBoundary
from .base import run

#: Qualitative palette for group allegiance. Distinguishable, and safe against the
#: common colour-vision deficiencies -- group identity is the whole signal in the
#: multi-group plots, so it must not depend on red/green discrimination.
GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_SINGLE_COLOR = "#2b6cb0"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as3d(a: np.ndarray) -> np.ndarray:
    """Embed (..., 2) in the z=0 plane so 2D runs are viewable. (..., 3) passes through."""
    a = np.asarray(a, dtype=float)
    if a.shape[-1] == 3:
        return a
    if a.shape[-1] == 2:
        return np.concatenate([a, np.zeros(a.shape[:-1] + (1,))], axis=-1)
    raise ValueError(f"viz3d needs dim 2 or 3, got {a.shape[-1]}")


def _colors_for(groups, n: int):
    if groups is None:
        return [_SINGLE_COLOR] * n
    g = np.asarray(groups, dtype=int)
    return [GROUP_COLORS[i % len(GROUP_COLORS)] for i in g]


def _draw_box(ax, boundary: Boundary):
    """Wireframe the fundamental domain, for the boundaries that have one."""
    if not isinstance(boundary, (PeriodicBoundary, ReflectingBoundary)):
        return
    L = boundary.L
    z_hi = L if boundary.dim >= 3 else 0.0
    style = dict(color="0.55", lw=0.8)
    if isinstance(boundary, PeriodicBoundary):
        style.update(ls="--", alpha=0.7)      # dashed: the walls are not real
    else:
        style.update(ls="-", alpha=0.9)       # solid: these walls are real
    corners = [(0, 0), (L, 0), (L, L), (0, L)]
    for (a, b), (c, d) in zip(corners, corners[1:] + corners[:1]):
        ax.plot([a, c], [b, d], [0, 0], **style)
        if z_hi:
            ax.plot([a, c], [b, d], [z_hi, z_hi], **style)
    if z_hi:
        for a, b in corners:
            ax.plot([a, a], [b, b], [0, z_hi], **style)


def _trail_break_mask(traj: np.ndarray, boundary: Boundary) -> np.ndarray:
    """True where a step is a periodic wrap rather than real motion.

    A bird that leaves at x=L and re-enters at x=0 moved a distance dx ~ 0, but its
    stored coordinates jumped by L. Drawing that segment paints a line across the
    whole box that the bird never flew. Returns a (T, N) mask; the caller NaNs those
    samples so the trail is cut instead of smeared.
    """
    if not isinstance(boundary, PeriodicBoundary):
        return np.zeros(traj.shape[:2], dtype=bool)
    jump = np.abs(np.diff(traj, axis=0)).max(axis=-1) > 0.5 * boundary.L
    return np.concatenate([np.zeros((1,) + jump.shape[1:], dtype=bool), jump], axis=0)


def _camera(traj: np.ndarray, boundary: Boundary, follow: bool, pad: float = 1.15,
            half_width=None, focus=None):
    """Return (centres (T,3), half_width) for the view box.

    Fixed half-width, moving centre. The fixed width is the point: it is what makes
    an expanding group look expanding rather than merely translating.

    focus      : agent indices the camera centres on (default: all). Use it when one
                 subgroup is the subject and the rest of the swarm has left for
                 somewhere else entirely.
    half_width : override the auto-fitted zoom, in data units.

                 USE THIS DELIBERATELY. The auto fit spans the whole run, which is
                 correct when the question is "did this group spread?" -- and useless
                 when the group also TRANSLATES a long way, because the span is then
                 set by the journey rather than by the flock. A 3-team UAV run that
                 flies 1.2 km while each team stays 6 m across auto-fits to ~1.2 km
                 and renders each team as one pixel.
                 The honesty cost is real: a hand-set zoom CAN hide expansion, which
                 is exactly what the auto fit exists to prevent. So when you set it,
                 keep R_g in the HUD -- the number is then the check on the picture.
    """
    if isinstance(boundary, (PeriodicBoundary, ReflectingBoundary)):
        L = boundary.L
        c = np.tile([0.5 * L, 0.5 * L, 0.5 * L if boundary.dim >= 3 else 0.0],
                    (len(traj), 1))
        return c, (half_width if half_width is not None else 0.5 * L * pad)

    sub = traj if focus is None else traj[:, np.asarray(focus), :]
    cents = sub.mean(axis=1)                                    # (T, 3)
    if not follow:
        cents = np.tile(sub.reshape(-1, 3).mean(axis=0), (len(traj), 1))
    if half_width is not None:
        return cents, float(half_width)
    spread = np.linalg.norm(sub - cents[:, None, :], axis=-1).max()
    return cents, max(float(spread) * pad, 1e-3)


def _apply_camera(ax, centre, half, boundary):
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))     # equal aspect: no visual distortion of shape
    except AttributeError:               # matplotlib < 3.3
        pass


def _style_axes(ax):
    ax.set_xlabel("x", fontsize=8, labelpad=-6)
    ax.set_ylabel("y", fontsize=8, labelpad=-6)
    ax.set_zlabel("z", fontsize=8, labelpad=-6)
    ax.tick_params(labelsize=0, length=0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.04)
    ax.grid(alpha=0.15)


# --------------------------------------------------------------------------
# single frame
# --------------------------------------------------------------------------

def snapshot3d(positions, velocities, boundary: Boundary, groups=None, ax=None,
               title: str = "", arrow: float = 0.5, follow: bool = True,
               elev: float = 22.0, azim: float = -60.0, dot: float = 9.0):
    """Draw one frame: a dot per bird, an arrow along its heading.

    Returns the Axes3D. Colour is group allegiance when `groups` is given.
    """
    x = _as3d(np.asarray(positions))
    v = _as3d(np.asarray(velocities))
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    cols = _colors_for(groups, len(x))
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=cols, s=dot, depthshade=True,
               edgecolors="none")
    h = v / np.where(np.linalg.norm(v, axis=1, keepdims=True) > 1e-12,
                     np.linalg.norm(v, axis=1, keepdims=True), 1.0)
    ax.quiver(x[:, 0], x[:, 1], x[:, 2], h[:, 0], h[:, 1], h[:, 2],
              length=arrow, colors=cols, linewidth=0.7, arrow_length_ratio=0.35)

    _draw_box(ax, boundary)
    centres, half = _camera(x[None, ...], boundary, follow)
    _apply_camera(ax, centres[0], half, boundary)
    _style_axes(ax)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# --------------------------------------------------------------------------
# static strip of snapshots
# --------------------------------------------------------------------------

def panels3d(hist, boundary: Boundary, out="results/viz3d_panels.png",
             n_panels: int = 4, title: str = "", follow: bool = True,
             arrow: float = 0.5, subtitle_fn=None):
    """Snapshots at equally spaced times, side by side. The still version of the movie.

    `hist` must come from `run(..., record_traj=True)`. `subtitle_fn(i_frame)` may
    return an extra line per panel (used by exp2 to print the live group metrics).
    """
    traj = _as3d(hist["trajectory"])
    vtraj = _as3d(hist["velocity_trajectory"])
    gtraj = hist.get("group_trajectory")
    t = hist["t"]
    idx = np.linspace(0, len(traj) - 1, n_panels).astype(int)

    fig = plt.figure(figsize=(4.2 * n_panels, 4.6))
    centres, half = _camera(traj, boundary, follow)
    for p, k in enumerate(idx):
        ax = fig.add_subplot(1, n_panels, p + 1, projection="3d")
        g = gtraj[k] if gtraj is not None else None
        cols = _colors_for(g, traj.shape[1])
        x, v = traj[k], vtraj[k]
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=cols, s=8, edgecolors="none")
        h = v / np.where(np.linalg.norm(v, axis=1, keepdims=True) > 1e-12,
                         np.linalg.norm(v, axis=1, keepdims=True), 1.0)
        ax.quiver(x[:, 0], x[:, 1], x[:, 2], h[:, 0], h[:, 1], h[:, 2],
                  length=arrow, colors=cols, linewidth=0.6, arrow_length_ratio=0.35)
        _draw_box(ax, boundary)
        _apply_camera(ax, centres[k], half, boundary)
        _style_axes(ax)
        ax.view_init(elev=22, azim=-60)
        sub = subtitle_fn(k) if subtitle_fn else ""
        ax.set_title(f"t = {t[k]:.1f}" + (f"\n{sub}" if sub else ""), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94 if title else 1.0])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# --------------------------------------------------------------------------
# movie
# --------------------------------------------------------------------------

def animate3d(hist, boundary: Boundary, out="results/viz3d.gif", fps: int = 18,
              trail: int = 12, title: str = "", follow: bool = True,
              arrow: float = 0.5, spin: float = 0.12, elev: float = 22.0,
              azim: float = -60.0, hud_fn=None, dpi: int = 110,
              half_width=None, focus=None, stride: int = 1):
    """Animate a recorded run. `hist` from `run(..., record_traj=True)`.

    trail  : frames of position history drawn behind each bird (0 = none). Trails are
             cut at periodic wraps, never drawn across the box.
    spin   : degrees of azimuth per frame. A slow orbit; 3D structure (a mill, a
             sheet, a shell) is genuinely ambiguous from one fixed viewpoint.
    hud_fn : hud_fn(i_frame) -> str, printed in the corner each frame. Pass the live
             metrics here so the picture and the numbers cannot drift apart. NOTE it
             is called with the index into the ORIGINAL history, not the strided one.
    stride : keep every n-th recorded frame. A run recorded finely for metrics does
             not need every sample rendered -- 8000 steps at record_every=10 is 800
             frames and a 33 MB GIF nobody will open.
    half_width, focus : see `_camera`.
    """
    traj = _as3d(hist["trajectory"])
    vtraj = _as3d(hist["velocity_trajectory"])
    gtraj = hist.get("group_trajectory")
    t = hist["t"]

    if stride > 1:
        keep = np.arange(0, len(traj), stride)
        traj, vtraj, t = traj[keep], vtraj[keep], t[keep]
        if gtraj is not None:
            gtraj = gtraj[keep]
        if hud_fn is not None:
            _hud, _keep = hud_fn, keep
            hud_fn = lambda k: _hud(int(_keep[k]))    # noqa: E731

    T, N, _ = traj.shape
    breaks = _trail_break_mask(traj, boundary)
    centres, half = _camera(traj, boundary, follow, half_width=half_width,
                            focus=focus)

    fig = plt.figure(figsize=(7.2, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    # Let the 3D axes fill the frame; title and HUD are figure-level text placed in
    # the margins, so they cannot collide with each other or with the flock.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
    if title:
        fig.text(0.5, 0.975, title, fontsize=11.5, fontweight="bold",
                 ha="center", va="top")
    hud = fig.text(0.015, 0.015, "", fontsize=9, family="monospace", va="bottom")

    # Set the scene up ONCE. The per-frame callback must never clear the axes: the
    # trail artists live here and would be destroyed with it.
    _draw_box(ax, boundary)
    _style_axes(ax)

    static_cols = _colors_for(gtraj[0] if gtraj is not None else None, N)
    trail_lines = []
    if trail > 0:
        for i in range(N):
            (ln,) = ax.plot([], [], [], lw=0.6, alpha=0.35, color=static_cols[i])
            trail_lines.append(ln)

    state = {"quiver": None, "scat": None}

    def draw(k):
        if state["quiver"] is not None:
            state["quiver"].remove()
        if state["scat"] is not None:
            state["scat"].remove()

        x, v = traj[k], vtraj[k]
        g = gtraj[k] if gtraj is not None else None
        cols = _colors_for(g, N)

        state["scat"] = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=cols, s=10,
                                   edgecolors="none", depthshade=True)
        h = v / np.where(np.linalg.norm(v, axis=1, keepdims=True) > 1e-12,
                         np.linalg.norm(v, axis=1, keepdims=True), 1.0)
        state["quiver"] = ax.quiver(x[:, 0], x[:, 1], x[:, 2],
                                    h[:, 0], h[:, 1], h[:, 2], length=arrow,
                                    colors=cols, linewidth=0.7,
                                    arrow_length_ratio=0.35)

        if trail > 0:
            lo = max(0, k - trail)
            seg = traj[lo:k + 1].copy()                       # (m, N, 3)
            seg[breaks[lo:k + 1]] = np.nan                    # cut, do not smear
            for i, ln in enumerate(trail_lines):
                ln.set_data(seg[:, i, 0], seg[:, i, 1])
                ln.set_3d_properties(seg[:, i, 2])
                if g is not None:
                    ln.set_color(GROUP_COLORS[int(g[i]) % len(GROUP_COLORS)])

        _apply_camera(ax, centres[k], half, boundary)
        ax.view_init(elev=elev, azim=azim + spin * k)
        txt = f"t = {t[k]:6.1f}"
        if hud_fn:
            txt += "\n" + hud_fn(k)
        hud.set_text(txt)
        return []

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 / fps, blit=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _save(anim, out, fps, dpi)
    plt.close(fig)
    return out


def _save(anim, out, fps, dpi):
    out = str(out)
    if out.endswith(".mp4"):
        try:
            anim.save(out, writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"wrote {out}")
            return
        except Exception as e:      # ffmpeg missing or misconfigured
            out = out[:-4] + ".gif"
            print(f"ffmpeg unavailable ({type(e).__name__}); falling back to {out}")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# convenience: simulate and render in one call
# --------------------------------------------------------------------------

def render(model, state, steps: int, dt: float, r_link: float,
           out="results/viz3d.gif", record_every: int = 10, panels: bool = True,
           title: str = None, **kw):
    """Run `model` and write the movie (and, by default, the still strip).

    Returns (final_state, hist) exactly like `core.run`, so the numbers behind the
    picture stay available to the caller.
    """
    title = title if title is not None else f"{model.name}  --  {model.boundary}"
    final, hist = run(model, state, steps=steps, dt=dt, r_link=r_link,
                      record_every=record_every, record_traj=True)
    animate3d(hist, model.boundary, out=out, title=title, **kw)
    if panels:
        stem = str(Path(out).with_suffix(""))
        panels3d(hist, model.boundary, out=stem + "_panels.png", title=title)
    return final, hist
