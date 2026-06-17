"""
flow_viz.py — physically-interpretable steady flow-field visualisation.

The MOC solve already computes the full supersonic flow state (Mach,
flow angle, hence pressure/temperature) at every characteristic-net
node.  This module turns that scattered data into a real field:

* smooth filled **Mach** and **temperature** contours, interpolated
  from the MOC net onto a grid and masked to the nozzle interior;
* the **characteristic lines** (the Mach waves) overlaid — the visible
  physics of MOC, including the throat expansion fan;
* **streamlines** integrated through the flow-angle field (the gas
  paths);
all mirrored about the axis for a full nozzle-slice view.

Desktop use: pass ``show=True`` for a pop-up window (matplotlib's
interactive backend — MacOSX/TkAgg/Qt), or ``save_path=...`` to write a
PNG.  Pure NumPy + SciPy + Matplotlib.
"""

from __future__ import annotations

import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
)


# --------------------------------------------------------------------------- #
# Gather the scattered MOC field data                                          #
# --------------------------------------------------------------------------- #
def _field_nodes(solution):
    """(x, r, M, theta) over the characteristic net + throat kernel."""
    xs, rs, Ms, ths = [], [], [], []
    for row in (getattr(solution, "characteristic_net", None) or []):
        pts = row.all_points() if hasattr(row, "all_points") else list(row)
        for p in pts:
            xs.append(p.x); rs.append(p.r); Ms.append(p.M); ths.append(p.theta)
    for p in (getattr(solution, "kernel_points", None) or []):
        xs.append(p.x); rs.append(p.r); Ms.append(p.M); ths.append(p.theta)
    return (np.asarray(xs), np.asarray(rs),
            np.asarray(Ms), np.asarray(ths))


def _wall_polyline(solution, geometry="raw"):
    wall = (solution.wall_raw if geometry == "raw"
            else solution.wall_export)
    wall = np.asarray(wall, dtype=float)
    order = np.argsort(wall[:, 0])
    return wall[order, 0], wall[order, 1]


def _streamlines(x, r, theta, wx, wr, *, n_lines=11, ds=None, max_steps=4000):
    """Integrate streamlines through the flow-angle field from a rake of
    seeds at the upstream edge to the exit (or the wall)."""
    from scipy.interpolate import LinearNDInterpolator
    th_interp = LinearNDInterpolator(np.column_stack([x, r]), theta,
                                     fill_value=np.nan)
    x0, x1 = float(x.min()), float(x.max())
    if ds is None:
        ds = (x1 - x0) / 600.0
    wall_at = lambda xx: float(np.interp(xx, wx, wr))
    seeds = np.linspace(0.04, 0.92, n_lines)        # fraction of local radius
    lines = []
    x_seed = x0 + 0.02 * (x1 - x0)
    for frac in seeds:
        xc, rc = x_seed, frac * wall_at(x_seed)
        xs, rs = [xc], [rc]
        for _ in range(max_steps):
            th = th_interp(xc, rc)
            if not np.isfinite(th):
                th = 0.0
            xc += ds * np.cos(th)
            rc += ds * np.sin(th)
            rc = max(rc, 0.0)
            if xc >= x1 or rc > wall_at(xc) * 1.001:
                break
            xs.append(xc); rs.append(rc)
        if len(xs) > 3:
            lines.append((np.asarray(xs), np.asarray(rs)))
    return lines


def _field_arrays(solution, gamma, *, anchor=True):
    """Filtered (x, r, M, theta) field nodes + wall polyline, shared by
    the static plot and the animations.  Drops non-physical/runaway
    seed-march nodes and (optionally) anchors the near-wall gap with the
    quasi-1-D wall Mach."""
    from raosim.gas_dynamics import mach_from_area_ratio
    x, r, M, theta = _field_nodes(solution)
    if x.size < 8:
        raise ValueError("flow field needs a populated characteristic net "
                         "(solve with evaluate_moc=True)")
    wx, wr = _wall_polyline(solution)
    Rt_est = max(float(wr.min()), 1e-9)
    eps_est = (float(wr.max()) / Rt_est) ** 2
    try:
        M_cap = 1.3 * mach_from_area_ratio(eps_est, gamma, supersonic=True)
    except Exception:
        M_cap = 8.0
    wall_r_at = np.interp(x, wx, wr)
    keep = (np.isfinite(M) & np.isfinite(theta) & (M >= 1.0) & (M <= M_cap)
            & (x >= wx.min() - 1e-9) & (x <= wx.max() + 1e-9)
            & (r >= -1e-9) & (r <= wall_r_at * 1.05))
    x, r, M, theta = x[keep], r[keep], M[keep], theta[keep]
    if x.size < 8:
        raise ValueError("flow field has too few physical nodes after "
                         "filtering; try a converged solve (max_nfev > 0)")
    if anchor:
        xa = np.linspace(x.min(), wx.max(), 80)
        wra = np.interp(xa, wx, wr)
        eps_a = np.maximum((wra / Rt_est) ** 2, 1.0001)
        M_wall = np.array([mach_from_area_ratio(float(e), gamma, supersonic=True)
                           for e in eps_a])
        th_wall = np.arctan2(np.gradient(wra), np.gradient(xa))
        x = np.concatenate([x, xa]); r = np.concatenate([r, wra])
        M = np.concatenate([M, M_wall]); theta = np.concatenate([theta, th_wall])
    return x, r, M, theta, wx, wr


def _grid_field(x, r, vals, wx, wr, *, nx=320, nr=120):
    """Interpolate scattered ``vals`` onto a grid, NaN outside the wall."""
    from scipy.interpolate import griddata
    xi = np.linspace(x.min(), x.max(), nx)
    ri = np.linspace(0.0, wr.max() * 1.001, nr)
    Xg, Rg = np.meshgrid(xi, ri)
    Vg = griddata((x, r), vals, (Xg, Rg), method="linear")
    wall_at = np.interp(Xg, wx, wr)
    Vg = np.where(Rg <= wall_at, Vg, np.nan)
    return Xg, Rg, Vg


# --------------------------------------------------------------------------- #
# The composite static flow-field figure                                       #
# --------------------------------------------------------------------------- #
def plot_flowfield(
    solution,
    gamma: float = 1.4,
    *,
    Tc: float | None = None,
    show_characteristics: bool = True,
    show_streamlines: bool = True,
    n_streamlines: int = 11,
    nx: int = 320,
    nr: int = 120,
    save_path: str | None = None,
    show: bool = False,
):
    """Render the steady flow field: filled Mach + temperature contours
    (MOC net interpolated onto a grid), characteristic lines, and
    streamlines, mirrored about the axis.

    ``Tc`` gives absolute temperatures [K]; otherwise the temperature
    panel is the stagnation ratio T/T0.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, r, M, theta, wx, wr = _field_arrays(solution, gamma, anchor=True)

    Xg, Rg, Mg = _grid_field(x, r, M, wx, wr, nx=nx, nr=nr)
    Tg = isentropic_temperature_ratio(np.clip(Mg, 1.0, None), gamma)
    if Tc is not None:
        Tg = Tg * Tc
    streams = (_streamlines(x, r, theta, wx, wr, n_lines=n_streamlines)
               if show_streamlines else [])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    panels = [
        (axes[0], Mg, "turbo", "Mach number", None),
        (axes[1], Tg, "inferno",
         "Temperature" + (" [K]" if Tc else "  T/T0"), None),
    ]
    for ax, field, cmap, label, _ in panels:
        levels = np.linspace(np.nanmin(field), np.nanmax(field), 30)
        for sgn in (1.0, -1.0):                     # mirror about the axis
            cf = ax.contourf(Xg, sgn * Rg, field, levels=levels, cmap=cmap,
                             extend="both")
        cb = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.05)
        cb.set_label(label)
        # wall (both halves) + axis
        ax.plot(wx, wr, color="k", lw=1.6); ax.plot(wx, -wr, color="k", lw=1.6)
        ax.axhline(0.0, color="w", lw=0.4, ls=":")
        if show_streamlines:
            for xs, rs in streams:
                ax.plot(xs, rs, color="w", lw=0.7, alpha=0.8)
                ax.plot(xs, -rs, color="w", lw=0.7, alpha=0.8)
        ax.set_ylabel("r [m]")
        ax.set_aspect("equal")

    # characteristic lines (the Mach waves / expansion fan) on the Mach
    # panel — clipped to physical points (a seed march can emit nodes
    # past the exit).
    x_lo, x_hi = float(wx.min()), float(wx.max())
    if show_characteristics:
        for row in (getattr(solution, "characteristic_net", None) or []):
            pts = row.all_points() if hasattr(row, "all_points") else list(row)
            seg = [(p.x, p.r) for p in pts
                   if x_lo - 1e-9 <= p.x <= x_hi + 1e-9
                   and 0.0 <= p.r <= np.interp(p.x, wx, wr) * 1.05]
            if len(seg) < 2:
                continue
            px = [s[0] for s in seg]; pr = [s[1] for s in seg]
            axes[0].plot(px, pr, color="k", lw=0.3, alpha=0.35)
            axes[0].plot(px, [-v for v in pr], color="k", lw=0.3, alpha=0.35)

    for ax in axes:                                  # frame the nozzle
        ax.set_xlim(x_lo - 0.03 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
        ax.set_ylim(-1.15 * float(wr.max()), 1.15 * float(wr.max()))

    Me = float(np.nanmax(Mg))
    axes[0].set_title(
        f"Steady flow field — Mach (top) & temperature (bottom)   "
        f"exit M≈{Me:.2f}"
        + (f", Cf={solution.thrust_coefficient:.3f}"
           if hasattr(solution, "thrust_coefficient") else ""),
        fontsize=11)
    axes[1].set_xlabel("x [m]")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Animations (FuncAnimation — pop up with show=True, or save mp4/gif)          #
# --------------------------------------------------------------------------- #
def _save_or_show(anim, fig, save_path, show, fps):
    if save_path is not None:
        import matplotlib.animation as manim
        writer = ("pillow" if str(save_path).lower().endswith(".gif")
                  else "ffmpeg")
        anim.save(str(save_path), writer=writer, fps=fps)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return anim


def animate_moc_march(
    solution, gamma: float = 1.4, *,
    interval: int = 140, fps: int = 8,
    save_path: str | None = None, show: bool = False,
):
    """Animate the method-of-characteristics march: the characteristic
    net (Mach waves) is revealed row by row from the throat, so you
    watch the expansion fan propagate down the bell.  Nodes coloured by
    Mach.  Pop up with ``show=True`` or save mp4/gif via ``save_path``.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    wx, wr = _wall_polyline(solution)
    x_lo, x_hi = float(wx.min()), float(wx.max())
    # physical per-row polylines (x, r, M)
    rows = []
    for row in (getattr(solution, "characteristic_net", None) or []):
        pts = row.all_points() if hasattr(row, "all_points") else list(row)
        seg = [(p.x, p.r, p.M) for p in pts
               if x_lo - 1e-9 <= p.x <= x_hi + 1e-9
               and 0.0 <= p.r <= np.interp(p.x, wx, wr) * 1.05
               and np.isfinite(p.M)]
        if len(seg) >= 2:
            rows.append(np.asarray(seg))
    if len(rows) < 2:
        raise ValueError("need a populated characteristic net to animate")
    Mmax = max(float(s[:, 2].max()) for s in rows)
    Mmin = min(float(s[:, 2].min()) for s in rows)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(wx, wr, "k", lw=1.6); ax.plot(wx, -wr, "k", lw=1.6)
    ax.axhline(0, color="0.7", lw=0.4, ls=":")
    ax.set_xlim(x_lo - 0.03 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
    ax.set_ylim(-1.15 * float(wr.max()), 1.15 * float(wr.max()))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    cmap = plt.get_cmap("turbo")
    from matplotlib import colors as mcolors, cm
    norm = mcolors.Normalize(Mmin, Mmax)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 pad=0.01, fraction=0.04, label="Mach number")

    def update(i):
        # redraw rows 0..i (cheap: ~20 rows)
        for a in list(ax.lines[2:]):       # keep the two wall lines
            a.remove()
        for a in list(ax.collections):
            a.remove()
        for s in rows[: i + 1]:
            ax.plot(s[:, 0], s[:, 1], color="0.25", lw=0.4, alpha=0.5)
            ax.plot(s[:, 0], -s[:, 1], color="0.25", lw=0.4, alpha=0.5)
            ax.scatter(s[:, 0], s[:, 1], c=s[:, 2], cmap=cmap, norm=norm, s=6)
            ax.scatter(s[:, 0], -s[:, 1], c=s[:, 2], cmap=cmap, norm=norm, s=6)
        ax.set_title(f"MOC march — characteristic row {i + 1}/{len(rows)} "
                     f"(the expansion fan propagating)", fontsize=11)
        return []

    anim = FuncAnimation(fig, update, frames=len(rows),
                         interval=interval, blit=False, repeat=True)
    return _save_or_show(anim, fig, save_path, show, fps)


def animate_particles(
    solution, gamma: float = 1.4, *,
    n_particles: int = 260, n_frames: int = 220, interval: int = 40, fps: int = 25,
    Tc: float | None = None, save_path: str | None = None, show: bool = False,
):
    """Advect tracer particles released at the throat through the steady
    flow field — they follow the true streamlines, coloured by local
    Mach, re-seeding as they exit.  ``show=True`` pops a window.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.interpolate import LinearNDInterpolator

    x, r, M, theta, wx, wr = _field_arrays(solution, gamma, anchor=True)
    pts = np.column_stack([x, r])
    th_i = LinearNDInterpolator(pts, theta, fill_value=0.0)
    M_i = LinearNDInterpolator(pts, M, fill_value=1.0)
    x_lo, x_hi = float(wx.min()), float(wx.max())
    wall_at = lambda xx: np.interp(xx, wx, wr)

    rng = np.random.default_rng(0)
    x_seed = x_lo + 0.01 * (x_hi - x_lo)

    def seed(n):
        fr = rng.uniform(0.03, 0.95, n)
        return np.column_stack([np.full(n, x_seed), fr * float(wall_at(x_seed))])

    P = seed(n_particles)
    # speed ∝ local gas velocity M·sqrt(T); normalise so a fast particle
    # crosses the nozzle in ~half the frames.
    def speed(P):
        m = np.clip(M_i(P[:, 0], P[:, 1]), 1.0, None)
        T = isentropic_temperature_ratio(m, gamma)
        return m * np.sqrt(T)
    v0 = float(np.nanmean(speed(P)))
    dt = (x_hi - x_lo) / (0.55 * n_frames * max(v0, 1e-6))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    Xg, Rg, Mg = _grid_field(x, r, M, wx, wr, nx=240, nr=90)
    for sgn in (1.0, -1.0):
        ax.contourf(Xg, sgn * Rg, Mg, levels=24, cmap="turbo", alpha=0.30)
    ax.plot(wx, wr, "k", lw=1.6); ax.plot(wx, -wr, "k", lw=1.6)
    ax.set_xlim(x_lo - 0.03 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
    ax.set_ylim(-1.15 * float(wr.max()), 1.15 * float(wr.max()))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    ax.set_title("Particle advection through the steady flow field", fontsize=11)
    from matplotlib import cm, colors as mcolors
    norm = mcolors.Normalize(1.0, float(np.nanmax(Mg)))
    sc = ax.scatter(P[:, 0], P[:, 1], c=speed(P) * 0 + 1.0, cmap="inferno",
                    norm=norm, s=10, edgecolors="none")
    sc2 = ax.scatter(P[:, 0], -P[:, 1], c=np.ones(len(P)), cmap="inferno",
                     norm=norm, s=10, edgecolors="none")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="inferno"), ax=ax,
                 pad=0.01, fraction=0.04, label="Mach at particle")

    def update(_frame):
        nonlocal P
        th = th_i(P[:, 0], P[:, 1]); th = np.where(np.isfinite(th), th, 0.0)
        v = speed(P)
        P = P + dt * v[:, None] * np.column_stack([np.cos(th), np.sin(th)])
        P[:, 1] = np.clip(P[:, 1], 0.0, None)
        # re-seed particles that left the nozzle (exit or through the wall)
        gone = (P[:, 0] >= x_hi) | (P[:, 1] > wall_at(P[:, 0]) * 1.02)
        if gone.any():
            P[gone] = seed(int(gone.sum()))
        m = np.clip(M_i(P[:, 0], P[:, 1]), 1.0, None)
        sc.set_offsets(P); sc.set_array(m)
        sc2.set_offsets(np.column_stack([P[:, 0], -P[:, 1]])); sc2.set_array(m)
        return sc, sc2

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=True)
    return _save_or_show(anim, fig, save_path, show, fps)
