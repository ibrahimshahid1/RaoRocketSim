"""
flow_viz.py — physically-interpretable steady flow-field visualisation.

The MOC solve computes the resolved supersonic flow state (Mach, flow
angle, hence pressure/temperature) at every characteristic-net node.
This module turns that scattered data into a field while explicitly
checking that the solved nodes cover the nozzle from axis to wall:

* smooth filled **Mach**, **pressure**, **flow-angle**, and **temperature**
  contours, interpolated from the MOC net and masked to the nozzle interior;
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
def _bde_artifacts(solution):
    diagnostics = getattr(solution, "construction_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None, None
    artifacts = diagnostics.get("bde_artifacts")
    if not isinstance(artifacts, dict):
        return None, None
    return artifacts.get("kernel"), artifacts.get("bde_region")


def _row_points(row):
    return row.all_points() if hasattr(row, "all_points") else list(row)


def _deduplicate_nodes(x, r, M, theta):
    """Merge coincident nodes shared by adjacent construction regions."""
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    M = np.asarray(M, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if not x.size:
        return x, r, M, theta
    scale = max(float(np.ptp(x)), float(np.ptp(r)), 1.0)
    key = np.round(np.column_stack([x, r]) / scale, decimals=13)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(float)

    def averaged(values):
        return np.bincount(inverse, weights=values) / count

    return averaged(x), averaged(r), averaged(M), averaged(theta)


def _field_node_rows(solution):
    """Return solved rows and their provenance.

    The BDE wall path stores its complete construction in diagnostics rather
    than in ``RaoSolution.characteristic_net``.  Its physical field is the
    union of the throat kernel, the wall-to-DE B-D-E strip, and the valid
    DE-to-axis continuation.  Only the continuation at/after ``iD`` is used;
    its raw prefix lies outside the extracted wall and is diagnostic-only.
    """
    kernel, bde = _bde_artifacts(solution)
    if kernel is not None and bde is not None:
        rows = list(getattr(kernel, "rrcs", ()) or ())
        rows.extend(list(getattr(bde, "grid_rows", ()) or ()))
        iD = int(getattr(bde, "iD", 0) or 0)
        for row in (getattr(bde, "full_grid_rows", ()) or ()):
            pts = _row_points(row)
            if len(pts) > iD:
                rows.append(pts[iD:])
        wall = list(getattr(bde, "wall_contour", ()) or ())
        if wall:
            rows.append(wall)
        return rows, "bde"

    rows = []
    for row in (getattr(solution, "characteristic_net", None) or []):
        rows.append(_row_points(row))
    kernel_points = list(getattr(solution, "kernel_points", None) or [])
    if kernel_points:
        rows.append(kernel_points)
    return rows, "legacy"


def _field_nodes(solution):
    """(x, r, M, theta) over every available solved MOC region."""
    xs, rs, Ms, ths = [], [], [], []
    rows, _ = _field_node_rows(solution)
    for row in rows:
        for p in _row_points(row):
            xs.append(p.x); rs.append(p.r); Ms.append(p.M); ths.append(p.theta)
    return _deduplicate_nodes(xs, rs, Ms, ths)


def _characteristic_polylines(solution):
    """Characteristic polylines used by the field overlay.

    BDE solutions expose both the row and column families.  The exterior
    prefix of a full BDE row is harmless here because the plotting filter
    clips it to the actual nozzle wall.
    """
    kernel, bde = _bde_artifacts(solution)
    if kernel is None or bde is None:
        return [
            ("net", _row_points(row))
            for row in (getattr(solution, "characteristic_net", None) or [])
        ]

    lines = [("rrc", _row_points(row))
             for row in (getattr(kernel, "rrcs", ()) or ())]
    full_rows = [
        _row_points(row)
        for row in (getattr(bde, "full_grid_rows", ()) or ())
    ]
    lines.extend(("rrc", _row_points(row))
                 for row in (getattr(bde, "grid_rows", ()) or ()))
    lines.extend(("rrc", row) for row in full_rows)
    max_len = max((len(row) for row in full_rows), default=0)
    for index in range(max_len):
        column = [row[index] for row in full_rows if index < len(row)]
        if len(column) >= 2:
            lines.append(("lrc", column))
    return lines


def _wall_polyline(solution, geometry="raw"):
    wall = (solution.wall_raw if geometry == "raw"
            else solution.wall_export)
    wall = np.asarray(wall, dtype=float)
    order = np.argsort(wall[:, 0])
    return wall[order, 0], wall[order, 1]


def _streamlines(
    x, r, theta, wx, wr, *, n_lines=11, ds=None, max_steps=4000,
    x_start=None,
):
    """Integrate streamlines through the flow-angle field from a rake of
    seeds at the upstream edge to the exit (or the wall)."""
    from scipy.interpolate import LinearNDInterpolator
    th_interp = LinearNDInterpolator(np.column_stack([x, r]), theta,
                                     fill_value=np.nan)
    x0, x1 = float(wx.min()), float(wx.max())
    if ds is None:
        ds = (x1 - x0) / 600.0
    wall_at = lambda xx: float(np.interp(xx, wx, wr))
    seeds = np.linspace(0.04, 0.92, n_lines)        # fraction of local radius
    lines = []
    if x_start is None or not np.isfinite(x_start):
        x_start = x0
    x_seed = max(float(x_start), x0) + 0.005 * (x1 - x0)
    for frac in seeds:
        xc, rc = x_seed, frac * wall_at(x_seed)
        xs, rs = [xc], [rc]
        for _ in range(max_steps):
            th = th_interp(xc, rc)
            if not np.isfinite(th):
                break
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
    seed-march nodes and (optionally, for the legacy path only) anchors the
    near-wall gap with the quasi-1-D wall Mach.  A BDE field never receives
    synthetic wall states."""
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
    _, source = _field_node_rows(solution)
    bde_field = source == "bde"
    length = max(float(wx.max() - wx.min()), 1e-12)
    support_hi = float(wx.max() + (0.20 * length if bde_field else 1e-9))
    wall_r_at = np.interp(x, wx, wr)
    keep = (np.isfinite(x) & np.isfinite(r) & np.isfinite(M)
            & np.isfinite(theta) & (M >= 1.0)
            & (M <= (20.0 if bde_field else M_cap))
            & (x >= wx.min() - 1e-9) & (x <= support_hi)
            & (r >= -1e-9) & (r <= wall_r_at * 1.05))
    x, r, M, theta = x[keep], r[keep], M[keep], theta[keep]
    if x.size < 8:
        raise ValueError("flow field has too few physical nodes after "
                         "filtering; try a converged solve (max_nfev > 0)")
    x, r, M, theta = _deduplicate_nodes(x, r, M, theta)
    if anchor and not bde_field:
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
    # Support nodes slightly downstream of the exit are retained so the exit
    # plane is interpolated rather than extrapolated, but the rendered domain
    # is always exactly the nozzle wall extent.
    xi = np.linspace(wx.min(), wx.max(), nx)
    ri = np.linspace(0.0, wr.max() * 1.001, nr)
    Xg, Rg = np.meshgrid(xi, ri)
    Vg = griddata((x, r), vals, (Xg, Rg), method="linear")
    wall_at = np.interp(Xg, wx, wr)
    Vg = np.where(Rg <= wall_at, Vg, np.nan)
    return Xg, Rg, Vg


def _field_coverage_report(
    Xg, Rg, field, wx, wr, *, min_radial_fraction=0.98,
    min_resolved_axial_fraction=0.70,
):
    """Measure contiguous wall-to-axis coverage through the exit."""
    inside = Rg <= np.interp(Xg, wx, wr) + 1e-12
    finite = np.isfinite(field) & inside
    counts = np.count_nonzero(inside, axis=0)
    radial_fraction = np.divide(
        np.count_nonzero(finite, axis=0), np.maximum(counts, 1),
    )
    axis_finite = finite[0, :]
    full_column = (radial_fraction >= min_radial_fraction) & axis_finite
    tail_is_full = np.logical_and.accumulate(full_column[::-1])[::-1]
    candidates = np.flatnonzero(tail_is_full)
    start_index = int(candidates[0]) if candidates.size else None
    x0 = float(Xg[0, 0])
    x1 = float(Xg[0, -1])
    if start_index is None or x1 <= x0:
        resolved_fraction = 0.0
        resolved_x_min = None
    else:
        resolved_x_min = float(Xg[0, start_index])
        resolved_fraction = float((x1 - resolved_x_min) / (x1 - x0))
    passes = bool(
        start_index is not None
        and resolved_fraction >= min_resolved_axial_fraction
        and full_column[-1]
    )
    return {
        "passes": passes,
        "overall_fraction": float(
            np.count_nonzero(finite) / max(np.count_nonzero(inside), 1)
        ),
        "axis_column_fraction": float(np.mean(axis_finite)),
        "exit_radial_fraction": float(radial_fraction[-1]),
        "resolved_x_min_m": resolved_x_min,
        "resolved_axial_fraction": resolved_fraction,
        "minimum_radial_fraction": float(min_radial_fraction),
        "minimum_resolved_axial_fraction": float(
            min_resolved_axial_fraction
        ),
        "radial_fraction_by_column": radial_fraction,
    }


def _bde_support_report(solution, exit_x):
    """Verify that the actual BDE construction reaches the exit plane."""
    kernel, bde = _bde_artifacts(solution)
    if kernel is None or bde is None:
        return {"available": False, "passes": True}
    full_rows = list(getattr(bde, "full_grid_rows", ()) or ())
    iD = int(getattr(bde, "iD", 0) or 0)
    frontiers = [
        float(_row_points(row)[-1].x)
        for row in full_rows if len(_row_points(row)) > iD
    ]
    axis_x = [
        float(_row_points(row)[-1].x)
        for row in (getattr(kernel, "rrcs", ()) or ()) if _row_points(row)
    ]
    diagnostics = getattr(solution, "construction_diagnostics", {})
    net_report = (
        diagnostics.get("net_report", {})
        if isinstance(diagnostics, dict) else {}
    )
    wall_complete = bool(getattr(bde, "wall_contour_complete", False))
    physical_complete = bool(
        net_report.get("bde_physical_mesh_complete", wall_complete)
    )
    frontier_min = min(frontiers) if frontiers else float("nan")
    continuation_to_exit = bool(
        getattr(bde, "complete_remaining_mesh", False)
        or (np.isfinite(frontier_min) and frontier_min >= exit_x - 1e-9)
    )
    kernel_axis_to_exit = bool(axis_x and max(axis_x) >= exit_x - 1e-9)
    passes = bool(
        wall_complete and physical_complete and full_rows
        and continuation_to_exit and kernel_axis_to_exit
    )
    return {
        "available": True,
        "passes": passes,
        "wall_contour_complete": wall_complete,
        "physical_bde_complete": physical_complete,
        "continuation_complete_to_axis": bool(
            getattr(bde, "complete_remaining_mesh", False)
        ),
        "continuation_frontier_min_x_m": (
            frontier_min if np.isfinite(frontier_min) else None
        ),
        "continuation_reaches_exit": continuation_to_exit,
        "kernel_axis_reaches_exit": kernel_axis_to_exit,
    }


def _axisymmetric_exit_mean(Rg, Mg) -> float:
    """Area-weighted Mach on the last populated axial grid cut."""
    for column in range(Mg.shape[1] - 1, -1, -1):
        radius = np.asarray(Rg[:, column], dtype=float)
        mach = np.asarray(Mg[:, column], dtype=float)
        finite = np.isfinite(radius) & np.isfinite(mach)
        if np.count_nonzero(finite) < 2:
            continue
        radius = radius[finite]
        mach = mach[finite]
        denominator = float(np.trapezoid(radius, radius))
        if denominator > 0.0:
            return float(np.trapezoid(mach * radius, radius) / denominator)
        return float(np.mean(mach))
    return float("nan")


# --------------------------------------------------------------------------- #
# The composite static flow-field figure                                       #
# --------------------------------------------------------------------------- #
def plot_flowfield(
    solution,
    gamma: float = 1.4,
    *,
    Tc: float | None = None,
    exit_mach: float | None = None,
    show_characteristics: bool = True,
    show_streamlines: bool = True,
    n_streamlines: int = 11,
    nx: int = 320,
    nr: int = 120,
    save_path: str | None = None,
    show: bool = False,
    allow_partial: bool = False,
):
    """Render the resolved steady supersonic MOC field.

    Mach, pressure, flow angle, and temperature are interpolated from the
    solved nodes, with characteristic lines and streamlines mirrored about
    the axis.  By default a partial construction is rejected instead of being
    labelled as a full steady field.  ``allow_partial=True`` is an explicit
    diagnostic override and changes the figure title accordingly.

    ``Tc`` gives absolute temperatures [K]; otherwise the temperature
    panel is the stagnation ratio T/T0.  ``exit_mach`` should be the design
    exit value when available; otherwise an axisymmetric area-weighted value
    is measured from the final populated field cut.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, r, M, theta, wx, wr = _field_arrays(solution, gamma, anchor=True)

    Xg, Rg, Mg = _grid_field(x, r, M, wx, wr, nx=nx, nr=nr)
    _, _, Thetag = _grid_field(x, r, theta, wx, wr, nx=nx, nr=nr)
    coverage = _field_coverage_report(Xg, Rg, Mg, wx, wr)
    bde_support = _bde_support_report(solution, float(wx.max()))
    full_field = bool(coverage["passes"] and bde_support["passes"])
    if not full_field and not allow_partial:
        if save_path is not None:
            from pathlib import Path
            Path(save_path).unlink(missing_ok=True)
        raise ValueError(
            "partial MOC coverage: refusing to label it as a steady nozzle "
            "flow field "
            f"(overall={100.0 * coverage['overall_fraction']:.1f}%, "
            f"exit wall-to-axis={100.0 * coverage['exit_radial_fraction']:.1f}%, "
            f"axis columns={100.0 * coverage['axis_column_fraction']:.1f}%)"
        )

    Pg = isentropic_pressure_ratio(np.clip(Mg, 1.0, None), gamma)
    Tg = isentropic_temperature_ratio(np.clip(Mg, 1.0, None), gamma)
    if Tc is not None:
        Tg = Tg * Tc
    resolved_x_min = coverage["resolved_x_min_m"]
    streams = (
        _streamlines(
            x, r, theta, wx, wr, n_lines=n_streamlines,
            x_start=resolved_x_min,
        )
        if show_streamlines else []
    )

    fig, axes_grid = plt.subplots(
        2, 2, figsize=(12, 8.5), sharex=True, sharey=True,
    )
    axes = axes_grid.ravel()
    theta_deg = np.degrees(Thetag)
    theta_limit = max(float(np.nanmax(np.abs(theta_deg))), 1e-6)
    panels = [
        (axes[0], Mg, "turbo", "Mach number", False, None),
        (axes[1], Pg, "viridis", "Pressure ratio  p/p0", False, None),
        (axes[2], theta_deg, "coolwarm", "Flow angle  θ [deg]", True,
         np.linspace(-theta_limit, theta_limit, 31)),
        (axes[3], Tg, "inferno",
         "Temperature" + (" [K]" if Tc else "  T/T0"), False, None),
    ]
    for ax, field, cmap, label, mirror_sign, explicit_levels in panels:
        finite_values = np.asarray(field)[np.isfinite(field)]
        if not finite_values.size:
            raise ValueError(f"{label} field has no finite values")
        if explicit_levels is not None:
            levels = explicit_levels
        else:
            low = float(np.min(finite_values))
            high = float(np.max(finite_values))
            if high - low <= 1e-12 * max(abs(low), abs(high), 1.0):
                high = low + 1e-9
            levels = np.linspace(low, high, 30)
        for sgn in (1.0, -1.0):                     # mirror about the axis
            plotted = sgn * field if mirror_sign else field
            cf = ax.contourf(Xg, sgn * Rg, plotted, levels=levels, cmap=cmap,
                             extend="both")
        cb = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.05)
        cb.set_label(label)
        # wall (both halves) + axis
        ax.plot(wx, wr, color="k", lw=1.6); ax.plot(wx, -wr, color="k", lw=1.6)
        ax.axhline(0.0, color="0.35", lw=0.5, ls=":")
        if show_streamlines:
            for xs, rs in streams:
                ax.plot(xs, rs, color="w", lw=0.7, alpha=0.8)
                ax.plot(xs, -rs, color="w", lw=0.7, alpha=0.8)
        ax.set_ylabel("r [m]")
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")

    # characteristic lines (the Mach waves / expansion fan) on the Mach
    # panel — clipped to physical points (a seed march can emit nodes
    # past the exit).
    x_lo, x_hi = float(wx.min()), float(wx.max())
    if show_characteristics:
        all_lines = _characteristic_polylines(solution)
        family_counts = {
            family: sum(kind == family for kind, _ in all_lines)
            for family, _ in all_lines
        }
        family_seen = {family: 0 for family in family_counts}
        for family, pts in all_lines:
            line_index = family_seen[family]
            family_seen[family] += 1
            stride = max(1, int(np.ceil(family_counts[family] / 36.0)))
            if (line_index % stride != 0
                    and line_index != family_counts[family] - 1):
                continue
            seg = [(p.x, p.r) for p in pts
                   if x_lo - 1e-9 <= p.x <= x_hi + 1e-9
                   and 0.0 <= p.r <= np.interp(p.x, wx, wr) * 1.05]
            if len(seg) < 2:
                continue
            px = [s[0] for s in seg]; pr = [s[1] for s in seg]
            color = "white" if family == "lrc" else "black"
            alpha = 0.24 if family == "lrc" else 0.25
            axes[0].plot(px, pr, color=color, lw=0.25, alpha=alpha)
            axes[0].plot(px, [-v for v in pr], color=color, lw=0.25,
                         alpha=alpha)

    for ax in axes:                                  # frame the nozzle
        ax.set_xlim(x_lo - 0.03 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
        ax.set_ylim(-1.15 * float(wr.max()), 1.15 * float(wr.max()))

    measured_exit_mach = (
        _axisymmetric_exit_mean(Rg, Mg) if coverage["passes"] else float("nan")
    )
    title_parts = [
        "Steady supersonic MOC field" if full_field
        else "Partial MOC construction field",
    ]
    if resolved_x_min is not None:
        title_parts.append(
            f"wall-to-axis coverage from x={1e3 * resolved_x_min:.1f} mm"
        )
    if exit_mach is not None and np.isfinite(exit_mach):
        title_parts.append(f"design exit M≈{float(exit_mach):.2f}")
    if np.isfinite(measured_exit_mach):
        title_parts.append(f"resolved exit ⟨M⟩A≈{measured_exit_mach:.2f}")
    if hasattr(solution, "thrust_coefficient"):
        title_parts.append(f"Cf={solution.thrust_coefficient:.3f}")
    fig.suptitle("   |   ".join(title_parts), fontsize=11)
    bottom = 0.045 if (
        resolved_x_min is not None and resolved_x_min > x_lo + 1e-9
    ) else 0.0
    if bottom:
        fig.text(
            0.5, 0.012,
            "Blank throat wedge: transonic core upstream of the first "
            "supersonic characteristic; no flow state is fabricated there.",
            ha="center", va="bottom", fontsize=8, color="0.25",
        )
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.96))
    fig.flowfield_coverage = {
        **{key: value for key, value in coverage.items()
           if key != "radial_fraction_by_column"},
        "bde_support": bde_support,
        "full_field": full_field,
    }

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
    for family, pts in _characteristic_polylines(solution):
        if family == "lrc":
            continue
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
    allow_partial: bool = False,
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
    th_i = LinearNDInterpolator(pts, theta, fill_value=np.nan)
    M_i = LinearNDInterpolator(pts, M, fill_value=np.nan)
    x_lo, x_hi = float(wx.min()), float(wx.max())
    wall_at = lambda xx: np.interp(xx, wx, wr)

    Xg, Rg, Mg = _grid_field(x, r, M, wx, wr, nx=240, nr=90)
    coverage = _field_coverage_report(Xg, Rg, Mg, wx, wr)
    bde_support = _bde_support_report(solution, x_hi)
    full_field = bool(coverage["passes"] and bde_support["passes"])
    if not full_field and not allow_partial:
        raise ValueError(
            "partial MOC coverage: particle advection requires an explicit "
            "allow_partial=True diagnostic override"
        )

    rng = np.random.default_rng(0)
    resolved_x_min = coverage["resolved_x_min_m"]
    x_seed = (
        float(resolved_x_min) if resolved_x_min is not None else x_lo
    ) + 0.005 * (x_hi - x_lo)

    def seed(n):
        fr = rng.uniform(0.03, 0.95, n)
        return np.column_stack([np.full(n, x_seed), fr * float(wall_at(x_seed))])

    P = seed(n_particles)
    # speed ∝ local gas velocity M·sqrt(T); normalise so a fast particle
    # crosses the nozzle in ~half the frames.
    def speed(P):
        m = np.asarray(M_i(P[:, 0], P[:, 1]), dtype=float)
        m = np.where(np.isfinite(m), np.clip(m, 1.0, None), 1.0)
        T = isentropic_temperature_ratio(m, gamma)
        return m * np.sqrt(T)
    v0 = float(np.nanmean(speed(P)))
    dt = (x_hi - x_lo) / (0.55 * n_frames * max(v0, 1e-6))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for sgn in (1.0, -1.0):
        ax.contourf(Xg, sgn * Rg, Mg, levels=24, cmap="turbo", alpha=0.30)
    ax.plot(wx, wr, "k", lw=1.6); ax.plot(wx, -wr, "k", lw=1.6)
    ax.set_xlim(x_lo - 0.03 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
    ax.set_ylim(-1.15 * float(wr.max()), 1.15 * float(wr.max()))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("r [m]")
    ax.set_title(
        "Particle advection through the resolved MOC field"
        if full_field else "Particle advection through a partial MOC construction",
        fontsize=11,
    )
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
        th = np.asarray(th_i(P[:, 0], P[:, 1]), dtype=float)
        mach = np.asarray(M_i(P[:, 0], P[:, 1]), dtype=float)
        invalid = ~np.isfinite(th) | ~np.isfinite(mach)
        if invalid.any():
            P[invalid] = seed(int(invalid.sum()))
            th[invalid] = np.asarray(
                th_i(P[invalid, 0], P[invalid, 1]), dtype=float,
            )
            mach[invalid] = np.asarray(
                M_i(P[invalid, 0], P[invalid, 1]), dtype=float,
            )
        # A failed seed is held and retried next frame; it is never assigned a
        # fabricated zero flow angle.
        movable = np.isfinite(th) & np.isfinite(mach)
        v = speed(P)
        direction = np.zeros_like(P)
        direction[movable] = np.column_stack([
            np.cos(th[movable]), np.sin(th[movable]),
        ])
        P = P + dt * v[:, None] * direction
        P[:, 1] = np.clip(P[:, 1], 0.0, None)
        # re-seed particles that left the nozzle (exit or through the wall)
        field_valid = np.isfinite(M_i(P[:, 0], P[:, 1]))
        gone = ((P[:, 0] >= x_hi)
                | (P[:, 1] > wall_at(P[:, 0]) * 1.02)
                | ~field_valid)
        if gone.any():
            P[gone] = seed(int(gone.sum()))
        m = np.clip(M_i(P[:, 0], P[:, 1]), 1.0, None)
        sc.set_offsets(P); sc.set_array(m)
        sc2.set_offsets(np.column_stack([P[:, 0], -P[:, 1]])); sc2.set_array(m)
        return sc, sc2

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=True)
    return _save_or_show(anim, fig, save_path, show, fps)


def _obj_value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _obj_first(obj, names, default=None):
    for name in names:
        value = _obj_value(obj, name, None)
        if value is not None:
            return value
    return default


def animate_pump_particles(
    pump_result,
    *,
    role: str = "fuel",
    n_particles: int = 180,
    n_frames: int = 220,
    interval: int = 40,
    fps: int = 25,
    save_path: str | None = None,
    show: bool = False,
):
    """Animate a simplified electric-pump flow path.

    This is a sizing visualization, not CFD: particles enter axially through
    the inducer, pick up swirl, accelerate radially across the impeller, then
    slow through the diffuser/volute collection path.
    """
    import matplotlib
    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    lines = _obj_value(pump_result, "lines", {})
    if role not in lines:
        role = next(iter(lines))
    line = lines[role]
    imp = _obj_value(line, "impeller")
    ind = _obj_value(line, "inducer")
    dif = _obj_value(line, "diffuser_volute")
    if imp is None:
        raise ValueError("pump visualization needs a line with impeller geometry")

    d2 = float(_obj_first(imp, ("impeller_diameter", "impeller_diameter_m"), 0.05)
               or 0.05)
    d1 = float(_obj_first(imp, ("inlet_diameter", "inlet_diameter_m"), 0.35 * d2)
               or 0.35 * d2)
    r2 = max(0.5 * d2, 1e-4)
    r1 = max(0.5 * d1, 0.18 * r2)
    inducer_r = max(
        0.5 * float(_obj_first(ind, ("diameter", "diameter_m"), d1) or d1),
        r1,
    )
    volute_r = 1.55 * r2
    diffuser_r = 1.25 * r2
    entry_len = 1.35 * r2
    rng = np.random.default_rng(2)
    phase0 = rng.random(n_particles)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    lane = rng.normal(0.0, 0.18 * inducer_r, n_particles)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-entry_len * 1.20, volute_r * 1.35)
    ax.set_ylim(-volute_r * 1.25, volute_r * 1.25)
    ax.set_xlabel("axial entry / radial pump plane [m]")
    ax.set_ylabel("pump radius [m]")
    ax.set_title(f"{role} electric pump particle path", fontsize=11)

    # Axial inlet/inducer sketch.
    ax.plot([-entry_len, 0.0], [inducer_r, inducer_r], color="0.25", lw=1.1)
    ax.plot([-entry_len, 0.0], [-inducer_r, -inducer_r], color="0.25", lw=1.1)
    ax.fill_between([-entry_len, 0.0], -inducer_r, inducer_r,
                    color="tab:blue", alpha=0.06)
    # Impeller, diffuser, and volute.
    for radius, color, lw, ls in (
        (r1, "0.25", 1.0, "--"),
        (r2, "0.05", 1.8, "-"),
        (diffuser_r, "tab:green", 1.2, "--"),
        (volute_r, "tab:purple", 1.4, "-"),
    ):
        ax.add_patch(plt.Circle((0.0, 0.0), radius, fill=False,
                                color=color, lw=lw, ls=ls, alpha=0.85))
    blades = int(_obj_value(imp, "blade_count", 6) or 6)
    for k in range(max(3, blades)):
        th = 2.0 * np.pi * k / max(3, blades)
        rr = np.linspace(r1, r2, 80)
        tt = th + 0.75 * (rr - r1) / max(r2 - r1, 1e-9)
        ax.plot(rr * np.cos(tt), rr * np.sin(tt),
                color="0.15", lw=0.8, alpha=0.45)
    ax.text(-entry_len * 0.95, inducer_r * 1.18, "inducer", fontsize=9)
    ax.text(r2 * 0.15, -r2 * 0.20, "impeller", fontsize=9)
    ax.text(diffuser_r * 0.70, diffuser_r * 0.22,
            _obj_value(dif, "selection", "diffuser/volute"), fontsize=9)

    sc = ax.scatter([], [], s=14, c=[], cmap="plasma", vmin=0.0, vmax=1.0,
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.045,
                 label="normalized particle energy")

    def positions(frame):
        p = (phase0 + frame / n_frames) % 1.0
        x = np.empty(n_particles)
        y = np.empty(n_particles)
        energy = np.empty(n_particles)

        m = p < 0.25
        s = p[m] / 0.25
        x[m] = -entry_len * (1.0 - s)
        y[m] = lane[m] + 0.10 * inducer_r * np.sin(8.0 * np.pi * s + theta0[m])
        energy[m] = 0.15 + 0.20 * s

        m = (p >= 0.25) & (p < 0.62)
        s = (p[m] - 0.25) / 0.37
        rr = r1 + (r2 - r1) * s**1.35
        th = theta0[m] + 4.8 * np.pi * s
        x[m] = rr * np.cos(th)
        y[m] = rr * np.sin(th)
        energy[m] = 0.35 + 0.55 * s

        m = (p >= 0.62) & (p < 0.82)
        s = (p[m] - 0.62) / 0.20
        rr = r2 + (diffuser_r - r2) * s
        th = theta0[m] + 4.8 * np.pi + 0.65 * np.pi * s
        x[m] = rr * np.cos(th)
        y[m] = rr * np.sin(th)
        energy[m] = 0.90 - 0.20 * s

        m = p >= 0.82
        s = (p[m] - 0.82) / 0.18
        th = theta0[m] + 5.45 * np.pi + 1.3 * np.pi * s
        rr = diffuser_r + (volute_r - diffuser_r) * s
        x[m] = rr * np.cos(th) + 0.18 * r2 * s
        y[m] = rr * np.sin(th)
        energy[m] = 0.70 - 0.18 * s
        return np.column_stack([x, y]), energy

    def update(frame):
        xy, energy = positions(frame)
        sc.set_offsets(xy)
        sc.set_array(energy)
        return (sc,)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=True)
    return _save_or_show(anim, fig, save_path, show, fps)
