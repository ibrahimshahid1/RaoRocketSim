"""injector_plots.py - pintle injector diagnostic figures.

A full diagnostic set drawn from the sized :class:`InjectorDesignResult`:
cross-section, spray envelope, hydraulics, atomization/combustion development,
face/tip thermal stack, stability, manifold maldistribution, the gate
scorecard, and (optionally) the throttle map.  Kept separate from the
pure-numeric solver, matching the repo's plotting/solver split.
"""

from __future__ import annotations

import math
import os

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

_STATUS_COLOR = {"pass": "#31a354", "info": "#9e9e9e",
                 "warn": "#f0a30a", "fail": "#de2d26"}


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _metal(ax, xy, w, h, **kw):
    ax.add_patch(Rectangle(xy, w, h, facecolor="0.75", edgecolor="0.3",
                           hatch="////", linewidth=1.0, **kw))


def plot_pintle_cross_section(inj, *, show=False, save_path=None) -> "plt.Figure":
    """CAD-oriented meridional section of the sized pintle injector."""
    return plot_pintle_schematic(inj, show=show, save_path=save_path)


def plot_spray_envelope(inj, *, show=False, save_path=None) -> "plt.Figure":
    """Spray cone from the pintle tip against the chamber wall, with the
    interception point and clearance status."""
    Rc = inj.chamber_radius
    Lc = inj.chamber_length
    Dp = inj.pintle_diameter
    half = inj.spray_half_angle_deg
    wall = inj.spray_wall_axial_distance
    mm = 1e3
    x_tip = 1.5 * Dp                          # tip protrusion (schematic origin)

    fig, ax = plt.subplots(figsize=(8, 5))
    # chamber box
    ax.plot([0, Lc * mm], [Rc * mm, Rc * mm], "k-", lw=2)
    ax.plot([0, Lc * mm], [-Rc * mm, -Rc * mm], "k-", lw=2)
    ax.plot([0, 0], [-Rc * mm, Rc * mm], "k-", lw=2)        # injector face
    ax.axhline(0, color="0.6", lw=0.6, ls="--")
    # pintle post + tip
    ax.add_patch(Rectangle((0, -0.5 * Dp * mm), x_tip * mm, Dp * mm,
                           facecolor="0.8", edgecolor="0.3"))

    # spray cone from the tip at +/- half-angle
    if math.isfinite(wall) and 0.0 < half < 90.0:
        x_hit = min((x_tip + wall), Lc)
        r_hit = min(0.5 * Dp + (x_hit - x_tip) * math.tan(math.radians(half)),
                    Rc)
        hits_wall = (x_tip + wall) <= Lc
        frac = wall / Lc if Lc > 0 else float("inf")
        col = ("#31a354" if hits_wall and frac >= 0.05 else "#de2d26")
        for sgn in (1, -1):
            ax.add_patch(Polygon(
                [[x_tip * mm, sgn * 0.5 * Dp * mm],
                 [x_hit * mm, sgn * r_hit * mm],
                 [x_hit * mm, 0]], closed=True, facecolor=col, alpha=0.25,
                edgecolor=col, linewidth=1.2))
            ax.plot([x_tip * mm, x_hit * mm],
                    [sgn * 0.5 * Dp * mm, sgn * r_hit * mm], color=col, lw=1.5)
        if hits_wall:
            ax.plot([x_hit * mm], [r_hit * mm], "o", color=col)
            ax.plot([x_hit * mm], [-r_hit * mm], "o", color=col)
            label = (f"wall interception @ {wall*mm:.0f} mm "
                     f"({frac*100:.0f}% of Lc)")
            if frac < 0.05:
                label += "  ⚠ gouging risk"
        else:
            label = "spray does not reach the wall within the chamber"
    else:
        col = "#de2d26"
        label = "spray nearly axial / reversed"
    ax.text(0.02 * Lc * mm, Rc * 0.88 * mm,
            f"half-angle {half:.0f}°,  TMR {inj.total_momentum_ratio:.2f}\n{label}",
            fontsize=9, va="top")

    ax.set_xlabel("axial x [mm]")
    ax.set_ylabel("radius r [mm]")
    ax.set_title("Pintle spray envelope vs chamber wall")
    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * Lc * mm, Lc * 1.03 * mm)
    ax.set_ylim(-Rc * 1.2 * mm, Rc * 1.2 * mm)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def _streams_ordered(inj):
    """(fuel, oxidizer) StreamResults with display labels."""
    f, o = inj.streams["fuel"], inj.streams["oxidizer"]
    return [(f, f"fuel\n({f.geometry})"), (o, f"oxidizer\n({o.geometry})")]


def plot_injector_hydraulics(inj, *, show=False, save_path=None):
    """Per-stream areas, velocities, and Re/We/Oh."""
    rows = _streams_ordered(inj)
    labels = [lab for _, lab in rows]
    streams = [s for s, _ in rows]
    colors = ["#e6550d", "#3182bd"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    def bar(ax, vals, title, ylabel, log=False, fmt="{:.0f}"):
        x = np.arange(len(vals))
        b = ax.bar(x, vals, color=colors, edgecolor="0.3")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        for xi, v in zip(x, vals):
            if v == v and v > 0:
                ax.text(xi, v, fmt.format(v), ha="center", va="bottom",
                        fontsize=8)
        return b

    bar(axes[0, 0], [s.area * 1e6 for s in streams],
        "Flow area", "mm²", fmt="{:.1f}")
    bar(axes[0, 1], [s.velocity for s in streams],
        "Injection velocity", "m/s")
    bar(axes[1, 0], [s.reynolds for s in streams],
        "Reynolds number", "Re", log=True, fmt="{:.0f}")
    we = [s.weber if s.weber == s.weber else 0.0 for s in streams]
    bar(axes[1, 1], we, "Weber number", "We", log=True, fmt="{:.0f}")
    fig.suptitle(
        f"Pintle hydraulics   TMR={inj.total_momentum_ratio:.2f}   "
        f"slot/annulus width={inj.slot_to_annulus_width_ratio:.2f}",
        fontsize=11)
    return _finish(fig, save_path, show)


def plot_atomization(inj, *, show=False, save_path=None):
    """SMD per stream and the combustion-development length vs the chamber."""
    at = inj.atomization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    roles = [r for r in ("fuel", "oxidizer") if at.streams[r].applicable]
    color_by_role = {"fuel": "#e6550d", "oxidizer": "#3182bd"}
    colors = [color_by_role[r] for r in roles]
    if not roles:
        reasons = "\n".join(
            f"{r}: {at.streams[r].validity_reason}"
            for r in ("fuel", "oxidizer")
        )
        for ax in (ax1, ax2):
            ax.axis("off")
        ax1.text(
            0.02, 0.98,
            "Legacy liquid-droplet screen not applicable\n\n" + reasons,
            transform=ax1.transAxes, va="top", wrap=True, fontsize=9,
        )
        fig.suptitle("Spray atomization / vaporization applicability", fontsize=11)
        return _finish(fig, save_path, show)
    smd = [at.streams[r].sauter_mean_diameter * 1e6 for r in roles]
    x = np.arange(len(roles))
    ax1.bar(x, smd, color=colors, edgecolor="0.3")
    ax1.set_xticks(x); ax1.set_xticklabels(roles)
    ax1.set_ylabel("SMD d₃₂  [µm]")
    ax1.set_title("Sauter mean diameter (Hinze screen)")
    for xi, v in zip(x, smd):
        ax1.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    # stacked breakup + vaporization length vs chamber
    brk = [at.streams[r].breakup_length * 1e3 for r in roles]
    vap = [at.streams[r].vaporization_length * 1e3 for r in roles]
    ax2.bar(x, brk, color="#fdae6b", edgecolor="0.3", label="primary breakup")
    ax2.bar(x, vap, bottom=brk, color="#9ecae1", edgecolor="0.3",
            label="vaporization (→99%)")
    Lc = at.available_chamber_length * 1e3
    ax2.axhline(Lc, color="k", ls="--", lw=1.5,
                label=f"chamber L={Lc:.0f} mm")
    ax2.set_xticks(x); ax2.set_xticklabels(roles)
    ax2.set_ylabel("axial length  [mm]")
    ax2.set_title("Combustion-development length")
    ax2.set_ylim(0, max(Lc, max((b + v) for b, v in zip(brk, vap))) * 1.25)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.text(0.5, 0.62,
             f"limiting: {at.limiting_role}\nmargin {at.development_margin:.2f}\n"
             f"eta_vaporization≈{at.eta_vaporization:.2f}\n"
             "eta_cstar unresolved",
             transform=ax2.transAxes, ha="center", va="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fff7bc", ec="0.6"))
    fig.suptitle("Spray atomization / vaporization (screening)", fontsize=11)
    return _finish(fig, save_path, show)


def plot_face_tip_thermal(inj, *, show=False, save_path=None):
    """Gas → wall → coolant temperature stack for the pintle tip and face."""
    t = inj.thermal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    surfaces = [("tip", t.tip_wall_temperature, inj.feed[inj.slots.role],
                 t.tip_margin),
                ("face", t.face_wall_temperature, inj.feed[inj.annulus.role],
                 t.face_margin)]
    x = np.arange(2)
    T_aw = t.recovery_temperature
    Twg = [s[1] for s in surfaces]
    Tcool = [s[2].temperature for s in surfaces]
    ax1.bar(x, [T_aw] * 2, color="#fdd0a2", edgecolor="0.4",
            label="recovery gas T_aw")
    ax1.bar(x, Twg, color="#de2d26", edgecolor="0.4", label="wall T_wg")
    ax1.bar(x, Tcool, color="#3182bd", edgecolor="0.4", label="coolant T")
    ax1.axhline(t.wall_temperature_limit, color="k", ls="--", lw=1.5,
                label=f"limit {t.wall_temperature_limit:.0f} K")
    ax1.set_xticks(x); ax1.set_xticklabels([s[0] for s in surfaces])
    ax1.set_ylabel("temperature  [K]")
    ax1.set_title("Gas / wall / coolant temperatures")
    ax1.legend(fontsize=8)
    for xi, (name, twg, _, mg) in zip(x, surfaces):
        ax1.text(xi, twg, f"T_wg {twg:.0f} K\nmargin {mg:.2f}", ha="center",
                 va="bottom", fontsize=8)

    q = [t.tip_heat_flux / 1e6, t.face_heat_flux / 1e6]
    ax2.bar(x, q, color=["#756bb1", "#9e9ac8"], edgecolor="0.3")
    ax2.set_xticks(x); ax2.set_xticklabels([s[0] for s in surfaces])
    ax2.set_ylabel("heat flux  [MW/m²]")
    ax2.set_title("Incident heat flux")
    for xi, v in zip(x, q):
        ax2.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Face / pintle-tip thermal screen   "
                 f"({t.limiting} governs, margin {t.governing_margin:.2f})",
                 fontsize=11)
    return _finish(fig, save_path, show)


def plot_stability(inj, *, show=False, save_path=None):
    """Chamber acoustic modes, chug decoupling, and the n-τ band."""
    st = inj.stability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))
    modes = ["L1", "L2", "T1", "R1"]
    freqs = [st.f_L1, st.f_L2, st.f_T1, st.f_R1]
    x = np.arange(4)
    ax1.bar(x, freqs, color="#54278f", edgecolor="0.3")
    ax1.set_xticks(x); ax1.set_xticklabels(modes)
    ax1.set_ylabel("frequency  [Hz]")
    ax1.set_title(f"Chamber acoustic modes (a={st.sound_speed:.0f} m/s)")
    for xi, v in zip(x, freqs):
        ax1.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    chi = st.injector_decoupling_fraction
    col = ("#31a354" if chi >= 0.2 else "#f0a30a" if chi >= 0.1 else "#de2d26")
    ax2.barh([0.4], [chi], color=col, edgecolor="0.3", height=0.35)
    ax2.axvline(0.20, color="0.3", ls="--", lw=1)
    ax2.axvline(0.10, color="0.6", ls=":", lw=1)
    ax2.text(0.205, 0.85, "good ≥0.2", fontsize=8, va="center")
    ax2.text(0.105, 0.85, "chug <0.1", fontsize=8, va="center", ha="right")
    ax2.text(chi, 0.4, f" min(χ)={chi:.2f}", fontsize=8, va="center")
    ax2.set_ylim(-1.2, 1.1)
    ax2.set_yticks([]); ax2.set_xlim(0, max(0.35, chi * 1.25))
    ax2.set_xlabel("injector decoupling  min(χ)")
    ax2.set_title("Feed-system chug margin")
    ntxt = (f"combustion lag τ ≈ {st.combustion_time_lag*1e3:.2f} ms\n"
            f"τ·f_L1 = {st.reduced_frequency_L1:.2f}  "
            + ("(SENSITIVE n-τ band)" if st.sensitive_band else "(outside band)"))
    ax2.text(0.5, -0.6, ntxt, transform=ax2.get_yaxis_transform(),
             ha="center", va="center", fontsize=8,
             bbox=dict(boxstyle="round", fc="#fff7bc", ec="0.6"))
    fig.suptitle(f"Stability screen   chug: {st.chug_status}", fontsize=11)
    return _finish(fig, save_path, show)


def plot_manifold(inj, *, show=False, save_path=None):
    """Per-manifold element-to-element flow maldistribution."""
    m = inj.manifold
    fig, ax = plt.subplots(figsize=(8, 4.6))
    roles = ["fuel", "oxidizer"]
    colors = ["#e6550d", "#3182bd"]
    x = np.arange(2)
    spreads = []
    for r in roles:
        mr = m.streams[r]
        s = mr.maldistribution_fraction
        spreads.append(s if s == s else 0.0)
    ax.bar(x, [s * 100 for s in spreads], color=colors, edgecolor="0.3")
    ax.axhline(10, color="0.3", ls="--", lw=1, label="10% (good)")
    ax.axhline(25, color="0.6", ls=":", lw=1, label="25% (limit)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n({m.streams[r].feeds}, "
                        f"{m.streams[r].port_count} ports)" for r in roles])
    ax.set_ylabel("element flow spread  [%]")
    ax.set_title("Manifold maldistribution (annular two-header network)")
    ax.legend(fontsize=8)
    for xi, r, s in zip(x, roles, spreads):
        mr = m.streams[r]
        ax.text(xi, s * 100, f"{s*100:.1f}%\n[{mr.min_flow_ratio:.2f}–"
                f"{mr.max_flow_ratio:.2f}×]", ha="center", va="bottom",
                fontsize=8)
    return _finish(fig, save_path, show)


def plot_injector_gates(inj, *, show=False, save_path=None):
    """Scorecard of every injector gate, colored by status."""
    gates = list(inj.gates)
    n = len(gates)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.34 * n + 1)))
    counts = {k: sum(g.status == k for g in gates)
              for k in ("pass", "warn", "fail", "info")}
    for i, g in enumerate(reversed(gates)):
        y = i
        ax.scatter([0], [y], s=130, color=_STATUS_COLOR.get(g.status, "0.5"),
                   edgecolor="0.3", zorder=3)
        ax.text(0.4, y, g.name, va="center", fontsize=8, fontweight="bold")
        detail = g.detail if len(g.detail) <= 78 else g.detail[:75] + "…"
        ax.text(0.4, y - 0.32, detail, va="center", fontsize=6.5, color="0.35")
    ax.set_xlim(-0.5, 10); ax.set_ylim(-1, n)
    ax.axis("off")
    verdict = "FEASIBLE" if inj.feasible else "INFEASIBLE"
    vcol = "#31a354" if inj.feasible else "#de2d26"
    ax.set_title(
        f"Injector gate scorecard — {verdict}   "
        f"({counts['pass']} pass, {counts['warn']} warn, "
        f"{counts['fail']} fail, {counts['info']} info)",
        color=vcol, fontsize=11)
    return _finish(fig, save_path, show)


def plot_throttle_map(tm, *, show=False, save_path=None):
    """Throttle curves with architecture-appropriate command semantics."""
    pts = tm.points
    f = [p.throttle for p in pts]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(f, [p.v_annulus for p in pts], "o-", label="annulus")
    axes[0, 0].plot(f, [p.v_slots for p in pts], "s-", label="radial")
    axes[0, 0].set_ylabel("injection velocity [m/s]"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Velocity")
    axes[0, 1].plot(f, [p.total_momentum_ratio for p in pts], "o-",
                    color="#756bb1")
    axes[0, 1].set_ylabel("TMR"); axes[0, 1].set_title("Total momentum ratio")
    axes[1, 0].plot(f, [p.smd_limiting * 1e6 for p in pts], "o-",
                    color="#e6550d")
    axes[1, 0].set_ylabel("SMD [µm]"); axes[1, 0].set_title("Atomization (SMD)")
    axes[1, 1].plot(f, [p.eta_vaporization for p in pts], "o-",
                    color="#31a354", label="eta_vaporization")
    if tm.kinematic_model is not None:
        axes[1, 1].plot(
            f, [p.actuator_stroke_fraction for p in pts], "s--",
            color="0.4", label="physical L_open/L_stop",
        )
        axes[1, 1].plot(
            f, [p.required_axial_controller_dp_fraction for p in pts], "^:",
            color="#636363", label="upstream axial controller dP/Pc",
        )
        command_title = "Vaporization, travel, and axial controller"
        figure_title = "Fixed-hardware Son movable-pintle throttle map"
    else:
        axes[1, 1].plot(
            f, [p.annulus_area_command_fraction for p in pts], "s--",
            color="0.4", label="annulus area command",
        )
        axes[1, 1].plot(
            f, [p.slot_area_command_fraction for p in pts], "^:",
            color="#636363", label="radial area command",
        )
        command_title = "Vaporization screen and area commands"
        figure_title = "Commanded effective-area throttle map"
    axes[1, 1].set_ylabel("fraction"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(command_title)
    for ax in axes.ravel():
        ax.set_xlabel("throttle  f = ṁ/ṁ_full"); ax.grid(alpha=0.3)
        for p in pts:
            if not p.feasible:
                ax.axvspan(p.throttle - 0.02, p.throttle + 0.02,
                           color="#de2d26", alpha=0.10)
    fig.suptitle(
        f"{figure_title}  (Pc∝f^{tm.pc_exponent:g}; "
        "red = infeasible)", fontsize=11,
    )
    return _finish(fig, save_path, show)


def export_all_injector_figures(inj, out_dir, *, prefix="injector_",
                                show=False, throttle=None):
    """Write the full diagnostic figure set for a sized pintle injector.

    Returns the list of written filenames.  Each figure is guarded so one
    failure never blocks the rest.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("cross_section", plot_pintle_cross_section, inj),
        ("spray", plot_spray_envelope, inj),
        ("hydraulics", plot_injector_hydraulics, inj),
        ("atomization", plot_atomization, inj),
        ("thermal", plot_face_tip_thermal, inj),
        ("stability", plot_stability, inj),
        ("manifold", plot_manifold, inj),
        ("gates", plot_injector_gates, inj),
    ]
    if throttle is not None:
        jobs.append(("throttle_map", plot_throttle_map, throttle))
    written = []
    for name, fn, obj in jobs:
        fname = f"{prefix}{name}.png"
        try:
            fig = fn(obj, save_path=str(out_dir / fname), show=show)
            plt.close(fig)
            written.append(fname)
        except Exception:
            continue
    return written


def _dim_line(ax, p0, p1, text, *, color="0.25", off=0.0, fs=8, va="center",
              ha="center", rot=0.0):
    """Draw a dimension line with arrowheads at both ends + a centered label."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    ax.text(mx, my + off, text, color=color, fontsize=fs, ha=ha, va=va,
            rotation=rot, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                    ec="none", alpha=0.85))


def plot_pintle_schematic(inj, *, geom=None, spec=None, show=False,
                          save_path=None) -> "plt.Figure":
    """Dimensioned meridional section of the sized pintle (drafting style).

    Geometry stations come from
    :func:`raosim.injector_cad.resolve_reference_pintle_layout` — the same
    record the DXF profile and the 3-D reference assembly consume — and the
    labeled values from :func:`pintle_reference_geometry`, so the drawing,
    the CSV/JSON tables, and the solids agree.  Main panel: zoomed section
    around the metering end with true dimension lines (hatched metal, flow
    arrows, impinging point).  Right column: whole-injector context with the
    spray fan, and the tip end view with the slot pattern.
    """
    from raosim.injector_cad import resolve_reference_pintle_layout

    if geom is None:
        from raosim.injector_export import pintle_reference_geometry
        geom = pintle_reference_geometry(inj, spec=spec)
    lay = resolve_reference_pintle_layout(inj, spec)

    mm = 1.0e3
    Rp = lay["pintle_radius_m"] * mm
    Ri = lay["bore_radius_m"] * mm
    Ro = lay["annulus_outer_radius_m"] * mm
    Rs = lay["sleeve_outer_radius_m"] * mm
    gap = lay["annulus_gap_m"] * mm
    t_sl = lay["sleeve_wall_m"] * mm
    t_face = lay["face_thickness_m"] * mm
    Rf = lay["face_outer_radius_m"] * mm
    Rc = lay["chamber_radius_m"] * mm
    Lb = lay["body_length_m"] * mm
    bs = lay["body_straight_m"] * mm
    tip_flat = lay["tip_flat_radius_m"] * mm
    th_pt = lay["deflector_angle_deg"]
    radial_style = str(getattr(inj.slots, "geometry", "slots")).lower()
    if radial_style == "holes":
        hole_d = float(inj.slots.detail["hole_diameter"]) * mm
        ws = hs = hole_d
    else:
        ws = lay["slot_width_m"] * mm
        hs = lay["slot_height_m"] * mm
    ns = int(lay["slot_count"])
    z_st = lay["z_slot_top_m"] * mm
    z_sc = lay["z_slot_center_m"] * mm
    z_se = lay["z_sleeve_exit_m"] * mm
    L_skip = lay["skip_length_m"] * mm
    bore_end = lay["bore_end_m"] * mm
    half = lay["spray_half_angle_deg"] or 0.0
    Dp = 2.0 * Rp

    web = float(inj.slots.detail.get("web", 0.0)) * mm
    bf = float(inj.slots.detail.get("blockage_factor", 0.0))
    radial_name = getattr(inj, "radial_stream", "fuel")
    axial_name = "oxidizer" if radial_name == "fuel" else "fuel"

    fig = plt.figure(figsize=(13.6, 9.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0],
                          height_ratios=[1.05, 1.0],
                          wspace=0.14, hspace=0.20,
                          left=0.055, right=0.985, top=0.92, bottom=0.075)
    axm = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1])

    HATCH = dict(facecolor="white", edgecolor="0.15", hatch="//////",
                 linewidth=1.1)
    DIM = "0.20"
    LIQ = "#1f6fb5"     # central / radial stream
    GAS = "#c23b22"     # annular stream
    SPRAY = "#2ca25f"

    def poly(ax, pts, **kw):
        ax.add_patch(Polygon(np.asarray(pts), closed=True, **kw))

    def rect(ax, x0, z0, w, h, **kw):
        ax.add_patch(Rectangle((x0, z0), w, h, **kw))

    def ext_line(ax, x, z0, z1):
        ax.plot([x, x], [z0, z1], color="0.55", lw=0.6)

    def dim_h(ax, x0, x1, z, text, *, ext_from=None, fs=8.3, above=True):
        if ext_from is not None:
            ext_line(ax, x0, ext_from, z)
            ext_line(ax, x1, ext_from, z)
        ax.annotate("", xy=(x1, z), xytext=(x0, z),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.0))
        dz = -0.014 * Lb if above else 0.030 * Lb
        ax.text(0.5 * (x0 + x1), z + dz, text, color=DIM, fontsize=fs,
                ha="center", va="bottom" if above else "top",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.9))

    def dim_v(ax, z0, z1, x, text, *, ext_from=None, fs=8.3, side=1):
        if ext_from is not None:
            ax.plot([ext_from, x], [z0, z0], color="0.55", lw=0.6)
            ax.plot([ext_from, x], [z1, z1], color="0.55", lw=0.6)
        ax.annotate("", xy=(x, z1), xytext=(x, z0),
                    arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.0))
        ax.text(x + side * 0.010 * Lb, 0.5 * (z0 + z1), text, color=DIM,
                fontsize=fs, ha="left" if side > 0 else "right", va="center",
                rotation=90,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.9))

    def leader(ax, xy, xytext, text, *, color=DIM, fs=8.3):
        ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fs,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.9))

    # ---- metal (hatched) -------------------------------------------------
    Rf_draw = min(Rf, 2.4 * Ro)
    ch = lay["sleeve_exit_chamfer_m"] * mm
    for s in (+1, -1):
        # faceplate (clipped for the zoom; full extent in the context view)
        rect(axm, s * Rs if s > 0 else -Rf_draw, -t_face, Rf_draw - Rs,
             t_face, **HATCH)
        # sleeve with converging exit
        poly(axm, [(s * Ro, -t_face), (s * Rs, -t_face),
                   (s * Rs, z_se - ch), (s * Ro, z_se)], **HATCH)
        # pintle half-section: outer wall, tip cone flank, nose flat to the
        # axis, back up the closed nose and the bore wall
        poly(axm, [(s * Ri, -t_face), (s * Rp, -t_face), (s * Rp, bs),
                   (s * tip_flat, Lb), (0.0, Lb), (0.0, bore_end),
                   (s * Ri, bore_end), ], **HATCH)
        # slot window through the pintle wall
        rect(axm, min(s * Ri, s * Rp), z_st, abs(s * Rp - s * Ri), hs,
             facecolor="white", edgecolor="0.15", lw=1.0)

    axm.axvline(0.0, color="0.5", lw=0.7, ls="-.")
    axm.plot([-Rf_draw, Rf_draw], [0.0, 0.0], color="0.35", lw=0.9, ls=":")
    axm.text(Rf_draw - 0.01 * Lb, 0.015 * Lb, "injector face  z = 0",
             fontsize=7.5, color="0.35", ha="right", va="top")

    # ---- flow arrows -----------------------------------------------------
    rb = 0.45 * Ri
    axm.annotate("", xy=(rb, z_sc), xytext=(rb, -t_face - 0.36 * Lb),
                 arrowprops=dict(arrowstyle="-", color=LIQ, lw=2.2))
    axm.annotate("", xy=(Rp + 0.9 * gap, z_sc), xytext=(rb, z_sc),
                 arrowprops=dict(arrowstyle="->", color=LIQ, lw=2.2))
    axm.text(rb + 0.02 * Lb, -t_face - 0.365 * Lb,
             f"{radial_name}  (central bore → {ns} radial {radial_style})",
             color=LIQ, fontsize=8.4, ha="left", va="bottom")
    r_ann = 0.5 * (Rp + Ro)
    for s in (+1, -1):
        axm.annotate("", xy=(s * r_ann, z_se + 0.6 * L_skip),
                     xytext=(s * r_ann, -t_face - 0.31 * Lb),
                     arrowprops=dict(arrowstyle="->", color=GAS, lw=2.0))
    axm.text(-r_ann - 0.02 * Lb, -t_face - 0.315 * Lb,
             f"{axial_name}  (annular gap)", color=GAS, fontsize=8.4,
             ha="right", va="bottom")

    # impinging point + spray fan
    x_imp, z_imp = Rp + 0.9 * gap, z_sc
    axm.plot([x_imp], [z_imp], "o", color=SPRAY, ms=7, zorder=6)
    L_ar = 0.24 * Lb
    for dth in (-8.0, 0.0, 8.0):
        a = math.radians(half + dth)
        axm.annotate("", xy=(x_imp + L_ar * math.sin(a),
                             z_imp + L_ar * math.cos(a)),
                     xytext=(x_imp, z_imp),
                     arrowprops=dict(arrowstyle="->", color=SPRAY, lw=1.5,
                                     ls="--"))
    leader(axm, (x_imp, z_imp),
           (x_imp + 0.40 * Lb, z_imp - 0.16 * Lb),
           f"impinging point\nθ_s = {half:.1f}° spray half-angle",
           color=SPRAY)

    # ---- dimension lines ---------------------------------------------
    row = -t_face - 0.075 * Lb
    step = 0.062 * Lb
    dim_h(axm, -Ri, Ri, row, f"D_cg  Ø{2 * Ri:.2f}", ext_from=-t_face)
    dim_h(axm, -Rp, Rp, row - step, f"D_pr  Ø{Dp:.2f}", ext_from=-t_face)
    dim_h(axm, -Ro, Ro, row - 2 * step, f"D_ann_o  Ø{2 * Ro:.2f}",
          ext_from=-t_face)
    dim_h(axm, -Rs, Rs, row - 3 * step, f"D_ob  Ø{2 * Rs:.2f}",
          ext_from=-t_face)
    dim_h(axm, -tip_flat, tip_flat, Lb + 0.05 * Lb,
          f"Ø{2 * tip_flat:.2f} tip flat", ext_from=Lb, above=False)

    xv2 = Rs + 0.24 * Dp
    xv3 = Rs + 0.42 * Dp
    dim_v(axm, z_se, z_st, xv2, f"L_skip {L_skip:.2f}", ext_from=Rs)
    dim_v(axm, 0.0, Lb, xv3, f"L_body {Lb:.1f}", ext_from=None)
    opening_label = "d_hole" if radial_style == "holes" else "h_slot"
    leader(axm, (Rp + 0.3 * (Ro - Rp), z_sc),
           (xv3 + 0.12 * Dp, z_sc + 0.06 * Lb),
           f"{opening_label} {hs:.2f}")
    ax_t = -(Rf_draw + 0.06 * Dp)
    dim_v(axm, -t_face, 0.0, ax_t, f"t_face {t_face:.2f}",
          ext_from=-Rf_draw, side=-1)

    leader(axm, (0.5 * (Rp + Ro), 0.30 * bs),
           (Rs + 0.10 * Dp, 0.24 * bs), f"δ_ann {gap:.3f}")
    leader(axm, (0.5 * (Ro + Rs), 0.12 * bs),
           (Rs + 0.10 * Dp, 0.08 * bs), f"δ_sleeve {t_sl:.2f}")
    radial_callout = (
        f"{ns} × Ø{ws:.2f} holes"
        if radial_style == "holes"
        else f"{ns} × {ws:.2f} × {hs:.2f} slots"
    )
    leader(axm, (-0.5 * (Ri + Rp), z_sc),
           (-Rs - 0.72 * Dp, z_sc - 0.12 * Lb),
           f"{radial_callout}\n(web {web:.2f}, BF {bf:.0%})")

    # tip cone angle callout
    from matplotlib.patches import Arc
    flank_ang = math.degrees(math.atan2(Lb - bs, Rp - tip_flat))
    axm.plot([tip_flat, Rp + 0.16 * Dp], [Lb, Lb], color="0.55", lw=0.6,
             ls="--")
    axm.add_patch(Arc((tip_flat, Lb), 0.30 * Dp, 0.30 * Dp,
                      angle=0.0, theta1=-flank_ang, theta2=0.0,
                      color=DIM, lw=1.0))
    leader(axm, (tip_flat + 0.16 * Dp * math.cos(math.radians(0.5 * flank_ang)),
                 Lb - 0.16 * Dp * math.sin(math.radians(0.5 * flank_ang))),
           (-Rp - 0.62 * Dp, Lb + 0.075 * Lb),
           f"θ_pt {th_pt:.0f}° deflector")

    axm.set_aspect("equal")
    axm.set_xlim(-(Rf_draw + 0.75 * Dp), Rf_draw + 0.95 * Dp)
    axm.set_ylim(Lb + 0.30 * Lb, -t_face - 0.44 * Lb)
    axm.set_xlabel("radius r [mm]")
    axm.set_ylabel("axial z from injector face [mm]")
    axm.set_title("Dimensioned meridional section  (values in mm; "
                  "matches pintle_dimensions.csv)", fontsize=10)

    # ---- context view ------------------------------------------------
    Lc = float(inj.chamber_length) * mm
    try:
        x_wall = float(inj.spray_wall_axial_distance) * mm
    except Exception:
        x_wall = float("nan")
    z_end = min(Lc, (x_wall if math.isfinite(x_wall) else 0.6 * Lc)
                + Lb + 0.25 * Lc)
    for s in (+1, -1):
        axc.plot([s * Rc, s * Rc], [0.0, z_end], color="0.1", lw=1.8)
        rect(axc, s * Rs if s > 0 else -Rf, -t_face, Rf - Rs, t_face, **HATCH)
        poly(axc, [(s * Ri, 0.0), (s * Rp, 0.0), (s * Rp, bs),
                   (s * tip_flat, Lb), (0.0, Lb), (0.0, bore_end),
                   (s * Ri, bore_end)],
             facecolor="0.85", edgecolor="0.2", lw=0.8)
    if math.isfinite(x_wall):
        z_hit = z_sc + x_wall
        for s in (+1, -1):
            axc.plot([s * x_imp, s * Rc], [z_imp, z_hit], color=SPRAY,
                     lw=1.4, ls="--")
        axc.plot([Rc, -Rc], [z_hit, z_hit], lw=0, marker="o", ms=5,
                 color=SPRAY)
        leader(axc, (Rc, z_hit), (0.15 * Rc, z_hit + 0.16 * Lc),
               f"wall intercept\nz = {z_hit:.0f} ({100 * z_hit / Lc:.0f}% L_c)",
               color=SPRAY)
    dim_h(axc, -Rc, Rc, z_end - 0.03 * Lc, f"D_c  Ø{2 * Rc:.1f}",
          above=True)
    axc.set_aspect("equal")
    axc.set_xlim(-1.25 * Rf, 1.25 * Rf)
    axc.set_ylim(z_end + 0.06 * Lc, -t_face - 0.12 * Lc)
    axc.set_title("chamber context", fontsize=9)
    axc.tick_params(labelsize=7)

    # ---- tip end view -------------------------------------------------
    tt = np.linspace(0.0, 2.0 * math.pi, 181)
    axe.plot(Rp * np.cos(tt), Rp * np.sin(tt), color="0.15", lw=1.4)
    axe.plot(Ri * np.cos(tt), Ri * np.sin(tt), color="0.4", lw=0.9, ls="--")
    axe.plot(Ro * np.cos(tt), Ro * np.sin(tt), color="0.55", lw=0.8)
    axe.plot(Rs * np.cos(tt), Rs * np.sin(tt), color="0.15", lw=1.0)
    for i in range(ns):
        a = 2.0 * math.pi * i / ns
        ca, sa = math.cos(a), math.sin(a)
        if radial_style == "holes":
            from matplotlib.patches import Circle
            axe.add_patch(Circle(
                (0.5 * (Ri + Rp) * ca, 0.5 * (Ri + Rp) * sa),
                0.5 * ws, facecolor="white", edgecolor="0.15", lw=0.7,
            ))
        else:
            na, ta = (ca, sa), (-sa, ca)
            r0, r1 = Ri, Rp
            hw = 0.5 * ws
            pts = [(r0 * na[0] + hw * ta[0], r0 * na[1] + hw * ta[1]),
                   (r1 * na[0] + hw * ta[0], r1 * na[1] + hw * ta[1]),
                   (r1 * na[0] - hw * ta[0], r1 * na[1] - hw * ta[1]),
                   (r0 * na[0] - hw * ta[0], r0 * na[1] - hw * ta[1])]
            poly(axe, pts, facecolor="white", edgecolor="0.15", lw=0.7)
    leader(axe, (Rp * math.cos(0.4), Rp * math.sin(0.4)),
           (1.25 * Rs, 0.75 * Rs),
           f"{radial_callout}\nweb {web:.2f} mm\nBF {bf:.0%}")
    axe.set_aspect("equal")
    lim = 1.9 * Rs
    axe.set_xlim(-lim, 2.4 * Rs)
    axe.set_ylim(-lim, lim)
    axe.set_title(f"tip end view ({radial_style} pattern)", fontsize=9)
    axe.axis("off")

    tmr = float(inj.total_momentum_ratio)
    fig.suptitle(
        f"Pintle injector — D_pr Ø{Dp:.2f} mm, {ns} {radial_style}, "
        f"TMR {tmr:.2f}, radial stream: {radial_name}   "
        f"(fixed annulus + radial {radial_style}; reference geometry, "
        "not hardware-qualified)", fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig
