"""injector_plots.py - pintle injector diagnostic figures.

A full diagnostic set drawn from the sized :class:`InjectorDesignResult`:
cross-section, spray envelope, hydraulics, atomization/combustion development,
face/tip thermal stack, stability, manifold maldistribution, the gate
scorecard, and (optionally) the throttle map.  Kept separate from the
pure-numeric solver, matching the repo's plotting/solver split.
"""

from __future__ import annotations

import math

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
    """Meridional half-section of the sized pintle: faceplate, hollow pintle
    body, tip, axial annulus and the radial slots, with the chamber wall."""
    Dp = inj.pintle_diameter
    Rp = 0.5 * Dp
    gap = inj.annulus.detail.get("gap", 0.1 * Rp)
    Ro = inj.annulus.detail.get("outer_diameter", Dp + 2 * gap) * 0.5
    Rc = inj.chamber_radius
    slot_w = inj.slots.detail.get("slot_width", 0.4 * gap)
    slot_h = inj.slots.detail.get("slot_height", slot_w)
    tip_r = 0.5 * Dp
    body_len = 3.0 * Dp                      # pintle protrusion into the chamber
    t_face = max(0.4 * Dp, 2 * gap)          # faceplate thickness (schematic)
    mm = 1e3

    fig, ax = plt.subplots(figsize=(8, 5))
    # faceplate (behind the face plane x=0), from the sleeve OD out to the wall
    _metal(ax, (-t_face * mm, Ro * mm), t_face * mm, (Rc - Ro) * mm)
    ax.add_patch(Rectangle((-t_face * mm, -Rc * mm), t_face * mm,
                           (Rc - Ro) * mm, facecolor="0.75", edgecolor="0.3",
                           hatch="////", linewidth=1.0))
    # outer sleeve walls bounding the annulus (thin)
    t_sleeve = max(0.15 * gap, 0.2e-3)
    for sgn in (1, -1):
        _metal(ax, (0, sgn * Ro * mm), body_len * 0.55 * mm,
               sgn * t_sleeve * mm)
    # hollow pintle body (the post), mirrored; the body stops where the
    # rounded tip begins so the nose reads as a rounded cap, not a notch.
    body_straight = body_len - tip_r
    for sgn in (1, -1):
        ax.add_patch(Rectangle((0, 0), body_straight * mm, sgn * Rp * mm,
                               facecolor="0.82", edgecolor="0.3",
                               linewidth=1.0))
    # hollow internal feed passage
    ax.add_patch(Rectangle((0, -0.55 * Rp * mm), (body_len - tip_r) * mm,
                           1.10 * Rp * mm, facecolor="#cfe8ff",
                           edgecolor="none"))
    # rounded tip
    th = np.linspace(-math.pi / 2, math.pi / 2, 40)
    ax.add_patch(Polygon(
        np.column_stack([((body_len - tip_r) + tip_r * np.cos(th)) * mm,
                         tip_r * np.sin(th) * mm]),
        closed=True, facecolor="0.82", edgecolor="0.3", linewidth=1.0))

    # axial annulus passages (between pintle OD and sleeve ID)
    for sgn in (1, -1):
        ax.add_patch(Rectangle((-t_face * mm, sgn * Rp * mm),
                               (body_len * 0.55 + t_face) * mm,
                               sgn * gap * mm, facecolor="#9ecae1",
                               edgecolor="none", alpha=0.9))
    # radial slots near the tip (the slotted stream exits the pintle wall)
    n_show = min(int(inj.slot_count), 6)
    xs = np.linspace(body_len - tip_r - slot_h, body_len - tip_r,
                     n_show + 1)[:-1]
    for x in xs:
        for sgn in (1, -1):
            ax.add_patch(Rectangle((x * mm, sgn * Rp * mm), slot_w * mm,
                                   sgn * slot_h * mm, facecolor="#fdae6b",
                                   edgecolor="0.4", linewidth=0.5))
    # chamber wall
    ax.plot([-t_face * mm, body_len * 1.15 * mm], [Rc * mm, Rc * mm],
            "k-", lw=2)
    ax.plot([-t_face * mm, body_len * 1.15 * mm], [-Rc * mm, -Rc * mm],
            "k-", lw=2)
    ax.axhline(0, color="0.6", lw=0.6, ls="--")

    # annotations
    ax.annotate(f"pintle Ø{Dp*mm:.1f} mm", (0.5 * body_len * mm, 0),
                ha="center", va="center", fontsize=9)
    ax.annotate(f"annulus gap {gap*mm:.2f} mm\n({inj.annulus.role})",
                (0.25 * body_len * mm, (Rp + 0.5 * gap) * mm),
                (0.25 * body_len * mm, (Rc * 0.65) * mm), fontsize=8,
                ha="center", arrowprops=dict(arrowstyle="->", color="#3182bd"))
    ax.annotate(
        f"{inj.slot_count}× slots {slot_w*mm:.2f}×{slot_h*mm:.2f} mm\n"
        f"({inj.slots.role})",
        ((body_len - tip_r) * mm, (Rp + slot_h) * mm),
        ((body_len - tip_r) * mm, (Rc * 0.78) * mm), fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="#e6550d"))
    ax.text(body_len * 1.0 * mm, Rc * 0.92 * mm, "chamber wall",
            fontsize=8, ha="right")

    ax.set_xlabel("axial x [mm]")
    ax.set_ylabel("radius r [mm]")
    ax.set_title(f"Pintle cross-section  (TMR={inj.total_momentum_ratio:.2f}, "
                 f"radial={inj.radial_stream})")
    ax.set_aspect("equal")
    ax.set_xlim(-t_face * mm * 1.5, body_len * 1.2 * mm)
    ax.set_ylim(-Rc * 1.15 * mm, Rc * 1.15 * mm)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


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
    roles = ["fuel", "oxidizer"]
    colors = ["#e6550d", "#3182bd"]
    smd = [at.streams[r].sauter_mean_diameter * 1e6 for r in roles]
    x = np.arange(2)
    ax1.bar(x, smd, color=colors, edgecolor="0.3")
    ax1.set_xticks(x); ax1.set_xticklabels(roles)
    ax1.set_ylabel("SMD d₃₂  [µm]")
    ax1.set_title("Sauter mean diameter (Hinze)")
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
             f"predicted η_c*≈{at.predicted_cstar_efficiency:.2f}",
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
    """Throttle-sweep curves: velocity, TMR, SMD, η_c*, sleeve stroke."""
    pts = tm.points
    f = [p.throttle for p in pts]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(f, [p.v_annulus for p in pts], "o-", label="annulus")
    axes[0, 0].plot(f, [p.v_slots for p in pts], "s-", label="slots")
    axes[0, 0].set_ylabel("injection velocity [m/s]"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Velocity")
    axes[0, 1].plot(f, [p.total_momentum_ratio for p in pts], "o-",
                    color="#756bb1")
    axes[0, 1].set_ylabel("TMR"); axes[0, 1].set_title("Total momentum ratio")
    axes[1, 0].plot(f, [p.smd_limiting * 1e6 for p in pts], "o-",
                    color="#e6550d")
    axes[1, 0].set_ylabel("SMD [µm]"); axes[1, 0].set_title("Atomization (SMD)")
    axes[1, 1].plot(f, [p.predicted_cstar_efficiency for p in pts], "o-",
                    color="#31a354", label="η_c*")
    axes[1, 1].plot(f, [p.sleeve_stroke_fraction for p in pts], "s--",
                    color="0.4", label="sleeve stroke")
    axes[1, 1].set_ylabel("fraction"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Predicted η_c* and sleeve stroke")
    for ax in axes.ravel():
        ax.set_xlabel("throttle  f = ṁ/ṁ_full"); ax.grid(alpha=0.3)
        for p in pts:
            if not p.feasible:
                ax.axvspan(p.throttle - 0.02, p.throttle + 0.02,
                           color="#de2d26", alpha=0.10)
    fig.suptitle(f"Movable-sleeve throttle map  (Pc∝f^{tm.pc_exponent:g}; "
                 "red = infeasible)", fontsize=11)
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
