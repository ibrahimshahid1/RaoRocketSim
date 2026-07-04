"""Explicit Rao construction topology (REWRITE_PLAN §11.7, Phase 12.6).

This module formalises the NASA/Rao contour construction as one explicit
object.  The field set follows the REWRITE_PLAN §11.7 spec exactly:

    TT_prime        initial data line (transonic start line)
    B               end-of-expansion wall point (throat arc exit)
    BF              final RRC of the kernel, wall -> axis
    D               selected point on BF (control-surface attachment)
    BD              wall-to-D portion of BF (the mass-flow curve)
    DE              control surface (left-running characteristic D -> E)
    E               nozzle exit point
    streamline_BE   wall contour from B to E -- THE bell wall
    theta_B         converged end-of-arc wall angle
    mass_BD / mass_DE   closure pair (equal at convergence)

``streamline_BE`` is NASA's ``CalcWallContour`` streamline traced from B
through the BFE mesh to E (Rice, *Three-Dimensional Nozzle Design Code*,
JHU/APL 2003, §3.4 — ported as :func:`raosim.nasa_moc.calc_bde_region`).
The full nozzle wall is the throat-arc wall (theta = 0 ... theta_B) +
``streamline_BE``; :meth:`RaoTopology.full_wall` assembles it.  Once the
solver default flips to the characteristic formulation, this field — not
``_construct_wall_from_ce`` — is the wall (see the DIRECTION block in
JAX_DIFFERENTIABLE_PLAN.md; the legacy constructor is deleted at that
point, not before).

Nodes are :class:`raosim.moc.CharPoint` (theta, M, nu, mu, compat±) so
plots and diagnostics can consume the topology directly.  The lighter
:class:`raosim.nasa_moc.RaoTopology` (FlowNode-based, produced by the
construction routines) remains the internal carrier; :func:`build_topology`
lifts it into the full-form object.  When ``raosim/jax`` absorbs the core
(DIRECTION item 1), this module is the intended public boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from raosim.gas_dynamics import mach_angle, prandtl_meyer
from raosim.moc import CharPoint, FlowNode
from raosim.nasa_moc import (
    BDERegion,
    MOCKernel,
    RaoTopology as _NasaRaoTopology,
    calc_bde_region,
    calc_massflow_along_rrc,
    calc_mdot_bd_grid,
    set_theta_b,
)

__all__ = ["RaoTopology", "build_topology", "build_reference_topology"]


def _char_point(x: float, r: float, theta: float, M: float,
                gamma: float) -> CharPoint:
    M = float(max(M, 1.000001))
    nu = prandtl_meyer(M, gamma)
    return CharPoint(
        x=float(x), r=float(r), theta=float(theta), M=M,
        nu=float(nu), mu=float(mach_angle(M)),
        compat_plus=float(theta + nu), compat_minus=float(theta - nu),
    )


def _from_flow(node: FlowNode, gamma: float) -> CharPoint:
    return _char_point(node.x, node.r, node.theta, node.M, gamma)


@dataclass(frozen=True)
class RaoTopology:
    """Explicit objects of the Rao construction (REWRITE_PLAN §11.7)."""

    TT_prime: tuple[CharPoint, ...]       # initial data line (wall-first)
    B: CharPoint                          # end-of-expansion wall point
    BF: tuple[CharPoint, ...]             # final kernel RRC (wall -> axis)
    D: CharPoint                          # selected point on BF
    BD: tuple[CharPoint, ...]             # wall -> D portion of BF
    DE: tuple[CharPoint, ...]             # control surface (D -> E)
    E: CharPoint                          # nozzle exit point
    streamline_BE: tuple[CharPoint, ...]  # wall contour from B to E
    theta_B: float                        # converged arc-end wall angle
    mass_BD: float
    mass_DE: float                        # == mass_BD at convergence
    # --- provenance (not in the §11.7 core set, kept for diagnostics) ---
    d_fraction: float = float("nan")      # arc-length fraction of D on BF
    arc_wall: tuple[CharPoint, ...] = ()  # throat-arc wall points (0..B)
    diagnostics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def full_wall(self) -> np.ndarray:
        """Manufacturable wall polyline: throat-arc wall + streamline_BE.

        Returns an (n, 2) array of (x, r).  The duplicate B at the seam
        is dropped; points are required to advance monotonically in x.
        """
        pts: list[tuple[float, float]] = [(p.x, p.r) for p in self.arc_wall]
        for p in self.streamline_BE:
            if pts and abs(p.x - pts[-1][0]) < 1e-12 \
                    and abs(p.r - pts[-1][1]) < 1e-12:
                continue
            pts.append((p.x, p.r))
        wall = np.asarray(pts, dtype=float)
        if wall.shape[0] >= 2 and not np.all(np.diff(wall[:, 0]) > -1e-12):
            raise ValueError("assembled wall is not monotone in x")
        return wall

    def closure_report(self) -> dict:
        """Closure metrics: mass pair, attachment seams, exit landing."""
        sBE = self.streamline_BE
        rep = {
            "mass_rel_mismatch": (
                abs(self.mass_DE - self.mass_BD)
                / max(abs(self.mass_BD), 1e-300)
            ),
            "BD_starts_at_B": math.hypot(self.BD[0].x - self.B.x,
                                         self.BD[0].r - self.B.r),
            "BD_ends_at_D": math.hypot(self.BD[-1].x - self.D.x,
                                       self.BD[-1].r - self.D.r),
            "DE_starts_at_D": math.hypot(self.DE[0].x - self.D.x,
                                         self.DE[0].r - self.D.r),
            "DE_ends_at_E": math.hypot(self.DE[-1].x - self.E.x,
                                       self.DE[-1].r - self.E.r),
        }
        if sBE:
            rep["wall_starts_at_B"] = math.hypot(sBE[0].x - self.B.x,
                                                 sBE[0].r - self.B.r)
            rep["wall_ends_at_E"] = math.hypot(sBE[-1].x - self.E.x,
                                               sBE[-1].r - self.E.r)
        return rep


def build_topology(
    kernel: MOCKernel,
    nasa_topology: _NasaRaoTopology,
    bde_region: BDERegion | None = None,
) -> RaoTopology:
    """Lift the construction outputs into the full-form §11.7 topology.

    Parameters
    ----------
    kernel
        Marched kernel (e.g. from ``set_theta_b`` or ``build_kernel``).
    nasa_topology
        The FlowNode-based topology from ``set_theta_b`` / ``calc_lrc_de``.
    bde_region
        Optional pre-computed BFE region.  Built via
        :func:`raosim.nasa_moc.calc_bde_region` when omitted.

    Notes
    -----
    * ``BF`` is the kernel's final RRC verbatim (wall-first).  At sharp
      downstream radii it may legitimately be non-monotone in r (the
      Phase 12.4 fold — theta crosses mu mid-row); consumers must not
      assume a monotone descent.
    * ``BD`` is BF truncated at D by *arc length* (``calc_mdot_bd_grid``
      convention: fraction 0 at the wall, 1 at the axis), with the exact
      interpolated D appended.
    * ``streamline_BE`` comes from ``bde_region.wall_contour``; B is
      prepended when the region march starts strictly below B.
    """
    gamma = float(kernel.gamma)

    if bde_region is None:
        bde_region = calc_bde_region(kernel, nasa_topology)

    tt_prime = tuple(
        _char_point(n.x, n.r, n.theta, n.M, gamma) for n in kernel.rrcs[0]
    )
    bf_nodes = kernel.bd
    BF = tuple(
        _char_point(n.x, n.r, n.theta, n.M, gamma) for n in bf_nodes
    )
    B = BF[0]
    D = _from_flow(nasa_topology.D, gamma)
    E = _from_flow(nasa_topology.E, gamma)
    DE = tuple(_from_flow(n, gamma) for n in nasa_topology.DE)

    # BD: wall -> D along BF by arc length.
    if kernel.massflow:
        massflow = kernel.massflow[-1]
    else:  # pragma: no cover - kernels always carry massflow today
        massflow = calc_massflow_along_rrc(bf_nodes, gamma)
    d_fraction = float(nasa_topology.d_fraction)
    _, D_interp, i_bracket, _ratio = calc_mdot_bd_grid(
        bf_nodes, massflow, d_fraction,
    )
    bd_points = list(BF[:i_bracket])
    d_node = _char_point(D_interp.x, D_interp.r, D_interp.theta,
                         D_interp.M, gamma)
    if not bd_points or math.hypot(bd_points[-1].x - d_node.x,
                                   bd_points[-1].r - d_node.r) > 1e-12:
        bd_points.append(d_node)
    BD = tuple(bd_points)

    # streamline_BE: the BFE wall contour, B prepended if absent.
    s_be = [_from_flow(p, gamma) for p in bde_region.wall_contour]
    if s_be and math.hypot(s_be[0].x - B.x, s_be[0].r - B.r) > 1e-9:
        s_be.insert(0, B)
    streamline_BE = tuple(s_be)

    # Throat-arc wall: the wall point of every kernel RRC up to B.
    arc_wall = tuple(
        _char_point(row[0].x, row[0].r, row[0].theta, row[0].M, gamma)
        for row in kernel.rrcs if row
    )

    return RaoTopology(
        TT_prime=tt_prime, B=B, BF=BF, D=D, BD=BD, DE=DE, E=E,
        streamline_BE=streamline_BE,
        theta_B=float(nasa_topology.theta_B),
        mass_BD=float(nasa_topology.mass_BD),
        mass_DE=float(nasa_topology.mass_DE),
        d_fraction=d_fraction,
        arc_wall=arc_wall,
        diagnostics={
            "bfe_complete_remaining_mesh": bool(
                bde_region.complete_remaining_mesh),
            "bfe_wall_contour_complete": bool(
                bde_region.wall_contour_complete),
            "bfe_negative_r_truncated_rows": int(
                bde_region.negative_r_truncated_rows),
            "kernel_reached_wall": bool(kernel.reached_wall),
            "kernel_fallback_used": bool(kernel.fallback_used),
        },
    )


def build_reference_topology(
    Rt: float,
    epsilon: float,
    length_pct: float,
    gamma: float,
    pa_over_p0: float = 0.01,
    *,
    n_kernel: int = 24,
    n_de_points: int = 24,
    theta_b_init_deg: float = 21.87,
    starting_line_method: str = "kliegel_levine",
    Ru_factor: float = 1.5,
    max_iter: int = 30,
) -> RaoTopology:
    """One-call fixed-(L, epsilon) construction -> full-form topology.

    Runs NASA's fixed-end closure (``set_theta_b``: outer secant on
    length, ``calc_lrc_de(end_condition="fixed_end")`` walking D along
    BD; converges post-Phase-12.4) and the BFE region march, then lifts
    everything via :func:`build_topology`.  This is the
    geometrically-sane characteristic closure at the commanded
    (epsilon, L%): mass-consistent, seam-closed, TOP-bell-shaped — but
    **not** Rao-stationary (the variational refinement is the BVP's
    job; chart theta_N/theta_E describe that optimal family).

    The exit-station length convention matches ``solve_rao_bvp``
    (``rao_variational._target_length``).
    """
    from raosim.rao_variational import _target_length

    Ln = _target_length(Rt, epsilon, length_pct)
    nasa_topo, kernel = set_theta_b(
        Rt, epsilon, length_pct, gamma, pa_over_p0,
        theta_b_init_deg=theta_b_init_deg,
        n_kernel=n_kernel, n_de_points=n_de_points,
        starting_line_method=starting_line_method,
        L_target=Ln, Ru=Ru_factor * Rt,
        end_condition="fixed_end", max_iter=max_iter,
    )
    return build_topology(kernel, nasa_topo)
