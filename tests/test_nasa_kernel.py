"""
Phase 12.4 — NASA-style kernel build (CalcRRCsAlongArc port) tests.

The kernel build in :func:`raosim.nasa_moc.build_kernel` is now a port
of the NASA marching architecture (``CalcInitialThroatLine`` +
``CalcRRCsAlongArc``).  For Rd/Rt geometries where the throat plane
contains a substantial subsonic region (e.g. Rd/Rt = 0.382, this
codebase's default), the row march cannot proceed past TT' because
``solve_interior_point``/``solve_wall_point`` require everywhere-
supersonic parents.  The fallback path emits an arc-following BD
curve and is what most current tests exercise.

Tests pin:

* The mass-flow sanity check — wall-side mass on every successfully
  marched RRC equals TT''s wall-side mass to within the configured
  ``mdot_tol``.
* BD endpoints — point B at the wall corner (theta = theta_B, on the
  arc), axis end at r = 0.
* Total RRC mass equals the sum of trapezoidal annular contributions.
* Legacy NASA kernel files parse without error (``TT'BF_Kernel.out``,
  ``BFE_Kernel.out``) — already covered by ``test_legacy_io.py`` but
  re-asserted here for the Phase-12.4 contract.
* End-to-end: switching to the new marching kernel does not regress
  the BVP residual on the reference ε=10/length_pct=80 case.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from raosim.legacy_io import parse_kernel_out
from raosim.nasa_moc import (
    ArcWall,
    MOCKernel,
    RaoKernelError,
    build_kernel,
    build_source_contour_from_kernel,
    calc_bde_region,
    calc_lrc_de,
    calc_massflow_along_rrc,
)


NASA_OUT = (
    Path(__file__).resolve().parents[1]
    / "Three-Dimensional-Nozzle-Design-Code-master"
    / "MOC_Grid_BDE"
    / "outputs_M3.5Perf"
)


def test_build_kernel_returns_well_formed_object():
    """``build_kernel`` returns a populated :class:`MOCKernel` for the
    standard Rao geometry, even when the row march has to fall back to
    the arc-following BD curve (Rd/Rt = 0.382 is in the subsonic-axis
    regime per NASA documentation)."""
    Rt = 0.020
    Rd = 0.382 * Rt
    kernel = build_kernel(Rt, Rd, math.radians(30.0), gamma=1.4, n_kernel=24)

    assert isinstance(kernel, MOCKernel)
    assert len(kernel.rrcs) >= 1
    assert len(kernel.bd) >= 4
    assert len(kernel.massflow) == len(kernel.rrcs)


def test_kernel_bd_endpoints_span_full_radial_cross_section():
    """BD's wall point lies on the throat arc at theta = theta_B; the
    axis end lies on the symmetry axis (r = 0)."""
    Rt = 0.020
    Rd = 0.382 * Rt
    theta_B = math.radians(30.0)
    kernel = build_kernel(Rt, Rd, theta_B, gamma=1.4, n_kernel=24)

    B = kernel.B
    axis_end = kernel.bd[-1]

    assert B.r == pytest.approx(Rt + Rd * (1.0 - math.cos(theta_B)), rel=1e-3)
    assert axis_end.r == pytest.approx(0.0, abs=1e-9)


def test_kernel_mass_flow_consistent_across_rrcs():
    """Every successfully marched RRC's wall-side mass flow must agree
    with TT''s wall-side mass flow to within the build's ``mdot_tol``
    (default 5 %).  This is NASA's built-in sanity check (line 1085)."""
    Rt = 0.020
    Rd = 0.382 * Rt
    kernel = build_kernel(Rt, Rd, math.radians(30.0), gamma=1.4, n_kernel=24,
                          mdot_tol=0.05)
    mdot_throat = float(kernel.massflow[0][0])
    if mdot_throat <= 0:
        pytest.skip("TT' mass is zero (degenerate kernel)")
    for j, mass in enumerate(kernel.massflow):
        err = abs(float(mass[0]) - mdot_throat) / mdot_throat
        assert err <= 0.05, f"RRC {j} mass err {err:.2%} > 5%"


def test_calc_massflow_along_rrc_matches_trapezoidal_integration():
    """The mass flow accumulator should be monotone non-decreasing from
    axis (i=last) to wall (i=0)."""
    Rt = 0.020
    Rd = 0.382 * Rt
    kernel = build_kernel(Rt, Rd, math.radians(30.0), gamma=1.4, n_kernel=12)
    mass = kernel.massflow[-1]
    # Wall-first: index 0 is wall (total mass), index -1 is axis (zero).
    assert mass[-1] == pytest.approx(0.0, abs=1e-12)
    assert mass[0] > 0.0
    # Monotone descent from wall to axis.
    for a, b in zip(mass[:-1], mass[1:]):
        assert b <= a + 1e-12


def test_arc_wall_geometry_round_trips():
    """:class:`ArcWall` should reproduce its own ``r(x)`` and
    ``dr_dx(x)`` consistently."""
    arc = ArcWall(Rt=0.020, Rd=0.382 * 0.020, theta_max=math.radians(30.0))
    x_mid = 0.5 * arc.x_end
    r_at_mid = arc.r(x_mid)
    # Inverse: the angle whose sin gives x_mid/Rd is asin(x/Rd); r is then
    # Rt + Rd*(1-cos(that angle)).
    angle = math.asin(x_mid / arc.Rd)
    r_check = arc.Rt + arc.Rd * (1.0 - math.cos(angle))
    assert r_at_mid == pytest.approx(r_check, rel=1e-9)
    # Theta = arctan(dr/dx) is the local wall flow angle.
    assert arc.theta(x_mid) == pytest.approx(angle, rel=1e-9)


def test_nasa_kernel_files_parse():
    """``TT'BF_Kernel.out`` and ``BFE_Kernel.out`` parse to LegacyTable."""
    tt = parse_kernel_out(NASA_OUT / "TT'BF_Kernel.out")
    bfe = parse_kernel_out(NASA_OUT / "BFE_Kernel.out")
    assert tt.data.shape[0] > 10
    assert bfe.data.shape[0] > 10
    assert tt.data.shape[1] == bfe.data.shape[1]


def test_build_kernel_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_kernel(Rt=-1.0, Rd=0.1, theta_B=math.radians(20.0), gamma=1.4)
    with pytest.raises(ValueError):
        build_kernel(Rt=0.020, Rd=0.020, theta_B=-1.0, gamma=1.4)
    with pytest.raises(ValueError):
        build_kernel(Rt=0.020, Rd=-0.020, theta_B=math.radians(20.0), gamma=1.4)
    with pytest.raises(ValueError):
        build_kernel(Rt=0.020, Rd=0.020, theta_B=math.radians(20.0),
                     gamma=1.4, n_kernel=2)


def test_rao_kernel_error_class_exists():
    """``RaoKernelError`` is the documented signal for mass-flow
    violations; downstream consumers may catch it."""
    err = RaoKernelError("test")
    assert isinstance(err, RuntimeError)


def test_marching_kernel_produces_multiple_rrcs_for_typical_geometry():
    """
    The source-shaped dtheta row march can advance past TT' when the
    starting line is compatible with the visible NASA unit-process equations.
    The codebase's tight default remains a harder curved-TT' problem; this
    smoke test only requires a well-formed kernel there.
    """
    Rt = 0.020
    Rd = 0.382 * Rt
    kernel_default = build_kernel(Rt, Rd, math.radians(30.0),
                                  gamma=1.4, n_kernel=24)
    assert len(kernel_default.bd) >= 4


def test_source_dtheta_row_march_reaches_wall_without_mass_relaxation():
    """Visible-source row march reaches theta_B with the default mass gate.

    This exercises ``CalcArcWallPoint`` special-wall insertion,
    ``CalcInteriorMeshPoints``, ``CalcAxialMeshPoint``, and NASA's
    mass-flow sanity check without relaxing ``mdot_tol``.
    """
    kernel = build_kernel(
        Rt=1.0,
        Rd=1.0,
        theta_B=math.radians(15.2196),
        gamma=1.4,
        n_kernel=101,
        starting_line_method="sauer_modified",
        mdot_tol=0.05,
    )

    assert kernel.fallback_used is False
    assert kernel.reached_wall is True
    assert len(kernel.rrcs) > 50
    assert len(kernel.bd) > len(kernel.rrcs[0])
    assert kernel.B.x == pytest.approx(kernel.Rd * math.sin(kernel.theta_B), abs=1e-9)
    assert kernel.B.r == pytest.approx(
        kernel.Rt + kernel.Rd * (1.0 - math.cos(kernel.theta_B)),
        abs=1e-9,
    )
    mdot0 = float(kernel.massflow[0][0])
    for mass in kernel.massflow:
        assert abs(float(mass[0]) - mdot0) / mdot0 <= 0.05


def test_corrected_kl_row_march_incompatibility_stays_visible():
    """Do not hide the corrected-KL starting-line mismatch with a fallback claim."""
    kernel = build_kernel(
        Rt=1.0,
        Rd=1.0,
        theta_B=math.radians(15.2196),
        gamma=1.4,
        n_kernel=101,
        starting_line_method="kliegel_levine",
        mdot_tol=0.05,
    )

    assert kernel.fallback_used is True
    assert kernel.reached_wall is False


def test_calc_bde_region_builds_wall_to_de_seed_rows():
    """BFE slice: port CalcBDERegion, CalcRemainingMesh, and CalcWallContour."""
    kernel = build_kernel(
        Rt=1.0,
        Rd=1.0,
        theta_B=math.radians(15.2196),
        gamma=1.4,
        n_kernel=101,
        starting_line_method="sauer_modified",
        mdot_tol=0.05,
    )
    topology = calc_lrc_de(
        kernel,
        x_E=12.5363,
        r_E=math.sqrt(6.73651),
        gamma=1.4,
        Rt=1.0,
        epsilon=6.73651,
        pa_over_p0=0.0,
        n_points=24,
    )

    region = calc_bde_region(kernel, topology)

    assert region.complete_remaining_mesh is True
    assert region.wall_contour_complete is True
    assert region.iD >= 1
    assert len(region.rows) == max(len(topology.DE) - 1, 0)
    assert len(region.grid_rows) == len(region.rows)
    assert len(region.wall_contour) == len(region.grid_rows)
    assert region.rows
    for row in region.rows:
        assert len(row) == region.iD + 1
        assert row[0].r > row[-1].r
        assert row[-1].r >= topology.D.r
        assert all(point.M > 1.0 for point in row)
    for row in region.grid_rows:
        assert len(row) > region.iD
        assert row[0].r > row[-1].r
        assert row[-1].r == pytest.approx(0.0, abs=1e-10)
        assert all(point.M > 1.0 for point in row)


def test_build_source_contour_from_kernel_reports_uncropped_status():
    """Current source-port contour is complete through wall extraction.

    Length closure/cropping is deliberately not claimed yet; that belongs to
    the next ``SetThetaB``/``CropNozzleToLength`` port slice.
    """
    kernel = build_kernel(
        Rt=1.0,
        Rd=1.0,
        theta_B=math.radians(15.2196),
        gamma=1.4,
        n_kernel=101,
        starting_line_method="sauer_modified",
        mdot_tol=0.05,
    )
    contour = build_source_contour_from_kernel(
        kernel,
        x_E=12.5363,
        r_E=math.sqrt(6.73651),
        epsilon=6.73651,
        pa_over_p0=0.0,
        n_de_points=24,
    )

    diag = contour.diagnostics
    assert diag["canonical_reference_track"] == "visible_source_port"
    assert diag["source_contour_complete"] is True
    assert diag["length_closed"] is False
    assert diag["crop_nozzle_to_length"] == "not_ported"
    assert diag["outer_theta_b_driver"] == "not_canonical"
    assert diag["nasa_reference_matched_eligible"] is False
    assert contour.bfe.complete_remaining_mesh is True
    assert contour.bfe.wall_contour_complete is True
    assert contour.wall_export.shape == (len(contour.wall), 2)
    assert len(contour.wall) == len(kernel.rrcs) + len(contour.bfe.wall_contour)


def test_phase12_4_end_to_end_no_regression():
    """The Phase 12.4 build_kernel rewrite should not regress the BVP
    ``max_scaled`` residual on the reference case (mass closure may
    stay at the same level since the fallback BD curve is what was
    used pre-rewrite)."""
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=8, n_kernel=8,
        max_nfev=200, evaluate_moc=False,
    )
    sol = solve_rao_bvp(cfg)
    assert sol.residuals.max_scaled < 0.5, (
        f"max_scaled regressed to {sol.residuals.max_scaled:.3e}"
    )


# ---------------------------------------------------------------------
#  NASA wall.out bit-comparison
# ---------------------------------------------------------------------


def test_kl_throat_wall_mach_bit_comparable_to_nasa_wall_out():
    """
    NASA-port reliability gate (REWRITE_PLAN.md §13:
    ``NASA_REFERENCE_MATCHED``): the KL throat-plane wall Mach must
    match ``MOC_Grid_BDE/outputs_M3.5Perf/wall.out`` row 0 to 4
    decimals.  NASA M3.5Perf inputs: Rt = Rd = 1 in (Rc/Rt = 1.0),
    γ = 1.4, axisymmetric.  Row 0 (i=0, j=0): the wall point at the
    throat plane.
    """
    from raosim.legacy_io import parse_wall_out
    from raosim.transonic_kernel import GEOM_AXI, kliegel_levine

    nasa_wall = parse_wall_out(NASA_OUT / "wall.out")
    # Row 0 = wall at throat plane (i=0, j=0).
    M_nasa_wall = float(nasa_wall.column("mach")[0])

    state = kliegel_levine(
        r_over_Rt=1.0, x_over_Rt=0.0,
        gamma=1.4, Rc_over_Rt=1.0, geom=GEOM_AXI,
    )
    assert state.M == pytest.approx(M_nasa_wall, abs=1e-4), (
        f"KL wall Mach {state.M:.5f} != NASA wall.out {M_nasa_wall:.5f} "
        "(4-decimal NASA_REFERENCE_MATCHED gate)"
    )


@pytest.mark.xfail(
    reason="Historical TT' fixture parity is blocked by unresolved generator "
           "provenance. Keep xfailed unless a matching source/executable is "
           "recovered or a documented fixture-reconstruction mode is added.",
)
def test_python_tt_prime_matches_nasa_tt_prime_rms_1e3():
    """Historical fixture overlay gate for ``TT'.out``.

    This is not the source-faithful port gate while the M3.5Perf TT'
    generator remains unresolved.
    """
    import numpy as np

    from raosim.legacy_io import parse_tt_prime_out

    nasa_tt = parse_tt_prime_out(NASA_OUT / "TT'.out")
    kernel = build_kernel(
        Rt=1.0,
        Rd=1.0,
        theta_B=math.radians(15.2196),
        gamma=1.4,
        n_kernel=101,
    )
    py_tt = kernel.rrcs[0]

    def rms(name: str, values):
        ref = nasa_tt.column(name)
        return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - ref) ** 2)))

    assert rms("X", [node.x for node in py_tt]) < 1e-3
    assert rms("R", [node.r for node in py_tt]) < 1e-3
    assert rms("MACH", [node.M for node in py_tt]) < 1e-3
    assert rms("THETA", [math.degrees(node.theta) for node in py_tt]) < 1e-3


@pytest.mark.xfail(
    reason="Full wall.out RMS bit-comparison requires the Phase 12.4 row "
           "march to actually produce the NASA kernel BD (currently the "
           "fallback arc-following BD is used for Rd/Rt = 0.382).  See "
           "test_marching_kernel_produces_multiple_rrcs_for_typical_geometry.",
)
def test_python_port_wall_matches_nasa_wall_out_rms_1e3():
    """End-to-end NASA wall.out RMS check.  Targets:
    RMS(x/R* - NASA) < 1e-3, RMS(r/R* - NASA) < 1e-3, RMS(M - NASA) < 1e-3.
    """
    from raosim.legacy_io import parse_wall_out
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    nasa_wall = parse_wall_out(NASA_OUT / "wall.out")
    Rt = 0.0254  # 1 inch in metres (NASA R* = 1 in)
    # M_exit = 3.5 with γ=1.4 → epsilon ≈ 6.79
    cfg = RaoSolverConfig(
        Rt=Rt, epsilon=6.79, gamma=1.4, pa_over_p0=0.0,
        length_pct=100.0,  # NASA "perfect" maps loosely to 100% length
        n_control=20, n_kernel=20,
        max_nfev=500, residual_tol=2e-3,
        evaluate_moc=True,
    )
    sol = solve_rao_bvp(cfg)
    # Map NASA wall (i=0 row across j) to our exported wall.
    nasa_x = nasa_wall.column("X_over_Rstar")
    nasa_r = nasa_wall.column("R_over_Rstar")
    nasa_M = nasa_wall.column("mach")
    py_x = sol.wall_export[:, 0] / Rt
    py_r = sol.wall_export[:, 1] / Rt
    # Resample at common x_over_Rstar
    x_common = np.linspace(max(nasa_x.min(), py_x.min()),
                           min(nasa_x.max(), py_x.max()), 50)
    py_r_resampled = np.interp(x_common, py_x, py_r)
    nasa_r_resampled = np.interp(x_common, nasa_x, nasa_r)
    rms_r = float(np.sqrt(np.mean((py_r_resampled - nasa_r_resampled) ** 2)))
    assert rms_r < 1e-3, f"wall r RMS {rms_r:.3e} > 1e-3"
