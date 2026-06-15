"""
J4 (JAX_DIFFERENTIABLE_PLAN.md §7): convergence of the differentiable backend
on the Phase-6 reference case (epsilon=10, length_pct=80, gamma=1.4,
PHYSICS_WEIGHT=1.0, couple_wall=True, kernel_d_fraction_max=0.7).

Status after the J4 spike (June 2026) — the plan's §10 diagnosis branch fired,
and productively.  Exact-Jacobian LM removed the FD-noise confound and the
stall decomposed into three layers, two of which are now fixed:

    max_scaled ~ 8      scipy FD Jacobian + degenerate kernel   (pre-spike)
    max_scaled ~ 2.8-3.5  exact-Jacobian LM, same degenerate kernel
    max_scaled ~ 2.5    after the KLThroat integer-division fix
                        (kernel march now runs; BD is a real RRC —
                        tests/test_nasa_kernel_march_parity.py)
    max_scaled ~ 0.5-0.7  after the upstream-radius (Ru) fix + sane
                        theta_B seed: real kernel, real BD anchor

The remaining gap to the 2e-3 gate is *seed topology*, not optimisation:
``calc_lrc_de`` / ``set_theta_b`` still collapse to a degenerate D~E
topology on marched kernels (mass_BD ~ 1e-8, DE of 1-2 nodes), so the CE
seed stays the legacy linear ramp far from the Rao basin.  That is
REWRITE_PLAN Phase-12 work (NASA find_point_e / CalcLRCDE), tracked by the
xfail below.  Homotopy over PHYSICS_WEIGHT, LM restarts, and both bound
treatments (sigmoid reparametrisation, vanishing barrier) all land on the
same stall point — evidence the obstruction is structural, exactly what
§10 predicted exact Jacobians would reveal.

The non-xfail tests pin today's gains as regression floors.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optimistix")

import raosim.rao_variational as rv  # noqa: E402
from raosim.rao_variational import ContourReliability, RaoSolverConfig  # noqa: E402


def _reference_config(**overrides):
    base = dict(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=12, n_kernel=24, n_wall=12,
        max_nfev=800, residual_tol=2e-3, evaluate_moc=False,
        couple_wall=True, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=24.0, solver_backend="jax",
    )
    base.update(overrides)
    return RaoSolverConfig(**base)


@pytest.fixture(scope="module")
def jax_solution_weight1():
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        yield rv.solve_rao_bvp(_reference_config())
    finally:
        rv.PHYSICS_WEIGHT = original


# --------------------------------------------------------------------------- #
# regression floors: what the spike achieved must not regress                  #
# --------------------------------------------------------------------------- #
def test_jax_backend_runs_end_to_end(jax_solution_weight1):
    sol = jax_solution_weight1
    assert sol.control_surface.solver_message.startswith("optimistix")
    assert np.isfinite(sol.residuals.max_scaled)
    # RaoSolution shape unchanged: downstream consumers read these fields.
    assert sol.wall_export.shape[1] == 2
    assert sol.construction_diagnostics["mass_closure"]["method"] == (
        "kernel_bd_curve_flux"
    )


def test_jax_backend_beats_pre_spike_stall(jax_solution_weight1):
    """Progression this floor protects: scipy + degenerate kernel ~8;
    exact-Jacobian LM on the degenerate kernel ~2.8; marched kernel +
    linear seed ~0.5-0.7; marched kernel + NASA fixed-end topology seed
    ~0.49 with the *physics* blocks (stationarity, C±, CE↔wall C+) at
    ~3e-2 and the residual concentrated in length/wall-endpoint.  Floor
    at 0.8 trips if the kernel march, topology seed, or backend regress."""
    sol = jax_solution_weight1
    assert sol.residuals.max_scaled < 0.8, (
        f"max_scaled={sol.residuals.max_scaled:.3g}; the JAX backend has "
        "regressed (kernel march / topology seed / solver)"
    )
    # The topology seed must leave the Rao physics nearly satisfied at the
    # solution — that structure (physics solved, misfit in length) is the
    # signature distinguishing today's state from every earlier stall.
    groups = {g["name"]: g for g in sol.residuals.group_summaries}
    assert groups["algebraic_stationarity"]["max"] < 0.2
    assert groups["moc_cplus"]["max"] < 0.2
    assert groups["moc_cminus"]["max"] < 0.2


def test_mass_closure_uses_real_kernel_bd(jax_solution_weight1):
    """The BD anchor must be a marched RRC, not the arc+sonic fallback.

    Degenerate-kernel signature was kernel_d_fraction -> 0.7 cap with D on
    a vertical sonic line at x=0 (M=1.0, theta=0).  With a real kernel the
    solved D sits at a supersonic interior point."""
    sol = jax_solution_weight1
    D = sol.construction_diagnostics["mass_closure"]["kernel_D"]
    assert D is not None
    assert D["M"] > 1.05, f"D is (near-)sonic: M={D['M']:.4f} — fallback BD?"
    assert abs(sol.residuals.mass_residual_rel) < 0.5


def test_backend_validation_rejects_unknown():
    with pytest.raises(ValueError, match="solver_backend"):
        rv.solve_rao_bvp(_reference_config(solver_backend="tensorflow"))


def test_jax_backend_is_default():
    """DIRECTION item 2 (2026-06-11): flipped after the J4 gate
    re-confirmed at 7.50e-4 on the post-12.4 seed."""
    assert RaoSolverConfig(Rt=0.02, epsilon=10.0).solver_backend == "jax"


def test_numpy_backend_still_available_opt_in():
    cfg = RaoSolverConfig(Rt=0.02, epsilon=10.0, solver_backend="numpy")
    assert cfg.solver_backend == "numpy"


# --------------------------------------------------------------------------- #
# characteristic formulation (converged-topology block set)                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def jax_characteristic_weight1():
    """The formulation fix + constraint-weight ladder (June 2026).

    Drops the two scaffold blocks that are structurally unsatisfiable at
    the Rao topology (C− applied along the C+ CE; CE→wall C+ pairing —
    see CHARACTERISTIC_RAO_RESIDUAL_BLOCKS' docstring for the literature
    grounding), and up-weights the integral constraints along a
    (1, 10, 30) ladder inside the JAX LM solve."""
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        yield rv.solve_rao_bvp(_reference_config(
            n_control=16, couple_wall=False, max_nfev=2500,
            formulation="characteristic",
            jax_constraint_weight_ladder=(1.0, 10.0, 30.0),
        ))
    finally:
        rv.PHYSICS_WEIGHT = original


def test_characteristic_formulation_reaches_3e3_floor(jax_characteristic_weight1):
    """Regression floor for the characteristic formulation + the
    exit-station length fix (Rao's L = z_C + ∫cot(φ)dr; the legacy
    Σdx = x_E − x_D double-counted x_D against the x_E pin and drove
    kernel_d_fraction to the throat plane).  Observed: max_scaled
    ~3.0e-3 (stationarity-only), mass ~5e-9, length ~3e-9, D interior
    at kdf ~0.27, resolution-independent (n=16/24, n_kernel=24/48), and
    theta_B-insensitive (refresh to the kernel-stationarity angle
    21.87 deg changes nothing — NOTE: that value is the stationarity
    diagnostic, NOT the chart theta_N, which is 30.0 deg at eps=10/L80;
    see the 2026-06-11 theta_N reconciliation in the plan STATUS) —
    floors at ~3x observed."""
    sol = jax_characteristic_weight1
    r = sol.residuals
    assert r.max_scaled < 0.01, f"max_scaled={r.max_scaled:.3g}"
    assert abs(r.mass_residual_rel) < 1e-3
    assert abs(r.length_residual_rel) < 1e-3
    # D must sit interior on the BD (the classical 0.3-0.6 band) — the
    # degenerate kdf -> 0.02 drift was the length-bookkeeping bug.
    assert 0.05 < sol.control_surface.kernel_d_fraction < 0.7
    groups = {g["name"]: g for g in r.group_summaries}
    assert groups["moc_cplus"]["max"] < 5e-3, (
        "the CE stopped being a consistent C+ characteristic"
    )
    # The dropped blocks must actually be absent (zero-length).
    assert groups["moc_cminus"]["count"] == 0
    assert groups["cplus_ce_to_wall"]["count"] == 0
    assert groups["wall_intersection"]["count"] == 0


def test_characteristic_formulation_is_default():
    """DIRECTION item 2c bundle (2026-06-11): characteristic formulation,
    J4 ladder, and full D-state continuity are the defaults."""
    cfg = RaoSolverConfig(Rt=0.02, epsilon=10.0)
    assert cfg.formulation == "characteristic"
    assert cfg.jax_constraint_weight_ladder == (1.0, 10.0, 30.0, 100.0)
    assert cfg.pin_d_theta is True
    assert cfg.pin_d_mach is True
    blocks = rv._enabled_residual_blocks(cfg)
    # The structurally-unsatisfiable scaffold blocks are gone by default.
    assert "moc_cminus" not in blocks
    assert "cplus_ce_to_wall" not in blocks


def test_legacy_formulation_still_available_opt_in():
    cfg = RaoSolverConfig(Rt=0.02, epsilon=10.0, formulation="legacy",
                          pin_d_theta=True,
                          jax_constraint_weight_ladder=None)
    assert cfg.formulation == "legacy"
    blocks = rv._enabled_residual_blocks(cfg)
    assert "moc_cminus" in blocks  # legacy stack reachable opt-in


def test_converged_solution_yields_closed_bde_wall():
    """End-to-end manufacturable-contour milestone: the gate-passing solve
    plus ``wall_method='bde'`` produces a monotone wall whose last point
    lands exactly on the commanded exit (L, Re) with a complete BFE mesh."""
    import math
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        sol = rv.solve_rao_bvp(_reference_config(
            n_control=24, couple_wall=False, max_nfev=4000,
            thetaN_guess_deg=21.87, evaluate_moc=True, wall_method="bde",
            formulation="characteristic", pin_d_theta=True,
            pin_d_mach=True,
            jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
        ))
    finally:
        rv.PHYSICS_WEIGHT = original
    assert sol.residuals.max_scaled <= 2e-3
    w = sol.wall_raw
    L = rv._target_length(0.020, 10.0, 80.0)
    Re = math.sqrt(10.0) * 0.020
    d = sol.construction_diagnostics
    assert d["bfe_complete_remaining_mesh"] and d["bfe_wall_contour_complete"]
    assert w.shape[0] > 50
    assert abs(w[-1, 0] - L) / L < 1e-3, f"wall exit x off: {w[-1,0]:.5f} vs {L:.5f}"
    assert abs(w[-1, 1] - Re) / Re < 1e-3, f"wall exit r off: {w[-1,1]:.5f} vs {Re:.5f}"
    assert np.all(np.diff(w[:, 1]) >= -1e-9), "wall radius not monotone"
    assert np.all(np.diff(w[:, 0]) >= -1e-9), "wall x not monotone"


def test_bde_wall_is_bell_shaped():
    """The corrected smooth solution must produce a bell, not a flare.

    The angle targets come from the solver-independent stationary-DE
    existence root.  The Rao chart values are parabola-fit parameters,
    not exact boundary-state constraints for this construction.
    """
    theta_b_smooth = 25.5659
    theta_e_smooth = 11.1193
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        sol = rv.solve_rao_bvp(_reference_config(
            n_control=24, couple_wall=False, max_nfev=4000,
            thetaN_guess_deg=21.87, evaluate_moc=True, wall_method="bde",
            formulation="characteristic", pin_d_theta=True,
            pin_d_mach=True,
            jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
        ))
    finally:
        rv.PHYSICS_WEIGHT = original
    w = sol.wall_raw
    ang = np.degrees(np.arctan2(np.diff(w[:, 1]), np.diff(w[:, 0])))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(w[:, 0]),
                                                  np.diff(w[:, 1])))])
    i_peak = int(np.argmax(ang))
    # Peak magnitude follows the solved kernel angle and occurs just
    # after the throat arc.
    assert ang.max() == pytest.approx(theta_b_smooth, abs=2.5), (
        f"wall peak {ang.max():.1f} deg vs smooth theta_B "
        f"{theta_b_smooth:.1f} deg"
    )
    assert s[i_peak] / s[-1] < 0.10, (
        f"peak at {s[i_peak] / s[-1]:.0%} of length — mid-bell flare"
    )
    post = ang[i_peak:]
    assert np.all(np.diff(post) <= 0.25), "wall angle not monotone decreasing"
    assert ang[-1] == pytest.approx(theta_e_smooth, abs=2.5), (
        f"exit angle {ang[-1]:.1f} deg vs smooth theta_E "
        f"{theta_e_smooth:.1f} deg"
    )


def test_position_only_diagnostic_closes_but_moves_d_toward_b():
    """The relaxed D-state branch is numerical diagnostic only.

    Characteristic formulation + exit-station length + constraint-weight
    ladder + position-only D attachment can still close algebraically,
    but after the characteristic correction it drives D toward B.  That
    is why this branch must not be used as the physical default.
    """
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        sol = rv.solve_rao_bvp(_reference_config(
            n_control=24, couple_wall=False, max_nfev=4000,
            thetaN_guess_deg=21.87,
            formulation="characteristic",
            pin_d_theta=False,
            pin_d_mach=False,
            jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
        ))
    finally:
        rv.PHYSICS_WEIGHT = original
    r = sol.residuals
    assert r.max_scaled <= 2e-3, (
        f"J4 gate regressed: max_scaled={r.max_scaled:.4g} > 2e-3"
    )
    assert abs(r.mass_residual_rel) < 1e-6
    assert abs(r.length_residual_rel) < 1e-6
    assert 0.0 < sol.control_surface.kernel_d_fraction < 0.15
    assert sol.control_surface.converged


# --------------------------------------------------------------------------- #
# the J4 gate itself                                                           #
# --------------------------------------------------------------------------- #
def test_j4_residual_gate_reference_case_converges(jax_solution_weight1):
    """The BVP gate passes even when the optional wall audit is skipped."""
    sol = jax_solution_weight1
    assert sol.residuals.max_scaled <= 2e-3
    assert sol.control_surface.converged
    assert sol.reliability == ContourReliability.GEOMETRIC_APPROXIMATION
    assert "MOC wall evaluation skipped" in " ".join(sol.warnings)
