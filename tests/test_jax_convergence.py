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


def test_numpy_backend_unchanged_default():
    assert RaoSolverConfig(Rt=0.02, epsilon=10.0).solver_backend == "numpy"


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
    theta_B-insensitive (refresh to the chart angle 21.87 deg changes
    nothing) — floors at ~3x observed."""
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


def test_legacy_formulation_is_unchanged_default():
    cfg = RaoSolverConfig(Rt=0.02, epsilon=10.0)
    assert cfg.formulation == "legacy"
    assert cfg.jax_constraint_weight_ladder is None
    blocks = rv._enabled_residual_blocks(cfg)
    assert "moc_cminus" in blocks  # legacy stack untouched by default


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
            formulation="characteristic", pin_d_theta=False,
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "BDE wall SHAPE defect: the converged solve's wall ends exactly on "
        "the commanded exit and is monotone in (x, r), but the wall angle "
        "RISES along the BFE section (23 -> 35.6 deg) and kinks down to "
        "4.6 deg on the final segment — a flare, not a bell (a Rao bell "
        "peaks at theta_N ~ 22 deg after the throat arc and decreases "
        "monotonically to theta_E ~ 8-10 deg).  The converged DE itself is "
        "bell-consistent (theta 13.6 -> 7.3 deg), so the defect is in "
        "calc_bde_region/_calc_wall_contour_rows wall-point placement — "
        "consistent with the forward-MOC audit's crossings.  Next work item."
    ),
)
def test_bde_wall_is_bell_shaped():
    """Wall angle must peak near theta_N right after the throat arc and
    decrease monotonically to theta_E ~ 8-10 deg (Rao TOP shape)."""
    import math
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        sol = rv.solve_rao_bvp(_reference_config(
            n_control=24, couple_wall=False, max_nfev=4000,
            thetaN_guess_deg=21.87, evaluate_moc=True, wall_method="bde",
            formulation="characteristic", pin_d_theta=False,
            jax_constraint_weight_ladder=(1.0, 10.0, 30.0, 100.0),
        ))
    finally:
        rv.PHYSICS_WEIGHT = original
    w = sol.wall_raw
    ang = np.degrees(np.arctan2(np.diff(w[:, 1]), np.diff(w[:, 0])))
    # After the throat arc (angle first reaches ~theta_N), the wall angle
    # must never exceed theta_N + 2 deg and must end at 6-12 deg.
    i_peak = int(np.argmax(ang))
    assert ang.max() < 24.0, f"wall flares to {ang.max():.1f} deg"
    post = ang[i_peak:]
    assert np.all(np.diff(post) <= 0.25), "wall angle not monotone decreasing"
    assert 6.0 <= ang[-1] <= 12.0, f"exit angle {ang[-1]:.1f} deg"


def test_j4_gate_passes_with_position_only_attachment():
    """**The J4 gate (max_scaled <= 2e-3), closed.**

    Characteristic formulation + exit-station length + constraint-weight
    ladder + position-only D attachment (``pin_d_theta=False``: r pinned
    to D, theta/M free).  Rationale: the kernel BD is itself approximate
    (KL start line + n_kernel march), so pinning the CE start *angle* to
    an interpolated approximate kernel value imports kernel
    discretization error into the stationarity chain at full weight; the
    kernel's physical role — D's position and the B→D mass budget —
    stays enforced.  Observed: max_scaled ~1.16e-3, mass ~2e-9,
    length ~6e-10, kdf interior at ~0.30 (the classical 0.3-0.6 band).
    """
    original = rv.PHYSICS_WEIGHT
    try:
        rv.PHYSICS_WEIGHT = 1.0
        sol = rv.solve_rao_bvp(_reference_config(
            n_control=24, couple_wall=False, max_nfev=4000,
            thetaN_guess_deg=21.87,
            formulation="characteristic",
            pin_d_theta=False,
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
    assert 0.1 < sol.control_surface.kernel_d_fraction < 0.6
    assert sol.control_surface.converged


# --------------------------------------------------------------------------- #
# the J4 gate itself                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=False,
    reason=(
        "J4 gate (max_scaled <= 2e-3).  Infrastructure blockers are all "
        "cleared: exact-Jacobian LM (8 -> 2.8), KLThroat int-division + "
        "upstream-radius kernel fixes (-> ~0.5), and the NASA fixed-end "
        "topology seed (calc_lrc_de end_condition='fixed_end' + "
        "set_theta_b) now place the BVP in a basin where the Rao physics "
        "blocks solve to ~3e-2.  What remains is the genuine variational "
        "tension: the kernel march's unit-process edge caps theta_B at "
        "~24 deg for Rd=0.382Rt, the fixed-end topology at that cap runs "
        "~9% long, and at PHYSICS_WEIGHT=1.0 LM trades the length "
        "residual (~0.5) against stationarity.  Candidate next levers: "
        "length continuation from the topology's natural length, kernel "
        "march robustness past the theta cap (Phase 12.4 "
        "CalcRRCsAlongArc completion), or transversality/multiplier "
        "blocks to pin the fixed-length optimum (REWRITE_PLAN Phase 6/12)."
    ),
)
def test_j4_gate_reference_case_converges(jax_solution_weight1):
    sol = jax_solution_weight1
    assert sol.residuals.max_scaled <= 2e-3
    assert sol.reliability in (
        ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED,
        ContourReliability.BENCHMARK_VALIDATED,
    )
