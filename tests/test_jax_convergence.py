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
    """Pre-spike: scipy stalled at max_scaled ~ 8 (Phase-6 xfail text);
    exact Jacobians on the degenerate kernel: ~2.8.  With the marched
    kernel the solve must stay below 1.5 — a coarse floor that trips if
    the kernel march or the backend regresses."""
    sol = jax_solution_weight1
    assert sol.residuals.max_scaled < 1.5, (
        f"max_scaled={sol.residuals.max_scaled:.3g}; the JAX backend has "
        "regressed toward the pre-spike stall (kernel march broken again?)"
    )


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
# the J4 gate itself                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    strict=False,
    reason=(
        "J4 gate (max_scaled <= 2e-3): blocked by seed topology, not by the "
        "optimizer — exact-Jacobian LM reduced the stall 8 -> ~0.5 after the "
        "KLThroat int-division and upstream-radius kernel fixes, and all "
        "optimizer-side strategies (PHYSICS_WEIGHT homotopy, LM restarts, "
        "sigmoid vs barrier bounds) converge to the same point.  Remaining "
        "blocker: calc_lrc_de/set_theta_b collapse to a degenerate D~E "
        "topology on marched kernels, leaving the CE seed on the legacy "
        "linear ramp outside the Rao basin (REWRITE_PLAN Phase 12; "
        "JAX_DIFFERENTIABLE_PLAN §10 diagnosis branch)."
    ),
)
def test_j4_gate_reference_case_converges(jax_solution_weight1):
    sol = jax_solution_weight1
    assert sol.residuals.max_scaled <= 2e-3
    assert sol.reliability in (
        ContourReliability.RAO_VARIATIONAL_RESIDUAL_SOLVED,
        ContourReliability.BENCHMARK_VALIDATED,
    )
