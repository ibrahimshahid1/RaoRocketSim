"""
Constrained differentiable nozzle design (raosim.jax.thermal +
raosim.jax.design_opt).

* the JAX thermal physics matches the NumPy originals (Bartz σ /
  Sieder-Tate / Schmucker) at the float-parity level;
* gradients are exact (jacfwd vs central FD);
* the constrained optimiser (Optimistix BFGS + penalty) maximises Cf
  and lets the constraints SHAPE the design: loose -> unconstrained Cf
  optimum; tight separation -> ε capped at the limit; tight cooling ->
  throat opened (and infeasibility reported when geometry alone can't
  satisfy it, the physical finding from raosim.thermal_design).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from raosim.jax import thermal as T  # noqa: E402
from raosim.jax.design_opt import (  # noqa: E402
    constrained_nozzle_design,
    design_gradients,
    normalize_objective_weights,
    thrust_targeted_design,
)
from raosim.physics import (  # noqa: E402
    bartz_sigma as np_bartz_sigma,
    gas_transport_properties,
)
from raosim.propellants import custom_propellant
from raosim.separation import schmucker_separation_ratio  # noqa: E402

GAMMA = 1.24


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=GAMMA, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def args(prop):
    cp, Pr, mu = gas_transport_properties(prop)
    return dict(
        Rt=0.020, Pc=70e5, Pa=101325.0, gamma=GAMMA, c_star=prop.c_star,
        Tc=3500.0, cp_gas=cp, Pr_gas=Pr, mu_gas=mu, h_c=80e3,
        coolant_temperature=320.0, t_wall=0.001, k_wall=350.0,
    )


# ---------------------------------------------------------------------
#  JAX physics parity with the NumPy originals.
# ---------------------------------------------------------------------


def test_bartz_sigma_parity():
    for M in (1.0, 2.0, 3.5):
        assert float(T.bartz_sigma(M, GAMMA, 900.0, 3500.0)) == pytest.approx(
            float(np_bartz_sigma(M, GAMMA, 900.0, 3500.0)), rel=1e-12)


def test_schmucker_parity():
    # Re-baselined 2026-07-22: criterion corrected to Östlund (2002) Eq. 30,
    # p_sep/Pa = (1.88·Me − 1)^(−0.64)  (was a cross-labeled (Pa/Pc)^0.8/Me;
    # see docs/DIFFERENTIABLE_MDO_PLAN_EVALUATION_2026-07-22.md §A.2.1).
    eps = 12.0
    Pc, Pa = 70e5, 101325.0
    from raosim.gas_dynamics import (
        isentropic_pressure_ratio as np_pr,
        mach_from_area_ratio as np_mach,
    )
    Me = np_mach(eps, GAMMA, supersonic=True)
    # Corrected closed form, pinned.
    np_ratio = schmucker_separation_ratio(Me, Pa / Pc)
    assert np_ratio == pytest.approx(
        (1.88 * Me - 1.0) ** (-0.64) * (Pa / Pc), rel=1e-12)
    # JAX helper preserves the raw attached-flow ratio for its legacy API.
    np_margin = np_pr(Me, GAMMA) / np_ratio
    jx_margin = float(T.schmucker_separation_margin(eps, GAMMA, Pc, Pa))
    assert jx_margin == pytest.approx(np_margin, rel=1e-8)
    # The ambient-referenced correlation degenerates in vacuum; the MDO
    # margin must remain finite and positive rather than emit inf/nan.
    vacuum = float(T.schmucker_separation_margin(eps, GAMMA, Pc, 0.0))
    assert np.isfinite(vacuum) and vacuum > 0.0


def test_thrust_coefficient_has_interior_optimum(args):
    Cf = lambda e: float(T.ambient_thrust_coefficient(
        e, GAMMA, args["Pc"], args["Pa"]))
    # Cf rises then falls with ε at fixed Pa (optimum expansion).
    assert Cf(8.0) > Cf(4.0)
    assert Cf(8.0) > Cf(20.0)


# ---------------------------------------------------------------------
#  Exact gradients.
# ---------------------------------------------------------------------


def test_design_gradients_match_finite_difference(args):
    a = dict(args, T_wall_limit=1100.0)
    eps, rd = 8.31, 0.5
    g = design_gradients(a, eps, rd)

    def fd_eps(fn, de=1e-4):
        return (fn(eps + de) - fn(eps - de)) / (2 * de)

    Cf = lambda e: float(T.ambient_thrust_coefficient(e, GAMMA, a["Pc"], a["Pa"]))
    sep = lambda e: float(T.schmucker_separation_margin(e, GAMMA, a["Pc"], a["Pa"]))
    assert g["Cf"]["d_d_epsilon"] == pytest.approx(fd_eps(Cf), abs=1e-7)
    assert g["separation_margin"]["d_d_epsilon"] == pytest.approx(
        fd_eps(sep), rel=1e-5)
    # Opening the throat cools (dT_wg/dRd < 0).
    assert g["throat_wall_temperature"]["d_d_Rd_factor"] < 0.0


# ---------------------------------------------------------------------
#  Constrained optimisation: constraints shape the design.
# ---------------------------------------------------------------------


def test_loose_constraints_reach_unconstrained_cf_optimum(args):
    r = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                  sep_margin_min=1.0)
    assert r.feasible
    assert not r.separation_active
    # Cf optimum near ε ≈ 8 for 70 bar at sea level.
    assert r.epsilon == pytest.approx(8.3, abs=1.5)
    # And it is a true optimum: dCf/dε ≈ 0 there.
    g = design_gradients(dict(args, T_wall_limit=1100.0),
                         r.epsilon, r.Rd_factor)
    assert abs(g["Cf"]["d_d_epsilon"]) < 1e-3


def test_tight_separation_constraint_caps_expansion(args):
    loose = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                      sep_margin_min=1.0)
    # Threshold re-baselined 2026-07-22: the corrected Schmucker criterion
    # (Östlund Eq. 30) gives ~2.7 at this case's unconstrained Cf optimum
    # (the old cross-labeled form gave ~1.34), so "tight" is now 4.0.
    tight = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                      sep_margin_min=4.0)
    assert tight.separation_active
    assert tight.feasible
    assert tight.separation_margin == pytest.approx(4.0, abs=0.1)
    # The constraint pushed ε below the unconstrained optimum...
    assert tight.epsilon < loose.epsilon
    # ... at a small Cf cost.
    assert tight.Cf < loose.Cf


def test_tight_cooling_opens_throat_then_reports_infeasible(args):
    loose = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                      sep_margin_min=1.0)
    tight = constrained_nozzle_design(**args, T_wall_limit=900.0,
                                      sep_margin_min=1.0)
    # Cooling pressure opens the throat curvature (weak lever)...
    assert tight.Rd_factor > loose.Rd_factor
    # ... but geometry alone can't reach 900 K here -> infeasible,
    # reproducing the thermal_design finding rigorously.
    assert not tight.feasible
    assert tight.cooling_active


def test_explicit_heat_flux_limit_constraint(args):
    # A q limit below the loose-design throat flux must drive the throat
    # open (q ∝ (Dt/rc)^0.1 falls as rc=Rd·Rt grows).
    loose = constrained_nozzle_design(**args, T_wall_limit=1100.0)
    q_target = 0.9 * loose.throat_heat_flux
    r = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                  q_limit=q_target)
    assert r.Rd_factor > loose.Rd_factor
    assert r.throat_heat_flux < loose.throat_heat_flux


# ---------------------------------------------------------------------
#  Fixed-thrust, multi-objective design.
# ---------------------------------------------------------------------


def _target_args(args):
    return dict(
        target_thrust=10_000.0,
        Pc=args["Pc"], Pa=args["Pa"], gamma=GAMMA,
        c_star=args["c_star"], Tc=args["Tc"],
        cp_gas=args["cp_gas"], Pr_gas=args["Pr_gas"], mu_gas=args["mu_gas"],
        h_c=args["h_c"], coolant_temperature=args["coolant_temperature"],
        t_wall=args["t_wall"], k_wall=args["k_wall"], T_wall_limit=1100.0,
    )


def test_thrust_targeted_design_meets_target_by_construction(args):
    r = thrust_targeted_design(**_target_args(args), objectives={"cf": 1.0})
    assert r.feasible
    assert r.positive_thrust_coefficient
    assert r.thrust == pytest.approx(r.target_thrust, rel=1e-12)
    assert r.At == pytest.approx(np.pi * r.Rt ** 2, rel=1e-12)
    assert r.exit_radius == pytest.approx(np.sqrt(r.epsilon) * r.Rt, rel=1e-12)
    assert set(r.objective_values) == {"cf", "isp", "length", "mass"}
    assert r.scalar_objective == pytest.approx(sum(r.objective_terms.values()))


def test_thrust_targeted_objective_changes_geometry(args):
    cf = thrust_targeted_design(**_target_args(args), objectives={"cf": 1.0})
    compact = thrust_targeted_design(
        **_target_args(args), objectives={"length": 1.0})
    assert cf.epsilon > compact.epsilon
    assert cf.Cf > compact.Cf
    assert compact.nozzle_length < cf.nozzle_length
    assert compact.thrust == pytest.approx(10_000.0, rel=1e-12)


def test_thrust_targeted_input_and_objective_validation(args):
    assert normalize_objective_weights({"CF": 2, "length": 0}) == {"cf": 2.0}
    with pytest.raises(ValueError, match="unknown objective"):
        normalize_objective_weights({"banana": 1})
    with pytest.raises(ValueError, match="target_thrust"):
        thrust_targeted_design(**dict(_target_args(args), target_thrust=0.0))
    with pytest.raises(ValueError, match="eps_bounds"):
        thrust_targeted_design(**_target_args(args), eps_bounds=(0.9, 20.0))
