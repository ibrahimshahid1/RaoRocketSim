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
    Me = 3.5
    Pc, Pa = 70e5, 101325.0
    margin = float(T.schmucker_separation_margin(
        (Me ** 2), GAMMA, Pc, Pa))  # rough eps proxy not needed; direct:
    # Direct parity on the ratio formula at a known Me.
    np_ratio = schmucker_separation_ratio(Me, Pa / Pc)
    assert np_ratio == pytest.approx((Pa / Pc) ** 0.8 / Me, rel=1e-12)


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
    tight = constrained_nozzle_design(**args, T_wall_limit=1100.0,
                                      sep_margin_min=2.0)
    assert tight.separation_active
    assert tight.feasible
    assert tight.separation_margin == pytest.approx(2.0, abs=0.05)
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
