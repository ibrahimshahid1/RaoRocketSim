"""
tests/test_mdo_of_variable.py — O/F as a design variable (plan item 1).

SP-125 §2.1 lists mixture ratio among the nine requirement parameters, but its
own derivation makes it an *output*: the optimum balances energy release
against molecular weight and is then moved off that optimum for cooling —
"The temperatures resulting from stoichiometric or near-stoichiometric mixture
ratios ... may impose severe demands on the chamber-wall cooling system.  A
lower temperature, therefore, may be desired and obtained by selecting a
suitable ratio."  That is the trade this optimiser resolves, so O/F belongs on
the design vector.

The property surfaces are what make it real.  With constant gamma/T_c/R the
thermochemistry is FLAT in O/F, so moving O/F would change the propellant mass
split without changing combustion — a wrong model, not a coarse one.  These
tests pin the two-regime behaviour:

* no CEA table  -> O/F is **absent** from the design space, and the solver uses
  the mission's fixed value;
* CEA table     -> O/F is the eleventh variable, bounded by the sampled domain,
  and it is **alive** (non-zero constraint/objective sensitivity).

The synthetic surface below is not a physics claim — it exists only to give the
wiring something with real (Pc, O/F) curvature to differentiate through.  The
production table comes from ``scripts/sample_cea_surface.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.schema import (
    DesignVector, MissionSpec, default_design_space,
)

_THRUST = 5.0e3
_OF_LO, _OF_HI = 1.8, 3.0
_PC_LO, _PC_HI = 1.5e6, 6.0e6


@pytest.fixture(scope="module")
def synthetic_table(tmp_path_factory):
    """A C1-sampleable stand-in with realistic O/F curvature.

    Shapes only, not validated values: T_c peaks near O/F ~= 2.6 for LOX/RP-1
    and falls either side; molecular weight rises with O/F; gamma drifts down
    as T_c rises.  Enough structure that a flat-surface bug cannot pass.
    """
    from raosim.mdo.properties import save_tables

    Pc = np.linspace(_PC_LO, _PC_HI, 9)
    OF = np.linspace(_OF_LO, _OF_HI, 9)
    P, O = np.meshgrid(Pc, OF, indexing="ij")

    Tc = 3700.0 - 900.0 * (O - 2.6) ** 2 + 40.0 * (P / 1.0e6 - 3.0)
    Mw = 21.0 + 1.6 * (O - 1.8)                      # g/mol, rising with O/F
    R_gas = 8314.462618 / Mw
    gamma = 1.26 - 0.02 * (Tc - 3400.0) / 300.0

    path = tmp_path_factory.mktemp("cea") / "synthetic_lox_rp1.npz"
    save_tables(str(path), {"Pc_grid": Pc, "OF_grid": OF, "gamma": gamma,
                            "Tc": Tc, "R_gas": R_gas},
                oxidizer="LOX", fuel="RP-1")
    return str(path)


# --------------------------------------------------------------------------- #
# Regime A — no surfaces: O/F must NOT be offered as a lever                   #
# --------------------------------------------------------------------------- #
def test_of_absent_from_design_space_without_surfaces():
    m = MissionSpec.for_thrust(_THRUST)
    names = [s.name for s in default_design_space(m)]
    assert "OF" not in names
    assert len(names) == 10


def test_solver_uses_mission_of_when_not_a_variable():
    """The class default is a -1 sentinel, so if the solver ever read it
    instead of the mission the flow split would go negative and this test
    would fail loudly rather than drift quietly."""
    from raosim.mdo.engine import solve_engine

    m = MissionSpec.for_propellant("LOX/LCH4", _THRUST)      # OF_default 3.5
    x = DesignVector.from_array(
        jnp.asarray([s.ref() for s in default_design_space(m)]))
    assert x.of_is_variable is False
    r = solve_engine(x, m)
    assert bool(r.finite)
    assert float(r.OF) == pytest.approx(m.OF)
    assert float(r.OF) == pytest.approx(3.5)


def test_short_array_never_activates_the_sentinel():
    x = DesignVector.from_array(jnp.asarray([3.0e6, 8.0, 0.2, 0.2]))
    assert x.of_is_variable is False
    assert float(x.OF) < 0.0        # sentinel, and it must stay unused


# --------------------------------------------------------------------------- #
# Regime B — surfaces loaded: O/F becomes the eleventh variable                #
# --------------------------------------------------------------------------- #
def test_of_enters_the_design_space_with_surfaces(synthetic_table):
    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    space = default_design_space(m)
    names = [s.name for s in space]
    assert names[-1] == "OF"
    assert len(names) == 11


def test_of_bounds_come_from_the_sampled_domain(synthetic_table):
    """Bounding O/F at the sampled box (rather than a physical band) keeps the
    bound and ``property_domain_margin`` from being two sources of truth about
    the same limit."""
    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    of_spec = [s for s in default_design_space(m) if s.name == "OF"][0]
    assert of_spec.lower == pytest.approx(_OF_LO)
    assert of_spec.upper == pytest.approx(_OF_HI)
    assert _OF_LO <= of_spec.ref() <= _OF_HI


def test_solver_uses_the_design_of_when_it_is_a_variable(synthetic_table):
    from raosim.mdo.engine import solve_engine

    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    arr = [s.ref() for s in default_design_space(m)]
    arr[-1] = 2.45                                   # deliberately != mission.OF
    x = DesignVector.from_array(jnp.asarray(arr))
    assert x.of_is_variable is True
    r = solve_engine(x, m)
    assert bool(r.finite)
    assert float(r.OF) == pytest.approx(2.45)
    assert abs(float(r.OF) - m.OF) > 1e-3            # not the mission's value


def test_thermochemistry_actually_responds_to_of(synthetic_table):
    """The whole point: with surfaces loaded, moving O/F must move c* and T_c.
    A flat response here means the surfaces were bypassed."""
    from raosim.mdo.engine import solve_engine

    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    base = [s.ref() for s in default_design_space(m)]

    def isp_at(of):
        arr = list(base)
        arr[-1] = of
        return float(solve_engine(
            DesignVector.from_array(jnp.asarray(arr)), m).Isp)

    lo, mid, hi = isp_at(2.0), isp_at(2.6), isp_at(2.9)
    assert abs(mid - lo) > 1.0        # seconds — a real response, not roundoff
    assert abs(mid - hi) > 1.0
    assert mid > lo and mid > hi      # peaked, matching the synthetic T_c shape


def test_of_is_not_a_dead_variable(synthetic_table):
    """The MDO_GUIDE dead-variable check, applied to the new column: if both
    the objective gradient and the constraint Jacobian column are zero, the
    variable is decorative."""
    from raosim.mdo.nlp import (
        _make_callables, CONSTRAINT_NAMES, DEFAULT_ENFORCED,
    )

    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    space = default_design_space(m)
    idx = tuple(CONSTRAINT_NAMES.index(n) for n in DEFAULT_ENFORCED)
    ss, obj, obj_grad, con, con_jac = _make_callables(m, 190.0, False, idx)

    u = np.full(len(space), 0.5)
    J = np.asarray(con_jac(jnp.asarray(u)))
    g = np.asarray(obj_grad(jnp.asarray(u)))
    col = [s.name for s in space].index("OF")

    assert J.shape == (len(CONSTRAINT_NAMES), len(space))
    assert np.all(np.isfinite(J[:, col]))
    assert np.linalg.norm(J[:, col]) > 0.0, "O/F has no constraint sensitivity"
    assert np.isfinite(g[col])


def test_host_bridge_reads_the_design_of_not_the_mission(synthetic_table):
    """``postprocess`` must hand the authoritative workflow the O/F that was
    optimised.  Re-deriving it from the mission would run a *different engine*
    through design_nozzle_v2 than the one the optimiser produced."""
    from raosim.mdo.postprocess import _effective_of

    m = MissionSpec.for_thrust(_THRUST, cea_table_path=synthetic_table)
    assert _effective_of({"Pc": 3.0e6, "OF": 2.45}, m) == pytest.approx(2.45)
    # and it falls back cleanly when O/F was not a variable
    assert _effective_of({"Pc": 3.0e6}, m) == pytest.approx(m.OF)
