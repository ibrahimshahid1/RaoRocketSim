"""
tests/test_mdo_nlp.py — Phase 8/9 gate for the ε-constraint NLP.

Checks the machinery the plan's Phase-9 gate needs: exact constraint Jacobians
(the Phase-8 deliverable — jacfwd through the two IFT solves matches finite
differences), a single min-mass solve reaching a feasible KKT point over the
enforced constraint set, and that coking is *reported* (not silently dropped).

The full multi-point Pareto sweep is host-only (compile + SLSQP over several
points exceeds the sandbox per-call budget); it is exercised via the CLI.  These
tests keep a single compile by fixing one Isp floor.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec
from raosim.mdo.nlp import (
    solve_min_mass, _make_callables, DEFAULT_ENFORCED, CONSTRAINT_NAMES,
)

_M = MissionSpec()
_ISP = 190.0        # feasible with coking enforced (film cooling costs Isp)
_NX = 10            # design-vector dimension
_IDX = tuple(CONSTRAINT_NAMES.index(n) for n in DEFAULT_ENFORCED)


@pytest.fixture(scope="module")
def callables():
    return _make_callables(_M, _ISP, False, _IDX)


def test_constraint_jacobian_matches_fd(callables):
    """Exact forward-mode constraint Jacobian ≈ central differences (plan §4.3):
    the derivatives flow through both IFT solves with no step-size noise."""
    ss, obj, obj_grad, con, con_jac = callables
    u = np.full(_NX, 0.5)
    J = np.asarray(con_jac(jnp.asarray(u)))
    # The primal contains two tightly converged implicit roots.  A unit-box
    # step of 1e-6 is below their useful subtraction scale and measures solver
    # roundoff, not truncation error; 1e-4 is in the observed central-difference
    # plateau while remaining far from chart knots at this fixture.
    h = 1e-4
    for k in range(_NX):
        up, um = u.copy(), u.copy()
        up[k] += h
        um[k] -= h
        fd = (np.asarray(con(jnp.asarray(up))) - np.asarray(con(jnp.asarray(um)))) / (2 * h)
        denom = np.maximum(np.abs(fd), 1e-6)
        assert np.max(np.abs(J[:, k] - fd) / denom) < 1e-5, f"col {k}"


def test_objective_gradient_matches_fd(callables):
    ss, obj, obj_grad, con, con_jac = callables
    u = np.full(_NX, 0.5)
    g = np.asarray(obj_grad(jnp.asarray(u)))
    h = 1e-6
    for k in range(_NX):
        up, um = u.copy(), u.copy()
        up[k] += h
        um[k] -= h
        fd = (float(obj(jnp.asarray(up))) - float(obj(jnp.asarray(um)))) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_min_mass_solve_feasible_with_coking_enforced():
    """One min-mass solve over the 10-var space with coking HARD-enforced: it
    reaches a feasible KKT point by dialing in film cooling, so the coking
    margin is satisfied (≥0) and the film fraction is active (>0).  A genuinely
    thermal-limited optimum.  Single solve to stay within the compile budget."""
    assert "coking" in DEFAULT_ENFORCED     # now enforced via film cooling
    r = solve_min_mass(_M, _ISP, maxiter=100)
    assert r.feasible                       # ALL enforced constraints, incl coking
    assert r.max_violation < 1e-5
    assert r.constraints["coking"] >= -1e-3  # coking satisfied (was infeasible w/o film)
    assert r.design["film_frac"] > 1e-3     # film cooling is the active lever
    assert r.Isp >= _ISP - 1e-3             # ε-constraint respected
    assert r.objective_mass >= r.exact_electric_package_mass
    # Compatibility remains available but is explicitly the smooth objective.
    assert r.package_mass == pytest.approx(r.objective_mass)
