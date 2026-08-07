"""Phase-1 gate (plan §11 row 1): schema pytrees + scaling are jit / jacfwd /
jacrev clean with no host callbacks."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from raosim.gas_dynamics import characteristic_velocity  # noqa: E402
from raosim.mdo.scaling import ScaledSpace  # noqa: E402
from raosim.mdo.schema import (  # noqa: E402
    DesignVector,
    MissionSpec,
    default_design_space,
)


def test_design_vector_pytree_roundtrip():
    # 4-keyword construction keeps the whole-engine vars at their defaults
    # (the skeleton-compatible path); the pytree still carries all six leaves.
    x = DesignVector(Pc=jnp.asarray(3.0e6), eps=jnp.asarray(8.0),
                     dp_f_frac=jnp.asarray(0.2), dp_o_frac=jnp.asarray(0.2))
    leaves, treedef = jax.tree_util.tree_flatten(x)
    assert len(leaves) == 11          # 10 hardware vars + O/F
    x2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(x2.eps) == 8.0
    # ``of_is_variable`` is pytree AUX, not a leaf, so it survives the
    # roundtrip without ever entering the traced graph.
    assert x2.of_is_variable is False
    assert float(x2.D_pintle) == pytest.approx(0.020)     # defaulted
    assert float(x2.N_rpm) == pytest.approx(30000.0)      # defaulted
    assert float(x2.film_frac) == pytest.approx(0.0)      # defaulted (pure regen)
    # full 6-long array roundtrip sets all variables
    arr = DesignVector(
        Pc=jnp.asarray(3.0e6), eps=jnp.asarray(8.0), dp_f_frac=jnp.asarray(0.2),
        dp_o_frac=jnp.asarray(0.2), D_pintle=jnp.asarray(0.018),
        N_rpm=jnp.asarray(4.0e4)).to_array()
    x3 = DesignVector.from_array(arr)
    assert float(x3.Pc) == pytest.approx(3.0e6)
    assert float(x3.N_rpm) == pytest.approx(4.0e4)
    # a 4-long array still builds — defaults fill the rest (skeleton path)
    x4 = DesignVector.from_array(jnp.asarray([3.0e6, 8.0, 0.2, 0.2]))
    assert float(x4.D_pintle) == pytest.approx(0.020)
    assert float(x4.film_frac) == pytest.approx(0.0)
    assert DesignVector.names() == (
        "Pc", "eps", "dp_f_frac", "dp_o_frac", "D_pintle", "N_rpm",
        "channel_width", "channel_height", "film_frac", "t_wall", "OF")
    # A short array leaves O/F non-variable, so the solver reads the mission's
    # fixed value instead of the class sentinel.  That is the only safe
    # behaviour: the sentinel is -1 precisely so a caller who marks it variable
    # without supplying it fails loudly rather than mis-splitting the flow.
    assert x4.of_is_variable is False
    assert float(x4.OF) < 0.0


def test_scaled_space_roundtrip_and_membership():
    space = default_design_space()
    S = ScaledSpace.from_specs(space)
    x = jnp.asarray([3.0e6, 8.0, 0.2, 0.25, 0.020, 3.0e4, 5.0e-4, 1.5e-3, 0.1,
                     8.0e-4])
    z = S.to_unit(x)
    assert np.all((np.asarray(z) >= 0) & (np.asarray(z) <= 1))
    x2 = S.to_physical(z)
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), rtol=1e-14)
    assert bool(S.contains(x))
    bad = jnp.asarray([9.9e6, 8.0, 0.2, 0.25, 0.020, 3.0e4, 5.0e-4, 1.5e-3, 0.1,
                       8.0e-4])
    assert not bool(S.contains(bad))


def test_mission_cstar_matches_gas_dynamics_oracle():
    m = MissionSpec()
    oracle = characteristic_velocity(m.gamma, m.R_gas, m.Tc)
    assert m.c_star_ideal() == pytest.approx(oracle, rel=1e-14)


def test_mission_uses_distinct_traditional_top_throat_fillet_factors():
    m = MissionSpec()
    assert m.throat_ru_factor == pytest.approx(1.5)
    assert m.throat_rd_factor == pytest.approx(0.382)


def test_two_branch_fuel_architecture_rejects_an_untracked_third_bypass():
    with pytest.raises(ValueError, match="cooling_fraction must therefore equal 1.0"):
        MissionSpec(cooling_fraction=0.9)


def test_phase1_gate_jit_and_jacobians_no_callbacks():
    """A jitted scalar of the design pytree; jacfwd and jacrev both run and
    agree — the plan's Phase-1 completion gate on the schema layer."""
    space = default_design_space()
    S = ScaledSpace.from_specs(space)

    @jax.jit
    def evaluate(z):
        x = DesignVector.from_array(S.to_physical(z))
        # algebra spanning all ten fields, smooth in the box
        return (x.Pc / 1e6) ** 0.8 * jnp.sqrt(x.eps) \
            + 10.0 * x.dp_f_frac * x.dp_o_frac \
            + 1.0e3 * x.D_pintle + x.N_rpm / 1.0e5 \
            + 1.0e3 * x.channel_width + 1.0e2 * x.channel_height \
            + 2.0 * x.film_frac + 1.0e2 * x.t_wall

    z0 = jnp.full((10,), 0.5)
    v = evaluate(z0)
    assert np.isfinite(float(v))
    g_fwd = jax.jacfwd(evaluate)(z0)
    g_rev = jax.jacrev(evaluate)(z0)
    np.testing.assert_allclose(np.asarray(g_fwd), np.asarray(g_rev),
                               rtol=1e-12)
    assert np.all(np.isfinite(np.asarray(g_fwd)))
