"""
tests/test_mdo_cea_surfaces.py — Phase-2 property-surface wiring.

The C¹ (P_c, O/F) surface machinery itself is gated in ``test_mdo_properties.py``.
This file gates the *engine wiring*: when a saved CEA table is supplied via
``MissionSpec.cea_table_path``, the engine consumes it and **O/F becomes a real
physical lever** (γ, T_c and hence c* vary with mixture ratio, so the flame
temperature moves both performance and the coking margin — the second coking
lever identified in the 2026-07-24 literature audit).  Without a table the
constant fallback is used and its provenance says so.

A *synthetic but physically shaped* table is used here (T_c peaking near
O/F ≈ 2.6 for LOX/RP-1); generating a real table needs RocketCEA on the host
(`scripts/sample_cea_surface.py`), which is the documented Phase-2 gate.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.properties import save_tables, load_chamber_surfaces
from raosim.mdo.schema import MissionSpec, DesignVector
from raosim.mdo.engine import solve_engine, chamber_surfaces_for


@pytest.fixture(scope="module")
def table_path():
    Pc = np.linspace(1.5e6, 6.0e6, 5)
    OF = np.linspace(1.6, 3.2, 7)
    _, O = np.meshgrid(Pc, OF, indexing="ij")
    Tc = 3700.0 - 900.0 * (O - 2.6) ** 2      # K, peak near O/F 2.6
    gamma = 1.24 - 0.02 * (O - 2.6)
    R_gas = np.full_like(Tc, 346.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "synthetic_lox_rp1.npz")
    save_tables(p, {"Pc_grid": Pc, "OF_grid": OF, "gamma": gamma, "Tc": Tc,
                    "R_gas": R_gas}, oxidizer="LOX", fuel="RP-1")
    return p


def _x(**over):
    kw = dict(Pc=jnp.asarray(3.0e6), eps=jnp.asarray(8.0),
              dp_f_frac=jnp.asarray(0.2), dp_o_frac=jnp.asarray(0.2),
              film_frac=jnp.asarray(0.05), channel_height=jnp.asarray(3.0e-3))
    kw.update(over)
    return DesignVector(**kw)


def test_fallback_used_without_table():
    m = MissionSpec()
    assert m.cea_table_path == ""
    assert "constant_fallback" in chamber_surfaces_for(m).provenance


def test_table_is_loaded_and_shapes_properties(table_path):
    s = load_chamber_surfaces(table_path)
    assert "cea_frozen_table" in s.provenance
    # T_c peaks near O/F 2.6 and falls either side (the physical shape)
    t = [float(s.Tc(jnp.asarray(3.0e6), jnp.asarray(o))) for o in (1.8, 2.6, 3.2)]
    assert t[1] > t[0] and t[1] > t[2]
    m = MissionSpec(cea_table_path=table_path)
    assert "cea_frozen_table" in chamber_surfaces_for(m).provenance


def test_of_becomes_a_real_lever(table_path):
    """With a table, mixture ratio moves BOTH Isp and the coking margin —
    running fuel-rich cools the wall at a performance cost."""
    m = MissionSpec(cea_table_path=table_path)
    out = {}
    for of in (1.8, 2.6):
        r = solve_engine(_x(), dataclasses.replace(m, OF=of))
        out[of] = (float(r.Isp), float(r.constraints["coking_margin_min"]))
    # richer (lower O/F) → cooler flame → better coking margin, lower Isp
    assert out[1.8][1] > out[2.6][1]      # coking margin improves fuel-rich
    assert out[1.8][0] < out[2.6][0]      # ...at an Isp cost


def test_engine_still_runs_on_fallback():
    """No table → constant surfaces → the engine still solves (CI path)."""
    r = solve_engine(_x(), MissionSpec())
    assert float(r.Isp) > 0.0
