"""Walking-skeleton gates (plan §12.5): the coupled residual converges, the
solved state matches the NumPy closed-form oracle (the system is currently
triangular by design), and end-to-end AD through the CONVERGED state matches
re-solved central differences — the first mass–Isp gradient of the MDO layer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optimistix")
import jax.numpy as jnp  # noqa: E402

from raosim import gas_dynamics as gd  # noqa: E402
from raosim import physics as ph  # noqa: E402
from raosim.mdo.assembly import (  # noqa: E402
    StateScales,
    make_engine_fn,
    readouts,
    solve_states,
)
from raosim.mdo.properties import constant_chamber_surfaces  # noqa: E402
from raosim.mdo.schema import DesignVector, MissionSpec  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    mission = MissionSpec()
    surfaces = constant_chamber_surfaces(
        gamma=mission.gamma, Tc=mission.Tc, R_gas=mission.R_gas)
    scales = StateScales.from_mission(mission)
    x = DesignVector(Pc=jnp.asarray(3.0e6), eps=jnp.asarray(6.0),
                     dp_f_frac=jnp.asarray(0.2), dp_o_frac=jnp.asarray(0.2))
    return mission, surfaces, scales, x


# --------------------------------------------------------------------------- #
# NumPy oracle (closed forms; the skeleton system is triangular today)         #
# --------------------------------------------------------------------------- #
def _oracle(mission: MissionSpec, x) -> dict:
    g = mission.gamma
    Pc = float(x.Pc)
    eps = float(x.eps)
    Me = gd.mach_from_area_ratio(eps, g, supersonic=True)
    Pe_ratio = gd.isentropic_pressure_ratio(Me, g)
    Cf = gd.thrust_coefficient(Me, g, Pe_ratio, mission.Pa / Pc, eps)
    Rt = math.sqrt(mission.thrust / (Cf * Pc * math.pi))
    c_star_del = mission.eta_cstar * gd.characteristic_velocity(
        g, mission.R_gas, mission.Tc)
    mdot = Pc * math.pi * Rt * Rt / c_star_del

    # Converged throat thermal fixed point on the NumPy originals.
    r = mission.Pr_gas ** (1.0 / 3.0)
    f1 = 1.0 + 0.5 * (g - 1.0)
    Taw = mission.Tc * (1.0 + r * 0.5 * (g - 1.0)) / f1
    T_wg = 0.6 * Taw
    for _ in range(200):
        h_g = ph.bartz_heat_transfer_coefficient(
            1.0, 1.0, Dt=2.0 * Rt, Pc=Pc, c_star=c_star_del,
            cp=mission.cp_gas, Pr=mission.Pr_gas, mu=mission.mu_gas,
            gamma=g, Tc=mission.Tc, wall_temperature=T_wg,
            throat_curvature_radius=mission.throat_rd_factor * Rt)
        R_tot = 1.0 / h_g + mission.t_wall / mission.k_wall + 1.0 / mission.h_c
        q = (Taw - mission.coolant_temperature) / R_tot
        T_new = Taw - q / h_g
        if abs(T_new - T_wg) < 1e-12 * mission.Tc:
            T_wg = T_new
            break
        T_wg = T_new
    return {"Cf": Cf, "Rt": Rt, "mdot": mdot, "T_wg": T_wg, "q": q}


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def test_states_converge_tight(setup):
    mission, surfaces, scales, x = setup
    y, res = solve_states(x, mission, surfaces, scales)
    assert float(jnp.max(jnp.abs(res))) < 1e-10


def test_solved_states_match_numpy_oracle(setup):
    mission, surfaces, scales, x = setup
    y, _ = solve_states(x, mission, surfaces, scales)
    out = readouts(y, x, mission, surfaces, scales)
    ora = _oracle(mission, x)
    assert float(out["Rt"]) == pytest.approx(ora["Rt"], rel=1e-9)
    assert float(out["mdot"]) == pytest.approx(ora["mdot"], rel=1e-9)
    assert float(out["T_wg"]) == pytest.approx(ora["T_wg"], rel=1e-8)
    assert float(out["Cf"]) == pytest.approx(ora["Cf"], rel=1e-10)


def test_readout_ledger_and_stability_fields(setup):
    mission, surfaces, scales, x = setup
    y, _ = solve_states(x, mission, surfaces, scales)
    out = readouts(y, x, mission, surfaces, scales)
    led = out["mass_ledger"]
    # §3 ledger: excluded items present as explicit zeros.
    assert float(led["tanks_excluded"]) == 0.0
    assert float(led["pressurant_excluded"]) == 0.0
    # battery branches exposed separately (no max() in the core).
    assert float(out["m_battery_energy_limited"]) > 0.0
    assert float(out["m_battery_power_limited"]) > 0.0
    # chug screens reported per stream.
    assert float(out["chi_fuel"]) == pytest.approx(0.2)
    assert float(out["chi_ox"]) == pytest.approx(0.2)
    # separation margin uses the corrected criterion and is finite/positive.
    assert float(out["separation_margin"]) > 0.0


def test_end_to_end_gradient_matches_resolved_fd(setup):
    """AD of outputs THROUGH the converged state (IFT) vs central differences
    of the fully re-solved engine — the §12.1 rule-4 acceptance comparison
    (plan tolerance ~1e-4; we gate tighter)."""
    mission, *_ = setup
    f = make_engine_fn(mission, outputs=("Isp_delivered",
                                         "package_mass_report",
                                         "m_battery_energy_limited",
                                         "T_wg"))
    x0 = jnp.asarray([3.0e6, 6.0, 0.2, 0.2])
    J = jax.jacfwd(f)(x0)
    J = np.asarray(J)
    steps = np.array([1.0e3, 1.0e-4, 1.0e-5, 1.0e-5])
    for k, h in enumerate(steps):
        e = np.zeros(4)
        e[k] = h
        fp = np.asarray(f(x0 + e))
        fm = np.asarray(f(x0 - e))
        fd = (fp - fm) / (2 * h)
        scale = np.maximum(np.abs(fd), 1e-12)
        rel = np.max(np.abs(J[:, k] - fd) / scale)
        assert rel < 1e-5, f"column {k}: AD vs FD rel err {rel:.2e}"


def test_engine_fn_is_jittable_and_jacrev_agrees(setup):
    mission, *_ = setup
    f = make_engine_fn(mission, outputs=("Isp_delivered",
                                         "package_mass_report"))
    x0 = jnp.asarray([3.0e6, 6.0, 0.2, 0.2])
    fj = jax.jit(f)
    np.testing.assert_allclose(np.asarray(fj(x0)), np.asarray(f(x0)),
                               rtol=1e-12)
    Jf = np.asarray(jax.jacfwd(f)(x0))
    Jr = np.asarray(jax.jacrev(f)(x0))
    np.testing.assert_allclose(Jf, Jr, rtol=1e-8, atol=1e-14)


def test_cstar_convention_direction(setup):
    """Rule 2 sanity: higher eta_cstar (better combustion) -> lower mdot at
    fixed thrust, higher delivered Isp."""
    mission, surfaces, scales, x = setup
    import dataclasses

    hi = dataclasses.replace(mission, eta_cstar=0.97)
    lo = dataclasses.replace(mission, eta_cstar=0.90)
    out_hi = readouts(*solve_states(x, hi, surfaces, scales)[:1], x, hi,
                      surfaces, scales)
    out_lo = readouts(*solve_states(x, lo, surfaces, scales)[:1], x, lo,
                      surfaces, scales)
    assert float(out_hi["mdot"]) < float(out_lo["mdot"])
    assert float(out_hi["Isp_delivered"]) > float(out_lo["Isp_delivered"])
