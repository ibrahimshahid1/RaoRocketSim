"""
tests/test_mdo_cooling.py — Phase-4a acceptance gate for ``raosim.mdo.cooling``.

Two comparisons per the §12.1 rule-4 discipline:

1. **NumPy parity oracle.** A plain-Python re-implementation of the same clean
   upwind finite-volume coolant march, built on the *audited* NumPy primitives
   (``physics.bartz_heat_transfer_coefficient``, ``physics.fin_efficiency``, the
   Sieder & Tate 1936 turbulent correlation at unit property ratio), must match
   the JAX ``cooling_march`` to ~1e-9.  Because the oracle uses ``physics.*`` and
   the block uses the ``jax.thermal.*`` mirrors, this also cross-validates the
   mirror claim.
2. **AD-vs-FD gate** (in test_mdo_cooling_gate.py) — kept separate so this file
   stays a pure parity check.

Also pins the two defects fixed while completing the block: the upwind march
crosses each of the n-1 wall segments exactly once (no double-counted segment),
and the channel-fit land width is *exposed* (``land_min``), never clipped away.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim import physics
from raosim.mdo.schema import MissionSpec
from raosim.mdo.grid import build_station_grid
from raosim.mdo.cooling import cooling_march, solve_cooling


# --------------------------------------------------------------------------- #
# Independent NumPy oracle (audited physics.* + explicit upwind march)         #
# --------------------------------------------------------------------------- #
def _numpy_cooling_march(T_wg, grid, mission, *, Pc, gamma, Tc, c_star_del,
                         mdot_cool):
    r = np.asarray(grid.r, dtype=float)
    ar = np.asarray(grid.area_ratio, dtype=float)
    mach = np.asarray(grid.mach, dtype=float)
    dseg = np.asarray(grid.dseg, dtype=float)
    T_wg = np.asarray(T_wg, dtype=float)
    n = r.shape[0]
    Rt = float(r[grid.throat_index])
    Pr = mission.Pr_gas

    # recovery temperature, turbulent r = Pr^(1/3)
    rec = Pr ** (1.0 / 3.0)
    fM = 1.0 + 0.5 * (gamma - 1.0) * mach ** 2
    T_aw = Tc * (1.0 + rec * 0.5 * (gamma - 1.0) * mach ** 2) / fM

    # gas-side h_g via the audited Bartz oracle, station by station
    h_g = np.array([
        physics.bartz_heat_transfer_coefficient(
            float(mach[i]), float(1.0 / ar[i]), Dt=2.0 * Rt, Pc=Pc,
            c_star=c_star_del, cp=mission.cp_gas, Pr=Pr, mu=mission.mu_gas,
            gamma=gamma, Tc=Tc, wall_temperature=float(T_wg[i]),
            throat_curvature_radius=mission.throat_rd_factor * Rt)
        for i in range(n)
    ])

    # channel geometry (fixed N, w, h)
    N = mission.n_channels
    w = mission.channel_width
    h = mission.channel_height
    r_cool = r + mission.t_wall
    pitch = 2.0 * np.pi * r_cool / N
    land_raw = pitch - w
    land_min = float(land_raw.min())
    land = np.maximum(land_raw, 1e-5)
    A_chan = N * w * h
    Dh = 2.0 * w * h / (w + h)
    G = mdot_cool / A_chan
    u = G / mission.rho_cool
    Re = G * Dh / mission.mu_cool

    # coolant-side Sieder-Tate (0.027 Re^0.8 Pr^1/3, unit property ratio)
    Pr_c = mission.mu_cool * mission.cp_cool / mission.k_cool
    Nu = 0.027 * Re ** 0.8 * Pr_c ** (1.0 / 3.0)
    h_c = Nu * mission.k_cool / Dh
    eta_f = physics.fin_efficiency(h_c, mission.k_wall, land, h)
    area_enh = (w + 2.0 * eta_f * h) / pitch
    R_tot = 1.0 / h_g + mission.t_wall / mission.k_wall + 1.0 / (h_c * area_enh)

    # upwind counterflow march (each of the n-1 segments crossed once)
    dA = 2.0 * np.pi * r[:-1] * dseg
    T_c = np.empty(n)
    T_c[n - 1] = mission.coolant_temperature
    cap = mdot_cool * mission.cp_cool
    for j in range(n - 2, -1, -1):
        q_up = (T_aw[j + 1] - T_c[j + 1]) / R_tot[j + 1]
        T_c[j] = T_c[j + 1] + q_up * dA[j] / cap
    q = (T_aw - T_c) / R_tot
    residual = T_wg - (T_aw - q / h_g)
    T_wc = T_wg - q * (mission.t_wall / mission.k_wall)
    coking_margin = mission.rp1_coking_wall_temp_K - T_wc

    # Darcy Δp (smooth blend replicated exactly; uniform f, u)
    Re_c = max(Re, 1.0)
    f_lam = 64.0 / Re_c
    f_turb = 0.3164 / Re_c ** 0.25
    wgt = min(max((Re - 2100.0) / 400.0, 0.0), 1.0)
    s = wgt * wgt * (3.0 - 2.0 * wgt)
    f_darcy = (1.0 - s) * f_lam + s * f_turb
    dp = f_darcy * float(dseg.sum()) / Dh * 0.5 * mission.rho_cool * u ** 2

    return dict(T_coolant=T_c, q_flux=q, h_g=h_g, h_c=h_c, T_aw=T_aw,
                T_wc=T_wc, area_enh=area_enh, coking_margin=coking_margin,
                dp_total=dp, T_coolant_exit=T_c[0], land_min=land_min,
                residual=residual)


# --------------------------------------------------------------------------- #
def _grid_and_case():
    m = MissionSpec()
    g = build_station_grid(jnp.asarray(0.025), jnp.asarray(8.0), m)
    case = dict(Pc=3.0e6, gamma=1.24, Tc=3550.0, c_star_del=1750.0,
                mdot_cool=1.02)
    return m, g, case


def test_cooling_march_parity_numpy():
    """JAX cooling_march ≈ audited NumPy oracle to ~1e-9 at a fixed T_wg."""
    m, g, case = _grid_and_case()
    n = g.r.shape[0]
    # a non-trivial, non-constant wall-temperature probe
    T_wg = 700.0 + 300.0 * np.linspace(0.0, 1.0, n) ** 2

    jax_out = cooling_march(
        jnp.asarray(T_wg), g, Pc=jnp.asarray(case["Pc"]),
        gamma=jnp.asarray(case["gamma"]), Tc=jnp.asarray(case["Tc"]),
        c_star_del=jnp.asarray(case["c_star_del"]),
        mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    ref = _numpy_cooling_march(T_wg, g, m, **case)

    for key in ("T_coolant", "q_flux", "h_g", "h_c", "T_aw", "T_wc",
                "area_enh", "coking_margin", "residual"):
        got = np.asarray(getattr(jax_out, key))
        np.testing.assert_allclose(
            got, ref[key], rtol=1e-9, atol=1e-9,
            err_msg=f"parity mismatch on {key}")
    assert float(jax_out.dp_total) == pytest.approx(ref["dp_total"], rel=1e-9)
    assert float(jax_out.T_coolant_exit) == pytest.approx(
        ref["T_coolant_exit"], rel=1e-9)
    assert float(jax_out.land_min) == pytest.approx(ref["land_min"], rel=1e-9)


def test_cooling_solve_converges():
    """solve_cooling drives the stationwise residual to ~machine zero."""
    m, g, case = _grid_and_case()
    T_wg, march = solve_cooling(
        g, Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
        Tc=jnp.asarray(case["Tc"]), c_star_del=jnp.asarray(case["c_star_del"]),
        mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    assert float(jnp.max(jnp.abs(march.residual))) < 1e-8
    # physical ordering: coolant floor < wall < recovery, everywhere
    assert float(jnp.min(T_wg)) > float(m.coolant_temperature)
    assert bool(jnp.all(T_wg < march.T_aw))
    # coolant heats monotonically from the nozzle-exit inlet to the injector
    assert float(march.T_coolant_exit) > float(m.coolant_temperature)


def test_cooling_coolant_heats_along_flow():
    """Counterflow: coolant strictly warms from station n-1 (inlet) to 0."""
    m, g, case = _grid_and_case()
    _, march = solve_cooling(
        g, Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
        Tc=jnp.asarray(case["Tc"]), c_star_del=jnp.asarray(case["c_star_del"]),
        mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    Tc_arr = np.asarray(march.T_coolant)
    # flow order is n-1 -> 0, so reversed array must be monotincreasing
    dT = np.diff(Tc_arr[::-1])
    assert np.all(dT >= -1e-9)


def test_land_fit_constraint_is_exposed_not_clipped():
    """Over-packing the wall drives land_min negative — it must be reported,
    not silently clipped to the numerical floor (plan §9)."""
    m = MissionSpec(n_channels=4000)      # absurd count -> negative land
    g = build_station_grid(jnp.asarray(0.025), jnp.asarray(8.0), m)
    march = cooling_march(
        jnp.full((g.r.shape[0],), 800.0), g, Pc=jnp.asarray(3.0e6),
        gamma=jnp.asarray(1.24), Tc=jnp.asarray(3550.0),
        c_star_del=jnp.asarray(1750.0), mdot_cool=jnp.asarray(1.0), mission=m)
    assert float(march.land_min) < 0.0    # infeasible packing is visible


def test_coking_constraint_exposed_and_active():
    """SP-8087 RP-1 liquid-wall coking limit (728 K) is exposed as a stationwise
    margin; the fixed *screening* channel geometry violates it near the throat
    (T_wc ≫ limit), so the optimiser sees an active constraint instead of a
    hidden infeasibility — the ~733 K coolant finding made explicit."""
    m, g, case = _grid_and_case()
    T_wg, march = solve_cooling(
        g, Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
        Tc=jnp.asarray(case["Tc"]), c_star_del=jnp.asarray(case["c_star_del"]),
        mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    T_wc = np.asarray(march.T_wc)
    Twg = np.asarray(T_wg)
    Tcool = np.asarray(march.T_coolant)
    # series-circuit ordering everywhere: T_coolant ≤ T_wc ≤ T_wg
    assert np.all(T_wc <= Twg + 1e-6)
    assert np.all(T_wc >= Tcool - 1e-6)
    # constraint is active: the peak-flux throat exceeds the 728 K coking limit
    assert float(march.coking_margin.min()) < 0.0
    assert float(T_wc[g.throat_index]) > m.rp1_coking_wall_temp_K
    # margin is exactly limit − T_wc (sign convention: ≥0 feasible)
    np.testing.assert_allclose(
        np.asarray(march.coking_margin),
        m.rp1_coking_wall_temp_K - T_wc, rtol=1e-12, atol=1e-9)


def test_gaseous_film_correlation_is_inapplicable_to_a_liquid_film():
    """The classical ε = C(X/VR)^(−0.8)Re^(0.2) family (Stollery/Hartnett/
    Tribus; Hatch & Papell TN D-130) is fitted to **gaseous** coolants over
    velocity ratios 0.45–33.3.  For a liquid RP-1 film, continuity through the
    tangential slot gives VR ≈ 1e−3 for *any* slot height in the tested range —
    three orders below that band — which is why the block uses the liquid
    phase-change model instead.  This test pins that finding."""
    from raosim.mdo.cooling import film_slot_validity
    m, g, _ = _grid_and_case()
    for sh in (5.0e-4, 2.0e-3, 1.27e-2):     # the Hatch–Papell slot range
        d = film_slot_validity(jnp.asarray(0.05), jnp.asarray(sh), g, m,
                               mdot_film=jnp.asarray(0.053),
                               gamma=jnp.asarray(1.24), Tc=jnp.asarray(3550.0))
        assert float(d["velocity_ratio"]) < 0.45        # outside the fitted band
        assert not bool(d["gaseous_correlation_applicable"])
        assert float(d["v_film"]) < float(d["v_core"])  # liquid film is slow


def test_design_margins_default_nominal_and_are_conservative():
    """§10.3 — SP-8087/Mirzamoghadam design margins.  Defaults are 1.0 so
    nominal results are bit-identical; switching them on (+10 % heat flux for
    streaking, −10 % channel flow for maldistribution) must make the wall
    strictly hotter, i.e. the answer strictly more conservative."""
    import dataclasses
    m, g, case = _grid_and_case()
    assert (m.heat_flux_margin, m.channel_flow_margin) == (1.0, 1.0)
    kw = dict(Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
              Tc=jnp.asarray(case["Tc"]),
              c_star_del=jnp.asarray(case["c_star_del"]),
              mdot_cool=jnp.asarray(case["mdot_cool"]))
    _, nom = solve_cooling(g, mission=m, **kw)
    m_marg = dataclasses.replace(m, heat_flux_margin=1.10,
                                 channel_flow_margin=0.90)
    _, marg = solve_cooling(g, mission=m_marg, **kw)
    assert float(jnp.max(marg.T_wc)) > float(jnp.max(nom.T_wc))
    assert float(jnp.min(marg.coking_margin)) < float(jnp.min(nom.coking_margin))


def test_thermal_stress_dominates_pressure_and_t_wall_matters():
    """§10.2 — the binding wall criterion is the THERMAL gradient, not pressure
    (the audit's original hoop-stress recommendation was wrong): plate bending
    across the channel is orders below the constrained-expansion stress.  Also
    pins that t_wall actually moves the solution."""
    m, g, case = _grid_and_case()
    kw = dict(Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
              Tc=jnp.asarray(case["Tc"]),
              c_star_del=jnp.asarray(case["c_star_del"]),
              mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    _, thin = solve_cooling(g, t_wall=jnp.asarray(5.0e-4), **kw)
    _, thick = solve_cooling(g, t_wall=jnp.asarray(1.5e-3), **kw)
    # thermal stress ≫ pressure bending
    assert float(jnp.max(thin.sigma_thermal)) > 50.0 * float(thin.sigma_pressure)
    # a thicker wall raises the through-wall ΔT (more thermal stress) and lowers
    # the coolant-side wall temperature (helps coking) — the real design trade
    assert float(jnp.max(thick.sigma_thermal)) > float(jnp.max(thin.sigma_thermal))
    assert float(jnp.max(thick.T_wc)) < float(jnp.max(thin.T_wc))


def test_coolant_mach_diagnostic_is_reported_and_far_below_limit():
    """§10.4 — coolant Mach is a *diagnostic*, not a constraint: the margin is
    ~2 orders, so constraining it would only add a near-dead Jacobian column."""
    m, g, case = _grid_and_case()
    _, mar = solve_cooling(g, Pc=jnp.asarray(case["Pc"]),
                           gamma=jnp.asarray(case["gamma"]),
                           Tc=jnp.asarray(case["Tc"]),
                           c_star_del=jnp.asarray(case["c_star_del"]),
                           mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    assert 0.0 < float(mar.coolant_mach) < 0.35      # Mirzamoghadam limit
    assert float(mar.coolant_mach) < 0.05            # comfortably non-binding
    assert float(mar.coolant_velocity) > 0.0


def test_film_cooling_efficiency_follows_knuth_transition():
    """η_fc is *derived* from the film state, not assumed: Stechman (1969) makes
    the liquid-coolant efficiency factor a function of coolant Reynolds number,
    and Grisson (AEDC-TR-91-1 §2.1) supplies the mechanism — below Knuth's
    critical flow-per-circumference Γ_cr = 1.01e5 μ_v²/μ_ℓ the film is smooth,
    above it waves shear droplets off and mass loss is 2–4× evaporation (η
    falls to ≈1/2–1/4)."""
    from raosim.mdo.cooling import film_cooling_efficiency
    m, g, _ = _grid_and_case()
    gamma_cr = m.film_knuth_coeff * m.film_mu_vapor ** 2 / m.mu_cool
    D = 2.0 * float(g.r[0])
    # a flow well below and well above the critical value
    mdot_lo = 0.1 * gamma_cr * np.pi * D
    mdot_hi = 10.0 * gamma_cr * np.pi * D
    eta_lo = float(film_cooling_efficiency(jnp.asarray(mdot_lo), g, m))
    eta_hi = float(film_cooling_efficiency(jnp.asarray(mdot_hi), g, m))
    assert eta_lo > eta_hi                       # waves reduce efficiency
    assert eta_lo == pytest.approx(m.film_eta_smooth, abs=0.05)
    assert eta_hi == pytest.approx(m.film_eta_wavy, abs=0.05)
    # the wavy limit is Grisson's 2-4x mass loss, i.e. eta ~ 1/2 .. 1/4
    assert 0.25 <= m.film_eta_wavy <= 0.5
    # monotone and smooth across the transition (C1 for the optimiser)
    flows = np.linspace(0.2 * mdot_lo, 5.0 * mdot_hi, 60)
    etas = np.array([float(film_cooling_efficiency(jnp.asarray(f), g, m))
                     for f in flows])
    assert np.all(np.diff(etas) <= 1e-12)        # monotone decreasing


def test_liquid_film_effectiveness_follows_phase_change_balance():
    """ε comes from the liquid phase-change energy balance (Shine & Nidhi §4.3;
    Huzel & Huang SP-125 Eq. 4-34): more film flow buys a longer film-cooled
    length, so effectiveness rises with film fraction and decays downstream."""
    from raosim.mdo.cooling import film_effectiveness
    m, g, case = _grid_and_case()
    kw = dict(slot_height=jnp.asarray(2.0e-3), gamma=jnp.asarray(case["gamma"]),
              Tc=jnp.asarray(case["Tc"]),
              h_g=jnp.full((g.r.shape[0],), 1.0e4),
              T_aw=jnp.full((g.r.shape[0],), 3400.0))
    e_lo = np.asarray(film_effectiveness(jnp.asarray(0.03), g, m,
                                         mdot_film=jnp.asarray(0.03), **kw))
    e_hi = np.asarray(film_effectiveness(jnp.asarray(0.10), g, m,
                                         mdot_film=jnp.asarray(0.10), **kw))
    assert e_hi.mean() > e_lo.mean()            # more film -> more protection
    assert e_hi[0] >= e_hi[-1]                  # decays downstream
    assert np.all((e_hi >= 0.0) & (e_hi <= 1.0))
    # zero film is an exact no-op
    zero = film_effectiveness(jnp.asarray(0.0), g, m,
                              mdot_film=jnp.asarray(0.0), **kw)
    assert float(jnp.max(zero)) == 0.0


def test_film_wall_temperature_is_monotone_and_smooth():
    """T_wc falls monotonically with film fraction and the axial profile has no
    artificial cliff (the correlation's X^(−0.8) decay replaced an on/off
    coverage mask that produced a non-physical jump)."""
    m, g, case = _grid_and_case()
    kw = dict(Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
              Tc=jnp.asarray(case["Tc"]),
              c_star_del=jnp.asarray(case["c_star_del"]),
              mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    peaks = []
    for ff in (0.0, 0.02, 0.05, 0.10):
        _, mar = solve_cooling(g, film_frac=jnp.asarray(ff), **kw)
        peaks.append(float(jnp.max(mar.T_wc)))
    assert all(b < a for a, b in zip(peaks, peaks[1:])), peaks   # monotone down
    # no cliff: neighbouring-station jumps stay bounded
    _, mar = solve_cooling(g, film_frac=jnp.asarray(0.05), **kw)
    jumps = np.abs(np.diff(np.asarray(mar.T_wc)))
    assert float(jumps.max()) < 150.0


def test_film_cooling_reduces_wall_temperature():
    """film_frac > 0 lowers the coolant-side wall temperature (SP-8087/Papell
    screening film model over the protected chamber→throat region); the
    default film_frac = 0 is an exact no-op."""
    m, g, case = _grid_and_case()
    kw = dict(Pc=jnp.asarray(case["Pc"]), gamma=jnp.asarray(case["gamma"]),
              Tc=jnp.asarray(case["Tc"]),
              c_star_del=jnp.asarray(case["c_star_del"]),
              mdot_cool=jnp.asarray(case["mdot_cool"]), mission=m)
    _, dry = solve_cooling(g, film_frac=jnp.asarray(0.0), **kw)
    _, wet = solve_cooling(g, film_frac=jnp.asarray(0.2), **kw)
    assert float(jnp.max(wet.T_wc)) < float(jnp.max(dry.T_wc)) - 50.0
    _, base = solve_cooling(g, **kw)          # default film_frac=0 == explicit 0
    assert float(jnp.max(base.T_wc)) == pytest.approx(
        float(jnp.max(dry.T_wc)), rel=1e-9)
