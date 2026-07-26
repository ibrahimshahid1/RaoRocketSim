"""
raosim.mdo.engine — Phase 7: the coupled whole-engine solve.

Integrates the four discipline blocks (nozzle performance, regen cooling,
pintle injector, electric pump feed) into ONE differentiable evaluation, so a
single gradient of any objective/constraint flows from the design vector x
through every block and both implicit-function-theorem solves (plan §12.3, gate
"all residual blocks converge without manual sequential iteration").

Coupling honesty (plan §4.1: "do not enlarge the nonlinear system merely to
appear coupled").  At the current physics fidelity the data flow is:

    outer state (Rt, mdot)  ──►  grid ──►  cooling  ─┐  (inner IFT: T_wg vector)
                                          injector   ├─►  Δp_regen, Δp_inj
                                                     ▼
             pump feed  ◄───  Δp_rise = Pc(1+χ) + Δp_regen + Δp_line − P_tank
                                                     ▼
                          P_electric, battery, drive masses, §3 ledger

So the §5 **hydraulic edge is now genuinely closed** — the cooling jacket Δp and
the injector Δp feed the pump rise (previously ``regen_dp_allowance = 0``
placeholders).  The wall-temperature vector is a real inner implicit solve.
Everything else is the explicit chain the physics actually is: the heat load and
feed pressures do not change the throat area or chamber mass flow, so (Rt, mdot)
stay a small outer root — kept implicit as the seam where the ONE genuine
two-way feedback lives.

The **combustion/mass-flow loop** (spray → η_c* → mdot → spray, plan §5) is the
only genuine 2-way coupling.  It is *off by default* (``couple_eta_cstar=False``
⇒ frozen ``mission.eta_cstar``): the plan flags it as the strongest edge but the
weakest physics and asks for the ablation.  With it on, R2's η_c* depends on the
injector TMR(mdot), so the outer Newton solves a real fixed point and the
``ablation_delta`` (coupled − frozen) is directly measurable (RQ1).

Nothing here touches CAD or the reporting workflow (plan rule 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp
import optimistix as optx

from raosim.jax import thermal as jt
from raosim.jax.primitives import isentropic_pressure_ratio, mach_from_area_ratio
from raosim.mdo.assembly import StateScales
from raosim.mdo.properties import ChamberSurfaces, constant_chamber_surfaces
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.grid import build_station_grid, GridTopology
from raosim.mdo.cooling import (
    solve_cooling, CoolingMarch, film_cooling_efficiency,
)
from raosim.mdo.injector import injector_readouts, InjectorReadout
from raosim.mdo.pump import electric_feed, ElectricFeed

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Optional spray → c* feedback surrogate (default OFF — see module doc)         #
# --------------------------------------------------------------------------- #
def eta_cstar_coupled(TMR: Array, mission: MissionSpec) -> Array:
    """Screening c*-efficiency vs total momentum ratio, peaked at TMR_opt
    (smooth log-Gaussian).  A *screening knob*, not validated physics — the
    corpus (Sakaki 2016/2017, Hwang s42405) shows η_c* varies with TMR with a
    config-dependent optimum near TMR≈1; used only when the coupling is on."""
    z = jnp.log(jnp.maximum(TMR, 1e-9) / mission.eta_cstar_tmr_opt) \
        / mission.eta_cstar_tmr_width
    return mission.eta_cstar_max * jnp.exp(-0.5 * z * z)


def _split(mdot: Array, mission: MissionSpec) -> tuple[Array, Array]:
    mdot_f = mdot / (1.0 + mission.OF)
    mdot_o = mdot * mission.OF / (1.0 + mission.OF)
    return mdot_f, mdot_o


def _eta_cstar(mdot: Array, x: DesignVector, mission: MissionSpec,
               surfaces: ChamberSurfaces, couple: bool) -> Array:
    if not couple:
        base = jnp.asarray(mission.eta_cstar, dtype=jnp.float64)
    else:
        mdot_f, mdot_o = _split(mdot, mission)
        inj = injector_readouts(Pc=x.Pc, chi_f=x.dp_f_frac, chi_o=x.dp_o_frac,
                                D_pintle=x.D_pintle, mdot_fuel=mdot_f,
                                mdot_ox=mdot_o, mission=mission)
        base = eta_cstar_coupled(inj.momentum_ratio, mission)
    # film-cooling c* penalty: the wall-film fuel burns fuel-rich and does not
    # fully contribute to core combustion (§6.2b; the film's performance cost).
    return base * (1.0 - mission.film_cstar_penalty * x.film_frac)


# --------------------------------------------------------------------------- #
# Outer coupled residual on (Rt, mdot)                                          #
# --------------------------------------------------------------------------- #
def engine_residual(y: Array, x: DesignVector, mission: MissionSpec,
                    surfaces: ChamberSurfaces, scales: StateScales,
                    couple: bool) -> Array:
    """R(y, x) on the 2-vector y = [Rt_hat, mdot_hat] (scaled).

    R1 thrust closure; R2 c*-mass-flow closure.  When ``couple`` the η_c* in R2
    depends on TMR(mdot) → a genuine fixed point; otherwise triangular."""
    Rt = y[0] * scales.Rt_ref
    mdot = y[1] * scales.mdot_ref
    Pc, eps = x.Pc, x.eps

    gamma = surfaces.gamma(Pc, mission.OF)
    cstar_ideal = surfaces.c_star_ideal(Pc, mission.OF)
    eta_cs = _eta_cstar(mdot, x, mission, surfaces, couple)
    cstar_del = eta_cs * cstar_ideal

    Cf = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    At = jnp.pi * Rt * Rt
    R1 = (mission.thrust - Cf * Pc * At) / mission.thrust
    R2 = (mdot - Pc * At / cstar_del) / scales.mdot_ref
    return jnp.stack([R1, R2])


def _initial_state(x: DesignVector, mission: MissionSpec,
                   surfaces: ChamberSurfaces, scales: StateScales) -> Array:
    gamma = surfaces.gamma(x.Pc, mission.OF)
    Cf = jt.ambient_thrust_coefficient(x.eps, gamma, x.Pc, mission.Pa)
    Rt = jnp.sqrt(mission.thrust / (Cf * x.Pc * jnp.pi))
    cstar_del = mission.eta_cstar * surfaces.c_star_ideal(x.Pc, mission.OF)
    mdot = x.Pc * jnp.pi * Rt * Rt / cstar_del
    return jnp.stack([Rt / scales.Rt_ref, mdot / scales.mdot_ref])


# --------------------------------------------------------------------------- #
# Full engine result                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineResult:
    # converged state + performance
    Rt: Array
    mdot: Array
    T_wg: Array           # (n,) solved gas-side wall temperatures [K]
    eta_cstar: Array
    Cf: Array
    Isp: Array
    Me: Array
    Pe: Array
    thrust_residual: Array
    # blocks
    cooling: CoolingMarch
    injector: InjectorReadout
    feed: ElectricFeed
    # routed hydraulic edge (closed §5 loop)
    dp_regen: Array
    dp_rise_fuel: Array
    dp_rise_ox: Array
    # mass ledger (reporting)
    package_mass: Array
    mass_ledger: dict
    # constraint margins (reporting scalars; ≥0 feasible)
    constraints: dict
    # reported diagnostics (not constraints)
    diagnostics: dict


def chamber_surfaces_for(mission: MissionSpec) -> ChamberSurfaces:
    """Resolve the property surfaces for a mission (Phase 2).

    If ``mission.cea_table_path`` points at a saved CEA sampling (produced
    host-side by ``scripts/sample_cea_surface.py``), the C¹ (P_c, O/F) surfaces
    are loaded from it, and O/F becomes a *physically meaningful* lever (γ, T_c
    and hence c* vary with mixture ratio).  Otherwise the constant fallback is
    used — correct for a fixed O/F, but flat in O/F, and its provenance says so.
    """
    if mission.cea_table_path:
        from raosim.mdo.properties import load_chamber_surfaces
        return load_chamber_surfaces(mission.cea_table_path)
    return constant_chamber_surfaces(gamma=mission.gamma, Tc=mission.Tc,
                                     R_gas=mission.R_gas)


def solve_engine(x: DesignVector, mission: MissionSpec, *,
                 couple_eta_cstar: bool = False,
                 surfaces: ChamberSurfaces | None = None,
                 topo: GridTopology = GridTopology(),
                 rtol: float = 1e-12, atol: float = 1e-12,
                 max_steps: int = 64) -> EngineResult:
    """Solve the coupled engine at design point ``x`` and assemble every block.

    Differentiable end-to-end: the outer (Rt, mdot) root and the inner cooling
    wall-temperature root are each closed by the IFT, so ``jax.grad`` of any
    output flows through both plus the explicit injector/pump chain.
    """
    surfaces = surfaces if surfaces is not None else chamber_surfaces_for(mission)
    scales = StateScales.from_mission(mission)

    def fn(y, args):
        return engine_residual(y, args, mission, surfaces, scales,
                               couple_eta_cstar)

    sol = optx.root_find(fn, optx.Newton(rtol=rtol, atol=atol),
                         _initial_state(x, mission, surfaces, scales),
                         args=x, max_steps=max_steps, throw=False)
    y = sol.value
    resid = fn(y, x)
    Rt = y[0] * scales.Rt_ref
    mdot = y[1] * scales.mdot_ref
    Pc, eps = x.Pc, x.eps

    gamma = surfaces.gamma(Pc, mission.OF)
    Tc = surfaces.Tc(Pc, mission.OF)
    eta_cs = _eta_cstar(mdot, x, mission, surfaces, couple_eta_cstar)
    cstar_del = eta_cs * surfaces.c_star_ideal(Pc, mission.OF)

    # --- performance ------------------------------------------------------- #
    Cf = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    Pe = Pc * isentropic_pressure_ratio(Me, gamma)
    Isp = Cf * cstar_del / mission.g0
    sep_margin = jt.schmucker_separation_margin(eps, gamma, Pc, mission.Pa)

    mdot_f, mdot_o = _split(mdot, mission)

    # --- cooling (inner IFT solve on the stationwise wall temperatures) ----- #
    grid = build_station_grid(Rt, eps, mission, topo)
    mdot_cool = mission.cooling_fraction * mdot_f
    T_wg, cooling = solve_cooling(grid, Pc=Pc, gamma=gamma, Tc=Tc,
                                  c_star_del=cstar_del, mdot_cool=mdot_cool,
                                  mission=mission, channel_width=x.channel_width,
                                  channel_height=x.channel_height,
                                  film_frac=x.film_frac, t_wall=x.t_wall)
    dp_regen = cooling.dp_total

    # --- injector ---------------------------------------------------------- #
    injector = injector_readouts(
        Pc=Pc, chi_f=x.dp_f_frac, chi_o=x.dp_o_frac, D_pintle=x.D_pintle,
        mdot_fuel=mdot_f, mdot_ox=mdot_o, mission=mission)

    # --- CLOSE the §5 hydraulic edge: cooling Δp + injector Δp → pump rise --- #
    dp_rise_f = (Pc * (1.0 + x.dp_f_frac) + dp_regen
                 + mission.line_dp_allowance - mission.P_tank_fuel)
    dp_rise_o = (Pc * (1.0 + x.dp_o_frac)
                 + mission.line_dp_allowance - mission.P_tank_ox)
    feed = electric_feed(mdot_fuel=mdot_f, mdot_ox=mdot_o,
                         dp_rise_fuel=dp_rise_f, dp_rise_ox=dp_rise_o,
                         N_rpm=x.N_rpm, mission=mission)

    # --- §3 mass ledger (battery max = reporting epigraph only) ------------- #
    m_batt = jnp.maximum(feed.battery.energy_limited_mass,
                         feed.battery.power_limited_mass) \
        * mission.battery_structural_margin
    ledger = {
        "pumps": feed.pump_mass, "motors": feed.motor_mass,
        "inverters": feed.inverter_mass, "battery": m_batt,
        "thrust_chamber_placeholder": jnp.asarray(0.0),
        "injector_placeholder": jnp.asarray(0.0),
    }
    package_mass = (feed.pump_mass + feed.motor_mass + feed.inverter_mass
                    + m_batt)

    # --- constraint margins (reporting scalars; ≥0 feasible) ---------------- #
    constraints = {
        "thrust_residual": resid[0],
        "separation_margin": sep_margin,
        "coking_margin_min": jnp.min(cooling.coking_margin),
        "land_min": cooling.land_min,
        "chug_margin_min": jnp.minimum(injector.chug_margin_fuel,
                                       injector.chug_margin_ox),
        "pintle_transition_margin": injector.transition_margin,
        "nss_margin_min": jnp.minimum(feed.fuel.nss_margin, feed.ox.nss_margin),
        "tip_speed_margin_min": jnp.minimum(feed.fuel.tip_speed_margin,
                                            feed.ox.tip_speed_margin),
        # HARCC aspect-ratio validity cap (Pizzarelli/Carlile/Mirzamoghadam)
        "aspect_ratio_margin": (mission.channel_aspect_ratio_max
                                - x.channel_height / x.channel_width),
        # blockage-factor band — BF with TMR are the two master pintle knobs
        # (Hwang 2022; Freeberg 2019).  Two-sided, so D_pintle is a *live*
        # design variable rather than one with no path to any constraint.
        "blockage_lo_margin": (injector.blockage_factor
                               - mission.blockage_factor_min),
        "blockage_hi_margin": (mission.blockage_factor_max
                               - injector.blockage_factor),
        # liner thermal stress (§10.2) — the binding wall criterion; the
        # pressure term is reported separately and is ~2 orders smaller.
        "thermal_stress_margin": (mission.liner_sigma_allow
                                  - jnp.max(cooling.sigma_thermal)),
        # allowable gas-side wall temperature (Mirzamoghadam: a primary design
        # criterion).  Required now that t_wall is a variable — a thicker wall
        # lowers T_wc but RAISES T_wg, so both sides must be bounded.
        "wall_temp_margin": mission.liner_T_wg_max - jnp.max(T_wg),
    }
    # reported diagnostics (NOT constraints — the coolant-Mach margin is ~173x,
    # so constraining it would only add a dead Jacobian column, §10.4)
    diagnostics = {
        "coolant_mach": cooling.coolant_mach,
        "coolant_mach_limit": jnp.asarray(0.35),
        "coolant_velocity": cooling.coolant_velocity,
        "sigma_thermal_max": jnp.max(cooling.sigma_thermal),
        "sigma_pressure": cooling.sigma_pressure,
        "film_capacity_required": (x.film_frac * mission.film_capacity_margin),
        "eta_film_cooling": film_cooling_efficiency(
            mission.cooling_fraction * mdot_f * x.film_frac
            / jnp.maximum(1.0 - x.film_frac, 1e-6), grid, mission),
    }

    return EngineResult(
        Rt=Rt, mdot=mdot, T_wg=T_wg, eta_cstar=eta_cs, Cf=Cf, Isp=Isp, Me=Me, Pe=Pe,
        thrust_residual=resid[0], cooling=cooling, injector=injector, feed=feed,
        dp_regen=dp_regen, dp_rise_fuel=dp_rise_f, dp_rise_ox=dp_rise_o,
        package_mass=package_mass, mass_ledger=ledger, constraints=constraints,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# Scalar accessor for AD / NLP (jit/jacfwd/jacrev-safe)                         #
# --------------------------------------------------------------------------- #
_SCALARS = {
    "Isp": lambda r: r.Isp,
    "package_mass": lambda r: r.package_mass,
    "P_electric": lambda r: r.feed.P_electric_total,
    "Rt": lambda r: r.Rt,
    "mdot": lambda r: r.mdot,
    "eta_cstar": lambda r: r.eta_cstar,
    "dp_regen": lambda r: r.dp_regen,
    "coking_margin_min": lambda r: r.constraints["coking_margin_min"],
    "separation_margin": lambda r: r.constraints["separation_margin"],
    "T_wc_max": lambda r: jnp.max(r.cooling.T_wc),
}


def engine_outputs(x_arr: Array, mission: MissionSpec, *,
                   couple_eta_cstar: bool = False,
                   outputs: tuple[str, ...] = ("Isp", "package_mass")) -> Array:
    """``f(x_array) -> stacked scalar outputs`` through the coupled solve."""
    for k in outputs:
        if k not in _SCALARS:
            raise KeyError(f"output '{k}' not in {tuple(_SCALARS)}")
    r = solve_engine(DesignVector.from_array(x_arr), mission,
                     couple_eta_cstar=couple_eta_cstar)
    return jnp.stack([_SCALARS[k](r) for k in outputs])


def ablation_delta(x: DesignVector, mission: MissionSpec,
                   key: str = "Isp") -> Array:
    """RQ1 probe: (coupled − frozen) for a scalar output, bounding how much of
    the result rests on the η_c* correlation (plan §5)."""
    on = solve_engine(x, mission, couple_eta_cstar=True)
    off = solve_engine(x, mission, couple_eta_cstar=False)
    return _SCALARS[key](on) - _SCALARS[key](off)
