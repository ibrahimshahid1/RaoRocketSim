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
import jax.scipy as jsp
import optimistix as optx

from raosim.jax import thermal as jt
from raosim.jax.primitives import isentropic_pressure_ratio, mach_from_area_ratio
from raosim.mdo.assembly import StateScales
from raosim.mdo.properties import ChamberSurfaces, constant_chamber_surfaces
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.grid import (
    build_station_grid, chamber_barrel_length, GridTopology,
)
from raosim.mdo.cooling import (
    solve_cooling, CoolingMarch, film_cooling_efficiency,
)
from raosim.mdo.injector import injector_readouts, InjectorReadout
from raosim.mdo.mass import chamber_mass, ChamberMassBreakdown
from raosim.mdo.envelope import (
    chamber_envelope, envelope_margins, fractional_margin, ChamberEnvelope,
)
from raosim.mdo.pump import electric_feed, ElectricFeed
from raosim.mdo.structures import nozzle_collapse_screen, NozzleCollapseScreen

Array = jnp.ndarray


def _smooth_min(values: Array, sharpness: float) -> Array:
    """Conservative differentiable lower envelope (never above ``min``)."""
    v = jnp.asarray(values, dtype=jnp.float64)
    return -jsp.special.logsumexp(-sharpness * v) / sharpness


def _smooth_max(values: Array, sharpness: float) -> Array:
    """Conservative differentiable upper envelope (never below ``max``)."""
    v = jnp.asarray(values, dtype=jnp.float64)
    return jsp.special.logsumexp(sharpness * v) / sharpness


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


def _resolve_of(x: DesignVector, mission: MissionSpec) -> Array:
    """Effective mixture ratio for this solve.

    ``x.of_is_variable`` is static pytree aux, so this is a *Python* branch
    resolved at trace time -- it never becomes a design-dependent switch inside
    the graph (plan §0.1).  When O/F is not a live variable the mission's fixed
    value is authoritative: ``DesignVector.OF``'s class default is an RP-1
    number and would silently mis-split the propellant flow on any other
    combination.
    """

    return (jnp.asarray(x.OF, dtype=jnp.float64) if x.of_is_variable
            else jnp.asarray(mission.OF, dtype=jnp.float64))


def _split(mdot: Array, mission: MissionSpec,
           OF: Array | None = None) -> tuple[Array, Array]:
    """Propellant mass-flow split at the effective mixture ratio."""

    of = jnp.asarray(mission.OF if OF is None else OF, dtype=jnp.float64)
    mdot_f = mdot / (1.0 + of)
    mdot_o = mdot * of / (1.0 + of)
    return mdot_f, mdot_o


def _eta_cstar(mdot: Array, x: DesignVector, mission: MissionSpec,
               surfaces: ChamberSurfaces, couple: bool) -> Array:
    if not couple:
        base = jnp.asarray(mission.eta_cstar, dtype=jnp.float64)
    else:
        mdot_f, mdot_o = _split(mdot, mission, _resolve_of(x, mission))
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
    of = _resolve_of(x, mission)

    gamma = surfaces.gamma(Pc, of)
    cstar_ideal = surfaces.c_star_ideal(Pc, of)
    eta_cs = _eta_cstar(mdot, x, mission, surfaces, couple)
    cstar_del = eta_cs * cstar_ideal

    Cf_ideal = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    Cf = mission.eta_CF * Cf_ideal
    At = jnp.pi * Rt * Rt
    R1 = (mission.thrust - Cf * Pc * At) / mission.thrust
    R2 = (mdot - Pc * At / cstar_del) / scales.mdot_ref
    return jnp.stack([R1, R2])


def _initial_state(x: DesignVector, mission: MissionSpec,
                   surfaces: ChamberSurfaces, scales: StateScales) -> Array:
    of = _resolve_of(x, mission)
    gamma = surfaces.gamma(x.Pc, of)
    Cf = mission.eta_CF * jt.ambient_thrust_coefficient(
        x.eps, gamma, x.Pc, mission.Pa)
    Rt = jnp.sqrt(mission.thrust / (Cf * x.Pc * jnp.pi))
    cstar_del = mission.eta_cstar * surfaces.c_star_ideal(x.Pc, of)
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
    Cf_ideal: Array
    Cf: Array
    Isp: Array
    #: Mixture ratio this solve actually used.  When O/F is a design variable
    #: this is the optimiser's value, NOT ``mission.OF`` -- the host bridge must
    #: read it from here or it will re-derive a stale split.
    OF: Array
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
    # Smooth electric-feed objective.  This is deliberately not labelled as a
    # physical package total: its battery branch is a log-sum-exp surrogate.
    objective_mass: Array
    electric_package_exact_mass: Array
    # Thrust-chamber structural metal, integrated on the station grid.  Real
    # hardware mass, not a surrogate -- but conditional on the closeout
    # thickness assumption documented in :mod:`raosim.mdo.mass`.
    chamber_mass: ChamberMassBreakdown
    # Smallest enclosing cylinder of the cooled chamber (SP-125 §2.1 item 6).
    # A lower bound on the installed envelope -- no flange, injector body or
    # feed hardware.  See raosim.mdo.envelope.
    envelope: ChamberEnvelope
    mass_ledger: dict
    # constraint margins (reporting scalars; ≥0 feasible)
    constraints: dict
    # reported diagnostics (not constraints)
    diagnostics: dict
    # Explicit numerical status: feasibility must not be inferred from a
    # partially converged root merely because its reported physics is finite.
    solver_residual_max: Array
    solver_status_ok: Array
    solver_converged: Array
    finite: Array

    @property
    def package_mass(self) -> Array:
        """Deprecated compatibility alias for :attr:`objective_mass`."""

        return self.objective_mass


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
    of = _resolve_of(x, mission)

    gamma = surfaces.gamma(Pc, of)
    Tc = surfaces.Tc(Pc, of)
    eta_cs = _eta_cstar(mdot, x, mission, surfaces, couple_eta_cstar)
    cstar_del = eta_cs * surfaces.c_star_ideal(Pc, of)

    # --- performance ------------------------------------------------------- #
    Cf_ideal = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    Cf = mission.eta_CF * Cf_ideal
    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    Pe = Pc * isentropic_pressure_ratio(Me, gamma)
    Isp = Cf * cstar_del / mission.g0
    # The helper exposes the raw attached-flow ratio Pe/p_sep.  Design
    # admission requires the SP-8120-style 20% reserve, while vacuum is an
    # automatic finite pass because an ambient-referenced separation criterion
    # has no physical threshold there.
    sep_raw_margin = jt.schmucker_separation_margin(eps, gamma, Pc, mission.Pa)
    sep_margin = jnp.where(
        mission.Pa <= 0.0, 1.0,
        sep_raw_margin - mission.separation_design_margin,
    )

    mdot_f, mdot_o = _split(mdot, mission, of)

    # --- cooling (inner IFT solve on the stationwise wall temperatures) ----- #
    grid = build_station_grid(Rt, eps, mission, topo, gamma=gamma)
    # Defined architecture: ``film_frac`` is a fuel branch that bypasses the
    # regenerative jacket.  Only the remainder reaches that jacket; sending
    # total fuel through it double-counts film and overstates heat capacity.
    mdot_film = mdot_f * x.film_frac
    mdot_cool = mission.cooling_fraction * (mdot_f - mdot_film)
    T_wg, cooling = solve_cooling(grid, Pc=Pc, gamma=gamma, Tc=Tc,
                                  c_star_del=cstar_del, mdot_cool=mdot_cool,
                                  mission=mission, channel_width=x.channel_width,
                                  channel_height=x.channel_height,
                                  film_frac=x.film_frac, mdot_film=mdot_film,
                                  t_wall=x.t_wall,
                                  coolant_outlet_pressure=(
                                      Pc * (1.0 + x.dp_f_frac)
                                  ))
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
    m_batt_objective = _smooth_max(jnp.stack([
        feed.battery.energy_limited_mass, feed.battery.power_limited_mass]), 2.0) \
        * mission.battery_structural_margin
    m_batt_energy_installed = (
        feed.battery.energy_limited_mass * mission.battery_structural_margin
    )
    m_batt_power_installed = (
        feed.battery.power_limited_mass * mission.battery_structural_margin
    )
    m_batt_exact = jnp.maximum(
        m_batt_energy_installed, m_batt_power_installed
    )
    # Thrust-chamber structure is now integrated from the same station grid the
    # cooling march uses (SP-125 eq. 8-32 shell mass; see mdo/mass.py), so the
    # optimizer can trade wall thickness and channel geometry against feed-
    # system mass instead of minimising only the electric package.  The
    # injector remains a hydraulic sizing here -- its hardware mass needs the
    # resolved machined layout, which lives on the host side
    # (raosim.mass_ledger.injector_mass_ledger) -- so it stays unavailable
    # rather than being written as a misleading zero.
    chamber = chamber_mass(
        grid, mission,
        t_wall=x.t_wall,
        channel_width=x.channel_width,
        channel_height=x.channel_height,
        # The jacket is sized against the SOLVED jacket pressure, so the
        # structure and the hydraulics cannot disagree about the load.
        coolant_pressure=cooling.coolant_pressure,
    )
    # SP-8087 sec. 2.1.3 nozzle hoop-compression collapse.  The jacket alone
    # is screened (conservative: crediting the land-stiffened liner/jacket
    # sandwich needs SP-8007 sec. 4.4), over the full divergent length, which
    # assumes no retainer bands.
    collapse = nozzle_collapse_screen(
        grid, mission,
        gas_pressure=cooling.gas_pressure,
        shell_thickness=chamber.closeout_thickness,
        shell_length=(mission.length_pct / 100.0)
        * (Rt * jnp.sqrt(x.eps) - Rt) / jnp.tan(jnp.deg2rad(15.0)),
    )
    # SP-125 §2.1 item 6: the smallest enclosing cylinder of the cooled thrust
    # chamber.  A LOWER BOUND on the installed envelope -- it has no flange,
    # injector body or feed hardware; see raosim.mdo.envelope.
    envelope = chamber_envelope(
        grid,
        t_wall=x.t_wall,
        channel_height=x.channel_height,
        # Reuse the jacket the mass ledger already solved, so the envelope and
        # the mass cannot describe different jackets.
        closeout_thickness=chamber.closeout_thickness,
    )
    envelope_d_margin, envelope_l_margin = envelope_margins(envelope, mission)
    ledger = {
        "pumps": feed.pump_mass, "motors": feed.motor_mass,
        "inverters": feed.inverter_mass,
        "battery_energy_installed": m_batt_energy_installed,
        "battery_power_installed": m_batt_power_installed,
        "battery_selected_exact": m_batt_exact,
        "battery_objective_smooth": m_batt_objective,
        "thrust_chamber_liner": chamber.liner,
        "thrust_chamber_lands": chamber.lands,
        "thrust_chamber_closeout": chamber.closeout,
        "thrust_chamber": chamber.total,
    }
    objective_mass = (
        feed.pump_mass
        + feed.motor_mass
        + feed.inverter_mass
        + m_batt_objective
    )
    electric_package_exact_mass = (
        feed.pump_mass
        + feed.motor_mass
        + feed.inverter_mass
        + m_batt_exact
    )

    # --- constraint margins (reporting scalars; ≥0 feasible) ---------------- #
    outer_residual_max = jnp.max(jnp.abs(resid))
    outer_status_ok = sol.result == optx.RESULTS.successful
    finite = (jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(resid))
              & jnp.isfinite(Cf) & jnp.isfinite(Isp) & cooling.finite)
    # The root API is intentionally called with throw=False so infeasible NLP
    # probes remain differentiable.  Residual closure plus finiteness is the
    # JIT-safe solver-success predicate carried into feasibility below.
    outer_tol = jnp.asarray(max(rtol, atol), dtype=jnp.float64) * 10.0
    solver_status_ok = outer_status_ok & cooling.solver_status_ok
    solver_converged = (solver_status_ok & finite
                        & (outer_residual_max <= outer_tol)
                        & cooling.solver_converged)

    chart_scale = jnp.stack([
        grid.chart_domain_violation[0] / 46.0,
        grid.chart_domain_violation[1] / 46.0,
        grid.chart_domain_violation[2] / 40.0,
        grid.chart_domain_violation[3] / 40.0,
    ])
    property_violation = surfaces.domain_violation(Pc, of)
    property_scale = jnp.stack([
        property_violation[0] / (surfaces.gamma.xg[-1] - surfaces.gamma.xg[0]),
        property_violation[1] / (surfaces.gamma.xg[-1] - surfaces.gamma.xg[0]),
        property_violation[2] / (surfaces.gamma.yg[-1] - surfaces.gamma.yg[0]),
        property_violation[3] / (surfaces.gamma.yg[-1] - surfaces.gamma.yg[0]),
    ])

    constraints = {
        "thrust_residual": resid[0],
        "separation_margin": sep_margin,
        # Retain the exact stationwise extrema for IFT wall constraints: a
        # broad smooth envelope perturbs their root sensitivities.  The active
        # stations are fixed-topology and tied only at measure-zero points.
        "coking_margin_min": jnp.min(cooling.coking_margin),
        "land_min": cooling.land_min,
        "chug_margin_min": _smooth_min(jnp.stack([
            injector.chug_margin_fuel, injector.chug_margin_ox]), 100.0),
        "pintle_transition_margin": injector.transition_margin,
        "nss_margin_min": _smooth_min(jnp.stack([
            feed.fuel.nss_margin, feed.ox.nss_margin]), 100.0),
        "tip_speed_margin_min": _smooth_min(jnp.stack([
            feed.fuel.tip_speed_margin, feed.ox.tip_speed_margin]), 10.0),
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
        # SP-125 combined liner stress. ``liner_sigma_allow`` is already the
        # post-FOS allowable; post-processing reconstructs the traditional
        # material yield as allowable*FOS so both pipelines enforce the same
        # governing inequality.
        "structural_stress_margin": (
            mission.liner_sigma_allow
            - jnp.max(
                cooling.sigma_thermal
                + jnp.abs(cooling.sigma_pressure_profile)
            )
        ),
        # allowable gas-side wall temperature (Mirzamoghadam: a primary design
        # criterion).  Required now that t_wall is a variable — a thicker wall
        # lowers T_wc but RAISES T_wg, so both sides must be bounded.
        "wall_temp_margin": mission.liner_T_wg_max - jnp.max(T_wg),
        # Film system must provide the required flow with its configured
        # capacity margin (SP-8087 design point commonly sets this to 2x).
        "film_capacity_margin": (mission.film_system_capacity_fraction
                                 - mission.film_capacity_margin * x.film_frac),
        # Both interpolant and chart evaluators clamp only for numerical safety;
        # their actual valid domains are hard NLP constraints.
        "property_domain_margin": -_smooth_max(property_scale, 40.0),
        "chart_domain_margin": -_smooth_max(chart_scale, 40.0),
        "wall_monotonic_margin": grid.wall_monotonic_margin,
        # The chamber barrel must have non-negative length.  SP-125 defines the
        # chamber volume as injector face to throat plane (printed p. 88), so
        # the shoulder, convergent cone and upstream throat arc already consume
        # part of L*.A_t; at small L*, high contraction ratio or a shallow
        # convergent angle they can consume all of it.  Reported as a margin
        # rather than clamped: clamping would manufacture chamber volume the
        # design does not have.
        "chamber_volume_margin": chamber_barrel_length(Rt, mission),
        # SP-125 (printed p. 336): the thin-shell hoop treatment used to size
        # the structural jacket is only valid while t/r <= ~1/15.  A jacket
        # thick enough to violate it is reporting that the alloy or the jacket
        # pressure is wrong -- e.g. a copper closeout at this Pc needs 5-7 mm
        # on a 91 mm radius, which is outside the model.
        "jacket_thin_shell_margin": chamber.closeout_thin_shell_margin,
        # SP-8087 sec. 2.1.3's third structural job: "hoop support about the
        # expansion nozzle to resist collapse from hoop compression ... during
        # operation at sea level, where jet separation occurs during start and
        # shutdown and the nozzle runs overexpanded".  SP-8120 sec. 2.2 records
        # the failure.  This is NOT the separation constraint -- separation asks
        # whether the flow detaches, collapse asks whether the shell survives
        # the external pressure while attached and overexpanded.
        "nozzle_collapse_margin": collapse.normalized_margin,
        # --- SP-125 §2.1 requirement screens (items 5 and 6) ---------------- #
        # FRACTIONAL margins (1 - value/limit), dimensionless and O(1) at every
        # thrust class -- see raosim.mdo.envelope.fractional_margin for why the
        # absolute form is unusable here.  Inert at the MissionSpec sentinel
        # defaults; they bind only when a requirement supplies a real limit.
        #
        # All three screen a LOWER BOUND on the true installed quantity, so a
        # satisfied margin does NOT prove the requirement is met.  That is why
        # the requirement layer classifies them as partially enforced rather
        # than enforced, and why the mass one is named for the partial quantity
        # it actually bounds.  Reporting them as full requirement satisfaction
        # would be exactly the "fake zero" failure mode the output contract
        # exists to prevent.
        "envelope_diameter_margin": envelope_d_margin,
        "envelope_length_margin": envelope_l_margin,
        # dry_mass_partial = smooth electric-feed objective + thrust-chamber
        # structure.  Missing: injector hardware, manifolds, valves, lines,
        # gimbal, mounts (see docs/HARDWARE_MASS_LEDGER.md `excludes`).
        "dry_mass_partial_margin": fractional_margin(
            objective_mass + chamber.total, mission.dry_mass_max
        ),
        "engine_residual_margin": outer_tol - outer_residual_max,
        "cooling_residual_margin": (jnp.asarray(max(rtol, atol), dtype=jnp.float64)
                                    * 10.0 - cooling.solver_residual_max),
        "solver_status_margin": jnp.where(solver_status_ok, 1.0, -1.0),
        "finite_margin": jnp.where(finite, 1.0, -1.0),
    }
    # reported diagnostics (NOT constraints — the coolant-Mach margin is ~173x,
    # so constraining it would only add a dead Jacobian column, §10.4)
    diagnostics = {
        "coolant_mach": cooling.coolant_mach,
        "coolant_mach_limit": jnp.asarray(0.35),
        "coolant_velocity": cooling.coolant_velocity,
        "sigma_thermal_max": jnp.max(cooling.sigma_thermal),
        "sigma_pressure": cooling.sigma_pressure,
        "sigma_combined_max": jnp.max(
            cooling.sigma_thermal
            + jnp.abs(cooling.sigma_pressure_profile)
        ),
        "film_capacity_required": (x.film_frac * mission.film_capacity_margin),
        "film_system_capacity": jnp.asarray(mission.film_system_capacity_fraction),
        "eta_film_cooling": film_cooling_efficiency(
            mdot_film, grid, mission),
        # Physical counterparts of the three fractional requirement margins,
        # kept here so a report can print metres and kilograms while the NLP
        # consumes the dimensionless form.  All three are LOWER BOUNDS on the
        # installed quantity (no flange, injector body or feed hardware).
        "envelope_diameter_partial": envelope.diameter,
        "envelope_length_partial": envelope.length,
        "dry_mass_partial": objective_mass + chamber.total,
    }

    return EngineResult(
        Rt=Rt, mdot=mdot, T_wg=T_wg, eta_cstar=eta_cs, Cf_ideal=Cf_ideal,
        Cf=Cf, Isp=Isp, Me=Me, Pe=Pe,
        OF=of,
        thrust_residual=resid[0], cooling=cooling, injector=injector, feed=feed,
        dp_regen=dp_regen, dp_rise_fuel=dp_rise_f, dp_rise_ox=dp_rise_o,
        objective_mass=objective_mass,
        electric_package_exact_mass=electric_package_exact_mass,
        chamber_mass=chamber,
        envelope=envelope,
        mass_ledger=ledger,
        constraints=constraints,
        diagnostics=diagnostics, solver_residual_max=outer_residual_max,
        solver_status_ok=solver_status_ok, solver_converged=solver_converged,
        finite=finite,
    )


# --------------------------------------------------------------------------- #
# Scalar accessor for AD / NLP (jit/jacfwd/jacrev-safe)                         #
# --------------------------------------------------------------------------- #
_SCALARS = {
    "Isp": lambda r: r.Isp,
    "objective_mass": lambda r: r.objective_mass,
    # Backward-compatible output name.  New callers should request
    # ``objective_mass`` so a smooth surrogate cannot be mistaken for hardware.
    "package_mass": lambda r: r.package_mass,
    "P_electric": lambda r: r.feed.P_electric_total,
    "Rt": lambda r: r.Rt,
    "mdot": lambda r: r.mdot,
    "eta_cstar": lambda r: r.eta_cstar,
    "OF": lambda r: r.OF,
    "dp_regen": lambda r: r.dp_regen,
    "coking_margin_min": lambda r: r.constraints["coking_margin_min"],
    "separation_margin": lambda r: r.constraints["separation_margin"],
    "T_wc_max": lambda r: jnp.max(r.cooling.T_wc),
    # Thrust-chamber structural metal (SP-125 eq. 8-32 shell integral).
    "thrust_chamber_mass": lambda r: r.chamber_mass.total,
    # Electric feed package + thrust-chamber structure.  Still NOT a complete
    # engine dry mass: injector hardware, manifolds, valves, lines, gimbal and
    # mounts are absent, and the injector branch is only resolvable host-side.
    "dry_mass_partial": lambda r: r.objective_mass + r.chamber_mass.total,
    # SP-125 §2.1 item 6.  Lower bounds on the installed envelope: the cooled
    # chamber only, with no flange, injector body or feed hardware.
    "envelope_diameter_partial": lambda r: r.envelope.diameter,
    "envelope_length_partial": lambda r: r.envelope.length,
}


def engine_outputs(x_arr: Array, mission: MissionSpec, *,
                   couple_eta_cstar: bool = False,
                   outputs: tuple[str, ...] = ("Isp", "objective_mass")) -> Array:
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
