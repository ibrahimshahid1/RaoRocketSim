"""
raosim.mdo.assembly — the walking-skeleton coupled engine residual (plan §12.5).

One residual signature (plan §12.1 rule 2): everything is R(y, x) = 0 stacked
into a single vector, solved by a square Newton root-find, and differentiated
via the implicit function theorem (Optimistix implicit diff — evaluation
report amendment #6: root-find, not least-squares implicit diff).

Skeleton state vector (scaled, fixed shape):

    y = [ Rt_hat, mdot_hat, Twg_hat ]

with residuals

    R1  thrust closure     F_target − CF(ε, γ, Pc, Pa) · Pc · π Rt²      = 0
    R2  c* mass-flow       mdot − Pc·At / (η_c* · c*_ideal(Pc, O/F))     = 0
    R3  throat thermal     T_wg − [T_aw − q(T_wg)/h_g(T_wg)]             = 0

R1/R2 are *naturally explicit* today (plan §4.1 warns against enlarging the
system to appear coupled) — they are carried as implicit states deliberately,
as integration scaffolding: the feedback edges that make them genuinely
implicit (regen Δp → feed → mdot; T_fuel_out → injector → η_c*; §5 loops)
land behind this interface in Phases 4a/5 without changing the solve/IFT
plumbing.  The parity test exploits the current triangularity: the solved
states must match the NumPy closed forms to solver tolerance.

Everything downstream of the state (injector velocities, pump duty, electric
power, battery/motor masses, the §3 mass ledger with explicit placeholders)
is EXPLICIT algebra in ``readouts`` — no ``max()`` in any differentiated
constraint path (plan rule 5): the battery power- and energy-limited masses
are exposed separately for the epigraph treatment at the NLP layer;
``package_mass_report`` (which does take the max) is a *reporting* scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import raosim.jax  # noqa: F401  -- x64 on
import jax
import jax.numpy as jnp
import optimistix as optx

from raosim.jax import thermal as jt
from raosim.jax.primitives import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
)
from raosim.mdo.properties import ChamberSurfaces, constant_chamber_surfaces
from raosim.mdo.schema import DesignVector, MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Reference scales (state nondimensionalization)                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateScales:
    Rt_ref: float
    mdot_ref: float
    T_ref: float

    @classmethod
    def from_mission(cls, mission: MissionSpec) -> "StateScales":
        # Crude but stable references: a 1.5 kN/cm²-class throat and the
        # c*-closure flow at the reference Pc; exact values are irrelevant
        # (only conditioning), so plain constants keep them static under jit.
        cstar = mission.eta_cstar * mission.c_star_ideal()
        Rt_ref = (mission.thrust / (1.4 * 3.0e6 * jnp.pi)) ** 0.5
        mdot_ref = 3.0e6 * jnp.pi * Rt_ref**2 / cstar
        return cls(float(Rt_ref), float(mdot_ref), float(mission.Tc))


# --------------------------------------------------------------------------- #
# Residual vector                                                              #
# --------------------------------------------------------------------------- #
def assemble_residual(y: Array, x: DesignVector, mission: MissionSpec,
                      surfaces: ChamberSurfaces, scales: StateScales) -> Array:
    """Stacked scaled residual R(y, x); shape (3,), fixed topology."""
    Rt = y[0] * scales.Rt_ref
    mdot = y[1] * scales.mdot_ref
    T_wg = y[2] * scales.T_ref

    Pc = x.Pc
    eps = x.eps

    gamma = surfaces.gamma(Pc, mission.OF)
    Tc = surfaces.Tc(Pc, mission.OF)
    c_star_del = mission.eta_cstar * surfaces.c_star_ideal(Pc, mission.OF)

    # R1 — thrust closure (CF at ambient, plan §6.1).
    Cf = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    At = jnp.pi * Rt * Rt
    R1 = (mission.thrust - Cf * Pc * At) / mission.thrust

    # R2 — pinned c* convention: mdot = Pc·At / c*_delivered (plan rule 2).
    R2 = (mdot - Pc * At / c_star_del) / scales.mdot_ref

    # R3 — series thermal circuit at the throat, solved implicitly (the same
    # equation jax.thermal.throat_wall_temperature fixed-points; here it is a
    # residual so the converged state is differentiated by the IFT instead of
    # unrolling — plan §4.4).
    Taw = jt.recovery_temperature(1.0, gamma, Tc, mission.Pr_gas)
    h_g = jt.bartz_hg(
        1.0, 1.0, Dt=2.0 * Rt, Pc=Pc, c_star=c_star_del,
        cp=mission.cp_gas, Pr=mission.Pr_gas, mu=mission.mu_gas,
        gamma=gamma, Tc=Tc, wall_temperature=T_wg,
        throat_curvature_radius=mission.throat_rd_factor * Rt,
    )
    R_tot = 1.0 / h_g + mission.t_wall / mission.k_wall + 1.0 / mission.h_c
    q = (Taw - mission.coolant_temperature) / R_tot
    R3 = (T_wg - (Taw - q / h_g)) / scales.T_ref

    return jnp.stack([R1, R2, R3])


# --------------------------------------------------------------------------- #
# State solve (square Newton + IFT)                                            #
# --------------------------------------------------------------------------- #
def initial_state(x: DesignVector, mission: MissionSpec,
                  surfaces: ChamberSurfaces, scales: StateScales) -> Array:
    """Closed-form seed (the current system is triangular; see module doc)."""
    gamma = surfaces.gamma(x.Pc, mission.OF)
    Cf = jt.ambient_thrust_coefficient(x.eps, gamma, x.Pc, mission.Pa)
    Rt = jnp.sqrt(mission.thrust / (Cf * x.Pc * jnp.pi))
    c_star_del = mission.eta_cstar * surfaces.c_star_ideal(x.Pc, mission.OF)
    mdot = x.Pc * jnp.pi * Rt * Rt / c_star_del
    Tc = surfaces.Tc(x.Pc, mission.OF)
    return jnp.stack([Rt / scales.Rt_ref, mdot / scales.mdot_ref,
                      0.6 * Tc / scales.T_ref])


def solve_states(x: DesignVector, mission: MissionSpec,
                 surfaces: ChamberSurfaces, scales: StateScales,
                 *, rtol: float = 1e-12, atol: float = 1e-12,
                 max_steps: int = 64) -> tuple[Array, Array]:
    """Newton root-find on the square R(y, x) = 0.

    Inner tolerances are set ≥100× tighter than any optimizer feasibility
    tolerance this layer will see (plan rule 6).  Returns (y*, R(y*, x)).
    """
    def fn(y, args):
        return assemble_residual(y, args, mission, surfaces, scales)

    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(fn, solver, initial_state(x, mission, surfaces, scales),
                         args=x, max_steps=max_steps, throw=False)
    y = sol.value
    return y, fn(y, x)


# --------------------------------------------------------------------------- #
# Explicit readouts + §3 mass ledger                                           #
# --------------------------------------------------------------------------- #
def readouts(y: Array, x: DesignVector, mission: MissionSpec,
             surfaces: ChamberSurfaces, scales: StateScales) -> dict:
    """Explicit post-state algebra.  Pure jnp; safe inside jit/grad."""
    Rt = y[0] * scales.Rt_ref
    mdot = y[1] * scales.mdot_ref
    T_wg = y[2] * scales.T_ref
    Pc, eps = x.Pc, x.eps

    gamma = surfaces.gamma(Pc, mission.OF)
    c_star_del = mission.eta_cstar * surfaces.c_star_ideal(Pc, mission.OF)
    Cf = jt.ambient_thrust_coefficient(eps, gamma, Pc, mission.Pa)
    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    Pe = Pc * isentropic_pressure_ratio(Me, gamma)
    Isp = Cf * c_star_del / mission.g0

    # Separation margin (corrected Schmucker, Östlund Eq. 30).
    sep_margin = jt.schmucker_separation_margin(eps, gamma, Pc, mission.Pa)

    # Injector split + velocities (orifice algebra; Cd folded into the
    # metering-area sizing downstream, velocities here are ideal jet speeds).
    mdot_f = mdot / (1.0 + mission.OF)
    mdot_o = mdot * mission.OF / (1.0 + mission.OF)
    dp_f = x.dp_f_frac * Pc
    dp_o = x.dp_o_frac * Pc
    v_f = jnp.sqrt(2.0 * dp_f / mission.rho_fuel)
    v_o = jnp.sqrt(2.0 * dp_o / mission.rho_ox)
    momentum_ratio = (mdot_o * v_o) / (mdot_f * v_f)

    # Feed ledger (§6.4; regen/line terms are explicit placeholders until
    # Phases 4a/5 supply them — always positive inside the design box, so no
    # clamping is needed or used here).
    rise_f = Pc * (1.0 + x.dp_f_frac) + mission.regen_dp_allowance \
        + mission.line_dp_allowance - mission.P_tank_fuel
    rise_o = Pc * (1.0 + x.dp_o_frac) + mission.line_dp_allowance \
        - mission.P_tank_ox
    Q_f = mdot_f / mission.rho_fuel
    Q_o = mdot_o / mission.rho_ox
    P_shaft_f = Q_f * rise_f / mission.eta_pump
    P_shaft_o = Q_o * rise_o / mission.eta_pump
    P_shaft = P_shaft_f + P_shaft_o
    P_elec = P_shaft / (mission.eta_motor * mission.eta_inverter)

    # Battery (Lee 2021 two-driver structure) — both branches exposed; the
    # NLP treats m_battery as an epigraph variable ≥ both (plan §6.4).
    E_req = P_elec * mission.burn_time / mission.eta_discharge
    m_batt_energy = E_req / mission.battery_energy_density
    m_batt_power = P_elec / mission.battery_power_density

    m_motor = P_shaft / mission.motor_power_density
    m_inverter = P_elec / mission.inverter_power_density
    m_pumps = P_shaft / mission.pump_specific_mass

    # §3 ledger — excluded items carried as explicit zeros (report §A.2.3).
    ledger = {
        "pumps_inducers": m_pumps,
        "motors": m_motor,
        "inverters": m_inverter,
        "battery_energy_limited": m_batt_energy,
        "battery_power_limited": m_batt_power,
        "thrust_chamber_placeholder": jnp.asarray(0.0),
        "injector_placeholder": jnp.asarray(0.0),
        "tanks_excluded": jnp.asarray(0.0),
        "pressurant_excluded": jnp.asarray(0.0),
        "valves_ignition_avionics_excluded": jnp.asarray(0.0),
    }
    m_batt_report = jnp.maximum(m_batt_energy, m_batt_power) \
        * mission.battery_structural_margin  # reporting only (module doc)
    package_mass_report = (m_pumps + m_motor + m_inverter + m_batt_report)

    return {
        "Rt": Rt, "mdot": mdot, "T_wg": T_wg,
        "Cf": Cf, "Me": Me, "Pe": Pe, "Isp_delivered": Isp,
        "separation_margin": sep_margin,
        "mdot_fuel": mdot_f, "mdot_ox": mdot_o,
        "v_fuel": v_f, "v_ox": v_o,
        "momentum_ratio_ox_over_fuel": momentum_ratio,
        "chi_fuel": x.dp_f_frac, "chi_ox": x.dp_o_frac,
        "pump_rise_fuel": rise_f, "pump_rise_ox": rise_o,
        "P_shaft": P_shaft, "P_electric": P_elec,
        "mass_ledger": ledger,
        "m_battery_energy_limited": m_batt_energy,
        "m_battery_power_limited": m_batt_power,
        "package_mass_report": package_mass_report,
        "propellant_mass": mdot * mission.burn_time,
    }


# --------------------------------------------------------------------------- #
# Differentiable end-to-end evaluation (the Phase-1/skeleton gate object)      #
# --------------------------------------------------------------------------- #
_SCALAR_OUTPUTS = (
    "Isp_delivered", "package_mass_report", "m_battery_energy_limited",
    "m_battery_power_limited", "P_electric", "Rt", "mdot", "T_wg",
    "separation_margin",
)


def make_engine_fn(mission: MissionSpec,
                   surfaces: ChamberSurfaces | None = None,
                   *, outputs: tuple[str, ...] = ("Isp_delivered",
                                                  "package_mass_report"),
                   ) -> Callable[[Array], Array]:
    """Build ``f(x_array) -> outputs`` through the CONVERGED engine state.

    The returned callable is jit/jacfwd/jacrev-safe; derivatives of the
    solved state come from Optimistix's implicit differentiation of the
    Newton root (plan §4.3 — never through iterations).
    """
    surfaces = surfaces if surfaces is not None else constant_chamber_surfaces(
        gamma=mission.gamma, Tc=mission.Tc, R_gas=mission.R_gas)
    scales = StateScales.from_mission(mission)
    for k in outputs:
        if k not in _SCALAR_OUTPUTS:
            raise KeyError(f"output '{k}' not in {_SCALAR_OUTPUTS}")

    def f(x_arr: Array) -> Array:
        if int(x_arr.shape[0]) != 4:
            raise ValueError("walking-skeleton engine function requires 4 values")
        # This historical four-variable skeleton has an explicit fixed-O/F
        # layout; it never infers O/F intent from the array length.
        x = DesignVector(
            Pc=x_arr[0],
            eps=x_arr[1],
            dp_f_frac=x_arr[2],
            dp_o_frac=x_arr[3],
            OF=mission.OF,
        )
        y, _ = solve_states(x, mission, surfaces, scales)
        out = readouts(y, x, mission, surfaces, scales)
        return jnp.stack([out[k] for k in outputs])

    return f
