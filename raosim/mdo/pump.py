"""
raosim.mdo.pump — Phase 6: differentiable pump + electric-feed block.

Closed-form jnp (no implicit state → plain jacfwd/jacrev), mirroring the
``raosim.pumps`` duty/specific-speed/NPSH conventions but with the binned
efficiency estimator replaced by a C¹ surrogate.  Physics (plan §6.4):

* duty: Q = ṁ/ρ, required head H = Δp_pump/(ρ g0), Δp_pump the §6.4 ledger rise
  (P_c + Δp_inj + Δp_regen + Δp_line − P_tank), ω = 2π N/60;
* dimensionless specific speed Ns = ω√Q/(g0 H)^¾ and suction specific speed
  Nss = ω√Q/(g0·NPSH)^¾ with NPSH = (p_in − p_v)/(ρ g0) (``pumps.py`` L2410
  form; SP-8052 suction screen);
* **C¹ efficiency** η(Ns) — a smooth log-Gaussian peaked at Ns_opt, replacing
  ``pumps._estimate_pump_efficiency`` (a C0-discontinuous step in Q that cannot
  be differentiated — plan §6.4 note).  Calibrated to the SP-125 rocket-pump
  band (Huzel & Huang: overall efficiency 60–85 %, rising with capacity/Ns; ~10 %
  below industrial) and the SP-8109 low-Ns penalty.  It is a *smooth screening
  surrogate*, not a fit to SP-125 Fig. 6-23 (image-only);
* electrical chain: P_hyd = ṁ Δp/ρ, P_shaft = P_hyd/η_pump,
  P_elec = P_shaft/(η_motor η_inv);
* tip-speed sizing U₂ = √(g0 H/ψ) with a stress-limited screen U₂ ≤ U₂,max
  (SP-8109), and the suction screen Nss ≤ Nss,max (SP-8052) — both exposed as
  margins, never clamped;
* **battery as an epigraph** (Lee 2021): the power- and energy-limited masses
  are returned *separately* (m_b ≥ Σ P_e Δt/(η_disch ρ_E) and m_b ≥ P_e/ρ_P);
  the NLP takes m_battery ≥ both — no ``max()`` in the differentiable core
  (plan rule 5).

Parity oracle: tests/test_mdo_pump.py checks Q/H/Ns/Nss/power against the
``pumps.py`` forms where they coincide, and pins that the new η(Ns) is C¹
(continuous value AND derivative) across the old bin edges where the estimator
jumps.
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# C¹ efficiency surrogate (replaces pumps._estimate_pump_efficiency)          #
# --------------------------------------------------------------------------- #
def pump_efficiency(Ns: Array, mission: MissionSpec) -> Array:
    """Smooth η(Ns): a log-Gaussian peaked at ``pump_ns_opt`` (C∞ ⊂ C¹).

    η = η_peak · exp(−½ (ln(Ns/Ns_opt)/σ)²).  Rises with specific speed to the
    SP-125 rocket-pump band and rolls off smoothly either side (Huzel & Huang
    60–85 %; SP-8109 low-Ns penalty).  Unlike ``pumps._estimate_pump_efficiency``
    this has no discontinuities, so it is safe inside the differentiable core.
    """
    Ns = jnp.maximum(Ns, 1e-9)
    z = jnp.log(Ns / mission.pump_ns_opt) / mission.pump_eta_ns_width
    return mission.pump_eta_peak * jnp.exp(-0.5 * z * z)


# --------------------------------------------------------------------------- #
# Per-stream pump duty + screens                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PumpStream:
    """Differentiable pump-duty readouts for one propellant stream."""

    Q: Array                  # volumetric flow [m³/s]
    head: Array               # required head [m]
    omega: Array              # shaft speed [rad/s]
    specific_speed: Array     # Ns = ω√Q/(g0 H)^¾  (dimensionless)
    efficiency: Array         # η(Ns)  (C¹)
    P_hydraulic: Array        # W
    P_shaft: Array            # W
    P_electric: Array         # W
    npsh_available: Array     # NPSH head [m]
    suction_specific_speed: Array  # Nss (dimensionless)
    nss_margin: Array         # Nss_max − Nss   (≥0 feasible, SP-8052)
    tip_speed: Array          # U₂ = √(g0 H/ψ) [m/s]
    tip_speed_margin: Array   # U₂,max − U₂     (≥0 feasible, SP-8109)


def pump_stream(*, mdot: Array, dp_rise: Array, rho: float, p_inlet: Array,
                p_vapor: float, N_rpm: Array, mission: MissionSpec) -> PumpStream:
    """One-stream pump duty, efficiency, power, and cavitation/stress screens.

    ``dp_rise`` is the §6.4 ledger pump rise (differentiable input); ``rho`` and
    ``p_vapor`` are per-propellant constants; ``N_rpm`` is the (continuous)
    design speed.  Pure jnp — jit/jacfwd/jacrev-safe.
    """
    g0 = mission.g0
    Q = mdot / rho
    head = dp_rise / (rho * g0)
    omega = 2.0 * jnp.pi * N_rpm / 60.0

    Ns = omega * jnp.sqrt(jnp.maximum(Q, 0.0)) / (g0 * head) ** 0.75
    eta = pump_efficiency(Ns, mission)

    P_hyd = mdot * dp_rise / rho
    P_shaft = P_hyd / eta
    P_elec = P_shaft / (mission.eta_motor * mission.eta_inverter)

    npsh = jnp.maximum(p_inlet - p_vapor, 0.0) / (rho * g0)
    Nss = omega * jnp.sqrt(jnp.maximum(Q, 0.0)) / (g0 * npsh) ** 0.75
    nss_margin = mission.pump_nss_max - Nss

    tip_speed = jnp.sqrt(g0 * head / mission.pump_head_coefficient)
    tip_margin = mission.pump_tip_speed_max - tip_speed

    return PumpStream(
        Q=Q, head=head, omega=omega, specific_speed=Ns, efficiency=eta,
        P_hydraulic=P_hyd, P_shaft=P_shaft, P_electric=P_elec,
        npsh_available=npsh, suction_specific_speed=Nss, nss_margin=nss_margin,
        tip_speed=tip_speed, tip_speed_margin=tip_margin,
    )


# --------------------------------------------------------------------------- #
# Battery epigraph (Lee 2021) — both branches exposed, no max() in core        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatteryEpigraph:
    energy_limited_mass: Array   # Σ P_e Δt /(η_disch ρ_E)
    power_limited_mass: Array    # P_e / ρ_P


def battery_masses(P_electric_total: Array, mission: MissionSpec) -> BatteryEpigraph:
    """Lee-2021 two-driver battery mass; the governing branch is selected by the
    NLP epigraph (m_b ≥ both), never a ``max()`` here (plan rule 5)."""
    e_req = P_electric_total * mission.burn_time / mission.eta_discharge
    m_energy = e_req / mission.battery_energy_density
    m_power = P_electric_total / mission.battery_power_density
    return BatteryEpigraph(energy_limited_mass=m_energy, power_limited_mass=m_power)


# --------------------------------------------------------------------------- #
# Two-stream electric feed (fuel + oxidiser) + drive masses                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ElectricFeed:
    fuel: PumpStream
    ox: PumpStream
    P_electric_total: Array
    battery: BatteryEpigraph
    motor_mass: Array
    inverter_mass: Array
    pump_mass: Array


def electric_feed(*, mdot_fuel: Array, mdot_ox: Array, dp_rise_fuel: Array,
                  dp_rise_ox: Array, N_rpm: Array,
                  mission: MissionSpec) -> ElectricFeed:
    """Full electric feed: both pump streams + battery epigraph + drive masses
    (specific-power screening masses, §3 ledger)."""
    f = pump_stream(mdot=mdot_fuel, dp_rise=dp_rise_fuel, rho=mission.rho_fuel,
                    p_inlet=mission.P_tank_fuel, p_vapor=mission.p_vapor_fuel,
                    N_rpm=N_rpm, mission=mission)
    o = pump_stream(mdot=mdot_ox, dp_rise=dp_rise_ox, rho=mission.rho_ox,
                    p_inlet=mission.P_tank_ox, p_vapor=mission.p_vapor_ox,
                    N_rpm=N_rpm, mission=mission)
    P_elec = f.P_electric + o.P_electric
    P_shaft = f.P_shaft + o.P_shaft
    batt = battery_masses(P_elec, mission)
    return ElectricFeed(
        fuel=f, ox=o, P_electric_total=P_elec, battery=batt,
        motor_mass=P_shaft / mission.motor_power_density,
        inverter_mass=P_elec / mission.inverter_power_density,
        pump_mass=P_shaft / mission.pump_specific_mass,
    )
