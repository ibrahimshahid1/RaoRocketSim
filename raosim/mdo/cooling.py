"""
raosim.mdo.cooling — Phase 4a: 1-D + fin regenerative-cooling residual block.

Physics (plan §6.2, all corpus-anchored, mirroring the audited NumPy oracle in
``raosim.physics``):

* gas side — full Bartz (1957) h_g with the σ property-variation factor
  (``jax.thermal.bartz_hg`` / ``bartz_sigma``, exponents 0.68/0.12 at ω=0.6),
  station Mach/area from ``mdo.grid``; turbulent recovery T_aw with r = Pr^(1/3)
  (``jax.thermal.recovery_temperature``);
* series thermal circuit per station through the hot wall (SP-8087 regen
  framing: coolant, Δp, wall temperature and geometry as one coupled
  allocation), with the rib/land **fin efficiency** η_f = tanh(mH)/(mH),
  m = √(2 h_c/(k_w t_land)), carrying the aspect-ratio (HARCC) trend
  (Huzel & Huang SP-125 §4; Pizzarelli 2011 / Carlile 1992 resolve the same
  first-order mechanism in full 2-D — that is the 4b deepening, not 4a);
* coolant side — **Sieder–Tate** h_c via the audited ``jax.thermal.sieder_tate_hc``
  primitive (0.027·Re^0.8·Pr^(1/3)·(μ_b/μ_w)^0.14), evaluated with μ_wall = μ_bulk
  at this screening fidelity so the property ratio is unity (Sieder & Tate 1936;
  ``physics.sieder_tate_coefficient``);
* hydraulics — Darcy Δp with 64/Re laminar and Blasius/Swamee–Jain turbulent
  (explicit, smooth — deliberately not the implicit Colebrook; White §6);
* energy march — ṁ_cool·cp·dT_c = q″·dA over each wall segment, **counterflow**
  (coolant enters at the nozzle-exit manifold, exits at the injector end),
  discretised as a first-order **upwind finite-volume** march in the coolant
  flow direction (each of the n−1 wall segments is crossed exactly once, the
  circuit for a crossing evaluated at its upwind node).  The march is explicit
  algebra given the stationwise wall temperatures — nothing is unrolled through
  solver iterations (plan rule 3): the T_wg array lives in the state vector and
  is driven to R = 0 by the outer Newton/IFT (``solve_cooling``).

Fixed-geometry consequence: with static channel count/section (N, w, h) and
constant screening coolant properties, the coolant mass flux G, Re, Dh, velocity
and hence h_c and the Darcy factor are *uniform* along the jacket; only the fin
augmentation (through the land width, which follows the local radius), h_g and
T_aw vary station-to-station.  A width-tapered (constant-aspect) channel and
variable properties are the 4b/Phase-2 deepenings.

Channel-fit constraint (plan §9 — never ``clip()`` invalid geometry): the land
between channels, t_land = 2πr_cool/N − w, must stay positive.  ``cooling_march``
still floors it for numerical safety in η_f, but ALSO returns the *true* minimum
land ``land_min`` so the NLP layer can carry ``land_min ≥ land_floor`` as an
explicit inequality instead of silently absorbing an infeasible packing.

Coolant coking constraint (SP-8087 recommended practice): RP-1 cokes on the
liquid side of the heated wall above ~850°F (728 K) liquid-wall temperature
(onset 800–900°F, 700–756 K; Sellers, ARS J. 31(5), 1961).  The block therefore
computes the coolant-side wall temperature T_wc = T_wg − q″·t_wall/k_wall and
returns the stationwise ``coking_margin`` = T_coke_limit − T_wc (≥0 feasible),
a differentiable inequality for the NLP.  With the *screening* (non-optimised)
channel geometry this margin is strongly negative near the throat — the finding
that the fixed-geometry jacket is thermally infeasible at P_c = 3 MPa; the
optimiser is meant to size N/w/h/t_wall (or lower P_c, add film cooling) to
recover it, which is precisely why the constraint must be visible rather than
absorbed.

Residual block:  R_i = T_wg,i − [T_aw,i − q″_i / h_g,i]   (i = 1..n),
with q″_i from the series circuit at the *marched* local coolant temperature.

NumPy twin: ``tests/test_mdo_cooling.py`` carries a plain-Python re-implementation
(same discretisation, same primitives) proving ~1e-9 parity — the §12.1 rule-1
oracle for this block.  Cross-validation against the richer
``physics.regenerative_cooling_analysis`` (curvature correction, variable
properties, 2-D wall options) is a host-side follow-up and the 4b ladder.
"""

from __future__ import annotations

from dataclasses import dataclass

import raosim.jax  # noqa: F401  -- enables float64
import jax.numpy as jnp
import optimistix as optx
from jax import lax

from raosim.jax import thermal as jt
from raosim.mdo.grid import StationGrid
from raosim.mdo.schema import MissionSpec

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Small smooth primitives (jnp mirrors of raosim.physics pieces)               #
# --------------------------------------------------------------------------- #
def fin_efficiency(h_c: Array, k_wall: float, land_width: Array,
                   channel_height: float) -> Array:
    """η_f = tanh(mH)/(mH), m = √(2 h_c/(k_w t_land)) — jnp mirror of
    ``physics.fin_efficiency`` (Huzel & Huang SP-125 §4)."""
    m = jnp.sqrt(2.0 * h_c / (k_wall * jnp.maximum(land_width, 1e-9)))
    mH = jnp.maximum(m * channel_height, 1e-12)
    return jnp.tanh(mH) / mH


def film_slot_state(mdot_film: Array, slot_height: Array, grid: StationGrid,
                    mission: MissionSpec, *, v_core: Array) -> tuple:
    """Film-slot velocity, velocity ratio and Reynolds number from **continuity**.

    Hatch & Papell (NASA TN D-130) size the tangential slot by continuity,
    S = Ẇ_c/(ρ_c V_c π D) (their sample calculation); inverted for a known slot
    height on an annular slot of chamber diameter D:

        A_slot = π D S,   V_c = ṁ_film/(ρ_c A_slot),
        VR = V_c/V_g,     Re_c = ρ_c V_c S / μ_c     (slot height is the
                                                      characteristic length)

    Returns ``(v_film, VR, Re_c, A_slot)``.  Pure jnp; the film injector is now
    real geometry rather than a scale factor.
    """
    D_ch = 2.0 * grid.r[0]
    A_slot = jnp.pi * D_ch * jnp.maximum(slot_height, 1e-9)
    v_film = mdot_film / (mission.rho_cool * A_slot)
    VR = v_film / jnp.maximum(v_core, 1e-9)
    Re_c = mission.rho_cool * v_film * slot_height / mission.mu_cool
    return v_film, VR, Re_c, A_slot


def film_cooling_efficiency(mdot_film: Array, grid: StationGrid,
                            mission: MissionSpec) -> Array:
    """η_fc — Stechman's "dimensionless liquid coolant efficiency factor",
    derived from the local film state rather than assumed.

    Stechman, Oberstone & Howell (AIAA J. Spacecraft 6(2), 1969) correct the
    liquid coolant flow by an efficiency factor "that is a function of coolant
    Reynolds number", Re_L = W_L/(π D μ_L).  Grisson (AEDC-TR-91-1 §2.1)
    supplies the mechanism and the numbers: below **Knuth's critical
    flow-per-circumference**

        Γ_cr = 1.01e5 · μ_v² / μ_ℓ

    the film is smooth and nearly all the coolant is used; above it large waves
    shear droplets from their crests and the mass-loss rate becomes "2 to 4
    times the normal evaporation rate", so the effective efficiency drops to
    ≈1/2–1/4.  This returns a C¹ blend between those two literature limits.
    """
    D_ch = 2.0 * grid.r[0]
    gamma_film = mdot_film / (jnp.pi * D_ch)                 # kg/(m s)
    gamma_cr = (mission.film_knuth_coeff * mission.film_mu_vapor ** 2
                / mission.mu_cool)
    # smooth logistic in log(Γ/Γ_cr): stable below, wavy above
    z = jnp.log(jnp.maximum(gamma_film, 1e-12) / gamma_cr) \
        / mission.film_transition_width
    s = 0.5 * (1.0 + jnp.tanh(z))
    return (1.0 - s) * mission.film_eta_smooth + s * mission.film_eta_wavy


def film_effectiveness(film_frac: Array, grid: StationGrid,
                       mission: MissionSpec, *, mdot_film: Array,
                       slot_height: Array, gamma: Array, Tc: Array,
                       h_g: Array, T_aw: Array) -> Array:
    """Film effectiveness for a **LIQUID** film, from a phase-change energy
    balance and a film-cooled length — *not* a gaseous velocity-ratio correlation.

    Why not the classical ε = C(X/VR)^(−0.8)Re_c^(0.2) family (Stollery,
    Hartnett, Tribus; Hatch & Papell NASA TN D-130)?  Those are fitted to
    **gaseous** coolants (air, helium) at velocity ratios 0.45–33.3.  An RP-1
    film is a *liquid*: at ρ ≈ 810 kg/m³ the continuity velocity through any
    sane annular slot gives VR ≈ 10⁻³ — three orders below the validated band,
    so applying them here is extrapolation, not validation.  (``film_slot_state``
    computes that VR explicitly and ``film_slot_validity`` reports it.)

    The literature-correct liquid model (Shine & Nidhi 2018 §4.3: Kinney,
    Graham, Sellers and Emmons all "equated the convective energy transfer on
    the surface of the liquid film from the hot gas stream to the energy
    utilized for the phase change of the liquid coolant"; Huzel & Huang
    SP-125 Eq. 4-34 / Shine Eq. 12) is:

    1. the film absorbs, per unit mass, the enthalpy
       ``H = c_p,l (T_wg − T_co) + ΔH_vap + c_p,v (T_aw − T_wg)``
       (sensible liquid heating + latent heat + vapour superheat);
    2. the hot-gas convective load per unit wall area is ``q = h_g (T_aw − T_wg)``;
    3. so the film survives a **film-cooled length** whose wetted area is
       ``A_FCL = η_fc · ṁ_film · H / q`` — the classic heat balance;
    4. within that length the wall is protected; beyond it the film is spent.
       A smooth (C¹) axial decay ``ε = ε₀ / (1 + (A/A_FCL)^p)`` replaces the
       hard on/off cut so the model stays differentiable while reproducing the
       physical picture of a finite, flow-rate-proportional protected length.

    ``η_fc`` (film-cooling efficiency, SP-125) is the fraction of coolant that
    cools rather than being entrained into the core — the one empirical constant,
    and the same quantity Mirzamoghadam's Aerojet entrainment fraction encodes.
    """
    ff = jnp.maximum(film_frac, 0.0)
    n = grid.r.shape[0]

    # (1) per-mass enthalpy absorption of the liquid film (SP-125 Eq. 4-34)
    T_wall_ref = jnp.asarray(mission.rp1_coking_wall_temp_K, dtype=jnp.float64)
    T_co = jnp.asarray(mission.coolant_temperature, dtype=jnp.float64)
    H = (mission.cp_cool * jnp.maximum(T_wall_ref - T_co, 0.0)
         + mission.film_latent_heat
         + mission.cp_cool_vapor * jnp.maximum(T_aw - T_wall_ref, 0.0))

    # (2)-(3) heat balance -> protected wetted area (film-cooled length).
    # η_fc is Stechman's coolant efficiency factor, derived from the film state
    # via Knuth's wave-transition criterion (see ``film_cooling_efficiency``).
    q = jnp.maximum(h_g * jnp.maximum(T_aw - T_wall_ref, 0.0), 1.0)
    eta_fc = film_cooling_efficiency(mdot_film, grid, mission)
    A_fcl = jnp.maximum(eta_fc * mdot_film * H / q[0], 1e-9)

    # cumulative wetted area from the injection plane
    dA = 2.0 * jnp.pi * grid.r[:-1] * grid.dseg
    A_cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(dA)])

    # (4) smooth decay over the protected length (C¹ stand-in for the cut-off)
    p = mission.film_decay_exponent
    eps = mission.film_effectiveness_max / (1.0 + (A_cum / A_fcl) ** p)
    return jnp.where(ff > 0.0, eps, 0.0)


def film_slot_validity(film_frac: Array, slot_height: Array, grid: StationGrid,
                       mission: MissionSpec, *, mdot_film: Array,
                       gamma: Array, Tc: Array) -> dict:
    """Report the slot state and whether a *gaseous* correlation would apply.

    Kept as an explicit diagnostic: it is what showed that the classical
    velocity-ratio family cannot be used for a liquid RP-1 film (VR ≪ 0.45).
    """
    a_core = jnp.sqrt(gamma * mission.R_gas * Tc)
    v_core = jnp.maximum(grid.mach[0] * a_core, 1e-9)
    v_film, VR, Re_c, A_slot = film_slot_state(mdot_film, slot_height, grid,
                                               mission, v_core=v_core)
    return {"v_film": v_film, "v_core": v_core, "velocity_ratio": VR,
            "Re_film": Re_c, "slot_area": A_slot,
            "gaseous_correlation_applicable": VR >= 0.45}


def darcy_friction_factor(Re: Array, rel_roughness: float = 0.0) -> Array:
    """64/Re laminar + smooth Blasius (or rough Swamee–Jain) turbulent, blended
    across Re≈2300 with a narrow smoothstep so the residual stays C¹ (the NumPy
    original switches discretely; the blend width is far below any physical
    fidelity claim).  White, *Fluid Mechanics* §6."""
    Re = jnp.maximum(Re, 1.0)
    f_lam = 64.0 / Re
    if rel_roughness > 0.0:
        f_turb = 0.25 / (jnp.log10(rel_roughness / 3.7
                                   + 5.74 / Re ** 0.9)) ** 2
    else:
        f_turb = 0.3164 / Re ** 0.25
    w = jnp.clip((Re - 2100.0) / 400.0, 0.0, 1.0)
    s = w * w * (3.0 - 2.0 * w)  # smoothstep
    return (1.0 - s) * f_lam + s * f_turb


# --------------------------------------------------------------------------- #
# Explicit outputs of the coolant march at fixed T_wg                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoolingMarch:
    """Explicit outputs of the coolant march at fixed T_wg (all (n,) unless
    noted as scalar)."""

    T_coolant: Array          # (n,) bulk coolant temperature at each station
    q_flux: Array             # (n,) W/m² hot-side heat flux
    h_g: Array                # (n,) gas-side film coefficient
    h_c: Array                # (n,) coolant-side film coefficient (uniform here)
    T_aw: Array               # (n,) adiabatic-wall (recovery) temperature
    T_wc: Array               # (n,) coolant-side ("liquid") wall temp [K]
    area_enh: Array           # (n,) fin area-augmentation (coolant/hot area)
    coking_margin: Array      # (n,) T_coke_limit − T_wc  [K]  (≥0 feasible)
    dp_total: Array           # scalar, jacket pressure drop [Pa]
    T_coolant_exit: Array     # scalar, jacket outlet (injector-end) temp [K]
    land_min: Array           # scalar, minimum inter-channel land width [m]
    residual: Array           # (n,) stationwise T_wg residuals [K]
    # --- structural + diagnostic outputs (§10.2 / §10.4) --------------------
    sigma_thermal: Array      # (n,) constrained-expansion thermal stress [Pa]
    sigma_pressure: Array     # scalar, liner plate bending across a channel [Pa]
    coolant_mach: Array       # scalar, v_cool/a_liquid (Mirzamoghadam ≤ 0.35)
    coolant_velocity: Array   # scalar [m/s]


def cooling_march(T_wg: Array, grid: StationGrid, *, Pc: Array, gamma: Array,
                  Tc: Array, c_star_del: Array, mdot_cool: Array,
                  mission: MissionSpec, channel_width: Array | None = None,
                  channel_height: Array | None = None,
                  film_frac: Array | None = None,
                  film_slot_height: Array | None = None,
                  t_wall: Array | None = None) -> CoolingMarch:
    """March the coolant along the jacket (counterflow, upwind FV) at fixed wall
    temperatures; return the circuit fields and the stationwise residual.

    ``channel_width``/``channel_height`` override the mission channel geometry
    (they are Phase-7b design variables); ``film_frac`` diverts that fraction of
    fuel to a wall film that reduces the gas-side driving temperature over the
    chamber→throat region (§6.2b).  Everything is a smooth function of its
    inputs; sequencing is a ``lax.scan`` over the STATIC station order.
    """
    n = grid.r.shape[0]
    Rt = grid.r[grid.throat_index]
    w = mission.channel_width if channel_width is None else channel_width
    h = mission.channel_height if channel_height is None else channel_height
    ff = 0.0 if film_frac is None else film_frac
    sh = (mission.film_slot_height_default if film_slot_height is None
          else film_slot_height)
    # Design margins (§10.3; both 1.0 by default ⇒ nominal, bit-identical).
    # Mirzamoghadam's "hot channel": +10 % heat flux for injector streaking and
    # −10 % channel flow for maldistribution.
    f_q = mission.heat_flux_margin
    mdot_cool = mdot_cool * mission.channel_flow_margin
    tw = mission.t_wall if t_wall is None else t_wall

    # --- gas side (vectorised over stations) -------------------------------- #
    T_aw = jt.recovery_temperature(grid.mach, gamma, Tc, mission.Pr_gas)
    # film cooling: an adiabatic-film effectiveness (saturating in the film
    # fraction) lowers the driving temperature over the protected chamber→throat
    # region; the film is entrained/consumed in the divergent (SP-8087/Papell
    # conservative screening surrogate, plan §6.2b).
    # Liquid-film protection.  The film-cooled length follows from the
    # phase-change energy balance (see ``film_effectiveness``), so the decay is
    # physical rather than an on/off coverage mask (an earlier mask produced a
    # non-physical wall-temperature cliff at the first unfilmed station).
    # Film mass flow: the film fraction is of the FUEL flow; the jacket carries
    # (1−film_frac) of it, so mdot_film = mdot_cool·ff/(1−ff) keeps the split
    # consistent with the caller's cooling flow.
    mdot_film = mdot_cool * ff / jnp.maximum(1.0 - ff, 1e-6)
    # h_g for the film balance is evaluated at the *unfilmed* recovery state
    # (the load the film has to absorb).
    h_g_bare = f_q * jt.bartz_hg(
        grid.mach, 1.0 / grid.area_ratio, Dt=2.0 * Rt, Pc=Pc,
        c_star=c_star_del, cp=mission.cp_gas, Pr=mission.Pr_gas,
        mu=mission.mu_gas, gamma=gamma, Tc=Tc, wall_temperature=T_wg,
        throat_curvature_radius=mission.throat_rd_factor * Rt,
    )
    eta_film = film_effectiveness(ff, grid, mission, mdot_film=mdot_film,
                                  slot_height=sh, gamma=gamma, Tc=Tc,
                                  h_g=h_g_bare, T_aw=T_aw)
    T_film_in = jnp.asarray(mission.coolant_temperature, dtype=jnp.float64)
    T_aw = T_aw - eta_film * (T_aw - T_film_in)
    # streaking margin multiplies the gas-side coefficient (Mirzamoghadam's
    # "+10 % increase in chamber heat flux" for the hot-channel scenario)
    h_g = f_q * jt.bartz_hg(
        grid.mach, 1.0 / grid.area_ratio, Dt=2.0 * Rt, Pc=Pc,
        c_star=c_star_del, cp=mission.cp_gas, Pr=mission.Pr_gas,
        mu=mission.mu_gas, gamma=gamma, Tc=Tc, wall_temperature=T_wg,
        throat_curvature_radius=mission.throat_rd_factor * Rt,
    )

    # --- channel geometry (N fixed/discrete; w, h are design variables) ------ #
    N = mission.n_channels
    r_cool = grid.r + tw
    pitch = 2.0 * jnp.pi * r_cool / N
    land_raw = pitch - w
    land_min = jnp.min(land_raw)                     # channel-fit constraint
    land = jnp.maximum(land_raw, 1e-5)               # numerical floor for η_f
    A_chan = N * w * h
    Dh = 2.0 * w * h / (w + h)
    G = mdot_cool / A_chan                            # uniform mass flux
    u_cool = G / mission.rho_cool
    Re = G * Dh / mission.mu_cool

    # coolant-side h_c via the audited Sieder–Tate primitive (μ_wall = μ_bulk
    # ⇒ property ratio unity at screening fidelity); uniform along the jacket.
    h_c = jt.sieder_tate_hc(G, Dh, k=mission.k_cool, cp=mission.cp_cool,
                            mu_bulk=mission.mu_cool, mu_wall=mission.mu_cool)
    eta_f = fin_efficiency(h_c, mission.k_wall, land, h)
    area_enh = (w + 2.0 * eta_f * h) / pitch          # (n,) coolant/hot area
    R_tot = 1.0 / h_g + tw / mission.k_wall + 1.0 / (h_c * area_enh)

    # --- counterflow upwind finite-volume coolant march --------------------- #
    # Segment j (dA[j], dseg[j]) joins stations j and j+1; coolant flows from
    # the nozzle exit (station n-1) to the injector (station 0), crossing each
    # segment once.  q for a crossing is taken at the upwind node (j+1).
    dA = 2.0 * jnp.pi * grid.r[:-1] * grid.dseg       # (n-1,) segment hot areas
    T_in = jnp.asarray(mission.coolant_temperature, dtype=jnp.float64)
    heat_cap = mdot_cool * mission.cp_cool

    def step(Tc_prev, j):                             # Tc_prev = T_c[j+1]
        q_up = (T_aw[j + 1] - Tc_prev) / R_tot[j + 1]
        Tc_j = Tc_prev + q_up * dA[j] / heat_cap
        return Tc_j, Tc_j

    js = jnp.arange(n - 2, -1, -1)                    # target nodes n-2 .. 0
    _, Tc_desc = lax.scan(step, T_in, js)             # in order j = n-2 .. 0
    # place: T_c[0..n-2] = Tc_desc reversed, T_c[n-1] = inlet
    T_coolant = jnp.concatenate([Tc_desc[::-1], T_in[None]])

    # stationwise heat flux + wall-temp residual (gas-side film closure)
    q_flux = (T_aw - T_coolant) / R_tot
    residual = T_wg - (T_aw - q_flux / h_g)

    # coolant-side ("liquid") wall temperature: drop the wall conduction from
    # the gas-side wall (equivalently T_coolant + q/(h_c·area_enh)).  This is
    # the temperature the RP-1 coking limit acts on (SP-8087: liquid-wall
    # ≤ 728 K).  Exposed as a stationwise margin — never clamped (plan §9).
    T_wc = T_wg - q_flux * (tw / mission.k_wall)
    coking_margin = mission.rp1_coking_wall_temp_K - T_wc

    # --- structural (§10.2) -------------------------------------------------- #
    # Constrained-thermal-expansion stress from the through-wall gradient — the
    # BINDING criterion for a regen liner (the pressure term below is ~2 orders
    # smaller).  SP-8087; the basis of the CR-134627 / Porowski LCF screens.
    dT_wall = jnp.maximum(T_wg - T_wc, 0.0)
    sigma_thermal = (mission.liner_E * mission.liner_alpha * dT_wall
                     / (2.0 * (1.0 - mission.liner_poisson)))

    # --- jacket Δp (uniform f, u ⇒ closed form over total wall length) ------- #
    f_darcy = darcy_friction_factor(Re, mission.channel_roughness)
    L_total = jnp.sum(grid.dseg)
    dp_total = f_darcy * L_total / Dh * 0.5 * mission.rho_cool * u_cool ** 2

    # Liner plate bending across the channel span (Mirzamoghadam: the wall is
    # sized on the pressure DIFFERENTIAL across it, not chamber pressure) — the
    # jacket Δp is the differential the liner actually sees.
    sigma_pressure = jnp.maximum(dp_total, 0.0) * w * w / (2.0 * tw * tw)
    # coolant Mach diagnostic (§10.4; Mirzamoghadam limit 0.35)
    coolant_mach = u_cool / mission.coolant_sound_speed

    return CoolingMarch(
        T_coolant=T_coolant, q_flux=q_flux, h_g=h_g,
        h_c=jnp.broadcast_to(h_c, (n,)), T_aw=T_aw, T_wc=T_wc,
        area_enh=area_enh, coking_margin=coking_margin, dp_total=dp_total,
        T_coolant_exit=T_coolant[0], land_min=land_min, residual=residual,
        sigma_thermal=sigma_thermal, sigma_pressure=sigma_pressure,
        coolant_mach=coolant_mach, coolant_velocity=u_cool,
    )


# --------------------------------------------------------------------------- #
# Solvable block: root-find the stationwise wall temperatures (square Newton)  #
# --------------------------------------------------------------------------- #
def solve_cooling(grid: StationGrid, *, Pc: Array, gamma: Array, Tc: Array,
                  c_star_del: Array, mdot_cool: Array, mission: MissionSpec,
                  channel_width: Array | None = None,
                  channel_height: Array | None = None,
                  film_frac: Array | None = None,
                  film_slot_height: Array | None = None,
                  t_wall: Array | None = None,
                  rtol: float = 1e-12, atol: float = 1e-12,
                  max_steps: int = 64) -> tuple[Array, CoolingMarch]:
    """Solve R(T_wg) = 0 for the (n,) hot-gas-side wall-temperature vector.

    The differentiable seam (plan §12.1 rule 3, §4.3): the converged state is
    differentiated by Optimistix's implicit differentiation of the Newton root —
    never through the iterations.  Inner tolerances are set ≥100× tighter than
    any optimizer feasibility tolerance this block will see (plan rule 6).  The
    differentiable inputs — including the channel-geometry and film-fraction
    design variables — are passed as solver ``args`` so total derivatives flow
    through the converged wall temperatures.  Returns (T_wg*, CoolingMarch).
    """
    w = mission.channel_width if channel_width is None else channel_width
    h = mission.channel_height if channel_height is None else channel_height
    ff = 0.0 if film_frac is None else film_frac
    sh = (mission.film_slot_height_default if film_slot_height is None
          else film_slot_height)
    tw = mission.t_wall if t_wall is None else t_wall
    T_aw0 = jt.recovery_temperature(grid.mach, gamma, Tc, mission.Pr_gas)
    T_wg_init = 0.35 * T_aw0            # cool-wall seed (regen wall ≪ T_aw)

    def fn(T_wg, args):
        Pc_, gamma_, Tc_, cstar_, mdot_, w_, h_, ff_, sh_, tw_ = args
        return cooling_march(T_wg, grid, Pc=Pc_, gamma=gamma_, Tc=Tc_,
                             c_star_del=cstar_, mdot_cool=mdot_, mission=mission,
                             channel_width=w_, channel_height=h_,
                             film_frac=ff_, film_slot_height=sh_,
                             t_wall=tw_).residual

    args = (Pc, gamma, Tc, c_star_del, mdot_cool, w, h, ff, sh, tw)
    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(fn, solver, T_wg_init, args=args,
                         max_steps=max_steps, throw=False)
    T_wg = sol.value
    march = cooling_march(T_wg, grid, Pc=Pc, gamma=gamma, Tc=Tc,
                          c_star_del=c_star_del, mdot_cool=mdot_cool,
                          mission=mission, channel_width=w, channel_height=h,
                          film_frac=ff, film_slot_height=sh, t_wall=tw)
    return T_wg, march
