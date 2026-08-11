"""
raosim.mdo.schema — design variables, bounds, and mission definition (Phase 1).

The MDF formulation (plan §4.1) separates:

* ``MissionSpec`` — fixed requirements and architecture constants for one
  operating point: target thrust, ambient pressure, burn time, tank states,
  efficiency/density constants for the electric feed.  Frozen dataclass of
  floats; a static argument to jitted functions.
* ``DesignVector`` — the continuous design variables ``x_h`` (skeleton subset
  of the plan's full list): ``Pc``, ``eps``, ``dp_f_frac``, ``dp_o_frac``.
  Registered as a JAX pytree so ``jit`` / ``jacfwd`` / ``jacrev`` flow through
  it with no host callbacks (the Phase-1 gate, tests/test_mdo_schema.py).
* ``VariableSpec`` / ``default_design_space`` — per-variable bounds used by
  ``scaling.ScaledSpace`` to map to the unit box.

Discrete architecture choices (channel counts, bus voltage class, movable vs
fixed pintle) are *not* variables here — plan rule 4: outer enumeration only.

Constant defaults below are screening values mirroring the NumPy models they
will be replaced by (``raosim.pumps`` drive/battery specs, Lee 2021 for the
battery energy/power densities); each is a placeholder in the §3 mass-ledger
sense: explicit, visible, and swapped for the audited model as the discipline
blocks deepen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterator, Mapping
import warnings

import jax
import jax.numpy as jnp

Array = jnp.ndarray


# Huzel & Huang, *Design of Liquid Propellant Rocket Engines* (1967),
# ``Optimum Mixture Ratio`` (source-PDF p. 29) and ``Mixture Ratio``
# (source-PDF p. 44), NASA SP-125, define the physical quantity used here as
# oxidizer weight flow divided by fuel weight flow.  The layout classes below
# are software-contract machinery around that physical scalar; a layout flag
# must never be encoded in the scalar itself.
_DESIGN_BASE_FIELDS = (
    "Pc", "eps", "dp_f_frac", "dp_o_frac", "D_pintle", "N_rpm",
    "channel_width", "channel_height", "film_frac", "t_wall",
)
_DESIGN_CONTRACT_FIELDS = _DESIGN_BASE_FIELDS + ("OF",)


def validate_mixture_ratio(value: object, *, name: str = "O/F") -> float:
    """Return a finite, strictly positive host-side mixture ratio.

    The differentiable core performs the equivalent check with JAX arrays and
    carries a failed value into its numerical-validity gate.  Host adapters use
    this helper so invalid O/F values fail before a fuel/oxidizer split or a
    traditional-analysis handoff is attempted.
    """

    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {resolved!r}")
    return resolved


@dataclass(frozen=True)
class DesignLayout:
    """Static description of the active optimizer vector.

    The optimizer owns either the ten hardware variables or those ten plus
    O/F.  Serialized/state contracts always use :attr:`contract_names`, which
    has eleven physical values and never contains an intent sentinel.
    """

    active_names: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed = (_DESIGN_BASE_FIELDS, _DESIGN_CONTRACT_FIELDS)
        if self.active_names not in allowed:
            raise ValueError(
                "DesignLayout active_names must be the ordered 10-variable "
                "hardware layout or that layout followed by OF"
            )

    @classmethod
    def fixed_of(cls) -> "DesignLayout":
        return cls(_DESIGN_BASE_FIELDS)

    @classmethod
    def variable_of(cls) -> "DesignLayout":
        return cls(_DESIGN_CONTRACT_FIELDS)

    @classmethod
    def for_mission(cls, mission: "MissionSpec") -> "DesignLayout":
        return (
            cls.variable_of()
            if mission.cea_table_path and not mission.of_is_pinned
            else cls.fixed_of()
        )

    @property
    def contract_names(self) -> tuple[str, ...]:
        return _DESIGN_CONTRACT_FIELDS

    @property
    def of_is_variable(self) -> bool:
        return self.active_names == _DESIGN_CONTRACT_FIELDS

    @property
    def active_size(self) -> int:
        return len(self.active_names)

    def resolve_of(self, design_of: Array, mission_of: float) -> Array:
        """Resolve the effective O/F using this explicit static layout."""

        value = design_of if self.of_is_variable else mission_of
        return jnp.asarray(value, dtype=jnp.float64)


# --------------------------------------------------------------------------- #
# Mission / architecture constants (static under jit)                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MissionSpec:
    """One operating point + architecture constants for the skeleton engine.

    All SI.  Efficiency and specific-mass constants are labeled screening
    values (see module docstring); ``regen_dp_allowance`` and
    ``line_dp_allowance`` are the §3 ledger placeholders for the cooling and
    feed-line pressure losses until Phases 4a/5 supply them from physics.
    """

    # Requirement
    thrust: float = 13.0e3          # N
    #: which combination the constants describe (set by ``for_propellant``)
    propellant_name: str = "LOX/RP-1"
    #: which alloys the traced wall constants describe (set by
    #: ``for_material``/``with_materials``).  ``None`` means the class
    #: defaults are in force and no catalog record backs them -- reported so a
    #: snapshot cannot imply a material was selected when none was.
    liner_material_name: str | None = None
    closeout_material_name: str | None = None
    Pa: float = 101325.0            # Pa ambient
    burn_time: float = 120.0        # s

    # Propellant / combustion constants (Phase-2 property surfaces replace
    # gamma/R_gas/Tc/c_star_ideal as functions of (Pc, OF); constants here
    # keep the skeleton runnable without RocketCEA).
    OF: float = 2.27                # oxidizer/fuel mass ratio
    gamma: float = 1.24
    R_gas: float = 379.6557077625571  # J/(kg K), Mw=21.9 g/mol
    Tc: float = 3571.0              # K
    eta_cstar: float = 0.975        # delivered = eta * ideal (plan rule 2)
    # Nozzle losses act on the ideal thrust coefficient, independently of the
    # combustion/c* efficiency above.  Thus Isp = eta_CF * Cf_ideal *
    # (eta_cstar * c*_ideal) / g0; do not fold this into eta_cstar.
    eta_CF: float = 0.985
    # Exit pressure must exceed the Schmucker separation-pressure estimate by
    # this factor.  SP-8120 p.82 recommends reducing expansion/selecting a
    # non-separating contour when the exit is within about 20% of separation.
    separation_design_margin: float = 1.20
    # Phase-2: path to a saved CEA property table (.npz) produced host-side by
    # ``scripts/sample_cea_surface.py``.  When set, γ/T_c/R (and hence c*) come
    # from C¹ surfaces over (P_c, O/F) and **O/F becomes a real design lever**
    # (flame temperature is a second coking lever).  When empty, the constants
    # above are used as flat surfaces — correct at the stated O/F, flat in O/F.
    cea_table_path: str = ""
    #: Feed architecture this mission describes.  Selects the chamber-pressure
    #: window in :mod:`raosim.mdo.bounds` -- Parsley & Zhang (2004) give a
    #: different pressure regime and limiting mechanism for each cycle.  Only
    #: ``electric_pump`` is implemented in the traced core;
    #: :func:`raosim.requirements.resolve_requirement` refuses the others.
    feed_architecture: str = "electric_pump"
    #: Optional user-selected numerical search window.  These values condition
    #: the optimizer only; they are not physical-validity claims.  When absent,
    #: :mod:`raosim.mdo.bounds` supplies its evidence-labelled recommended
    #: window.  Property-table domains remain separate hard constraints.
    chamber_pressure_search_min_pa: float | None = None
    chamber_pressure_search_max_pa: float | None = None
    #: Pin O/F at :attr:`OF` even when surfaces are loaded.  Set by
    #: :func:`raosim.requirements.resolve_requirement` when the user states a
    #: mixture ratio explicitly.  Without this a pinned requirement would be
    #: silently overridden the moment a CEA table appeared, which is the
    #: opposite of what "pinned" means.
    of_is_pinned: bool = False
    # OPTIONAL spray→c* feedback edge (Phase 7, default OFF).  Plan §5 flags this
    # as the strongest coupling but the *weakest physics* — a one-way correlation,
    # not energy-closed — and asks for an ablation (η_cstar frozen) to bound how
    # much Pareto gain rests on it.  So the engine defaults to the frozen
    # ``eta_cstar`` above; ``couple_eta_cstar=True`` swaps in this screening
    # surrogate, peaked at TMR_opt (Sakaki 2015: TMR "usually set around 1";
    # Sakaki 2016/2017 + Hwang s42405 show c*-eff varies with TMR, direction
    # config-dependent).  Documented as a screening knob, not validated physics.
    eta_cstar_max: float = 0.96          # peak c*-eff for the coupled surrogate
    eta_cstar_tmr_opt: float = 1.0       # TMR at peak (Sakaki design target)
    eta_cstar_tmr_width: float = 2.5     # log-TMR Gaussian width (mild)

    # Gas transport at the throat (Bartz inputs; Phase-2 surfaces later)
    cp_gas: float = 2000.0          # J/(kg K)
    Pr_gas: float = 0.55
    mu_gas: float = 9.0e-5          # Pa s

    # Throat thermal screen constants (used by the "throat0" fidelity; the
    # 4a march computes h_c and the coolant state itself)
    # Repository/traditional TOP convention: Ru/Rt=1.5 and Rd/Rt≈0.382.
    # They remain separate because Ru is the convergent/upstream fillet while
    # Rd starts the divergent arc.  (Rao's 1958 construction used Rd/Rt=0.45.)
    throat_ru_factor: float = 1.5   # upstream curvature / Rt
    throat_rd_factor: float = 0.382 # downstream curvature / Rt
    h_c: float = 8.0e4              # W/(m2 K) coolant-side (throat0 fidelity)
    coolant_temperature: float = 320.0  # K (throat0) / jacket inlet T (4a)
    t_wall: float = 8.0e-4          # m hot-gas wall
    k_wall: float = 320.0           # W/(m K)
    # RP-1 coking limit on the LIQUID-SIDE (coolant-side) wall temperature.
    # NASA SP-8087 recommended practice: "do not operate with liquid-wall
    # temperatures above 850°F (728 K) for RP-1" (coking onset 800–900°F,
    # 700–756 K; ref. Sellers, ARS J. 31(5), 1961).  Applied as a stationwise
    # inequality on T_wc; the design owns the margin, not a clamp (plan §9).
    rp1_coking_wall_temp_K: float = 728.0
    # --- film cooling (§6.2b; the coking lever) -------------------------------
    #   A fuel fraction ``film_frac`` injected at the wall reduces the gas-side
    #   driving temperature over the chamber→throat region.  SP-8087/SP-125:
    #   combined regen+film is standard at high Pc; SP-8087 states the film/
    #   combustion interaction cannot be modelled precisely, so a *conservative
    #   simple analytical model* is the accepted practice (Hatch & Papell
    #   correlation, ref. 21).  This is that conservative screening surrogate:
    #   an adiabatic-film effectiveness that saturates with the film fraction,
    #   T_aw,eff = T_aw − η_film·(T_aw − T_film_in), applied for stations up to
    #   the throat (film is entrained/consumed downstream).  Film fuel burns
    #   fuel-rich at the wall → a delivered-c* penalty (rule-of-thumb ≈ ½ % Isp
    #   per 1 % film).
    #   VALIDATED CORRELATION (2026-07-24, replaces the earlier surrogate).
    #   Adiabatic film-cooling effectiveness ε = (T_ad−T_g)/(T_c−T_g) from the
    #   classical tangential-slot family reviewed by Shine & Nidhi (2018),
    #   "Review on film cooling of liquid rocket engines", Propulsion and Power
    #   Research, Eqs. (1)–(5):
    #        ε = C · (X/VR)^(−0.8) · Re_c^(0.2)
    #   with X = x/D the non-dimensional distance from injection, VR the
    #   coolant/mainstream velocity ratio and Re_c the coolant Reynolds number.
    #   Published coefficients: Stollery & El-Ehwany C = 3.09, Hartnett C = 3.39,
    #   Tribus & Klein C = 4.62 (the spread reflects mixing assumptions — "the
    #   more the mixing, the lower the coefficient"), so the conservative
    #   Stollery value is the default.  Consistent with Hatch & Papell,
    #   NASA TN D-130 (1959) (±5 % for ε in 0.2–1.0, velocity ratio 0.45–33.3),
    #   which SP-8087 cites as the film-cooling reference.
    #   NOTE (2026-07-24): the classical ε = C(X/VR)^(−0.8)Re^(0.2) family is
    #   fitted to GASEOUS coolants (VR 0.45–33.3).  An RP-1 film is liquid, and
    #   continuity through any sane annular slot gives VR ≈ 1e−3 — three orders
    #   below that band — so the liquid model below is used instead and the
    #   velocity ratio is reported as a diagnostic (``film_slot_validity``).
    film_effectiveness_coeff: float = 3.09     # (gaseous family; diagnostic only)
    film_effectiveness_max: float = 1.0        # ε is a fraction by definition
    #   LIQUID film model (Shine & Nidhi 2018 §4.3 — Kinney/Graham/Sellers/
    #   Emmons phase-change energy balance; Huzel & Huang SP-125 Eq. 4-34):
    #   the film absorbs sensible + latent + vapour-superheat enthalpy and so
    #   protects a finite film-cooled length.
    film_latent_heat: float = 2.5e5        # J/kg, RP-1 heat of vaporisation
    cp_cool_vapor: float = 2.5e3           # J/(kg K), vapour-phase cp
    #   η_fc — the "dimensionless liquid coolant efficiency factor" (Stechman,
    #   Oberstone & Howell, AIAA J. Spacecraft 6(2), 1969, Eq. 2 and Fig. 2):
    #   the liquid film is "not completely ideal" because of film instability, so
    #   the coolant flow is corrected by an efficiency factor **that is a
    #   function of coolant Reynolds number** Re_L = W_L/(π D μ_L).
    #
    #   The mechanism is quantified by Grisson (AEDC-TR-91-1, 1991 §2.1): below
    #   Knuth's critical flow-per-circumference Γ_cr = 1.01e5·μ_v²/μ_ℓ the film
    #   is smooth and essentially all the coolant is used (η → η_max); above it
    #   large waves shear droplets off the crests and "the mass loss rate ... is
    #   **2 to 4 times the normal evaporation rate**", i.e. η falls to ≈1/2–1/4.
    #   (Woodmansee & Hanratty measured Γ_cr ≈ 3× lower than Knuth for water, so
    #   the transition location itself carries ~3× uncertainty.)
    #
    #   Implemented as a smooth (C¹) transition between those two literature
    #   limits, so η_fc is *derived from the local film state* rather than being
    #   a fitted constant.  Validation context: Shine & Nidhi report Stechman's
    #   model errs −20 % … +13 % depending on chamber configuration.
    film_eta_smooth: float = 0.9       # η below Knuth's critical flow (stable)
    film_eta_wavy: float = 0.33        # η above it (mass loss 2–4× ⇒ ~1/2–1/4)
    film_knuth_coeff: float = 1.01e5   # Γ_cr = coeff·μ_v²/μ_ℓ  (Knuth)
    film_mu_vapor: float = 1.0e-5      # Pa s, RP-1 vapour viscosity
    film_transition_width: float = 0.6 # log-Γ width of the C¹ blend
    film_decay_exponent: float = 2.0       # C¹ decay sharpness over the FCL
    #   Screening scales mapping the film *fraction* onto the correlation's
    #   VR (velocity ratio) and Re_c.  Chosen so ε spans ≈0.09–0.93 over film
    #   fractions 2–30 % at mid-chamber — i.e. the correlation is exercised
    #   inside its **validated** band (Hatch–Papell: ±5 % for ε ∈ [0.2, 1.0];
    #   the slot family is fitted over VR 0.45–33.3).  At a 10 % film this is
    #   VR ≈ 0.05 and Re_c ≈ 1e3 for the tangential slot.  These scales stand in
    #   for the film-injector slot sizing; a slot-height design variable would
    #   replace them with geometry.
    #   Default tangential-slot height when the caller does not pass the design
    #   variable (Hatch & Papell tested 0.063–0.50 in = 1.6–12.7 mm).
    film_slot_height_default: float = 2.0e-3   # m
    #   Specific-impulse / c* penalty: Coulbert's analysis (via Shine §4.5) finds
    #   the loss is *proportional to the quantity of coolant flow*; Morrell's
    #   experiments give 4 % Isp loss at 5 % water film (ratio 0.8), 4 % at 15 %
    #   alcohol (0.27) and 2 % at 15 % ammonia (0.13).  0.5 sits inside that
    #   measured band for a hydrocarbon film and keeps the linear (Coulbert) form.
    film_cstar_penalty: float = 0.5   # Δη_c* per unit film_frac (Coulbert-linear)
    #   Coverage: the whole subsonic run (chamber→throat) plus the early
    #   divergent up to this area ratio — film injected upstream is entrained
    #   through the throat and protects the initial expansion before it is
    #   consumed; the far, low-flux, cold-coolant divergent is not the limiter.
    film_coverage_ar_limit: float = 2.5
    # Coolant-channel aspect-ratio (height/width) validity cap.  The HARCC
    # models are validated to AR ≈ 8 (Pizzarelli 2011; Carlile 1992 tested 8 and
    # 15) and Mirzamoghadam 1991 (AIAA-91-1982) caps manufacturable depth/width
    # at 7.  Enforced so the optimiser cannot exploit un-validated AR.
    channel_aspect_ratio_max: float = 8.0

    # --- design margins (§10.3; thermal/flow default 1.0; film 2x) -----------
    #   Real chamber-cooling practice does not size on nominal conditions.
    #   Mirzamoghadam 1991 (AIAA-91-1982) sizes a "hot channel" worst case:
    #   injector **streaking** is accounted for by "a 10% increase in chamber
    #   heat flux", and flow **maldistribution** (inlet manifold + manufacturing
    #   tolerance + streaking) "by reducing the actual flow through the channel
    #   by 10%".  SP-8087 further *requires* (shall-language, p. 6082) that
    #   "film-cooling flow shall be capable of providing **twice** the estimated
    #   required quantity", with the injector "flexible enough to provide for
    #   additions of up to 100 percent more than the original estimate".
    #
    #   Thermal/flow defaults are 1.0 so nominal results remain reproducible.
    #   The film-system capacity reserve is 2x by default because SP-8087
    #   requires that installed capability, not merely a nominal condition.
    heat_flux_margin: float = 1.0        # Mirzamoghadam hot-channel: 1.10
    channel_flow_margin: float = 1.0     # Mirzamoghadam maldistribution: 0.90
    film_capacity_margin: float = 2.0    # SP-8087 film system capacity reserve
    # Installed wall-film circuit capacity as a fraction of total fuel flow.
    # Default 0.60 supports a 30% design film flow with SP-8087's 2x reserve;
    # it is an explicit architecture/sizing assumption, not a claim that the
    # system may route all fuel through the film manifold.
    film_system_capacity_fraction: float = 0.60

    # --- liner structural properties (§10.2) ---------------------------------
    #   The binding wall criterion for a regen liner is the THERMAL gradient,
    #   not pressure: at the current optimum the pressure bending across the
    #   channel is ~0.2 MPa while the thermal stress is ~27 MPa.  The classical
    #   constrained-thermal-expansion stress for a plate through-wall gradient is
    #       σ_th = E·α·ΔT_wall / (2(1−ν))
    #   (SP-8087; the same form the low-cycle-fatigue screens of NASA CR-134627
    #   and Porowski 1985 are built on — those are the §8 constraint-layer
    #   deepening).  Defaults are a copper-alloy liner (GRCop/NARloy class).
    liner_E: float = 110.0e9          # Pa, Young's modulus
    liner_alpha: float = 17.0e-6      # 1/K, thermal expansion
    liner_poisson: float = 0.33
    # ``liner_sigma_allow`` is the post-factor-of-safety allowable used by the
    # MDO inequality.  The traditional solver stores the corresponding material
    # yield strength and divides it by ``liner_structural_fos``.  Keeping both
    # values explicit prevents the former double-application of a hidden 1.5 FOS
    # during post-optimum parity checks.
    liner_sigma_allow: float = 200.0e6  # Pa, post-FOS allowable at temperature
    liner_structural_fos: float = 1.5
    #   Allowable GAS-SIDE wall temperature.  Mirzamoghadam names "the allowable
    #   design gas-side wall temperature ... based on sensitivity of the material
    #   yield strength" as a primary criterion and reports 900°F (755 K) average
    #   wall temperature as compatible for a copper H2/O2 chamber.  Needed once
    #   t_wall is a variable: a thicker wall lowers T_wc (helping coking) while
    #   RAISING T_wg, so without this the optimiser could trade the liner away.
    liner_T_wg_max: float = 800.0     # K, copper-alloy class
    liner_pressure_stress_report: bool = True  # report the (small) Δp bending
    #   Liquid RP-1 speed of sound, for the coolant-Mach diagnostic
    #   (Mirzamoghadam limits coolant Mach ≤ 0.35 to avoid sonic choking at
    #   bends; measured ≈0.002 here, i.e. ~173× margin — reported, not
    #   constrained, so it does not add a dead Jacobian column).
    coolant_sound_speed: float = 1300.0   # m/s

    # --- 4a station-grid geometry (analytic bulk-sweep contour, mdo/grid.py) --
    contraction_ratio: float = 8.0        # A_c/A_t
    #   Characteristic length L* = V_c/A_t (SP-125 §4).  Table 4-1 gives
    #   **40–50 in for LOX/RP-1**; 43 in ≈ 1.09 m is used.
    #
    #   NOTE (2026-07-31): ``chamber_length`` used to sit here as a prescribed
    #   barrel length ``= L*/CR``.  That is wrong.  SP-125 (printed p. 88) is
    #   explicit that "the combustion chamber volume includes the space between
    #   injector face I-I and the nozzle throat plane II-II", so the convergent
    #   section carries part of L*·A_t and the barrel must be SHORTER than
    #   L*/CR.  The barrel length is now solved from the volume closure by
    #   :func:`raosim.mdo.grid.chamber_barrel_length`, exactly as the
    #   traditional :func:`raosim.chamber_geometry.chamber_contour` does, so the
    #   two pipelines describe one chamber.  It is no longer an input.
    l_star: float = 1.09                  # m (43 in), LOX/RP-1 (SP-125 Tab 4-1)
    #   Fraction of the geometrically feasible shoulder fillet that is used,
    #   matching raosim.chamber_geometry.auto_shoulder_factor's default so both
    #   pipelines build the same contraction.  0.8 keeps a ~20% straight cone.
    shoulder_fill_fraction: float = 0.8
    #   Channel pitch target — the cooling channel COUNT scales with the
    #   chamber circumference so the pitch (and hence the land width) stays
    #   manufacturable as the engine grows.
    channel_pitch_ref: float = 2.87e-3    # m (13 kN baseline: 551 mm / 192)
    converging_half_angle_deg: float = 30.0
    length_pct: float = 80.0              # % of 15° cone (SP-8120 convention)

    # --- 4a regenerative jacket (fixed channel geometry; counts are DISCRETE
    #     architecture — outer enumeration only, plan rule 4).  Defaults echo
    #     the 13 kN baseline (194 x 0.50 mm channels; TEST_PROMPT.md). ---------
    n_channels: int = 192                 # discrete (static)
    channel_width: float = 5.0e-4         # m
    channel_height: float = 1.5e-3        # m
    # Defined MDO architecture: a wall-film branch bypasses the regenerative
    # jacket; this fraction is applied to the remaining fuel after the film
    # split.  Other plumbing topologies require a separate model, not a switch.
    cooling_fraction: float = 1.0
    # Coolant transport constants (RP-1-class screening values; Phase-2
    # CoolProp surfaces over (T, p) replace these)
    rho_cool: float = 810.0               # kg/m3
    cp_cool: float = 2093.0               # J/(kg K)
    k_cool: float = 0.11                  # W/(m K)
    mu_cool: float = 1.0e-3               # Pa s
    channel_roughness: float = 0.0        # relative (smooth -> Blasius branch)

    # --- thrust-chamber hardware mass (raosim.mass_ledger, differentiable
    #     mirror in mdo/mass.py) -------------------------------------------- #
    #   Mass is integrated geometrically as a solid of revolution (Pappus) over
    #   the same station grid the cooling model marches.  NASA SP-125 eq. 8-32
    #   uses the same mid-surface-area x thickness x density relation for a
    #   cylindrical propellant TANK; it corroborates this shell-volume geometry
    #   but does not prescribe a thrust-chamber mass model or wall thickness.
    #   ``rho_wall`` must match the liner alloy behind ``k_wall`` --
    #   the default pair (320 W/m-K, 9130 kg/m3) is the NARloy-Z / CuCrZr-class
    #   liner the MDO baseline uses (raosim.materials).
    rho_wall: float = 9130.0              # kg/m3 liner density
    #   Structural closeout over the channels.  SP-8087 sec. 2.1.3 gives the
    #   jacket three jobs -- chamber hoop support, throat bend/buckling support,
    #   and nozzle hoop-compression collapse resistance under sea-level
    #   overexpansion -- and sec. 2.1.3 quotes design factors of safety of
    #   1.0-1.32 (yield) and 1.3-1.8 (ultimate).  This repository does not yet
    #   size the jacket against those loads, so the closeout is carried as an
    #   explicit ratio of the hot-gas wall and reported as an assumption, not
    #   as a solved thickness.  The traditional path uses the same convention
    #   via RegenWallProfile.uniform's ``t_jacket``.
    #   --- structural closeout (jacket) -------------------------------------
    #   The jacket is now SIZED, not assumed.  SP-125 (printed p. 109): "the
    #   outer shell is subjected only to the hoop stress induced by the coolant
    #   pressure", so the thin-shell requirement at each station is
    #
    #       t_j(x) = FoS * p_coolant(x) * r_outer(x) / sigma_yield_closeout
    #
    #   floored at a manufacturing minimum.  SP-8087 sec. 2.1.3.1 confirms a
    #   tapered jacket is normal practice -- "The brazed jacket can be tapered
    #   for optimum strength and weight" -- so the thickness follows the local
    #   pressure and radius rather than being uniform.
    #
    #   ``closeout_thickness_ratio`` is retained ONLY as the legacy fallback
    #   used when ``closeout_sizing`` is "ratio"; the default is "hoop".
    closeout_sizing: str = "hoop"          # "hoop" | "ratio"
    closeout_thickness_ratio: float = 2.0  # t_closeout / t_wall [-] (legacy)
    closeout_thickness_min: float = 5.0e-4  # m manufacturing floor
    #   Jacket material.  SP-8087 sec. 2.1.3.1: "Hardenable materials often are
    #   used for jacket designs, where, after brazing, the strength can be
    #   increased considerably by agehardening" -- i.e. a soft
    #   high-conductivity liner inside a strong jacket, NOT one alloy for both.
    #   Defaults are Inconel 718 (raosim.materials), the repository's standard
    #   jacket/structure entry.  A copper jacket at these pressures needs a
    #   thickness that violates the thin-shell assumption; that is what the
    #   ``jacket_thin_shell_margin`` constraint reports.
    rho_closeout: float | None = 8190.0   # kg/m3 (None reuses rho_wall)
    closeout_sigma_yield: float = 1035.0e6  # Pa, Inconel 718 room-temp yield
    #   SP-8087 sec. 2.1.3 quotes yield factors of safety of 1.0-1.32 and
    #   ultimate 1.3-1.8.  The conservative end of the yield band is used.
    closeout_structural_fos: float = 1.32
    closeout_E: float = 200.0e9           # Pa, Inconel 718 Young's modulus
    closeout_poisson: float = 0.29
    #   --- nozzle collapse under sea-level overexpansion --------------------
    #   SP-8087 sec. 2.1.3 lists this as one of the three structural jobs of
    #   chamber reinforcement: "hoop support about the expansion nozzle to
    #   resist collapse from hoop compression.  The last condition occurs
    #   during operation at sea level, where jet separation occurs during start
    #   and shutdown and the nozzle runs overexpanded during steady-state
    #   operation."  SP-8120 sec. 2.2 records the consequence: "A typical
    #   failure of this kind is the collapse of the nozzle from overexpansion
    #   during ground testing".
    #   The screen uses NASA SP-8007 (rev. Aug 1968) sec. 4.2.3 with its
    #   recommended correlation factors: gamma = 0.75 for the moderate-length
    #   branch (eq. 16/20) and gamma = 0.90 for the long-cylinder oval mode
    #   (eq. 19/21).
    nozzle_collapse_fos: float = 1.5
    nozzle_collapse_gamma_moderate: float = 0.75   # SP-8007 eq. (20)
    nozzle_collapse_gamma_long: float = 0.90       # SP-8007 eq. (21)
    #   SP-125 (printed p. 336) states the membrane/thin-shell treatment holds
    #   while "the wall thickness of a pressure vessel is small compared to the
    #   radii of wall curvature (t/r <= 1/15)".  Beyond that the hoop formula
    #   above is no longer the right model, so it is a hard admissibility
    #   constraint rather than a warning.
    closeout_thin_shell_ratio_max: float = 1.0 / 15.0
    #   Land (rib) width between channels.  ``None`` uses the geometric rib
    #   pitch - w at the channel mid-radius, matching
    #   raosim.regen_profile.RegenWallProfile.uniform.
    land_width: float | None = None       # m

    # --- Requirement limits (SP-125 §2.1 items 5 and 6) ----------------------
    #   These are *requirements*, not physics: SP-125 (printed p. 31) lists
    #   "weight of engine system at burnout" and "envelope (size)" among the
    #   nine parameters an engine specification must cover.  They live on the
    #   MissionSpec so the traced core can screen them, but they are populated
    #   from a :class:`raosim.requirements.EngineRequirement`, never guessed.
    #
    #   Defaults are large FINITE sentinels rather than ``inf``.  An infinite
    #   limit would give an infinite margin, and while ``d(inf - x)/dx`` is
    #   well defined, an inf entry in the scaled constraint vector is not
    #   something SLSQP handles gracefully.  Large-and-finite keeps the row in
    #   the Jacobian, keeps it inert, and keeps every number printable.
    #
    #   All three screens are LOWER-BOUND screens on the true installed
    #   quantity -- see raosim.mdo.envelope for what the envelope excludes and
    #   the ``dry_mass_partial`` scalar in mdo/engine.py for what the mass
    #   excludes.  :class:`raosim.requirements.RequirementCoverage` records
    #   that, so a partially screened requirement can never be reported as
    #   satisfied outright.
    envelope_diameter_max: float = 1.0e3   # m (inert sentinel)
    envelope_length_max: float = 1.0e3     # m (inert sentinel)
    dry_mass_max: float = 1.0e9            # kg (inert sentinel)

    # --- Phase-5 pintle injector architecture (mdo/injector.py) --------------
    #   radial stream = FUEL through N slots, axial stream = OXIDIZER annulus
    #   (repo convention: PintleGeometrySpec.radial_stream="fuel").  Discrete
    #   counts are outer-enumeration architecture (plan rule 4).  Movable-pintle
    #   geometry (Son 2017) sizes the tip-opening branch; the fixed center-gap
    #   annulus caps it (the two branches of plan rule 8 / §6.3).
    injector_cd_fuel: float = 0.75        # radial-slot discharge coefficient
    injector_cd_ox: float = 0.75          # axial-annulus discharge coefficient
    pintle_diameter: float = 0.020        # m, D_pintle (BF anchor + post radius)
    pintle_post_thickness: float = 1.5e-3  # m, wall of the slotted post
    pintle_tip_angle_deg: float = 0.0     # Son Eq.(1) tip half-angle (0 = flat)
    pintle_slot_count: int = 24           # discrete (static) — radial openings
    pintle_slot_aspect_ratio: float = 1.0  # h_slot / w_slot
    pintle_deflector_angle_deg: float = 0.0  # radial-stream deflection
    pintle_center_gap_diameter: float = 0.018   # m, Dcg (center-gap outer)
    pintle_rod_diameter: float = 0.010          # m, Dpr (center rod)
    pintle_transition_area_fraction: float = 0.95  # stay below the Son cap
    # Conservative injector-drop design screen.  The 0.20 endpoint is the
    # 15--20% rule of thumb in Huzel & Huang, NASA SP-125 (1967), sec. 4.2,
    # source-PDF p.137 / printed p.128.  SP-194 (1972), sec. 6.2.3.1,
    # source-PDF pp.293--294 supports the stability coupling but cautions that
    # changing only one stream can destabilize; this is not a universal chug
    # boundary or a substitute for a coupled stability analysis.
    injector_dp_stability_min: float = 0.20
    #   Blockage factor BF = N·w/(π D_p) — with TMR, one of "the two master
    #   geometric knobs" of a pintle injector (Hwang et al., IJASS 23, 2022;
    #   Freeberg 2019 makes the spray angle a function of BF and TMR).  Real
    #   designs sit in a band: Ryu et al. swept BF 70 → 85 % (c*-efficiency
    #   +3.85 %), Kang 2022 used 0.6.  Constrained so the optimiser cannot park
    #   D_pintle at a value that no longer describes a pintle element.
    blockage_factor_min: float = 0.30
    blockage_factor_max: float = 0.90

    # Feed / tanks (per-stream; §3 ledger placeholders explicit)
    rho_fuel: float = 810.0         # kg/m3 (RP-1)
    rho_ox: float = 1141.0          # kg/m3 (LOX)
    P_tank_fuel: float = 4.0e5      # Pa
    P_tank_ox: float = 4.0e5        # Pa
    regen_dp_allowance: float = 0.0  # Pa, fuel side (placeholder until 4a)
    line_dp_allowance: float = 0.0   # Pa (placeholder)

    # --- Phase-6 pump / electric feed (mdo/pump.py) --------------------------
    #   C¹ efficiency-vs-specific-speed surrogate REPLACING the binned
    #   pumps.py:_estimate_pump_efficiency (which is C0-discontinuous and cannot
    #   live in the differentiable core — plan §6.4).  Smooth log-Gaussian in
    #   the dimensionless specific speed Ns = ω√Q/(g0 H)^¾, peaked at Ns_opt,
    #   calibrated to the SP-125 rocket-pump band (Huzel & Huang: "overall
    #   efficiency … ranges from 60 to 85 percent", ~10% below industrial;
    #   rises with capacity/Ns) with the SP-8109 low-Ns penalty.  Documented as
    #   a smooth screening surrogate — NOT a fit to SP-125 Fig. 6-23 ordinates
    #   (image-only, not text-extractable).  Ns is a continuous design/derived
    #   variable; pump speed N_rpm is the plan §4.1 continuous variable.
    pump_eta_peak: float = 0.82           # peak hydraulic efficiency (SP-125 band)
    pump_ns_opt: float = 0.55             # dimensionless Ns at peak
    pump_eta_ns_width: float = 1.5        # log-Ns Gaussian width
    pump_speed_rpm: float = 30000.0       # N (electric pumps run fast; Rutherford-class)
    pump_head_coefficient: float = 0.50   # ψ = g0 H / U2² (tip-speed sizing)
    pump_tip_speed_max: float = 400.0     # m/s, stress-limited screen (SP-8109)
    pump_nss_max: float = 4.0             # dimensionless suction-Ns cap (SP-8052)
    p_vapor_fuel: float = 3.0e3           # Pa, RP-1 vapour pressure (~300 K)
    p_vapor_ox: float = 1.0e5             # Pa, LOX vapour pressure (near-satur.)

    # Electric drive / battery screening constants
    eta_pump: float = 0.60
    eta_motor: float = 0.90
    eta_inverter: float = 0.95
    eta_discharge: float = 0.90
    motor_power_density: float = 3.0e3     # W/kg shaft
    inverter_power_density: float = 20.0e3  # W/kg electric
    pump_specific_mass: float = 2.5e3       # W/kg hydraulic (screening)
    battery_energy_density: float = 6.5e5   # J/kg  (~180 Wh/kg, Lee-2021 class)
    battery_power_density: float = 1.3e3    # W/kg
    battery_structural_margin: float = 1.2

    g0: float = 9.80665

    def __post_init__(self) -> None:
        validate_mixture_ratio(self.OF, name="MissionSpec.OF")
        search_overrides = (
            self.chamber_pressure_search_min_pa,
            self.chamber_pressure_search_max_pa,
        )
        if any(value is not None for value in search_overrides):
            from raosim.mdo.bounds import chamber_pressure_search_window

            recommended = chamber_pressure_search_window(
                self.feed_architecture
            )
            lo = (
                recommended.lower
                if self.chamber_pressure_search_min_pa is None
                else float(self.chamber_pressure_search_min_pa)
            )
            hi = (
                recommended.upper
                if self.chamber_pressure_search_max_pa is None
                else float(self.chamber_pressure_search_max_pa)
            )
            if not math.isfinite(lo) or lo <= 0.0:
                raise ValueError(
                    "chamber_pressure_search_min_pa must be finite and positive"
                )
            if not math.isfinite(hi) or hi <= 0.0:
                raise ValueError(
                    "chamber_pressure_search_max_pa must be finite and positive"
                )
            if not lo < hi:
                raise ValueError(
                    "chamber-pressure search window must satisfy min < max"
                )
        if not math.isclose(
            float(self.cooling_fraction), 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "MissionSpec defines a two-branch fuel topology: all non-film "
                "fuel passes through the regenerative jacket. "
                "cooling_fraction must therefore equal 1.0; a lower value "
                "requires an explicit third bypass branch and closure "
                "constraint."
            )

    def design_layout(self) -> DesignLayout:
        """Return the explicit active-variable layout for this mission."""

        return DesignLayout.for_mission(self)

    # ----------------------------------------------------------------------- #
    # Thrust-class scaling                                                     #
    # ----------------------------------------------------------------------- #
    @classmethod
    def for_propellant(cls, propellant: str, thrust: float, *,
                       Pc_ref: float = 3.0e6, **overrides) -> "MissionSpec":
        """Thrust-scaled architecture for a **named propellant combination**.

        Pulls the chamber gas (γ, T_c, R), L*, densities, coolant
        thermophysical properties and the coolant wall/coking limit from
        ``raosim.mdo.propellants`` (SP-125 Table 4-1 for L*, SP-8087 for the
        wall limits, the repo's Sutton-sourced table for the gases), then
        applies the thrust scaling of :meth:`for_thrust`.

        Hydrogen has no coking limit — ``coolant_wall_limit_K is None`` — so
        its coking row is not applicable.  This does **not** prove the gas-side
        material limit governs: methane/hydrogen coolant-side HTD requires the
        separate real-fluid coverage gate in :mod:`raosim.mdo.coolant_htd`.
        Until that model is available, authoritative feasibility is unknown.
        """
        from raosim.mdo.propellants import get_propellant
        from raosim.physics import default_coolant_inlet_temperature

        p = get_propellant(propellant)
        # None (cannot coke) → a coking screen that can never bind.  The
        # independent HTD applicability/coverage row remains authoritative.
        coke = (p.coolant_wall_limit_K if p.coolant_wall_limit_K is not None
                else 1.0e4)
        derived = dict(
            propellant_name=p.name,
            OF=p.OF_default, gamma=p.gamma, Tc=p.Tc, R_gas=p.R_gas,
            eta_cstar=p.eta_cstar, eta_CF=p.eta_CF,
            l_star=p.l_star,
            rho_fuel=p.rho_fuel, rho_ox=p.rho_ox,
            rho_cool=p.rho_cool, cp_cool=p.cp_cool, k_cool=p.k_cool,
            mu_cool=p.mu_cool,
            film_latent_heat=p.film_latent_heat,
            cp_cool_vapor=p.cp_cool_vapor,
            film_mu_vapor=p.film_mu_vapor,
            coolant_sound_speed=p.coolant_sound_speed,
            p_vapor_fuel=p.p_vapor_fuel, p_vapor_ox=p.p_vapor_ox,
            rp1_coking_wall_temp_K=coke,
            # Jacket inlet temperature from the SAME central resolver the
            # traditional pipeline uses (``raosim.physics
            # .default_coolant_inlet_temperature``): 120 K for methane, 25 K
            # for hydrogen, 300 K otherwise.
            #
            # Without this the MDO ran every propellant at the class default of
            # 320 K -- an RP-1 number.  That put methane in at T/T_c = 1.68
            # instead of 0.63 and hydrogen at 9.7 instead of 0.75, i.e. a
            # different fluid state entirely, with the coolant enthalpy rise
            # starting from the wrong place and the wall-temperature and coking
            # margins computed off it.  Exactly the R0 failure mode: a central
            # convention that one pipeline used and the other did not.
            coolant_temperature=default_coolant_inlet_temperature(
                p.coolant_name),
        )
        derived.update(overrides)
        return cls.for_thrust(thrust, Pc_ref=Pc_ref, **derived)

    def with_materials(
        self,
        *,
        liner: Any = None,
        closeout: Any = None,
    ) -> "MissionSpec":
        """Atomically retarget every traced liner/closeout wall property.

        This is the single mapper between the ``raosim.materials`` catalog and
        the differentiable core.  Either every field the selection owns is
        resolved from one catalog record, or the call raises and the mission is
        unchanged -- a partially applied material is an alloy that exists in no
        catalog and matches no qualification data.

        Omitting a role keeps that role's current selection.  Because the
        default closeout is a hardenable jacket alloy and the default liner is
        a high-conductivity copper (SP-8087 sec. 2.1.3.1), the two roles are
        never inherited from one another.
        """
        from dataclasses import replace

        from raosim.mdo.material_map import resolve_material_selection

        liner_choice = liner if liner is not None else self.liner_material_name
        closeout_choice = (
            closeout if closeout is not None else self.closeout_material_name
        )
        if liner_choice is None or closeout_choice is None:
            missing = "liner" if liner_choice is None else "structural closeout"
            raise ValueError(
                f"with_materials needs an explicit {missing} material: this "
                "mission still carries unattributed class-default wall "
                "constants, so there is nothing to inherit"
            )
        selection = resolve_material_selection(
            liner=liner_choice,
            closeout=closeout_choice,
            liner_structural_fos=self.liner_structural_fos,
        )
        return replace(self, **selection.fields)

    @classmethod
    def for_material(
        cls,
        liner: Any,
        thrust: float,
        *,
        closeout: Any = "Inconel 718",
        propellant: str | None = None,
        Pc_ref: float = 3.0e6,
        **overrides,
    ) -> "MissionSpec":
        """Thrust-scaled architecture for a **named liner/closeout pair**.

        The companion of :meth:`for_propellant`: that selects the working
        fluid, this selects the metal.  Both may be combined --
        ``for_material("GRCop-84", 13e3, propellant="LOX/LCH4")`` -- and the
        material mapping is applied last so it owns the wall constants
        outright.

        The closeout defaults to the repository's standard jacket entry rather
        than to the liner alloy, because a copper jacket at these pressures
        needs a thickness that violates the thin-shell assumption the
        ``jacket_thin_shell_margin`` constraint reports.
        """
        base = (
            cls.for_propellant(propellant, thrust, Pc_ref=Pc_ref, **overrides)
            if propellant is not None
            else cls.for_thrust(thrust, Pc_ref=Pc_ref, **overrides)
        )
        return base.with_materials(liner=liner, closeout=closeout)

    @classmethod
    def for_thrust(cls, thrust: float, *, Pc_ref: float = 3.0e6,
                   **overrides) -> "MissionSpec":
        """Derive a **thrust-scaled** architecture from first principles.

        The hard-coded defaults on this class describe a 13 kN engine.  Nothing
        in the physics is 13 kN-specific, but several *architecture* constants
        (channel count, pintle diameter, pump speed) and their bounds must move
        with the engine or the optimiser is handed a design space that contains
        no feasible point.  This factory sizes them:

        1. **Throat** from the thrust closure, A_t = F/(C_F P_c) at a reference
           C_F (the outer Newton re-solves it exactly; this only sets scale).
        2. **Chamber length** from the characteristic length, L ≈ L*/CR
           (SP-125 §4, Table 4-1: L* = 40–50 in for LOX/RP-1) — thrust-
           independent at fixed L* and CR, which is why it barely moves.
        3. **Channel count** ∝ chamber circumference, holding the pitch (and so
           the land width) at a manufacturable value — the same rationale
           Mirzamoghadam uses when he fixes a channel count for producibility.
        4. **Pintle diameter** sized so the blockage factor lands mid-band,
           BF = N w/(π D_p) — BF being one of the two master pintle knobs
           (Hwang 2022, Freeberg 2019), so this is the literature design rule
           rather than an invented diameter ratio.
        5. **Pump speed** from the specific speed, ω = N_s (g₀H)^¾/√Q — bigger
           engines pump more volume, so the *rpm falls*; a fixed 15–60 krpm
           band is simply wrong at 10× thrust.

        ``overrides`` are applied last, so anything explicit wins.
        """
        import math

        probe = cls(thrust=thrust, **overrides)
        CR = probe.contraction_ratio

        # 1. throat / chamber geometry (reference C_F for a sea-level bell)
        Cf_ref = 1.4
        At = thrust / (Cf_ref * Pc_ref)
        Rt = math.sqrt(At / math.pi)
        r_c = Rt * math.sqrt(CR)
        D_c = 2.0 * r_c

        # 2. (barrel length is no longer a MissionSpec field -- it is solved
        #    from the L* volume closure by raosim.mdo.grid.chamber_barrel_length
        #    so the MDO and traditional chambers match; see MissionSpec.l_star.)

        # 3. channel count from circumference at the reference pitch
        n_channels = max(int(round(math.pi * D_c / probe.channel_pitch_ref)), 8)

        # 4. pintle sized so BF lands mid-band
        cstar_del = probe.eta_cstar * probe.c_star_ideal()
        mdot = Pc_ref * At / cstar_del
        mdot_f = mdot / (1.0 + probe.OF)
        dp_f = 0.20 * Pc_ref
        G_f = probe.injector_cd_fuel * math.sqrt(2.0 * probe.rho_fuel * dp_f)
        A_f = mdot_f / G_f
        N_slot = probe.pintle_slot_count
        w_slot = math.sqrt(A_f / (N_slot * probe.pintle_slot_aspect_ratio))
        bf_target = 0.5 * (probe.blockage_factor_min + probe.blockage_factor_max)
        D_p = N_slot * w_slot / (math.pi * bf_target)
        # centre gap / rod follow the post (Son 2017 geometry family)
        Dcg = 0.9 * D_p
        Dpr = 0.5 * D_p

        # 5. pump speed from specific speed
        Q = mdot_f / probe.rho_fuel
        H = (Pc_ref * 1.2) / (probe.rho_fuel * probe.g0)
        omega = probe.pump_ns_opt * (probe.g0 * H) ** 0.75 / math.sqrt(max(Q, 1e-12))
        rpm = omega * 60.0 / (2.0 * math.pi)

        derived = dict(
            thrust=thrust,
            n_channels=n_channels,
            pintle_diameter=D_p,
            pintle_center_gap_diameter=Dcg,
            pintle_rod_diameter=Dpr,
            pump_speed_rpm=rpm,
        )
        derived.update(overrides)          # explicit user values win
        return cls(**derived)

    def scaled_design_space(self) -> tuple["VariableSpec", ...]:
        """Design-variable bounds **scaled to this mission's thrust class**.

        Absolute/manufacturing-limited variables (channel section, wall
        thickness, film fraction, injector Δp) keep physical bounds; the
        architecture-scaled ones (pintle diameter, pump speed) bracket the
        values derived by :meth:`for_thrust`.
        """
        from raosim.mdo.bounds import (
            chamber_pressure_bounds, expansion_ratio_bounds,
            expansion_ratio_reference,
        )

        pc = chamber_pressure_bounds(self.feed_architecture)
        pc_lo = (
            pc.lower
            if self.chamber_pressure_search_min_pa is None
            else float(self.chamber_pressure_search_min_pa)
        )
        pc_hi = (
            pc.upper
            if self.chamber_pressure_search_max_pa is None
            else float(self.chamber_pressure_search_max_pa)
        )
        eps_lo, eps_hi = expansion_ratio_bounds(self)
        return (
            # The default Pc search guidance is architecture-dependent
            # (Parsley & Zhang 2004), and caller overrides can widen/narrow it.
            # These are optimizer-search endpoints, not cycle hard limits;
            # live physics and sampled-property domains gate validity.  Epsilon
            # still comes from the Rao chart's own tabulated box.
            VariableSpec("Pc", pc_lo, pc_hi,
                         min(max(3.0e6, pc_lo), pc_hi)),
            VariableSpec("eps", eps_lo, eps_hi,
                         expansion_ratio_reference(self)),
            VariableSpec("dp_f_frac", 0.12, 0.45, 0.2),
            VariableSpec("dp_o_frac", 0.12, 0.45, 0.2),
            # bracket the BF-sized pintle by ±2x
            VariableSpec("D_pintle", 0.5 * self.pintle_diameter,
                         2.0 * self.pintle_diameter, self.pintle_diameter),
            # bracket the specific-speed-sized pump by ±2x
            VariableSpec("N_rpm", 0.5 * self.pump_speed_rpm,
                         2.0 * self.pump_speed_rpm, self.pump_speed_rpm),
            VariableSpec("channel_width", 3.0e-4, 1.2e-3, 5.0e-4),
            VariableSpec("channel_height", 8.0e-4, 5.0e-3, 1.5e-3),
            # Start an unconstrained solve inside the film branch.  The liquid
            # film model has a deliberately flat/no-film limit, so seeding
            # exactly at zero can leave SLSQP with no usable feasibility
            # direction even when the coking constraint requires film.
            VariableSpec("film_frac", 0.0, 0.30, 0.10),
            VariableSpec("t_wall", 4.0e-4, 2.0e-3, 8.0e-4),
        ) + self.of_design_space()

    def of_design_space(self) -> tuple["VariableSpec", ...]:
        """The O/F variable spec — empty unless CEA surfaces make it real.

        With constant gamma/T_c/R the property surfaces are flat in O/F, so an
        O/F design variable would change the propellant mass split without
        changing combustion at all.  That is not a conservative approximation,
        it is a wrong model, and the honest response is to not offer the lever:
        the returned tuple is empty and the design space stays 10-dimensional.

        Once ``cea_table_path`` points at a sampled table, O/F becomes a real
        variable bounded by the **sampled domain**.  Bounding it there rather
        than at some physical band is deliberate: outside the sampled box the
        C1 surfaces are extrapolating, and ``property_domain_margin`` would
        already be screening the design as inadmissible.  Two mechanisms
        disagreeing about the same limit is how the two pipelines drifted apart
        before, so the bound and the constraint are taken from one source.
        """

        if not self.cea_table_path or self.of_is_pinned:
            return ()
        from raosim.mdo.properties import load_chamber_surfaces

        surfaces = load_chamber_surfaces(
            self.cea_table_path,
            expected_propellant=self.propellant_name,
            require_of_dependence=True,
        )
        lo = float(surfaces.gamma.yg[0])
        hi = float(surfaces.gamma.yg[-1])
        ref = min(max(float(self.OF), lo), hi)
        return (VariableSpec("OF", lo, hi, ref),)

    def c_star_ideal(self) -> float:
        """Ideal c* from the constant-property chamber state (SP-125 form,
        algebraically = sqrt(R·Tc)/Gamma(gamma); parity with
        gas_dynamics.characteristic_velocity is test-pinned)."""
        g = self.gamma
        gp1, gm1 = g + 1.0, g - 1.0
        import math

        return math.sqrt(g * self.R_gas * self.Tc) / (
            g * math.sqrt((2.0 / gp1) ** (gp1 / gm1))
        )


# --------------------------------------------------------------------------- #
# Design vector (pytree)                                                       #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
@dataclass
class DesignVector:
    """Eleven-value physical design contract plus a static active layout.

    :attr:`layout` says whether O/F is active in the optimizer.  :attr:`OF`
    itself is always a physical value; it never doubles as a layout sentinel.
    """

    Pc: Array          # chamber pressure [Pa]
    eps: Array         # expansion ratio [-]
    dp_f_frac: Array   # fuel injector dp / Pc [-]
    dp_o_frac: Array   # oxidizer injector dp / Pc [-]
    # Phase-7 whole-engine variables (plan §4.1 x_h: D_pintle, pump speeds).
    # Defaulted so the 4-var skeleton path (assembly.py) is unaffected — a
    # 4-long from_array leaves these at the mission-matched defaults; the engine
    # / NLP pass all of them and optimise them.
    D_pintle: Array = 0.020         # pintle diameter [m]
    N_rpm: Array = 30000.0          # pump speed [rpm]
    # Phase-7b cooling variables (channel geometry — continuous; N stays discrete
    # per rule 4) and the film-cooling fuel fraction (the coking lever, §6.2b).
    channel_width: Array = 5.0e-4   # coolant channel width [m]
    channel_height: Array = 1.5e-3  # coolant channel height [m]
    film_frac: Array = 0.0          # fuel fraction diverted to wall film [-]
    # Hot-gas wall thickness.  Mirzamoghadam: "with a thicker wall the heat
    # transfer conductance is reduced, and with a thinner wall the channel is
    # designed to withstand the pressure differential" — a genuine two-sided
    # trade, so it is a design variable.  NOTE the binding criterion is the
    # THERMAL gradient / low-cycle fatigue, not pressure (§10.2: pressure
    # bending across the channel is ~0.2 MPa vs ~27 MPa thermal).
    t_wall: Array = 8.0e-4          # m
    # Mixture ratio.  SP-125 §2.1 (printed p. 31) lists it among the nine
    # requirement parameters, but §2.1's own derivation makes it an OUTPUT:
    # the optimum balances energy release against molecular weight, then moves
    # off that optimum for cooling -- "The temperatures resulting from
    # stoichiometric or near-stoichiometric mixture ratios ... may impose
    # severe demands on the chamber-wall cooling system.  A lower temperature,
    # therefore, may be desired and obtained by selecting a suitable ratio."
    # That is precisely the trade this optimiser already resolves with film
    # fraction, so O/F belongs on the design vector.
    #
    # It is only a *real* variable when C1 property surfaces are loaded: with
    # constant gamma/Tc/R the thermochemistry is FLAT in O/F, so moving it
    # would change the propellant mass split without changing combustion --
    # physically wrong.  ``of_is_variable`` (static pytree aux, never traced)
    # records which regime this vector is in; ``default_design_space`` only
    # emits an O/F spec when ``MissionSpec.cea_table_path`` is set.
    #
    # Constructor compatibility default for the class-default LOX/RP-1
    # mission.  Internal code constructs vectors from an explicit layout and
    # supplies the mission's actual fixed value.  This is a physical value,
    # never an intent sentinel.
    OF: Array = 2.27                # oxidizer/fuel mass ratio [-]

    #: Static (non-traced) design intent.  It is pytree auxiliary data, so the
    #: fixed/variable branch is resolved at trace time rather than becoming a
    #: design-dependent numerical switch.
    layout: DesignLayout = field(default_factory=DesignLayout.fixed_of)

    # NOTE — the film-injector slot height S is deliberately **not** a design
    # variable.  Stechman (1969) *derives* it ("the slot height S is determined
    # from the static density of the gas at the vaporization temperature and the
    # velocity of the coolant at the original injection point"), his liquid-
    # cooled-length Eq. (2) contains no S, and Knuth's wave-transition criterion
    # is written on flow-per-circumference Γ = Ẇ/(πD), also S-independent.  For
    # an evaporation-limited liquid film S is therefore a *sizing/validity*
    # quantity, not an optimiser lever: it lives on ``MissionSpec`` and is
    # reported by ``cooling.film_slot_validity``.  (It genuinely matters for a
    # *gaseous* film, where S enters Hatch & Papell's correlation directly.)
    _FIELDS = _DESIGN_CONTRACT_FIELDS

    @property
    def of_is_variable(self) -> bool:
        """Deprecated compatibility view; use ``layout.of_is_variable``."""

        return self.layout.of_is_variable

    # -- pytree protocol ---------------------------------------------------- #
    def tree_flatten(self):
        # ``layout`` is AUX, not a child: it is a configuration fact,
        # not a number to differentiate, and keeping it static is what stops
        # the O/F regime from becoming a traced branch (plan §0.1).
        return (tuple(getattr(self, f) for f in self._FIELDS),
                self.layout)

    @classmethod
    def tree_unflatten(cls, aux, children):
        # ``bool`` accepts treedefs produced by the pre-v2 in-process class;
        # persisted v1 EngineState values are intentionally not migrated.
        layout = (
            DesignLayout.variable_of() if bool(aux)
            else DesignLayout.fixed_of()
        ) if isinstance(aux, bool) else aux
        return cls(*children, layout=layout)

    # -- vector packing ------------------------------------------------------ #
    def to_contract_array(self, *, effective_of: Array | None = None) -> Array:
        """Return the fixed eleven-value physical state/snapshot contract."""

        of = self.OF if effective_of is None else effective_of
        values = [getattr(self, f) for f in _DESIGN_BASE_FIELDS] + [of]
        return jnp.stack([
            jnp.asarray(value, dtype=jnp.float64) for value in values
        ])

    def to_active_array(self) -> Array:
        """Return the explicit 10- or 11-value optimizer vector."""

        return jnp.stack([
            jnp.asarray(getattr(self, name), dtype=jnp.float64)
            for name in self.layout.active_names
        ])

    def to_array(self) -> Array:
        """Compatibility alias for the fixed physical contract vector."""

        return self.to_contract_array()

    @classmethod
    def from_active_array(
        cls,
        x: Array,
        layout: DesignLayout,
        *,
        fixed_of: float | Array | None = None,
    ) -> "DesignVector":
        """Build from an explicitly described 10- or 11-value active vector."""

        x = jnp.asarray(x)
        n = int(x.shape[0])
        if n != layout.active_size:
            raise ValueError(
                f"active design vector has length {n}; layout requires "
                f"{layout.active_size} ({', '.join(layout.active_names)})"
            )
        if layout.of_is_variable:
            return cls(*(x[i] for i in range(n)), layout=layout)
        if fixed_of is None:
            raise ValueError("fixed-O/F layout requires fixed_of=")
        return cls(
            *(x[i] for i in range(n)),
            OF=jnp.asarray(fixed_of, dtype=jnp.float64),
            layout=layout,
        )

    @classmethod
    def from_contract_array(
        cls,
        x: Array,
        layout: DesignLayout,
    ) -> "DesignVector":
        """Build from the fixed eleven-value physical contract + layout."""

        x = jnp.asarray(x)
        n = int(x.shape[0])
        if n != len(_DESIGN_CONTRACT_FIELDS):
            raise ValueError(
                "design contract vector must contain exactly 11 physical values"
            )
        return cls(*(x[i] for i in range(n)), layout=layout)

    @classmethod
    def from_array(cls, x: Array, *,
                   of_is_variable: bool | None = None) -> "DesignVector":
        """Deprecated compatibility adapter for 4–11 ordered values.

        Omitted trailing values retain their dataclass defaults for
        compatibility with the early four-/six-variable skeletons.

        Only this deprecated boundary infers layout from length.  Internal MDO
        code must use :meth:`from_active_array` or :meth:`from_contract_array`.
        """
        warnings.warn(
            "DesignVector.from_array() infers design layout from array length; "
            "use from_active_array(..., layout, fixed_of=...) or "
            "from_contract_array(..., layout)",
            DeprecationWarning,
            stacklevel=2,
        )
        x = jnp.asarray(x)
        n = int(x.shape[0])
        if n < 4 or n > len(cls._FIELDS):
            raise ValueError("legacy design vector length must be between 4 and 11")
        if of_is_variable is None:
            of_is_variable = n >= len(cls._FIELDS)
        if of_is_variable and n < len(cls._FIELDS):
            raise ValueError("variable-O/F design vector must include a physical OF")
        layout = (
            DesignLayout.variable_of()
            if of_is_variable else DesignLayout.fixed_of()
        )
        return cls(*(x[i] for i in range(n)), layout=layout)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return cls._FIELDS

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self._FIELDS}

    def as_contract_dict(self, *, effective_of: Array | None = None) -> dict:
        """Return all eleven physical values, overriding O/F after a solve."""

        out = {name: getattr(self, name) for name in _DESIGN_BASE_FIELDS}
        out["OF"] = self.OF if effective_of is None else effective_of
        return out


# --------------------------------------------------------------------------- #
# Bounds                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VariableSpec:
    name: str
    lower: float
    upper: float
    reference: float | None = None   # scaling reference; default midpoint

    def ref(self) -> float:
        return self.reference if self.reference is not None else (
            0.5 * (self.lower + self.upper))


def default_design_space(mission: "MissionSpec | None" = None
                         ) -> tuple[VariableSpec, ...]:
    """Skeleton bounds.

    Pc window brackets the 13 kN LOX/RP-1 baseline (Pc 3.0 MPa was shown
    thermally feasible; 7 MPa infeasible for catalog liners — TEST_PROMPT.md);
    dp fractions bracket the SP-125 15–20 % stability rule with headroom.

    Pass a ``mission`` to get **thrust-scaled** bounds (see
    ``MissionSpec.scaled_design_space``); without one the 13 kN-class defaults
    are returned, which is what the historical callers expect.
    """
    if mission is not None:
        return mission.scaled_design_space()
    return (
        VariableSpec("Pc", 1.5e6, 6.0e6, 3.0e6),
        VariableSpec("eps", 3.0, 40.0, 8.0),
        VariableSpec("dp_f_frac", 0.12, 0.45, 0.2),
        VariableSpec("dp_o_frac", 0.12, 0.45, 0.2),
        # whole-engine variables (D_pintle bracket a machinable 10–40 mm post;
        # N_rpm brackets electric-pump practice, Rutherford-class ~30–40 krpm)
        VariableSpec("D_pintle", 0.010, 0.040, 0.020),
        VariableSpec("N_rpm", 1.5e4, 6.0e4, 3.0e4),
        # cooling channel geometry (HARCC aspect up to ~8–15, Pizzarelli/Carlile)
        VariableSpec("channel_width", 3.0e-4, 1.2e-3, 5.0e-4),
        VariableSpec("channel_height", 8.0e-4, 5.0e-3, 1.5e-3),
        # film-cooling fuel fraction (0 = pure regen; upper ~30% before the c*
        # penalty dominates — SP-8087 combined regen+film practice)
        # Interior reference for optimizer robustness; zero remains a valid
        # bound and is still used by explicit no-film designs.
        VariableSpec("film_frac", 0.0, 0.30, 0.10),
        # hot-gas wall thickness: thin for conductance, thick for structure.
        # SP-8087-era liners run ~0.5–1.5 mm (Mirzamoghadam quotes 0.61–0.94 mm
        # for a tube wall); bracket that with machining floors.
        VariableSpec("t_wall", 4.0e-4, 2.0e-3, 8.0e-4),
    ) + MissionSpec().of_design_space()


def bounds_arrays(space: tuple[VariableSpec, ...]) -> tuple[Array, Array]:
    lo = jnp.asarray([s.lower for s in space], dtype=jnp.float64)
    hi = jnp.asarray([s.upper for s in space], dtype=jnp.float64)
    return lo, hi


def iter_space(space: tuple[VariableSpec, ...]) -> Iterator[VariableSpec]:
    yield from space
