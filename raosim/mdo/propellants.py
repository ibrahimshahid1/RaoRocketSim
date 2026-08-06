"""
raosim.mdo.propellants — propellant combinations as a first-class MDO input.

The MDO layer was originally written around a single LOX/RP-1 engine: the
coking limit, the coolant thermophysical properties, L* and the chamber gas
constants were all RP-1 values baked into ``MissionSpec``.  This module makes
the propellant a **selectable input**, so the same optimiser designs a
LOX/LH2 upper stage or an N2O4/MMH thruster without editing code.

Sources
-------
* **Chamber gases** (γ, M_w, T_c) come from the repository's existing audited
  table, ``raosim.propellants`` (Sutton Table 5-5 values) — not re-derived here.
* **L\\*** from **SP-125 (Huzel & Huang) Table 4-1**, "Recommended Combustion
  Chamber Characteristic Length for Various Propellant Combinations":
  LOX/RP-1 40–50 in, LOX/LH2 30–40 in (LH2 injection) or 22–28 in (GH2),
  N2O4/hydrazine-base 30–35 in, LOX/ammonia 30–40 in, H2O2/RP-1 60–70 in.
* **Coolant wall limits** from **SP-8087**: "do not operate with liquid-wall
  temperatures above 850°F (728 K) for RP-1, or above 600°F (589 K) for
  furfuryl alcohol.  Aerozine-50 should be used at wall temperatures below
  600°F (589 K)" — RP-1's coking onset is quoted as 800–900°F (700–756 K).

Honesty about coverage
----------------------
Coking is a **hydrocarbon** phenomenon.  Hydrogen has no carbon and therefore
no coking limit at all: for LOX/LH2 the limit is ``None`` and the gas-side
wall-temperature constraint governs instead — which is the physically correct
behaviour, not a missing number.  Methane post-dates SP-125/SP-8087, so its
L\\* and wall limit are flagged ``estimated=True``; treat those two numbers as
engineering estimates, not literature values.
"""

from __future__ import annotations

from dataclasses import dataclass

_IN = 0.0254   # m per inch
_R_UNIVERSAL = 8314.462618   # J/(kmol K)


@dataclass(frozen=True)
class PropellantSpec:
    """Everything the MDO needs to know about a propellant combination."""

    name: str
    # chamber gas (from raosim.propellants — Sutton Table 5-5)
    gamma: float
    Tc: float
    R_gas: float                    # J/(kg K)
    OF_default: float
    # Preserve the traditional repository split: combustion losses apply to
    # c*, nozzle losses to Cf.  Do not silently replace these with eta_Isp.
    eta_cstar: float
    eta_CF: float
    # chamber sizing
    l_star: float                   # m (SP-125 Table 4-1 unless estimated)
    # densities
    rho_fuel: float
    rho_ox: float
    # regen coolant (normally the fuel)
    coolant_name: str
    rho_cool: float
    cp_cool: float                  # J/(kg K)
    k_cool: float                   # W/(m K)
    mu_cool: float                  # Pa s
    # film-cooling (liquid phase-change model)
    film_latent_heat: float         # J/kg
    cp_cool_vapor: float            # J/(kg K)
    film_mu_vapor: float            # Pa s
    coolant_sound_speed: float      # m/s
    p_vapor_fuel: float             # Pa
    p_vapor_ox: float               # Pa
    #: liquid-side wall limit; ``None`` when the fluid cannot coke (e.g. H2),
    #: in which case only the gas-side material limit applies.
    coolant_wall_limit_K: float | None
    #: True when l_star / wall limit are engineering estimates, not literature.
    estimated: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Table                                                                        #
# --------------------------------------------------------------------------- #
def _R(Mw_kg_per_mol: float) -> float:
    return _R_UNIVERSAL / (Mw_kg_per_mol * 1000.0)


PROPELLANTS: dict[str, PropellantSpec] = {
    "lox/rp-1": PropellantSpec(
        name="LOX/RP-1",
        gamma=1.24, Tc=3571.0, R_gas=_R(0.0219), OF_default=2.27,
        eta_cstar=0.975, eta_CF=0.985,
        l_star=45.0 * _IN,                     # SP-125 Tab 4-1: 40–50 in
        rho_fuel=810.0, rho_ox=1141.0,
        coolant_name="RP-1",
        rho_cool=810.0, cp_cool=2093.0, k_cool=0.11, mu_cool=1.0e-3,
        film_latent_heat=2.5e5, cp_cool_vapor=2.5e3, film_mu_vapor=1.0e-5,
        coolant_sound_speed=1300.0, p_vapor_fuel=3.0e3, p_vapor_ox=1.0e5,
        coolant_wall_limit_K=728.0,            # SP-8087: 850 °F
        notes="SP-8087 coking onset 800–900 °F (700–756 K); design limit 850 °F.",
    ),
    "lox/lch4": PropellantSpec(
        name="LOX/LCH4",
        gamma=1.20, Tc=3533.0, R_gas=_R(0.0203), OF_default=3.5,
        eta_cstar=0.975, eta_CF=0.985,
        l_star=40.0 * _IN,                     # ESTIMATE (post-dates SP-125)
        rho_fuel=422.0, rho_ox=1141.0,
        coolant_name="methane",
        rho_cool=422.0, cp_cool=3500.0, k_cool=0.19, mu_cool=1.2e-4,
        film_latent_heat=5.1e5, cp_cool_vapor=2.2e3, film_mu_vapor=1.1e-5,
        coolant_sound_speed=1400.0, p_vapor_fuel=1.0e5, p_vapor_ox=1.0e5,
        coolant_wall_limit_K=950.0,            # ESTIMATE — CH4 cokes far less
        estimated=True,
        notes="L* and wall limit are ENGINEERING ESTIMATES: methane post-dates "
              "SP-125/SP-8087, which carry no LOX/LCH4 entry.",
    ),
    "lox/lh2": PropellantSpec(
        name="LOX/LH2",
        gamma=1.26, Tc=2999.0, R_gas=_R(0.0089), OF_default=4.83,
        eta_cstar=0.99, eta_CF=0.99,
        l_star=35.0 * _IN,                     # SP-125 Tab 4-1: 30–40 in (LH2 inj)
        rho_fuel=71.0, rho_ox=1141.0,
        coolant_name="hydrogen",
        rho_cool=71.0, cp_cool=9700.0, k_cool=0.10, mu_cool=1.3e-5,
        film_latent_heat=4.46e5, cp_cool_vapor=1.4e4, film_mu_vapor=8.8e-6,
        coolant_sound_speed=1100.0, p_vapor_fuel=1.0e5, p_vapor_ox=1.0e5,
        coolant_wall_limit_K=None,             # no carbon -> cannot coke
        notes="No coking limit (no carbon); the gas-side material wall-temp "
              "constraint governs instead.",
    ),
    "n2o4/mmh": PropellantSpec(
        name="N2O4/MMH",
        gamma=1.23, Tc=3122.0, R_gas=_R(0.0215), OF_default=2.17,
        eta_cstar=0.97, eta_CF=0.985,
        l_star=32.0 * _IN,                     # SP-125: N2O4/hydrazine-base 30–35 in
        rho_fuel=875.0, rho_ox=1442.0,
        coolant_name="MMH",
        rho_cool=875.0, cp_cool=2900.0, k_cool=0.25, mu_cool=7.8e-4,
        film_latent_heat=8.75e5, cp_cool_vapor=2.0e3, film_mu_vapor=1.0e-5,
        coolant_sound_speed=1400.0, p_vapor_fuel=6.6e3, p_vapor_ox=9.6e4,
        coolant_wall_limit_K=589.0,            # SP-8087 hydrazine family: 600 °F
        notes="SP-8087: Aerozine-50 detonated against walls above 600 °F "
              "(589 K); the hydrazine-family limit is applied here.",
    ),
    "n2o/ethanol": PropellantSpec(
        name="N2O/Ethanol",
        gamma=1.22, Tc=2950.0, R_gas=_R(0.0264), OF_default=5.5,
        eta_cstar=0.94, eta_CF=0.98,
        l_star=40.0 * _IN,                     # ESTIMATE (not in SP-125 Tab 4-1)
        rho_fuel=789.0, rho_ox=1220.0,
        coolant_name="ethanol",
        rho_cool=789.0, cp_cool=2440.0, k_cool=0.17, mu_cool=1.1e-3,
        film_latent_heat=8.4e5, cp_cool_vapor=1.9e3, film_mu_vapor=1.0e-5,
        coolant_sound_speed=1160.0, p_vapor_fuel=5.8e3, p_vapor_ox=5.1e6,
        coolant_wall_limit_K=589.0,            # SP-8087 alcohols: 600 °F
        estimated=True,
        notes="L* is an ESTIMATE (no SP-125 entry).  Wall limit uses the "
              "SP-8087 alcohol figure (furfuryl alcohol, 600 °F).",
    ),
}

_ALIASES = {
    "lox/rp1": "lox/rp-1", "loxrp1": "lox/rp-1", "rp1": "lox/rp-1",
    "kerolox": "lox/rp-1",
    "lox/ch4": "lox/lch4", "methalox": "lox/lch4", "lox/methane": "lox/lch4",
    "lox/h2": "lox/lh2", "hydrolox": "lox/lh2",
    "n2o4/hydrazine": "n2o4/mmh", "mmh": "n2o4/mmh",
    "n2o/etoh": "n2o/ethanol",
}


def get_propellant(name: str) -> PropellantSpec:
    """Look up a combination by name (case/alias tolerant)."""
    key = str(name).strip().lower().replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in PROPELLANTS:
        raise KeyError(
            f"unknown propellant {name!r}; available: "
            + ", ".join(sorted(PROPELLANTS)))
    return PROPELLANTS[key]


def available() -> tuple[str, ...]:
    return tuple(sorted(PROPELLANTS))
