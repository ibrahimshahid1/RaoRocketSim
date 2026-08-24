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

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

_IN = 0.0254   # m per inch
_R_UNIVERSAL = 8314.462618   # J/(kmol K)


class WallSideMechanism(str, Enum):
    """Which coolant-side wall mechanism governs, stated rather than guessed.

    This exists because inferring the mechanism from a coolant *name* is
    unsafe.  ``HTD_RELEVANT_COOLANTS`` is a fixed spelling list, so ``"LNG"``,
    ``"ethane"`` and ``"subcooled methane"`` all fall through it and would be
    handed the hydrocarbon coking limit instead of the supercritical
    heat-transfer-deterioration criterion -- silently, and for a fluid where
    the coking limit is not the governing physics.

    A registered propellant must therefore declare its mechanism.  ``UNKNOWN``
    is a legitimate answer and reduces feasibility to ``unknown``; it is never
    a pass.
    """

    #: Hydrocarbon liquid-side deposition limit (SP-8087 sec. 3.1.1.5.4).
    COKING = "coking"
    #: Supercritical deterioration (Nasuti & Pizzarelli 2021, Eq. 9).
    SUPERCRITICAL_HTD = "supercritical_htd"
    #: Neither mechanism has been demonstrated for this fluid.
    UNKNOWN = "unknown"


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
    #: Which coolant-side mechanism governs.  ``None`` means the legacy
    #: name-based inference is still in force; every catalog entry declares it
    #: explicitly and :func:`register_propellant` requires it.
    wall_side_mechanism: WallSideMechanism | None = None
    #: ``"catalog"`` for repository-validated entries, ``"user_registered"``
    #: for anything supplied at runtime.  A report must be able to say which.
    provenance: str = "catalog"
    #: Free-form evidence for a user-registered combination: where L*, the
    #: efficiencies, the wall limit and the material compatibility come from.
    evidence: Mapping[str, str] | None = None


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
        wall_side_mechanism=WallSideMechanism.COKING,
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
        wall_side_mechanism=WallSideMechanism.SUPERCRITICAL_HTD,
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
        wall_side_mechanism=WallSideMechanism.SUPERCRITICAL_HTD,
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
        wall_side_mechanism=WallSideMechanism.COKING,
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
        wall_side_mechanism=WallSideMechanism.COKING,
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
    if key in PROPELLANTS:
        return PROPELLANTS[key]
    if key in _REGISTERED:
        return _REGISTERED[key]
    raise KeyError(
        f"unknown propellant {name!r}; available: "
        + ", ".join(sorted(PROPELLANTS))
        + (f"; registered: {', '.join(sorted(_REGISTERED))}"
           if _REGISTERED else "")
        + ". Register a custom combination with "
          "raosim.mdo.propellants.register_propellant().")


def available() -> tuple[str, ...]:
    return tuple(sorted(PROPELLANTS))


# --------------------------------------------------------------------------- #
# Runtime registration of a custom combination                                 #
# --------------------------------------------------------------------------- #
class PropellantCoverageError(ValueError):
    """A custom propellant is incompletely or unsafely specified."""


#: Registered at runtime.  Kept separate from ``PROPELLANTS`` so a report can
#: always distinguish a repository-validated entry from a user-supplied one.
_REGISTERED: dict[str, PropellantSpec] = {}

#: Numeric fields a custom combination must supply.  There is no partial
#: registration: the same rule as the material mapper, for the same reason --
#: a half-specified propellant is a fluid that exists in no table and matches
#: no test data.
_REQUIRED_POSITIVE: tuple[str, ...] = (
    "gamma", "Tc", "R_gas", "OF_default", "eta_cstar", "eta_CF", "l_star",
    "rho_fuel", "rho_ox", "rho_cool", "cp_cool", "k_cool", "mu_cool",
    "film_latent_heat", "cp_cool_vapor", "film_mu_vapor",
    "coolant_sound_speed", "p_vapor_fuel", "p_vapor_ox",
)

#: Evidence a custom combination must cite before it is usable.
_REQUIRED_EVIDENCE: tuple[str, ...] = (
    "chamber_gas", "l_star_basis", "efficiency_basis",
    "coolant_properties", "wall_limit_basis", "material_compatibility",
)


def register_propellant(
    spec: PropellantSpec,
    *,
    evidence: Mapping[str, str] | None = None,
    replace_existing: bool = False,
) -> PropellantSpec:
    """Register a custom combination for this process, or raise.

    Registration is deliberately demanding.  A propellant selects the chamber
    gas, the chamber sizing, the coolant, the wall-side failure mechanism and
    the material compatibility all at once; accepting a partially specified one
    would let a design be optimized against a fluid nobody has characterized.

    ``wall_side_mechanism`` must be stated.  It cannot be inferred from the
    coolant name -- ``"LNG"``, ``"ethane"`` and ``"subcooled methane"`` all
    miss the built-in supercritical name list and would otherwise be handed the
    hydrocarbon coking limit, which is not their governing physics.  Declaring
    :attr:`WallSideMechanism.UNKNOWN` is allowed and honest; it makes coverage
    ``unknown`` rather than granting a pass.
    """

    if not isinstance(spec, PropellantSpec):
        raise PropellantCoverageError("spec must be a PropellantSpec")

    name = str(spec.name).strip()
    if "/" not in name:
        raise PropellantCoverageError(
            f"propellant name must be OXIDIZER/FUEL; got {spec.name!r}"
        )

    for field_name in _REQUIRED_POSITIVE:
        value = getattr(spec, field_name, None)
        if value is None:
            raise PropellantCoverageError(
                f"{name}: {field_name} is required; a custom propellant may "
                "not inherit a catalog default for it"
            )
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise PropellantCoverageError(
                f"{name}: {field_name} must be finite and positive; got {value!r}"
            )
    if not str(spec.coolant_name).strip():
        raise PropellantCoverageError(f"{name}: coolant_name is required")

    mechanism = spec.wall_side_mechanism
    if mechanism is None:
        raise PropellantCoverageError(
            f"{name}: wall_side_mechanism must be declared explicitly "
            f"({', '.join(m.value for m in WallSideMechanism)}). It cannot be "
            "inferred from the coolant name without risking the wrong "
            "wall-side physics."
        )
    mechanism = WallSideMechanism(mechanism)
    if mechanism is WallSideMechanism.COKING:
        limit = spec.coolant_wall_limit_K
        if limit is None or not math.isfinite(float(limit)) or float(limit) <= 0:
            raise PropellantCoverageError(
                f"{name}: mechanism 'coking' requires a finite positive "
                "coolant_wall_limit_K -- the limit IS the constraint"
            )

    supplied = dict(evidence or spec.evidence or {})
    missing = [key for key in _REQUIRED_EVIDENCE if not supplied.get(key)]
    if missing:
        raise PropellantCoverageError(
            f"{name}: missing evidence for {', '.join(missing)}. A custom "
            "propellant must cite where its chamber gas, L*, efficiencies, "
            "coolant properties, wall limit and material compatibility come "
            "from; unsourced numbers must not silently become a design basis."
        )

    key = name.lower().replace(" ", "")
    if key in PROPELLANTS:
        raise PropellantCoverageError(
            f"{name}: shadows the validated catalog entry {key!r}; choose a "
            "distinct name rather than overriding repository-validated data"
        )
    if key in _REGISTERED and not replace_existing:
        raise PropellantCoverageError(
            f"{name}: already registered; pass replace_existing=True to change it"
        )

    stored = replace(
        spec,
        wall_side_mechanism=mechanism,
        provenance="user_registered",
        evidence=dict(supplied),
        estimated=True,
    )
    _REGISTERED[key] = stored
    return stored


def unregister_propellant(name: str) -> None:
    """Remove a runtime registration (tests and interactive sessions)."""

    _REGISTERED.pop(str(name).strip().lower().replace(" ", ""), None)


def registered_propellants() -> tuple[str, ...]:
    return tuple(sorted(_REGISTERED))


def wall_side_mechanism_for(name: str) -> WallSideMechanism | None:
    """Declared mechanism for a combination, or ``None`` if it has none."""

    try:
        return get_propellant(name).wall_side_mechanism
    except KeyError:
        return None
