"""
raosim.electric_technology — one electric-drive technology source for both
pipelines.

Before this module the traditional sizing (``raosim.pumps.SCREENING_DEFAULTS``)
and the differentiable MDO (``raosim.mdo.schema.MissionSpec``) carried two
different, unsourced electric-drive sets — battery power density 3.0 vs
1.3 kW/kg, battery energy density 250 vs ≈180.6 Wh/kg, motor 2.5 vs 3.0 kW/kg,
inverter 15 vs 20 kW/kg (MDO issues audit 2026-10-16, §1.3 item 19).  About
94 % of the MDO objective is that electric package, so an MDO optimum and the
parts sized from it described two different technologies.

Each :class:`ElectricTechnologySet` here is one complete, internally
consistent set, and both pipelines read the same set:

* ``raosim.pumps.SCREENING_DEFAULTS`` takes its electric entries from
  :data:`DEFAULT_ELECTRIC_TECHNOLOGY`;
* ``MissionSpec.electric_technology`` names a set, its numeric electric fields
  default to that set, and ``MissionSpec`` refuses values that disagree with
  the declared set unless the set is :data:`CUSTOM_ELECTRIC_TECHNOLOGY_ID`;
* the MDO→traditional handoff (``raosim.mdo.postprocess._pump_spec``) passes
  the mission's set, so the re-evaluated pump package uses the numbers the
  optimizer used.

Power-density conventions (all sets, Lee et al. 2021, DOI
10.1007/s42405-020-00325-z, PDF p. 7 Eqs. (26)-(29) and PDF p. 8 Eqs.
(30)-(32), checked on the page):

* motor density on motor SHAFT OUTPUT power, ``m_m = P_m,out / δ_m``;
* inverter density on inverter OUTPUT power, which equals the motor
  electrical input: ``m_inv = P_m,out / (δ_inv η_m)``;
* battery power branch on battery output = inverter input,
  ``m_b,P = κ_b P_inv,out / (δ_P,b η_inv)``;
* battery energy branch ``m_b,E = κ_b P_inv,out t_b / (δ_E,b η_E η_inv)``;
  ``m_b = max(m_b,P, m_b,E)``, the structural margin κ_b applying to both.

Two sets are registered:

``repository_screening_v1`` (default)
    The values the traditional ``raosim.pumps`` path has always used.  They
    are a repository screening assumption with no single source; the
    published brackets below place each against the held literature.  Two of
    them lie outside every held value (inverter density below, inverter
    efficiency above) and the constant discharge efficiency stands in for a
    burn-time correlation.  Recorded, not hidden.
``lee2021_ijass_electric_pump``
    Lee et al. 2021, Table 2 (PDF p. 11): every battery, motor and inverter
    value read on the page.  Table 2's pump efficiencies (66 %/61 %) and 10 %
    pump margin are not part of a technology set and are not applied.  Its
    battery energy efficiency is Eq. (38) evaluated at the paper's 600 s burn
    (PDF p. 10); it is not a constant at other burn times, and a mission with
    a different burn time raises :class:`ElectricTechnologyValidityWarning`.

The MDO-only ``MissionSpec.pump_specific_mass`` is deliberately NOT part of a
technology set: no held source states a pump mass per pump shaft power on the
basis the MDO uses (Kwak 2018 and Pavlov Rachov 2013 give pump-only densities
of 15-22 kW/kg; Casalino 2019 and Solda & Lentini 2008 give combined
drive+pump masses per electric power), and the traditional path sizes pump
mass from hardware geometry instead.  See ``MissionSpec`` for its flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import warnings
from types import MappingProxyType
from typing import Any, Mapping

#: Basis classes a set may declare.
ELECTRIC_BASIS_CLASSES = ("literature", "repository_screening_assumption")

#: ``MissionSpec`` field  ->  :class:`ElectricTechnologySet` field.
MISSION_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("eta_motor", "motor_efficiency"),
    ("eta_inverter", "inverter_efficiency"),
    ("eta_discharge", "battery_discharge_efficiency"),
    ("motor_power_density", "motor_power_density"),
    ("inverter_power_density", "inverter_power_density"),
    ("battery_energy_density", "battery_energy_density"),
    ("battery_power_density", "battery_power_density"),
    ("battery_structural_margin", "battery_structural_margin"),
)

#: ``raosim.pumps.SCREENING_DEFAULTS`` key  ->  set field (identity names).
SCREENING_DEFAULT_KEYS: tuple[str, ...] = (
    "motor_efficiency",
    "inverter_efficiency",
    "motor_power_density",
    "inverter_power_density",
    "battery_energy_density",
    "battery_power_density",
    "battery_discharge_efficiency",
    "battery_structural_margin",
)

_NUMERIC_FIELDS = tuple(name for _, name in MISSION_FIELD_MAP)
_EFFICIENCY_FIELDS = (
    "motor_efficiency", "inverter_efficiency", "battery_discharge_efficiency",
)


@dataclass(frozen=True)
class TechnologySource:
    """One page-located statement in a held source PDF."""

    source_id: str
    #: Repository-relative path of the original PDF (the authority).
    pdf_path: str
    #: 1-based page of the PDF file (not the printed page number).
    pdf_page: int
    location: str


@dataclass(frozen=True)
class PublishedBracket:
    """Held published values that bound one parameter of a set.

    ``low``/``high`` are the smallest and largest values the held sources
    state for this quantity on a compatible basis; ``held_values`` names who
    states what and where.  :meth:`position` says where a set's value falls.
    """

    parameter: str
    low: float
    high: float
    held_values: str

    def position(self, value: float) -> str:
        if value < self.low:
            return "below"
        if value > self.high:
            return "above"
        return "inside"


@dataclass(frozen=True)
class ElectricTechnologySet:
    """A complete electric-drive and battery technology set."""

    id: str
    label: str
    basis_class: str
    #: W/kg — battery DC output (= inverter input) per pack mass, before κ_b.
    battery_power_density: float
    #: J/kg — usable pack energy per pack mass, before κ_b.
    battery_energy_density: float
    #: κ_b — multiplies both the power- and the energy-limited battery mass.
    battery_structural_margin: float
    #: η_E — divides the required energy, E = P t_b / η_E.
    battery_discharge_efficiency: float
    #: W/kg — motor SHAFT OUTPUT power per motor mass.
    motor_power_density: float
    #: motor shaft output / motor electrical input.
    motor_efficiency: float
    #: W/kg — inverter OUTPUT (= motor electrical input) per inverter mass.
    inverter_power_density: float
    #: inverter output / inverter DC input.
    inverter_efficiency: float
    sources: tuple[TechnologySource, ...] = ()
    notes: str = ""
    #: Burn time at which the source evaluated a burn-time-dependent
    #: discharge efficiency; ``None`` when the set states a plain constant.
    discharge_efficiency_burn_time_s: float | None = None
    published_brackets: tuple[PublishedBracket, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or self.id == CUSTOM_ELECTRIC_TECHNOLOGY_ID:
            raise ValueError("an electric technology set needs a reserved-free id")
        if self.basis_class not in ELECTRIC_BASIS_CLASSES:
            raise ValueError(
                f"basis_class must be one of {ELECTRIC_BASIS_CLASSES}; "
                f"got {self.basis_class!r}"
            )
        _validate_values({name: getattr(self, name) for name in _NUMERIC_FIELDS},
                         owner=f"electric technology {self.id!r}")
        if self.basis_class == "literature" and not self.sources:
            raise ValueError("a literature set must cite its source pages")
        for bracket in self.published_brackets:
            if bracket.parameter not in _NUMERIC_FIELDS:
                raise ValueError(f"unknown bracket parameter {bracket.parameter!r}")
            if not bracket.low <= bracket.high:
                raise ValueError(f"bracket for {bracket.parameter} is inverted")

    def mission_fields(self) -> dict[str, float]:
        """``MissionSpec`` keyword values for this set."""

        return {mission: float(getattr(self, name))
                for mission, name in MISSION_FIELD_MAP}

    def screening_defaults(self) -> dict[str, float]:
        """``raosim.pumps.SCREENING_DEFAULTS`` entries for this set."""

        return {key: float(getattr(self, key)) for key in SCREENING_DEFAULT_KEYS}

    def bracket_positions(self) -> dict[str, str]:
        """``inside``/``below``/``above`` the held published values."""

        return {
            bracket.parameter: bracket.position(float(getattr(self, bracket.parameter)))
            for bracket in self.published_brackets
        }


def _validate_values(values: Mapping[str, Any], *, owner: str) -> None:
    for name, raw in values.items():
        if isinstance(raw, bool):
            raise ValueError(f"{owner}: {name} must be a number, not bool")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{owner}: {name} must be finite and positive")
        if name in _EFFICIENCY_FIELDS and value > 1.0:
            raise ValueError(f"{owner}: {name} must lie in (0, 1]")
        if name == "battery_structural_margin" and value < 1.0:
            raise ValueError(f"{owner}: battery_structural_margin must be >= 1")


#: Reserved id: the numeric fields were supplied directly and belong to no set.
CUSTOM_ELECTRIC_TECHNOLOGY_ID = "custom"

_LEE = "propulsion_texts/10.1007_s42405-020-00325-z.pdf"
_KWAK = "propulsion_texts/fuel_pump_design/1-s2.0-S1270963817320953-main.pdf"
_CASALINO = "propulsion_texts/fuel_pump_design/aerospace-06-00036.pdf"
_PAVLOV = "propulsion_texts/10.2514_1.B34714.pdf"
_PARK = "propulsion_texts/10.1007_s42405-022-00561-5.pdf"
_SOLDA = "propulsion_texts/10.2514_1.35074.pdf"

_LEE_ID = "Lee-2021-IJASS-22-94-DOI-10.1007-s42405-020-00325-z"

# Published brackets for the screening set.  Every value is a page-located
# statement in a held PDF (research pass of 2026-10-16); "PDF p." is the page
# of the PDF file.  Kwak 2018 = DOI 10.1016/j.ast.2018.02.033; Casalino 2019 =
# DOI 10.3390/aerospace6030036; Pavlov Rachov 2013 = DOI 10.2514/1.B34714;
# Park 2023 = DOI 10.1007/s42405-022-00561-5; Solda & Lentini 2008 = DOI
# 10.2514/1.35074.
_SCREENING_BRACKETS = (
    PublishedBracket(
        "battery_power_density", 650.0, 6950.0,
        "Lee 2021 0.650 kW/kg (Table 2, PDF p.11); Kwak 2018 cells 1.53, "
        "1.64, 1.68 kW/kg (Table 2, PDF p.5) and cell D 6.95 kW/kg (Table 3, "
        "PDF p.9); Casalino 2019 set A 3.00 and set B 6.95 kW/kg (Table 1, "
        "PDF p.6); Pavlov Rachov 2013 Li-S 1.2, Li-ion 2.0, Li-Po 6.0 kW/kg "
        "(Table 1, PDF p.5)",
    ),
    PublishedBracket(
        "battery_energy_density", 90.0 * 3600.0, 350.0 * 3600.0,
        "Casalino 2019 set A 90.00 and set B 198.50 Wh/kg (Table 1, PDF p.6); "
        "Kwak 2018 cells 168.2-230.0 Wh/kg (Table 2, PDF p.5); Pavlov Rachov "
        "2013 Li-Po 130, Li-ion 220, Li-S 350 Wh/kg (Table 1, PDF p.5); Lee "
        "2021 325 Wh/kg (Table 2, PDF p.11)",
    ),
    PublishedBracket(
        "battery_structural_margin", 1.2, 1.2,
        "1.2 in every held set: Lee 2021 (PDF p.10-11), Kwak 2018 (Table 3, "
        "PDF p.9), Casalino 2019 (Eq. 17, PDF p.6), Solda & Lentini 2008 "
        "(PDF p.4), Pavlov Rachov 2013 (Table 2, PDF p.5)",
    ),
    PublishedBracket(
        "battery_discharge_efficiency", 0.925, 0.925,
        "No held source states a constant. Lee 2021 Eq. (38) (PDF p.10) and "
        "Kwak 2018 Eq. (4) (PDF p.5) give eta_E = 0.093 ln(t_b) + 0.3301; "
        "the only stated number is 0.925 at t_b = 600 s (Lee). The "
        "correlation gives 0.775 at 120 s and reaches 0.95 near 785 s",
    ),
    PublishedBracket(
        "motor_power_density", 875.0, 8200.0,
        "Lee 2021 0.875 kW/kg (Table 2, PDF p.11); Pavlov Rachov 2013 "
        "3.8 kW/kg (Table 1, PDF p.5); Kwak 2018 5.3 kW/kg (Table 3, PDF "
        "p.9), shaft output; Park 2023 8.20 kW/kg (Table 3, PDF p.6), "
        "basis unstated",
    ),
    PublishedBracket(
        "motor_efficiency", 0.80, 0.95,
        "Pavlov Rachov 2013 0.8 (Table 1, PDF p.5); Lee 2021 0.87 (Table 2, "
        "PDF p.11); Park 2023 0.90 (PDF p.10); Kwak 2018 0.95 (Table 3, "
        "PDF p.9)",
    ),
    PublishedBracket(
        "inverter_power_density", 26640.0, 60000.0,
        "Lee 2021 (PDF p.7, p.10-11), Kwak 2018 (Table 3, PDF p.9) and "
        "Pavlov Rachov 2013 (Table 1, PDF p.5): 60 kW/kg on inverter output; "
        "Park 2023 26.64 kW/kg, basis unstated (Table 3, PDF p.6)",
    ),
    PublishedBracket(
        "inverter_efficiency", 0.80, 0.85,
        "Lee 2021 (PDF p.10-11), Kwak 2018 (Table 3, PDF p.9), Pavlov "
        "Rachov 2013 (Table 1, PDF p.5): 0.85; Park 2023 0.80 (PDF p.11)",
    ),
)

#: The values the traditional ``raosim.pumps`` path has always used.
REPOSITORY_SCREENING_V1 = ElectricTechnologySet(
    id="repository_screening_v1",
    label="Repository screening assumption v1 (former raosim.pumps defaults)",
    basis_class="repository_screening_assumption",
    battery_power_density=3000.0,
    battery_energy_density=250.0 * 3600.0,
    battery_structural_margin=1.20,
    battery_discharge_efficiency=0.95,
    motor_power_density=2500.0,
    motor_efficiency=0.90,
    inverter_power_density=15000.0,
    inverter_efficiency=0.96,
    notes=(
        "Repository screening values with no single source. Battery power and "
        "energy density, motor density and efficiency lie inside the held "
        "published ranges and the 1.2 margin matches every held set; the "
        "inverter density (15 kW/kg) is BELOW and the inverter efficiency "
        "(0.96) ABOVE every held value; the constant discharge efficiency "
        "0.95 stands in for a burn-time correlation that reaches 0.95 only "
        "near 785 s. Replace with component data or a literature set."
    ),
    published_brackets=_SCREENING_BRACKETS,
)

#: Lee et al. 2021, Table 2: every battery, motor and inverter value read on
#: the page (pump efficiencies and margin are outside the set).
LEE2021_IJASS_ELECTRIC_PUMP = ElectricTechnologySet(
    id="lee2021_ijass_electric_pump",
    label=(
        "Lee et al. 2021, IJASS 22:94, 500 N LOX/RP-1 electric-pump case "
        "(Table 2)"
    ),
    basis_class="literature",
    battery_power_density=650.0,
    battery_energy_density=325.0 * 3600.0,
    battery_structural_margin=1.20,
    battery_discharge_efficiency=0.925,
    motor_power_density=875.0,
    motor_efficiency=0.87,
    inverter_power_density=60.0e3,
    inverter_efficiency=0.85,
    sources=(
        TechnologySource(
            _LEE_ID, _LEE, 11,
            "Table 2 Input parameter values: motor efficiency 87 %, power "
            "density 0.875 kW/kg; inverter efficiency 85 %, power density "
            "60 kW/kg; battery energy efficiency 92.5 %, power density "
            "0.650 kW/kg, energy density 325 W h/kg, structure margin 20 %",
        ),
        TechnologySource(
            _LEE_ID, _LEE, 7,
            "Sec. 2.1.4-2.1.5, Eqs. (26)-(29): motor density on motor output "
            "power; inverter density on inverter output = motor output / "
            "motor efficiency",
        ),
        TechnologySource(
            _LEE_ID, _LEE, 8,
            "Sec. 2.1.6, Eqs. (30)-(32): power- and energy-limited battery "
            "mass with structural margin kappa_b on both, m_b = MAX",
        ),
        TechnologySource(
            _LEE_ID, _LEE, 10,
            "Sec. 3.2 text and Eq. (38): eta_E = 0.093 ln(t_b) + 0.3301, "
            "0.925 at t_b = 600 s; inverter 60 kW/kg and 85 %; battery "
            "structure margin 20 %",
        ),
    ),
    notes=(
        "Commercial components (Vishan EC2060L-3660 BLDC motor, Amicell "
        "ABLP8474170H320 Li-polymer cell; Figs. 8-9, PDF p.11) chosen for a "
        "500 N, 20 bar, 600 s, 30 km design point (Table 1, PDF p.10). The "
        "battery energy efficiency is Eq. (38) at 600 s; at another burn "
        "time the same source gives a different value. Lee Table 4 (PDF "
        "p.12) motor 451.2 g and battery 985.6 g close with these values "
        "(tests/test_pumps.py)."
    ),
    discharge_efficiency_burn_time_s=600.0,
)

ELECTRIC_TECHNOLOGY_SETS: Mapping[str, ElectricTechnologySet] = MappingProxyType({
    technology.id: technology
    for technology in (REPOSITORY_SCREENING_V1, LEE2021_IJASS_ELECTRIC_PUMP)
})

DEFAULT_ELECTRIC_TECHNOLOGY = REPOSITORY_SCREENING_V1
DEFAULT_ELECTRIC_TECHNOLOGY_ID = DEFAULT_ELECTRIC_TECHNOLOGY.id


def electric_technology_ids(*, include_custom: bool = False) -> tuple[str, ...]:
    ids = tuple(ELECTRIC_TECHNOLOGY_SETS)
    return ids + ((CUSTOM_ELECTRIC_TECHNOLOGY_ID,) if include_custom else ())


def get_electric_technology(technology_id: str) -> ElectricTechnologySet:
    """Return a registered set; ``custom`` has no values and is refused."""

    try:
        return ELECTRIC_TECHNOLOGY_SETS[technology_id]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown electric technology {technology_id!r}; registered: "
            f"{', '.join(ELECTRIC_TECHNOLOGY_SETS)}"
            + (" ('custom' carries no values)"
               if technology_id == CUSTOM_ELECTRIC_TECHNOLOGY_ID else "")
        ) from None


class ElectricTechnologyValidityWarning(UserWarning):
    """A selected set is used outside the point its source states."""


def validate_mission_electric_fields(mission: Any) -> None:
    """``MissionSpec`` check: numeric fields agree with the declared set.

    A mission declaring ``custom`` must still carry physically valid values;
    one declaring a registered set must carry exactly that set's values, so a
    label can never describe numbers it does not hold.
    """

    declared = getattr(mission, "electric_technology", None)
    values = {name: getattr(mission, name) for name, _ in MISSION_FIELD_MAP}
    if declared == CUSTOM_ELECTRIC_TECHNOLOGY_ID:
        _validate_values(
            {field: values[mission] for mission, field in MISSION_FIELD_MAP},
            owner="MissionSpec (custom electric technology)",
        )
        return
    technology = get_electric_technology(declared)
    expected = technology.mission_fields()
    mismatched = [
        f"{name}={float(values[name])!r} (set: {expected[name]!r})"
        for name in expected
        if isinstance(values[name], bool)
        or not math.isclose(float(values[name]), expected[name],
                            rel_tol=1e-12, abs_tol=0.0)
    ]
    if mismatched:
        raise ValueError(
            f"MissionSpec electric fields disagree with electric_technology "
            f"{declared!r}: {', '.join(mismatched)}. Use "
            "MissionSpec.with_electric_technology(id) to select a set, or "
            f"electric_technology={CUSTOM_ELECTRIC_TECHNOLOGY_ID!r} for "
            "values that belong to no registered set"
        )
    valid_at = technology.discharge_efficiency_burn_time_s
    burn_time = getattr(mission, "burn_time", None)
    if (
        valid_at is not None and burn_time is not None
        and not math.isclose(float(burn_time), valid_at, rel_tol=0.01)
    ):
        # The source states its discharge efficiency only at this burn time
        # (Lee 2021 Eq. 38 is a burn-time correlation).  Warn rather than
        # refuse: the set stays power-limited over most burn times, where the
        # energy branch, and so this efficiency, does not set the mass.
        warnings.warn(
            f"electric technology {technology.id!r} states its discharge "
            f"efficiency ({technology.battery_discharge_efficiency:g}) only "
            f"at a {valid_at:g} s burn; this mission burns {float(burn_time):g} "
            "s, so the energy-limited battery branch is outside the source's "
            "stated point",
            ElectricTechnologyValidityWarning,
            stacklevel=3,
        )


def identify_electric_technology(values: Mapping[str, float]) -> str:
    """Id of the registered set whose screening values equal ``values``.

    ``values`` uses :data:`SCREENING_DEFAULT_KEYS` names.  Anything that
    matches no set exactly is ``custom``.
    """

    for technology in ELECTRIC_TECHNOLOGY_SETS.values():
        reference = technology.screening_defaults()
        try:
            if all(
                math.isclose(float(values[key]), reference[key],
                             rel_tol=1e-12, abs_tol=0.0)
                for key in SCREENING_DEFAULT_KEYS
            ):
                return technology.id
        except (KeyError, TypeError, ValueError):
            break
    return CUSTOM_ELECTRIC_TECHNOLOGY_ID


__all__ = [
    "CUSTOM_ELECTRIC_TECHNOLOGY_ID",
    "DEFAULT_ELECTRIC_TECHNOLOGY",
    "DEFAULT_ELECTRIC_TECHNOLOGY_ID",
    "ELECTRIC_BASIS_CLASSES",
    "ELECTRIC_TECHNOLOGY_SETS",
    "ElectricTechnologySet",
    "ElectricTechnologyValidityWarning",
    "LEE2021_IJASS_ELECTRIC_PUMP",
    "MISSION_FIELD_MAP",
    "PublishedBracket",
    "REPOSITORY_SCREENING_V1",
    "SCREENING_DEFAULT_KEYS",
    "TechnologySource",
    "electric_technology_ids",
    "get_electric_technology",
    "identify_electric_technology",
    "validate_mission_electric_fields",
]
