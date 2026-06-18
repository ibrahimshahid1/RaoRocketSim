"""materials.py — thrust-chamber / nozzle construction materials catalog.

A regeneratively cooled wall is a *two-fluid heat exchanger that also
carries pressure and thermal stress*, so the metal it is built from sets
the whole problem, not just one number.  The wall properties enter every
physical concept the design tool screens:

* **Channel sizing (thermal).**  The gas-side wall temperature is driven
  by the conduction term ``t_w / k_w`` in the series resistance circuit
  (NASA SP-125, Huzel & Huang, eqs. 4-19 and 4-22; printed p. 104):

      q = (k/t)(T_wg - T_wc),   H = 1 / (1/h_gc + t/k + 1/h_c)

  A high-conductivity copper liner runs a *cool* gas-side wall at a given
  flux; a low-k superalloy runs a much hotter one.  This is why the
  channel auto-sizer (:func:`raosim.thermal_design.size_cooling_channels`)
  takes ``wall_k`` — and why it must come from the chosen material, not a
  disconnected flag.

* **Service temperature limit.**  SP-125 (printed p. 105) states that for
  the metals "commonly used in thrust chamber walls, such as stainless
  steel, nickel, and Inconel, the limiting hot-gas-side wall temperature
  is around 1500deg-1800deg F" (1089-1255 K).  Copper alloys conduct far
  better but soften far earlier, so their service limit is set by creep /
  strength, not melting.  The cooling *margin* is (limit / peak wall T),
  so the limit is a per-material number.

* **Structural / thermal stress.**  SP-125's coaxial-shell wall design
  (eq. 4-31, printed p. 109) sizes the inner liner against the combined
  pressure-differential and thermal stress

      S_c = (p_co - p_g) R / t  +  E a q t / (2 (1 - v) k)

  which needs the elastic modulus ``E``, thermal-expansion coefficient
  ``a``, Poisson ratio ``v`` and conductivity ``k`` of the wall — see
  :func:`raosim.physics.coaxial_shell_wall_stress`.  The longitudinal
  thermal stress S_l = E a dT (eq. 4-28) and the outer-jacket hoop stress
  use the same properties.

Provenance of the numbers
-------------------------
The *relations* above are all grounded in the local literature
(``propulsion_texts/19710019929.pdf`` = NASA SP-125).  SP-125 also names
the classic wall materials (stainless steel, nickel, Inconel; sample
calculation 4-4 picks Inconel X for an RP-1-cooled tube) and gives the
1500deg-1800deg F class limit, so the superalloy/steel entries are grounded
directly.  SP-125 predates the modern copper combustion-chamber alloys
(NARloy-Z, GRCop-84, CuCrZr), so their property values are representative
handbook / NASA-Glenn values from each alloy's own literature and are
flagged as such in ``source``; they are screening-grade, not a heat of
certified material.  All values are single representative points near the
service temperature — temperature-dependent k(T), S_y(T) tables are a
documented future extension (see JAX_DIFFERENTIABLE_PLAN.md).

Units are SI throughout: conductivity W/(m.K), strength/modulus Pa,
temperature K, density kg/m^3, expansion 1/K, heat flux W/m^2.
"""

from __future__ import annotations

from dataclasses import dataclass


# Material categories — the regen wall is usually a high-conductivity
# liner backed by a strong structural jacket (SP-125 coaxial-shell /
# tube-and-jacket construction).  The category tells the design which
# role a material is meant to play.
CATEGORY_COPPER_LINER = "copper_liner"
CATEGORY_SUPERALLOY = "superalloy"
CATEGORY_STEEL = "steel"
CATEGORY_NICKEL = "nickel"


@dataclass(frozen=True)
class MaterialProperties:
    """One construction material, with the properties every screen needs.

    ``conductivity``, ``elastic_modulus``, ``thermal_expansion``,
    ``poisson_ratio`` and ``yield_strength`` are single representative
    points near the service temperature (real design uses T-dependent
    tables; that is a future extension).  ``max_temperature`` is the
    maximum allowable *gas-side wall* temperature (the cooling margin is
    ``max_temperature / peak_wall_T``), set by oxidation/creep/strength —
    NOT the melting point, which is reported separately.
    """

    name: str
    category: str
    conductivity: float          # k   [W/(m.K)]  near service temperature
    yield_strength: float        # S_y [Pa]
    max_temperature: float       # max allowable gas-side wall T [K]
    density: float               # rho [kg/m^3]
    elastic_modulus: float       # E   [Pa]
    thermal_expansion: float     # a   [1/K]
    poisson_ratio: float         # v   [-]
    max_heat_flux: float         # screening flux capability [W/m^2]
    melting_point: float         # [K]  (reference only)
    source: str                  # provenance / citation
    # Optional strain-life data.  There are deliberately NO generic
    # defaults: SP-125 identifies low-cycle thermal fatigue as a governing
    # failure mode, but it does not provide alloy/temperature-specific
    # Coffin-Manson coefficients.  A cycle-life number is only evaluated
    # when all four coefficients and an explicit source are supplied.
    # ``fatigue_design_qualified`` must additionally be True before that
    # result is allowed to gate feasibility.
    fatigue_strength_coeff: float | None = None   # sigma_f' [Pa]
    fatigue_strength_exp: float | None = None     # b  (Basquin, < 0)
    fatigue_ductility_coeff: float | None = None  # eps_f' [-]
    fatigue_ductility_exp: float | None = None    # c  (Coffin-Manson, < 0)
    fatigue_source: str | None = None
    fatigue_data_temperature: float | None = None  # [K]
    fatigue_design_qualified: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def is_liner(self) -> bool:
        """True for the high-conductivity copper liner alloys."""
        return self.category == CATEGORY_COPPER_LINER


# --------------------------------------------------------------------------- #
#  The catalog.  Copper liners first (high k, low service T), then the         #
#  structural superalloys / steels (low k, high service T).  The contrast      #
#  IS the design trade: copper cools the wall but can't take the heat;         #
#  superalloys take the heat but conduct poorly so the wall runs hot.          #
# --------------------------------------------------------------------------- #
_CATALOG: tuple[MaterialProperties, ...] = (
    # ---- high-conductivity copper liners --------------------------------- #
    MaterialProperties(
        name="OFHC Copper",
        category=CATEGORY_COPPER_LINER,
        conductivity=365.0, yield_strength=70e6, max_temperature=700.0,
        density=8940.0, elastic_modulus=117e9, thermal_expansion=17.0e-6,
        poisson_ratio=0.34, max_heat_flux=120e6, melting_point=1356.0,
        source="OFHC C10200 handbook (high k, low yield -> needs jacket "
               "support per SP-125 coaxial-shell design)",
        aliases=("copper", "cu", "ofhc", "c10200"),
    ),
    MaterialProperties(
        name="NARloy-Z",
        category=CATEGORY_COPPER_LINER,
        conductivity=320.0, yield_strength=125e6, max_temperature=810.0,
        density=9130.0, elastic_modulus=108e9, thermal_expansion=18.0e-6,
        poisson_ratio=0.34, max_heat_flux=160e6, melting_point=1340.0,
        source="Cu-3Ag-0.5Zr; SSME main combustion chamber liner "
               "(NASA literature; not in SP-125)",
        aliases=("narloy", "narloyz", "narloy_z", "cu-ag-zr"),
    ),
    MaterialProperties(
        name="GRCop-84",
        category=CATEGORY_COPPER_LINER,
        conductivity=285.0, yield_strength=186e6, max_temperature=1000.0,
        density=8756.0, elastic_modulus=140e9, thermal_expansion=16.5e-6,
        poisson_ratio=0.33, max_heat_flux=150e6, melting_point=1370.0,
        source="Cu-8Cr-4Nb (at.%); NASA Glenn (Ellis) AM chamber alloy; "
               "higher service T than NARloy-Z (not in SP-125)",
        aliases=("grcop", "grcop84", "grcop_84", "cu-cr-nb"),
    ),
    MaterialProperties(
        name="CuCrZr",
        category=CATEGORY_COPPER_LINER,
        conductivity=320.0, yield_strength=300e6, max_temperature=770.0,
        density=8900.0, elastic_modulus=128e9, thermal_expansion=17.0e-6,
        poisson_ratio=0.34, max_heat_flux=120e6, melting_point=1350.0,
        source="Cu-Cr-Zr (C18150); European regen liners / ITER "
               "(not in SP-125)",
        aliases=("cucrzr", "cu-cr-zr", "c18150"),
    ),
    # ---- structural superalloys / steels (jacket, high-strength tubes) --- #
    MaterialProperties(
        name="Inconel 718",
        category=CATEGORY_SUPERALLOY,
        conductivity=15.0, yield_strength=1035e6, max_temperature=1255.0,
        density=8190.0, elastic_modulus=200e9, thermal_expansion=13.0e-6,
        poisson_ratio=0.29, max_heat_flux=30e6, melting_point=1610.0,
        source="Ni-base superalloy; SP-125 names Inconel as a wall metal "
               "(1500-1800F class limit, printed p.105). RT yield; "
               "knock down ~10-15% at 650C.",
        aliases=("inconel", "inconel718", "in718", "718"),
    ),
    MaterialProperties(
        name="Inconel X-750",
        category=CATEGORY_SUPERALLOY,
        conductivity=12.0, yield_strength=815e6, max_temperature=1200.0,
        density=8280.0, elastic_modulus=214e9, thermal_expansion=12.6e-6,
        poisson_ratio=0.29, max_heat_flux=28e6, melting_point=1665.0,
        source="SP-125 sample calculation 4-4 picks Inconel X for the "
               "RP-1-cooled A-1 tube wall (printed p.109)",
        aliases=("inconelx", "inconel-x", "x750", "x-750", "inconel_x750"),
    ),
    MaterialProperties(
        name="Stainless 316L",
        category=CATEGORY_STEEL,
        conductivity=16.3, yield_strength=290e6, max_temperature=1100.0,
        density=8000.0, elastic_modulus=193e9, thermal_expansion=16.0e-6,
        poisson_ratio=0.27, max_heat_flux=16e6, melting_point=1644.0,
        source="SP-125 names stainless steel as a common wall metal "
               "(printed p.105); 316L handbook properties",
        aliases=("316", "316l", "ss316", "ss316l", "stainless", "steel"),
    ),
    MaterialProperties(
        name="Nickel 200",
        category=CATEGORY_NICKEL,
        conductivity=70.0, yield_strength=148e6, max_temperature=870.0,
        density=8890.0, elastic_modulus=207e9, thermal_expansion=13.3e-6,
        poisson_ratio=0.31, max_heat_flux=40e6, melting_point=1728.0,
        source="SP-125 names nickel as a common wall metal (printed "
               "p.105); Nickel 200 handbook properties",
        aliases=("nickel", "ni", "nickel200", "ni200"),
    ),
)


# name (lowercased) + every alias -> MaterialProperties
_BY_KEY: dict[str, MaterialProperties] = {}
for _m in _CATALOG:
    _BY_KEY[_m.name.lower()] = _m
    for _a in _m.aliases:
        _BY_KEY[_a.lower()] = _m


def list_materials() -> list[MaterialProperties]:
    """All catalog entries, copper liners first then structural metals."""
    return list(_CATALOG)


def material_names() -> list[str]:
    """Canonical display names of every catalog material."""
    return [m.name for m in _CATALOG]


def get_material(name: str) -> MaterialProperties:
    """Look up a material by canonical name or any alias (case-insensitive).

    Raises ``KeyError`` with the available names if ``name`` is unknown.
    """
    key = str(name).strip().lower().replace(" ", "")
    # try the raw key, then a space-tolerant form of every catalog key
    if key in _BY_KEY:
        return _BY_KEY[key]
    for k, m in _BY_KEY.items():
        if k.replace(" ", "").replace("-", "") == key.replace("-", ""):
            return m
    raise KeyError(
        f"unknown material {name!r}; available: {', '.join(material_names())}"
    )


def material_table() -> str:
    """A formatted catalog table for the CLI ``--list-materials`` output."""
    rows = [
        ("MATERIAL", "CATEGORY", "k", "S_y", "T_max", "rho", "ROLE"),
        ("", "", "W/m.K", "MPa", "K", "kg/m3", ""),
    ]
    for m in _CATALOG:
        rows.append((
            m.name, m.category.replace("_", " "),
            f"{m.conductivity:g}", f"{m.yield_strength/1e6:g}",
            f"{m.max_temperature:g}", f"{m.density:g}",
            "liner" if m.is_liner else "jacket/structure",
        ))
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 1:
            lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    return "\n".join(lines)
