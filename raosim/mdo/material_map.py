"""The single typed mapper from a catalog material to traced MDO properties.

A regeneratively cooled wall is two different alloys doing two different jobs
-- NASA SP-8087 sec. 2.1.3.1: *"Hardenable materials often are used for jacket
designs, where, after brazing, the strength can be increased considerably by
agehardening"* -- so the liner and the structural closeout are selected
separately and mapped separately here.

The contract this module exists to enforce is **atomicity**.  Before it, a
host-side material label reached the traditional pipeline through
``MaterialSpec.from_catalog`` while the differentiable core kept flat scalar
defaults, so ``--material GRCop-84`` optimized a NARloy-Z-class liner: a 12
percent conductivity error, a 21 percent modulus error, and a 200 K error in
the allowable gas-side wall temperature.  That is the same failure class as the
O/F sentinel -- one concept with two definitions that silently disagree.

Therefore either **every** field a selection owns is resolved from one catalog
record, or the selection is rejected.  There is deliberately no partial
application and no per-field fallback to a class default: a half-applied
material is a mongrel alloy that exists in no catalog and matches no
qualification data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CLOSEOUT_MISSION_FIELDS",
    "LINER_MISSION_FIELDS",
    "MaterialCoverageError",
    "ResolvedMaterialSelection",
    "closeout_mission_fields",
    "liner_mission_fields",
    "resolve_material_selection",
]


class MaterialCoverageError(ValueError):
    """A selected material cannot populate every property it owns."""


#: Traced ``MissionSpec`` fields owned by the liner selection.
LINER_MISSION_FIELDS: tuple[str, ...] = (
    "k_wall",
    "rho_wall",
    "liner_E",
    "liner_alpha",
    "liner_poisson",
    "liner_sigma_allow",
    "liner_T_wg_max",
)

#: Traced ``MissionSpec`` fields owned by the structural closeout selection.
CLOSEOUT_MISSION_FIELDS: tuple[str, ...] = (
    "rho_closeout",
    "closeout_sigma_yield",
    "closeout_E",
    "closeout_poisson",
)

#: Catalog attributes required before either role may be applied.
_REQUIRED_CATALOG_ATTRS: tuple[str, ...] = (
    "conductivity",
    "yield_strength",
    "max_temperature",
    "density",
    "elastic_modulus",
    "thermal_expansion",
    "poisson_ratio",
)


@dataclass(frozen=True)
class ResolvedMaterialSelection:
    """One atomically resolved liner/closeout pair plus its provenance."""

    liner_name: str
    closeout_name: str
    liner_source: str
    closeout_source: str
    fields: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "liner_material": self.liner_name,
            "closeout_material": self.closeout_name,
            "liner_source": self.liner_source,
            "closeout_source": self.closeout_source,
            "fields": dict(self.fields),
        }


def _catalog_material(material: Any) -> Any:
    """Accept a catalog name, a ``MaterialProperties``, or a ``MaterialSpec``.

    A name is resolved through the catalog; an object is used exactly as
    given.  Re-resolving a supplied object by its ``name`` would silently
    discard a caller's deliberate property override and substitute the catalog
    record -- the same silent-fallback failure this module exists to prevent.
    Completeness of a supplied object is enforced by validation instead.
    """

    from raosim.materials import get_material

    if material is None:
        raise MaterialCoverageError("no material was supplied")
    if isinstance(material, str):
        return get_material(material)
    return material


def _positive(material: Any, attr: str, role: str) -> float:
    value = getattr(material, attr, None)
    name = getattr(material, "name", material)
    if value is None:
        raise MaterialCoverageError(
            f"{role} material {name!r} has no {attr}; every traced "
            f"{role} property must come from one catalog record, so the "
            "selection is rejected rather than partially applied"
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterialCoverageError(
            f"{role} material {name!r} has a non-numeric {attr}: {value!r}"
        ) from exc
    if not math.isfinite(out) or out <= 0.0:
        raise MaterialCoverageError(
            f"{role} material {name!r} has a non-physical {attr}: {out!r}"
        )
    return out


def _validate_coverage(material: Any, role: str) -> None:
    for attr in _REQUIRED_CATALOG_ATTRS:
        _positive(material, attr, role)


def liner_mission_fields(
    material: Any, *, structural_fos: float = 1.5
) -> dict[str, float]:
    """Every traced liner property, or an exception.

    ``liner_sigma_allow`` is the **post**-factor-of-safety allowable the MDO
    inequality consumes, so the catalog yield strength is divided once here.
    ``liner_structural_fos`` is carried alongside it so the traditional solver
    can recover the yield strength without applying the factor a second time.

    ``max_temperature`` maps to ``liner_T_wg_max`` because the catalog defines
    it as the maximum allowable *gas-side wall* temperature set by
    oxidation/creep/strength -- not the melting point, which it reports
    separately.
    """

    mat = _catalog_material(material)
    _validate_coverage(mat, "liner")
    fos = float(structural_fos)
    if not math.isfinite(fos) or fos <= 0.0:
        raise MaterialCoverageError(
            f"liner structural factor of safety must be finite and positive; "
            f"got {structural_fos!r}"
        )
    return {
        "k_wall": _positive(mat, "conductivity", "liner"),
        "rho_wall": _positive(mat, "density", "liner"),
        "liner_E": _positive(mat, "elastic_modulus", "liner"),
        "liner_alpha": _positive(mat, "thermal_expansion", "liner"),
        "liner_poisson": _positive(mat, "poisson_ratio", "liner"),
        "liner_sigma_allow": _positive(mat, "yield_strength", "liner") / fos,
        "liner_structural_fos": fos,
        "liner_T_wg_max": _positive(mat, "max_temperature", "liner"),
    }


def closeout_mission_fields(material: Any) -> dict[str, float]:
    """Every traced structural-closeout property, or an exception.

    ``closeout_sigma_yield`` is the raw catalog yield: the jacket hoop sizing
    applies ``closeout_structural_fos`` itself (SP-8087 sec. 2.1.3 quotes yield
    factors of 1.0--1.32).  That asymmetry with the liner is the existing
    contract and is preserved deliberately, so retargeting a material cannot
    change how many times a factor of safety is applied.
    """

    mat = _catalog_material(material)
    _validate_coverage(mat, "closeout")
    return {
        "rho_closeout": _positive(mat, "density", "closeout"),
        "closeout_sigma_yield": _positive(mat, "yield_strength", "closeout"),
        "closeout_E": _positive(mat, "elastic_modulus", "closeout"),
        "closeout_poisson": _positive(mat, "poisson_ratio", "closeout"),
    }


def resolve_material_selection(
    *,
    liner: Any,
    closeout: Any,
    liner_structural_fos: float = 1.5,
) -> ResolvedMaterialSelection:
    """Resolve both roles together so a mission is never half-retargeted."""

    liner_mat = _catalog_material(liner)
    closeout_mat = _catalog_material(closeout)
    fields: dict[str, Any] = {}
    fields.update(liner_mission_fields(
        liner_mat, structural_fos=liner_structural_fos
    ))
    fields.update(closeout_mission_fields(closeout_mat))
    liner_name = str(getattr(liner_mat, "name", liner))
    closeout_name = str(getattr(closeout_mat, "name", closeout))
    fields["liner_material_name"] = liner_name
    fields["closeout_material_name"] = closeout_name
    return ResolvedMaterialSelection(
        liner_name=liner_name,
        closeout_name=closeout_name,
        liner_source=str(getattr(liner_mat, "source", "") or ""),
        closeout_source=str(getattr(closeout_mat, "source", "") or ""),
        fields=fields,
    )
