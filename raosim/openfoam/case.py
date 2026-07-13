"""Version-pinned OpenFOAM VOF case generation for movable-pintle studies.

This module deliberately separates two stages in Radhakrishnan et al. (2018):

* Section 2.3 used VOF with a *water-only* mass-flow inlet to obtain liquid
  sheet thickness.  The surrounding phase was initially quiescent air.
* Section 2.4 used the 3.03 g/s annular-air stream as the Eulerian carrier for
  the later Lagrangian droplet calculation.

The first template implemented here is a reduced external-gap wedge.  It maps
the mechanical opening to a radial water boundary and is useful for checking
interface transport and external sheet evolution.  It does not contain the
paper's internal center-gap turning passage, so it cannot claim to predict the
sheet at formation.  That limitation is machine-readable in every manifest.

OpenFOAM is an external solver.  Builders in this module are pure and
deterministic; :func:`write_openfoam_case` only writes text files and never
spawns a process.  The target is OpenFOAM Foundation v13 at patch/tag
``20260624`` using ``foamRun`` with ``solver incompressibleVoF``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from raosim.spray.benchmarks import load_spray_benchmark


OPENFOAM_DISTRIBUTION = "OpenFOAM Foundation"
OPENFOAM_MAJOR_VERSION = 13
OPENFOAM_PATCH_TAG = "20260624"
OPENFOAM_SOLVER = "incompressibleVoF"
OPENFOAM_RUNNER = "foamRun"
TEMPLATE_ID = "lrekit.openfoam13.radial_sheet_external_wedge"
TEMPLATE_VERSION = 1
MANIFEST_NAME = "raosim_openfoam_manifest.json"

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SAFE_ARTIFACTS = {
    "0/U",
    "0/alpha.water",
    "0/epsilon",
    "0/k",
    "0/nut",
    "0/p_rgh",
    "Allclean",
    "Allrun",
    "README.md",
    "constant/g",
    "constant/momentumTransport",
    "constant/phaseProperties",
    "constant/physicalProperties.air",
    "constant/physicalProperties.water",
    "system/blockMeshDict",
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
}


class OpenFOAMExportError(ValueError):
    """Raised when a case input or filesystem destination is unsafe."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise OpenFOAMExportError(f"{name} must be a finite number, not bool")
    value = float(value)
    if not math.isfinite(value):
        raise OpenFOAMExportError(f"{name} must be finite")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise OpenFOAMExportError(f"{name} must be > 0")
    return value


def _fraction(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0.0 < value < 1.0:
        raise OpenFOAMExportError(f"{name} must be in (0, 1)")
    return value


def _count(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpenFOAMExportError(f"{name} must be an integer >= {minimum}")
    return value


def _fmt(value: float) -> str:
    value = _finite("OpenFOAM scalar", value)
    if value == 0.0:
        return "0"
    return format(value, ".12g")


def _text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value if value.endswith("\n") else value + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class OpenFOAMFluidProperties:
    """Constant incompressible properties passed to OpenFOAM in SI units."""

    name: str
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    temperature_k: float
    absolute_pressure_pa: float
    provenance: str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        provenance = str(self.provenance).strip()
        if name not in {"water", "air"}:
            raise OpenFOAMExportError("fluid name must be 'water' or 'air'")
        if not provenance or any(c in provenance for c in "\r\n\x00"):
            raise OpenFOAMExportError("fluid provenance must be one safe line")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "density_kg_m3", _positive("density_kg_m3", self.density_kg_m3)
        )
        object.__setattr__(
            self,
            "dynamic_viscosity_pa_s",
            _positive("dynamic_viscosity_pa_s", self.dynamic_viscosity_pa_s),
        )
        object.__setattr__(
            self, "temperature_k", _positive("temperature_k", self.temperature_k)
        )
        object.__setattr__(
            self,
            "absolute_pressure_pa",
            _positive("absolute_pressure_pa", self.absolute_pressure_pa),
        )
        object.__setattr__(self, "provenance", provenance)

    @property
    def kinematic_viscosity_m2_s(self) -> float:
        return self.dynamic_viscosity_pa_s / self.density_kg_m3

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "density_kg_m3": self.density_kg_m3,
            "dynamic_viscosity_pa_s": self.dynamic_viscosity_pa_s,
            "kinematic_viscosity_m2_s": self.kinematic_viscosity_m2_s,
            "temperature_k": self.temperature_k,
            "absolute_pressure_pa": self.absolute_pressure_pa,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class MovablePintleVOFGeometry:
    """Full literature geometry contract, including dimensions not yet meshed."""

    post_diameter_m: float
    center_gap_diameter_m: float
    pintle_rod_diameter_m: float
    pintle_tip_diameter_m: float
    annular_gap_thickness_m: float
    post_angle_deg: float
    pintle_tip_angle_deg: float
    pintle_tip_thickness_m: float
    post_recess_length_m: float
    post_thickness_m: float
    opening_distance_m: float
    axial_domain_length_m: float
    radial_domain_radius_m: float

    def __post_init__(self) -> None:
        for name in (
            "post_diameter_m",
            "center_gap_diameter_m",
            "pintle_rod_diameter_m",
            "pintle_tip_diameter_m",
            "annular_gap_thickness_m",
            "pintle_tip_thickness_m",
            "post_recess_length_m",
            "post_thickness_m",
            "opening_distance_m",
            "axial_domain_length_m",
            "radial_domain_radius_m",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("post_angle_deg", "pintle_tip_angle_deg"):
            angle = _finite(name, getattr(self, name))
            if not 0.0 < angle < 90.0:
                raise OpenFOAMExportError(f"{name} must be in (0, 90) deg")
            object.__setattr__(self, name, angle)
        if not self.pintle_rod_diameter_m < self.center_gap_diameter_m:
            raise OpenFOAMExportError(
                "pintle rod diameter must be smaller than center-gap diameter"
            )
        if self.center_gap_diameter_m >= self.post_diameter_m:
            raise OpenFOAMExportError(
                "center-gap diameter must be smaller than post diameter"
            )
        if self.opening_distance_m >= self.pintle_tip_thickness_m:
            raise OpenFOAMExportError(
                "opening distance must be smaller than pintle-tip thickness"
            )
        if self.sheet_exit_radius_m >= self.radial_domain_radius_m:
            raise OpenFOAMExportError(
                "radial domain must extend beyond the pintle-tip radius"
            )
        if (
            self.post_recess_length_m + self.opening_distance_m
            >= self.axial_domain_length_m
        ):
            raise OpenFOAMExportError(
                "opening must lie strictly inside the axial domain"
            )

    @property
    def sheet_exit_radius_m(self) -> float:
        return 0.5 * self.pintle_tip_diameter_m

    @property
    def liquid_center_gap_area_m2(self) -> float:
        return 0.25 * math.pi * (
            self.center_gap_diameter_m**2 - self.pintle_rod_diameter_m**2
        )

    @property
    def gas_annulus_area_m2(self) -> float:
        outer = self.post_diameter_m + 2.0 * self.annular_gap_thickness_m
        return 0.25 * math.pi * (outer**2 - self.post_diameter_m**2)

    @property
    def external_sheet_inlet_area_360_m2(self) -> float:
        return (
            2.0
            * math.pi
            * self.sheet_exit_radius_m
            * self.opening_distance_m
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_diameter_m": self.post_diameter_m,
            "center_gap_diameter_m": self.center_gap_diameter_m,
            "pintle_rod_diameter_m": self.pintle_rod_diameter_m,
            "pintle_tip_diameter_m": self.pintle_tip_diameter_m,
            "annular_gap_thickness_m": self.annular_gap_thickness_m,
            "post_angle_deg": self.post_angle_deg,
            "pintle_tip_angle_deg": self.pintle_tip_angle_deg,
            "pintle_tip_thickness_m": self.pintle_tip_thickness_m,
            "post_recess_length_m": self.post_recess_length_m,
            "post_thickness_m": self.post_thickness_m,
            "opening_distance_m": self.opening_distance_m,
            "axial_domain_length_m": self.axial_domain_length_m,
            "radial_domain_radius_m": self.radial_domain_radius_m,
            "derived": {
                "sheet_exit_radius_m": self.sheet_exit_radius_m,
                "liquid_center_gap_area_m2": self.liquid_center_gap_area_m2,
                "gas_annulus_area_m2": self.gas_annulus_area_m2,
                "external_sheet_inlet_area_360_m2": (
                    self.external_sheet_inlet_area_360_m2
                ),
            },
        }


@dataclass(frozen=True)
class OpenFOAMWedgeControls:
    """Numerical and property inputs for the reduced external-gap wedge."""

    wedge_angle_deg: float = 5.0
    upstream_cells: int = 30
    opening_cells: int = 6
    downstream_cells: int = 220
    radial_cells: int = 240
    radial_expansion_ratio: float = 10.0
    max_total_cells: int = 1_000_000
    initial_delta_t_s: float = 1.0e-7
    max_delta_t_s: float = 5.0e-6
    end_time_s: float = 2.0e-2
    write_interval_s: float = 5.0e-4
    max_courant: float = 0.5
    max_alpha_courant: float = 0.25
    surface_tension_n_m: float = 0.07197
    turbulence_intensity: float = 0.05
    turbulence_length_scale_m: float = 5.0e-4
    water: OpenFOAMFluidProperties = OpenFOAMFluidProperties(
        name="water",
        density_kg_m3=998.2,
        dynamic_viscosity_pa_s=1.002e-3,
        temperature_k=300.0,
        absolute_pressure_pa=101325.0,
        provenance="repository ambient-property assumption; publication omits values",
    )
    air: OpenFOAMFluidProperties = OpenFOAMFluidProperties(
        name="air",
        density_kg_m3=1.1766,
        dynamic_viscosity_pa_s=1.846e-5,
        temperature_k=300.0,
        absolute_pressure_pa=101325.0,
        provenance="repository ambient-property assumption; publication omits values",
    )

    def __post_init__(self) -> None:
        angle = _finite("wedge_angle_deg", self.wedge_angle_deg)
        if not 0.0 < angle <= 10.0:
            raise OpenFOAMExportError("wedge_angle_deg must be in (0, 10]")
        object.__setattr__(self, "wedge_angle_deg", angle)
        for name, minimum in (
            ("upstream_cells", 2),
            ("opening_cells", 4),
            ("downstream_cells", 2),
            ("radial_cells", 4),
            ("max_total_cells", 1),
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name), minimum=minimum)
            )
        ratio = _positive("radial_expansion_ratio", self.radial_expansion_ratio)
        if ratio > 20.0:
            raise OpenFOAMExportError("radial_expansion_ratio must be <= 20")
        object.__setattr__(self, "radial_expansion_ratio", ratio)
        for name in (
            "initial_delta_t_s",
            "max_delta_t_s",
            "end_time_s",
            "write_interval_s",
            "surface_tension_n_m",
            "turbulence_length_scale_m",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        object.__setattr__(
            self,
            "turbulence_intensity",
            _fraction("turbulence_intensity", self.turbulence_intensity),
        )
        for name in ("max_courant", "max_alpha_courant"):
            value = _positive(name, getattr(self, name))
            if value > 0.5:
                raise OpenFOAMExportError(f"{name} must be <= 0.5")
            object.__setattr__(self, name, value)
        if self.max_alpha_courant > 0.25:
            raise OpenFOAMExportError("max_alpha_courant must be <= 0.25")
        if self.initial_delta_t_s > self.max_delta_t_s:
            raise OpenFOAMExportError(
                "initial_delta_t_s must be <= max_delta_t_s"
            )
        if self.write_interval_s > self.end_time_s:
            raise OpenFOAMExportError("write_interval_s must be <= end_time_s")
        if self.total_cells > self.max_total_cells:
            raise OpenFOAMExportError(
                f"requested {self.total_cells} cells exceeds max_total_cells"
            )

    @property
    def total_cells(self) -> int:
        return (
            self.upstream_cells + self.opening_cells + self.downstream_cells
        ) * self.radial_cells

    @property
    def wedge_fraction(self) -> float:
        return math.radians(self.wedge_angle_deg) / (2.0 * math.pi)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wedge_angle_deg": self.wedge_angle_deg,
            "azimuthal_cells": 1,
            "upstream_cells": self.upstream_cells,
            "opening_cells": self.opening_cells,
            "downstream_cells": self.downstream_cells,
            "radial_cells": self.radial_cells,
            "radial_expansion_ratio": self.radial_expansion_ratio,
            "total_cells": self.total_cells,
            "max_total_cells": self.max_total_cells,
            "initial_delta_t_s": self.initial_delta_t_s,
            "max_delta_t_s": self.max_delta_t_s,
            "end_time_s": self.end_time_s,
            "write_interval_s": self.write_interval_s,
            "max_courant": self.max_courant,
            "max_alpha_courant": self.max_alpha_courant,
            "surface_tension_n_m": self.surface_tension_n_m,
            "turbulence_model": "RAS.realizableKE",
            "turbulence_intensity": self.turbulence_intensity,
            "turbulence_length_scale_m": self.turbulence_length_scale_m,
            "water": self.water.as_dict(),
            "air": self.air.as_dict(),
        }


@dataclass(frozen=True)
class OpenFOAMCasePackage:
    """In-memory deterministic case files and immutable manifest."""

    files: Mapping[str, str]
    manifest: Mapping[str, Any]
    fingerprint: str

    def __post_init__(self) -> None:
        files = {str(path): _text(str(content)) for path, content in self.files.items()}
        expected = _SAFE_ARTIFACTS | {MANIFEST_NAME}
        if set(files) != expected:
            missing = sorted(expected - set(files))
            extra = sorted(set(files) - expected)
            raise OpenFOAMExportError(
                f"case artifact set mismatch; missing={missing}, extra={extra}"
            )
        for path, content in files.items():
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts or "\\" in path:
                raise OpenFOAMExportError(f"unsafe case artifact path: {path}")
            if "\x00" in content or "\r" in content:
                raise OpenFOAMExportError(f"unsafe text content in {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise OpenFOAMExportError("case fingerprint must be SHA-256 hex")
        object.__setattr__(self, "files", MappingProxyType(files))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class OpenFOAMCaseWriteResult:
    destination: Path
    fingerprint: str
    written: bool


def build_radhakrishnan2018_sheet_vof_case(
    row_case_id: str = "case_1",
    *,
    controls: OpenFOAMWedgeControls | None = None,
) -> OpenFOAMCasePackage:
    """Build one reduced, water-only external-gap VOF screening wedge.

    ``sheet_thickness_full_mm`` and ``water_velocity_m_s`` from the fixture are
    author VOF outputs.  They are recorded as targets and are never prescribed
    to the generated fields.  The mechanical ``lopen_mm`` is the inlet opening.
    """

    if not _CASE_ID_RE.fullmatch(str(row_case_id)):
        raise OpenFOAMExportError(f"invalid row_case_id: {row_case_id!r}")
    dataset = load_spray_benchmark("radhakrishnan2018_water_air")
    row = dataset.row(row_case_id)
    controls = controls or OpenFOAMWedgeControls()
    source_geometry = dataset.manifest["injector_geometry"]
    geometry = MovablePintleVOFGeometry(
        post_diameter_m=float(source_geometry["post_diameter_mm"]) * 1.0e-3,
        center_gap_diameter_m=(
            float(source_geometry["center_gap_diameter_mm"]) * 1.0e-3
        ),
        pintle_rod_diameter_m=(
            float(source_geometry["pintle_rod_diameter_mm"]) * 1.0e-3
        ),
        pintle_tip_diameter_m=(
            float(source_geometry["pintle_tip_diameter_mm"]) * 1.0e-3
        ),
        annular_gap_thickness_m=(
            float(source_geometry["annular_gap_thickness_mm"]) * 1.0e-3
        ),
        post_angle_deg=float(source_geometry["post_angle_deg"]),
        pintle_tip_angle_deg=float(source_geometry["pintle_tip_angle_deg"]),
        pintle_tip_thickness_m=(
            float(source_geometry["pintle_tip_thickness_mm"]) * 1.0e-3
        ),
        post_recess_length_m=(
            float(source_geometry["post_recess_length_mm"]) * 1.0e-3
        ),
        post_thickness_m=(
            float(source_geometry["post_thickness_mm"]) * 1.0e-3
        ),
        opening_distance_m=float(row["lopen_mm"]) * 1.0e-3,
        axial_domain_length_m=80.0e-3,
        radial_domain_radius_m=120.0e-3,
    )
    liquid_mass_flow_kg_s = float(row["mdot_water_g_s"]) * 1.0e-3
    wedge_mass_flow = liquid_mass_flow_kg_s * controls.wedge_fraction
    liquid_speed = (
        liquid_mass_flow_kg_s
        / controls.water.density_kg_m3
        / geometry.external_sheet_inlet_area_360_m2
    )
    inlet_area_wedge = (
        geometry.external_sheet_inlet_area_360_m2 * controls.wedge_fraction
    )
    achieved_wedge_mass_flow = (
        controls.water.density_kg_m3 * inlet_area_wedge * liquid_speed
    )
    mass_residual = abs(achieved_wedge_mass_flow - wedge_mass_flow) / wedge_mass_flow
    if mass_residual > 1.0e-12:
        raise OpenFOAMExportError("analytical wedge liquid-mass closure failed")

    turbulent_k = 1.5 * (controls.turbulence_intensity * liquid_speed) ** 2
    turbulent_epsilon = (
        0.09**0.75
        * turbulent_k**1.5
        / controls.turbulence_length_scale_m
    )
    ambient_k = max(turbulent_k * 1.0e-4, 1.0e-10)
    ambient_epsilon = max(turbulent_epsilon * 1.0e-4, 1.0e-10)

    render_state = {
        "geometry": geometry,
        "controls": controls,
        "liquid_speed_m_s": liquid_speed,
        "turbulent_k_m2_s2": turbulent_k,
        "turbulent_epsilon_m2_s3": turbulent_epsilon,
        "ambient_k_m2_s2": ambient_k,
        "ambient_epsilon_m2_s3": ambient_epsilon,
    }
    artifacts = _render_artifacts(**render_state)
    artifact_hashes = {
        path: _sha256_bytes(content.encode("utf-8"))
        for path, content in sorted(artifacts.items())
    }

    author_sheet_target_m = float(row["sheet_thickness_full_mm"]) * 1.0e-3
    author_velocity_target = float(row["water_velocity_m_s"])
    manifest_core: dict[str, Any] = {
        "schema_version": 1,
        "template": {
            "id": TEMPLATE_ID,
            "version": TEMPLATE_VERSION,
            "profile": "radhakrishnan2018_sheet_vof_external_screen",
            "geometry_fidelity": "reduced_external_gap_not_internal_turning_passage",
        },
        "solver": {
            "distribution": OPENFOAM_DISTRIBUTION,
            "major_version": OPENFOAM_MAJOR_VERSION,
            "patch_tag": OPENFOAM_PATCH_TAG,
            "runner": OPENFOAM_RUNNER,
            "module": OPENFOAM_SOLVER,
            "source_url": (
                "https://github.com/OpenFOAM/OpenFOAM-13/tree/20260624"
            ),
            "runtime_version_verified": False,
        },
        "benchmark": {
            "dataset_case_id": dataset.case_id,
            "row_case_id": row.case_id,
            "validation_role": dataset.validation_role,
            "source_pdf_sha256": dataset.source_sha256,
            "source_doi": dataset.manifest["source"]["doi"],
            "citation": dataset.manifest["source"]["citation"],
            "paper_stage": "Section 2.3 water-only VOF sheet-thickness calculation",
            "paper_stage_mapping": {
                "mechanical_opening_input": {
                    "field": "lopen_mm",
                    "value_m": geometry.opening_distance_m,
                    "origin": dataset.manifest["column_provenance"]["lopen_mm"],
                },
                "liquid_mass_flow_input": {
                    "field": "mdot_water_g_s",
                    "value_kg_s": liquid_mass_flow_kg_s,
                    "origin": dataset.manifest["column_provenance"][
                        "mdot_water_g_s"
                    ],
                },
                "author_vof_sheet_thickness_output_target": {
                    "field": "sheet_thickness_full_mm",
                    "value_m": author_sheet_target_m,
                    "origin": dataset.manifest["column_provenance"][
                        "sheet_thickness_full_mm"
                    ],
                    "prescribed_to_case": False,
                },
                "author_vof_water_velocity_output_target": {
                    "field": "water_velocity_m_s",
                    "value_m_s": author_velocity_target,
                    "origin": dataset.manifest["column_provenance"][
                        "water_velocity_m_s"
                    ],
                    "prescribed_to_case": False,
                },
                "lagrangian_air_mass_flow": {
                    "field": "mdot_air_g_s",
                    "value_kg_s": float(row["mdot_air_g_s"]) * 1.0e-3,
                    "prescribed_to_case": False,
                    "reason": "belongs to the paper's later Lagrangian carrier stage",
                },
                "wave_constants": {
                    "b0": float(row["wave_b0"]),
                    "b1": float(row["wave_b1"]),
                    "prescribed_to_case": False,
                    "reason": "downstream parcel-breakup inputs, not VOF inputs",
                },
            },
        },
        "geometry": geometry.as_dict(),
        "mesh_and_numerics": controls.as_dict(),
        "boundary_flux": {
            "liquid_mass_flow_360_kg_s": liquid_mass_flow_kg_s,
            "wedge_fraction": controls.wedge_fraction,
            "liquid_mass_flow_wedge_kg_s": wedge_mass_flow,
            "liquid_inlet_area_360_m2": (
                geometry.external_sheet_inlet_area_360_m2
            ),
            "liquid_inlet_area_wedge_m2": inlet_area_wedge,
            "liquid_radial_velocity_m_s": liquid_speed,
            "analytical_wedge_mass_flow_kg_s": achieved_wedge_mass_flow,
            "relative_mass_residual": mass_residual,
        },
        "assumptions": [
            {
                "id": "external_gap_reduction",
                "origin": "repository_model_choice",
                "statement": (
                    "The internal center-gap turn, sloped post, and pintle tip are "
                    "not meshed; the mechanical opening is a radial boundary at "
                    "the tip radius."
                ),
            },
            {
                "id": "ambient_properties",
                "origin": "repository_assumption_publication_missing",
                "statement": (
                    "Constant water and air properties at 300 K and 1 atm are "
                    "explicit inputs; the paper does not publish the Fluent values."
                ),
            },
            {
                "id": "turbulence_inlet",
                "origin": "repository_assumption_publication_missing",
                "statement": (
                    "Realizable k-epsilon follows the paper, but turbulence "
                    "intensity and length scale are explicit unvalidated assumptions."
                ),
            },
            {
                "id": "isothermal_incompressible",
                "origin": "selected_solver_scope",
                "statement": (
                    "incompressibleVoF is isothermal; supply pressures are retained "
                    "as provenance and are not imposed as compressible thermodynamics."
                ),
            },
        ],
        "gates": {
            "fixture_pdf_sha256_verified": True,
            "mechanical_opening_not_author_vof_output_used_as_input": True,
            "analytical_wedge_mass_flux_closed": True,
            "static_dictionary_set_complete": True,
            "internal_injector_geometry_resolved": False,
            "exact_openfoam_runtime_verified": False,
            "block_mesh_executed": False,
            "check_mesh_passed": False,
            "solver_executed": False,
            "phase_mass_conservation_verified_from_results": False,
            "mesh_convergence_verified": False,
            "time_step_convergence_verified": False,
            "domain_size_convergence_verified": False,
            "statistical_stationarity_verified": False,
            "author_vof_component_targets_reproduced": False,
            "experimental_smd_validated": False,
            "vof_to_lagrangian_handoff_verified": False,
            "reacting_spray_validated": False,
            "lox_gch4_applicable": False,
            "hardware_qualified": False,
        },
        "prohibited_claims": [
            "This reduced external-gap wedge reproduces the paper's internal VOF geometry.",
            "A finite OpenFOAM result is experimental SMD validation.",
            "This ambient water-air case validates LOX/GCH4 or reacting combustion.",
            "The 3.03 g/s Lagrangian carrier-air stream was imposed in the VOF stage.",
        ],
        "artifact_sha256": artifact_hashes,
    }
    fingerprint_payload = {
        "schema_version": manifest_core["schema_version"],
        "template": manifest_core["template"],
        "solver": manifest_core["solver"],
        "benchmark": manifest_core["benchmark"],
        "geometry": manifest_core["geometry"],
        "mesh_and_numerics": manifest_core["mesh_and_numerics"],
        "boundary_flux": manifest_core["boundary_flux"],
        "artifact_sha256": artifact_hashes,
    }
    fingerprint = _sha256_bytes(_canonical_json_bytes(fingerprint_payload))
    manifest = {**manifest_core, "case_fingerprint_sha256": fingerprint}
    manifest_text = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    files = {**artifacts, MANIFEST_NAME: manifest_text}
    return OpenFOAMCasePackage(files=files, manifest=manifest, fingerprint=fingerprint)


def _render_artifacts(
    *,
    geometry: MovablePintleVOFGeometry,
    controls: OpenFOAMWedgeControls,
    liquid_speed_m_s: float,
    turbulent_k_m2_s2: float,
    turbulent_epsilon_m2_s3: float,
    ambient_k_m2_s2: float,
    ambient_epsilon_m2_s3: float,
) -> dict[str, str]:
    artifacts = {
        "system/blockMeshDict": _render_block_mesh(geometry, controls),
        "system/controlDict": _render_control_dict(controls),
        "system/fvSchemes": _render_fv_schemes(),
        "system/fvSolution": _render_fv_solution(),
        "constant/phaseProperties": _render_phase_properties(controls),
        "constant/physicalProperties.water": _render_physical_properties(
            controls.water
        ),
        "constant/physicalProperties.air": _render_physical_properties(controls.air),
        "constant/momentumTransport": _render_momentum_transport(),
        "constant/g": _render_gravity(),
        "0/U": _render_velocity(liquid_speed_m_s),
        "0/alpha.water": _render_alpha(),
        "0/p_rgh": _render_pressure(),
        "0/k": _render_k(turbulent_k_m2_s2, ambient_k_m2_s2),
        "0/epsilon": _render_epsilon(
            turbulent_epsilon_m2_s3, ambient_epsilon_m2_s3
        ),
        "0/nut": _render_nut(),
        "Allrun": _render_allrun(),
        "Allclean": _render_allclean(),
        "README.md": _render_readme(),
    }
    if set(artifacts) != _SAFE_ARTIFACTS:
        raise AssertionError("internal OpenFOAM artifact renderer drift")
    return {path: _text(content) for path, content in artifacts.items()}


def _foam_header(*, object_name: str, location: str | None, cls: str) -> str:
    location_line = "" if location is None else f'    location    "{location}";\n'
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _render_block_mesh(
    geometry: MovablePintleVOFGeometry, controls: OpenFOAMWedgeControls
) -> str:
    x_values = (
        0.0,
        geometry.post_recess_length_m,
        geometry.post_recess_length_m + geometry.opening_distance_m,
        geometry.axial_domain_length_m,
    )
    radii = (geometry.sheet_exit_radius_m, geometry.radial_domain_radius_m)
    half = 0.5 * math.radians(controls.wedge_angle_deg)
    cos_half = math.cos(half)
    sin_half = math.sin(half)
    vertices: list[tuple[float, float, float]] = []
    for x in x_values:
        vertices.extend(
            [
                (x, radii[0] * cos_half, -radii[0] * sin_half),
                (x, radii[1] * cos_half, -radii[1] * sin_half),
                (x, radii[0] * cos_half, radii[0] * sin_half),
                (x, radii[1] * cos_half, radii[1] * sin_half),
            ]
        )
    vertex_lines = "\n".join(
        f"    ({_fmt(x)} {_fmt(y)} {_fmt(z)})" for x, y, z in vertices
    )
    edges: list[str] = []
    for station, x in enumerate(x_values):
        base = 4 * station
        edges.append(
            f"    arc {base} {base + 2} ({_fmt(x)} {_fmt(radii[0])} 0)"
        )
        edges.append(
            f"    arc {base + 1} {base + 3} ({_fmt(x)} {_fmt(radii[1])} 0)"
        )
    edge_lines = "\n".join(edges)
    axial_counts = (
        controls.upstream_cells,
        controls.opening_cells,
        controls.downstream_cells,
    )
    block_lines: list[str] = []
    for index, nx in enumerate(axial_counts):
        a = 4 * index
        b = 4 * (index + 1)
        block_lines.extend(
            [
                (
                    f"    hex ({a} {b} {b + 1} {a + 1} "
                    f"{a + 2} {b + 2} {b + 3} {a + 3})"
                ),
                (
                    f"    ({nx} {controls.radial_cells} 1) "
                    f"simpleGrading (1 {_fmt(controls.radial_expansion_ratio)} 1)"
                ),
            ]
        )
    blocks = "\n".join(block_lines)

    def inner_face(index: int) -> str:
        a = 4 * index
        b = 4 * (index + 1)
        return f"({a} {b} {b + 2} {a + 2})"

    def outer_face(index: int) -> str:
        a = 4 * index
        b = 4 * (index + 1)
        return f"({a + 1} {a + 3} {b + 3} {b + 1})"

    def front_face(index: int) -> str:
        a = 4 * index
        b = 4 * (index + 1)
        return f"({a} {a + 1} {b + 1} {b})"

    def back_face(index: int) -> str:
        a = 4 * index
        b = 4 * (index + 1)
        return f"({a + 2} {b + 2} {b + 3} {a + 3})"

    outer_faces = "\n".join(f"            {outer_face(i)}" for i in range(3))
    front_faces = "\n".join(f"            {front_face(i)}" for i in range(3))
    back_faces = "\n".join(f"            {back_face(i)}" for i in range(3))
    return _foam_header(object_name="blockMeshDict", location=None, cls="dictionary") + f"""convertToMeters 1;

vertices
(
{vertex_lines}
);

edges
(
{edge_lines}
);

blocks
(
{blocks}
);

boundary
(
    upstreamAmbient
    {{
        type patch;
        faces
        (
            (0 2 3 1)
        );
    }}

    downstreamOutlet
    {{
        type patch;
        faces
        (
            (12 13 15 14)
        );
    }}

    innerWallUpstream
    {{
        type wall;
        faces
        (
            {inner_face(0)}
        );
    }}

    waterInlet
    {{
        type patch;
        faces
        (
            {inner_face(1)}
        );
    }}

    innerWallDownstream
    {{
        type wall;
        faces
        (
            {inner_face(2)}
        );
    }}

    outerAtmosphere
    {{
        type patch;
        faces
        (
{outer_faces}
        );
    }}

    wedgeFront
    {{
        type wedge;
        faces
        (
{front_faces}
        );
    }}

    wedgeBack
    {{
        type wedge;
        faces
        (
{back_faces}
        );
    }}
);

mergePatchPairs ();

// ************************************************************************* //
"""


def _render_control_dict(controls: OpenFOAMWedgeControls) -> str:
    return _foam_header(
        object_name="controlDict", location="system", cls="dictionary"
    ) + f"""solver          {OPENFOAM_SOLVER};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {_fmt(controls.end_time_s)};
deltaT          {_fmt(controls.initial_delta_t_s)};

writeControl    adjustableRunTime;
writeInterval   {_fmt(controls.write_interval_s)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable yes;

adjustTimeStep  yes;
maxCo           {_fmt(controls.max_courant)};
maxAlphaCo      {_fmt(controls.max_alpha_courant)};
maxDeltaT       {_fmt(controls.max_delta_t_s)};

DebugSwitches
{{
    MULES           1;
}}

// ************************************************************************* //
"""


def _render_fv_schemes() -> str:
    return _foam_header(
        object_name="fvSchemes", location="system", cls="dictionary"
    ) + """ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(phi,alpha)  Gauss interfaceCompression vanLeer 1;
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,epsilon) Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""


def _render_fv_solution() -> str:
    return _foam_header(
        object_name="fvSolution", location="system", cls="dictionary"
    ) + """solvers
{
    "alpha.water.*"
    {
        nCorrectors     2;
        nSubCycles      1;
        MULESCorr       yes;
        MULES
        {
            nIter           10;
            tolerance       1e-2;
        }
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }

    "pcorr.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
    }

    p_rgh
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        DIC;
    }

    p_rghFinal
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-8;
        relTol          0;
    }

    "(U|k|epsilon).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0;
        minIter         1;
    }
}

PIMPLE
{
    momentumPredictor          no;
    nOuterCorrectors           1;
    nCorrectors                3;
    nNonOrthogonalCorrectors   1;
}

relaxationFactors
{
    equations
    {
        ".*"            1;
    }
}

// ************************************************************************* //
"""


def _render_phase_properties(controls: OpenFOAMWedgeControls) -> str:
    return _foam_header(
        object_name="phaseProperties", location="constant", cls="dictionary"
    ) + f"""phases (water air);

sigma {_fmt(controls.surface_tension_n_m)};

// ************************************************************************* //
"""


def _render_physical_properties(fluid: OpenFOAMFluidProperties) -> str:
    return _foam_header(
        object_name=f"physicalProperties.{fluid.name}",
        location="constant",
        cls="dictionary",
    ) + f"""viscosityModel constant;

nu {_fmt(fluid.kinematic_viscosity_m2_s)};

rho {_fmt(fluid.density_kg_m3)};

// ************************************************************************* //
"""


def _render_momentum_transport() -> str:
    return _foam_header(
        object_name="momentumTransport", location="constant", cls="dictionary"
    ) + """simulationType  RAS;

RAS
{
    model           realizableKE;
    turbulence      on;
    printCoeffs     on;
}

// ************************************************************************* //
"""


def _render_gravity() -> str:
    return _foam_header(
        object_name="g",
        location="constant",
        cls="uniformDimensionedVectorField",
    ) + """dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);

// ************************************************************************* //
"""


_PATCHES = (
    "upstreamAmbient",
    "downstreamOutlet",
    "innerWallUpstream",
    "waterInlet",
    "innerWallDownstream",
    "outerAtmosphere",
    "wedgeFront",
    "wedgeBack",
)


def _render_velocity(liquid_speed_m_s: float) -> str:
    return _foam_header(object_name="U", location="0", cls="volVectorField") + f"""dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    upstreamAmbient
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    downstreamOutlet
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    innerWallUpstream
    {{
        type            noSlip;
    }}
    waterInlet
    {{
        type            fixedValue;
        value           uniform (0 {_fmt(liquid_speed_m_s)} 0);
    }}
    innerWallDownstream
    {{
        type            noSlip;
    }}
    outerAtmosphere
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    wedgeFront
    {{
        type            wedge;
    }}
    wedgeBack
    {{
        type            wedge;
    }}
}}

// ************************************************************************* //
"""


def _render_alpha() -> str:
    return _foam_header(
        object_name="alpha.water", location="0", cls="volScalarField"
    ) + """dimensions      [];

internalField   uniform 0;

boundaryField
{
    upstreamAmbient
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    downstreamOutlet
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    innerWallUpstream
    {
        type            zeroGradient;
    }
    waterInlet
    {
        type            fixedValue;
        value           uniform 1;
    }
    innerWallDownstream
    {
        type            zeroGradient;
    }
    outerAtmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    wedgeFront
    {
        type            wedge;
    }
    wedgeBack
    {
        type            wedge;
    }
}

// ************************************************************************* //
"""


def _render_pressure() -> str:
    return _foam_header(
        object_name="p_rgh", location="0", cls="volScalarField"
    ) + """dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    upstreamAmbient
    {
        type            prghTotalPressure;
        p0              uniform 0;
    }
    downstreamOutlet
    {
        type            prghTotalPressure;
        p0              uniform 0;
    }
    innerWallUpstream
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    waterInlet
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    innerWallDownstream
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    outerAtmosphere
    {
        type            prghTotalPressure;
        p0              uniform 0;
    }
    wedgeFront
    {
        type            wedge;
    }
    wedgeBack
    {
        type            wedge;
    }
}

// ************************************************************************* //
"""


def _render_k(inlet_value: float, ambient_value: float) -> str:
    return _foam_header(object_name="k", location="0", cls="volScalarField") + _render_turbulence_scalar_body(
        dimensions="[0 2 -2 0 0 0 0]",
        inlet_value=inlet_value,
        ambient_value=ambient_value,
        wall_type="kqRWallFunction",
    )


def _render_epsilon(inlet_value: float, ambient_value: float) -> str:
    return _foam_header(
        object_name="epsilon", location="0", cls="volScalarField"
    ) + _render_turbulence_scalar_body(
        dimensions="[0 2 -3 0 0 0 0]",
        inlet_value=inlet_value,
        ambient_value=ambient_value,
        wall_type="epsilonWallFunction",
    )


def _render_turbulence_scalar_body(
    *, dimensions: str, inlet_value: float, ambient_value: float, wall_type: str
) -> str:
    return f"""dimensions      {dimensions};

internalField   uniform {_fmt(ambient_value)};

boundaryField
{{
    upstreamAmbient
    {{
        type            inletOutlet;
        inletValue      uniform {_fmt(ambient_value)};
        value           uniform {_fmt(ambient_value)};
    }}
    downstreamOutlet
    {{
        type            inletOutlet;
        inletValue      uniform {_fmt(ambient_value)};
        value           uniform {_fmt(ambient_value)};
    }}
    innerWallUpstream
    {{
        type            {wall_type};
        value           uniform {_fmt(ambient_value)};
    }}
    waterInlet
    {{
        type            fixedValue;
        value           uniform {_fmt(inlet_value)};
    }}
    innerWallDownstream
    {{
        type            {wall_type};
        value           uniform {_fmt(ambient_value)};
    }}
    outerAtmosphere
    {{
        type            inletOutlet;
        inletValue      uniform {_fmt(ambient_value)};
        value           uniform {_fmt(ambient_value)};
    }}
    wedgeFront
    {{
        type            wedge;
    }}
    wedgeBack
    {{
        type            wedge;
    }}
}}

// ************************************************************************* //
"""


def _render_nut() -> str:
    return _foam_header(
        object_name="nut", location="0", cls="volScalarField"
    ) + """dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    upstreamAmbient
    {
        type            calculated;
        value           uniform 0;
    }
    downstreamOutlet
    {
        type            calculated;
        value           uniform 0;
    }
    innerWallUpstream
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
    waterInlet
    {
        type            calculated;
        value           uniform 0;
    }
    innerWallDownstream
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
    outerAtmosphere
    {
        type            calculated;
        value           uniform 0;
    }
    wedgeFront
    {
        type            wedge;
    }
    wedgeBack
    {
        type            wedge;
    }
}

// ************************************************************************* //
"""


def _render_allrun() -> str:
    return """#!/bin/sh
set -eu
cd "${0%/*}" || exit 1

if [ "$(foamVersion)" != "13" ]; then
    echo "This case requires OpenFOAM Foundation v13 (target patch 20260624)." >&2
    exit 2
fi

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

runApplication blockMesh
runApplication checkMesh -allTopology -allGeometry
runApplication foamRun
"""


def _render_allclean() -> str:
    return """#!/bin/sh
set -eu
cd "${0%/*}" || exit 1

. "$WM_PROJECT_DIR/bin/tools/CleanFunctions"
cleanCase
"""


def _render_readme() -> str:
    return """# RaoRocketSim OpenFOAM radial-sheet screening wedge

This deterministic case targets OpenFOAM Foundation v13, patch/tag `20260624`,
and runs `foamRun` with `solver incompressibleVoF`.

It represents the water-only VOF stage described in Radhakrishnan et al. (2018),
Section 2.3: water enters a quiescent-air domain. The paper's 3.03 g/s annular
air stream belongs to the later Lagrangian carrier calculation and is not
silently imposed here.

Important limitation: this first case is an external-gap screening model. The
mechanical opening is a radial water boundary at the tip radius. The internal
center-gap turn and sloped post/tip passages are recorded in the manifest but
are not meshed. Consequently this case can screen external VOF transport and
sheet evolution; it cannot be called a reproduction of the paper's internal
sheet-formation solution.

Run only under the exact pinned distribution:

```sh
./Allrun
```

Before using any result, update external evidence for: exact patch identity,
strict `checkMesh`, phase-mass closure, mesh/time-step/domain convergence,
stationarity, and comparison to the author-VOF sheet thickness and velocity.
The generated manifest intentionally leaves those gates false. It confers no
SMD, reacting-spray, LOX/GCH4, hot-fire, or hardware qualification.
"""


def write_openfoam_case(
    package: OpenFOAMCasePackage, destination: str | Path
) -> OpenFOAMCaseWriteResult:
    """Atomically write a package, or accept an exactly identical directory.

    Existing non-identical directories and symlinked path components fail
    closed.  No OpenFOAM executable is invoked.
    """

    requested_path = Path(destination).expanduser().absolute()
    if requested_path == requested_path.parent:
        raise OpenFOAMExportError("destination cannot be a filesystem root")
    _reject_nearest_symlink(requested_path)
    parent = requested_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _reject_nearest_symlink(requested_path)
    # macOS exposes trusted system aliases such as /var -> /private/var.  Resolve
    # the existing parent after rejecting a user-controlled nearest symlink so
    # normal temporary directories remain usable without serializing aliases.
    destination_path = parent.resolve(strict=True) / requested_path.name

    if destination_path.exists():
        if destination_path.is_symlink() or not destination_path.is_dir():
            raise OpenFOAMExportError("destination must be a normal directory")
        if _directory_matches_package(destination_path, package):
            return OpenFOAMCaseWriteResult(
                destination=destination_path,
                fingerprint=package.fingerprint,
                written=False,
            )
        raise OpenFOAMExportError(
            "destination exists and is not the exact same case package"
        )

    staging = Path(tempfile.mkdtemp(prefix=".raosim-openfoam-", dir=parent))
    try:
        for relative, content in package.files.items():
            target = staging.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
            if relative in {"Allrun", "Allclean"}:
                target.chmod(0o755)
        os.replace(staging, destination_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return OpenFOAMCaseWriteResult(
        destination=destination_path,
        fingerprint=package.fingerprint,
        written=True,
    )


def _reject_nearest_symlink(path: Path) -> None:
    """Reject the nearest existing destination component when it is a link.

    This catches a caller-controlled ``linked/case`` destination while allowing
    platform-owned aliases above an already-real parent (notably macOS
    ``/var``).  The real parent is used for the atomic publish immediately
    afterwards.
    """

    current = path
    while not current.exists() and not current.is_symlink():
        if current == current.parent:
            break
        current = current.parent
    if current.is_symlink():
        raise OpenFOAMExportError(
            f"symlinked destination component is not allowed: {current}"
        )


def _directory_matches_package(
    destination: Path, package: OpenFOAMCasePackage
) -> bool:
    found: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            return False
        if path.is_file():
            found.add(path.relative_to(destination).as_posix())
    if found != set(package.files):
        return False
    return all(
        (destination / relative).read_bytes() == content.encode("utf-8")
        for relative, content in package.files.items()
    )


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI for the pinned water-air VOF case exporter."""

    parser = argparse.ArgumentParser(
        prog="lrekit export-openfoam-spray",
        description=(
            "Export a deterministic OpenFOAM Foundation v13 water-only VOF "
            "screening wedge for the Radhakrishnan 2018 fixture."
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case-row", choices=("case_1", "case_2", "case_3", "case_4"), default="case_1"
    )
    parser.add_argument("--wedge-angle-deg", type=float, default=5.0)
    parser.add_argument("--opening-cells", type=int, default=6)
    parser.add_argument("--radial-cells", type=int, default=240)
    args = parser.parse_args(argv)
    controls = OpenFOAMWedgeControls(
        wedge_angle_deg=args.wedge_angle_deg,
        opening_cells=args.opening_cells,
        radial_cells=args.radial_cells,
    )
    package = build_radhakrishnan2018_sheet_vof_case(
        args.case_row, controls=controls
    )
    result = write_openfoam_case(package, args.output)
    action = "wrote" if result.written else "verified identical"
    print(f"{action}: {result.destination}")
    print(f"case fingerprint: {result.fingerprint}")
    print("OpenFOAM was not run; execution and validation gates remain false.")
    return 0
