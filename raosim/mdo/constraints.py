"""Authoritative constraint and feasibility contract for the engine MDO.

The discipline solve, NLP, numerical :mod:`raosim.mdo.state`, and host
snapshot all consume this ordered manifest.  A constraint therefore cannot be
added to reporting without making an explicit decision about whether the
optimizer enforces it and whether missing model coverage is a failure or an
unknown.

The manifest is software policy around equations implemented in the discipline
modules; it does not invent new physical limits.  ``source_id`` points back to
the source already documented beside each underlying equation.  The two new
policy-sensitive rows have the following exact source map (the original PDFs,
not the OCR, were checked):

* ``NASA-SP-125-1967-sec-4.2-PDF137-printed128`` — Dieter K. Huzel and
  David H. Huang, *Design of Liquid Propellant Rocket Engines* (1967), NASA
  SP-125, sec. 4.2, "Injection Pressure Drop," source-PDF p. 137 /
  printed p. 128.  The stated 15--20 percent range supports the numerical 0.20 design
  screen.  It is not presented here as a universal chug-stability boundary.
* ``NASA-SP-194-1972-sec-6.2.3.1-PDF293-294`` — David T. Harrje (editor) and
  Frederick H. Reardon (associate editor), *Liquid Propellant Rocket
  Combustion Instability* (1972), NASA SP-194, sec. 6.2.3.1, "Injector
  Impedance," source-PDF pp. 293--294.  It supports the qualitative coupling
  between injector pressure drop and low-frequency stability and explicitly
  warns that increasing only one stream's drop can be destabilizing.
* ``Nasuti-Pizzarelli-2021-sec-2.2-Eq9-PDF6`` — Francesco Nasuti and Marco
  Pizzarelli, *Pseudo-boiling and heat transfer deterioration while heating
  supercritical liquid rocket engine propellants* (2021), sec. 2.2, Eq. (9),
  source-PDF p. 6, DOI 10.1016/j.supflu.2020.105066.  Its ``(beta/cp)_b`` input
  is a real-fluid bulk property and cannot be reconstructed from the cooling
  march's constant properties.
* ``NASA-SP-8087-1972-sec-3.1.1.5.4-PDF78`` — *Liquid Rocket Engine
  Fluid-Cooled Combustion Chambers* (NASA SP-8087, April 1972), sec. 3.1.1.5.4,
  "Wall Temperatures," source-PDF p. 78.  It directly gives the 728 K RP-1
  liquid-wall coking limit already implemented by the cooling discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Sequence

import numpy as np


ConstraintKind = Literal["inequality", "equality", "availability_gate"]
OptimizerRole = Literal["hard", "final_gate", "report_only"]


@dataclass(frozen=True)
class ConstraintSpec:
    """One stable row in the engine constraint contract.

    ``name`` is the public/NLP name.  ``engine_key`` is the key produced by
    :func:`raosim.mdo.engine.solve_engine`; ``None`` denotes a requirement-layer
    row assembled outside ``EngineResult`` (currently the Isp epsilon row).
    """

    name: str
    engine_key: str | None
    kind: ConstraintKind
    optimizer_role: OptimizerRole
    category: str
    scale: float
    units: str | None
    source_id: str | None
    applicability: str = "always"


class ConstraintReasonCode(IntEnum):
    """JAX-safe reason codes stored in :class:`ConstraintState`."""

    NONE = 0
    NOT_APPLICABLE = 1
    MODEL_UNAVAILABLE = 2
    REQUIREMENT_INACTIVE = 3
    COOLANT_UNRESOLVED = 4


# Ordered once, consumed everywhere.  The row scales are conditioning
# references, not limits; physical limits remain in the discipline equations.
CONSTRAINT_MANIFEST: tuple[ConstraintSpec, ...] = (
    ConstraintSpec("isp_epsilon", None, "inequality", "hard", "requirement",
                   50.0, "s", "NASA-SP-125-sec-2.1-item-2", "isp_floor"),
    ConstraintSpec("thrust_closure", "thrust_residual", "equality",
                   "report_only", "numerical", 1.0e-10, None,
                   "NASA-SP-125-sec-2.1-item-1"),
    ConstraintSpec("separation", "separation_margin", "inequality", "hard",
                   "aerodynamics", 1.0, None, "NASA-SP-8120"),
    ConstraintSpec("coking", "coking_margin_min", "inequality", "hard",
                   "thermal", 300.0, "K",
                   "NASA-SP-8087-1972-sec-3.1.1.5.4-PDF78",
                   "coking_coolant"),
    ConstraintSpec("land_fit", "land_min", "inequality", "hard",
                   "geometry", 5.0e-4, "m", None),
    ConstraintSpec("chug", "chug_margin_min", "inequality", "hard",
                   "injector", 0.2, None,
                   "NASA-SP-125-1967-sec-4.2-PDF137-printed128;"
                   "NASA-SP-194-1972-sec-6.2.3.1-PDF293-294"),
    ConstraintSpec("pintle_transition", "pintle_transition_margin",
                   "inequality", "hard", "injector", 5.0e-5, "m^2", None),
    ConstraintSpec("pump_suction", "nss_margin_min", "inequality", "hard",
                   "feed", 2.0, None, "NASA-SP-8052"),
    ConstraintSpec("pump_tip_speed", "tip_speed_margin_min", "inequality",
                   "hard", "feed", 300.0, "m/s", "NASA-SP-8109"),
    ConstraintSpec("aspect_ratio", "aspect_ratio_margin", "inequality",
                   "hard", "thermal", 8.0, None, None),
    ConstraintSpec("blockage_lo", "blockage_lo_margin", "inequality", "hard",
                   "injector", 0.3, None, None),
    ConstraintSpec("blockage_hi", "blockage_hi_margin", "inequality", "hard",
                   "injector", 0.3, None, None),
    ConstraintSpec("structural_stress", "structural_stress_margin",
                   "inequality", "hard", "structures", 1.0e8, "Pa",
                   "NASA-SP-125"),
    ConstraintSpec("wall_temp", "wall_temp_margin", "inequality", "hard",
                   "thermal", 100.0, "K", None),
    ConstraintSpec("film_capacity", "film_capacity_margin", "inequality",
                   "hard", "thermal", 0.1, None, "NASA-SP-8087"),
    ConstraintSpec("property_domain", "property_domain_margin", "inequality",
                   "hard", "properties", 1.0, None, None,
                   "sampled_property_surface"),
    ConstraintSpec("chart_domain", "chart_domain_margin", "inequality", "hard",
                   "geometry", 0.1, None, None),
    ConstraintSpec("wall_monotonic", "wall_monotonic_margin", "inequality",
                   "hard", "geometry", 1.0e-4, "m", None),
    ConstraintSpec("chamber_volume", "chamber_volume_margin", "inequality",
                   "hard", "geometry", 0.05, "m", "NASA-SP-125-sec-4.1"),
    ConstraintSpec("jacket_thin_shell", "jacket_thin_shell_margin",
                   "inequality", "hard", "structures", 0.05, None,
                   "NASA-SP-125-p336"),
    ConstraintSpec("nozzle_collapse", "nozzle_collapse_margin", "inequality",
                   "hard", "structures", 1.0, None, "NASA-SP-8087-sec-2.1.3"),
    ConstraintSpec("envelope_diameter", "envelope_diameter_margin",
                   "inequality", "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-6", "envelope_diameter_requirement"),
    ConstraintSpec("envelope_length", "envelope_length_margin", "inequality",
                   "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-6", "envelope_length_requirement"),
    ConstraintSpec("dry_mass_partial", "dry_mass_partial_margin", "inequality",
                   "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-5", "dry_mass_requirement"),
    ConstraintSpec("engine_residual", "engine_residual_margin", "inequality",
                   "final_gate", "numerical", 1.0e-11, None, None),
    ConstraintSpec("cooling_residual", "cooling_residual_margin", "inequality",
                   "final_gate", "numerical", 1.0e-11, "K", None),
    ConstraintSpec("solver_status", "solver_status_margin", "availability_gate",
                   "final_gate", "numerical", 1.0, None, None),
    ConstraintSpec("finite", "finite_margin", "availability_gate", "final_gate",
                   "numerical", 1.0, None, None),
    ConstraintSpec("coolant_htd", "coolant_htd_margin", "availability_gate",
                   "final_gate", "thermal", 1.0, None,
                   "Nasuti-Pizzarelli-2021-sec-2.2-Eq9-PDF6", "htd_coolant"),
)


ENGINE_CONSTRAINT_SPECS = tuple(
    spec for spec in CONSTRAINT_MANIFEST if spec.engine_key is not None
)
ENGINE_CONSTRAINT_NAMES = tuple(
    spec.engine_key for spec in ENGINE_CONSTRAINT_SPECS
)
OPTIMIZER_CONSTRAINT_SPECS = tuple(
    spec for spec in CONSTRAINT_MANIFEST if spec.optimizer_role == "hard"
)
OPTIMIZER_CONSTRAINT_NAMES = tuple(spec.name for spec in OPTIMIZER_CONSTRAINT_SPECS)
OPTIMIZER_CONSTRAINT_SCALES = np.asarray(
    [spec.scale for spec in OPTIMIZER_CONSTRAINT_SPECS], dtype=float
)


def _resolve_coolant(mission: object) -> tuple[str, bool]:
    """Return ``(coolant_name, resolved)`` for the mission's propellant.

    A propellant outside the catalog — a typo, or a custom ``OXIDIZER/FUEL``
    pair supplied with its own property table — has no catalog coolant
    identity.  The caller must not guess one: which wall-side mechanism
    governs (coking or HTD) is then genuinely unknown, and this contract
    reduces unknown coverage to ``unknown``, never to a passing constraint.
    """

    from raosim.mdo.propellants import get_propellant

    name = getattr(mission, "propellant_name", None)
    try:
        return str(get_propellant(name).coolant_name), True
    except (KeyError, AttributeError, TypeError):
        return str(name or ""), False


def _coolant_name(mission: object) -> str:
    return _resolve_coolant(mission)[0]


def constraint_metadata(
    mission: object,
    surfaces: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-shape applicability/availability/required/reason arrays.

    This function is host/static policy.  Its outputs may be embedded as JAX
    constants by the engine-state converter; no Python decision depends on a
    traced design variable.
    """

    applicable: list[bool] = []
    available: list[bool] = []
    required: list[bool] = []
    reasons: list[int] = []
    from raosim.mdo.coolant_htd import htd_is_applicable

    coolant_name, coolant_resolved = _resolve_coolant(mission)
    htd_applicable = htd_is_applicable(coolant_name) if coolant_resolved else False
    sampled_domain = getattr(surfaces, "domain_policy", "sampled_bounded") == (
        "sampled_bounded"
    )

    for spec in ENGINE_CONSTRAINT_SPECS:
        app = True
        reason = ConstraintReasonCode.NONE
        # With an unresolved coolant identity neither wall-side mechanism can
        # be ruled out, so both rows stay applicable and become unavailable
        # below.  Ruling one out here is what previously let a mistyped
        # methane/hydrogen mission lose its governing HTD gate.
        if spec.applicability == "coking_coolant":
            app = True if not coolant_resolved else not htd_applicable
        elif spec.applicability == "htd_coolant":
            app = True if not coolant_resolved else htd_applicable
        elif spec.applicability == "sampled_property_surface":
            app = sampled_domain
        elif spec.applicability == "envelope_diameter_requirement":
            app = float(getattr(mission, "envelope_diameter_max", 1.0e3)) < 1.0e3
            if not app:
                reason = ConstraintReasonCode.REQUIREMENT_INACTIVE
        elif spec.applicability == "envelope_length_requirement":
            app = float(getattr(mission, "envelope_length_max", 1.0e3)) < 1.0e3
            if not app:
                reason = ConstraintReasonCode.REQUIREMENT_INACTIVE
        elif spec.applicability == "dry_mass_requirement":
            app = float(getattr(mission, "dry_mass_max", 1.0e9)) < 1.0e9
            if not app:
                reason = ConstraintReasonCode.REQUIREMENT_INACTIVE

        avail = app
        if not app and reason == ConstraintReasonCode.NONE:
            reason = ConstraintReasonCode.NOT_APPLICABLE
        if (
            spec.applicability in {"coking_coolant", "htd_coolant"}
            and not coolant_resolved
        ):
            avail = False
            reason = ConstraintReasonCode.COOLANT_UNRESOLVED
        elif spec.applicability == "htd_coolant" and app:
            # No differentiable real-fluid coolant bundle is wired yet.  A path
            # string is intentionally not accepted as proof of model coverage.
            avail = False
            reason = ConstraintReasonCode.MODEL_UNAVAILABLE

        req = app and spec.optimizer_role in {"hard", "final_gate"}
        applicable.append(app)
        available.append(avail)
        required.append(req)
        reasons.append(int(reason))

    return (
        np.asarray(applicable, dtype=bool),
        np.asarray(available, dtype=bool),
        np.asarray(required, dtype=bool),
        np.asarray(reasons, dtype=np.int32),
    )


def applicable_optimizer_names(mission: object, surfaces: object) -> tuple[str, ...]:
    """Hard-row names applicable to this exact mission/property contract."""

    app, available, _, _ = constraint_metadata(mission, surfaces)
    engine_flags = {
        spec.engine_key: bool(app[i] and available[i])
        for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
    }
    return tuple(
        spec.name
        for spec in OPTIMIZER_CONSTRAINT_SPECS
        if spec.engine_key is None or engine_flags.get(spec.engine_key, False)
    )


def reason_text(code: int, spec: ConstraintSpec, mission: object) -> str:
    """Map a numerical state reason to a stable human-readable explanation."""

    reason = ConstraintReasonCode(int(code))
    if reason == ConstraintReasonCode.NOT_APPLICABLE:
        if spec.name == "coking":
            return "coking is not applicable to this coolant; HTD is the wall-side mechanism"
        if spec.name == "coolant_htd":
            return "HTD is not applicable to this coolant; the coking limit governs"
        if spec.name == "property_domain":
            return "constant fallback axes are interpolation scaffolding, not physical bounds"
        return "constraint is not applicable to this engine architecture"
    if reason == ConstraintReasonCode.MODEL_UNAVAILABLE:
        if spec.name != "coolant_htd":
            return "constraint output is unavailable in this result version"
        return (
            f"HTD is applicable to coolant {_coolant_name(mission)!r}, but the "
            "MDO has no validated real-fluid beta/cp(T,p) surface required by "
            "Nasuti & Pizzarelli (2021), Eq. (9)"
        )
    if reason == ConstraintReasonCode.REQUIREMENT_INACTIVE:
        return "no active requirement supplied this constraint limit"
    if reason == ConstraintReasonCode.COOLANT_UNRESOLVED:
        return (
            f"propellant {getattr(mission, 'propellant_name', None)!r} is not in "
            "the catalog, so no coolant identity resolves and neither the coking "
            "limit nor the HTD criterion can be shown to govern this wall"
        )
    return ""


def status_from_rows(
    values: Sequence[float],
    applicable: Sequence[bool],
    available: Sequence[bool],
    required: Sequence[bool],
    *,
    indices: Sequence[int] | None = None,
    nonfinite: Literal["fail", "unknown"] = "fail",
) -> Literal["pass", "fail", "unknown"]:
    """Reduce rows without collapsing unavailable physics into a pass.

    ``nonfinite='unknown'`` is used for physics and requirement categories when
    a separate numerical gate owns contamination.  A finite negative margin is
    still a demonstrated failure and takes precedence over unknown rows.
    """

    if nonfinite not in {"fail", "unknown"}:
        raise ValueError("nonfinite must be 'fail' or 'unknown'")

    v = np.asarray(values, dtype=float)
    app = np.asarray(applicable, dtype=bool)
    avail = np.asarray(available, dtype=bool)
    req = np.asarray(required, dtype=bool)
    if indices is not None:
        idx = np.asarray(tuple(indices), dtype=int)
        v, app, avail, req = v[idx], app[idx], avail[idx], req[idx]
    active = app & req
    known = active & avail
    if np.any(known & np.isfinite(v) & (v < 0.0)):
        return "fail"
    if np.any(known & ~np.isfinite(v)):
        return nonfinite
    if np.any(active & ~avail):
        return "unknown"
    return "pass"


__all__ = [
    "ConstraintReasonCode",
    "ConstraintSpec",
    "CONSTRAINT_MANIFEST",
    "ENGINE_CONSTRAINT_NAMES",
    "ENGINE_CONSTRAINT_SPECS",
    "OPTIMIZER_CONSTRAINT_NAMES",
    "OPTIMIZER_CONSTRAINT_SCALES",
    "OPTIMIZER_CONSTRAINT_SPECS",
    "applicable_optimizer_names",
    "constraint_metadata",
    "reason_text",
    "status_from_rows",
]
