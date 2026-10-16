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
* ``Son-2017-JPP-33-4-DOI-10.2514-1.B36301`` — Son, Radhakrishnan, Koo, Kwon
  and Kim,
  *Design Procedure of a Movable Pintle Injector for Liquid Rocket Engines*,
  JPP 33(4) (2017), local PDF ``propulsion_texts/pintle_injector/son2017.pdf``
  (SHA-256 ``bc10dd3b…a2b1a7``).  Eq. (1) and the transition discussion
  (PDF p. 4: the controllable opening range lies below the transition) support
  ``pintle_transition``; PDF p. 2 (no uniform sheet within 0.1 mm) and PDF
  p. 10 (a schedule below 0.1 mm rejected as infeasible) support
  ``pintle_min_opening``; PDF p. 9 lists both "minimum area ... lower than the
  center gap area" and "chamber diameter should be larger than the injector
  module diameter" (``pintle_module_fit``) among its feasibility checks.  The
  0.95 transition fraction and the 0.1 mm floor applied to a new geometry are
  repository applicability choices, not universal limits.

The pintle rows are architecture-specific (plan rule 4).  ``pintle_slotted``
rows apply only to the fixed slotted post and ``pintle_movable`` rows only to
the Son continuous-gap valve; a row that does not apply is returned inert by
the engine and marked not applicable here.

Basis of every row (``basis_class``; MDO issues audit 2026-10-16, §1.4 item
22).  A ``source_id`` is a citation route; ``basis_class`` says what kind of
claim the implemented limit is, so a row with no literature behind its number
is labelled rather than left as an unknown basis:

* ``literature`` — the limit (or its governing allowable) is stated in the
  cited source;
* ``geometric_definition`` — a packing/ordering identity of the modelled
  geometry (non-negative land, non-decreasing divergent wall, ...);
* ``model_validity_domain`` — the domain over which an implemented model or
  table is defined (sampled CEA box, Rao/TOP chart extent);
* ``repository_rule`` — a repository choice; any ``source_id`` cites only the
  criterion's form, not the implemented number;
* ``numerical_contract`` — a software gate on the solve itself;
* ``requirement`` — a user requirement row.

Rows corrected or newly attributed in the 2026-10-16 pass (original PDFs,
page-verified):

* ``blockage_lo`` (BF ≥ 0.30): Freeberg 2019, ``propulsion_texts/
  pintle_injector/freeberg2019.pdf``, PDF pp. 5-6 relays TRW advice of
  0.3 < BF < 0.7 (second-hand, via Escher), PDF p. 10 (conclusions) repeats
  the 0.3-0.7 band and PDF p. 2 Eq. (1) defines BF.
* ``blockage_hi`` (BF ≤ 0.90): no held source states 0.90.  The held bounds
  are 0.7 (the same TRW advice) and the Ryu optimum near BF 85 % that Hwang
  et al. report second-hand together with thermal damage of an 86 % tip
  (``pintle_injector/s42405-022-00489-w.pdf`` PDF pp. 2, 4, 8).  Repository
  rule; the value is unchanged pending an owner decision.
* ``wall_temp`` (800 K): the criterion form — an allowable gas-side wall
  temperature set from the material's yield-strength sensitivity — is
  Mirzamoghadam 1991 (PDF p. 5; Table 2, PDF p. 7, lists the 1050 °F
  hot-channel value as "Depends on Material").  800 K itself is not in the
  paper (its 900 °F = 755 K is an H2/O2 copper compatibility value, PDF
  p. 4); repository rule.
* ``chart_domain``: ε ∈ [4, 50], L_f ∈ [60, 100] % match the extent of the
  held Rao/TOP angle charts (SP-125 Fig. 4-14, PDF p. 100; SP-8120 Fig. 5(b),
  PDF p. 29), graphs only.  The extent is cited, not the tabulated
  ordinates (see the adjudication note in ``raosim.nozzle_geometry``).
* ``land_fit``, ``wall_monotonic``: geometric definitions;
  ``pintle_wall_fit``: the host CAD layout cap 0.8 r_o
  (``injector_cad.resolve_machined_pintle_layout``), a repository rule;
  ``property_domain``: the sampled CEA table box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Sequence

import numpy as np


ConstraintKind = Literal["inequality", "equality", "availability_gate"]
OptimizerRole = Literal["hard", "final_gate", "report_only"]
BasisClass = Literal[
    "literature",
    "geometric_definition",
    "model_validity_domain",
    "repository_rule",
    "numerical_contract",
    "requirement",
]
BASIS_CLASSES: tuple[str, ...] = (
    "literature",
    "geometric_definition",
    "model_validity_domain",
    "repository_rule",
    "numerical_contract",
    "requirement",
)


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
    #: What kind of claim the implemented limit is (module docstring).
    #: Appended and defaulted to ``None`` (undeclared) so positional
    #: construction is unchanged; every manifest row declares one.
    basis_class: BasisClass | None = None

    def __post_init__(self) -> None:
        if self.basis_class is not None and self.basis_class not in BASIS_CLASSES:
            raise ValueError(
                f"{self.name}: basis_class must be one of {BASIS_CLASSES}"
            )
        if self.basis_class == "literature" and not self.source_id:
            raise ValueError(f"{self.name}: a literature row needs a source_id")


class ConstraintReasonCode(IntEnum):
    """JAX-safe reason codes stored in :class:`ConstraintState`."""

    NONE = 0
    NOT_APPLICABLE = 1
    MODEL_UNAVAILABLE = 2
    REQUIREMENT_INACTIVE = 3
    COOLANT_UNRESOLVED = 4
    CONTOUR_PROVIDER_FAILED = 5


# Ordered once, consumed everywhere.  The row scales are conditioning
# references, not limits; physical limits remain in the discipline equations.
CONSTRAINT_MANIFEST: tuple[ConstraintSpec, ...] = (
    ConstraintSpec("isp_epsilon", None, "inequality", "hard", "requirement",
                   50.0, "s", "NASA-SP-125-sec-2.1-item-2", "isp_floor",
                   basis_class="requirement"),
    ConstraintSpec("thrust_closure", "thrust_residual", "equality",
                   "report_only", "numerical", 1.0e-10, None,
                   "NASA-SP-125-sec-2.1-item-1",
                   basis_class="numerical_contract"),
    ConstraintSpec("separation", "separation_margin", "inequality", "hard",
                   "aerodynamics", 1.0, None, "NASA-SP-8120",
                   basis_class="literature"),
    ConstraintSpec("coking", "coking_margin_min", "inequality", "hard",
                   "thermal", 300.0, "K",
                   "NASA-SP-8087-1972-sec-3.1.1.5.4-PDF78",
                   "coking_coolant", basis_class="literature"),
    # Positive land between adjacent channels at every station: a packing
    # identity of the channel layout, not a literature limit.
    ConstraintSpec("land_fit", "land_min", "inequality", "hard",
                   "geometry", 5.0e-4, "m", None,
                   basis_class="geometric_definition"),
    ConstraintSpec("chug", "chug_margin_min", "inequality", "hard",
                   "injector", 0.2, None,
                   "NASA-SP-125-1967-sec-4.2-PDF137-printed128;"
                   "NASA-SP-194-1972-sec-6.2.3.1-PDF293-294",
                   basis_class="literature"),
    ConstraintSpec("pintle_transition", "pintle_transition_margin",
                   "inequality", "hard", "injector", 5.0e-5, "m^2",
                   "Son-2017-JPP-33-4-DOI-10.2514-1.B36301-Eq1-PDF4-PDF9",
                   "pintle_movable", basis_class="literature"),
    ConstraintSpec("pump_suction", "nss_margin_min", "inequality", "hard",
                   "feed", 2.0, None, "NASA-SP-8052",
                   basis_class="literature"),
    ConstraintSpec("pump_tip_speed", "tip_speed_margin_min", "inequality",
                   "hard", "feed", 300.0, "m/s", "NASA-SP-8109",
                   basis_class="literature"),
    # AR <= 8 is a MANUFACTURING bound, not a model-validity bound: Wadel 1997
    # cut analytic designs "from 40 down to the limit of 8" only when milling
    # constraints were imposed, and Carlile 1992 tested 8 and 15.
    ConstraintSpec("aspect_ratio", "aspect_ratio_margin", "inequality",
                   "hard", "manufacturing", 8.0, None,
                   "Wadel-1997-milled-channel-fabrication-constraints;"
                   "Carlile-1992-AIAA-92-3154;Mirzamoghadam-1991-AIAA-91-1982",
                   basis_class="literature"),
    # BF >= 0.30: TRW advice relayed by Freeberg (module docstring).
    ConstraintSpec("blockage_lo", "blockage_lo_margin", "inequality", "hard",
                   "injector", 0.3, None,
                   "Freeberg-2019-AIAA-2019-0152-PDF5-6-TRW-BF-0.3-to-0.7-"
                   "secondhand-via-Escher;Freeberg-2019-PDF10-conclusions",
                   "pintle_slotted", basis_class="literature"),
    # BF <= 0.90: no held source states 0.90 (module docstring).
    ConstraintSpec("blockage_hi", "blockage_hi_margin", "inequality", "hard",
                   "injector", 0.3, None, None, "pintle_slotted",
                   basis_class="repository_rule"),
    # Pintle component rows (raosim.mdo.pintle).  The post screen is the
    # outer-radius thin pressure-vessel hoop form, which bounds the elastic
    # Lame bore stress, against the closeout alloy's SP-8087 yield allowable.
    ConstraintSpec("pintle_post_stress", "pintle_post_stress_margin",
                   "inequality", "hard", "structures", 1.0e8, "Pa",
                   "outer-radius-hoop-screening-assumption;"
                   "NASA-SP-8087-sec-2.1.3-yield-FOS",
                   # The allowable is sourced; the governing stress form is a
                   # screening choice, so the row as a whole is not literature.
                   basis_class="repository_rule"),
    # Host machined-layout wall cap (0.8 r_o); a repository layout rule.
    ConstraintSpec("pintle_wall_fit", "pintle_wall_fit_margin", "inequality",
                   "hard", "geometry", 1.0e-3, "m", None,
                   basis_class="repository_rule"),
    ConstraintSpec("pintle_min_opening", "pintle_min_opening_margin",
                   "inequality", "hard", "injector", 1.0e-4, "m",
                   "Son-2017-JPP-33-4-DOI-10.2514-1.B36301-PDF2-PDF10",
                   "pintle_movable", basis_class="literature"),
    ConstraintSpec("pintle_center_gap_fit", "pintle_center_gap_fit_margin",
                   "inequality", "hard", "geometry", 1.0e-3, "m",
                   "Son-2017-JPP-33-4-DOI-10.2514-1.B36301-Fig2-geometry",
                   "pintle_movable", basis_class="geometric_definition"),
    ConstraintSpec("pintle_module_fit", "pintle_module_fit_margin",
                   "inequality", "hard", "geometry", 0.05, "m",
                   "Son-2017-JPP-33-4-DOI-10.2514-1.B36301-PDF9",
                   basis_class="literature"),
    ConstraintSpec("structural_stress", "structural_stress_margin",
                   "inequality", "hard", "structures", 1.0e8, "Pa",
                   "NASA-SP-125", basis_class="literature"),
    # The criterion form is Mirzamoghadam's; the 800 K value is a repository
    # choice the paper does not state (module docstring).
    ConstraintSpec("wall_temp", "wall_temp_margin", "inequality", "hard",
                   "thermal", 100.0, "K",
                   "Mirzamoghadam-1991-AIAA-91-1982-PDF5-PDF7-Table2-"
                   "allowable-wall-temperature-criterion-form",
                   basis_class="repository_rule"),
    ConstraintSpec("film_capacity", "film_capacity_margin", "inequality",
                   "hard", "thermal", 0.1, None, "NASA-SP-8087",
                   basis_class="literature"),
    # The sampled (Pc, O/F) CEA table box: the domain of the property model.
    ConstraintSpec("property_domain", "property_domain_margin", "inequality",
                   "hard", "properties", 1.0, None, None,
                   "sampled_property_surface",
                   basis_class="model_validity_domain"),
    # Extent of the held Rao/TOP angle charts (graphs only; module docstring).
    ConstraintSpec("chart_domain", "chart_domain_margin", "inequality", "hard",
                   "geometry", 0.1, None,
                   "NASA-SP-125-sec-4.3-Fig4-14-PDF100-printed91-extent;"
                   "NASA-SP-8120-1976-Fig5b-PDF29-printed15-extent",
                   basis_class="model_validity_domain"),
    # A bell wall must not turn back toward the axis downstream of the throat.
    ConstraintSpec("wall_monotonic", "wall_monotonic_margin", "inequality",
                   "hard", "geometry", 1.0e-4, "m", None,
                   basis_class="geometric_definition"),
    # Non-negative barrel length under the SP-125 chamber-volume definition
    # (injector face to throat plane, printed p. 88).
    ConstraintSpec("chamber_volume", "chamber_volume_margin", "inequality",
                   "hard", "geometry", 0.05, "m", "NASA-SP-125-sec-4.1",
                   basis_class="geometric_definition"),
    ConstraintSpec("jacket_thin_shell", "jacket_thin_shell_margin",
                   "inequality", "hard", "structures", 0.05, None,
                   "NASA-SP-125-p336", basis_class="literature"),
    ConstraintSpec("nozzle_collapse", "nozzle_collapse_margin", "inequality",
                   "hard", "structures", 1.0, None, "NASA-SP-8087-sec-2.1.3",
                   basis_class="literature"),
    ConstraintSpec("envelope_diameter", "envelope_diameter_margin",
                   "inequality", "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-6", "envelope_diameter_requirement",
                   basis_class="requirement"),
    ConstraintSpec("envelope_length", "envelope_length_margin",
                   "inequality", "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-6", "envelope_length_requirement",
                   basis_class="requirement"),
    ConstraintSpec("dry_mass_partial", "dry_mass_partial_margin", "inequality",
                   "hard", "requirement", 1.0, None,
                   "NASA-SP-125-sec-2.1-item-5", "dry_mass_requirement",
                   basis_class="requirement"),
    # The provider outcome is a mandatory post-solve availability gate rather
    # than a differentiable hard row.  A future contour implementation may
    # report a failed candidate with a fixed-shape carrier grid; this gate
    # prevents that carrier from being accepted as successful geometry
    # (plan §0.1 rule 1: no silent contour substitution).
    ConstraintSpec(
        "contour_provider", "contour_provider_margin", "availability_gate",
        "final_gate", "geometry", 1.0, None, None,
        basis_class="numerical_contract",
    ),
    # The differentiable port is a mandatory post-solve screen, but not yet an
    # SLSQP row.  Adding the extra hard row changes the active-set path enough
    # to hit a pre-existing non-finite implicit-root derivative at a valid box
    # corner (tests/test_mdo_nlp.py Phase-9 gate).  Hard enforcement is deferred
    # until that corner is robust and NASA SP-8068 qualifies the classical
    # plate coefficient; a negative final margin still cannot pass verification.
    ConstraintSpec(
        "liner_buckling", "liner_buckling_margin", "inequality", "final_gate",
        "structures", 1.0, None,
        "NASA-SP-8087-sec-2.1.1.4;"
        "classical-long-plate-relation-screening-assumption",
        # SP-8087 names the failure mode; the long-plate relation and its
        # coefficient are a screening assumption pending SP-8068.
        basis_class="repository_rule",
    ),
    # Plan §0.1 rule 6 / §11: implicit states solved >= 100x tighter than the
    # optimizer feasibility tolerance.
    ConstraintSpec("engine_residual", "engine_residual_margin", "inequality",
                   "final_gate", "numerical", 1.0e-11, None, None,
                   basis_class="numerical_contract"),
    ConstraintSpec("cooling_residual", "cooling_residual_margin", "inequality",
                   "final_gate", "numerical", 1.0e-11, "K", None,
                   basis_class="numerical_contract"),
    ConstraintSpec("solver_status", "solver_status_margin", "availability_gate",
                   "final_gate", "numerical", 1.0, None, None,
                   basis_class="numerical_contract"),
    ConstraintSpec("finite", "finite_margin", "availability_gate", "final_gate",
                   "numerical", 1.0, None, None,
                   basis_class="numerical_contract"),
    ConstraintSpec("coolant_htd", "coolant_htd_margin", "availability_gate",
                   "final_gate", "thermal", 1.0, None,
                   "Nasuti-Pizzarelli-2021-sec-2.2-Eq9-PDF6", "htd_coolant",
                   basis_class="literature"),
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
    from raosim.mdo.propellants import WallSideMechanism, wall_side_mechanism_for

    coolant_name, coolant_resolved = _resolve_coolant(mission)
    # Prefer the propellant's *declared* wall-side mechanism.  Falling back to
    # the coolant-name list is unsafe on its own: "LNG", "ethane" and
    # "subcooled methane" all miss it and would be handed the coking limit
    # instead of the supercritical HTD criterion.  Catalog entries all declare
    # a mechanism, so the name list is now only a legacy backstop.
    declared = (
        wall_side_mechanism_for(getattr(mission, "propellant_name", ""))
        if coolant_resolved else None
    )
    if declared is WallSideMechanism.UNKNOWN:
        # Neither mechanism demonstrated: treat exactly like an unresolved
        # coolant so feasibility reduces to unknown rather than to a pass.
        coolant_resolved = False
        htd_applicable = False
    elif declared is not None:
        htd_applicable = declared is WallSideMechanism.SUPERCRITICAL_HTD
    else:
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
        elif spec.applicability in {"pintle_slotted", "pintle_movable"}:
            architecture = getattr(
                mission, "pintle_architecture", "fixed_slotted"
            )
            app = architecture == (
                "son_movable" if spec.applicability == "pintle_movable"
                else "fixed_slotted"
            )
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
    if reason == ConstraintReasonCode.CONTOUR_PROVIDER_FAILED:
        return (
            "the selected contour provider reported this candidate unavailable "
            "or infeasible; its fixed-shape grid is only a numerical carrier"
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
    "BASIS_CLASSES",
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
