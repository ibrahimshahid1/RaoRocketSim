"""One versioned engine-job input that every front end compiles into.

The guided wizard, ordinary CLI flags, and advanced JSON/YAML all produce an
:class:`EngineJobSpec`; :func:`resolve_engine_job` turns it into a
:class:`ResolvedEngineJob`.  Nothing downstream should branch on which front
end was used, because that is how ``--propellant`` and ``--mdo-propellant``
came to resolve the same concept differently.

``ResolvedEngineJob`` **contains** a
:class:`~raosim.mdo.resolved_inputs.ResolvedEngineInputs` rather than sitting
beside one.  A job is workflow intent (analyze / optimize / pareto, variable
policies, targets, output policy) layered on top of the physical inputs; if the
two were peers they would become a third competing definition of the same
conventions, which is precisely the failure this contract exists to end.

Design intent is always explicit.  A variable states its mode rather than being
inferred from whether a value happens to be present, and O/F intent is an enum
rather than an overloaded ``None`` -- the same lesson as the fixed-O/F
sentinel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from raosim.units import UnitError, parse_quantity

__all__ = [
    "ENGINE_JOB_SCHEMA_VERSION",
    "ArchitectureSelection",
    "ConstraintPolicy",
    "EngineJobSpec",
    "JobValidationError",
    "MissionRequest",
    "ModelSelection",
    "ObjectiveRequest",
    "OperatingPointSpec",
    "OutputPolicy",
    "ResolvedEngineJob",
    "TargetRelation",
    "TargetRequest",
    "VariableMode",
    "VariableRequest",
    "Workflow",
    "resolve_engine_job",
]

ENGINE_JOB_SCHEMA_VERSION = "1.1.0"


class JobValidationError(ValueError):
    """A job could not be validated into a runnable resolved form."""


class Workflow(str, Enum):
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    PARETO = "pareto"


class VariableMode(str, Enum):
    """How the optimizer may treat one design quantity."""

    FIXED = "fixed"          # held at an explicit value
    FREE = "free"            # optimized within the model's validity domain
    BOUNDED = "bounded"      # optimized within caller-supplied bounds
    DERIVED = "derived"      # computed by a documented closed-form resolver
    ENUMERATE = "enumerate"  # discrete outer loop, never in the gradient


class TargetRelation(str, Enum):
    EQUAL_TO = "equal_to"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    RANGE = "range"


class ConstraintPolicy(str, Enum):
    """Which physical constraints apply.

    The user chooses the design objective, not whether physical validity
    matters.  ``ALL_APPLICABLE`` is the only policy that may yield a
    ``physics_feasible`` verdict; an ablation produces
    ``custom_screen_feasible`` instead and can never claim requirements met.
    """

    ALL_APPLICABLE = "all_applicable"
    CUSTOM_ABLATION = "custom_ablation"


#: Design quantities a job may address, with the dimension each is parsed in.
#: A name outside this table is rejected rather than silently ignored.
VARIABLE_DIMENSIONS: Mapping[str, str] = {
    "Pc": "pressure",
    "eps": "dimensionless",
    "dp_f_frac": "dimensionless",
    "dp_o_frac": "dimensionless",
    "D_pintle": "length",
    "N_rpm": "rotational_speed",
    "channel_width": "length",
    "channel_height": "length",
    "film_frac": "dimensionless",
    "t_wall": "length",
    "OF": "dimensionless",
}

#: Quantities with a documented closed-form resolver, so ``DERIVED`` has a
#: defined meaning.  Anything else in DERIVED mode is rejected rather than
#: quietly treated as free.
DERIVABLE_VARIABLES: frozenset[str] = frozenset({"N_rpm"})

#: Discrete selections that must stay outside the gradient vector.
ENUMERABLE_SELECTIONS: frozenset[str] = frozenset(
    {"propellant", "liner_material", "closeout_material", "channel_count"}
)


def _quantity(value: Any, *, name: str, dimension: str) -> float:
    try:
        resolved = parse_quantity(value, dimension=dimension, field=name).value
    except UnitError as exc:
        raise JobValidationError(str(exc)) from exc
    # Unit conversion can overflow even when the written magnitude is finite.
    if not math.isfinite(resolved):
        raise JobValidationError(f"{name}: converted value must be finite")
    return resolved


@dataclass(frozen=True)
class VariableRequest:
    """An explicit policy for one design quantity.

    Intent is stated, never inferred from whether ``value`` is present.
    """

    name: str
    mode: VariableMode | str
    value: Any = None
    lower: Any = None
    upper: Any = None
    values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", VariableMode(self.mode))
        if self.name not in VARIABLE_DIMENSIONS:
            raise JobValidationError(
                f"unknown design variable {self.name!r}; known variables are "
                f"{', '.join(sorted(VARIABLE_DIMENSIONS))}"
            )
        dim = VARIABLE_DIMENSIONS[self.name]
        mode = self.mode
        if not isinstance(self.values, (list, tuple)):
            raise JobValidationError(f"{self.name}: values must be a sequence")
        supplied = {
            "value": self.value is not None,
            "lower": self.lower is not None,
            "upper": self.upper is not None,
            "values": bool(self.values),
        }
        accepted = {
            VariableMode.FIXED: {"value"},
            VariableMode.BOUNDED: {"lower", "upper"},
            VariableMode.FREE: set(),
            VariableMode.DERIVED: set(),
            VariableMode.ENUMERATE: {"values"},
        }[mode]
        unused = [key for key, present in supplied.items()
                  if present and key not in accepted]
        if unused:
            raise JobValidationError(
                f"{self.name}: mode {mode.value!r} does not consume "
                + ", ".join(unused)
            )

        if mode is VariableMode.FIXED:
            if self.value is None:
                raise JobValidationError(
                    f"{self.name}: mode 'fixed' requires an explicit value"
                )
            object.__setattr__(
                self, "value", _quantity(self.value, name=self.name, dimension=dim)
            )
        elif mode is VariableMode.BOUNDED:
            if self.lower is None or self.upper is None:
                raise JobValidationError(
                    f"{self.name}: mode 'bounded' requires both lower and upper"
                )
            lo = _quantity(self.lower, name=f"{self.name}.lower", dimension=dim)
            hi = _quantity(self.upper, name=f"{self.name}.upper", dimension=dim)
            if not lo < hi:
                raise JobValidationError(
                    f"{self.name}: lower ({lo}) must be below upper ({hi})"
                )
            object.__setattr__(self, "lower", lo)
            object.__setattr__(self, "upper", hi)
        elif mode is VariableMode.DERIVED:
            if self.name not in DERIVABLE_VARIABLES:
                raise JobValidationError(
                    f"{self.name}: mode 'derived' needs a documented resolver; "
                    f"only {', '.join(sorted(DERIVABLE_VARIABLES))} have one. "
                    "Use 'free' or 'bounded' instead of implying a derivation "
                    "that does not exist."
                )
        elif mode is VariableMode.ENUMERATE:
            if len(self.values) < 2:
                raise JobValidationError(
                    f"{self.name}: mode 'enumerate' needs at least two values"
                )
            object.__setattr__(self, "values", tuple(
                _quantity(v, name=self.name, dimension=dim) for v in self.values
            ))
            if len(set(self.values)) != len(self.values):
                raise JobValidationError(f"{self.name}: enumeration values must be distinct")
        values = ((self.value,) if mode is VariableMode.FIXED else
                  self.values if mode is VariableMode.ENUMERATE else ())
        for value in values:
            valid = (0.0 <= value < 1.0 if self.name == "film_frac" else
                     value > 1.0 if self.name == "eps" else value > 0.0)
            if not valid:
                raise JobValidationError(f"{self.name}: value is outside its physical range")


@dataclass(frozen=True)
class TargetRequest:
    """A requested output condition, expressed against a registered metric."""

    metric: str
    relation: TargetRelation | str
    value: Any = None
    upper_value: Any = None
    dimension: str | None = None
    condition: str | None = None

    def __post_init__(self) -> None:
        from raosim.mdo.metrics import MetricNotAvailable, resolve_metric

        object.__setattr__(self, "relation", TargetRelation(self.relation))
        # The registry owns each metric's dimension, so a target cannot be
        # parsed in units the metric does not have.
        try:
            definition = resolve_metric(self.metric, role="target")
        except (LookupError, MetricNotAvailable) as exc:
            raise JobValidationError(str(exc)) from exc
        if self.dimension is not None and self.dimension != definition.dimension:
            raise JobValidationError(
                f"{self.metric}: registered dimension is "
                f"{definition.dimension!r}, not {self.dimension!r}"
            )
        object.__setattr__(self, "dimension", definition.dimension)
        if self.relation is TargetRelation.RANGE:
            if self.value is None or self.upper_value is None:
                raise JobValidationError(
                    f"{self.metric}: relation 'range' needs value and upper_value"
                )
            lo = _quantity(self.value, name=self.metric, dimension=self.dimension)
            hi = _quantity(
                self.upper_value, name=self.metric, dimension=self.dimension
            )
            if not lo < hi:
                raise JobValidationError(
                    f"{self.metric}: range lower ({lo}) must be below upper ({hi})"
                )
            object.__setattr__(self, "value", lo)
            object.__setattr__(self, "upper_value", hi)
        else:
            if self.upper_value is not None:
                raise JobValidationError(
                    f"{self.metric}: upper_value is only consumed by 'range'"
                )
            if self.value is None:
                raise JobValidationError(
                    f"{self.metric}: relation {self.relation.value!r} needs a value"
                )
            object.__setattr__(self, "value", _quantity(
                self.value, name=self.metric, dimension=self.dimension
            ))


@dataclass(frozen=True)
class ObjectiveRequest:
    """The single primary scalar objective.

    Version 1 permits exactly one.  Additional desired outputs become targets,
    and two-objective studies use an epsilon-constraint Pareto sweep -- a
    weighted sum of physically incommensurate quantities has no defined
    meaning without stated exchange rates.
    """

    metric: str
    direction: Literal["min", "max"] = "min"

    def __post_init__(self) -> None:
        from raosim.mdo.metrics import MetricNotAvailable, resolve_metric

        if self.direction not in ("min", "max"):
            raise JobValidationError(
                f"objective direction must be 'min' or 'max'; got "
                f"{self.direction!r}"
            )
        # An objective must be differentiable, registered, and modellable.
        # "Minimize engine dry mass" has to fail here rather than silently
        # optimize a partial subtotal under a whole-engine name.
        try:
            resolve_metric(self.metric, role="objective")
        except (LookupError, MetricNotAvailable) as exc:
            raise JobValidationError(str(exc)) from exc
        if self.direction != "min":
            raise JobValidationError(
                f"{self.metric}: objective direction {self.direction!r} is not "
                "implemented; this objective is minimization-only"
            )


@dataclass(frozen=True)
class MissionRequest:
    """The SP-125 sec. 2.1 requirement set, in engineering notation."""

    thrust: Any
    # None retains omitted/default provenance so an explicit operating point
    # can supply these values without overriding a conflicting user request.
    thrust_condition: str | None = None
    altitude: Any = None
    burn_duration: Any = None
    qualification_duration: Any = None
    isp_min: Any = None
    dry_mass_max: Any = None
    envelope_diameter_max: Any = None
    envelope_length_max: Any = None

    def resolved(self) -> dict[str, Any]:
        condition = self.thrust_condition if self.thrust_condition is not None else "sea_level"
        out: dict[str, Any] = {
            "thrust": _quantity(self.thrust, name="thrust", dimension="force"),
            "thrust_condition": condition,
            "burn_duration": _quantity(
                self.burn_duration if self.burn_duration is not None else "120 s",
                name="burn_duration", dimension="time"
            ),
        }
        if condition not in ("sea_level", "vacuum", "altitude"):
            raise JobValidationError(
                "thrust_condition must be 'sea_level', 'vacuum' or 'altitude'; "
                f"got {self.thrust_condition!r}"
            )
        if condition == "altitude":
            if self.altitude is None:
                raise JobValidationError(
                    "thrust_condition 'altitude' requires an altitude"
                )
            out["altitude"] = _quantity(
                self.altitude, name="altitude", dimension="length"
            )
            # The existing simplified atmosphere reproduces its stated lapse
            # layers through 32 km. Beyond that it does not implement the next
            # ISA lapse segment, and its high-altitude tail is approximate.
            if not 0.0 <= out["altitude"] <= 32000.0:
                raise JobValidationError(
                    "altitude must be in [0, 32000] m for the selected "
                    "raosim.atmosphere simplified ISA model; supply a single "
                    "operating point with explicit ambient_pressure otherwise"
                )
        elif self.altitude is not None:
            raise JobValidationError(
                "altitude is only consumed when thrust_condition is 'altitude'"
            )
        optional = (
            ("qualification_duration", "time"),
            ("isp_min", "time"),
            ("dry_mass_max", "mass"),
            ("envelope_diameter_max", "length"),
            ("envelope_length_max", "length"),
        )
        for name, dim in optional:
            raw = getattr(self, name)
            if raw is not None:
                out[name] = _quantity(raw, name=name, dimension=dim)
        for name, value in out.items():
            if name not in {"thrust_condition", "altitude"} and value <= 0.0:
                raise JobValidationError(f"{name} must be positive")
        qual = out.get("qualification_duration")
        if qual is not None and qual < out["burn_duration"]:
            raise JobValidationError(
                f"qualification duration ({qual} s) cannot be below the flight "
                f"duration ({out['burn_duration']} s)"
            )
        return out


@dataclass(frozen=True)
class ArchitectureSelection:
    """Discrete selections.  These never enter the gradient vector."""

    propellant: str = "LOX/RP-1"
    feed: str = "electric_pump"
    injector: str = "pintle"
    cooling: str = "regenerative"
    liner_material: str | None = None
    closeout_material: str | None = None

    def __post_init__(self) -> None:
        from raosim.mdo.capabilities import require_capability

        for name in ("propellant", "liner_material", "closeout_material"):
            value = getattr(self, name)
            if value is None and name != "propellant":
                continue
            choices = value if isinstance(value, (list, tuple)) else (value,)
            if not choices or any(not isinstance(item, str) or not item.strip() for item in choices):
                raise JobValidationError(f"{name} requires a name or a nonempty sequence of names")
        for name in ("feed", "injector", "cooling"):
            try:
                require_capability(name, getattr(self, name))
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        if self.closeout_material is not None and self.liner_material is None:
            raise JobValidationError(
                "closeout_material requires liner_material so both selected "
                "materials can reach the wall model"
            )


@dataclass(frozen=True)
class ModelSelection:
    """Model/coverage choices that decide what a verdict is allowed to say."""

    chamber_property_table: str | None = None
    mixture_ratio_mode: str = "nominal"
    contour_provider: str = "rao_top_bezier_fixed_topology"
    postsolve_contour_validator: str | None = None
    rao_convergence_domain: str | None = None
    contour_max_radius_error_over_rt: float | None = None
    contour_rms_radius_error_over_rt: float | None = None
    contour_relative_length_error: float | None = None
    contour_relative_cf_error: float | None = None
    constraint_policy: ConstraintPolicy | str = ConstraintPolicy.ALL_APPLICABLE
    allow_incomplete_physics: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "constraint_policy", ConstraintPolicy(self.constraint_policy)
        )
        if not isinstance(self.allow_incomplete_physics, bool):
            raise JobValidationError("allow_incomplete_physics must be a boolean")
        if self.constraint_policy is ConstraintPolicy.CUSTOM_ABLATION:
            raise JobValidationError(
                "custom_ablation is not implemented: no constraint-selection "
                "mask is available; use all_applicable"
            )
        if self.chamber_property_table is not None and (
            not isinstance(self.chamber_property_table, str)
            or not self.chamber_property_table.strip()
        ):
            raise JobValidationError("chamber_property_table must be a nonempty path")
        if self.mixture_ratio_mode not in ("nominal", "pinned", "optimize"):
            raise JobValidationError(
                "mixture_ratio_mode must be 'nominal', 'pinned' or 'optimize'; "
                f"got {self.mixture_ratio_mode!r}"
            )
        # Model selection is an identity, not a hint.  Reject unknown contour
        # providers here so execution can never interpret one as a request to
        # fall back to the analytic TOP/Bézier path.
        from raosim.mdo.nozzle import resolve_contour_provider

        if not isinstance(self.contour_provider, str):
            raise JobValidationError("contour_provider must be a provider id string")
        try:
            resolve_contour_provider(self.contour_provider)
        except (TypeError, ValueError) as exc:
            raise JobValidationError(str(exc)) from exc

        # The variational method is deliberately a once-per-candidate host
        # validator, not another JAX inner-loop contour provider.  Selection is
        # nevertheless an exact model identity: a misspelled id, a missing
        # hash-checked domain artifact, or an unstated discrepancy threshold is
        # rejected instead of being interpreted as "use TOP only".
        validator = self.postsolve_contour_validator
        validation_fields = {
            "rao_convergence_domain": self.rao_convergence_domain,
            "contour_max_radius_error_over_rt": (
                self.contour_max_radius_error_over_rt
            ),
            "contour_rms_radius_error_over_rt": (
                self.contour_rms_radius_error_over_rt
            ),
            "contour_relative_length_error": self.contour_relative_length_error,
            "contour_relative_cf_error": self.contour_relative_cf_error,
        }
        if validator is None:
            supplied = [name for name, value in validation_fields.items()
                        if value is not None]
            if supplied:
                raise JobValidationError(
                    "post-solve contour-validation settings were supplied "
                    "without postsolve_contour_validator: "
                    + ", ".join(supplied)
                )
        else:
            from raosim.mdo.contour_validation import (
                ContourDiscrepancyTolerances,
                VARIATIONAL_VALIDATOR_MODEL_ID,
            )
            from raosim.mdo.nozzle import ANALYTIC_TOP_PROVIDER_ID

            if validator != VARIATIONAL_VALIDATOR_MODEL_ID:
                raise JobValidationError(
                    f"unknown post-solve contour validator {validator!r}; "
                    f"available: {VARIATIONAL_VALIDATOR_MODEL_ID!r}"
                )
            if self.contour_provider != ANALYTIC_TOP_PROVIDER_ID:
                raise JobValidationError(
                    f"{validator} compares the optimized analytic TOP wall and "
                    f"therefore requires contour_provider="
                    f"{ANALYTIC_TOP_PROVIDER_ID!r}"
                )
            if not isinstance(self.rao_convergence_domain, str) or not (
                self.rao_convergence_domain.strip()
            ):
                raise JobValidationError(
                    f"{validator} requires rao_convergence_domain"
                )
            missing = [
                name for name, value in validation_fields.items()
                if name != "rao_convergence_domain" and value is None
            ]
            if missing:
                raise JobValidationError(
                    f"{validator} requires explicit discrepancy thresholds: "
                    + ", ".join(missing)
                )
            try:
                ContourDiscrepancyTolerances(
                    max_radius_error_over_rt=(
                        self.contour_max_radius_error_over_rt
                    ),
                    rms_radius_error_over_rt=(
                        self.contour_rms_radius_error_over_rt
                    ),
                    relative_divergent_length_error=(
                        self.contour_relative_length_error
                    ),
                    relative_thrust_coefficient_error=(
                        self.contour_relative_cf_error
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise JobValidationError(
                    f"invalid post-solve contour-validation policy: {exc}"
                ) from exc
        if (
            self.mixture_ratio_mode == "optimize"
            and not self.chamber_property_table
        ):
            raise JobValidationError(
                "mixture_ratio_mode 'optimize' requires a validated "
                "chamber_property_table: without O/F-dependent thermochemistry "
                "the mixture ratio is not a physical lever"
            )


@dataclass(frozen=True)
class OperatingPointSpec:
    """One condition the engine must satisfy.

    Multipoint optimization is not implemented; a job carrying more than one
    point is rejected rather than silently analyzed at the first, so a caller
    cannot believe a throttle range was honoured when it was not.
    """

    id: str
    ambient_pressure: Any
    thrust_fraction: float = 1.0
    duration: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise JobValidationError("operating point id must be a nonempty string")
        pressure = _quantity(
            self.ambient_pressure, name=f"{self.id}.ambient_pressure",
            dimension="pressure",
        )
        if pressure < 0.0:
            raise JobValidationError("ambient_pressure must be nonnegative and absolute")
        fraction = _quantity(
            self.thrust_fraction, name=f"{self.id}.thrust_fraction",
            dimension="dimensionless",
        )
        if fraction != 1.0:
            raise JobValidationError(
                "thrust_fraction other than 1 is not implemented: the solver "
                "sizes one design point and cannot evaluate throttled operation "
                "of the same hardware"
            )
        if self.duration is not None:
            duration = _quantity(self.duration, name=f"{self.id}.duration", dimension="time")
            if duration <= 0.0:
                raise JobValidationError("operating point duration must be positive")


@dataclass(frozen=True)
class OutputPolicy:
    """What artifacts a run should produce, and at what fidelity."""

    parts: str = "none"
    cad_formats: tuple[str, ...] = ()
    cad_fidelity: str = "L0_numerical_only"
    reports: tuple[str, ...] = ()
    output_dir: str | None = None
    allow_partial_package: bool = False

    def __post_init__(self) -> None:
        from raosim.mdo.capabilities import require_capability

        if not isinstance(self.allow_partial_package, bool):
            raise JobValidationError("allow_partial_package must be a boolean")
        for kind, value in (("parts", self.parts), ("cad_fidelity", self.cad_fidelity)):
            try:
                require_capability(kind, value)
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        if not isinstance(self.cad_formats, (list, tuple)):
            raise JobValidationError("cad_formats must be a sequence of format names")
        if self.cad_formats:
            raise JobValidationError(
                "cad_formats are not implemented by the unified job execution "
                "service; no CAD request will be silently omitted"
            )
        if not isinstance(self.reports, (list, tuple)):
            raise JobValidationError("reports must be a sequence of report names")
        for report in self.reports:
            try:
                require_capability("report", report)
            except ValueError as exc:
                raise JobValidationError(str(exc)) from exc
        if self.output_dir is not None and (
            not isinstance(self.output_dir, str) or not self.output_dir.strip()
        ):
            raise JobValidationError("output_dir must be a nonempty directory path")
        if self.reports and self.output_dir is None:
            raise JobValidationError("requested reports require output_dir")
        object.__setattr__(self, "reports", tuple(self.reports))
        object.__setattr__(self, "cad_formats", tuple(self.cad_formats))


@dataclass(frozen=True)
class EngineJobSpec:
    """The one versioned job every front end compiles into."""

    mission: MissionRequest
    workflow: Workflow | str = Workflow.ANALYZE
    architecture: ArchitectureSelection = field(
        default_factory=ArchitectureSelection
    )
    models: ModelSelection = field(default_factory=ModelSelection)
    variables: tuple[VariableRequest, ...] = ()
    targets: tuple[TargetRequest, ...] = ()
    objective: ObjectiveRequest | None = None
    operating_points: tuple[OperatingPointSpec, ...] = ()
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    schema_version: str = ENGINE_JOB_SCHEMA_VERSION
    # Retain custom catalog definitions in the submitted job and its digest,
    # so replay does not depend on an earlier process's registry mutation.
    custom_propellants: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        from raosim.mdo.capabilities import require_capability

        try:
            require_capability("schema_version", self.schema_version)
        except ValueError as exc:
            raise JobValidationError(str(exc)) from exc
        object.__setattr__(self, "workflow", Workflow(self.workflow))
        if (not isinstance(self.custom_propellants, (list, tuple))
                or any(not isinstance(item, Mapping) for item in self.custom_propellants)):
            raise JobValidationError("custom_propellants must be a sequence of definitions")
        object.__setattr__(self, "custom_propellants", tuple(dict(item) for item in self.custom_propellants))
        names = [v.name for v in self.variables]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise JobValidationError(
                "a design variable may carry only one policy; duplicated: "
                + ", ".join(sorted(duplicates))
            )
        if self.workflow in (Workflow.OPTIMIZE, Workflow.PARETO):
            if self.objective is None:
                raise JobValidationError(
                    f"workflow {self.workflow.value!r} requires an objective"
                )
        if self.workflow is Workflow.ANALYZE and self.objective is not None:
            raise JobValidationError(
                "workflow 'analyze' evaluates a stated design and cannot carry "
                "an objective; use 'optimize'"
            )
        of_mode = self.models.mixture_ratio_mode
        of_var = next((v for v in self.variables if v.name == "OF"), None)
        if of_var is not None and of_mode == "nominal":
            raise JobValidationError(
                "an OF variable policy requires models.mixture_ratio_mode="
                f"'optimize'; got {of_mode!r}"
            )
        if of_mode == "pinned" and (
            of_var is None or of_var.mode is not VariableMode.FIXED
        ):
            raise JobValidationError(
                "mixture_ratio_mode 'pinned' requires a fixed OF variable value"
            )
        if len(self.operating_points) > 1:
            raise JobValidationError(
                "multipoint operation is not implemented: this job states "
                f"{len(self.operating_points)} operating points but the solver "
                "is single-design-point. Submit one point, or wait for the "
                "multipoint layer rather than receiving a single-point answer "
                "labelled as a throttle envelope."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_document(self) -> dict[str, Any]:
        """A replayable CLI document, restoring units to normalized scalars."""
        from raosim.units import canonical_unit

        def quantity(value: Any, dimension: str) -> Any:
            if value is None:
                return None
            unit = canonical_unit(dimension)
            return f"{float(value):.17g} {unit}" if unit else float(value)

        document = self.to_dict()
        document["variables"] = {}
        for variable in self.variables:
            payload = asdict(variable)
            payload.pop("name")
            dimension = VARIABLE_DIMENSIONS[variable.name]
            for key in ("value", "lower", "upper"):
                payload[key] = quantity(payload[key], dimension)
            payload["values"] = [quantity(value, dimension) for value in variable.values]
            document["variables"][variable.name] = payload
        for target in document["targets"]:
            for key in ("value", "upper_value"):
                target[key] = quantity(target[key], target["dimension"])
        return document

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedEngineJob:
    """A validated job plus the physical inputs it resolves to.

    ``inputs`` is present only once a design point exists (an ``analyze`` job,
    or an optimum). For an unsolved ``optimize`` job it is ``None`` and
    ``mission`` plus ``active_variables`` describe the problem to be solved.
    """

    spec: EngineJobSpec
    mission: Any
    active_variables: tuple[str, ...]
    fixed_variables: Mapping[str, float]
    bounds: Mapping[str, tuple[float, float]]
    enumerations: Mapping[str, tuple[Any, ...]]
    constraint_policy: ConstraintPolicy
    effective_targets: tuple[TargetRequest, ...] = ()
    requirement_coverage: tuple[Mapping[str, Any], ...] = ()
    rao_convergence_domain_sha256: str | None = None
    inputs: Any = None
    notes: tuple[str, ...] = ()

    @property
    def job_digest(self) -> str:
        spec_digest = self.spec.digest()
        if self.rao_convergence_domain_sha256 is None:
            return spec_digest
        payload = {
            "engine_job_spec_sha256": spec_digest,
            "rao_convergence_domain_sha256": (
                self.rao_convergence_domain_sha256
            ),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def design_vector(self) -> Any:
        """The design point for an ``analyze`` job.

        Only defined once every variable is fixed; an ``optimize`` job has no
        single design point until it is solved.
        """
        import jax.numpy as jnp

        from raosim.mdo.schema import DesignVector

        if self.active_variables:
            raise JobValidationError(
                "a design vector exists only when every variable is fixed; "
                f"still active: {', '.join(self.active_variables)}"
            )
        layout = self.mission.design_layout()
        missing = [n for n in layout.active_names if n not in self.fixed_variables]
        if missing:
            raise JobValidationError(
                "fixed design point is incomplete; missing: "
                + ", ".join(missing)
            )
        values = jnp.asarray(
            [float(self.fixed_variables[n]) for n in layout.active_names]
        )
        return DesignVector.from_active_array(
            values,
            layout,
            fixed_of=None if layout.of_is_variable else float(self.mission.OF),
        )

    def with_inputs(self, inputs: Any) -> "ResolvedEngineJob":
        """Attach the resolved physical inputs once a design point exists.

        This is the containment relationship made concrete: the job carries the
        workflow intent, and the physical conventions live in one
        ``ResolvedEngineInputs`` rather than being restated here.
        """
        from dataclasses import replace as _replace

        return _replace(self, inputs=inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.spec.schema_version,
            "job_digest": self.job_digest,
            "workflow": self.spec.workflow.value,
            "active_variables": list(self.active_variables),
            "fixed_variables": dict(self.fixed_variables),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "enumerations": {k: list(v) for k, v in self.enumerations.items()},
            "constraint_policy": self.constraint_policy.value,
            "effective_targets": [asdict(target) for target in self.effective_targets],
            "requirement_coverage": [dict(row) for row in self.requirement_coverage],
            "resolved_operating_conditions": {
                "thrust_N": self.mission.thrust,
                "ambient_pressure_Pa": self.mission.Pa,
                "burn_duration_s": self.mission.burn_time,
            },
            "rao_convergence_domain_sha256": (
                self.rao_convergence_domain_sha256
            ),
            "resolved_inputs": (
                self.inputs.to_dict() if self.inputs is not None else None
            ),
            "notes": list(self.notes),
        }


def resolve_engine_job(spec: EngineJobSpec) -> ResolvedEngineJob:
    """Validate a job and resolve it into a runnable, replayable form."""
    try:
        return _resolve_engine_job(spec)
    except JobValidationError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise JobValidationError(str(exc)) from exc


def _resolve_engine_job(spec: EngineJobSpec) -> ResolvedEngineJob:
    for entry in spec.custom_propellants:
        from raosim.mdo.propellants import PropellantSpec, WallSideMechanism, register_propellant

        payload = dict(entry)
        evidence = payload.pop("evidence", None)
        mechanism = payload.pop("wall_side_mechanism", None)
        if mechanism is not None:
            payload["wall_side_mechanism"] = WallSideMechanism(mechanism)
        register_propellant(PropellantSpec(**payload), evidence=evidence, replace_existing=True)

    from raosim.mdo.schema import MissionSpec, default_design_space

    from dataclasses import replace as _replace

    request = spec.mission
    if spec.operating_points and request.burn_duration is None:
        point_duration = spec.operating_points[0].duration
        if point_duration is not None:
            request = _replace(request, burn_duration=point_duration)
    requirement = request.resolved()
    notes: list[str] = []
    coverage: list[Mapping[str, Any]] = []
    effective_targets = list(spec.targets)
    rao_convergence_domain_sha256: str | None = None

    # This mapping is deliberately total over the request schema, including
    # currently unavailable requirements. Adding a field without declaring its
    # consumer must fail resolution rather than dropping a new user request.
    mission_consumers = {
        "thrust": "mission.thrust",
        "thrust_condition": "mission.Pa",
        "altitude": "raosim.atmosphere.pressure -> mission.Pa",
        "burn_duration": "mission.burn_time",
        "qualification_duration": "qualification test evidence",
        "isp_min": "performance.isp_delivered target",
        "dry_mass_max": "masses.engine_dry_mass",
        "envelope_diameter_max": "geometry.installed_envelope",
        "envelope_length_max": "geometry.installed_envelope",
    }
    if set(mission_consumers) != {item.name for item in fields(MissionRequest)}:
        raise JobValidationError("MissionRequest fields lack a total input-consumer mapping")
    for name, value in requirement.items():
        row: dict[str, Any] = {
            "input": f"mission.{name}", "value": value,
            "consumer": mission_consumers[name], "status": "supported",
        }
        if name in {"dry_mass_max", "envelope_diameter_max", "envelope_length_max"}:
            from raosim.mdo.metrics import METRIC_REGISTRY

            metric = METRIC_REGISTRY[mission_consumers[name]]
            raise JobValidationError(
                f"mission.{name} is unavailable: {metric.unavailable_reason}"
            )
        if name == "qualification_duration":
            row.update(status="unavailable", reason=(
                "qualification_duration requires configuration-bound test evidence; "
                "a steady design calculation cannot establish qualified duration"
            ))
            notes.append(str(row["reason"]))
        if name == "isp_min":
            effective_targets.append(TargetRequest(
                "performance.isp_delivered", TargetRelation.AT_LEAST,
                f"{value:.17g} s",
            ))
        coverage.append(row)

    mission = MissionSpec.for_propellant(
        spec.architecture.propellant, requirement["thrust"]
    )
    overrides: dict[str, Any] = {"burn_time": requirement["burn_duration"]}
    if requirement["thrust_condition"] == "vacuum":
        overrides["Pa"] = 0.0
    elif requirement["thrust_condition"] == "altitude":
        from raosim.atmosphere import pressure

        overrides["Pa"] = pressure(requirement["altitude"])
        notes.append(
            "ambient pressure resolved from raosim.atmosphere.pressure "
            "(simplified ISA, accepted altitude range 0–32000 m): "
            f"{requirement['altitude']:g} m -> {overrides['Pa']:.9g} Pa"
        )
    if spec.operating_points:
        point = spec.operating_points[0]
        ambient = _quantity(point.ambient_pressure, name=f"{point.id}.ambient_pressure",
                            dimension="pressure")
        if spec.mission.thrust_condition is not None and not math.isclose(
            ambient, overrides.get("Pa", mission.Pa), rel_tol=1e-9, abs_tol=1e-6
        ):
            raise JobValidationError(
                "operating point ambient_pressure conflicts with mission "
                "thrust_condition; one job cannot size different reference conditions"
            )
        overrides["Pa"] = ambient
        notes.append(
            f"operating point {point.id!r} explicitly selects the single design "
            f"ambient pressure ({ambient:g} Pa); this takes precedence over "
            "an omitted ambient condition and does not establish off-design coverage"
        )
        for name, value in (("id", point.id), ("ambient_pressure", ambient),
                            ("thrust_fraction", 1.0)):
            coverage.append({"input": f"operating_points.{point.id}.{name}",
                             "value": value, "status": "supported",
                             "consumer": "single design operating point"})
        if point.duration is not None:
            duration = _quantity(point.duration, name=f"{point.id}.duration", dimension="time")
            if duration != requirement["burn_duration"]:
                raise JobValidationError(
                    "operating point duration conflicts with mission burn_duration; "
                    "set both to the intended single design-point duration"
                )
            coverage.append({"input": f"operating_points.{point.id}.duration",
                             "value": duration, "status": "supported",
                             "consumer": "mission.burn_time"})
            overrides["burn_time"] = duration
    selected_condition = (spec.operating_points[0].id if spec.operating_points
                          else requirement["thrust_condition"])
    for target in effective_targets:
        if target.condition is not None and target.condition != selected_condition:
            raise JobValidationError(
                f"target {target.metric}: condition {target.condition!r} is not "
                f"the selected single operating condition {selected_condition!r}"
            )
    if spec.models.chamber_property_table:
        overrides["cea_table_path"] = str(spec.models.chamber_property_table)
    overrides["of_is_pinned"] = spec.models.mixture_ratio_mode != "optimize"
    if spec.models.mixture_ratio_mode == "pinned":
        overrides["OF"] = float(next(v.value for v in spec.variables if v.name == "OF"))
        if overrides["OF"] <= 0.0:
            raise JobValidationError("pinned OF must be positive")
        if not spec.models.chamber_property_table and not math.isclose(overrides["OF"], mission.OF):
            raise JobValidationError(
                "a nonnominal pinned OF requires a validated chamber_property_table; "
                "catalog gas properties describe only the nominal mixture ratio"
            )
    mission = _replace(mission, **overrides)

    # Validate model selection during job resolution, including nominal or
    # pinned jobs whose design space does not otherwise need to load a table.
    # This makes ``lrekit validate`` fail closed for a missing explicit path or
    # an unavailable catalog pair instead of deferring the error to execution.
    if mission.cea_table_path:
        from raosim.mdo.properties import load_chamber_surfaces

        load_chamber_surfaces(
            mission.cea_table_path,
            expected_propellant=mission.propellant_name,
            require_of_dependence=not mission.of_is_pinned,
        )

    if spec.models.postsolve_contour_validator is not None:
        from raosim.rao_existence_scan import RaoConvergenceDomain

        domain_path = str(spec.models.rao_convergence_domain)
        try:
            domain = RaoConvergenceDomain.read_json(domain_path)
        except (OSError, TypeError, ValueError) as exc:
            raise JobValidationError(
                "post-solve Rao convergence-domain artifact is invalid: "
                f"{exc}"
            ) from exc
        notes.append(
            "post-solve contour validation selected: "
            f"{spec.models.postsolve_contour_validator}; B5 domain "
            f"sha256={domain.artifact_sha256}. The analytic TOP contour "
            "remains the differentiable inner-loop geometry."
        )
        rao_convergence_domain_sha256 = domain.artifact_sha256

    liner = spec.architecture.liner_material
    closeout = spec.architecture.closeout_material
    if liner is not None:
        mission = mission.with_materials(
            liner=liner, closeout=closeout or "Inconel 718"
        )
    else:
        notes.append(
            "no liner material was selected, so the traced wall constants are "
            "class defaults that correspond to no single catalog alloy"
        )

    space = {v.name: v for v in default_design_space(mission)}
    policies = {v.name: v for v in spec.variables
                if not (v.name == "OF" and spec.models.mixture_ratio_mode == "pinned")}

    fixed: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    enumerations: dict[str, tuple[Any, ...]] = {}
    active: list[str] = []
    defaulted: list[str] = []

    for name, var in space.items():
        policy = policies.get(name)
        if policy is None:
            # No stated policy: analyze holds the reference point, optimize
            # searches the model's own validity domain.  A defaulted analyze
            # variable is recorded by name below -- it is a value the user did
            # not choose, and a design report must not present it as one.
            if spec.workflow is Workflow.ANALYZE:
                fixed[name] = float(var.ref())
                defaulted.append(name)
            else:
                active.append(name)
                bounds[name] = (float(var.lower), float(var.upper))
            continue
        if policy.mode is VariableMode.FIXED:
            fixed[name] = float(policy.value)
        elif policy.mode is VariableMode.FREE:
            active.append(name)
            bounds[name] = (float(var.lower), float(var.upper))
        elif policy.mode is VariableMode.BOUNDED:
            lo, hi = float(policy.lower), float(policy.upper)
            if lo < float(var.lower) or hi > float(var.upper):
                notes.append(
                    f"{name}: requested bounds [{lo:g}, {hi:g}] extend beyond "
                    f"the configured search domain [{float(var.lower):g}, "
                    f"{float(var.upper):g}]; the search is clipped to the "
                    "domain; these search bounds do not establish physical validation"
                )
                lo = max(lo, float(var.lower))
                hi = min(hi, float(var.upper))
                if not lo < hi:
                    raise JobValidationError(
                        f"{name}: requested bounds do not intersect the "
                        "configured search domain"
                    )
            active.append(name)
            bounds[name] = (lo, hi)
        elif policy.mode is VariableMode.ENUMERATE:
            enumerations[name] = policy.values
        elif policy.mode is VariableMode.DERIVED:
            raise JobValidationError(
                f"{name}: derived mode has no implemented job-level resolver; "
                "state a fixed value or choose free/bounded optimization"
            )

    unknown = set(policies) - set(space)
    if unknown:
        # ``OF`` is legitimately absent from the space unless a validated
        # O/F-dependent table made it a lever.
        raise JobValidationError(
            "these variables are not in the design space for this "
            f"configuration: {', '.join(sorted(unknown))}. "
            "OF becomes a variable only with a validated chamber property "
            "table."
        )

    if spec.workflow is Workflow.ANALYZE and active:
        raise JobValidationError(
            "workflow 'analyze' needs every design variable fixed; unresolved: "
            + ", ".join(sorted(active))
        )
    if defaulted:
        notes.append(
            "these variables were not stated and took the design-space "
            "reference value, so they are defaults rather than choices: "
            + ", ".join(sorted(defaulted))
            + ". State them explicitly, or use workflow 'optimize' to let the "
            "solver select them."
        )

    # Milling practice is not a physical constraint, so it is reported rather
    # than enforced -- but a search box whose corner cannot be milled will
    # eventually be explored, and that must not be silent.
    from raosim.manufacturing_limits import check_channel_geometry, check_design_space

    for finding in check_design_space(space.values()):
        notes.append(f"manufacturability: {finding.message} [{finding.source_id}]")
    if fixed:
        for finding in check_channel_geometry(
            channel_width=fixed.get("channel_width"),
            channel_height=fixed.get("channel_height"),
        ):
            notes.append(
                f"manufacturability: {finding.message} [{finding.source_id}]"
            )

    if spec.models.constraint_policy is ConstraintPolicy.CUSTOM_ABLATION:
        notes.append(
            "constraint policy is a custom ablation, so any verdict is "
            "'custom_screen_feasible' and never 'physics_feasible'"
        )

    return ResolvedEngineJob(
        spec=spec,
        mission=mission,
        active_variables=tuple(active),
        fixed_variables=fixed,
        bounds=bounds,
        enumerations=enumerations,
        constraint_policy=spec.models.constraint_policy,
        effective_targets=tuple(effective_targets),
        requirement_coverage=tuple(coverage),
        rao_convergence_domain_sha256=rao_convergence_domain_sha256,
        inputs=None,
        notes=tuple(notes),
    )
