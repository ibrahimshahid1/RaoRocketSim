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
from dataclasses import asdict, dataclass, field
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

ENGINE_JOB_SCHEMA_VERSION = "1.0.0"


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
        return parse_quantity(value, dimension=dimension, field=name).value
    except UnitError as exc:
        raise JobValidationError(str(exc)) from exc


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
    thrust_condition: str = "sea_level"
    altitude: Any = None
    burn_duration: Any = "120 s"
    qualification_duration: Any = None
    isp_min: Any = None
    dry_mass_max: Any = None
    envelope_diameter_max: Any = None
    envelope_length_max: Any = None

    def resolved(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "thrust": _quantity(self.thrust, name="thrust", dimension="force"),
            "thrust_condition": str(self.thrust_condition),
            "burn_duration": _quantity(
                self.burn_duration, name="burn_duration", dimension="time"
            ),
        }
        if self.thrust_condition not in ("sea_level", "vacuum", "altitude"):
            raise JobValidationError(
                "thrust_condition must be 'sea_level', 'vacuum' or 'altitude'; "
                f"got {self.thrust_condition!r}"
            )
        if self.thrust_condition == "altitude":
            if self.altitude is None:
                raise JobValidationError(
                    "thrust_condition 'altitude' requires an altitude"
                )
            out["altitude"] = _quantity(
                self.altitude, name="altitude", dimension="length"
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


@dataclass(frozen=True)
class ModelSelection:
    """Model/coverage choices that decide what a verdict is allowed to say."""

    chamber_property_table: str | None = None
    mixture_ratio_mode: str = "nominal"
    contour_provider: str = "rao_top_bezier_fixed_topology"
    constraint_policy: ConstraintPolicy | str = ConstraintPolicy.ALL_APPLICABLE
    allow_incomplete_physics: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "constraint_policy", ConstraintPolicy(self.constraint_policy)
        )
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


@dataclass(frozen=True)
class OutputPolicy:
    """What artifacts a run should produce, and at what fidelity."""

    parts: str = "none"
    cad_formats: tuple[str, ...] = ()
    cad_fidelity: str = "L0_numerical_only"
    reports: tuple[str, ...] = ()
    output_dir: str | None = None
    allow_partial_package: bool = False


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow", Workflow(self.workflow))
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
        if of_var is not None and of_mode != "optimize":
            raise JobValidationError(
                "an OF variable policy requires models.mixture_ratio_mode="
                f"'optimize'; got {of_mode!r}"
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
    inputs: Any = None
    notes: tuple[str, ...] = ()

    @property
    def job_digest(self) -> str:
        return self.spec.digest()

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
            "resolved_inputs": (
                self.inputs.to_dict() if self.inputs is not None else None
            ),
            "notes": list(self.notes),
        }


def resolve_engine_job(spec: EngineJobSpec) -> ResolvedEngineJob:
    """Validate a job and resolve it into a runnable, replayable form."""

    from raosim.mdo.schema import MissionSpec, default_design_space

    requirement = spec.mission.resolved()
    notes: list[str] = []

    mission = MissionSpec.for_propellant(
        spec.architecture.propellant, requirement["thrust"]
    )
    overrides: dict[str, Any] = {"burn_time": requirement["burn_duration"]}
    if requirement["thrust_condition"] == "vacuum":
        overrides["Pa"] = 0.0
    if spec.models.chamber_property_table:
        overrides["cea_table_path"] = str(spec.models.chamber_property_table)
    overrides["of_is_pinned"] = spec.models.mixture_ratio_mode != "optimize"
    from dataclasses import replace as _replace

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
    policies = {v.name: v for v in spec.variables}

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
                    f"the validated model domain [{float(var.lower):g}, "
                    f"{float(var.upper):g}]; the search is clipped to the "
                    "domain because outside it the models are not valid"
                )
                lo = max(lo, float(var.lower))
                hi = min(hi, float(var.upper))
                if not lo < hi:
                    raise JobValidationError(
                        f"{name}: requested bounds do not intersect the "
                        "validated model domain"
                    )
            active.append(name)
            bounds[name] = (lo, hi)
        elif policy.mode is VariableMode.ENUMERATE:
            enumerations[name] = policy.values
        elif policy.mode is VariableMode.DERIVED:
            notes.append(
                f"{name}: resolved by its documented closed-form relation "
                "rather than optimized"
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
        inputs=None,
        notes=tuple(notes),
    )
