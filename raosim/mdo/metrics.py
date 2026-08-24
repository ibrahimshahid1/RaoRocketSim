"""The registry of selectable objectives and targets.

"Any user-selected output target" cannot safely mean any arbitrary state
field.  A quantity is selectable only if something can actually be said about
it: where it lives in the snapshot contract, whether it is differentiable,
whether a counterpart exists in the traditional pipeline, and which models must
be available before a verdict about it means anything.

Two structural rules keep this from becoming a third competing definition:

* **Constraint metrics are derived from** :data:`~raosim.mdo.constraints
  .CONSTRAINT_MANIFEST`, never re-listed.  A constraint is a metric plus a
  relation, so the manifest stays the single authority on names, scales,
  applicability and optimizer role.
* **Every reportable metric is validated against**
  :data:`~raosim.mdo.snapshot.SNAPSHOT_FIELD_MANIFEST`.  A registry entry
  pointing at a snapshot field that does not exist is worse than no entry, so
  the module refuses to build one.

Quantities the repository cannot yet model are registered as *unavailable with
a reason* rather than omitted.  Omission reads as "not thought about"; an
explicit unavailable row lets a catalog command explain why full engine dry
mass, cost, or reliability is not on offer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

__all__ = [
    "METRIC_REGISTRY",
    "MetricDefinition",
    "MetricNotAvailable",
    "objective_capable_metrics",
    "resolve_metric",
    "target_capable_metrics",
    "unavailable_metrics",
]


class MetricNotAvailable(LookupError):
    """A metric is known but cannot be optimized or targeted yet."""


@dataclass(frozen=True)
class MetricDefinition:
    """One selectable output quantity."""

    id: str
    dimension: str
    snapshot_section: str | None
    snapshot_field: str | None
    differentiable: bool = False
    objective_capable: bool = False
    target_capable: bool = False
    #: ``EngineResult`` constraint key when this metric is a constraint margin.
    engine_key: str | None = None
    #: Traditional-pipeline counterpart, for parity comparison.
    exact_counterpart: str | None = None
    required_models: tuple[str, ...] = ()
    source_id: str | None = None
    unavailable_reason: str | None = None
    category: str = "reported"

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def snapshot_path(self) -> str | None:
        if self.snapshot_section is None or self.snapshot_field is None:
            return None
        return f"{self.snapshot_section}.{self.snapshot_field}"


# --------------------------------------------------------------------------- #
# Reportable metrics.  Every snapshot path here is checked at import.          #
# --------------------------------------------------------------------------- #
_REPORTED: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "performance.isp_delivered", "time", "performance",
        "specific_impulse_delivered_s", differentiable=True,
        target_capable=True,
        exact_counterpart="performance.specific_impulse_delivered_s",
        source_id="NASA-SP-125-sec-2.1-item-2", category="performance",
    ),
    MetricDefinition(
        "performance.isp_ideal", "time", "performance",
        "specific_impulse_ideal_s", differentiable=True, target_capable=True,
        category="performance",
    ),
    MetricDefinition(
        "performance.thrust", "force", "performance", "thrust_n",
        differentiable=True, target_capable=True, category="performance",
    ),
    MetricDefinition(
        "performance.chamber_pressure", "pressure", "performance",
        "chamber_pressure_pa", differentiable=True, target_capable=True,
        category="performance",
    ),
    MetricDefinition(
        "performance.mixture_ratio", "dimensionless", "performance",
        "mixture_ratio", differentiable=True, target_capable=True,
        category="performance",
    ),
    MetricDefinition(
        "performance.c_star_delivered", "velocity", "performance",
        "c_star_delivered_m_s", differentiable=True, target_capable=True,
        category="performance",
    ),
    MetricDefinition(
        "geometry.expansion_ratio", "dimensionless", "geometry",
        "expansion_ratio", differentiable=True, target_capable=True,
        category="geometry",
    ),
    MetricDefinition(
        "geometry.throat_radius", "length", "geometry", "throat_radius_m",
        differentiable=True, target_capable=True, category="geometry",
    ),
    MetricDefinition(
        "geometry.exit_radius", "length", "geometry", "exit_radius_m",
        differentiable=True, target_capable=True, category="geometry",
    ),
    MetricDefinition(
        "geometry.chamber_volume", "volume", "geometry",
        "chamber_volume_m3", differentiable=True, target_capable=True,
        category="geometry",
    ),
    MetricDefinition(
        "thermal.wall_temperature_max", "temperature", "thermal",
        "gas_side_wall_temperature_max_k", differentiable=True,
        target_capable=True, category="thermal",
    ),
    MetricDefinition(
        "thermal.heat_flux_max", "heat_flux", "thermal",
        "heat_flux_max_w_m2", differentiable=True, target_capable=True,
        category="thermal",
    ),
    MetricDefinition(
        "cooling.coolant_pressure_drop", "pressure", "cooling",
        "coolant_pressure_drop_pa", differentiable=True, target_capable=True,
        category="cooling",
    ),
    MetricDefinition(
        "feed.electric_power_total", "power", "feed_electrical",
        "electric_power_total_w", differentiable=True, target_capable=True,
        category="feed",
    ),
    # --- mass objectives: names must say exactly what they include --------- #
    MetricDefinition(
        "masses.electric_package", "mass", "masses",
        "electric_package_objective_mass_kg", differentiable=True,
        objective_capable=True, target_capable=True,
        required_models=("electric_feed",), category="mass",
    ),
    MetricDefinition(
        "masses.dry_mass_partial", "mass", "masses",
        "dry_mass_partial_objective_mass_kg", differentiable=True,
        objective_capable=True, target_capable=True,
        required_models=("electric_feed", "regen_wall"), category="mass",
    ),
    MetricDefinition(
        "masses.battery", "mass", "masses", "battery_objective_mass_kg",
        differentiable=True, target_capable=True,
        category="mass",
    ),
    MetricDefinition(
        "masses.thrust_chamber", "mass", "masses", "thrust_chamber_mass_kg",
        differentiable=True, target_capable=True, category="mass",
    ),
)


# --------------------------------------------------------------------------- #
# Known but not yet modellable.  Registered so a catalog can say why.          #
# --------------------------------------------------------------------------- #
_UNAVAILABLE: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "masses.engine_dry_mass", "mass", "masses",
        "total_engine_package_mass_kg", category="mass",
        unavailable_reason=(
            "valves, lines, brackets and the gimbal have no mass model yet; "
            "docs/PLAN_VALVES_LINES_BRACKETS_GIMBAL.md specifies them and "
            "needs no new physics, only implementation"
        ),
    ),
    MetricDefinition(
        "geometry.installed_envelope", "length", None, None,
        category="geometry",
        unavailable_reason=(
            "a complete installed envelope needs every component placed in the "
            "engine assembly, which depends on the missing hardware branches"
        ),
    ),
    MetricDefinition(
        "performance.engine_system_isp", "time", None, None,
        category="performance",
        unavailable_reason=(
            "engine-system Isp must account for overboard and purge flows; the "
            "repository has no cycle-level mass balance for them"
        ),
    ),
    MetricDefinition(
        "life.qualified_reusable_cycles", "dimensionless", None, None,
        category="life",
        unavailable_reason=(
            "closest of the unavailable set: Coffin-Manson/Basquin is "
            "implemented (raosim.physics) and two liners carry sourced "
            "strain-life data, but a duty cycle definition and a gated "
            "screening contract are still missing"
        ),
    ),
    MetricDefinition(
        "life.cycles_to_failure", "dimensionless", None, None,
        category="life",
        source_id="NASA-CR-134627;Porowski-1985",
        unavailable_reason=(
            "the MDO structural_stress constraint is a static screening "
            "stress, not a cycle count; cycles to failure require a defined "
            "duty cycle, a stabilized local elastic-plastic strain range, "
            "and a sourced strain-life law wired into the differentiable "
            "model"
        ),
    ),
    MetricDefinition(
        "stability.combustion", "dimensionless", None, None,
        category="stability",
        unavailable_reason=(
            "chamber acoustic modes and injector coupling could give a "
            "screening margin (NASA SP-194), but SP-194 treats stability as "
            "empirical, so an authoritative verdict is not available at any "
            "effort"
        ),
    ),
    MetricDefinition(
        "operability.restart_capability", "dimensionless", None, None,
        category="operability",
        unavailable_reason=(
            "requires transient chill-down, purge and ignition sequencing; "
            "that is a time-marching solver, not a constraint on a steady solve"
        ),
    ),
    MetricDefinition(
        "aerodynamics.side_load", "force", None, None, category="aerodynamics",
        unavailable_reason=(
            "side loads are driven by the change in separation pattern "
            "(FSS<->RSS transition). The implemented separation correlations "
            "predict only an onset pressure ratio, which is a free-shock "
            "description and cannot detect the transition. Frey & Hagemann "
            "(JPP 16(3), 2000) report restricted shock separation in "
            "full-scale thrust-optimized nozzles, the contour family this "
            "repository generates. A shock/boundary-layer solver is required "
            "first; a correlation would not be honest here"
        ),
    ),
    MetricDefinition(
        "cost.recurring", "dimensionless", None, None, category="programmatic",
        unavailable_reason=(
            "parametric cost needs a cost database the repository does not "
            "have; this is data, not physics, and should not be presented as "
            "an engineering result"
        ),
    ),
    MetricDefinition(
        "reliability.probability", "dimensionless", None, None,
        category="programmatic",
        unavailable_reason=(
            "requires failure-rate data and an FMEA; not derivable from the "
            "design models and recommended to stay out of scope"
        ),
    ),
)


def _constraint_metrics() -> tuple[MetricDefinition, ...]:
    """Derive one metric per constraint margin from the single manifest."""

    from raosim.mdo.constraints import CONSTRAINT_MANIFEST

    out: list[MetricDefinition] = []
    for spec in CONSTRAINT_MANIFEST:
        out.append(MetricDefinition(
            id=f"constraints.{spec.name}_margin",
            dimension="dimensionless",
            # Margins are reported together in one contract field rather than
            # as individual snapshot fields.
            snapshot_section="constraints_gates",
            snapshot_field="constraint_margins",
            differentiable=spec.kind != "availability_gate",
            objective_capable=False,
            # A constraint is already enforced by the manifest; re-targeting it
            # would let a caller silently weaken a mandatory physical limit.
            target_capable=False,
            engine_key=spec.engine_key,
            source_id=spec.source_id,
            category=f"constraint:{spec.category}",
        ))
    return tuple(out)


def _build_registry() -> Mapping[str, MetricDefinition]:
    from raosim.mdo.snapshot import SNAPSHOT_FIELD_MANIFEST

    registry: dict[str, MetricDefinition] = {}
    for metric in (*_REPORTED, *_UNAVAILABLE, *_constraint_metrics()):
        if metric.id in registry:
            raise ValueError(f"duplicate metric id {metric.id!r}")
        section, fieldname = metric.snapshot_section, metric.snapshot_field
        if section is not None:
            if section not in SNAPSHOT_FIELD_MANIFEST:
                raise ValueError(
                    f"metric {metric.id!r} points at unknown snapshot section "
                    f"{section!r}"
                )
            if fieldname not in SNAPSHOT_FIELD_MANIFEST[section]:
                raise ValueError(
                    f"metric {metric.id!r} points at {section}.{fieldname}, "
                    "which is not in the snapshot contract"
                )
        if metric.objective_capable and not metric.differentiable:
            raise ValueError(
                f"metric {metric.id!r} is objective-capable but not "
                "differentiable"
            )
        if metric.objective_capable and not metric.available:
            raise ValueError(
                f"metric {metric.id!r} is unavailable and cannot be an objective"
            )
        registry[metric.id] = metric
    return registry


METRIC_REGISTRY: Mapping[str, MetricDefinition] = _build_registry()


def resolve_metric(
    metric_id: str,
    *,
    role: Literal["objective", "target", "report"] = "report",
) -> MetricDefinition:
    """Look up a metric and confirm it may be used in ``role``."""

    try:
        metric = METRIC_REGISTRY[metric_id]
    except KeyError as exc:
        raise LookupError(
            f"unknown metric {metric_id!r}; see the metric catalog for the "
            f"{len(METRIC_REGISTRY)} registered ids"
        ) from exc
    if not metric.available:
        raise MetricNotAvailable(
            f"{metric_id} is registered but not available: "
            f"{metric.unavailable_reason}"
        )
    if role == "objective" and not metric.objective_capable:
        raise MetricNotAvailable(
            f"{metric_id} cannot be a primary objective"
        )
    if role == "target" and not metric.target_capable:
        raise MetricNotAvailable(
            f"{metric_id} cannot be a target"
            + (
                "; physical constraints are enforced by the manifest and must "
                "not be re-targeted, which would silently weaken a mandatory "
                "limit"
                if metric.category.startswith("constraint:") else ""
            )
        )
    return metric


def objective_capable_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(m for m in METRIC_REGISTRY.values() if m.objective_capable)


def target_capable_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(m for m in METRIC_REGISTRY.values() if m.target_capable)


def unavailable_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(m for m in METRIC_REGISTRY.values() if not m.available)
