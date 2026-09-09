"""The guided workflow: describe the engine, not the solver.

The wizard asks only what SP-125 sec. 2.1 calls the defining engine
requirements -- thrust and the condition it is quoted at, duration, propellant,
mixture-ratio intent, mass and envelope limits -- plus the architecture and
output choices a user genuinely owns.

Mission-target mode leaves design variables to the optimizer. Partial and
complete-design modes explicitly collect the values the user elects to pin.

Prompting is injected rather than calling ``input`` directly, so the whole flow
is testable with scripted answers and cannot hang a non-interactive run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

__all__ = ["GuidedAbort", "GuidedSession", "Question", "run_guided"]


class GuidedAbort(RuntimeError):
    """The user declined to continue, or input ended."""


@dataclass
class Question:
    key: str
    prompt: str
    default: str | None = None
    choices: tuple[str, ...] = ()
    dimension: str | None = None
    optional: bool = False


def _catalog_propellants() -> tuple[str, ...]:
    from raosim.mdo.propellants import PROPELLANTS

    return tuple(spec.name for spec in PROPELLANTS.values())


def _catalog_liners() -> tuple[str, ...]:
    """Copper-alloy liners: high conductivity, low service temperature."""
    from raosim.materials import CATEGORY_COPPER_LINER, list_materials

    return tuple(
        m.name for m in list_materials()
        if m.category == CATEGORY_COPPER_LINER
    )


def _catalog_jackets() -> tuple[str, ...]:
    """Structural closeout alloys -- SP-8087 sec. 2.1.3.1 hardenable jackets."""
    from raosim.materials import CATEGORY_COPPER_LINER, list_materials

    return tuple(
        m.name for m in list_materials()
        if m.category != CATEGORY_COPPER_LINER
    )


@dataclass
class GuidedSession:
    """One guided interview, driven by injectable prompt/output callables."""

    ask: Callable[[str], str]
    out: Callable[[str], None] = print
    answers: dict[str, Any] = field(default_factory=dict)

    # -- prompting ---------------------------------------------------------- #
    def _ask_one(self, question: Question) -> str | None:
        suffix = ""
        if question.choices:
            suffix += "\n    " + "\n    ".join(
                f"{i + 1}. {choice}" for i, choice in enumerate(question.choices)
            )
        hint = f" [{question.default}]" if question.default is not None else ""
        if question.optional and question.default is None:
            hint = " [optional, blank to skip]"
        while True:
            raw = self.ask(f"{question.prompt}{suffix}{hint}: ").strip()
            if not raw:
                if question.default is not None:
                    raw = question.default
                elif question.optional:
                    return None
                else:
                    self.out("    a value is required")
                    continue
            if question.choices:
                if raw.isdigit() and 1 <= int(raw) <= len(question.choices):
                    return question.choices[int(raw) - 1]
                match = [c for c in question.choices if c.lower() == raw.lower()]
                if match:
                    return match[0]
                self.out(f"    choose one of: {', '.join(question.choices)}")
                continue
            if question.dimension is not None:
                from raosim.units import UnitError, parse_quantity

                try:
                    parse_quantity(
                        raw, dimension=question.dimension, field=question.key
                    )
                except UnitError as exc:
                    # Re-prompting here is the whole point: a unit mistake
                    # should be corrected while the user is present, not
                    # discovered as a 1000x error in a report.
                    self.out(f"    {exc}")
                    continue
            return raw

    # -- interview ---------------------------------------------------------- #
    def run(self) -> dict[str, Any]:
        self.out("What are you providing?")
        mode = self._ask_one(Question(
            "mode", "  select", default="1",
            choices=(
                "mission targets - let the optimizer select the design",
                "partial design - fix what you know, optimize the rest",
                "complete design point - analyze and report",
            ),
        ))
        workflow = "analyze" if mode.startswith("complete") else "optimize"

        thrust = self._ask_one(Question(
            "thrust", "Target thrust (with units, e.g. 13 kN)",
            dimension="force",
        ))
        condition = self._ask_one(Question(
            "condition", "Condition the thrust is quoted at",
            default="sea_level", choices=("sea_level", "vacuum", "altitude"),
        ))
        altitude = None
        if condition == "altitude":
            altitude = self._ask_one(Question(
                "altitude", "Altitude", dimension="length",
            ))
        duration = self._ask_one(Question(
            "burn_duration", "Burn duration", default="120 s", dimension="time",
        ))
        propellant = self._ask_one(Question(
            "propellant", "Propellant", default="LOX/RP-1",
            choices=_catalog_propellants(),
        ))
        liner = self._ask_one(Question(
            "liner", "Liner alloy", default="GRCop-84", choices=_catalog_liners(),
        ))
        jacket = self._ask_one(Question(
            "jacket", "Structural closeout alloy", default="Inconel 718",
            choices=_catalog_jackets(),
        ))
        isp_min = self._ask_one(Question(
            "isp_min", "Minimum delivered thrust-chamber Isp",
            dimension="time", optional=True,
        ))

        objective = None
        if workflow == "optimize":
            from raosim.mdo.metrics import objective_capable_metrics

            objective = self._ask_one(Question(
                "objective", "Primary objective",
                default="masses.dry_mass_partial",
                choices=tuple(metric.id for metric in objective_capable_metrics()),
            ))

        document: dict[str, Any] = {
            "mission": {"thrust": thrust, "thrust_condition": condition,
                        "burn_duration": duration},
            "workflow": workflow,
            "architecture": {
                "propellant": propellant,
                "liner_material": liner,
                "closeout_material": jacket,
            },
        }
        if altitude is not None:
            document["mission"]["altitude"] = altitude
        if isp_min is not None:
            document["targets"] = [{
                "metric": "performance.isp_delivered",
                "relation": "at_least",
                "value": isp_min,
            }]
        if objective is not None:
            direction = (
                "max" if objective == "performance.isp_delivered" else "min"
            )
            document["objective"] = {"metric": objective,
                                     "direction": direction}
        if mode.startswith(("partial", "complete")):
            from raosim.mdo.job import VARIABLE_DIMENSIONS, VariableRequest

            self.out("Enter the design values you are supplying. Dimensional values need units.")
            variables = {}
            for name, dimension in VARIABLE_DIMENSIONS.items():
                if name == "OF":
                    continue  # nominal catalog O/F is declared in the preview
                while True:
                    value = self._ask_one(Question(
                        name, f"Fixed {name}", dimension=dimension,
                        optional=mode.startswith("partial"),
                    ))
                    if value is None:
                        break
                    try:
                        VariableRequest(name, "fixed", value)
                    except ValueError as exc:
                        self.out(f"    {exc}")
                        continue
                    variables[name] = {"mode": "fixed", "value": value}
                    break
            document["variables"] = variables
        self.answers = document
        return document

    # -- preview ------------------------------------------------------------ #
    def preview(self, document: Mapping[str, Any]) -> tuple[str, ...]:
        """Report model coverage before anything is solved.

        A user should learn that their coolant has no validated real-fluid
        surface *before* waiting for a solve that can only return ``unknown``.
        """
        from raosim.mdo.coolant_htd import htd_availability
        from raosim.mdo.propellants import get_propellant

        lines: list[str] = []
        architecture = document.get("architecture", {})
        propellant = architecture.get("propellant", "LOX/RP-1")
        try:
            spec = get_propellant(propellant)
        except KeyError:
            lines.append(f"propellant {propellant!r} is not in the catalog")
            return tuple(lines)

        if getattr(spec, "estimated", False):
            lines.append(
                f"{spec.name}: some constants are engineering ESTIMATES "
                "(see the propellant catalog notes), so results are screening "
                "grade"
            )
        available, reason = htd_availability(
            spec.coolant_name, has_real_fluid_properties=False
        )
        if not available:
            lines.append(f"{spec.name}: {reason}")
            lines.append(
                "  physics feasibility will be reported as UNKNOWN, not as a "
                "pass; an explicit screening opt-in is required to proceed"
            )
        if document.get("workflow") == "optimize" and not document.get(
            "models", {}
        ).get("chamber_property_table"):
            lines.append(
                "no chamber property table supplied, so O/F stays at the "
                "catalog nominal and is not an optimization lever"
            )
        lines.append(
            "engine dry mass, installed envelope, cost and reliability are "
            "not modelled; run 'lrekit catalog unavailable' for why"
        )
        return tuple(lines)


def run_guided(
    ask: Callable[[str], str],
    out: Callable[[str], None] = print,
    *,
    confirm: bool = True,
) -> dict[str, Any]:
    """Run the interview and return a job document, or raise ``GuidedAbort``."""

    session = GuidedSession(ask=ask, out=out)
    document = session.run()

    # The preview and saved job must obey exactly the same contract as run.
    from lrekit.job_cli import _build_spec
    from raosim.mdo.job import resolve_engine_job

    resolved = resolve_engine_job(_build_spec(document))

    out("")
    out("Coverage preview")
    for line in session.preview(document):
        out(f"  - {line}")
    for note in resolved.notes:
        out(f"  - {note}")
    out("")

    if confirm:
        reply = ask("Proceed with this job? [y/N]: ").strip().lower()
        if reply not in {"y", "yes"}:
            raise GuidedAbort("cancelled before solving")
    return document
