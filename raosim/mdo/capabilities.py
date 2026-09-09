"""Executable job capabilities shared by validation and user-facing catalogs.

An accepted selection names an implemented execution path. Unsupported choices
must acquire an implementation here before a client can advertise them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobCapability:
    kind: str
    id: str
    implemented_by: str
    scope: str


JOB_CAPABILITIES = (
    JobCapability("schema_version", "1.1.0", "raosim.mdo.job.EngineJobSpec",
                  "single-design-point engine jobs"),
    JobCapability("feed", "electric_pump", "raosim.mdo.feed",
                  "electric pump feed sizing; detailed assembly remains partial"),
    JobCapability("injector", "pintle", "raosim.mdo.injector",
                  "pintle hydraulic sizing and screening"),
    JobCapability("cooling", "regenerative", "raosim.mdo.cooling",
                  "regenerative cooling with the selected coolant coverage"),
    JobCapability("parts", "none", "raosim.mdo.execute",
                  "unified jobs do not export hardware packages yet"),
    JobCapability("cad_fidelity", "L0_numerical_only", "raosim.mdo.execute",
                  "numerical analysis reports; no CAD fidelity claim"),
    *(JobCapability("report", name, "raosim.mdo.execute", "JSON report")
      for name in ("summary", "resolved_inputs", "snapshot", "comparison", "coverage")),
)


def require_capability(kind: str, value: str) -> JobCapability:
    """Return an implemented selection, or refuse it without a fallback."""
    available = tuple(row for row in JOB_CAPABILITIES if row.kind == kind)
    for row in available:
        if isinstance(value, str) and row.id == value:
            return row
    choices = ", ".join(repr(row.id) for row in available) or "none"
    raise ValueError(
        f"unsupported {kind} {value!r}; implemented capabilities: {choices}. "
        "The requested selection will not be replaced by a default."
    )
