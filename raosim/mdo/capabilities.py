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
    # The Son movable pintle valve (MissionSpec.pintle_architecture
    # "son_movable") is optimizable through the API and CLI, but unified jobs
    # always run the authoritative host re-evaluation, which has no movable
    # handoff yet, so it is deliberately not an executable job selection.
    JobCapability("injector", "pintle", "raosim.mdo.injector",
                  "fixed slotted pintle hydraulic sizing, post hardware and "
                  "screening"),
    JobCapability("cooling", "regenerative", "raosim.mdo.cooling",
                  "regenerative cooling with the selected coolant coverage"),
    JobCapability("parts", "none", "raosim.mdo.execute",
                  "no hardware package; numerical reports only"),
    # The authoritative re-evaluation (raosim.mdo.postprocess.reevaluate ->
    # design_nozzle_v2) writes its reference CAD into the run directory and
    # raosim.mdo.build records the engine part manifest.  That module accepts
    # only path-and-hash evidence today, which it records as UNVERIFIED, so no
    # part earns geometry fidelity and the package is always partial; the job
    # must opt into a partial package (allow_partial_package).
    JobCapability("parts", "reference_cad", "raosim.mdo.execute",
                  "authoritative re-evaluation reference CAD plus the "
                  "raosim.mdo.build engine part manifest; unverified "
                  "artifacts, always a partial package"),
    JobCapability("cad_format", "step", "raosim.mdo.postprocess.reevaluate",
                  "STEP from the authoritative re-evaluation; requires "
                  "parts='reference_cad'"),
    # No achieved-fidelity claim beyond L0 is made by a job: the build
    # manifest reports what each part actually reached.
    JobCapability("cad_fidelity", "L0_numerical_only", "raosim.mdo.execute",
                  "no CAD fidelity claim; the part manifest reports achieved "
                  "fidelity per part"),
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
