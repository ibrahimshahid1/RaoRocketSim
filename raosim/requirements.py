"""raosim.requirements — Layer 0: what the user asks for, not what they design.

The requirement set is not invented here.  NASA SP-125 §2.1, *The Major Rocket
Engine Design Parameters* (printed p. 31, ``propulsion_texts/19710019929.pdf``,
PDF p. 40) fixes it:

    "To fit the engine system properly into a vehicle system, engine systems
    design and development specifications will have to cover the following
    parameters above all:
    (1) Thrust level  (2) Performance (specific impulse)  (3) Run duration
    (4) Propellant mixture ratio  (5) Weight of engine system at burnout
    (6) Envelope (size)  (7) Reliability  (8) Cost
    (9) Availability (time table-schedule)"

Note what is *absent* from that list: chamber pressure, expansion ratio, ``L*``,
contraction ratio, channel geometry, pintle diameter, pump speed.  SP-125 treats
those as design outputs, which is exactly the split :mod:`raosim.mdo` already
implements.  This module supplies the missing left-hand side.

This module is **host-side**.  It never runs inside a traced computation; it
produces a :class:`~raosim.mdo.schema.MissionSpec` and a constraint selection,
and the differentiable layer takes it from there (plan rule 10).

The central discipline
----------------------
A requirement that cannot be screened must say so.  Three statuses:

``ENFORCED``
    A hard NLP constraint on the actual requirement quantity.
``PARTIALLY_ENFORCED``
    A hard NLP constraint on a **lower bound** of the requirement quantity.
    Satisfaction is necessary but not sufficient — the real quantity may still
    violate.  Reported with the reason and with what is missing.
``UNSUPPORTED``
    No screen exists.  The requirement is carried into the report and ignored
    by the optimiser.  It is never silently dropped and never reported as met.

This mirrors the output-contract rule the repository already follows: an
unavailable quantity stays unavailable rather than becoming a convenient zero
(``docs/MDO_REMEDIATION_AND_OUTPUT_CONTRACT.md`` §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "Coverage",
    "EngineRequirement",
    "RequirementCoverage",
    "RequirementResult",
    "ResolvedRequirement",
    "resolve_requirement",
    "solve_requirement",
]


# --------------------------------------------------------------------------- #
# Coverage reporting                                                           #
# --------------------------------------------------------------------------- #
class Coverage(str, Enum):
    """How completely a stated requirement is actually screened."""

    ENFORCED = "enforced"
    PARTIALLY_ENFORCED = "partially_enforced"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class RequirementCoverage:
    """One requirement's screening status.

    ``sp125_item`` is the numbered SP-125 §2.1 parameter this maps to, so a
    report can be read against the monograph.  ``None`` for requirements that
    are not on SP-125's list (throttling, restarts, reusable cycles) — those
    come from later practice, and the field records that honestly.
    """

    requirement: str
    coverage: Coverage
    sp125_item: int | None = None
    constraint: str | None = None
    reason: str | None = None
    missing: tuple[str, ...] = ()

    @property
    def satisfied_implies_met(self) -> bool:
        """Whether a satisfied constraint proves the requirement is met."""

        return self.coverage is Coverage.ENFORCED

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "coverage": self.coverage.value,
            "sp125_item": self.sp125_item,
            "constraint": self.constraint,
            "reason": self.reason,
            "missing": list(self.missing),
            "satisfied_implies_met": self.satisfied_implies_met,
        }


# --------------------------------------------------------------------------- #
# The requirement                                                              #
# --------------------------------------------------------------------------- #
#: Ambient pressure conventions.  SP-125 §2.1: "Thrust levels for first-stage
#: booster engines, which start at or near sea-level altitude and stop at a
#: specified higher altitude, are usually quoted for sea-level conditions...
#: The nominal thrust of engines in stages starting and operating at or near-
#: vacuum conditions is quoted for that environment."  So the condition is part
#: of the requirement, not metadata.
_SEA_LEVEL_PA = 101325.0
#: Not exactly zero: the separation and nozzle-collapse screens divide by or
#: compare against ambient, and a hard zero makes the vacuum case a different
#: code path.  1 Pa is ~1e-5 atm — below any pressure that changes a design —
#: and keeps every screen on one branch.
_VACUUM_PA = 1.0

_SUPPORTED_ISP_BASIS = ("thrust_chamber",)
_SUPPORTED_FEED = ("electric_pump",)
_SUPPORTED_OBJECTIVE = ("min_mass",)


@dataclass(frozen=True)
class EngineRequirement:
    """What the user wants, in SP-125 §2.1 terms.

    Every field is a *requirement*.  Nothing here is a design variable; the
    optimiser owns ``Pc``, ``eps``, ``L*``, channel geometry, pintle diameter
    and pump speed.

    Parameters
    ----------
    thrust, thrust_condition
        SP-125 item (1).  ``thrust_condition`` is ``"sea_level"``, ``"vacuum"``,
        or ``("altitude", h_m)``.  Required, because SP-125 gives no default:
        the same engine has different thrust at different back-pressures.
    isp_min, isp_basis
        Item (2).  ``isp_basis`` records whether the floor refers to the
        complete engine system or the thrust chamber only — SP-125: *"It is
        important to state whether a specified value of I_s refers to the
        complete engine system, or to the thrust chamber only."*  Only
        ``"thrust_chamber"`` is currently screenable.
    flight_duration, qualification_duration
        Item (3), which is two numbers.  SP-125 puts most large liquid engines
        in a 50–400 s flight band, but adds that qualification requires
        *"accumulated duration times... many times the comparatively short
        rated flight duration"* and that **those** specifications *"govern most
        engine design considerations"*.  Flight duration sizes the battery;
        cumulative duration governs life.
    of
        Item (4).  ``None`` (the default and the recommended value) lets the
        propellant supply its optimum.  SP-125 derives O/F from a balance of
        energy release against molecular weight, *modified downward for
        cooling*: *"The temperatures resulting from stoichiometric or near-
        stoichiometric mixture ratios... may impose severe demands on the
        chamber-wall cooling system.  A lower temperature, therefore, may be
        desired and obtained by selecting a suitable ratio."*  That is the trade
        the optimiser should resolve, so pinning O/F removes a real lever.
    burnout_mass_max
        Item (5).
    envelope_diameter_max, envelope_length_max
        Item (6).  SP-125 defines the envelope as *"a hypothetical smallest
        cylinder, cube, or sphere into which the engine would fit"*; a cylinder
        is the natural choice for an axisymmetric engine.
    propellant, feed_architecture
        Architecture selections.  Discrete by construction — plan §0.1 keeps
        discrete choices outside the traced core, so these select a
        configuration rather than entering the gradient.
    throttle_range, restarts, reusable_cycles
        Operability requirements not on SP-125's §2.1 list.  Carried and
        reported; none is screenable today.
    objective
        Which requirement becomes the objective.  Everything else becomes a
        constraint.
    """

    # --- SP-125 §2.1 (1)-(2) ------------------------------------------------ #
    thrust: float
    thrust_condition: str | tuple[str, float]
    isp_min: float | None = None
    isp_basis: str = "thrust_chamber"

    # --- (3) run duration: two numbers -------------------------------------- #
    flight_duration: float = 120.0
    qualification_duration: float | None = None

    # --- (4) mixture ratio: an output unless deliberately pinned ------------ #
    of: float | None = None

    # --- (5)-(6) mass and envelope ------------------------------------------ #
    burnout_mass_max: float | None = None
    envelope_diameter_max: float | None = None
    envelope_length_max: float | None = None

    # --- architecture (discrete; outer enumeration) ------------------------- #
    propellant: str = "LOX/RP-1"
    feed_architecture: str = "electric_pump"

    # --- operability (not on the SP-125 §2.1 list) -------------------------- #
    throttle_range: tuple[float, float] | None = None
    restarts: int | None = None
    reusable_cycles: int | None = None

    # --- what to optimise --------------------------------------------------- #
    objective: str = "min_mass"

    #: Optional passthrough overrides applied last to the derived MissionSpec,
    #: for cases the requirement vocabulary does not cover (e.g. pinning a
    #: liner alloy).  Explicit wins, exactly as ``MissionSpec.for_thrust``.
    mission_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.thrust > 0.0:
            raise ValueError("thrust must be positive")
        if self.isp_min is not None and not self.isp_min > 0.0:
            raise ValueError("isp_min must be positive when given")
        if not self.flight_duration > 0.0:
            raise ValueError("flight_duration must be positive")
        if (self.qualification_duration is not None
                and self.qualification_duration < self.flight_duration):
            raise ValueError(
                "qualification_duration is cumulative demonstrated duration "
                "and cannot be shorter than one flight duration (SP-125 §2.1: "
                "'many times the comparatively short rated flight duration')"
            )
        for name in ("burnout_mass_max", "envelope_diameter_max",
                     "envelope_length_max"):
            v = getattr(self, name)
            if v is not None and not v > 0.0:
                raise ValueError(f"{name} must be positive when given")
        if self.throttle_range is not None:
            lo, hi = self.throttle_range
            if not (0.0 < lo <= hi):
                raise ValueError("throttle_range must satisfy 0 < lo <= hi")
        _ambient_pressure(self.thrust_condition)   # validates early


def _ambient_pressure(condition: str | tuple[str, float]) -> float:
    """Ambient pressure for a stated thrust condition [Pa]."""

    if isinstance(condition, str):
        key = condition.strip().lower()
        if key in {"sea_level", "sea-level", "sl"}:
            return _SEA_LEVEL_PA
        if key in {"vacuum", "vac"}:
            return _VACUUM_PA
        raise ValueError(
            f"unknown thrust_condition {condition!r}; use 'sea_level', "
            "'vacuum', or ('altitude', h_metres)"
        )
    kind, value = condition
    if str(kind).strip().lower() != "altitude":
        raise ValueError(
            f"unknown thrust_condition {condition!r}; the tuple form is "
            "('altitude', h_metres)"
        )
    from raosim.atmosphere import pressure

    h = float(value)
    if h < 0.0:
        raise ValueError("altitude must be non-negative")
    return max(float(pressure(h)), _VACUUM_PA)


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedRequirement:
    """A requirement mapped onto what the solver can actually be handed."""

    requirement: EngineRequirement
    mission: Any                       # MissionSpec
    isp_floor: float
    objective: str
    coverage: tuple[RequirementCoverage, ...]

    @property
    def unsupported(self) -> tuple[RequirementCoverage, ...]:
        return tuple(c for c in self.coverage
                     if c.coverage is Coverage.UNSUPPORTED)

    @property
    def partial(self) -> tuple[RequirementCoverage, ...]:
        return tuple(c for c in self.coverage
                     if c.coverage is Coverage.PARTIALLY_ENFORCED)

    @property
    def fully_screened(self) -> bool:
        """True only if every stated requirement is fully enforced."""

        return all(c.coverage is Coverage.ENFORCED for c in self.coverage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "isp_floor_s": self.isp_floor,
            "objective": self.objective,
            "ambient_pressure_Pa": float(self.mission.Pa),
            "fully_screened": self.fully_screened,
            "coverage": [c.as_dict() for c in self.coverage],
        }

    def report(self) -> str:
        """Human-readable coverage table for the CLI."""

        width = max((len(c.requirement) for c in self.coverage), default=10)
        lines = ["requirement coverage (SP-125 §2.1 numbering):"]
        for c in sorted(self.coverage, key=lambda c: (c.sp125_item or 99,
                                                      c.requirement)):
            item = f"({c.sp125_item})" if c.sp125_item else " - "
            mark = {"enforced": "✓", "partially_enforced": "~",
                    "unsupported": "✗"}[c.coverage.value]
            lines.append(
                f"  {mark} {item:>4} {c.requirement:<{width}}  "
                f"{c.coverage.value}"
                + (f" [{c.constraint}]" if c.constraint else "")
            )
            if c.reason:
                lines.append(f"        {c.reason}")
            if c.missing:
                lines.append(f"        missing: {', '.join(c.missing)}")
        return "\n".join(lines)


def resolve_requirement(req: EngineRequirement) -> ResolvedRequirement:
    """Map an :class:`EngineRequirement` onto a mission + constraint selection.

    Raises only for requirements that would make the run *meaningless*
    (an unsupported objective, an unimplemented feed architecture).  Everything
    else that cannot be screened is recorded as ``UNSUPPORTED`` coverage and
    reported, so the run still produces a design and the user can see exactly
    which of their requirements the answer does not speak to.
    """

    from raosim.mdo.propellants import available, get_propellant
    from raosim.mdo.schema import MissionSpec

    cov: list[RequirementCoverage] = []

    # --- architecture: hard-fail, because it changes what is being designed - #
    if req.feed_architecture not in _SUPPORTED_FEED:
        raise NotImplementedError(
            f"feed_architecture={req.feed_architecture!r} is not implemented; "
            f"supported: {_SUPPORTED_FEED}. Turbine-driven cycles need a "
            "turbine model (NASA SP-8110) and a gas-generator model "
            "(NASA SP-8081); pressure-fed needs SP-125 ch. V plus a non-regen "
            "chamber (NASA SP-8124). See "
            "docs/LITERATURE_REQUESTS_GENERALIZATION.md"
        )
    if req.objective not in _SUPPORTED_OBJECTIVE:
        raise NotImplementedError(
            f"objective={req.objective!r} is not implemented; supported: "
            f"{_SUPPORTED_OBJECTIVE}. The ε-constraint driver minimises mass "
            "at an Isp floor; a max-Isp driver would swap which of the two is "
            "the epigraph."
        )
    try:
        prop = get_propellant(req.propellant)
    except KeyError as exc:
        raise ValueError(
            f"{exc}. Available: {', '.join(available())}"
        ) from exc

    # --- (1) thrust: an internal equality, always closed -------------------- #
    Pa = _ambient_pressure(req.thrust_condition)
    cov.append(RequirementCoverage(
        "thrust", Coverage.ENFORCED, sp125_item=1,
        constraint="outer Newton thrust closure",
        reason=("closed inside every engine evaluation, so it costs the NLP "
                "no constraint row"),
    ))

    # --- (2) specific impulse ----------------------------------------------- #
    if req.isp_basis not in _SUPPORTED_ISP_BASIS:
        cov.append(RequirementCoverage(
            "isp_min", Coverage.UNSUPPORTED, sp125_item=2,
            reason=(f"isp_basis={req.isp_basis!r}: the solver reports "
                    "thrust-chamber Isp only"),
            missing=("turbine/gas-generator overboard flow (open cycles)",
                     "complete engine mass flow accounting"),
        ))
        isp_floor = 0.0
    elif req.isp_min is None:
        isp_floor = 0.0
        cov.append(RequirementCoverage(
            "isp_min", Coverage.ENFORCED, sp125_item=2,
            constraint="isp_epsilon",
            reason="no floor stated; the ε-constraint is inactive",
        ))
    else:
        isp_floor = float(req.isp_min)
        cov.append(RequirementCoverage(
            "isp_min", Coverage.ENFORCED, sp125_item=2,
            constraint="isp_epsilon",
            reason="thrust-chamber basis, matching isp_basis",
        ))

    # --- (3) run duration: two numbers, one of which is screenable ---------- #
    cov.append(RequirementCoverage(
        "flight_duration", Coverage.ENFORCED, sp125_item=3,
        constraint="battery energy sizing (MissionSpec.burn_time)",
        reason="sizes the installed electric-feed energy, not a margin",
    ))
    if req.qualification_duration is not None:
        cov.append(RequirementCoverage(
            "qualification_duration", Coverage.UNSUPPORTED, sp125_item=3,
            reason=("SP-125 §2.1 says cumulative demonstrated duration "
                    "'govern[s] most engine design considerations', but the "
                    "MDO carries no cumulative-life model: the coking screen "
                    "is steady-state and structural_stress is a static "
                    "combined-stress screen, not a cycle count"),
            missing=("Coffin-Manson / SP-8087 low-cycle-fatigue constraint in "
                     "the traced core (roadmap R5, gap 12.11 neighbours)",),
        ))

    # --- (4) mixture ratio --------------------------------------------------- #
    OF = float(req.of) if req.of is not None else float(prop.OF_default)

    # --- assemble the mission ------------------------------------------------ #
    overrides: dict[str, Any] = {
        "Pa": float(Pa),
        "burn_time": float(req.flight_duration),
    }
    if req.of is not None:
        overrides["OF"] = OF
    if req.envelope_diameter_max is not None:
        overrides["envelope_diameter_max"] = float(req.envelope_diameter_max)
    if req.envelope_length_max is not None:
        overrides["envelope_length_max"] = float(req.envelope_length_max)
    if req.burnout_mass_max is not None:
        overrides["dry_mass_max"] = float(req.burnout_mass_max)
    overrides.update(dict(req.mission_overrides))

    mission = MissionSpec.for_propellant(
        req.propellant, float(req.thrust), **overrides
    )

    # O/F coverage depends on whether the surfaces can actually see it, so it
    # is classified after the mission exists.
    if req.of is not None:
        cov.append(RequirementCoverage(
            "of", Coverage.ENFORCED, sp125_item=4,
            constraint="pinned MissionSpec.OF",
            reason=("pinned by the user; note SP-125 §2.1 derives O/F as an "
                    "output balanced against cooling, so pinning it removes a "
                    "design lever"),
        ))
    elif mission.cea_table_path:
        cov.append(RequirementCoverage(
            "of", Coverage.ENFORCED, sp125_item=4,
            constraint="propellant default (CEA surfaces loaded)",
            reason=("γ/T_c/R vary with O/F, so the cooling-vs-performance "
                    "trade SP-125 describes is resolvable"),
        ))
    else:
        cov.append(RequirementCoverage(
            "of", Coverage.PARTIALLY_ENFORCED, sp125_item=4,
            constraint="propellant default (constant properties)",
            reason=("no CEA table on MissionSpec.cea_table_path, so γ/T_c/R "
                    "are FLAT IN O/F: the value is correct at the stated O/F "
                    "but the mixture-ratio trade cannot be resolved"),
            missing=("scripts/sample_cea_surface.py output",
                     "O/F as a design variable"),
        ))

    # --- (5) burnout mass ---------------------------------------------------- #
    if req.burnout_mass_max is not None:
        cov.append(RequirementCoverage(
            "burnout_mass_max", Coverage.PARTIALLY_ENFORCED, sp125_item=5,
            constraint="dry_mass_partial",
            reason=("the screened quantity is a LOWER BOUND on engine dry "
                    "mass, so a satisfied margin does not prove the "
                    "requirement is met"),
            missing=("injector hardware", "manifolds", "valves", "lines",
                     "gimbal", "mounts"),
        ))

    # --- (6) envelope --------------------------------------------------------- #
    for name, value, constraint in (
        ("envelope_diameter_max", req.envelope_diameter_max,
         "envelope_diameter"),
        ("envelope_length_max", req.envelope_length_max, "envelope_length"),
    ):
        if value is None:
            continue
        missing = (("bolted-interface flange OD (host-side; the largest "
                    "diameter on the 13 kN baseline)", "injector body")
                   if constraint == "envelope_diameter"
                   else ("injector body depth", "dome", "feed package"))
        cov.append(RequirementCoverage(
            name, Coverage.PARTIALLY_ENFORCED, sp125_item=6,
            constraint=constraint,
            reason=("screens the cooled thrust chamber only, which is a LOWER "
                    "BOUND on the installed envelope"),
            missing=missing,
        ))

    # --- operability: carried, not screened ---------------------------------- #
    if req.throttle_range is not None:
        cov.append(RequirementCoverage(
            "throttle_range", Coverage.UNSUPPORTED,
            reason=("the MDO solves a single design point; throttling is a "
                    "multipoint problem and the movable-pintle model lives "
                    "host-side"),
            missing=("multipoint mission (roadmap R5d)",
                     "throttled operating point in the traced core"),
        ))
    if req.restarts is not None:
        cov.append(RequirementCoverage(
            "restarts", Coverage.UNSUPPORTED,
            reason="no start/shutdown transient model; the solver is steady",
            missing=("ignition model", "start-transient model"),
        ))
    if req.reusable_cycles is not None:
        cov.append(RequirementCoverage(
            "reusable_cycles", Coverage.UNSUPPORTED,
            reason=("structural_stress is a static combined-stress screen, "
                    "not a cycle count"),
            missing=("low-cycle-fatigue constraint in the traced core",),
        ))

    return ResolvedRequirement(
        requirement=req,
        mission=mission,
        isp_floor=isp_floor,
        objective=req.objective,
        coverage=tuple(cov),
    )


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RequirementResult:
    """An NLP solution plus what it does and does not say about the ask."""

    resolved: ResolvedRequirement
    nlp: Any                           # NLPResult

    @property
    def requirements_met(self) -> bool | None:
        """Whether the stated requirements are met.

        ``True``/``False`` only when every stated requirement is fully
        enforced.  ``None`` when any requirement is partially enforced or
        unsupported — because in that case the optimiser's feasibility flag
        answers a *different, weaker* question than the one asked, and
        collapsing the two would be the exact failure this module exists to
        prevent.  Callers must handle ``None`` rather than treating it as
        truthy.
        """

        if not self.nlp.feasible:
            return False
        if not self.resolved.fully_screened:
            return None
        return True

    def summary(self) -> str:
        r, n = self.resolved, self.nlp
        verdict = {True: "requirements MET",
                   False: "requirements NOT met",
                   None: "feasible against the screened subset only"}[
            self.requirements_met]
        return "\n".join([
            r.report(),
            "",
            f"verdict: {verdict}",
            f"  Isp             {n.Isp:8.2f} s   (floor {n.isp_min:.1f})",
            f"  objective mass  {n.objective_mass:8.3f} kg",
            f"  max violation   {n.max_violation:8.2e}",
            f"  solver          {n.message}",
        ])


def solve_requirement(req: EngineRequirement, *,
                      couple_eta_cstar: bool = False,
                      **kw: Any) -> RequirementResult:
    """Resolve a requirement and run the ε-constraint NLP against it.

    Extra keyword arguments are forwarded to
    :func:`raosim.mdo.nlp.solve_min_mass`.
    """

    from raosim.mdo.nlp import solve_min_mass

    resolved = resolve_requirement(req)
    nlp = solve_min_mass(resolved.mission, resolved.isp_floor,
                         couple_eta_cstar=couple_eta_cstar, **kw)
    return RequirementResult(resolved=resolved, nlp=nlp)
