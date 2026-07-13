"""Opt-in fixed-point coupling between injector vaporization and cycle mass flow.

The default design path remains independent of the screening atomization
correlation.  When a user supplies explicit mixing and chemical-completion
efficiencies, this module can close the otherwise one-way loop

``eta_cstar -> mdot -> injector geometry -> eta_vaporization -> eta_cstar``.

It does not infer mixing or combustion efficiency and it refuses to use an
inapplicable gas/transcritical droplet result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Literal

from raosim.spray.handoff import SprayCycleHandoff


@dataclass(frozen=True)
class SprayCStarCouplingSpec:
    enabled: bool = False
    eta_mixing: float | None = None
    eta_combustion: float | None = None
    relaxation: float = 0.5
    relative_tolerance: float = 1.0e-4
    max_iterations: int = 25
    minimum_eta_cstar: float = 0.40
    require_convergence: bool = True
    source: Literal["legacy_screen", "lagrangian"] = "legacy_screen"

    def validate(self) -> None:
        if self.source not in {"legacy_screen", "lagrangian"}:
            raise ValueError("source must be 'legacy_screen' or 'lagrangian'")
        if not self.enabled:
            return
        for name, value in (
            ("eta_mixing", self.eta_mixing),
            ("eta_combustion", self.eta_combustion),
        ):
            if value is None or not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be supplied in (0, 1]")
        if not math.isfinite(self.relaxation) or not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")
        if self.relative_tolerance <= 0.0 or not math.isfinite(self.relative_tolerance):
            raise ValueError("relative_tolerance must be finite and > 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not 0.0 < self.minimum_eta_cstar <= 1.0:
            raise ValueError("minimum_eta_cstar must be in (0, 1]")


@dataclass(frozen=True)
class SprayCouplingIteration:
    iteration: int
    eta_cstar_in: float
    eta_vaporization: float
    eta_cstar_raw: float
    eta_cstar_out: float
    required_mass_flow: float
    relative_change: float
    observation_fingerprint: str | None = None
    state_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


@dataclass
class SprayCStarCouplingResult:
    converged: bool
    eta_cstar: float
    eta_vaporization: float
    eta_mixing: float
    eta_combustion: float
    effective_cstar: float
    required_mass_flow: float
    relative_closure_error: float
    iterations: list[SprayCouplingIteration] = field(default_factory=list)
    payload: Any = None
    spray_observation: Any = None
    source: str = "legacy_screen"
    model: str = "relaxed_legacy_screen_spray_cycle_fixed_point"

    def to_dict(self) -> dict[str, Any]:
        final_state = _payload_state_summary(self.payload)
        return {
            "enabled": True,
            "converged": self.converged,
            "eta_cstar": self.eta_cstar,
            "eta_vaporization": self.eta_vaporization,
            "eta_mixing": self.eta_mixing,
            "eta_combustion": self.eta_combustion,
            "effective_cstar_m_s": self.effective_cstar,
            "required_mass_flow_kg_s": self.required_mass_flow,
            "relative_closure_error": self.relative_closure_error,
            "iteration_count": len(self.iterations),
            "iterations": [item.to_dict() for item in self.iterations],
            "model": self.model,
            "source": self.source,
            "spray_observation": (
                self.spray_observation.to_dict()
                if isinstance(self.spray_observation, SprayCycleHandoff)
                else None
            ),
            "scope": (
                final_state.get("outer_loop_scope")
                if final_state is not None else
                "injector_and_cycle_mass_flow_no_regen"
            ),
            "final_state_summary": final_state,
        }


def _payload_state_summary(payload: Any) -> dict[str, Any] | None:
    callback = getattr(payload, "coupling_summary", None)
    if not callable(callback):
        return None
    summary = callback()
    if not isinstance(summary, dict):
        raise TypeError("payload coupling_summary() must return a dict")
    return summary


def _legacy_eta(atomization: Any, *, final: bool) -> float:
    """Validate the existing correlation-screen observation."""

    streams = getattr(atomization, "streams", None)
    if isinstance(streams, dict):
        inapplicable = [
            str(role)
            for role, stream in streams.items()
            if not bool(getattr(stream, "applicable", False))
        ]
        if inapplicable:
            prefix = "final spray/c-star state has" if final else (
                "spray/c-star coupling requires an applicable spray model "
                "for every propellant stream;"
            )
            if final:
                raise RuntimeError(
                    prefix + " inapplicable propellant streams: "
                    + ", ".join(sorted(inapplicable))
                )
            raise RuntimeError(
                prefix + " inapplicable streams: "
                + ", ".join(sorted(inapplicable))
                + ". Use a real-fluid/transcritical model rather than "
                "renormalizing eta_vaporization over the remaining streams."
            )
        out_of_regime = [
            str(role)
            for role, stream in streams.items()
            if hasattr(stream, "regime")
            and getattr(stream, "regime") != "aerodynamic atomization"
        ]
        if out_of_regime:
            if final:
                raise RuntimeError(
                    "final spray/c-star state left the aerodynamic-atomization "
                    "regime for: " + ", ".join(sorted(out_of_regime))
                )
            raise RuntimeError(
                "spray/c-star coupling requires every liquid stream to be "
                "inside the implemented aerodynamic-atomization regime; "
                "out-of-regime streams: " + ", ".join(sorted(out_of_regime))
            )
    eta = float(getattr(atomization, "eta_vaporization", math.nan))
    if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
        message = (
            "final spray/c-star state has no finite applicable "
            "eta_vaporization in (0, 1]"
            if final else
            "spray/c-star coupling requires a finite applicable liquid "
            "eta_vaporization in (0, 1]"
        )
        raise RuntimeError(message)
    return eta


def _observation_eta(
    spec: SprayCStarCouplingSpec, observation: Any, *, final: bool
) -> tuple[float, str | None]:
    if spec.source == "legacy_screen":
        return _legacy_eta(observation, final=final), None
    if not isinstance(observation, SprayCycleHandoff):
        raise RuntimeError(
            "lagrangian spray/c-star coupling requires a typed "
            "SprayCycleHandoff; an eta-like object is not sufficient"
        )
    if not observation.coupling_eligible:
        failed = [
            gate.name for gate in observation.required_gates
            if not gate.passed
        ]
        raise RuntimeError(
            "lagrangian SprayCycleHandoff is not coupling eligible; failed "
            "gates: " + ", ".join(failed)
        )
    eta = float(observation.eta_vaporization)
    if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
        raise RuntimeError(
            "lagrangian SprayCycleHandoff has eta_vaporization outside (0, 1]"
        )
    return eta, observation.fingerprint


def solve_spray_cstar_fixed_point(
    spec: SprayCStarCouplingSpec,
    *,
    initial_eta_cstar: float,
    ideal_cstar: float,
    chamber_pressure: float,
    throat_area: float,
    evaluator: Callable[[float, float], tuple[Any, Any]],
) -> SprayCStarCouplingResult:
    """Iterate an explicit injector evaluator to a relaxed efficiency fixed point.

    ``evaluator(eta_cstar, required_mdot)`` must return
    ``(atomization, payload)`` where ``atomization.eta_vaporization`` is a
    finite applicable liquid-mass fraction.  The payload is returned unchanged
    so an integrated design can retain its final injector result.
    """

    spec.validate()
    if not spec.enabled:
        raise ValueError("spray/c-star fixed point requested with coupling disabled")
    for name, value in (
        ("initial_eta_cstar", initial_eta_cstar),
        ("ideal_cstar", ideal_cstar),
        ("chamber_pressure", chamber_pressure),
        ("throat_area", throat_area),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")

    eta = float(min(max(initial_eta_cstar, spec.minimum_eta_cstar), 1.0))
    history: list[SprayCouplingIteration] = []
    payload = None
    converged = False
    for iteration in range(1, spec.max_iterations + 1):
        mdot = chamber_pressure * throat_area / (ideal_cstar * eta)
        observation, payload = evaluator(eta, mdot)
        eta_vap, observation_fingerprint = _observation_eta(
            spec, observation, final=False
        )
        raw = eta_vap * float(spec.eta_mixing) * float(spec.eta_combustion)
        if raw < spec.minimum_eta_cstar:
            raise RuntimeError(
                f"coupled eta_cstar={raw:.6g} is below configured minimum "
                f"{spec.minimum_eta_cstar:.6g}; correlation is outside the "
                "accepted cycle-design envelope"
            )
        eta_next = eta + spec.relaxation * (raw - eta)
        eta_next = float(min(max(eta_next, spec.minimum_eta_cstar), 1.0))
        relative_change = abs(eta_next - eta) / max(abs(eta), 1.0e-12)
        history.append(SprayCouplingIteration(
            iteration=iteration,
            eta_cstar_in=eta,
            eta_vaporization=eta_vap,
            eta_cstar_raw=raw,
            eta_cstar_out=eta_next,
            required_mass_flow=mdot,
            relative_change=relative_change,
            observation_fingerprint=observation_fingerprint,
            state_summary=_payload_state_summary(payload),
        ))
        eta = eta_next
        if relative_change <= spec.relative_tolerance:
            converged = True
            break

    final_mdot = chamber_pressure * throat_area / (ideal_cstar * eta)
    # Re-evaluate at the reported fixed-point state so the returned injector
    # payload and its mass flow correspond exactly to the final efficiency.
    final_observation, payload = evaluator(eta, final_mdot)
    final_eta_vap, _ = _observation_eta(
        spec, final_observation, final=True
    )
    final_raw = final_eta_vap * float(spec.eta_mixing) * float(spec.eta_combustion)
    closure_error = abs(final_raw - eta) / max(abs(eta), 1.0e-12)
    converged = converged and closure_error <= max(
        spec.relative_tolerance, 2.0 * spec.relative_tolerance / spec.relaxation
    )
    if not converged and spec.require_convergence:
        raise RuntimeError(
            "spray/c-star fixed point did not converge in "
            f"{spec.max_iterations} iterations; final closure error="
            f"{closure_error:.3e}"
        )

    return SprayCStarCouplingResult(
        converged=converged,
        eta_cstar=eta,
        eta_vaporization=final_eta_vap,
        eta_mixing=float(spec.eta_mixing),
        eta_combustion=float(spec.eta_combustion),
        effective_cstar=ideal_cstar * eta,
        required_mass_flow=final_mdot,
        relative_closure_error=closure_error,
        iterations=history,
        payload=payload,
        spray_observation=final_observation,
        source=spec.source,
        model=(
            "relaxed_lagrangian_spray_cycle_fixed_point"
            if spec.source == "lagrangian"
            else "relaxed_legacy_screen_spray_cycle_fixed_point"
        ),
    )


__all__ = [
    "SprayCStarCouplingSpec",
    "SprayCouplingIteration",
    "SprayCStarCouplingResult",
    "solve_spray_cstar_fixed_point",
]
