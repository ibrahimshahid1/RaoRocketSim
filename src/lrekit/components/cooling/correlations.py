"""Source-scoped thermal correlation choices and applicability reports.

The numeric bounds below are Bartz's preliminary-estimation guidance, not a
physical validation envelope. Original PDF identities and pages are recorded in
knowledge/reports/THERMAL_VERIFICATION_IMPLEMENTATION_2026-10-17.md.
"""

from __future__ import annotations

import math

from lrekit.core.coolants import canonical_coolant_name


# Bartz 1957 Eq. [9], PDF p.2; SP-125 Eq. (4-16), PDF p.110.
# Exact international pound/inch definitions; source precision stays 3 digits.
GAS_VISCOSITY_SI_COEFFICIENT = 46.6e-10 * (0.45359237 / 0.0254) * 1.8**0.6
SP125_RP1_C1 = 0.0214  # SP-125 PDF p.119, printed p.110.
BARTZ_CONTRACTION_ANGLE_RANGE_DEG = (15.0, 45.0)
BARTZ_EXPANSION_ANGLE_RANGE_DEG = (7.5, 22.5)
BARTZ_MAX_DIAMETER_CURVATURE_RATIO = 3.0  # source says "about 3"


def resolve_sp125_c1(*, coolant: str | None = None,
                     c1: float | None = None) -> float:
    """Resolve a coolant-specific Eq. (4-23) coefficient, without guessing.

    A supplied value is an explicit user input; it does not acquire literature
    support by passing this function. Only the held RP-1 value is built in.
    """
    if c1 is not None:
        value = float(c1)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("coolant_heat_transfer_c1 must be finite and positive")
        return value
    if coolant is not None and canonical_coolant_name(coolant) == "rp1":
        return SP125_RP1_C1
    raise ValueError(
        "UNAVAILABLE: SP-125 Eq. (4-23) needs a coolant-specific C1; "
        "the held value 0.0214 is for RP-1 only. Supply "
        "coolant_heat_transfer_c1 with an appropriate source."
    )


def bartz_applicability(*, Dt: float, throat_curvature_radius: float,
                        contraction_half_angle_deg: float | None = None,
                        expansion_half_angle_deg: float | None = None,
                        includes_chamber: bool = False) -> dict:
    """Assess author-stated geometric guidance, preserving missing context.

    The source compares a nozzle with constant contraction/expansion half
    angles. A curved bell does not have one expansion angle; mapping its exit
    tangent to that quantity does not establish equivalence.
    """
    ratio = (float(Dt) / float(throat_curvature_radius)
             if throat_curvature_radius > 0.0 else float("nan"))
    supplied = {
        "diameter_over_curvature": ratio,
        "contraction_half_angle_deg": contraction_half_angle_deg,
        "expansion_half_angle_deg": expansion_half_angle_deg,
    }
    checks = [math.isfinite(ratio) and 0.0 < ratio <= BARTZ_MAX_DIAMETER_CURVATURE_RATIO]
    missing = []
    for key, bounds in (
        ("contraction_half_angle_deg", BARTZ_CONTRACTION_ANGLE_RANGE_DEG),
        ("expansion_half_angle_deg", BARTZ_EXPANSION_ANGLE_RANGE_DEG),
    ):
        value = supplied[key]
        if value is None:
            missing.append(key)
        else:
            checks.append(math.isfinite(float(value)) and bounds[0] <= float(value) <= bounds[1])
    state = "OUTSIDE_VALIDITY" if not all(checks) else "UNKNOWN" if missing else "PASS"
    return {
        "geometry_state": state,
        "state": state if state != "PASS" else "REPORT_ONLY",
        "source": "DOC_DOI_10_2514_8_12572",
        "source_pages": [2, 3],
        "basis": "AUTHOR_ESTIMATED_PRELIMINARY_GUIDANCE",
        "inputs": supplied,
        "missing_inputs": missing,
        "bounds": {
            "contraction_half_angle_deg": list(BARTZ_CONTRACTION_ANGLE_RANGE_DEG),
            "expansion_half_angle_deg": list(BARTZ_EXPANSION_ANGLE_RANGE_DEG),
            "diameter_over_curvature_max_approximate": BARTZ_MAX_DIAMETER_CURVATURE_RATIO,
        },
        "includes_chamber": bool(includes_chamber),
        "chamber_heat_flux_warning": (
            "Bartz reports chamber predictions only 40-50% of measured values "
            "for RFNA-ammonia tests; no universal multiplier is established."
            if includes_chamber else None
        ),
        "additional_unquantified_effects": [
            "strong gas radiation", "dissociation and near-wall recombination",
            "high-frequency flow instability", "insulating deposits",
            "incomplete chamber combustion",
        ],
        "authoritative_thermal_design_available": False,
        "physical_validation": "NOT_ESTABLISHED",
    }
