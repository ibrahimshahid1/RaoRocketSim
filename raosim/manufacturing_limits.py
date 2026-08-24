"""Manufacturing limits, kept distinct from physical validity limits.

A milled channel that cannot be cut is not a physics failure -- it is a
geometry no shop will produce.  Mixing the two is how a "feasible" optimum
becomes unbuildable, so these limits live here with their own provenance and
their own applicability rather than inside the physics constraints.

Source
------
Wadel, M. F., *Comparison of High Aspect Ratio Cooling Channel Designs for a
Rocket Combustion Chamber* (NASA TM-107276 / AIAA-97-????, 1997), the
"fabrication constraints for milled channels" list:

    Aspect ratios <8; coolant channel heights <0.51 cm (0.20 in.); coolant
    channel widths >0.051 cm (0.02 in.); coolant channel landwidths
    >0.051 cm (0.02 in.); no sharp changes in coolant channel width or height

Applicability, stated plainly: Wadel's study is an analytical investigation of
a **liquid-hydrogen-cooled** chamber, and the companion validation hardware
(ref. 2 therein) is an 89 kN (20 000 lbf) class combustor.  Applying these
numbers to a 13 kN RP-1 engine is an extrapolation in both coolant and scale.
They are quoted here as *milling practice*, which travels better than a
coolant-specific thermal result, but the extrapolation is recorded rather than
hidden.

Note what Wadel does **not** say: that AR>8 is unphysical.  The paper's own
designs reached AR 40 analytically and were cut back "from 40 down to the limit
of 8" only when fabrication constraints were imposed, and Carlile (1992) tested
AR 8 and 15 experimentally.  Eight is therefore a manufacturing bound, not a
model-validity bound, and this module labels it as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = [
    "MILLED_CHANNEL_LIMITS",
    "ManufacturingFinding",
    "ManufacturingLimit",
    "check_channel_geometry",
    "check_design_space",
]

WADEL_SOURCE = "Wadel-1997-milled-channel-fabrication-constraints"
WADEL_APPLICABILITY = (
    "established for milled channels on a liquid-hydrogen-cooled, 89 kN-class "
    "combustor; applying it to another coolant or thrust class is an "
    "extrapolation"
)


@dataclass(frozen=True)
class ManufacturingLimit:
    """One process limit on a geometric quantity."""

    quantity: str
    minimum: float | None
    maximum: float | None
    units: str
    process: str
    source_id: str
    applicability: str

    def violation(self, value: float) -> str | None:
        """Return a message when ``value`` is outside the limit."""
        if value is None or not math.isfinite(float(value)):
            return f"{self.quantity} is not a finite value"
        v = float(value)
        if self.minimum is not None and v < self.minimum:
            return (
                f"{self.quantity} {v:g} {self.units} is below the {self.process} "
                f"minimum {self.minimum:g} {self.units}"
            )
        if self.maximum is not None and v > self.maximum:
            return (
                f"{self.quantity} {v:g} {self.units} exceeds the {self.process} "
                f"maximum {self.maximum:g} {self.units}"
            )
        return None


#: Wadel's milled-channel list, in SI.
MILLED_CHANNEL_LIMITS: tuple[ManufacturingLimit, ...] = (
    ManufacturingLimit("channel_width", 0.51e-3, None, "m", "milled_channel",
                       WADEL_SOURCE, WADEL_APPLICABILITY),
    ManufacturingLimit("channel_height", None, 5.1e-3, "m", "milled_channel",
                       WADEL_SOURCE, WADEL_APPLICABILITY),
    ManufacturingLimit("land_width", 0.51e-3, None, "m", "milled_channel",
                       WADEL_SOURCE, WADEL_APPLICABILITY),
    ManufacturingLimit("channel_aspect_ratio", None, 8.0, "-",
                       "milled_channel", WADEL_SOURCE, WADEL_APPLICABILITY),
)

_BY_QUANTITY: Mapping[str, ManufacturingLimit] = {
    limit.quantity: limit for limit in MILLED_CHANNEL_LIMITS
}


@dataclass(frozen=True)
class ManufacturingFinding:
    """One way a geometry or a search box conflicts with milling practice."""

    quantity: str
    message: str
    source_id: str
    applicability: str
    kind: str = "geometry"      # "geometry" | "search_box"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "message": self.message,
            "source_id": self.source_id,
            "applicability": self.applicability,
            "kind": self.kind,
        }


def check_channel_geometry(
    *,
    channel_width: float | None = None,
    channel_height: float | None = None,
    land_width: float | None = None,
    process: str = "milled_channel",
) -> tuple[ManufacturingFinding, ...]:
    """Check one resolved channel section against milling practice.

    Aspect ratio is derived here rather than accepted, so a caller cannot
    supply a height and width that disagree with the ratio they claim.
    """

    if process != "milled_channel":
        raise ValueError(
            f"no manufacturing limits are recorded for process {process!r}"
        )

    findings: list[ManufacturingFinding] = []
    supplied = {
        "channel_width": channel_width,
        "channel_height": channel_height,
        "land_width": land_width,
    }
    for quantity, value in supplied.items():
        if value is None:
            continue
        message = _BY_QUANTITY[quantity].violation(value)
        if message:
            limit = _BY_QUANTITY[quantity]
            findings.append(ManufacturingFinding(
                quantity, message, limit.source_id, limit.applicability,
            ))

    if channel_width and channel_height and float(channel_width) > 0.0:
        ratio = float(channel_height) / float(channel_width)
        message = _BY_QUANTITY["channel_aspect_ratio"].violation(ratio)
        if message:
            limit = _BY_QUANTITY["channel_aspect_ratio"]
            findings.append(ManufacturingFinding(
                "channel_aspect_ratio", message, limit.source_id,
                limit.applicability,
            ))
    return tuple(findings)


def check_design_space(space: Iterable[Any]) -> tuple[ManufacturingFinding, ...]:
    """Report where an optimizer's search box admits unmillable geometry.

    This is the difference between "the optimum happened to be buildable" and
    "the optimizer could not have chosen otherwise".  A box whose corner is
    unmillable will eventually be explored.
    """

    findings: list[ManufacturingFinding] = []
    for variable in space:
        name = getattr(variable, "name", None)
        limit = _BY_QUANTITY.get(str(name))
        if limit is None:
            continue
        lower = getattr(variable, "lower", None)
        upper = getattr(variable, "upper", None)
        if limit.minimum is not None and lower is not None:
            if float(lower) < limit.minimum:
                findings.append(ManufacturingFinding(
                    str(name),
                    f"search box lower bound {float(lower):g} {limit.units} is "
                    f"below the {limit.process} minimum {limit.minimum:g} "
                    f"{limit.units}, so the optimizer may select geometry that "
                    "cannot be milled",
                    limit.source_id, limit.applicability, kind="search_box",
                ))
        if limit.maximum is not None and upper is not None:
            if float(upper) > limit.maximum:
                findings.append(ManufacturingFinding(
                    str(name),
                    f"search box upper bound {float(upper):g} {limit.units} "
                    f"exceeds the {limit.process} maximum {limit.maximum:g} "
                    f"{limit.units}",
                    limit.source_id, limit.applicability, kind="search_box",
                ))
    return tuple(findings)
