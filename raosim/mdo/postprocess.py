"""
raosim.mdo.postprocess — Phase 11: MDO optimum → authoritative re-evaluation.

Plan §10 lists ``postprocess.py -> LREKit result / CAD conversion``; this is it.
It closes the loop between the differentiable MDO layer and the existing
LREKit design/report/CAD pipeline:

    MDO optimum (Pc, eps, channels, film, t_wall, …)
            │  to_design_input()
            ▼
    design_nozzle_v2  →  authoritative Rao contour, thermochemistry,
                         cooling, injector, CAD, reports
            │  compare_margins()
            ▼
    discrepancy report:  do the MDO's constraint margins survive on the
                         authoritative models?

**Why the re-evaluation matters** (plan §11): a gradient optimiser *exploits
model error* — it drives designs into exactly the corners where the cheap
screening models are least accurate.  The MDO's margins are computed on its own
station grid, C¹ property surfaces and screening correlations; the LREKit
pipeline uses the audited NumPy models (and, in validated mode, real CEA).  A
partial degradation is a **finding, not a failure** — but it must be measured
rather than assumed, which is what ``compare_margins`` is for.

Nothing here is differentiated: this runs strictly *after* the optimiser has
converged, which is also why CAD generation is allowed to live here at all
(plan rule 10 — CAD/Boolean geometry is post-optimum only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from raosim.mdo.schema import MissionSpec


# --------------------------------------------------------------------------- #
# MDO result  ->  LREKit DesignInput                                           #
# --------------------------------------------------------------------------- #
def to_design_input(design: dict, mission: MissionSpec, *,
                    mdot_cool: float | None = None,
                    propellant: str | None = None,
                    mode: str | None = None,
                    thermo_mode: str | None = None) -> Any:
    """Build a LREKit ``DesignInput`` from an MDO design dict.

    ``design`` is ``NLPResult.design`` (or any mapping with the same keys).
    The MDO's own architecture constants (channel count, contraction ratio,
    L*, injector geometry) come from ``mission`` so the two descriptions of the
    engine agree.  Imports are deferred: this module must stay importable in the
    jitted layer's environment without pulling in the CAD stack.
    """
    from raosim.design import (
        DesignInput, ThermoSpec, CoolingSpec, ManufacturingSpec,
        DESIGN_MODE_PRELIMINARY,
    )
    from raosim.injector import InjectorSpec, PintleGeometrySpec

    # jacket flow: the fuel side less whatever the film takes (the MDO's own
    # split).  Estimated from the thrust closure when the caller has no solved
    # mass flow to hand.
    if mdot_cool is None:
        At = mission.thrust / (1.4 * float(design["Pc"]))
        mdot = float(design["Pc"]) * At / (mission.eta_cstar
                                           * mission.c_star_ideal())
        mdot_f = mdot / (1.0 + mission.OF)
        mdot_cool = (mission.cooling_fraction * mdot_f
                     * (1.0 - float(design.get("film_frac", 0.0))))

    thermo = ThermoSpec(
        mode=thermo_mode or "constant_gamma",
        propellant_name=propellant or "LOX/RP-1",
        mixture_ratio=mission.OF,
    )
    cooling = CoolingSpec(
        method="regenerative",
        coolant="RP-1",
        channel_count=int(mission.n_channels),
        channel_width=float(design["channel_width"]),
        channel_height=float(design["channel_height"]),
        coolant_inlet_temperature=float(mission.coolant_temperature),
        coolant_wall_temperature_limit=float(mission.rp1_coking_wall_temp_K),
        injector_pressure_drop=float(design["dp_f_frac"] * design["Pc"]),
        coolant_mass_flow=float(mdot_cool),
    )
    manufacturing = ManufacturingSpec(wall_thickness=float(design["t_wall"]))
    injector = InjectorSpec(
        geometry=PintleGeometrySpec(
            pintle_diameter=float(design["D_pintle"]),
            slot_count=int(mission.pintle_slot_count),
        ),
    )
    return DesignInput(
        thermo=thermo,
        Pc=float(design["Pc"]),
        target_thrust=float(mission.thrust),
        epsilon=float(design["eps"]),
        mode=mode or DESIGN_MODE_PRELIMINARY,
        length_pct=float(mission.length_pct),
        contraction_ratio=float(mission.contraction_ratio),
        L_star=float(mission.l_star),
        cooling=cooling,
        manufacturing=manufacturing,
        injector=injector,
    )


# --------------------------------------------------------------------------- #
# Authoritative re-evaluation                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReEvaluation:
    """Side-by-side of the MDO screening answer and the authoritative pipeline."""

    mdo: dict                 # key quantities as the MDO reported them
    authoritative: dict       # the same quantities from design_nozzle_v2
    deltas: dict              # authoritative − mdo
    result: Any               # the full ValidatedDesignResult (contour, CAD, …)
    warnings: tuple


def reevaluate(design: dict, mission: MissionSpec, *, mdo_summary: dict,
               **kw) -> ReEvaluation:
    """Re-run an MDO optimum through ``design_nozzle_v2`` and diff the answers.

    ``mdo_summary`` carries what the MDO believed (Isp, Rt, mdot, T_wc_max …);
    whatever the authoritative pipeline also reports is compared.  Keys the
    pipeline does not produce are simply omitted from ``deltas`` rather than
    faked.
    """
    from raosim.design import design_nozzle_v2

    di = to_design_input(design, mission, **kw)
    res = design_nozzle_v2(di)

    auth: dict = {}
    # geometry / performance the pipeline always produces
    contour = getattr(res, "contour", None)
    if contour is not None:
        auth["Rt"] = float(contour.get("Rt", np.nan))
        auth["eps"] = float(contour.get("epsilon", np.nan))
    for name in ("Isp", "isp", "c_star", "thrust"):
        val = getattr(res, name, None)
        if isinstance(val, (int, float)):
            auth[name] = float(val)
    perf = getattr(res, "performance", None)
    if isinstance(perf, dict):
        for k in ("Isp", "Isp_sl", "Isp_vac", "c_star", "Cf", "thrust"):
            if k in perf and isinstance(perf[k], (int, float)):
                auth.setdefault(k, float(perf[k]))

    deltas = {k: auth[k] - mdo_summary[k]
              for k in auth if k in mdo_summary
              and isinstance(mdo_summary[k], (int, float))}
    warns = tuple(getattr(res, "warnings", ()) or ())
    return ReEvaluation(mdo=dict(mdo_summary), authoritative=auth,
                        deltas=deltas, result=res, warnings=warns)


def summarise(reev: ReEvaluation) -> str:
    """Human-readable discrepancy report (Phase-11 style)."""
    lines = ["  MDO screening vs authoritative pipeline:"]
    if not reev.deltas:
        lines.append("    (no directly comparable scalars were produced)")
    for k, d in sorted(reev.deltas.items()):
        base = reev.mdo.get(k, float("nan"))
        pct = (100.0 * d / base) if base else float("nan")
        flag = "  <-- check" if abs(pct) > 5.0 else ""
        lines.append(f"    {k:<12s} mdo={base:12.4g}  auth={reev.authoritative[k]:12.4g}"
                     f"  Δ={d:+.4g} ({pct:+.1f}%){flag}")
    if reev.warnings:
        lines.append("  pipeline warnings:")
        for w in reev.warnings[:8]:
            lines.append(f"    - {w}")
    return "\n".join(lines)
