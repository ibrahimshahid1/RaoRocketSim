"""
thermal_design.py — cooling-coupled contour selection.

The screening workflow in :mod:`raosim.design` fixes the contour first
(chart Bézier) and evaluates heat flux / cooling AFTER, as pass/fail
gates.  This module inverts that: it computes the **cooling physics
before the contour is fixed** and lets it shape the geometry, so the
returned contour is one the thermal design can actually survive.

Two physical levers, co-designed against the cooling margin (peak
gas-side wall temperature vs the channel-wall limit):

* **Throat curvature** ``Rd_factor`` — opening the throat radius of
  curvature lowers the Bartz peak flux through the ``(D*/r_c)^0.1``
  term.  This is a *weak* lever (~20 % flux relief from 0.382→3.0 Rt):
  geometry alone cannot cool a throat, which is exactly why engines
  rely on active cooling.  It is applied first because it costs nothing
  hydraulically.
* **Channel mass flux** ``G`` — shrinking the channel cross-section at
  fixed coolant flow raises the coolant velocity and the Sieder-Tate
  ``h_c ∝ G^0.8``.  This is the *strong* lever and carries the rest.

The result reports the selected contour, the converged cooling state,
the binding lever, and the full search history.  This is the NumPy
feedback precursor to the rigorous differentiable constrained design
(the J-series gradient loop): same coupling direction (physics →
geometry), heuristic search instead of exact gradients.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis


# --------------------------------------------------------------------------- #
# Channel auto-sizing — solve channel count/width FROM the requirement         #
# --------------------------------------------------------------------------- #
def coolant_flow_from_cycle(
    Pc: float, Rt: float, c_star: float, mixture_ratio: float,
    *, cooling_fraction: float = 1.0,
) -> tuple[float, float]:
    """Coolant mass flow [kg/s] derived from the engine cycle, not chosen.

    The regenerative coolant is the fuel before injection, so it is set
    by the propellant flow, not a free knob:

        mdot_total = Pc · A_t / c*          (c* = Pc·A_t/mdot)
        mdot_fuel  = mdot_total / (1 + MR)  (MR = oxidiser/fuel ratio)
        mdot_cool  = mdot_fuel · cooling_fraction

    Returns ``(mdot_cool, mdot_total)``.
    """
    At = math.pi * Rt * Rt
    mdot_total = Pc * At / max(c_star, 1e-9)
    mdot_fuel = mdot_total / (1.0 + max(mixture_ratio, 0.0))
    return mdot_fuel * cooling_fraction, mdot_total


def size_cooling_channels(
    contour: dict,
    prop: Any,
    Pc: float,
    *,
    margin_target: float = 1.2,
    dp_budget_bar: float = 150.0,
    wall_temp_limit: float = 1100.0,
    mixture_ratio: float = 2.6,
    cooling_fraction: float = 1.0,
    channel_height: float = 0.0025,
    wall_thickness: float = 0.001,
    wall_k: float = 350.0,
    coolant: str = "rp1",
    w_min: float = 0.0005,
    w_max: float = 0.0020,
    land_min: float = 0.0005,
    n_w: int = 6,
    n_count: int = 12,
    objective: str = "min_dp",
    n_iter: int = 15,
) -> dict:
    """Solve for the channel count ``N`` and width ``w`` that meet the
    cooling requirement, instead of taking them as inputs.

    The coolant flow is derived from the cycle (:func:`coolant_flow_from
    _cycle`).  Over a grid of (``N``, ``w``) — bounded by the
    manufacturing floors ``w_min`` / ``land_min`` and the throat
    circumference (channels must fit) — the coupled Sieder-Tate cooling
    is evaluated and a design is chosen that satisfies

        cooling margin ≥ ``margin_target``   (peak wall T ≤ limit/target)
        pressure drop  ≤ ``dp_budget_bar``
        channels fit + w ≥ w_min

    optimising ``objective`` among the feasible set: ``"min_dp"`` (least
    pump penalty, default), ``"max_margin"`` (coolest), or
    ``"min_channels"`` (simplest).  If nothing is feasible, returns the
    best-margin attempt with ``feasible=False`` and a diagnosis.
    """
    Rt = float(contour["Rt"])
    circ = 2.0 * math.pi * Rt
    mdot_cool, mdot_total = coolant_flow_from_cycle(
        Pc, Rt, float(prop.c_star), mixture_ratio,
        cooling_fraction=cooling_fraction)
    hf = bartz_heat_flux(contour, Pc, prop, wall_temperature=900.0)
    material = SimpleNamespace(conductivity=wall_k)

    candidates: list[dict] = []
    for w in np.linspace(w_min, min(w_max, circ / 8.0), n_w):
        n_fit = int(math.floor(circ / (w + land_min)))   # channels that fit
        if n_fit < 8:
            continue
        for N in np.unique(np.linspace(8, n_fit, n_count).astype(int)):
            spec = SimpleNamespace(
                method="regenerative", coolant=coolant, channel_count=int(N),
                channel_width=float(w), channel_height=channel_height,
                coolant_mass_flow=mdot_cool, coolant_cp=None,
                coolant_inlet_temperature=300.0,
                max_wall_temperature=wall_temp_limit,
                coolant_density=None, coolant_viscosity=None,
                coolant_conductivity=None)
            res = regenerative_cooling_analysis(
                hf, contour, spec, material, wall_thickness, prop, Pc,
                n_iter=n_iter)
            dp_bar = res["coolant_pressure_drop"] / 1e5
            fits = not any("do not fit" in m for m in res["warnings"])
            feasible = (res["cooling_margin"] >= margin_target
                        and dp_bar <= dp_budget_bar and fits)
            candidates.append({
                "N": int(N), "w": float(w),
                "margin": float(res["cooling_margin"]),
                "dp_bar": float(dp_bar),
                "peak_wall_T": float(res["peak_gas_side_wall_temperature"]),
                "fits": bool(fits), "feasible": bool(feasible),
            })

    feas = [c for c in candidates if c["feasible"]]
    if feas:
        key = {"min_dp": lambda c: c["dp_bar"],
               "max_margin": lambda c: -c["margin"],
               "min_channels": lambda c: c["N"]}[objective]
        best = min(feas, key=key)
        diagnosis = "feasible"
    else:
        best = max(candidates, key=lambda c: c["margin"]) if candidates else None
        # Why did it fail? — the closest-margin design tells us.
        if best is None:
            diagnosis = "no candidate channels fit the throat"
        elif best["margin"] < margin_target:
            diagnosis = ("cooling requirement unmet at any geometry: coolant "
                         "flow too low or heat flux too high — raise mixture "
                         "ratio / cooling fraction, lower Pc, or use a higher-k "
                         "wall / film cooling")
        else:
            diagnosis = "margin reachable but only above the pressure-drop budget"

    return {
        "mdot_cool": mdot_cool, "mdot_total": mdot_total,
        "channel_count": best["N"] if best else None,
        "channel_width": best["w"] if best else None,
        "channel_height": channel_height,
        "margin": best["margin"] if best else float("nan"),
        "pressure_drop_bar": best["dp_bar"] if best else float("nan"),
        "peak_wall_T": best["peak_wall_T"] if best else float("nan"),
        "feasible": bool(feas),
        "objective": objective,
        "diagnosis": diagnosis,
        "requirement": {"margin_target": margin_target,
                        "dp_budget_bar": dp_budget_bar,
                        "wall_temp_limit": wall_temp_limit},
        "candidates": candidates,
        "model": "channel_auto_size",
    }


def _evaluate(Rt, epsilon, Pc, prop, cooling, material, *,
              length_pct, wall_thickness, Rd_factor, channel_scale):
    """Generate a contour at the trial throat curvature and evaluate the
    coupled cooling, with the channel cross-section scaled by
    ``channel_scale`` (≤ 1 shrinks the channels, raising the mass flux)."""
    contour = bell_nozzle_contour(
        Rt, epsilon, length_pct=length_pct, Rd_factor=Rd_factor,
        gamma=float(prop.gamma),
    )
    scaled_cooling = replace(
        cooling,
        channel_width=float(cooling.channel_width) * channel_scale,
        channel_height=float(cooling.channel_height) * channel_scale,
    ) if _is_dataclass(cooling) else _scaled_namespace(cooling, channel_scale)
    thermal = bartz_heat_flux(contour, Pc, prop)
    cool = regenerative_cooling_analysis(
        thermal, contour, scaled_cooling, material, wall_thickness, prop, Pc,
    )
    return contour, thermal, cool, scaled_cooling


def _is_dataclass(obj) -> bool:
    return hasattr(type(obj), "__dataclass_fields__")


def _scaled_namespace(cooling, scale):
    from types import SimpleNamespace
    fields = {k: getattr(cooling, k) for k in dir(cooling)
              if not k.startswith("__") and not callable(getattr(cooling, k))}
    fields["channel_width"] = float(cooling.channel_width) * scale
    fields["channel_height"] = float(cooling.channel_height) * scale
    return SimpleNamespace(**fields)


def cooling_coupled_contour(
    Rt: float,
    epsilon: float,
    Pc: float,
    prop: Any,
    cooling: Any,
    material: Any,
    *,
    length_pct: float = 80.0,
    wall_thickness: float = 0.001,
    cooling_margin_target: float = 1.2,
    rd_factor_bounds: tuple[float, float] = (0.382, 3.0),
    channel_scale_bounds: tuple[float, float] = (0.4, 1.0),
    n_throat_steps: int = 6,
    n_channel_steps: int = 8,
) -> dict:
    """Select a contour whose cooling margin meets ``cooling_margin_target``,
    co-designing throat curvature then channel mass flux.

    Returns a dict with the selected ``contour``, ``thermal`` (Bartz),
    ``cooling`` (coupled Sieder-Tate state), the chosen ``Rd_factor`` /
    ``channel_scale``, ``cooling_margin``, ``feasible`` (whether the
    target was met within the lever bounds), ``binding_lever``, and the
    ``history`` of (Rd_factor, channel_scale, margin, peak_wall_T).
    """
    rd_lo, rd_hi = rd_factor_bounds
    cs_lo, cs_hi = channel_scale_bounds
    history: list[dict] = []

    def step(rd, cs):
        contour, thermal, cool, used = _evaluate(
            Rt, epsilon, Pc, prop, cooling, material,
            length_pct=length_pct, wall_thickness=wall_thickness,
            Rd_factor=rd, channel_scale=cs,
        )
        rec = {
            "Rd_factor": float(rd),
            "channel_scale": float(cs),
            "cooling_margin": float(cool["cooling_margin"]),
            "peak_wall_T": float(cool["peak_gas_side_wall_temperature"]),
        }
        history.append(rec)
        return contour, thermal, cool, used, rec

    best = None
    binding = "throat_curvature"

    # Lever 1: open the throat curvature (cheap, weak).
    rds = [rd_lo + (rd_hi - rd_lo) * i / (n_throat_steps - 1)
           for i in range(n_throat_steps)]
    for rd in rds:
        contour, thermal, cool, used, rec = step(rd, cs_hi)
        best = (contour, thermal, cool, used, rd, cs_hi)
        if rec["cooling_margin"] >= cooling_margin_target:
            return _result(best, history, True, "throat_curvature")

    # Lever 2: shrink the channels (strong) at the most-open throat.
    binding = "channel_mass_flux"
    css = [cs_hi + (cs_lo - cs_hi) * i / (n_channel_steps - 1)
           for i in range(n_channel_steps)]
    for cs in css:
        contour, thermal, cool, used, rec = step(rd_hi, cs)
        best = (contour, thermal, cool, used, rd_hi, cs)
        if rec["cooling_margin"] >= cooling_margin_target:
            return _result(best, history, True, "channel_mass_flux")

    # Target not met within bounds: return the best (lowest peak wall T)
    # so the caller sees the binding limit.
    best_rec = min(history, key=lambda r: r["peak_wall_T"])
    contour, thermal, cool, used, _r = step(
        best_rec["Rd_factor"], best_rec["channel_scale"])
    best = (contour, thermal, cool, used,
            best_rec["Rd_factor"], best_rec["channel_scale"])
    return _result(best, history, False, binding)


def _result(best, history, feasible, binding):
    contour, thermal, cool, used, rd, cs = best
    return {
        "contour": contour,
        "thermal": thermal,
        "cooling": cool,
        "cooling_spec_used": used,
        "Rd_factor": float(rd),
        "channel_scale": float(cs),
        "cooling_margin": float(cool["cooling_margin"]),
        "peak_wall_temperature": float(cool["peak_gas_side_wall_temperature"]),
        "feasible": bool(feasible),
        "binding_lever": binding,
        "history": history,
        "model": "cooling_coupled_contour",
    }
