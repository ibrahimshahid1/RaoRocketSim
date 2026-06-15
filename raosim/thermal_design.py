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

from dataclasses import replace
from typing import Any

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis


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
