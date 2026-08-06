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
from raosim.physics import (
    bartz_heat_flux,
    channel_pressure_hoop_radius,
    coaxial_shell_wall_stress_profile,
    coffin_manson_cycles,
    regenerative_cooling_analysis,
    resolve_coolant_inlet_temperature,
    rib_supported_liner_buckling_profile,
    sp125_inelastic_buckling_critical_stress,
    thermal_fatigue_strain,
    total_strain_life_cycles,
)
from raosim.materials import cyclic_tangent_modulus
from raosim.regen_profile import (
    RegenWallProfile,
    helix_passage_lengths,
    helix_stretch_factors,
    normal_offset_contour,
)


def _fuel_injector_drop_from_split(
    Pc: float,
    *,
    fuel_injector_dp_fraction: float | None,
    injector_pressure_drop: float,
) -> float:
    """Resolve the fuel-side regen outlet dP from the split injector model."""
    legacy_drop = float(injector_pressure_drop or 0.0)
    if legacy_drop != 0.0:
        raise ValueError(
            "injector_pressure_drop is deprecated and no longer controls "
            "thermal sizing; use fuel_injector_dp_fraction or an explicit "
            "coolant_outlet_pressure."
        )
    if fuel_injector_dp_fraction is None:
        return 0.0
    if fuel_injector_dp_fraction <= 0.0:
        raise ValueError("fuel_injector_dp_fraction must be positive")
    return float(fuel_injector_dp_fraction) * float(Pc)


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
    channel_roughness: float = 0.0,
    gate_coolant_chemistry: bool = False,
    curvature_correction: bool = False,
    wall_thickness: float = 0.001,
    wall_k: float = 350.0,
    coolant: str = "rp1",
    helix_turns: float = 0.0,
    cooling_options: dict[str, Any] | None = None,
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
            spec_fields = dict(
                method="regenerative", coolant=coolant, channel_count=int(N),
                channel_width=float(w), channel_height=channel_height,
                channel_roughness=channel_roughness,
                coolant_mass_flow=mdot_cool, coolant_cp=None,
                coolant_inlet_temperature=None,
                max_wall_temperature=wall_temp_limit,
                coolant_density=None, coolant_viscosity=None,
                coolant_conductivity=None)
            spec_fields.update(cooling_options or {})
            spec = SimpleNamespace(**spec_fields)
            res = regenerative_cooling_analysis(
                hf, contour, spec, material, wall_thickness, prop, Pc,
                n_iter=n_iter, helix_turns=helix_turns,
                curvature_correction=curvature_correction)
            dp_bar = res["coolant_pressure_drop"] / 1e5
            fits = not any("do not fit" in m for m in res["warnings"])
            feasible = (
                res["cooling_margin"] >= margin_target
                and (
                    not gate_coolant_chemistry
                    or res["coolant_chemistry_feasible"]
                )
                and dp_bar <= dp_budget_bar
                and fits
            )
            candidates.append({
                "N": int(N), "w": float(w),
                "margin": float(res["cooling_margin"]),
                "dp_bar": float(dp_bar),
                "peak_wall_T": float(res["peak_gas_side_wall_temperature"]),
                "coolant_chemistry_margin":
                    float(res["coolant_chemistry_margin"]),
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
        elif (
            gate_coolant_chemistry
            and best["coolant_chemistry_margin"] < 1.0
        ):
            diagnosis = (
                "coolant-chemistry-limited: the RP-1/kerosene coolant-side "
                "wall exceeds the conservative coking screen"
            )
        elif best["margin"] < margin_target:
            diagnosis = ("cooling requirement unmet at any geometry: coolant "
                         "flow too low or heat flux too high — lower O/F "
                         "(more fuel coolant at fixed total flow), raise the "
                         "cooling fraction, lower Pc, or use a higher-k wall / "
                         "film cooling; re-run CEA when O/F changes because the "
                         "combustion state also changes")
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
        "coolant_chemistry_margin": (
            best["coolant_chemistry_margin"] if best else float("nan")
        ),
        "feasible": bool(feas),
        "objective": objective,
        "diagnosis": diagnosis,
        "requirement": {"margin_target": margin_target,
                        "dp_budget_bar": dp_budget_bar,
                        "wall_temp_limit": wall_temp_limit,
                        "channel_roughness": channel_roughness,
                        "gate_coolant_chemistry":
                            bool(gate_coolant_chemistry),
                        "curvature_correction":
                            bool(curvature_correction),
                        "coolant_inlet_temperature":
                            resolve_coolant_inlet_temperature(
                                SimpleNamespace(
                                    coolant=coolant,
                                    coolant_inlet_temperature=(
                                        cooling_options or {}
                                    ).get("coolant_inlet_temperature"),
                                )
                            )},
        "candidates": candidates,
        "helix_turns": float(helix_turns),
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
        curvature_correction=False,
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


# --------------------------------------------------------------------------- #
# Joint wall + channel design — size t_hot AND the channels together against    #
# the coupled thermal + structural limits (SP-125 coaxial-shell wall).         #
# --------------------------------------------------------------------------- #
def _resolve_material(material: Any) -> Any:
    """Accept a catalog name or a populated MaterialSpec/MaterialProperties.

    The joint design needs the elastic/thermal properties for the SP-125
    eq. 4-31 stress, so a bare ``SimpleNamespace(conductivity=...)`` is
    not enough; a string is looked up in :mod:`raosim.materials`."""
    if isinstance(material, str):
        from raosim.materials import get_material
        return get_material(material)
    for attr in ("conductivity", "yield_strength", "max_temperature",
                 "elastic_modulus", "thermal_expansion", "poisson_ratio"):
        if getattr(material, attr, None) is None:
            raise ValueError(
                f"material missing '{attr}'; pass a catalog material name "
                "or a fully-populated MaterialSpec/MaterialProperties")
    return material


def _wall_mass(
    contour: dict,
    t_hot: float,
    t_jacket: float,
    channel_height: float,
    density: float,
    *,
    channel_count: int | None = None,
    channel_width: float | None = None,
    land_width=None,
) -> float:
    """Liner + channel-land + jacket metal mass [kg].

    Corrected 2026-07-31.  The previous implementation had three defects, all
    of which biased the channel auto-sizer's ``min_mass`` objective:

    1. **Quadrature.**  It summed ``hypot(gradient(x), gradient(y))``, which
       gives each end node a full segment and over-counts the meridian by one
       grid interval -- the failure mode
       :func:`raosim.regen_profile._nodal_weights_from_segments` was written to
       avoid.  Trapezoidal nodal weights are used instead, so the weights sum
       to the true arc length.
    2. **Shell radius.**  It used the gas-side radius rather than the
       mid-surface radius, under-counting each shell by ``t/2``.  NASA SP-125
       eq. 8-32 (``W_c = 2 pi a l_c t_c rho``, printed p. 339) takes ``a`` as
       the *nominal* radius, i.e. Pappus's centroid theorem.
    3. **Missing land metal.**  The ribs between coolant channels were absent
       entirely.  On the 13 kN baseline the lands are about a third of the
       thrust-chamber mass, so a mass objective that ignored them was ranking
       channel layouts by the wrong quantity -- and *systematically* so, since
       narrower channels mean wider lands and therefore more metal.

    When ``channel_count`` and ``channel_width`` are supplied the land term is
    included; otherwise the result is liner + jacket only, as before, and the
    caller is responsible for knowing that.  See :mod:`raosim.mass_ledger` for
    the full ledger this mirrors.
    """

    from raosim.regen_profile import _nodal_weights_from_segments

    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    ds = _nodal_weights_from_segments(np.hypot(np.diff(x), np.diff(y)))

    r_ch_in = y + t_hot
    r_ch_out = r_ch_in + channel_height
    liner = float(np.sum(2.0 * math.pi * (y + 0.5 * t_hot) * t_hot * ds))
    jacket = float(np.sum(
        2.0 * math.pi * (r_ch_out + 0.5 * t_jacket) * t_jacket * ds
    ))

    lands = 0.0
    if channel_count and channel_width and channel_width > 0.0:
        if land_width is None:
            r_mid = y + t_hot + 0.5 * channel_height
            pitch = 2.0 * math.pi * np.maximum(r_mid, 1e-12) / int(channel_count)
            band = np.maximum(pitch - float(channel_width), 0.0)
        else:
            band = np.asarray(land_width, dtype=float)
            if band.ndim == 0:
                band = np.full_like(y, float(band))
        annulus = math.pi * (r_ch_out ** 2 - r_ch_in ** 2)
        fraction = band / np.maximum(band + float(channel_width), 1e-30)
        lands = float(np.sum(annulus * fraction * ds))

    return density * (liner + lands + jacket)


def wall_feasibility_band(
    t_values, thermal_margins, structural_margins, *,
    thermal_target: float, structural_fos: float, t_mfg: float,
    cycles=None, cycles_required: float | None = None,
) -> dict:
    """The SP-125 hot-wall thickness squeeze, made explicit.

    The thermal side puts an UPPER bound on the hot-wall thickness — a
    thicker liner runs a hotter gas-side wall (``q t_w/k_w``), so it fails
    the temperature limit beyond some ``t_thermal``.  The structural side
    puts a LOWER bound — a thinner liner carries more pressure stress
    ``(p_co−p_g)R/t`` — and, through the thermal stress ``E α q t/(2(1−ν)k)``
    which *grows* with ``t``, an upper bound too.  When ``cycles`` (the
    Coffin-Manson ``N_f`` at each ``t``) and ``cycles_required`` are given,
    low-cycle FATIGUE adds its own interval (thermal strain ∝ ΔT ∝ t worsens
    life thick-side, pressure strain ∝ 1/t thin-side).  With manufacturing's
    floor this is

        max(t_pressure, t_fatigue, t_manufacturing) ≤ t_hot ≤ t_thermal

    (SP-125 printed pp. 108-110).  The feasible band is the true
    intersection of all the per-constraint OK sets over ``t_values``
    (fixed channels); per-constraint sub-bounds are reported too.
    """
    t = np.asarray(t_values, dtype=float)
    th = np.asarray(thermal_margins, dtype=float)
    st = np.asarray(structural_margins, dtype=float)
    thermal_ok = th >= thermal_target
    struct_ok = st >= structural_fos
    all_ok = thermal_ok & struct_ok & (t >= t_mfg)
    out = {
        "t_thermal_max": float(t[thermal_ok].max()) if np.any(thermal_ok) else float("nan"),
        "t_structural_lo": float(t[struct_ok].min()) if np.any(struct_ok) else float("nan"),
        "t_structural_hi": float(t[struct_ok].max()) if np.any(struct_ok) else float("nan"),
        "t_manufacturing": float(t_mfg),
    }
    if cycles is not None and cycles_required is not None:
        fat_ok = np.asarray(cycles, dtype=float) >= float(cycles_required)
        all_ok = all_ok & fat_ok
        out["t_fatigue_lo"] = float(t[fat_ok].min()) if np.any(fat_ok) else float("nan")
        out["t_fatigue_hi"] = float(t[fat_ok].max()) if np.any(fat_ok) else float("nan")
    feas_t = t[all_ok]
    out["feasible_lo"] = float(feas_t.min()) if feas_t.size else float("nan")
    out["feasible_hi"] = float(feas_t.max()) if feas_t.size else float("nan")
    out["feasible"] = bool(feas_t.size > 0)
    return out


def joint_wall_channel_design(
    contour: dict,
    prop: Any,
    Pc: float,
    *,
    material: Any,
    mixture_ratio: float = 2.6,
    cooling_fraction: float = 1.0,
    coolant: str = "rp1",
    thermal_margin: float = 1.2,
    structural_fos: float = 1.0,
    required_cycles: float = 100.0,
    life_fos: float = 4.0,
    dp_budget_bar: float = 200.0,
    helix_turns: float = 0.0,
    channel_height: float = 0.0030,
    channel_roughness: float = 0.0,
    gate_coolant_chemistry: bool = False,
    curvature_correction: bool = False,
    channel_height_min: float | None = None,
    channel_height_max: float | None = None,
    n_height: int = 1,
    t_hot_min: float = 0.0005,
    t_hot_max: float = 0.0030,
    n_t: int = 6,
    w_min: float = 0.0005,
    w_max: float = 0.0020,
    land_min: float = 0.0005,
    n_w: int = 4,
    n_count: int = 6,
    coolant_outlet_pressure: float | None = None,
    fuel_injector_dp_fraction: float | None = None,
    injector_pressure_drop: float = 0.0,
    cooling_options: dict[str, Any] | None = None,
    objective: str = "min_mass",
    n_iter: int = 12,
) -> dict:
    """Co-size ``t_hot`` and channel ``N``/``w``/**``h``** against the
    coupled thermal and structural limits — the step
    that ``size_cooling_channels`` (cooling only, fixed wall) cannot take.

    For each ``(t_hot, N, w, h)`` on the grid the coupled Sieder-Tate cooling
    is solved (helix-aware Δp) and the SP-125 eq. 4-31 combined wall stress
    is evaluated station by station, giving four screens:

    * thermal:    ``material.max_temperature / peak_wall_T ≥ thermal_margin``
    * structural: ``yield / combined_stress ≥ structural_fos``
    * hydraulic:  ``Δp ≤ dp_budget_bar`` (rises with ``helix_turns``)
    * geometric:  the channels fit the throat circumference

    plus the manufacturing floor ``t_hot ≥ t_hot_min``.  Among the feasible
    set ``objective`` picks ``"min_mass"`` (default), ``"min_dp"``,
    ``"max_thermal_margin"`` or ``"min_wall_temp"``.  The result reports the
    SP-125 thickness feasibility band (:func:`wall_feasibility_band`), the
    chosen design and, when nothing is feasible, the best-effort design with
    a diagnosis of the binding limit.

    Note on the structural gate: at a real throat flux (~50 MW/m²) the
    eq. 4-31 *thermal* stress alone approaches copper's yield, so a
    high-Pc copper throat will not clear a yield factor of safety — real
    regen liners are designed to low-cycle *fatigue* life, running near or
    above yield.  ``structural_fos`` therefore defaults to 1.0 (yield as a
    screening proxy).  Coffin-Manson life is reported only when the caller
    supplies a complete, explicitly sourced coefficient set, and it gates
    feasibility only when that set is marked design-qualified.  The
    chamber-pressure hoop is carried by the outer jacket, not this liner.
    The coolant-gas load is computed station by station from the hydraulic
    pressure march and the wall gas pressure.

    ``material`` is a catalog name (e.g. ``"grcop-84"``) or a populated
    MaterialSpec/MaterialProperties (it must carry E, α, ν for the stress).
    """
    valid_objectives = {"min_mass", "min_dp", "max_thermal_margin", "min_wall_temp"}
    if objective not in valid_objectives:
        raise ValueError(
            f"objective must be one of {sorted(valid_objectives)}, got {objective!r}")
    mat = _resolve_material(material)
    h_lo = float(channel_height if channel_height_min is None else channel_height_min)
    h_hi = float(channel_height if channel_height_max is None else channel_height_max)
    if not (0.0 < h_lo <= h_hi):
        raise ValueError("channel-height bounds must satisfy 0 < min <= max")
    height_grid = np.linspace(h_lo, h_hi, max(int(n_height), 1))
    Rt = float(contour["Rt"])
    circ = 2.0 * math.pi * Rt
    mdot_cool, mdot_total = coolant_flow_from_cycle(
        Pc, Rt, float(prop.c_star), mixture_ratio, cooling_fraction=cooling_fraction)
    injector_drop = _fuel_injector_drop_from_split(
        Pc,
        fuel_injector_dp_fraction=fuel_injector_dp_fraction,
        injector_pressure_drop=injector_pressure_drop,
    )
    hf = bartz_heat_flux(contour, Pc, prop, wall_temperature=900.0)
    T_max = float(getattr(mat, "max_temperature"))
    E = float(mat.elastic_modulus); alpha = float(mat.thermal_expansion)
    nu = float(mat.poisson_ratio); k = float(mat.conductivity)
    Sy = float(mat.yield_strength); rho = float(getattr(mat, "density", 8000.0) or 8000.0)
    cool_material = SimpleNamespace(conductivity=k)
    # Coffin-Manson fatigue is only evaluated with a complete, explicitly
    # sourced coefficient set.  It only gates feasibility when the material
    # marks those data as design-qualified.  SP-125 establishes thermal LCF
    # as a governing mode but does not provide alloy-specific coefficients.
    f_sf = getattr(mat, "fatigue_strength_coeff", None)
    f_b = getattr(mat, "fatigue_strength_exp", None)
    f_ef = getattr(mat, "fatigue_ductility_coeff", None)
    f_c = getattr(mat, "fatigue_ductility_exp", None)
    total_strain_curves = tuple(
        getattr(mat, "fatigue_total_strain_curves", ()) or ()
    )
    fatigue_source = getattr(mat, "fatigue_source", None)
    has_cm_fatigue = (
        None not in (f_sf, f_b, f_ef, f_c) and bool(fatigue_source)
    )
    has_direct_fatigue = bool(total_strain_curves and fatigue_source)
    has_fatigue = has_cm_fatigue or has_direct_fatigue
    fatigue_gates = bool(
        has_fatigue
        and (
            getattr(mat, "fatigue_design_qualified", False)
            or getattr(mat, "fatigue_screening_gate", False)
        )
    )
    cyc_target = float(required_cycles) * float(life_fos)

    t_grid = np.linspace(t_hot_min, t_hot_max, max(n_t, 2))
    candidates: list[dict] = []
    for h in height_grid:
        for w in np.linspace(w_min, min(w_max, circ / 8.0), n_w):
            n_fit = int(math.floor(circ / (w + land_min)))
            if n_fit < 8:
                continue
            for N in np.unique(np.linspace(8, n_fit, n_count).astype(int)):
                for t_hot in t_grid:
                    spec_fields = dict(
                        method="regenerative", coolant=coolant, channel_count=int(N),
                        channel_width=float(w), channel_height=float(h),
                        channel_roughness=channel_roughness,
                        coolant_mass_flow=mdot_cool, coolant_cp=None,
                        coolant_inlet_temperature=None,
                        max_wall_temperature=T_max,
                        coolant_density=None, coolant_viscosity=None,
                        coolant_conductivity=None)
                    spec_fields.update(cooling_options or {})
                    spec = SimpleNamespace(**spec_fields)
                    res = regenerative_cooling_analysis(
                        hf, contour, spec, cool_material, float(t_hot), prop, Pc,
                        n_iter=n_iter, helix_turns=helix_turns,
                        curvature_correction=curvature_correction,
                        coolant_outlet_pressure=coolant_outlet_pressure,
                        injector_pressure_drop=injector_drop)
                    th_margin = float(res["cooling_margin"])
                    dp_bar = float(res["coolant_pressure_drop"] / 1e5)
                    fits = not any("do not fit" in m for m in res["warnings"])
                    stress = coaxial_shell_wall_stress_profile(
                        pressure_differential=res["liner_pressure_differential"],
                        # SP-125 eq. 4-27 hoop radius = channel tube radius,
                        # NOT the nozzle shell radius contour["y"].
                        inner_radius=channel_pressure_hoop_radius(w, t_hot),
                        wall_thickness=float(t_hot), heat_flux=res["q"],
                        elastic_modulus=E, thermal_expansion=alpha, poisson_ratio=nu,
                        conductivity=k, yield_strength=Sy)
                    st_margin = float(stress["stress_margin"])
                    # Low-cycle fatigue: strain from the worst through-wall ΔT
                    # (T_wg − T_wc) plus the mechanical (pressure) strain →
                    # Coffin-Manson N_f, vs the required cyclic life × FoS.
                    Twg = np.asarray(res["gas_side_wall_temperature"], dtype=float)
                    Twc = np.asarray(res["coolant_side_wall_temperature"], dtype=float)
                    dT_wall_profile = Twg - Twc
                    fatigue_model = None
                    fatigue_curve = None
                    fatigue_model_status = None
                    if has_fatigue:
                        eps_profile = np.asarray([
                            thermal_fatigue_strain(
                                dT,
                                thermal_expansion=alpha,
                                poisson_ratio=nu,
                                mechanical_strain=float(sigma_p) / E,
                            )
                            for dT, sigma_p in zip(
                                dT_wall_profile,
                                np.abs(stress["pressure_stress_profile"]),
                            )
                        ])
                        fatigue_idx = int(np.argmax(eps_profile))
                        eps = float(eps_profile[fatigue_idx])
                        if has_cm_fatigue:
                            Nf = coffin_manson_cycles(
                                eps, elastic_modulus=E,
                                fatigue_strength_coeff=f_sf,
                                fatigue_strength_exp=f_b,
                                fatigue_ductility_coeff=f_ef,
                                fatigue_ductility_exp=f_c)
                            fatigue_model = "coffin_manson_basquin"
                            fatigue_model_status = "sourced_strain_life_coefficients"
                        else:
                            direct = total_strain_life_cycles(
                                eps,
                                total_strain_curves,
                                temperature=float(Twg[fatigue_idx]),
                            )
                            Nf = direct["cycles"]
                            fatigue_model = "direct_total_strain_life"
                            fatigue_curve = direct.get("curve")
                            fatigue_model_status = direct.get("status")
                    else:
                        eps, Nf = None, None
                    fatigue_ok = (
                        Nf is not None and Nf >= cyc_target
                    ) if fatigue_gates else True
                    feasible = (
                        th_margin >= thermal_margin
                        and (
                            not gate_coolant_chemistry
                            or res["coolant_chemistry_feasible"]
                        )
                        and res.get("boiling_chf_feasible", True)
                        and st_margin >= structural_fos
                        and fatigue_ok
                        and dp_bar <= dp_budget_bar
                        and fits
                    )
                    candidates.append({
                        "t_hot": float(t_hot), "N": int(N), "w": float(w),
                        "h": float(h),
                        "thermal_margin": th_margin,
                        "structural_margin": st_margin,
                        "N_f": float(Nf) if Nf is not None else None,
                        "strain_range": float(eps) if eps is not None else None,
                        "fatigue_ok": bool(fatigue_ok), "dp_bar": dp_bar,
                        "fatigue_model": fatigue_model,
                        "fatigue_curve": fatigue_curve,
                        "fatigue_model_status": fatigue_model_status,
                        "peak_wall_T": float(
                            res["peak_gas_side_wall_temperature"]
                        ),
                        "coolant_chemistry_margin": float(
                            res["coolant_chemistry_margin"]
                        ),
                        "combined_stress_MPa": float(
                            stress["combined_stress"] / 1e6
                        ),
                        "stress_governing_index": int(
                            stress["governing_index"]
                        ),
                        "max_liner_pressure_differential_bar": float(
                            np.max(res["liner_pressure_differential"]) / 1e5
                        ),
                        # Channel count and width are passed so the LAND metal
                        # is in the objective.  Without them a "min_mass"
                        # channel search is biased: narrower channels leave
                        # wider ribs, so ignoring lands makes fine channels look
                        # free when they are not.
                        "mass_kg": _wall_mass(
                            contour, float(t_hot), float(t_hot), float(h), rho,
                            channel_count=int(N), channel_width=float(w),
                        ),
                        "fits": bool(fits), "feasible": bool(feasible),
                    })

    feas = [c for c in candidates if c["feasible"]]
    keymap = {
        "min_mass": lambda c: c["mass_kg"],
        "min_dp": lambda c: c["dp_bar"],
        "max_thermal_margin": lambda c: -c["thermal_margin"],
        "min_wall_temp": lambda c: c["peak_wall_T"],
    }

    if feas:
        best = min(feas, key=keymap[objective])
        diagnosis = "feasible"
    elif candidates:
        # Closest miss: maximise the worst normalised margin (thermal,
        # structural and — when available — fatigue), penalising Δp
        # overshoot and non-fit.  The eq. 4-31 structural term is
        # width-independent (coaxial shell), so add a tiny cooling
        # tie-break — otherwise wide, poorly-cooled channels tie whenever
        # structure binds and the report is uninformative.
        def _norm(c):
            n = [c["thermal_margin"] / thermal_margin,
                 c["structural_margin"] / structural_fos]
            if fatigue_gates and cyc_target > 0:
                n.append(c["N_f"] / cyc_target)
            if gate_coolant_chemistry:
                n.append(c["coolant_chemistry_margin"])
            return n
        def score(c):
            s = min(_norm(c))
            s -= max(0.0, c["dp_bar"] - dp_budget_bar) / max(dp_budget_bar, 1e-9)
            s -= (10.0 if not c["fits"] else 0.0)
            return s + 1e-3 * (c["thermal_margin"] / thermal_margin)
        best = max(candidates, key=score)
        # The binder is the most-violated constraint at the best effort.
        viol = {"thermal": best["thermal_margin"] / thermal_margin,
                "structural": best["structural_margin"] / structural_fos}
        if fatigue_gates and cyc_target > 0:
            viol["fatigue"] = best["N_f"] / cyc_target
        if gate_coolant_chemistry:
            viol["coolant chemistry"] = best["coolant_chemistry_margin"]
        if not best["fits"]:
            diagnosis = "no channel geometry fits the throat at the required count"
        elif best["dp_bar"] > dp_budget_bar:
            diagnosis = "pressure-drop-limited: reduce helix turns or widen the channels"
        else:
            binder = min(viol, key=viol.get)
            if binder == "thermal":
                diagnosis = ("thermal-limited: the wall runs too hot even at the thinnest "
                             "liner — use a higher-k liner, more coolant flow, or lower Pc")
            elif binder == "fatigue":
                diagnosis = ("fatigue-limited: low-cycle (thermal-strain) life is below the "
                             "required cycles — thin the liner, cut ΔT / heat flux, or use "
                             "a tougher (superalloy) liner")
            elif binder == "coolant chemistry":
                diagnosis = (
                    "coolant-chemistry-limited: the RP-1/kerosene coolant-"
                    "side wall exceeds the conservative coking screen"
                )
            else:
                diagnosis = ("structurally-limited: the combined pressure + thermal stress "
                             "exceeds yield/FoS — a stronger alloy, lower Pc / coolant-gas "
                             "differential, or an LCF design (the jacket carries the hoop)")
    else:
        best = None
        diagnosis = "no candidate channels fit the throat"

    band = None
    if best is not None:
        sl = sorted((c for c in candidates
                     if c["N"] == best["N"] and c["w"] == best["w"]
                     and c["h"] == best["h"]),
                    key=lambda c: c["t_hot"])
        band = wall_feasibility_band(
            [c["t_hot"] for c in sl], [c["thermal_margin"] for c in sl],
            [c["structural_margin"] for c in sl],
            thermal_target=thermal_margin, structural_fos=structural_fos,
            t_mfg=t_hot_min,
            cycles=[c["N_f"] for c in sl] if fatigue_gates else None,
            cycles_required=cyc_target if fatigue_gates else None)

    if fatigue_gates and getattr(mat, "fatigue_design_qualified", False):
        fatigue_status = "design_qualified_gate"
    elif fatigue_gates:
        fatigue_status = "sourced_screening_gate"
    elif has_fatigue:
        fatigue_status = "screening_only_not_gating"
    else:
        fatigue_status = "not_evaluated_missing_sourced_coefficients"

    return {
        "material": getattr(mat, "name", "?"),
        "mdot_cool": mdot_cool, "mdot_total": mdot_total,
        "t_hot": best["t_hot"] if best else None,
        "channel_count": best["N"] if best else None,
        "channel_width": best["w"] if best else None,
        "channel_height": best["h"] if best else None,
        "thermal_margin": best["thermal_margin"] if best else float("nan"),
        "structural_margin": best["structural_margin"] if best else float("nan"),
        "pressure_drop_bar": best["dp_bar"] if best else float("nan"),
        "peak_wall_T": best["peak_wall_T"] if best else float("nan"),
        "coolant_chemistry_margin": (
            best["coolant_chemistry_margin"] if best else float("nan")
        ),
        "combined_stress_MPa": best["combined_stress_MPa"] if best else float("nan"),
        "fatigue_cycles": best["N_f"] if best else None,
        "strain_range": best["strain_range"] if best else None,
        "fatigue_status": fatigue_status,
        "fatigue_source": fatigue_source,
        "fatigue_model": best["fatigue_model"] if best else None,
        "fatigue_curve": best["fatigue_curve"] if best else None,
        "fatigue_model_status": (
            best["fatigue_model_status"] if best else None
        ),
        "fatigue_gates_feasibility": fatigue_gates,
        "mass_kg": best["mass_kg"] if best else float("nan"),
        "stress_governing_index": best["stress_governing_index"] if best else None,
        "max_liner_pressure_differential_bar": (
            best["max_liner_pressure_differential_bar"] if best else float("nan")
        ),
        "coolant_outlet_pressure": (
            float(coolant_outlet_pressure)
            if coolant_outlet_pressure is not None
            else float(Pc) + max(float(injector_drop), 0.0)
        ),
        "coolant_pressure_boundary_source": (
            "user_supplied_coolant_outlet_pressure"
            if coolant_outlet_pressure is not None
            else "minimum_injector_entry_pressure_Pc_plus_injector_drop"
        ),
        "feasible": bool(feas),
        "objective": objective,
        "diagnosis": diagnosis,
        "band": band,
        "helix_turns": float(helix_turns),
        "requirement": {"thermal_margin": thermal_margin,
                        "structural_fos": structural_fos,
                        "required_cycles": required_cycles,
                        "life_fos": life_fos,
                        "dp_budget_bar": dp_budget_bar,
                        "channel_roughness": channel_roughness,
                        "gate_coolant_chemistry":
                            bool(gate_coolant_chemistry),
                        "curvature_correction":
                            bool(curvature_correction),
                        "t_hot_min": t_hot_min},
        "candidates": candidates,
        "model": "joint_wall_channel_design",
    }


# --------------------------------------------------------------------------- #
# Variable hot-wall + jacket profile — size t_hot(x) AND t_jacket(x) along the  #
# contour, not one uniform number (the SP-125 "passage/wall varies along the    #
# chamber" design).                                                             #
# --------------------------------------------------------------------------- #
def size_wall_profile(
    contour: dict,
    prop: Any,
    Pc: float,
    *,
    material: Any,
    channel_count: int,
    channel_width: float,
    channel_height: float,
    channel_roughness: float = 0.0,
    gate_coolant_chemistry: bool = False,
    curvature_correction: bool = False,
    mixture_ratio: float = 2.6,
    cooling_fraction: float = 1.0,
    coolant: str = "rp1",
    thermal_margin: float = 1.2,
    structural_fos: float = 1.0,
    jacket_fos: float = 1.5,
    jacket_material: Any = None,
    helix_turns: float = 0.0,
    t_hot_min: float = 0.0005,
    t_hot_max: float = 0.0030,
    t_jacket_min: float = 0.0005,
    channel_height_min: float | None = None,
    channel_height_max: float | None = None,
    n_channel_height: int = 3,
    height_relief_min: float = 1.25,
    height_relief_max: float = 2.0,
    n_height_relief: int = 2,
    dp_budget_bar: float = 200.0,
    buckling_fos: float = 1.0,
    buckling_tangent_modulus_fraction: float = 0.10,
    buckling_plate_knockdown: float = 0.65,
    gate_sp125_429: bool = False,
    coolant_outlet_pressure: float | None = None,
    fuel_injector_dp_fraction: float | None = None,
    injector_pressure_drop: float = 0.0,
    cooling_options: dict[str, Any] | None = None,
    n_outer: int = 4,
    n_iter: int = 20,
) -> dict:
    """Size a VARIABLE hot-wall + jacket profile along the contour.

    For fixed channels ``(N, w, h)`` this returns a station-wise
    :class:`~raosim.regen_profile.RegenWallProfile` rather than one uniform
    ``t_hot`` — a preliminary variable wall profile of the kind SP-125
    describes (passage/wall vary along the chamber; the throat is often the
    critical thermal region).

    * **Hot-wall ``t_hot(x)``** is set to the *thinnest* liner that keeps
      the SP-125 eq. 4-31 combined stress within ``yield/structural_fos`` at
      each station.  Thinnest is simultaneously the *coolest* (``T_wg =
      T_aw − q/h_g`` rises with ``t``) and the *lightest*, so the structural
      lower bound governs.  Writing the combined stress as ``a/t + b·t``
      (``a=|p_co−p_g|R``, ``b=Eαq/(2(1−ν)k)``) the thin root of
      ``a/t + b·t = S_lim`` is that bound.  Because ``q`` itself drops as
      ``t`` grows, the cooling solve is re-run to a fixed point.
    * **Channel depth ``h(x)``** is heat-shaped: narrow near the peak-flux
      region to raise coolant velocity/film coefficient, and relieved away
      from it to recover pressure drop.  The throat depth and relief ratio
      are searched against the thermal and global ``dp_budget_bar`` gates.
    * **Buckling** includes SP-125 equation 4-29 as an explicitly
      equivalent-tube longitudinal screen and a separately labeled
      rib-supported liner-wrinkling gate under coolant-over-gas pressure.
      Equation 4-29 was derived for tubular walls and needs hot-wall tangent
      moduli; it therefore reports but does not gate a milled-channel design
      unless ``gate_sp125_429=True`` is requested.
    * **Jacket ``t_jacket(x)``** carries the coolant-pressure hoop on the
      OUTER shell (SP-125: "the outer shell is subjected only to the hoop
      stress induced by the coolant pressure"), so
      ``t_jacket = p_coolant·R_jacket·jacket_fos / S_y,jacket`` — optionally
      a stronger ``jacket_material`` (the classic copper-liner /
      Inconel-jacket split).

    Returns the ``profile`` plus the per-station thermal feasibility, the
    structural-margin profile, liner/jacket masses and a feasibility flag.
    """
    mat = _resolve_material(material)
    jmat = _resolve_material(jacket_material) if jacket_material is not None else mat
    if Pc <= 0.0:
        raise ValueError("Pc must be positive")
    if channel_count <= 0 or channel_width <= 0.0 or channel_height <= 0.0:
        raise ValueError("channel count, width, and height must be positive")
    if not (0.0 < t_hot_min <= t_hot_max):
        raise ValueError("hot-wall bounds must satisfy 0 < min <= max")
    if t_jacket_min <= 0.0:
        raise ValueError("t_jacket_min must be positive")
    if (structural_fos <= 0.0 or jacket_fos <= 0.0
            or thermal_margin <= 0.0 or buckling_fos <= 0.0):
        raise ValueError("thermal and structural safety factors must be positive")
    if not (0.0 < buckling_tangent_modulus_fraction <= 1.0):
        raise ValueError("buckling_tangent_modulus_fraction must be in (0, 1]")
    E = float(mat.elastic_modulus); alpha = float(mat.thermal_expansion)
    nu = float(mat.poisson_ratio); k = float(mat.conductivity)
    Sy = float(mat.yield_strength); rho = float(getattr(mat, "density", 8000.0) or 8000.0)
    Sy_j = float(jmat.yield_strength); rho_j = float(getattr(jmat, "density", 8000.0) or 8000.0)
    T_max = float(mat.max_temperature); T_limit = T_max / max(thermal_margin, 1e-9)
    Slim = Sy / max(structural_fos, 1e-9)

    x = np.asarray(contour["x"], dtype=float)
    r_inner = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    n = len(x)
    throat_idx = int(np.argmin(r_inner))
    N = int(channel_count)
    w = float(channel_width)
    h_nom = float(channel_height)
    h_min = float(0.60 * h_nom if channel_height_min is None else channel_height_min)
    h_max = float(1.80 * h_nom if channel_height_max is None else channel_height_max)
    if not (0.0 < h_min <= h_max):
        raise ValueError("channel-height bounds must satisfy 0 < min <= max")
    if not (1.0 <= height_relief_min <= height_relief_max):
        raise ValueError("height relief must satisfy 1 <= min <= max")

    mdot_cool, mdot_total = coolant_flow_from_cycle(
        Pc, Rt, float(prop.c_star), mixture_ratio,
        cooling_fraction=cooling_fraction,
    )
    injector_drop = _fuel_injector_drop_from_split(
        Pc,
        fuel_injector_dp_fraction=fuel_injector_dp_fraction,
        injector_pressure_drop=injector_pressure_drop,
    )
    hf = bartz_heat_flux(contour, Pc, prop, wall_temperature=900.0)
    heat_shape = np.asarray(hf["q"], dtype=float)
    heat_shape = heat_shape / max(float(np.max(heat_shape)), 1e-12)
    cool_material = SimpleNamespace(conductivity=k)
    Et_fallback = E * float(buckling_tangent_modulus_fraction)
    Ec_fallback = E * float(buckling_tangent_modulus_fraction)

    def _spec(h_arr):
        fields = dict(
            method="regenerative", coolant=coolant, channel_count=N,
            channel_width=w, channel_height=float(np.mean(h_arr)),
            channel_roughness=channel_roughness,
            coolant_mass_flow=mdot_cool, coolant_cp=None,
            coolant_inlet_temperature=None,
            max_wall_temperature=T_max, coolant_density=None,
            coolant_viscosity=None, coolant_conductivity=None,
            coolant_outlet_pressure=coolant_outlet_pressure,
            injector_pressure_drop=injector_drop,
        )
        fields.update(cooling_options or {})
        return SimpleNamespace(**fields)

    def _profile(t_hot, h_arr, t_jacket):
        stretch = helix_stretch_factors(
            x,
            r_inner,
            helix_turns=helix_turns,
            t_wall=t_hot,
            channel_height=h_arr,
        )
        _, r_mid = normal_offset_contour(
            x, r_inner, t_hot + 0.5 * h_arr
        )
        pitch_normal = (
            2.0 * math.pi * np.maximum(r_mid, 1e-9)
            / max(N, 1)
            / stretch
        )
        land = np.maximum(pitch_normal - w, 0.0)
        return RegenWallProfile(
            x=x, r_inner=r_inner, t_hot=t_hot,
            channel_width=np.full(n, w), channel_height=h_arr,
            land_width=land, t_jacket=t_jacket, channel_count=N,
            helix_turns=float(helix_turns), Rt=Rt,
        )

    def _solve(t_hot, h_arr):
        return regenerative_cooling_analysis(
            hf, contour, _spec(h_arr), cool_material, None, prop, Pc,
            n_iter=n_iter, helix_turns=helix_turns,
            curvature_correction=curvature_correction,
            wall_profile=_profile(t_hot, h_arr, t_hot),
            coolant_outlet_pressure=coolant_outlet_pressure,
            injector_pressure_drop=injector_drop,
        )

    def _evaluate_height_profile(h_arr):
        # Fixed-point: q and the buckling/strength thickness bounds depend on
        # the wall itself, so update and re-run the cooling march.
        t_hot = np.full(n, min(max(0.001, t_hot_min), t_hot_max))
        res = _solve(t_hot, h_arr)
        for _ in range(max(n_outer, 1)):
            q = np.asarray(res["q"], dtype=float)
            p_diff = np.abs(
                np.asarray(res["liner_pressure_differential"], dtype=float)
            )
            liner_hoop_radius = channel_pressure_hoop_radius(w, t_hot)
            a = p_diff * liner_hoop_radius
            b = E * alpha * q / (2.0 * (1.0 - nu) * k)
            disc = Slim * Slim - 4.0 * a * b
            with np.errstate(divide="ignore", invalid="ignore"):
                t_quad = (
                    Slim - np.sqrt(np.maximum(disc, 0.0))
                ) / (2.0 * np.maximum(b, 1e-30))
                t_lin = a / Slim
                t_star = np.sqrt(a / np.maximum(b, 1e-30))
            t_yield = np.where(b > 1e-12, t_quad, t_lin)
            t_yield = np.where(disc >= 0.0, t_yield, t_star)

            Twg = np.asarray(res["gas_side_wall_temperature"], dtype=float)
            Twc = np.asarray(res["coolant_side_wall_temperature"], dtype=float)
            longitudinal_thermal = E * alpha * np.maximum(Twg - Twc, 0.0)
            tangent = cyclic_tangent_modulus(
                mat,
                longitudinal_thermal,
                0.5 * (Twg + Twc),
            )
            if tangent["available"]:
                Et = np.asarray(tangent["tangent_modulus"], dtype=float)
                Ec = Et
            else:
                Et = np.full(n, Et_fallback)
                Ec = np.full(n, Ec_fallback)
            # SP-125 4-29 defines r as the tube radius.  A milled rectangular
            # channel has no unique tube radius, so h/2 is an explicit
            # equivalent-tube approximation, not the nozzle-shell radius.
            buckling_radius = np.maximum(0.5 * h_arr, t_hot_min)
            sc_per_t = sp125_inelastic_buckling_critical_stress(
                wall_thickness=np.ones(n),
                local_radius=buckling_radius,
                tangent_modulus_tension=Et,
                tangent_modulus_compression=Ec,
                poisson_ratio=nu,
            )
            t_sp125 = (
                longitudinal_thermal * buckling_fos
                / np.maximum(0.9 * sc_per_t, 1e-9)
            )

            plate_constant = (
                buckling_plate_knockdown * 4.0 * math.pi ** 2 * E
                / max(12.0 * (1.0 - nu ** 2), 1e-12)
            )
            t_pressure_buckle = np.cbrt(
                p_diff * r_inner * w ** 2 * buckling_fos
                / max(plate_constant, 1e-12)
            )
            lower_bounds = [
                t_yield,
                t_pressure_buckle,
                np.full(n, t_hot_min),
            ]
            if gate_sp125_429:
                lower_bounds.append(t_sp125)
            t_new = np.clip(
                np.maximum.reduce(lower_bounds),
                t_hot_min,
                t_hot_max,
            )
            moved = float(np.max(np.abs(t_new - t_hot)))
            t_hot = t_new
            res = _solve(t_hot, h_arr)
            if moved < 1e-6:
                break

        q = np.asarray(res["q"], dtype=float)
        p_diff = np.abs(
            np.asarray(res["liner_pressure_differential"], dtype=float)
        )
        Twg = np.asarray(res["gas_side_wall_temperature"], dtype=float)
        Twc = np.asarray(res["coolant_side_wall_temperature"], dtype=float)
        thermal_ok = Twg <= T_limit + 1e-6
        liner_hoop_radius = channel_pressure_hoop_radius(w, t_hot)
        combined = (
            p_diff * liner_hoop_radius / np.maximum(t_hot, 1e-12)
            + E * alpha * q * t_hot / (2.0 * (1.0 - nu) * k)
        )
        struct_margin = Sy / np.maximum(combined, 1e-9)

        longitudinal_thermal = E * alpha * np.maximum(Twg - Twc, 0.0)
        tangent = cyclic_tangent_modulus(
            mat,
            longitudinal_thermal,
            0.5 * (Twg + Twc),
        )
        if tangent["available"]:
            Et = np.asarray(tangent["tangent_modulus"], dtype=float)
            Ec = Et
        else:
            Et = np.full(n, Et_fallback)
            Ec = np.full(n, Ec_fallback)
        buckling_radius = np.maximum(0.5 * h_arr, t_hot_min)
        critical_429 = sp125_inelastic_buckling_critical_stress(
            wall_thickness=t_hot,
            local_radius=buckling_radius,
            tangent_modulus_tension=Et,
            tangent_modulus_compression=Ec,
            poisson_ratio=nu,
        )
        margin_429 = (
            0.9 * critical_429
            / np.maximum(longitudinal_thermal * buckling_fos, 1e-9)
        )
        external_buckling = rib_supported_liner_buckling_profile(
            pressure_differential=p_diff,
            inner_radius=r_inner,
            wall_thickness=t_hot,
            unsupported_span=np.full(n, w),
            elastic_modulus=E,
            poisson_ratio=nu,
            knockdown=buckling_plate_knockdown,
        )
        external_margin = (
            np.asarray(external_buckling["margin_profile"])
            / buckling_fos
        )

        p_coolant = np.asarray(res["coolant_pressure"], dtype=float)
        x_jacket_inner, R_jacket = normal_offset_contour(
            x, r_inner, t_hot + h_arr
        )
        t_jacket = np.maximum(
            p_coolant * R_jacket * jacket_fos / max(Sy_j, 1e-9),
            t_jacket_min,
        )
        jacket_stress = p_coolant * R_jacket / np.maximum(t_jacket, 1e-12)
        jacket_margin = Sy_j / np.maximum(jacket_stress, 1e-9)
        profile = _profile(t_hot, h_arr, t_jacket)

        x_liner_mid, r_liner_mid = normal_offset_contour(
            x, r_inner, 0.5 * t_hot
        )
        _, ds_liner = helix_passage_lengths(x_liner_mid, r_liner_mid)
        x_jacket_mid, r_jacket_mid = normal_offset_contour(
            x_jacket_inner, R_jacket, 0.5 * t_jacket
        )
        _, ds_jacket = helix_passage_lengths(x_jacket_mid, r_jacket_mid)
        liner_mass = float(
            rho * np.sum(2.0 * math.pi * r_liner_mid * t_hot * ds_liner)
        )
        jacket_mass = float(
            rho_j * np.sum(
                2.0 * math.pi * r_jacket_mid * t_jacket * ds_jacket
            )
        )
        dp_bar = float(res["coolant_pressure_drop"] / 1e5)
        fits = bool(profile.channels_fit()["fits"])
        feasible = bool(
            np.all(thermal_ok)
            and (
                not gate_coolant_chemistry
                or res["coolant_chemistry_feasible"]
            )
            and res.get("boiling_chf_feasible", True)
            and np.all(struct_margin >= structural_fos - 1e-3)
            and np.all(jacket_margin >= jacket_fos - 1e-3)
            and (
                not gate_sp125_429
                or np.all(margin_429 >= 1.0 - 1e-3)
            )
            and np.all(external_margin >= 1.0 - 1e-3)
            and dp_bar <= dp_budget_bar
            and fits
        )
        return {
            "profile": profile,
            "t_hot": t_hot,
            "t_jacket": t_jacket,
            "h": h_arr,
            "thermal_ok": thermal_ok,
            "structural_margin_profile": struct_margin,
            "liner_pressure_hoop_radius_profile": liner_hoop_radius,
            "jacket_stress_profile": jacket_stress,
            "jacket_margin_profile": jacket_margin,
            "sp125_429_critical_stress_profile": critical_429,
            "sp125_429_longitudinal_stress_profile": longitudinal_thermal,
            "sp125_429_margin_profile": margin_429,
            "sp125_429_tangent_modulus_profile": Et,
            "buckling_tangent_data": tangent,
            "external_buckling": external_buckling,
            "external_buckling_margin_profile": external_margin,
            "liner_mass_kg": liner_mass,
            "jacket_mass_kg": jacket_mass,
            "mass_kg": liner_mass + jacket_mass,
            "pressure_drop_bar": dp_bar,
            "peak_wall_T": float(np.max(Twg)),
            "coolant_chemistry_margin": float(
                res["coolant_chemistry_margin"]
            ),
            "cooling": res,
            "feasible": feasible,
            "fits": fits,
        }

    height_candidates = []
    for h_throat in np.linspace(h_min, h_max, max(n_channel_height, 1)):
        for relief in np.linspace(
            height_relief_min, height_relief_max, max(n_height_relief, 1)
        ):
            h_arr = np.clip(
                h_throat * (
                    1.0 + (float(relief) - 1.0)
                    * np.sqrt(np.maximum(1.0 - heat_shape, 0.0))
                ),
                h_min,
                h_max,
            )
            height_candidates.append(_evaluate_height_profile(h_arr))

    feasible_candidates = [c for c in height_candidates if c["feasible"]]
    if feasible_candidates:
        best = min(
            feasible_candidates,
            key=lambda c: c["mass_kg"] + 1e-3 * c["pressure_drop_bar"],
        )
    else:
        def _score(c):
            margins = [
                float(np.min(c["structural_margin_profile"])) / structural_fos,
                float(np.min(c["jacket_margin_profile"])) / jacket_fos,
                float(np.min(c["external_buckling_margin_profile"])),
                T_limit / max(c["peak_wall_T"], 1e-9),
                dp_budget_bar / max(c["pressure_drop_bar"], 1e-9),
            ]
            if gate_coolant_chemistry:
                margins.append(c["coolant_chemistry_margin"])
            if gate_sp125_429:
                margins.append(float(np.min(c["sp125_429_margin_profile"])))
            return min(margins) - (10.0 if not c["fits"] else 0.0)
        best = max(height_candidates, key=_score)

    profile = best["profile"]
    t_hot = best["t_hot"]
    t_jacket = best["t_jacket"]
    h_arr = best["h"]
    res = best["cooling"]
    return {
        "profile": profile,
        "material": getattr(mat, "name", "?"),
        "jacket_material": getattr(jmat, "name", "?"),
        "t_hot": t_hot, "t_jacket": t_jacket,
        "channel_height": h_arr,
        "t_hot_throat_mm": float(t_hot[throat_idx] * 1e3),
        "t_hot_min_mm": float(np.min(t_hot) * 1e3),
        "t_hot_max_mm": float(np.max(t_hot) * 1e3),
        "t_jacket_min_mm": float(np.min(t_jacket) * 1e3),
        "t_jacket_max_mm": float(np.max(t_jacket) * 1e3),
        "channel_height_throat_mm": float(h_arr[throat_idx] * 1e3),
        "channel_height_min_mm": float(np.min(h_arr) * 1e3),
        "channel_height_max_mm": float(np.max(h_arr) * 1e3),
        "thermal_ok": best["thermal_ok"],
        "thermal_feasible": bool(np.all(best["thermal_ok"])),
        "structural_margin_profile": best["structural_margin_profile"],
        "min_structural_margin": float(
            np.min(best["structural_margin_profile"])
        ),
        "liner_pressure_hoop_radius_profile":
            best["liner_pressure_hoop_radius_profile"],
        "liner_pressure_hoop_radius_basis":
            "channel_half_width_sp125_eq_4_27_4_31",
        "jacket_stress_profile": best["jacket_stress_profile"],
        "jacket_margin_profile": best["jacket_margin_profile"],
        "min_jacket_margin": float(np.min(best["jacket_margin_profile"])),
        "sp125_429_margin_profile": best["sp125_429_margin_profile"],
        "min_sp125_429_margin": float(
            np.min(best["sp125_429_margin_profile"])
        ),
        "external_buckling_margin_profile":
            best["external_buckling_margin_profile"],
        "min_external_buckling_margin": float(
            np.min(best["external_buckling_margin_profile"])
        ),
        "buckling_tangent_modulus_fraction":
            float(buckling_tangent_modulus_fraction),
        "buckling_data_status": best["buckling_tangent_data"]["status"],
        "buckling_tangent_modulus_profile":
            best["sp125_429_tangent_modulus_profile"],
        "buckling_tangent_modulus_source":
            best["buckling_tangent_data"].get("source"),
        "sp125_429_geometry_status":
            "equivalent_tube_radius_half_channel_height",
        "sp125_429_temperature_status":
            "surface_temperature_drop_used_as_zone_mean_proxy",
        "sp125_429_gates_feasibility": bool(gate_sp125_429),
        "external_buckling_gates_feasibility": True,
        "peak_wall_T": best["peak_wall_T"],
        "cooling_margin": float(res["cooling_margin"]),
        "coolant_chemistry_margin": best["coolant_chemistry_margin"],
        "coolant_chemistry_gates_feasibility":
            bool(gate_coolant_chemistry),
        "pressure_drop_bar": best["pressure_drop_bar"],
        "liner_mass_kg": best["liner_mass_kg"],
        "jacket_mass_kg": best["jacket_mass_kg"],
        "mass_kg": best["mass_kg"],
        "mdot_cool": mdot_cool, "mdot_total": mdot_total,
        "feasible": bool(best["feasible"]),
        "requirement": {
            "thermal_margin": thermal_margin,
            "structural_fos": structural_fos,
            "jacket_fos": jacket_fos,
            "buckling_fos": buckling_fos,
            "gate_sp125_429": bool(gate_sp125_429),
            "dp_budget_bar": dp_budget_bar,
            "t_hot_min": t_hot_min,
            "t_hot_max": t_hot_max,
            "channel_height_min": h_min,
            "channel_height_max": h_max,
            "channel_roughness": channel_roughness,
            "gate_coolant_chemistry": bool(gate_coolant_chemistry),
            "curvature_correction": bool(curvature_correction),
        },
        "height_candidates": height_candidates,
        "cooling": res,
        "model": "variable_wall_and_channel_height_profile",
    }
