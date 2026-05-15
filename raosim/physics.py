"""
physics.py - Screening physics for validated preliminary nozzle design.

The models in this module are intentionally conservative early-design checks.
They are suitable for design review triage, not hardware qualification.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
    mach_from_area_ratio,
)
from raosim.separation import check_separation


def boundary_layer_displacement(
    contour: dict,
    Pc: float,
    prop: Any,
    *,
    wall_temperature: float | None = None,
) -> dict:
    """Estimate turbulent boundary-layer displacement and effective area ratio."""
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    gamma = float(prop.gamma)
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    x0 = float(x[throat_idx])
    mu_ref = _gas_viscosity(float(prop.Tc))
    R = float(prop.R_gas)
    wall_temperature = wall_temperature or 900.0

    delta_star = np.zeros_like(x)
    mach = np.zeros_like(x)
    for i in range(throat_idx, len(x)):
        area_ratio = max((float(y[i]) / Rt) ** 2, 1.0 + 1e-8)
        try:
            mach[i] = mach_from_area_ratio(area_ratio, gamma, supersonic=True)
        except Exception:
            mach[i] = 1.0
        T = float(prop.Tc) * isentropic_temperature_ratio(float(mach[i]), gamma)
        p = Pc * isentropic_pressure_ratio(float(mach[i]), gamma)
        rho = p / max(R * T, 1e-30)
        V = float(mach[i]) * math.sqrt(max(gamma * R * T, 0.0))
        s = max(float(x[i] - x0), 1e-6)
        Re_x = max(rho * V * s / max(mu_ref, 1e-12), 1.0)
        compressibility = math.sqrt(max(T / wall_temperature, 0.2))
        delta_star[i] = 0.046 * s / (Re_x ** 0.2) * compressibility

    effective_radius = np.maximum(y - delta_star, 0.05 * Rt)
    effective_epsilon = float((effective_radius[-1] / Rt) ** 2)
    return {
        "x": x,
        "delta_star": delta_star,
        "max_delta_star": float(np.max(delta_star)),
        "exit_delta_star": float(delta_star[-1]),
        "effective_exit_radius": float(effective_radius[-1]),
        "effective_epsilon": effective_epsilon,
        "epsilon_loss_fraction": float(
            max(0.0, (float(contour["epsilon"]) - effective_epsilon)
                / max(float(contour["epsilon"]), 1e-30))
        ),
        "mach": mach,
        "model": "turbulent_flat_plate_displacement_screening",
    }


def bartz_heat_flux(
    contour: dict,
    Pc: float,
    prop: Any,
    *,
    wall_temperature: float = 900.0,
) -> dict:
    """Return a Bartz-style heat-flux screening distribution."""
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    gamma = float(prop.gamma)
    Dt = 2.0 * Rt
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    c_star = max(float(prop.c_star), 1.0)
    Tc = float(prop.Tc)
    recovery = 0.9
    Taw = Tc * (1.0 + recovery * 0.5 * (gamma - 1.0)) / (1.0 + 0.5 * (gamma - 1.0))
    delta_t = max(Taw - wall_temperature, 1.0)

    # Screening correlation scaled to produce rocket-nozzle order-of-magnitude
    # heat fluxes while preserving Bartz sensitivities to pressure and throat size.
    throat_q = (
        0.026
        * (Pc / c_star) ** 0.8
        * (1.0 / max(Dt, 1e-9)) ** 0.2
        * (Tc / 3000.0) ** 0.68
        * (delta_t / 2500.0)
        * 1.0e3
    )
    area_scale = np.power(np.maximum(Rt / np.maximum(y, 1e-12), 0.05), 0.35)
    axial_scale = 1.0 / (1.0 + 2.0 * np.maximum(x - x[throat_idx], 0.0) / max(Rt, 1e-9))
    q = throat_q * np.maximum(area_scale, axial_scale)
    q[:throat_idx] = np.linspace(0.35 * throat_q, throat_q, throat_idx) if throat_idx else q[:throat_idx]
    peak_idx = int(np.argmax(q))
    return {
        "x": x,
        "q": q,
        "q_max": float(q[peak_idx]),
        "x_q_max": float(x[peak_idx]),
        "throat_q": float(q[throat_idx]),
        "adiabatic_wall_temperature": float(Taw),
        "model": "bartz_style_screening",
    }


def regenerative_cooling_screen(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
) -> dict:
    """Screen rectangular regenerative cooling channels."""
    method = getattr(cooling, "method", "none")
    coolant_inlet = float(getattr(cooling, "coolant_inlet_temperature", 293.0))
    if method != "regenerative":
        Taw = float(heat_flux["adiabatic_wall_temperature"])
        return {
            "method": method,
            "estimated_wall_temperature": Taw,
            "coolant_outlet_temperature": None,
            "channel_flow_area": 0.0,
            "cooling_margin": 0.0,
            "warnings": ["No active cooling model selected."],
        }

    channel_count = int(getattr(cooling, "channel_count", 0) or 0)
    channel_width = float(getattr(cooling, "channel_width", 0.0) or 0.0)
    channel_height = float(getattr(cooling, "channel_height", 0.0) or 0.0)
    coolant_mdot = float(getattr(cooling, "coolant_mass_flow", 0.0) or 0.0)
    coolant_cp = float(getattr(cooling, "coolant_cp", 3500.0) or 3500.0)
    flow_area = max(channel_count * channel_width * channel_height, 0.0)

    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    q = np.asarray(heat_flux["q"], dtype=float)
    area_weight = 2.0 * math.pi * y
    total_heat = float(np.trapezoid(q * area_weight, x))
    heat_capacity = max(coolant_mdot * coolant_cp, 1e-9)
    coolant_rise = total_heat / heat_capacity if coolant_mdot > 0 else float("inf")
    coolant_out = coolant_inlet + coolant_rise

    k_wall = max(float(getattr(material, "conductivity", 15.0) or 15.0), 1e-9)
    thickness = max(float(wall_thickness or 0.0), 1e-9)
    h_cool = 1000.0 + 2.0e8 * flow_area
    wall_drop = float(heat_flux["q_max"]) * (thickness / k_wall + 1.0 / h_cool)
    wall_temp = coolant_inlet + coolant_rise + wall_drop
    max_wall = float(getattr(cooling, "max_wall_temperature", 950.0) or 950.0)
    margin = max_wall / max(wall_temp, 1e-9)

    warnings: list[str] = []
    if channel_count <= 0 or channel_width <= 0.0 or channel_height <= 0.0:
        warnings.append("Regenerative cooling requires positive channel geometry.")
    if coolant_mdot <= 0.0:
        warnings.append("Regenerative cooling requires positive coolant mass flow.")

    return {
        "method": method,
        "estimated_wall_temperature": float(wall_temp),
        "coolant_outlet_temperature": float(coolant_out) if math.isfinite(coolant_out) else None,
        "coolant_temperature_rise": float(coolant_rise) if math.isfinite(coolant_rise) else None,
        "channel_flow_area": float(flow_area),
        "total_heat_load": total_heat,
        "cooling_margin": float(margin),
        "warnings": warnings,
    }


def structural_screen(
    contour: dict,
    Pc: float,
    Pa: float,
    prop: Any,
    material: Any,
    wall_thickness: float | None,
    thermal: dict,
    cooling: dict,
) -> dict:
    """Evaluate simple thermal and hoop-stress screening margins."""
    y = np.asarray(contour["y"], dtype=float)
    radius = float(np.max(y))
    thickness = float(wall_thickness or 0.0)
    if thickness > 0:
        hoop_stress = Pc * radius / thickness
    else:
        hoop_stress = float("inf")
    yield_strength = float(getattr(material, "yield_strength", 0.0) or 0.0)
    max_material_temp = float(getattr(material, "max_temperature", 0.0) or 0.0)
    max_heat_flux = float(getattr(material, "max_heat_flux", 0.0) or 0.0)
    wall_temp = float(cooling.get("estimated_wall_temperature") or thermal["adiabatic_wall_temperature"])

    try:
        sep = check_separation(contour, Pc, Pa, prop.gamma)
        sep_margin = float(sep["margin"])
    except Exception:
        sep_margin = 0.0

    stress_margin = yield_strength / max(hoop_stress, 1e-9) if yield_strength > 0 else 0.0
    temperature_margin = max_material_temp / max(wall_temp, 1e-9) if max_material_temp > 0 else 0.0
    heat_flux_margin = max_heat_flux / max(float(thermal["q_max"]), 1e-9) if max_heat_flux > 0 else 0.0

    return {
        "hoop_stress": float(hoop_stress),
        "stress_margin": float(stress_margin),
        "temperature_margin": float(temperature_margin),
        "heat_flux_margin": float(heat_flux_margin),
        "separation_margin": sep_margin,
        "wall_temperature": wall_temp,
        "passed": bool(
            stress_margin >= 1.5
            and temperature_margin >= 1.0
            and heat_flux_margin >= 1.0
            and sep_margin >= 1.0
            and cooling.get("cooling_margin", 0.0) >= (1.0 if cooling.get("method") == "regenerative" else 0.0)
        ),
        "model": "thin_wall_thermal_structural_screening",
    }


def _gas_viscosity(T: float) -> float:
    """Sutherland-like gas viscosity estimate for hot combustion products."""
    mu0 = 1.716e-5
    T0 = 273.15
    S = 110.4
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
