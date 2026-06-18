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
from raosim.regen_profile import helix_passage_lengths
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


def prandtl_number_estimate(gamma: float) -> float:
    """Prandtl number of combustion products, Pr = 4γ/(9γ−5).

    The standard Eucken-relation estimate used in the Bartz workflow
    (Huzel & Huang, *Modern Engineering for Design of Liquid-Propellant
    Rocket Engines*, NASA SP-125, Eq. 4-18).  Supply a CEA/measured Pr
    instead when available.
    """
    return 4.0 * gamma / (9.0 * gamma - 5.0)


def combustion_gas_viscosity(T_kelvin: float, Mw_g_per_mol: float) -> float:
    """Dynamic viscosity μ [Pa·s] of combustion gas at temperature ``T``.

    The Huzel & Huang correlation (NASA SP-125, Eq. 4-23):

        μ = 46.6e-10 · M^0.5 · T^0.6     [lbm/(in·s), T in °R]

    converted to SI exactly (1 lbm/(in·s) = 17.8579 Pa·s; °R = 1.8 K):

        μ = 1.1929e-7 · M^0.5 · T_K^0.6  [Pa·s]

    with ``M`` the mean molecular weight in g/mol.  This is the
    viscosity Bartz pairs with his correlation; supply a CEA/measured μ
    (evaluated at chamber stagnation temperature) for better accuracy.
    """
    return 1.1929e-7 * math.sqrt(max(Mw_g_per_mol, 1e-9)) * (
        max(T_kelvin, 1.0) ** 0.6
    )


def gas_transport_properties(
    prop: Any,
    *,
    cp: float | None = None,
    Pr: float | None = None,
    mu: float | None = None,
) -> tuple[float, float, float]:
    """Return ``(cp, Pr, mu)`` for the Bartz correlation at chamber
    stagnation conditions, from the :class:`Propellant` (γ, R_gas, Mw,
    Tc) with documented estimates, each overridable when CEA/measured
    values are on hand.

    * ``cp = γ R_gas / (γ−1)``           [J/(kg·K)]  (perfect-gas)
    * ``Pr = 4γ/(9γ−5)``                  (Eucken; :func:`prandtl_number_estimate`)
    * ``μ``  Huzel–Huang at ``Tc``        [Pa·s]      (:func:`combustion_gas_viscosity`)
    """
    gamma = float(prop.gamma)
    R_gas = float(getattr(prop, "R_gas", 0.0)) or 0.0
    if cp is None:
        if R_gas <= 0.0:
            raise ValueError("propellant lacks R_gas; pass cp explicitly")
        cp = gamma * R_gas / (gamma - 1.0)
    if Pr is None:
        Pr = prandtl_number_estimate(gamma)
    if mu is None:
        # Propellant.Mw is kg/mol; the correlation wants g/mol.
        Mw_g = float(getattr(prop, "Mw", 0.0)) * 1000.0
        mu = combustion_gas_viscosity(float(prop.Tc), Mw_g)
    return float(cp), float(Pr), float(mu)


def bartz_sigma(
    mach,
    gamma: float,
    wall_temperature: float,
    Tc: float,
    *,
    omega: float = 0.6,
):
    """Bartz boundary-layer property-variation correction factor σ.

        σ = 1 / { [½ (T_wg/T_c)(1 + ½(γ−1)M²) + ½]^(0.8−ω/5)
                  · [1 + ½(γ−1)M²]^(ω/5) }

    (Bartz, *Jet Propulsion* 27(1), 1957; general viscosity-exponent ω
    form per Huzel & Huang.  ω ≈ 0.6 recovers the classic 0.68/0.12
    exponents.)  ``T_wg`` is the gas-side wall temperature, ``M`` the
    local Mach number.  Accepts scalars or arrays.
    """
    M = np.asarray(mach, dtype=float)
    f = 1.0 + 0.5 * (gamma - 1.0) * M * M
    tw_ratio = wall_temperature / max(Tc, 1e-9)
    base = 0.5 * tw_ratio * f + 0.5
    return 1.0 / (base ** (0.8 - omega / 5.0) * f ** (omega / 5.0))


def bartz_heat_transfer_coefficient(
    mach,
    area_throat_over_area,
    *,
    Dt: float,
    Pc: float,
    c_star: float,
    cp: float,
    Pr: float,
    mu: float,
    gamma: float,
    Tc: float,
    wall_temperature: float,
    throat_curvature_radius: float,
    omega: float = 0.6,
):
    """Gas-side convective heat-transfer coefficient h_g [W/(m²·K)] from
    the **full Bartz (1957) correlation**::

        h_g = (0.026 / Dt^0.2)
              · (μ^0.2 cp / Pr^0.6)          property group
              · (Pc / c*)^0.8                 chamber mass-flux term
              · (Dt / r_c)^0.1                throat-curvature term
              · (At/A)^0.9                    local area-ratio term
              · σ                             property-variation factor

    In SI the gravitational conversion g_c = 1, so the imperial
    ``(Pc·g/c*)`` reduces to ``(Pc/c*)``.  All inputs SI; the 0.026
    coefficient is the standard Colburn/Bartz constant and the formula
    is dimensionally consistent in any coherent unit system.  ``mach``
    and ``area_throat_over_area`` (= A_t/A_local ≤ 1) may be arrays.

    Refs: Bartz, *Jet Propulsion* 27(1) 1957; Huzel & Huang, NASA
    SP-125 §4.  See :func:`bartz_sigma` for σ and
    :func:`gas_transport_properties` for cp/Pr/μ.
    """
    At_over_A = np.asarray(area_throat_over_area, dtype=float)
    sigma = bartz_sigma(mach, gamma, wall_temperature, Tc, omega=omega)
    coeff = (
        0.026
        / max(Dt, 1e-12) ** 0.2
        * (mu ** 0.2 * cp / Pr ** 0.6)
        * (Pc / max(c_star, 1e-9)) ** 0.8
        * (Dt / max(throat_curvature_radius, 1e-12)) ** 0.1
    )
    return coeff * np.power(np.clip(At_over_A, 1e-9, 1.0), 0.9) * sigma


def bartz_heat_flux(
    contour: dict,
    Pc: float,
    prop: Any,
    *,
    wall_temperature: float = 900.0,
    cp: float | None = None,
    Pr: float | None = None,
    mu: float | None = None,
    throat_curvature_radius: float | None = None,
    omega: float = 0.6,
) -> dict:
    """Gas-side heat-flux distribution from the full Bartz (1957)
    correlation along a nozzle contour.

    For each station: the local Mach is recovered from the isentropic
    area ratio (subsonic upstream of the throat, supersonic downstream),
    h_g from :func:`bartz_heat_transfer_coefficient`, the adiabatic-wall
    (recovery) temperature with turbulent recovery factor r = Pr^(1/3),

        T_aw = Tc · (1 + r·½(γ−1)M²) / (1 + ½(γ−1)M²),

    and the heat flux q = h_g · (T_aw − T_wg).

    Gas transport properties default to the documented Bartz estimates
    (:func:`gas_transport_properties`); pass ``cp``/``Pr``/``mu`` to use
    CEA/measured values.  ``throat_curvature_radius`` defaults to the
    contour's downstream throat-arc radius ``Rd`` (Bartz's r_c; the
    (Dt/r_c)^0.1 term is weak, so the choice is second-order).

    Backward-compatible keys (``q``, ``q_max``, ``x_q_max``,
    ``throat_q``, ``adiabatic_wall_temperature``) are preserved for the
    cooling/structural screens; ``model`` is now ``"bartz_1957"``.
    """
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    gamma = float(prop.gamma)
    Dt = 2.0 * Rt
    Tc = float(prop.Tc)
    c_star = max(float(prop.c_star), 1.0)
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    rc = float(
        throat_curvature_radius
        if throat_curvature_radius is not None
        else contour.get("Rd", 0.382 * Rt)
    )
    cp_v, Pr_v, mu_v = gas_transport_properties(prop, cp=cp, Pr=Pr, mu=mu)

    # Local Mach from the area ratio (A/At = (y/Rt)^2), branch by side
    # of the throat.  At/A = (Rt/y)^2 is the Bartz area term (<= 1).
    mach = np.ones_like(x)
    for i in range(len(x)):
        area_ratio = max((float(y[i]) / Rt) ** 2, 1.0 + 1e-12)
        supersonic = i >= throat_idx
        try:
            mach[i] = mach_from_area_ratio(area_ratio, gamma,
                                           supersonic=supersonic)
        except Exception:
            mach[i] = 1.0
    At_over_A = np.clip((Rt / np.maximum(y, 1e-12)) ** 2, 1e-9, 1.0)

    h_g = bartz_heat_transfer_coefficient(
        mach, At_over_A, Dt=Dt, Pc=Pc, c_star=c_star, cp=cp_v, Pr=Pr_v,
        mu=mu_v, gamma=gamma, Tc=Tc, wall_temperature=wall_temperature,
        throat_curvature_radius=rc, omega=omega,
    )

    # Adiabatic-wall (recovery) temperature, turbulent r = Pr^(1/3).
    r_factor = Pr_v ** (1.0 / 3.0)
    f = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    Taw = Tc * (1.0 + r_factor * 0.5 * (gamma - 1.0) * mach * mach) / f
    q = h_g * np.maximum(Taw - wall_temperature, 0.0)

    peak_idx = int(np.argmax(q))
    return {
        "x": x,
        "q": q,
        "h_g": h_g,
        "mach": mach,
        "recovery_temperature": Taw,
        "q_max": float(q[peak_idx]),
        "x_q_max": float(x[peak_idx]),
        "throat_q": float(q[throat_idx]),
        "throat_h_g": float(h_g[throat_idx]),
        "adiabatic_wall_temperature": float(Taw[throat_idx]),
        "recovery_factor": float(r_factor),
        "gas_properties": {"cp": cp_v, "Pr": Pr_v, "mu": mu_v,
                           "throat_curvature_radius": rc},
        "wall_temperature": (
            np.asarray(wall_temperature, dtype=float)
            if np.ndim(wall_temperature) else float(wall_temperature)
        ),
        "units": {"q": "W/m^2", "h_g": "W/(m^2.K)", "T": "K"},
        "model": "bartz_1957",
        "reference": ("Bartz, Jet Propulsion 27(1) 1957; "
                      "Huzel & Huang NASA SP-125 sec. 4"),
    }


# --------------------------------------------------------------------------- #
# Regenerative cooling — Sieder-Tate coolant side + 1-D wall conduction        #
# --------------------------------------------------------------------------- #
#
# Reference values at ~300 K (liquid), for screening when CEA/measured
# coolant properties are not supplied.  ``andrade_B`` is the Andrade
# liquid-viscosity activation parameter for μ(T) = μ_ref·exp(B·(1/T −
# 1/T_ref)) (the Sieder-Tate (μ_b/μ_w)^0.14 term needs μ at both bulk
# and wall temperature).  Cryogens use a milder dependence.
COOLANT_PROPERTIES: dict[str, dict[str, float]] = {
    # name:       rho[kg/m3] mu_ref[Pa.s] T_ref[K]  andrade_B[K]  k[W/mK] cp[J/kgK]
    "rp1":      {"rho": 810.0, "mu_ref": 1.6e-3, "T_ref": 300.0, "andrade_B": 1200.0, "k": 0.12, "cp": 2010.0},
    "kerosene": {"rho": 810.0, "mu_ref": 1.6e-3, "T_ref": 300.0, "andrade_B": 1200.0, "k": 0.12, "cp": 2010.0},
    "ethanol":  {"rho": 789.0, "mu_ref": 1.1e-3, "T_ref": 300.0, "andrade_B": 1600.0, "k": 0.17, "cp": 2440.0},
    "methane":  {"rho": 423.0, "mu_ref": 1.1e-4, "T_ref": 110.0, "andrade_B": 400.0,  "k": 0.19, "cp": 3450.0},
    "water":    {"rho": 1000.0, "mu_ref": 8.9e-4, "T_ref": 300.0, "andrade_B": 1900.0, "k": 0.61, "cp": 4186.0},
    "lh2":      {"rho": 71.0,  "mu_ref": 1.3e-5, "T_ref": 25.0,  "andrade_B": 60.0,   "k": 0.10, "cp": 9800.0},
    "hydrogen": {"rho": 71.0,  "mu_ref": 1.3e-5, "T_ref": 25.0,  "andrade_B": 60.0,   "k": 0.10, "cp": 9800.0},
}
_DEFAULT_COOLANT = "rp1"


def resolve_coolant_properties(cooling: Any) -> dict[str, float]:
    """Coolant ρ, μ(T) parameters, k, cp from a ``CoolingSpec``.

    Looks up :data:`COOLANT_PROPERTIES` by ``cooling.coolant`` name
    (case-insensitive; defaults to RP-1) and overlays any explicit
    ``coolant_density`` / ``coolant_viscosity`` / ``coolant_conductivity``
    / ``coolant_cp`` fields the spec carries.  Supply CEA/measured
    values via those fields for accuracy.
    """
    name = str(getattr(cooling, "coolant", None) or _DEFAULT_COOLANT).lower()
    props = dict(COOLANT_PROPERTIES.get(name, COOLANT_PROPERTIES[_DEFAULT_COOLANT]))
    rho = getattr(cooling, "coolant_density", None)
    mu = getattr(cooling, "coolant_viscosity", None)
    k = getattr(cooling, "coolant_conductivity", None)
    cp = getattr(cooling, "coolant_cp", None)
    if rho:
        props["rho"] = float(rho)
    if mu:                       # an explicit μ disables the Andrade T-model
        props["mu_ref"] = float(mu)
        props["andrade_B"] = 0.0
    if k:
        props["k"] = float(k)
    if cp:
        props["cp"] = float(cp)
    return props


def coolant_viscosity(props: dict[str, float], T: float) -> float:
    """Andrade liquid viscosity μ(T) = μ_ref·exp(B·(1/T − 1/T_ref)) [Pa·s]."""
    B = float(props.get("andrade_B", 0.0))
    if B <= 0.0:
        return float(props["mu_ref"])
    T_ref = float(props["T_ref"])
    return float(props["mu_ref"]) * math.exp(
        B * (1.0 / max(T, 1.0) - 1.0 / T_ref)
    )


def hydraulic_diameter(width: float, height: float) -> float:
    """Hydraulic diameter of a rectangular channel: D_h = 2wh/(w+h)."""
    w = max(float(width), 1e-12)
    h = max(float(height), 1e-12)
    return 2.0 * w * h / (w + h)


def sieder_tate_coefficient(
    mass_flux, D_h: float, props: dict[str, float],
    *, mu_bulk, mu_wall,
):
    """Coolant-side film coefficient h_c [W/(m²·K)] from the Sieder-Tate
    turbulent forced-convection correlation::

        Nu = h_c D_h / k = 0.027 · Re^0.8 · Pr^(1/3) · (μ_b/μ_w)^0.14

    Re = G·D_h/μ_b (G = coolant mass flux in the channel), Pr =
    μ_b·cp/k.  The (μ_b/μ_w)^0.14 term is the Sieder-Tate property
    correction for the bulk↔wall viscosity difference across the heated
    film (Sieder & Tate, *Ind. Eng. Chem.* 28, 1936; standard regen
    practice, Huzel & Huang NASA SP-125 §4).  Accepts arrays.
    """
    G = np.asarray(mass_flux, dtype=float)
    k = float(props["k"])
    cp = float(props["cp"])
    mu_b = np.asarray(mu_bulk, dtype=float)
    mu_w = np.asarray(mu_wall, dtype=float)
    Re = G * D_h / np.maximum(mu_b, 1e-12)
    Pr = mu_b * cp / max(k, 1e-12)
    Nu = (0.027 * np.power(np.maximum(Re, 1.0), 0.8)
          * np.power(np.maximum(Pr, 1e-6), 1.0 / 3.0)
          * np.power(mu_b / np.maximum(mu_w, 1e-12), 0.14))
    return Nu * k / np.maximum(D_h, 1e-12)


def fin_efficiency(h_c, k_wall: float, land_width, channel_height) -> Any:
    """Fin (channel-land) efficiency η_f = tanh(mH)/(mH), m =
    √(2 h_c / (k_wall · t_land)).

    The lands (ribs) between coolant channels conduct heat from the
    inner wall down their height ``H`` into the coolant on both sides —
    a fin with coolant on two faces of thickness ``t_land``.  η_f → 1
    for short, thick, high-conductivity fins; < 1 as they get tall/thin
    (the tip runs hotter than the base).  Standard regen channel
    practice (Huzel & Huang NASA SP-125 §4).
    """
    import numpy as _np
    t_land = _np.maximum(_np.asarray(land_width, dtype=float), 1e-9)
    H = _np.asarray(channel_height, dtype=float)
    m = _np.sqrt(2.0 * _np.asarray(h_c, dtype=float) / (max(k_wall, 1e-9) * t_land))
    mH = _np.maximum(m * H, 1e-12)
    return _np.tanh(mH) / mH


def curvature_correction_factor(Re, D_h: float, R_c_signed) -> Any:
    """Dean-number coolant-side curvature correction ``C`` on the
    Nusselt number / h_c::

        C = (Re · (D_h / (2 |R_c|))²)^(±0.05)

    ``+`` exponent where the channel wall is **concave** toward the gas
    (the throat region — secondary/Dean vortices enhance the
    coolant-side transfer on the heated wall), ``−`` where **convex**.
    The sign is taken from ``R_c_signed`` (signed meridional radius of
    curvature: positive = concave on the gas side).  Mild correction
    (exponent ±0.05).  Curved-channel regen heat-transfer practice
    (Niino et al.; see the EUCASS / curved-passage literature).
    """
    import numpy as _np
    Re = _np.maximum(_np.asarray(Re, dtype=float), 1.0)
    Rc = _np.asarray(R_c_signed, dtype=float)
    sign = _np.sign(Rc)
    sign = _np.where(sign == 0.0, 1.0, sign)
    arg = _np.maximum(Re * (D_h / (2.0 * _np.maximum(_np.abs(Rc), 1e-9))) ** 2, 1e-12)
    return arg ** (0.05 * sign)


def darcy_friction_factor(Re) -> Any:
    """Blasius turbulent Darcy friction factor f = 0.316 Re^(−0.25)
    (smooth channel; the partner of the Sieder-Tate/Colburn family).
    Laminar fallback 64/Re below Re = 2300."""
    import numpy as _np
    Re = _np.maximum(_np.asarray(Re, dtype=float), 1.0)
    turb = 0.316 * Re ** (-0.25)
    return _np.where(Re < 2300.0, 64.0 / Re, turb)


def coolant_pressure_profile(
    pressure_loss_per_station,
    flow_order,
    *,
    outlet_pressure: float,
) -> np.ndarray:
    """Recover the absolute coolant pressure at each wall station.

    ``pressure_loss_per_station`` contains the Darcy-Weisbach loss assigned
    to each station and ``flow_order`` lists station indices from coolant
    inlet to coolant outlet.  Pressure decreases along that order.  The
    supplied ``outlet_pressure`` is the jacket pressure immediately before
    the coolant enters the injector/manifold downstream of the cooling
    passages.

    Station values are cell-centred: half of the local loss lies upstream
    and half downstream.  This gives the pressure required by the SP-125
    liner load ``p_coolant(x) - p_gas(x)`` without inventing a fixed
    fraction of chamber pressure.
    """
    losses = np.maximum(np.asarray(pressure_loss_per_station, dtype=float), 0.0)
    order = np.asarray(flow_order, dtype=int)
    if losses.ndim != 1 or order.ndim != 1 or len(losses) != len(order):
        raise ValueError("loss and flow-order arrays must be one-dimensional and equal length")
    loss_flow = losses[order]
    remaining = np.cumsum(loss_flow[::-1])[::-1]
    pressure_flow = float(outlet_pressure) + remaining - 0.5 * loss_flow
    pressure = np.empty_like(pressure_flow)
    pressure[order] = pressure_flow
    return pressure


def _wall_radius_of_curvature_signed(x, y):
    """Signed meridional radius of curvature R_c of the wall polyline.

    Positive where the wall is concave toward the gas (the throat
    region), negative where convex, large where nearly straight.  Uses
    the PARAMETRIC curvature κ = (x'y'' − y'x'')/(x'²+y'²)^1.5 with
    derivatives w.r.t. the node index, so it is robust to duplicate or
    non-monotone x (the convergent/throat segment has repeated x)."""
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    y = _np.asarray(y, dtype=float)
    xp = _np.gradient(x)
    yp = _np.gradient(y)
    xpp = _np.gradient(xp)
    ypp = _np.gradient(yp)
    denom = _np.maximum((xp * xp + yp * yp) ** 1.5, 1e-30)
    kappa = (xp * ypp - yp * xpp) / denom
    radius = _np.sign(kappa + 1e-30) * 1e9
    _np.divide(1.0, kappa, out=radius, where=_np.abs(kappa) > 1e-9)
    return radius


def regenerative_cooling_analysis(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
    prop: Any,
    Pc: float,
    *,
    flow_from: str = "exit",
    n_iter: int = 30,
    omega: float = 0.6,
    fin_correction: bool = True,
    curvature_correction: bool = True,
    pressure_drop: bool = True,
    helix_turns: float = 0.0,
    wall_profile: Any = None,
    coolant_outlet_pressure: float | None = None,
    injector_pressure_drop: float = 0.0,
) -> dict:
    """Coupled 1-D regenerative-cooling solve along the contour.

    The steady heat flux passes in series through the gas film, the
    wall, and the coolant film (Huzel & Huang NASA SP-125 §4):

        q = (T_aw − T_c) / (1/h_g + t_w/k_w + 1/h_c)
        T_wg = T_aw − q/h_g          (gas-side wall temperature)
        T_wc = T_c  + q/h_c          (coolant-side wall temperature)

    h_g is the **full Bartz** gas-side coefficient (re-evaluated at the
    local T_wg each pass, so the σ property factor is self-consistent);
    h_c is **Sieder-Tate** in the channels; the coolant bulk temperature
    T_c is marched along the flow path from the inlet
    (``flow_from='exit'`` = counter-current to the gas, the common
    single-pass layout; ``'throat'`` co-current).  Iterated to a fixed
    point.

    Returns per-station arrays (``q``, ``T_wg``, ``T_wc``,
    ``T_coolant``, ``h_c``) plus the screen-compatible summary keys.
    """
    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    throat_idx = int(np.argmin(np.abs(y - Rt)))

    n_st = len(x)
    mdot = float(getattr(cooling, "coolant_mass_flow", 0.0) or 0.0)
    inlet_T = float(getattr(cooling, "coolant_inlet_temperature", 293.0))
    k_wall = max(float(getattr(material, "conductivity", 15.0) or 15.0), 1e-9)
    cprops = resolve_coolant_properties(cooling)
    cp_cool = float(cprops["cp"])

    # Channel geometry as per-station arrays.  A RegenWallProfile can vary
    # t_hot/w/h along the contour (SP-125 passage-area tapering); scalars
    # broadcast to a constant array, so the uniform case is unchanged.
    land_override = None
    if wall_profile is not None:
        w_arr = np.asarray(wall_profile.channel_width, dtype=float)
        h_arr = np.asarray(wall_profile.channel_height, dtype=float)
        t_arr = np.maximum(np.asarray(wall_profile.t_hot, dtype=float), 1e-9)
        N = int(wall_profile.channel_count)
        helix_turns = float(wall_profile.helix_turns)
        land_override = np.asarray(wall_profile.land_width, dtype=float)
    else:
        N = int(getattr(cooling, "channel_count", 0) or 0)
        w_arr = np.full(n_st, float(getattr(cooling, "channel_width", 0.0) or 0.0))
        h_arr = np.full(n_st, float(getattr(cooling, "channel_height", 0.0) or 0.0))
        t_arr = np.full(n_st, max(float(wall_thickness or 0.0), 1e-9))
    w = float(np.mean(w_arr)); h = float(np.mean(h_arr))   # representative (msgs)
    t_wall = float(np.mean(t_arr))

    warnings: list[str] = []
    if N <= 0 or np.any(w_arr <= 0.0) or np.any(h_arr <= 0.0):
        warnings.append("Regenerative cooling requires positive channel geometry.")
    if mdot <= 0.0:
        warnings.append("Regenerative cooling requires positive coolant mass flow.")

    A_ch = np.maximum(w_arr * h_arr, 1e-12)             # per-channel area [m²]
    D_h = 2.0 * w_arr * h_arr / np.maximum(w_arr + h_arr, 1e-12)
    mdot_per_channel = mdot / max(N, 1)
    G = mdot_per_channel / A_ch                 # channel mass flux [kg/m²s]
    rho_cool = float(cprops["rho"])
    V_cool = G / max(rho_cool, 1e-9)            # channel velocity [m/s]

    # Level-1 channel-cross-section geometry (per station): the channel
    # pitch wraps the local circumference, so the land (rib) width and
    # the fin enhancement vary along the nozzle; the wall curvature sets
    # the Dean correction.
    pitch = 2.0 * math.pi * np.maximum(y, 1e-9) / max(N, 1)
    land_width_raw = land_override if land_override is not None else (pitch - w_arr)
    land_width = np.maximum(land_width_raw, 1e-5 * Rt)
    # Channels must physically fit the circumference: N·(w+land) ≤ 2πr.
    tight = int(np.argmin(land_width_raw))
    if float(np.min(land_width_raw)) <= 0.0:
        warnings_geom = (
            f"Channels do not fit: {N} × {w_arr[tight]*1e3:.2f} mm exceeds the "
            f"circumference at r={float(y[tight])*1e3:.1f} mm "
            f"(pitch {float(pitch[tight])*1e3:.2f} mm); reduce "
            "channel count or width.  Fin model degraded there."
        )
    else:
        warnings_geom = None
    if warnings_geom:
        warnings.append(warnings_geom)
    R_c_signed = _wall_radius_of_curvature_signed(x, y)
    # Per-station coolant path length: meridional (ds_arc) and, when the
    # channels are wound helically (helix_turns > 0), the longer helical
    # path (ds_path) that sets the Darcy-Weisbach Δp (SP-125 eq. 4-32).
    ds_path, ds_arc = helix_passage_lengths(
        x, y, helix_turns=helix_turns, t_wall=t_arr, channel_height=h_arr)

    # Gas-side heated area per station (the heat the coolant removes).
    area_weight = 2.0 * math.pi * np.maximum(y, 1e-9)

    # Coolant march order: index sequence in the direction of coolant flow.
    if flow_from == "throat":
        order = np.arange(len(x))
    else:                                       # 'exit' (counter-current)
        order = np.arange(len(x))[::-1]
    inv = np.argsort(order)

    Taw = np.asarray(heat_flux["recovery_temperature"], dtype=float)
    T_wg = 0.6 * Taw                            # initial guess
    T_c = np.full_like(x, inlet_T)

    for _ in range(max(n_iter, 1)):
        # Gas side, self-consistent σ at the current wall temperature.
        hf = bartz_heat_flux(contour, Pc, prop, wall_temperature=T_wg,
                             omega=omega)
        h_g = np.asarray(hf["h_g"], dtype=float)
        Taw = np.asarray(hf["recovery_temperature"], dtype=float)

        # Coolant side (Sieder-Tate); wall-side coolant temp for μ_w.
        T_wc = T_c + (Taw - T_c) * 0.3          # provisional, refined below
        mu_b = np.array([coolant_viscosity(cprops, float(t)) for t in T_c])
        mu_w = np.array([coolant_viscosity(cprops, float(t)) for t in T_wc])
        h_c_straight = sieder_tate_coefficient(G, D_h, cprops,
                                               mu_bulk=mu_b, mu_wall=mu_w)
        # Level-1 corrections: Dean curvature, then fin (land) area.
        Re_cool = G * D_h / np.maximum(mu_b, 1e-12)
        c_curv = (curvature_correction_factor(Re_cool, D_h, R_c_signed)
                  if curvature_correction else np.ones_like(x))
        h_c_film = h_c_straight * c_curv
        if fin_correction:
            eta_f = fin_efficiency(h_c_film, k_wall, land_width, h_arr)
            fin_factor = (w_arr + 2.0 * eta_f * h_arr) / np.maximum(pitch, 1e-12)
        else:
            eta_f = np.ones_like(x)
            fin_factor = np.ones_like(x)
        # Effective coolant-side conductance referred to the gas-side
        # area (channel base + fin-efficient side walls).
        h_c = h_c_film * fin_factor

        # Series thermal circuit.
        R_tot = 1.0 / np.maximum(h_g, 1e-9) + t_arr / k_wall + 1.0 / np.maximum(h_c, 1e-9)
        q = np.maximum((Taw - T_c) / R_tot, 0.0)
        T_wg_new = Taw - q / np.maximum(h_g, 1e-9)
        T_wc = T_c + q / np.maximum(h_c, 1e-9)

        # March the coolant bulk temperature along the flow path.
        dQ = q * area_weight                    # per-unit-length heat pickup
        ds = np.abs(np.gradient(x))
        dT = (dQ * ds) / max(mdot * cp_cool, 1e-9)
        T_c_marched = inlet_T + np.cumsum(dT[order])[inv]

        # Relax for stability.
        T_wg = 0.5 * T_wg + 0.5 * T_wg_new
        T_c = 0.5 * T_c + 0.5 * T_c_marched

    coolant_out = float(T_c[order[-1]])
    peak_T_wg = float(np.max(T_wg))
    max_wall = float(getattr(cooling, "max_wall_temperature", 950.0) or 950.0)
    margin = max_wall / max(peak_T_wg, 1e-9)
    total_heat = float(np.trapezoid(q * area_weight, x))

    # Coolant pressure drop (Darcy-Weisbach, friction only) along the
    # channel: Δp = Σ f (ds/D_h) (ρ V²/2).
    if pressure_drop:
        Re_dp = G * D_h / np.maximum(mu_b, 1e-12)
        f_profile = np.asarray(darcy_friction_factor(Re_dp), dtype=float)
        dynamic_head = 0.5 * rho_cool * V_cool ** 2
        # Δp = Σ f (ds_path / D_h) (ρ V²/2): ds_path is the HELICAL length
        # when the channels are wound, so more turns ⇒ proportionally more Δp.
        dP = f_profile * (ds_path / np.maximum(D_h, 1e-12)) * dynamic_head
        dP_meridional_reference = (
            f_profile * (ds_arc / np.maximum(D_h, 1e-12)) * dynamic_head
        )
        total_dP = float(np.sum(dP))
        f_darcy = float(np.mean(f_profile))
    else:
        f_darcy = 0.0
        dP = np.zeros_like(x)
        dP_meridional_reference = np.zeros_like(x)
        total_dP = 0.0

    # Absolute coolant pressure needs one boundary condition.  If the user
    # has not supplied the jacket outlet pressure, use Pc + injector Δp:
    # this is the minimum pressure that lets the coolant subsequently enter
    # the chamber.  It is an explicit conservative-boundary assumption, not
    # the former hard-coded ``0.5 Pc`` liner differential.
    explicit_outlet = coolant_outlet_pressure
    if explicit_outlet is None:
        explicit_outlet = getattr(cooling, "coolant_outlet_pressure", None)
    if explicit_outlet is None:
        spec_injector_drop = getattr(cooling, "injector_pressure_drop", None)
        used_injector_drop = (
            float(spec_injector_drop)
            if spec_injector_drop is not None
            else float(injector_pressure_drop)
        )
        outlet_pressure = float(Pc) + max(used_injector_drop, 0.0)
        pressure_boundary_source = "minimum_injector_entry_pressure_Pc_plus_injector_drop"
    else:
        outlet_pressure = float(explicit_outlet)
        pressure_boundary_source = "user_supplied_coolant_outlet_pressure"
    p_coolant = coolant_pressure_profile(
        dP, order, outlet_pressure=outlet_pressure
    )
    p_gas = float(Pc) * np.asarray(
        [isentropic_pressure_ratio(float(M), float(prop.gamma))
         for M in np.asarray(hf["mach"], dtype=float)],
        dtype=float,
    )
    p_diff = p_coolant - p_gas

    return {
        "method": getattr(cooling, "method", "regenerative"),
        "x": x,
        "q": q,
        "h_c": h_c,
        "h_c_film": h_c_film,
        "fin_efficiency": eta_f,
        "fin_area_factor": fin_factor,
        "curvature_factor": c_curv,
        "land_width": land_width,
        "gas_side_wall_temperature": T_wg,
        "coolant_side_wall_temperature": T_wc,
        "coolant_temperature": T_c,
        "estimated_wall_temperature": peak_T_wg,
        "peak_gas_side_wall_temperature": peak_T_wg,
        "x_peak_wall_temperature": float(x[int(np.argmax(T_wg))]),
        "throat_wall_temperature": float(T_wg[throat_idx]),
        "coolant_outlet_temperature": coolant_out,
        "coolant_temperature_rise": float(coolant_out - inlet_T),
        "channel_flow_area": float(N * A_ch[throat_idx]),
        "channel_hydraulic_diameter": float(D_h[throat_idx]),
        "channel_mass_flux": float(G[throat_idx]),
        "channel_velocity": float(V_cool[throat_idx]),
        "channel_velocity_profile": V_cool,
        "hydraulic_diameter_profile": D_h,
        "coolant_pressure_drop": total_dP,
        "darcy_friction_factor": f_darcy,
        "pressure_loss_profile": dP,
        "coolant_pressure": p_coolant,
        "coolant_flow_order": order,
        "coolant_outlet_pressure": float(outlet_pressure),
        "coolant_inlet_pressure": float(p_coolant[order[0]] + 0.5 * dP[order[0]]),
        "coolant_pressure_boundary_source": pressure_boundary_source,
        "gas_pressure": p_gas,
        "liner_pressure_differential": p_diff,
        "helix_turns": float(helix_turns),
        "passage_length": float(np.sum(ds_path)),
        "passage_length_axial": float(np.sum(ds_arc)),
        "passage_length_factor": float(np.sum(ds_path) / max(np.sum(ds_arc), 1e-12)),
        # This is the exact multiplier for this solver's friction integral.
        # It differs slightly from the purely geometric length factor when
        # D_h, velocity, viscosity, or friction factor vary by station.
        "pressure_drop_path_factor": float(
            np.sum(dP) / max(np.sum(dP_meridional_reference), 1e-12)
        ) if pressure_drop else 1.0,
        "throat_h_c": float(h_c[throat_idx]),
        "throat_fin_efficiency": float(eta_f[throat_idx]) if fin_correction else 1.0,
        "total_heat_load": total_heat,
        "cooling_margin": float(margin),
        "coolant_properties": cprops,
        "model": "sieder_tate_1d_regen",
        "fidelity": "1d_finned",
        "corrections": {"fin": fin_correction,
                        "curvature": curvature_correction,
                        "pressure_drop": pressure_drop},
        "warnings": warnings,
    }


def _solve_wall_cross_section(
    *, t_w, H, w_half, land_half, k_wall, h_g, T_aw, h_c, T_c,
    n_s=12, n_xi=16,
):
    """Steady 2-D conduction on one symmetric half-pitch of the
    channel-land wall cross-section (s = circumferential, ξ = radial).

    Cell-centred finite volume.  The L-shaped solid = the inner wall
    (0 ≤ ξ ≤ t_w over the full half-pitch) plus the land/fin (s ≥
    w_half, up to ξ = t_w+H).  BCs: gas convection (h_g, T_aw) on the
    hot face ξ=0; coolant convection (h_c, T_c) on every solid face
    bordering the channel void; adiabatic on the symmetry planes (s=0
    channel centre, s=p/2 land centre), the fin tip, and the outer
    closeout.  Returns the (n_s, n_xi) temperature field (NaN in the
    void) — the land-centre gas-side node is the real hot spot.
    """
    s_edges = np.linspace(0.0, w_half + land_half, n_s + 1)
    xi_edges = np.linspace(0.0, t_w + H, n_xi + 1)
    ds = s_edges[1] - s_edges[0]
    dxi = xi_edges[1] - xi_edges[0]
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    xi_c = 0.5 * (xi_edges[:-1] + xi_edges[1:])

    # Solid mask: base slab (ξ ≤ t_w) OR land (s ≥ w_half).
    S, X = np.meshgrid(s_c, xi_c, indexing="ij")
    solid = (X <= t_w) | (S >= w_half)
    idx = -np.ones((n_s, n_xi), dtype=int)
    ids = np.argwhere(solid)
    for n, (i, j) in enumerate(ids):
        idx[i, j] = n
    M = len(ids)
    A = np.zeros((M, M))
    b = np.zeros(M)

    G_s = k_wall * dxi / ds          # conductance, s-direction face
    G_x = k_wall * ds / dxi          # conductance, ξ-direction face

    def robin(h, length, dist):
        return length / (1.0 / max(h, 1e-30) + dist / (2.0 * k_wall))

    for n, (i, j) in enumerate(ids):
        diag = 0.0
        # s neighbours
        for di in (-1, 1):
            ii = i + di
            if 0 <= ii < n_s and solid[ii, j]:
                A[n, idx[ii, j]] += G_s
                diag += G_s
            elif 0 <= ii < n_s and not solid[ii, j]:
                # face borders the coolant void (channel side wall).
                g = robin(h_c, dxi, ds)
                diag += g
                b[n] += g * T_c
            # else domain edge in s -> adiabatic (symmetry); no term.
        # ξ neighbours
        for dj in (-1, 1):
            jj = j + dj
            if 0 <= jj < n_xi and solid[i, jj]:
                A[n, idx[i, jj]] += G_x
                diag += G_x
            elif jj < 0:
                # gas-side hot face (ξ = 0).
                g = robin(h_g, ds, dxi)
                diag += g
                b[n] += g * T_aw
            elif 0 <= jj < n_xi and not solid[i, jj]:
                # channel base (coolant void above the inner wall).
                g = robin(h_c, ds, dxi)
                diag += g
                b[n] += g * T_c
            # jj == n_xi (outer / fin tip) -> adiabatic.
        A[n, n] = -diag

    T = np.linalg.solve(A, -b)
    field = np.full((n_s, n_xi), np.nan)
    for n, (i, j) in enumerate(ids):
        field[i, j] = T[n]
    return field, s_c, xi_c


def regenerative_cooling_2d(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
    prop: Any,
    Pc: float,
    *,
    n_s: int = 12,
    n_xi: int = 16,
    stations: int | None = 40,
    **kwargs,
) -> dict:
    """Level-2 quasi-2-D regen analysis: the Level-1 axial coupling for
    T_c(x)/h_g(x)/h_c(x), then a 2-D wall cross-section conduction solve
    (:func:`_solve_wall_cross_section`) at each station to resolve the
    **circumferential** temperature distribution — the land-midpoint hot
    spot the 1-D series circuit averages away.

    Returns the 1-D result's keys plus per-station 2-D peaks
    (``T_wg_land`` hot spot, ``T_wg_channel`` cool spot, their spread)
    and the governing ``peak_gas_side_wall_temperature`` REPLACED by the
    2-D land hot spot (the failure-relevant temperature).  ``stations``
    subsamples the axial grid for the 2-D solves (None = all).
    """
    base = regenerative_cooling_analysis(
        heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
        **kwargs)
    x = np.asarray(base["x"]); y = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    throat_idx = int(np.argmin(np.abs(y - Rt)))
    k_wall = max(float(getattr(material, "conductivity", 15.0) or 15.0), 1e-9)
    t_w = max(float(wall_thickness or 0.0), 1e-9)
    H = float(getattr(cooling, "channel_height", 0.0) or 0.0)
    w = float(getattr(cooling, "channel_width", 0.0) or 0.0)
    N = int(getattr(cooling, "channel_count", 0) or 0)

    Taw = np.asarray(heat_flux["recovery_temperature"], dtype=float)
    hf = bartz_heat_flux(contour, Pc, prop,
                         wall_temperature=np.asarray(base["gas_side_wall_temperature"]))
    h_g = np.asarray(hf["h_g"], dtype=float)
    h_c_film = np.asarray(base["h_c_film"], dtype=float)   # true film, not fin-effective
    T_c = np.asarray(base["coolant_temperature"], dtype=float)

    n = len(x)
    peak_idx_1d = int(np.argmax(np.asarray(base["gas_side_wall_temperature"])))
    if stations is None:
        sel = np.arange(n)
    else:
        sel = np.unique(np.concatenate([
            np.linspace(0, n - 1, min(stations, n)).astype(int),
            [throat_idx, peak_idx_1d],
        ]))
    T_land = np.full(n, np.nan)
    T_chan = np.full(n, np.nan)
    for i in sel:
        pitch = 2.0 * math.pi * max(float(y[i]), 1e-9) / max(N, 1)
        w_half = 0.5 * w
        land_half = max(0.5 * (pitch - w), 1e-6)
        field, s_c, xi_c = _solve_wall_cross_section(
            t_w=t_w, H=H, w_half=w_half, land_half=land_half, k_wall=k_wall,
            h_g=float(h_g[i]), T_aw=float(Taw[i]),
            h_c=float(h_c_film[i]), T_c=float(T_c[i]),
            n_s=n_s, n_xi=n_xi)
        gas_row = field[:, 0]                  # ξ = 0 (gas-side hot face)
        T_land[i] = float(np.nanmax(gas_row))  # land midpoint (s = p/2)
        T_chan[i] = float(np.nanmin(gas_row))  # channel centre (s = 0)

    peak_land = float(np.nanmax(T_land))
    max_wall = float(getattr(cooling, "max_wall_temperature", 950.0) or 950.0)
    out = dict(base)
    out.update({
        "T_wg_land": T_land,
        "T_wg_channel": T_chan,
        "circumferential_spread": T_land - T_chan,
        "peak_gas_side_wall_temperature": peak_land,
        "estimated_wall_temperature": peak_land,
        "peak_land_wall_temperature": peak_land,
        "throat_circumferential_spread": float(T_land[throat_idx] - T_chan[throat_idx])
        if not np.isnan(T_land[throat_idx]) else float("nan"),
        "cooling_margin": float(max_wall / max(peak_land, 1e-9)),
        "stations_solved": int(len(sel)),
        "model": "regen_2d_cross_section",
        "fidelity": "2d_cross_section",
    })
    return out


def regenerative_cooling(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
    prop: Any,
    Pc: float,
    *,
    fidelity: str = "1d",
    **kwargs,
) -> dict:
    """Unified regenerative-cooling entry point with a fidelity ladder:

    * ``"1d"``  — :func:`regenerative_cooling_analysis`: coupled
      Sieder-Tate / 1-D wall conduction with the Level-1 fin (land)
      area, Dean curvature, and Darcy-Weisbach pressure-drop
      corrections.  Cheap; the engineering preliminary-design model.
    * ``"2d"``  — :func:`regenerative_cooling_2d`: adds a per-station
      2-D (r-θ) cross-section conduction solve resolving the
      circumferential land hot spot (the burnthrough-relevant peak the
      1-D circuit averages away).
    * ``"3d"``  — :func:`regenerative_cooling_3d`: a single 3-D
      conjugate conduction solve of the channel sector with **axial**
      conduction, which relieves the throat hot spot the per-station
      2-D model cannot share.  (3-D conduction with convective
      gas/coolant BCs; full coolant CFD is out of scope.)

    Extra ``kwargs`` pass through to the selected solver (grid sizes,
    correction toggles).
    """
    if fidelity == "1d":
        return regenerative_cooling_analysis(
            heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
            **kwargs)
    if fidelity == "2d":
        return regenerative_cooling_2d(
            heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
            **kwargs)
    if fidelity == "3d":
        return regenerative_cooling_3d(
            heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
            **kwargs)
    raise ValueError(f"fidelity must be '1d', '2d', or '3d', got {fidelity!r}")


def regenerative_cooling_3d(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
    prop: Any,
    Pc: float,
    *,
    n_x: int = 40,
    n_s: int = 10,
    n_xi: int = 12,
    n_outer: int = 6,
    flow_from: str = "exit",
    **kwargs,
) -> dict:
    """Level-3 **3-D conjugate** regen analysis: a single finite-volume
    conduction solve over the whole channel sector (axial x, circumferential
    s half-pitch, radial ξ), so the wall conducts in all three directions
    at once — including **axially**, which relieves the throat hot spot
    that the per-station Level-2 model cannot share between stations.

    Honest scope: this is 3-D conjugate *conduction* of the solid wall
    with convective gas (full Bartz) and coolant (Sieder-Tate) boundary
    conditions and a 1-D coolant energy march (outer-iterated to a fixed
    point).  Resolving the turbulent coolant *flow field* itself is
    CFD-tool territory; here the coolant enters through the validated
    convective BC.

    Returns the 1-D keys plus the 3-D field summary; the governing
    ``peak_gas_side_wall_temperature`` is the 3-D gas-side land hot spot
    (typically below the Level-2 value by the axial-conduction relief).
    """
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    base = regenerative_cooling_analysis(
        heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
        **kwargs)

    x_full = np.asarray(base["x"]); y_full = np.asarray(contour["y"], dtype=float)
    Rt = float(contour["Rt"])
    k_wall = max(float(getattr(material, "conductivity", 15.0) or 15.0), 1e-9)
    t_w = max(float(wall_thickness or 0.0), 1e-9)
    H = float(getattr(cooling, "channel_height", 0.0) or 0.0)
    w = float(getattr(cooling, "channel_width", 0.0) or 0.0)
    N = int(getattr(cooling, "channel_count", 0) or 0)
    mdot = float(getattr(cooling, "coolant_mass_flow", 0.0) or 0.0)
    inlet_T = float(getattr(cooling, "coolant_inlet_temperature", 293.0))
    cprops = resolve_coolant_properties(cooling)
    cp_cool = float(cprops["cp"])

    # Downsample the axial grid (always keep the throat).
    nf = len(x_full)
    throat_full = int(np.argmin(np.abs(y_full - Rt)))
    xs_idx = np.unique(np.concatenate([
        np.linspace(0, nf - 1, min(n_x, nf)).astype(int), [throat_full]]))
    nx = len(xs_idx)
    xg = x_full[xs_idx]; yg = y_full[xs_idx]
    Taw_g = np.asarray(heat_flux["recovery_temperature"], dtype=float)[xs_idx]
    # Gas-side h_g at the base wall temperature, on the coarse grid.
    hf_g = bartz_heat_flux(contour, Pc, prop,
                           wall_temperature=np.asarray(base["gas_side_wall_temperature"]))
    h_g_g = np.asarray(hf_g["h_g"], dtype=float)[xs_idx]
    h_c_g = np.asarray(base["h_c_film"], dtype=float)[xs_idx]
    dx = np.gradient(xg)
    dxi = (t_w + H) / n_xi
    throat_g = int(np.argmin(np.abs(yg - Rt)))

    # Per-station cross-section geometry.
    pitch = 2.0 * math.pi * np.maximum(yg, 1e-9) / max(N, 1)
    half_pitch = 0.5 * pitch
    w_half = 0.5 * w
    ds = half_pitch / n_s
    xi_c = (np.arange(n_xi) + 0.5) * dxi
    # Solid + channel-void masks per (k, i, j).
    solid = np.zeros((nx, n_s, n_xi), dtype=bool)
    for k in range(nx):
        s_c = (np.arange(n_s) + 0.5) * ds[k]
        S, X = np.meshgrid(s_c, xi_c, indexing="ij")
        solid[k] = (X <= t_w) | (S >= w_half)
    gid = -np.ones((nx, n_s, n_xi), dtype=int)
    cells = np.argwhere(solid)
    for n, (k, i, j) in enumerate(cells):
        gid[k, i, j] = n
    M = len(cells)

    def robin(h, area, dist):
        return area / (1.0 / max(h, 1e-30) + dist / (2.0 * k_wall))

    T_c = np.full(nx, inlet_T)
    order = np.arange(nx) if flow_from == "throat" else np.arange(nx)[::-1]
    inv = np.argsort(order)
    T = np.full(M, inlet_T)

    for _outer in range(max(n_outer, 1)):
        rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
        b = np.zeros(M)
        for n, (k, i, j) in enumerate(cells):
            diag = 0.0
            A_s = dxi * dx[k]            # s-face area
            A_xi = ds[k] * dx[k]         # ξ-face area
            A_x = ds[k] * dxi            # axial-face area
            # s neighbours (circumferential)
            for di in (-1, 1):
                ii = i + di
                if 0 <= ii < n_s and solid[k, ii, j]:
                    g = k_wall * A_s / ds[k]
                    rows.append(n); cols.append(gid[k, ii, j]); vals.append(g); diag += g
                elif 0 <= ii < n_s and not solid[k, ii, j]:
                    g = robin(h_c_g[k], A_s, ds[k]); diag += g; b[n] -= g * T_c[k]
                # domain edge in s -> adiabatic symmetry
            # ξ neighbours (radial)
            for dj in (-1, 1):
                jj = j + dj
                if 0 <= jj < n_xi and solid[k, i, jj]:
                    g = k_wall * A_xi / dxi
                    rows.append(n); cols.append(gid[k, i, jj]); vals.append(g); diag += g
                elif jj < 0:
                    g = robin(h_g_g[k], A_xi, dxi); diag += g; b[n] -= g * Taw_g[k]
                elif 0 <= jj < n_xi and not solid[k, i, jj]:
                    g = robin(h_c_g[k], A_xi, dxi); diag += g; b[n] -= g * T_c[k]
                # jj == n_xi (outer/fin tip) -> adiabatic
            # x neighbours (AXIAL conduction — the Level-3 physics)
            for dk in (-1, 1):
                kk = k + dk
                if 0 <= kk < nx and solid[kk, i, j]:
                    dist = abs(xg[kk] - xg[k]) + 1e-12
                    g = k_wall * A_x / dist
                    rows.append(n); cols.append(gid[kk, i, j]); vals.append(g); diag += g
            rows.append(n); cols.append(n); vals.append(-diag)
        A = _sp.csr_matrix((vals, (rows, cols)), shape=(M, M))
        T = _spla.spsolve(A, b)

        # Coolant energy march from the coolant-face heat pickup per station.
        Qk = np.zeros(nx)
        for n, (k, i, j) in enumerate(cells):
            A_s = dxi * dx[k]; A_xi = ds[k] * dx[k]
            for di in (-1, 1):
                ii = i + di
                if 0 <= ii < n_s and not solid[k, ii, j]:
                    g = robin(h_c_g[k], A_s, ds[k]); Qk[k] += g * (T[n] - T_c[k])
            jj = j + 1
            if jj < n_xi and not solid[k, i, jj]:
                g = robin(h_c_g[k], A_xi, dxi); Qk[k] += g * (T[n] - T_c[k])
        Qk_per_channel = np.maximum(Qk, 0.0)
        dT = Qk_per_channel / max(mdot / max(N, 1) * cp_cool, 1e-12)
        T_c = inlet_T + np.cumsum(dT[order])[inv]

    # Extract the gas-side (ξ=0) land hot spot and channel-centre per station.
    T_land = np.full(nx, np.nan); T_chan = np.full(nx, np.nan)
    for n, (k, i, j) in enumerate(cells):
        if j == 0:
            T_land[k] = np.nanmax([T_land[k], T[n]]) if not np.isnan(T_land[k]) else T[n]
            T_chan[k] = np.nanmin([T_chan[k], T[n]]) if not np.isnan(T_chan[k]) else T[n]

    peak_land = float(np.nanmax(T_land))
    max_wall = float(getattr(cooling, "max_wall_temperature", 950.0) or 950.0)
    out = dict(base)
    out.update({
        "x_3d": xg,
        "T_wg_land_3d": T_land,
        "T_wg_channel_3d": T_chan,
        "coolant_temperature_3d": T_c,
        "peak_gas_side_wall_temperature": peak_land,
        "estimated_wall_temperature": peak_land,
        "peak_land_wall_temperature": peak_land,
        "throat_wall_temperature": float(T_land[throat_g]),
        "coolant_outlet_temperature": float(T_c[order[-1]]),
        "cooling_margin": float(max_wall / max(peak_land, 1e-9)),
        "grid": {"n_x": nx, "n_s": n_s, "n_xi": n_xi, "cells": int(M)},
        "model": "regen_3d_conjugate",
        "fidelity": "3d_conjugate_conduction",
    })
    return out


def regenerative_cooling_screen(
    heat_flux: dict,
    contour: dict,
    cooling: Any,
    material: Any,
    wall_thickness: float | None,
    prop: Any | None = None,
    Pc: float | None = None,
) -> dict:
    """Screen rectangular regenerative cooling channels.

    When ``prop`` and ``Pc`` are supplied this runs the real coupled
    Sieder-Tate / 1-D wall-conduction analysis
    (:func:`regenerative_cooling_analysis`); the gas side is the full
    Bartz coefficient.  Without them (legacy two-argument callers) it
    falls back to a fixed-resistance estimate using the heat-flux dict's
    own gas-side coefficient — no coupled solve, but still grounded in
    h_g rather than the old ad-hoc ``1000 + 2e8·area`` film model.
    """
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

    if prop is not None and Pc is not None:
        return regenerative_cooling_analysis(
            heat_flux, contour, cooling, material, wall_thickness, prop, Pc,
        )

    # Legacy fallback (no gas thermochemistry passed): single-station
    # series resistance at the peak flux using Sieder-Tate for h_c.
    N = int(getattr(cooling, "channel_count", 0) or 0)
    w = float(getattr(cooling, "channel_width", 0.0) or 0.0)
    h = float(getattr(cooling, "channel_height", 0.0) or 0.0)
    mdot = float(getattr(cooling, "coolant_mass_flow", 0.0) or 0.0)
    coolant_cp = float(getattr(cooling, "coolant_cp", 3500.0) or 3500.0)
    k_wall = max(float(getattr(material, "conductivity", 15.0) or 15.0), 1e-9)
    thickness = max(float(wall_thickness or 0.0), 1e-9)
    cprops = resolve_coolant_properties(cooling)

    A_ch = max(w * h, 1e-12)
    D_h = hydraulic_diameter(w, h)
    G = (mdot / max(N, 1)) / A_ch
    mu_b = coolant_viscosity(cprops, coolant_inlet)
    h_c = float(sieder_tate_coefficient(G, D_h, cprops,
                                        mu_bulk=mu_b, mu_wall=mu_b))

    x = np.asarray(contour["x"], dtype=float)
    y = np.asarray(contour["y"], dtype=float)
    q = np.asarray(heat_flux["q"], dtype=float)
    h_g_arr = np.asarray(heat_flux.get("h_g", np.zeros_like(q)), dtype=float)
    area_weight = 2.0 * math.pi * y
    total_heat = float(np.trapezoid(q * area_weight, x))
    coolant_rise = total_heat / max(mdot * coolant_cp, 1e-9) if mdot > 0 else float("inf")
    coolant_out = coolant_inlet + coolant_rise

    q_max = float(heat_flux["q_max"])
    Taw = float(heat_flux["adiabatic_wall_temperature"])
    h_g_peak = float(h_g_arr.max()) if h_g_arr.size and h_g_arr.max() > 0 else (
        q_max / max(Taw - 900.0, 1.0))
    wall_temp = coolant_out + q_max * (thickness / k_wall + 1.0 / max(h_c, 1e-9))
    max_wall = float(getattr(cooling, "max_wall_temperature", 950.0) or 950.0)
    margin = max_wall / max(wall_temp, 1e-9)

    warnings: list[str] = []
    if N <= 0 or w <= 0.0 or h <= 0.0:
        warnings.append("Regenerative cooling requires positive channel geometry.")
    if mdot <= 0.0:
        warnings.append("Regenerative cooling requires positive coolant mass flow.")
    warnings.append("Coupled Sieder-Tate solve skipped (no gas thermochemistry "
                    "passed); reported wall temperature is a peak-flux estimate.")

    return {
        "method": method,
        "estimated_wall_temperature": float(wall_temp),
        "coolant_outlet_temperature": float(coolant_out) if math.isfinite(coolant_out) else None,
        "coolant_temperature_rise": float(coolant_rise) if math.isfinite(coolant_rise) else None,
        "channel_flow_area": float(N * A_ch),
        "channel_hydraulic_diameter": float(D_h),
        "throat_h_c": float(h_c),
        "total_heat_load": total_heat,
        "cooling_margin": float(margin),
        "coolant_properties": cprops,
        "model": "sieder_tate_peak_flux_estimate",
        "warnings": warnings,
    }


def coaxial_shell_wall_stress(
    *,
    pressure_differential: float,
    inner_radius: float,
    wall_thickness: float,
    heat_flux: float,
    elastic_modulus: float,
    thermal_expansion: float,
    poisson_ratio: float,
    conductivity: float,
    yield_strength: float | None = None,
) -> dict:
    """Combined inner-liner wall stress — NASA SP-125 eq. 4-31.

    For a coaxial-shell (liner + jacket) regen wall, the maximum stress is
    at the inner-wall surface of the inner shell and combines the
    pressure-differential stress with the thermal stress from the
    through-wall temperature gradient (Huzel & Huang, SP-125 printed
    p. 109, ``propulsion_texts/19710019929.pdf``):

        S_c = (p_co - p_g) R / t  +  E a q t / (2 (1 - v) k)
              \\_____pressure_____/    \\________thermal________/

    where ``p_co - p_g`` is the coolant/gas pressure differential, ``R``
    the liner radius, ``t`` the liner thickness, ``q`` the local heat
    flux, ``E`` the modulus, ``a`` the thermal-expansion coefficient,
    ``v`` Poisson's ratio and ``k`` the conductivity.  The thermal term
    grows with both ``q`` and ``t`` (a thicker wall runs a steeper
    gradient), while the conduction limit on wall temperature wants ``t``
    small — the two-sided squeeze on liner thickness SP-125 describes.

    All inputs SI; returns the two stress components, their sum and (if
    ``yield_strength`` is given) the yield margin ``S_y / S_c``.
    """
    t = max(float(wall_thickness), 1e-12)
    k = max(float(conductivity), 1e-12)
    pressure_stress = float(pressure_differential) * float(inner_radius) / t
    thermal_stress = (
        float(elastic_modulus) * float(thermal_expansion) * float(heat_flux) * t
        / (2.0 * (1.0 - float(poisson_ratio)) * k)
    )
    combined = abs(pressure_stress) + thermal_stress
    out = {
        "pressure_stress": pressure_stress,
        "thermal_stress": thermal_stress,
        "combined_stress": combined,
        "model": "sp125_coaxial_shell_eq_4_31",
    }
    if yield_strength:
        out["stress_margin"] = float(yield_strength) / max(combined, 1e-9)
    return out


def coaxial_shell_wall_stress_profile(
    *,
    pressure_differential,
    inner_radius,
    wall_thickness,
    heat_flux,
    elastic_modulus: float,
    thermal_expansion: float,
    poisson_ratio: float,
    conductivity: float,
    yield_strength: float | None = None,
) -> dict:
    """Station-wise NASA SP-125 eq. 4-31 liner stress.

    Unlike :func:`coaxial_shell_wall_stress`, every load/geometry field may
    be an array.  The returned governing stress is the maximum combined
    pressure-plus-thermal stress at a common station, with the full profiles
    retained for audit and CAD sizing.
    """
    dp = np.asarray(pressure_differential, dtype=float)
    radius = np.asarray(inner_radius, dtype=float)
    thickness = np.maximum(np.asarray(wall_thickness, dtype=float), 1e-12)
    q = np.asarray(heat_flux, dtype=float)
    dp, radius, thickness, q = np.broadcast_arrays(dp, radius, thickness, q)
    k = max(float(conductivity), 1e-12)
    pressure = dp * radius / thickness
    thermal = (
        float(elastic_modulus) * float(thermal_expansion) * q * thickness
        / (2.0 * (1.0 - float(poisson_ratio)) * k)
    )
    combined = np.abs(pressure) + thermal
    idx = int(np.nanargmax(combined))
    out = {
        "pressure_stress_profile": pressure,
        "thermal_stress_profile": thermal,
        "combined_stress_profile": combined,
        "pressure_stress": float(pressure[idx]),
        "thermal_stress": float(thermal[idx]),
        "combined_stress": float(combined[idx]),
        "governing_index": idx,
        "model": "sp125_coaxial_shell_eq_4_31_stationwise",
    }
    if yield_strength:
        out["stress_margin_profile"] = float(yield_strength) / np.maximum(combined, 1e-9)
        out["stress_margin"] = float(yield_strength) / max(float(combined[idx]), 1e-9)
    return out


def sp125_inelastic_buckling_critical_stress(
    *,
    wall_thickness,
    local_radius,
    tangent_modulus_tension,
    tangent_modulus_compression,
    poisson_ratio: float,
) -> np.ndarray:
    """SP-125 equation 4-29 longitudinal inelastic-buckling stress.

    SP-125 printed p. 108 defines

    ``S_c = [4 E_t E_c / (sqrt(E_t)+sqrt(E_c))²] *
             t / [sqrt(3(1-v²)) r]``.

    ``E_t`` and ``E_c`` are tangent moduli at wall temperature from the
    tensile and compressive stress-strain curves.  They are *not* the elastic
    modulus.  The caller must therefore either supply qualified values or
    explicitly identify a screening knockdown used to derive them.
    """
    t = np.maximum(np.asarray(wall_thickness, dtype=float), 1e-12)
    radius = np.maximum(np.asarray(local_radius, dtype=float), 1e-12)
    Et = np.maximum(np.asarray(tangent_modulus_tension, dtype=float), 1e-9)
    Ec = np.maximum(np.asarray(tangent_modulus_compression, dtype=float), 1e-9)
    equivalent_modulus = (
        4.0 * Et * Ec / np.maximum((np.sqrt(Et) + np.sqrt(Ec)) ** 2, 1e-18)
    )
    return equivalent_modulus * t / (
        math.sqrt(max(3.0 * (1.0 - float(poisson_ratio) ** 2), 1e-12))
        * radius
    )


def rib_supported_liner_buckling_profile(
    *,
    pressure_differential,
    inner_radius,
    wall_thickness,
    unsupported_span,
    elastic_modulus: float,
    poisson_ratio: float,
    plate_coefficient: float = 4.0,
    knockdown: float = 0.65,
) -> dict:
    """External-pressure compression screen for a liner between ribs.

    This is deliberately separate from SP-125 equation 4-29.  The local
    coolant-over-gas pressure creates circumferential compression
    ``|Δp|R/t`` in the inner shell.  The rib-supported strip is screened as a
    simply supported plate under in-plane compression:

    ``sigma_cr = knockdown * k*pi²*E/[12(1-v²)] * (t/b)²``.

    ``b`` is the unsupported channel span between adjacent ribs.  The model
    is a preliminary local-collapse gate, not a substitute for nonlinear
    shell buckling FEA with imperfections.
    """
    dp = np.abs(np.asarray(pressure_differential, dtype=float))
    radius = np.maximum(np.asarray(inner_radius, dtype=float), 1e-12)
    t = np.maximum(np.asarray(wall_thickness, dtype=float), 1e-12)
    span = np.maximum(np.asarray(unsupported_span, dtype=float), 1e-12)
    compressive = dp * radius / t
    coefficient = (
        max(float(knockdown), 0.0) * float(plate_coefficient) * math.pi ** 2
        * float(elastic_modulus)
        / max(12.0 * (1.0 - float(poisson_ratio) ** 2), 1e-12)
    )
    critical = coefficient * (t / span) ** 2
    margin = critical / np.maximum(compressive, 1e-9)
    idx = int(np.nanargmin(margin))
    return {
        "compressive_stress_profile": compressive,
        "critical_stress_profile": critical,
        "margin_profile": margin,
        "margin": float(margin[idx]),
        "governing_index": idx,
        "model": "rib_supported_liner_plate_buckling_screen",
        "plate_coefficient": float(plate_coefficient),
        "knockdown": float(knockdown),
    }


def thermal_fatigue_strain(
    delta_T_wall: float, *, thermal_expansion: float, poisson_ratio: float,
    mechanical_strain: float = 0.0,
) -> float:
    """Strain-controlled total strain range Δε for one regen-liner cycle.

    The dominant driver is the CONSTRAINED thermal expansion of the hot
    face over a start→steady→shutdown cycle: the hot face wants to grow
    ``α·ΔT`` relative to the cooler backing structure but is restrained,
    so a biaxially constrained strain ``α·ΔT/(1−ν)`` is imposed each cycle
    — the strain counterpart of SP-125 eq. 4-28's thermal stress
    ``E·α·ΔT`` (``propulsion_texts/19710019929.pdf`` printed p. 108).
    ``ΔT`` is the through-wall drop ``T_wg − T_wc = q·t_w/k_w``.  Any
    mechanical (pressure) strain adds.  Being strain-CONTROLLED is why it
    is *low*-cycle: the metal yields each cycle to accommodate ``ΔT``.
    """
    return (float(thermal_expansion) * float(delta_T_wall)
            / max(1.0 - float(poisson_ratio), 1e-6) + float(mechanical_strain))


def coffin_manson_cycles(
    strain_range: float, *, elastic_modulus: float,
    fatigue_strength_coeff: float, fatigue_strength_exp: float,
    fatigue_ductility_coeff: float, fatigue_ductility_exp: float,
    max_cycles: float = 1e9,
) -> float:
    """Cycles to failure ``N_f`` from the strain-life relation::

        Δε/2 = (σ_f'/E)(2N_f)^b + ε_f'(2N_f)^c

    the elastic (Basquin, ``σ_f'``/``b``) plus plastic (Coffin-Manson,
    ``ε_f'``/``c``) terms, ``b, c < 0``.  The right side decreases
    monotonically in ``N_f``, so it is inverted by bisection in
    ``log(2N_f)``.  Standard strain-life fatigue (Coffin 1954; Manson
    1953) — used here as an order-of-magnitude screen with handbook
    coefficients, NOT a certified life prediction.

    A strain range too large for even half a cycle returns ~0.5; one
    below the strain at ``max_cycles`` returns ``max_cycles`` (run-out).
    """
    half = 0.5 * float(strain_range)
    if half <= 0.0:
        return float(max_cycles)
    E = max(float(elastic_modulus), 1e-9)
    sf, b = float(fatigue_strength_coeff), float(fatigue_strength_exp)
    ef, c = float(fatigue_ductility_coeff), float(fatigue_ductility_exp)

    def rhs(two_Nf: float) -> float:        # predicted Δε/2 at this life
        return (sf / E) * two_Nf ** b + ef * two_Nf ** c

    lo, hi = 1.0, 2.0 * float(max_cycles)   # bounds on 2N_f
    if half >= rhs(lo):                     # over-strained: < half a cycle
        return 0.5
    if half <= rhs(hi):                     # under the run-out strain
        return float(max_cycles)
    for _ in range(200):                    # bisect (rhs is decreasing)
        mid = math.sqrt(lo * hi)
        if rhs(mid) > half:
            lo = mid
        else:
            hi = mid
    return 0.5 * math.sqrt(lo * hi)


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
    """Evaluate thermal and hoop-stress screening margins.

    The hoop check is the thin-wall membrane stress ``Pc R / t``.  When
    the material carries the elastic/thermal properties (``elastic_
    modulus``, ``thermal_expansion``, ``poisson_ratio``) the screen also
    reports the SP-125 eq. 4-31 combined stress (pressure differential +
    through-wall thermal stress); the pressure differential is taken as
    the chamber-pressure hoop ``Pc`` as a conservative stand-in when the
    coolant pressure is not supplied.  ``stress_margin`` then uses the
    combined stress (the tighter, more physical bound)."""
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

    # Combined SP-125 eq. 4-31 stress when the material has E, a, v.
    E = getattr(material, "elastic_modulus", None)
    alpha = getattr(material, "thermal_expansion", None)
    nu = getattr(material, "poisson_ratio", None)
    combined = None
    if thickness > 0 and E and alpha and nu is not None:
        # eq. 4-31 governs at the throat (peak flux), so co-locate the
        # radius with q_max there — the narrowest contour station.
        throat_radius = float(np.min(y))
        p_diff = cooling.get("liner_pressure_differential")
        q_profile = cooling.get("q")
        if p_diff is not None and q_profile is not None:
            combined = coaxial_shell_wall_stress_profile(
                pressure_differential=p_diff, inner_radius=y,
                wall_thickness=thickness, heat_flux=q_profile,
                elastic_modulus=float(E), thermal_expansion=float(alpha),
                poisson_ratio=float(nu),
                conductivity=float(getattr(material, "conductivity", 15.0) or 15.0),
                yield_strength=yield_strength or None,
            )
        else:
            combined = coaxial_shell_wall_stress(
                pressure_differential=Pc, inner_radius=throat_radius,
                wall_thickness=thickness, heat_flux=float(thermal["q_max"]),
                elastic_modulus=float(E), thermal_expansion=float(alpha),
                poisson_ratio=float(nu),
                conductivity=float(getattr(material, "conductivity", 15.0) or 15.0),
                yield_strength=yield_strength or None,
            )

    # Use the combined (pressure + thermal) stress for the margin when we
    # have it; otherwise fall back to the bare hoop stress.
    governing_stress = combined["combined_stress"] if combined else hoop_stress
    stress_margin = yield_strength / max(governing_stress, 1e-9) if yield_strength > 0 else 0.0
    temperature_margin = max_material_temp / max(wall_temp, 1e-9) if max_material_temp > 0 else 0.0
    heat_flux_margin = max_heat_flux / max(float(thermal["q_max"]), 1e-9) if max_heat_flux > 0 else 0.0

    result = {
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
    if combined:
        result["thermal_stress"] = float(combined["thermal_stress"])
        result["pressure_stress"] = float(combined["pressure_stress"])
        result["combined_stress"] = float(combined["combined_stress"])
        result["stress_model"] = combined["model"]
        if "governing_index" in combined:
            result["stress_governing_index"] = int(combined["governing_index"])
    return result


def _gas_viscosity(T: float) -> float:
    """Sutherland-like gas viscosity estimate for hot combustion products."""
    mu0 = 1.716e-5
    T0 = 273.15
    S = 110.4
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)
