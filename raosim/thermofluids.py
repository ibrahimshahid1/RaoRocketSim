"""Higher-fidelity coolant-network, real-fluid, radiation, and boiling tools.

These models deliberately expose their validity and data source.  They extend
the axisymmetric preliminary regen solver; they do not turn it into a
qualification CFD/CHT analysis.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from raosim.coolants import canonical_coolant_name


SIGMA_SB = 5.670374419e-8
G_STANDARD = 9.80665

_COOLPROP_NAMES = {
    "methane": "Methane",
    "hydrogen": "Hydrogen",
}


def real_fluid_coolant_state(
    coolant: str,
    temperature,
    pressure,
    *,
    backend: str = "auto",
) -> dict:
    """Return station-wise real-fluid properties using CoolProp.

    ``backend="auto"`` uses CoolProp when installed for methane or hydrogen
    and otherwise returns an explicit unavailable status.  No constant-
    property values are invented here; the caller owns any screening
    fallback.
    """
    name = canonical_coolant_name(coolant)
    fluid = _COOLPROP_NAMES.get(name)
    T, P = np.broadcast_arrays(
        np.asarray(temperature, dtype=float),
        np.asarray(pressure, dtype=float),
    )
    if backend not in {"auto", "coolprop", "constant"}:
        raise ValueError("real-fluid backend must be auto, coolprop, or constant")
    if backend == "constant":
        return {
            "available": False,
            "status": "constant_property_backend_requested",
            "backend": "constant",
        }
    if fluid is None:
        return {
            "available": False,
            "status": "coolant_not_supported_by_real_fluid_backend",
            "backend": "unavailable",
        }
    try:
        from CoolProp.CoolProp import PropsSI
    except Exception:
        return {
            "available": False,
            "status": "coolprop_not_installed",
            "backend": "unavailable",
        }

    def props(key: str):
        values = PropsSI(key, "T", T.ravel(), "P", P.ravel(), fluid)
        return np.asarray(values, dtype=float).reshape(T.shape)

    try:
        out = {
            "available": True,
            "status": "real_fluid_properties_evaluated",
            "backend": "CoolProp_HEOS",
            "fluid": fluid,
            "temperature": T,
            "pressure": P,
            "rho": props("Dmass"),
            "mu": props("VISCOSITY"),
            "k": props("CONDUCTIVITY"),
            "cp": props("Cpmass"),
            "h": props("Hmass"),
            "Pr": props("PRANDTL"),
            "critical_temperature": float(PropsSI("Tcrit", fluid)),
            "critical_pressure": float(PropsSI("Pcrit", fluid)),
            "reference":
                "CoolProp HEOS; Bell et al., Ind. Eng. Chem. Res. 53 (2014)",
        }
    except Exception as exc:
        return {
            "available": False,
            "status": f"coolprop_state_failure:{type(exc).__name__}",
            "backend": "CoolProp_HEOS",
        }
    return out


def spectral_band_radiative_heat_flux(
    gas_temperature,
    wall_temperature,
    path_length,
    bands: Iterable[dict[str, Any]],
    *,
    wall_emissivity: float = 1.0,
    view_factor: float = 1.0,
) -> dict:
    """Homogeneous nonscattering multiband radiative-transfer screen.

    For each band ``b``:

    ``q_b = eps_w F w_b sigma (T_g^4 - T_w^4) [1-exp(-kappa_b L)]``.

    This is the formal RTE solution for an isothermal homogeneous path with
    band-mean absorption.  It retains spectral bands, unlike a single gray
    multiplier, but does not ray-trace an inhomogeneous chamber.  Each band
    dictionary needs ``weight`` (Planck fraction) and
    ``absorption_coefficient`` [1/m], each scalar or station-wise.
    """
    Tg, Tw, L = np.broadcast_arrays(
        np.asarray(gas_temperature, dtype=float),
        np.asarray(wall_temperature, dtype=float),
        np.asarray(path_length, dtype=float),
    )
    if not (0.0 <= wall_emissivity <= 1.0):
        raise ValueError("wall_emissivity must be in [0, 1]")
    if not (0.0 <= view_factor <= 1.0):
        raise ValueError("view_factor must be in [0, 1]")
    band_fluxes = []
    names = []
    weight_sum = 0.0
    for i, band in enumerate(bands):
        weight = float(band["weight"])
        if weight < 0.0:
            raise ValueError("radiation band weights must be nonnegative")
        kappa = np.broadcast_to(
            np.asarray(band["absorption_coefficient"], dtype=float), Tg.shape
        )
        transmissivity = np.exp(-np.maximum(kappa, 0.0) * np.maximum(L, 0.0))
        q_band = (
            float(wall_emissivity)
            * float(view_factor)
            * weight
            * SIGMA_SB
            * (np.maximum(Tg, 0.0) ** 4 - np.maximum(Tw, 0.0) ** 4)
            * (1.0 - transmissivity)
        )
        band_fluxes.append(q_band)
        names.append(str(band.get("name", f"band_{i}")))
        weight_sum += weight
    if not band_fluxes:
        raise ValueError("at least one radiation band is required")
    if weight_sum > 1.0 + 1e-9:
        raise ValueError("radiation band weights must sum to no more than one")
    q = np.sum(np.stack(band_fluxes), axis=0)
    return {
        "q": q,
        "band_heat_fluxes": dict(zip(names, band_fluxes)),
        "band_weight_sum": float(weight_sum),
        "model": "homogeneous_nonscattering_multiband_rte",
        "fidelity": "spectral_band_screening",
        "reference":
            "Leccese et al., J. Propul. Power, doi:10.2514/1.B36589, Eqs. 3-7",
    }


def leccese_participating_gas_radiation(
    gas_temperature,
    wall_temperature,
    path_length,
    pressure,
    *,
    propellant_family: str,
    wall_emissivity: float = 1.0,
) -> dict:
    """Leccese-calibrated gray participating-gas radiation screen.

    Leccese et al. report nominal mixture absorption coefficients of
    2.4 1/m for O2/CH4 and 1.9 1/m for O2/H2 at 10 bar.  Their Eq. 7 makes
    mixture absorption proportional to pressure.  This preset applies that
    pressure scaling to a homogeneous path; it is intentionally labeled gray
    and should be replaced by :func:`spectral_band_radiative_heat_flux` when
    species/band data are available.
    """
    family = str(propellant_family).strip().lower().replace("-", "")
    if family in {"methane", "ch4", "loxch4", "o2ch4"}:
        kappa_ref = 2.4
        label = "oxygen_methane"
    elif family in {"hydrogen", "h2", "loxh2", "o2h2"}:
        kappa_ref = 1.9
        label = "oxygen_hydrogen"
    else:
        raise ValueError("Leccese preset supports oxygen/methane or oxygen/hydrogen")
    P = np.asarray(pressure, dtype=float)
    kappa = kappa_ref * P / 1.0e6
    result = spectral_band_radiative_heat_flux(
        gas_temperature,
        wall_temperature,
        path_length,
        [{"name": label, "weight": 1.0, "absorption_coefficient": kappa}],
        wall_emissivity=wall_emissivity,
    )
    result.update({
        "model": "leccese_gray_participating_gas_screen",
        "fidelity": "gray_homogeneous_calibrated_screen",
        "reference_absorption_coefficient": kappa_ref,
        "reference_pressure": 1.0e6,
        "propellant_family": label,
    })
    return result


def zuber_critical_heat_flux(
    *,
    latent_heat: float,
    vapor_density: float,
    liquid_density: float,
    surface_tension: float,
    acceleration: float = G_STANDARD,
    coefficient: float = 0.131,
) -> float:
    """Hydrodynamic saturated CHF (Zuber/Kutateladze) [W/m²].

    This is a conservative pool-boiling reference, not a forced-flow rocket
    channel correlation.  It is useful as a minimum-fidelity crisis screen
    when no validated cryogenic channel CHF correlation is available.
    """
    hfg = max(float(latent_heat), 0.0)
    rho_v = max(float(vapor_density), 1e-12)
    rho_l = max(float(liquid_density), rho_v)
    sigma = max(float(surface_tension), 0.0)
    accel = max(float(acceleration), 0.0)
    return (
        float(coefficient)
        * hfg
        * math.sqrt(rho_v)
        * (sigma * accel * max(rho_l - rho_v, 0.0)) ** 0.25
    )


def boiling_chf_screen(
    coolant: str,
    bulk_temperature,
    wall_temperature,
    pressure,
    heat_flux,
    *,
    backend: str = "auto",
    acceleration: float = G_STANDARD,
) -> dict:
    """Phase/boiling/CHF screen for methane and hydrogen coolants.

    Below the critical pressure, saturation properties come from CoolProp.
    ``T_wall >= T_sat`` marks a boiling-capable station and the required heat
    flux is compared with the Zuber saturated CHF reference.  At or above
    critical pressure there is no liquid-vapor CHF; the result instead flags
    the pseudo-critical/heat-transfer-deterioration regime for a dedicated
    supercritical correlation or CFD.
    """
    state = real_fluid_coolant_state(
        coolant, bulk_temperature, pressure, backend=backend
    )
    q, Tw, Tb, P = np.broadcast_arrays(
        np.asarray(heat_flux, dtype=float),
        np.asarray(wall_temperature, dtype=float),
        np.asarray(bulk_temperature, dtype=float),
        np.asarray(pressure, dtype=float),
    )
    empty = np.full(q.shape, np.nan)
    if not state.get("available"):
        return {
            "evaluated": False,
            "status": state["status"],
            "boiling_active": np.zeros(q.shape, dtype=bool),
            "saturation_temperature": empty,
            "critical_heat_flux": empty,
            "chf_margin": empty,
            "gates_feasibility": False,
            "reference":
                "Zuber hydrodynamic CHF; real-fluid saturation data required",
        }
    pcrit = float(state["critical_pressure"])
    supercritical = P >= pcrit
    Tsat = np.full(q.shape, np.nan)
    q_chf = np.full(q.shape, np.nan)
    fluid = state["fluid"]
    try:
        from CoolProp.CoolProp import PropsSI
        for index in np.ndindex(q.shape):
            if supercritical[index]:
                continue
            p_i = float(P[index])
            Tsat[index] = PropsSI("T", "P", p_i, "Q", 0, fluid)
            h_l = PropsSI("Hmass", "P", p_i, "Q", 0, fluid)
            h_v = PropsSI("Hmass", "P", p_i, "Q", 1, fluid)
            rho_l = PropsSI("Dmass", "P", p_i, "Q", 0, fluid)
            rho_v = PropsSI("Dmass", "P", p_i, "Q", 1, fluid)
            sigma = PropsSI("SURFACE_TENSION", "P", p_i, "Q", 0, fluid)
            q_chf[index] = zuber_critical_heat_flux(
                latent_heat=h_v - h_l,
                vapor_density=rho_v,
                liquid_density=rho_l,
                surface_tension=sigma,
                acceleration=acceleration,
            )
    except Exception as exc:
        return {
            "evaluated": False,
            "status": f"saturation_property_failure:{type(exc).__name__}",
            "boiling_active": np.zeros(q.shape, dtype=bool),
            "saturation_temperature": Tsat,
            "critical_heat_flux": q_chf,
            "chf_margin": empty,
            "gates_feasibility": False,
        }
    boiling = (~supercritical) & (Tw >= Tsat)
    margin = q_chf / np.maximum(q, 1e-12)
    finite = np.isfinite(margin)
    feasible = bool(np.all(margin[finite] >= 1.0)) if np.any(finite) else True
    return {
        "evaluated": True,
        "status": (
            "subcritical_zuber_chf_screen"
            if np.any(~supercritical)
            else "supercritical_no_liquid_vapor_chf_heat_transfer_deterioration_unresolved"
        ),
        "supercritical": supercritical,
        "boiling_active": boiling,
        "bulk_subcooling": Tsat - Tb,
        "wall_superheat": Tw - Tsat,
        "saturation_temperature": Tsat,
        "critical_heat_flux": q_chf,
        "chf_margin": margin,
        "minimum_chf_margin": (
            float(np.nanmin(margin)) if np.any(finite) else float("inf")
        ),
        "feasible": feasible,
        "gates_feasibility": bool(np.any(~supercritical)),
        "model": "real_fluid_phase_screen_plus_zuber_saturated_chf",
        "qualification_status":
            "screening_only_forced_flow_cryogenic_chf_correlation_required",
        "reference":
            "Zuber, AECU-4439 (1959); properties from CoolProp HEOS",
    }


def _edge_flow(delta_p: float, resistance: float) -> float:
    """Smooth signed square-law edge flow, Δp = R q |q|."""
    resistance = max(float(resistance), 1e-30)
    scale = math.sqrt(resistance * abs(float(delta_p)) + 1e-24)
    return float(delta_p) / scale


def solve_annular_manifold_network(
    *,
    channel_count: int,
    ports_per_manifold: int,
    total_mass_flow: float,
    density: float,
    channel_pressure_drop: float,
    channel_total_area: float,
    manifold_radius: float,
    port_area_ratio: float = 1.0,
    port_diameter: float | None = None,
    plenum_area_ratio: float = 2.0,
    plenum_hydraulic_diameter: float | None = None,
    roughness: float = 0.0,
    port_loss_coefficient: float = 1.5,
    channel_entry_loss_coefficient: float = 0.5,
    channel_exit_loss_coefficient: float = 1.0,
) -> dict:
    """Solve a two-ring inlet/channel/outlet hydraulic graph.

    The network contains every channel branch, both annular header rings,
    and each discrete inlet/outlet port.  Edge flows satisfy the square-law
    relation and node mass conservation; the solved branch-flow spread is a
    first-principles maldistribution metric for the assumed geometry.
    """
    n = int(channel_count)
    m = int(ports_per_manifold)
    mdot = float(total_mass_flow)
    rho = float(density)
    if n < 2 or m < 1 or mdot <= 0.0 or rho <= 0.0:
        raise ValueError("network needs N>=2, ports>=1, and positive flow/density")
    if channel_pressure_drop <= 0.0 or channel_total_area <= 0.0:
        raise ValueError("channel pressure drop and total area must be positive")
    A_channel_each = float(channel_total_area) / n
    q_channel_ref = mdot / n
    dynamic_channel = 0.5 * rho * (
        q_channel_ref / max(rho * A_channel_each, 1e-30)
    ) ** 2
    dp_channel_branch = (
        float(channel_pressure_drop)
        + (float(channel_entry_loss_coefficient)
           + float(channel_exit_loss_coefficient)) * dynamic_channel
    )
    R_channel = dp_channel_branch / max(q_channel_ref ** 2, 1e-30)

    if port_diameter is None:
        A_port_each = (
            float(port_area_ratio) * float(channel_total_area) / m
        )
        D_port = math.sqrt(4.0 * A_port_each / math.pi)
    else:
        D_port = float(port_diameter)
        A_port_each = math.pi * D_port ** 2 / 4.0
    A_plenum = float(plenum_area_ratio) * float(channel_total_area)
    D_plenum = (
        float(plenum_hydraulic_diameter)
        if plenum_hydraulic_diameter is not None
        else 2.0 * math.sqrt(A_plenum / math.pi)
    )
    q_port_ref = mdot / m
    R_port = float(port_loss_coefficient) / (
        2.0 * rho * max(A_port_each ** 2, 1e-30)
    )
    segment_length = 2.0 * math.pi * float(manifold_radius) / n
    q_ring_ref = mdot / max(2 * m, 1)
    V_ring_ref = q_ring_ref / max(rho * A_plenum, 1e-30)
    Re_ring = rho * abs(V_ring_ref) * D_plenum / 1.0e-3
    rel_rough = max(float(roughness), 0.0) / max(D_plenum, 1e-30)
    if Re_ring < 2300.0:
        f_ring = 64.0 / max(Re_ring, 1.0)
    else:
        f_ring = 0.25 / max(
            math.log10(rel_rough / 3.7 + 5.74 / max(Re_ring, 1.0) ** 0.9) ** 2,
            1e-12,
        )
    R_ring = (
        f_ring * segment_length / max(D_plenum, 1e-30)
        / (2.0 * rho * max(A_plenum ** 2, 1e-30))
    )

    source = 0
    inlet0 = 1
    outlet0 = 1 + n
    sink = 1 + 2 * n
    edges: list[tuple[int, int, float, str, int]] = []
    for i in range(n):
        edges.append((inlet0 + i, inlet0 + (i + 1) % n, R_ring, "inlet_ring", i))
        edges.append((outlet0 + i, outlet0 + (i + 1) % n, R_ring, "outlet_ring", i))
        edges.append((inlet0 + i, outlet0 + i, R_channel, "channel", i))
    port_indices = np.unique(np.floor(np.arange(m) * n / m).astype(int))
    for j, i in enumerate(port_indices):
        edges.append((source, inlet0 + int(i), R_port, "inlet_port", j))
        edges.append((outlet0 + int(i), sink, R_port, "outlet_port", j))

    unknown_nodes = list(range(sink))
    p_scale = (
        dp_channel_branch
        + 2.0 * R_port * q_port_ref ** 2
        + R_ring * q_ring_ref ** 2
    )
    guess = np.empty(sink)
    guess[source] = p_scale
    guess[inlet0:outlet0] = 0.75 * p_scale
    guess[outlet0:sink] = 0.25 * p_scale
    node_to_unknown = {node: i for i, node in enumerate(unknown_nodes)}
    smooth_dp = max(1e-10 * p_scale, 1e-6)

    def edge_flow_and_slope(delta_p, resistance):
        # Smooth q=sign(dp)*sqrt(|dp|/R) near zero so a Newton Jacobian
        # remains finite at symmetric no-flow ring segments.
        a = math.sqrt(delta_p * delta_p + smooth_dp * smooth_dp)
        root_R = math.sqrt(max(resistance, 1e-30))
        q_edge = delta_p / (root_R * math.sqrt(a))
        slope = (
            a ** -0.5 - 0.5 * delta_p * delta_p * a ** -2.5
        ) / root_R
        return q_edge, max(slope, 1e-18)

    def residual_and_jacobian(pressures, with_jacobian=True):
        p = np.zeros(sink + 1)
        p[unknown_nodes] = pressures
        balance = np.zeros(sink + 1)
        jac = np.zeros((sink, sink)) if with_jacobian else None
        for u, v, resistance, _, _ in edges:
            q_edge, slope = edge_flow_and_slope(p[u] - p[v], resistance)
            balance[u] += q_edge
            balance[v] -= q_edge
            if with_jacobian:
                iu = node_to_unknown.get(u)
                iv = node_to_unknown.get(v)
                if iu is not None:
                    jac[iu, iu] += slope
                    if iv is not None:
                        jac[iu, iv] -= slope
                if iv is not None:
                    jac[iv, iv] += slope
                    if iu is not None:
                        jac[iv, iu] -= slope
        balance[source] -= mdot
        balance[sink] += mdot
        return balance[unknown_nodes], jac

    pressures = guess.copy()
    converged = False
    iteration = 0
    for iteration in range(80):
        residual, jac = residual_and_jacobian(pressures)
        norm = float(np.linalg.norm(residual, ord=np.inf))
        if norm <= 1e-9 * mdot:
            converged = True
            break
        try:
            step = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, -residual, rcond=None)[0]
        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            trial = pressures + damping * step
            trial_residual, _ = residual_and_jacobian(
                trial, with_jacobian=False
            )
            if np.linalg.norm(trial_residual, ord=np.inf) < norm:
                pressures = trial
                accepted = True
                break
        if not accepted:
            pressures += 0.005 * step

    p = np.zeros(sink + 1)
    p[unknown_nodes] = pressures
    flows_by_kind: dict[str, list[tuple[int, float]]] = {}
    for u, v, resistance, kind, index in edges:
        q_edge = _edge_flow(p[u] - p[v], resistance)
        flows_by_kind.setdefault(kind, []).append((index, q_edge))
    channel_flows = np.array(
        [q for _, q in sorted(flows_by_kind["channel"])], dtype=float
    )
    mean = float(np.mean(channel_flows))
    spread = (
        float((np.max(channel_flows) - np.min(channel_flows)) / mean)
        if mean > 0.0 else float("inf")
    )
    return {
        "converged": converged,
        "status": (
            f"newton_converged_in_{iteration + 1}_iterations"
            if converged else "newton_iteration_limit"
        ),
        "total_pressure_drop": float(p[source]),
        "channel_branch_pressure_drop_reference": float(dp_channel_branch),
        "channel_mass_flows": channel_flows,
        "channel_flow_mean": mean,
        "channel_flow_min": float(np.min(channel_flows)),
        "channel_flow_max": float(np.max(channel_flows)),
        "maldistribution_fraction": spread,
        "minimum_channel_flow_ratio": float(np.min(channel_flows) / mean),
        "maximum_channel_flow_ratio": float(np.max(channel_flows) / mean),
        "inlet_ring_pressures": p[inlet0:outlet0],
        "outlet_ring_pressures": p[outlet0:sink],
        "source_pressure_relative_to_sink": float(p[source]),
        "port_area_each": float(A_port_each),
        "port_diameter": float(D_port),
        "plenum_cross_section_area": float(A_plenum),
        "plenum_hydraulic_diameter": float(D_plenum),
        "ring_friction_factor": float(f_ring),
        "edge_flows": flows_by_kind,
        "model": "nonlinear_two_annular_header_full_channel_network",
        "qualification_status":
            "one_dimensional_network_requires_3d_manifold_validation",
        "reference":
            "NASA SP-8087 regenerative-cooling hydraulics; "
            "Kang and Sun, JTHT 25 (2011)",
    }
