"""Reference CAD package for electric-pump feed hardware.

The pump sizing model solves a meanline, not a blade-to-blade design.  This
module therefore exports labeled reference geometry for layout, interfaces, and
trade review.  It deliberately does not claim production-ready blade surfaces.

Literature basis:
* NASA SP-8109, *Liquid Rocket Engine Centrifugal Flow Turbopumps*:
  centrifugal pump elements, specific speed/specific diameter, head and flow
  coefficients, impeller blade count/angles, diffuser/volute selection.
* NASA SP-8052, *Liquid Rocket Engine Turbopump Inducers*: inducer inlet eye,
  hub ratio, blade count, solidity, suction-specific-speed and NPSH screening.
* Lee et al. 2021 and Spiller et al. 2013: electric-pump package mass/energy
  closure and small-pump efficiency caution.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from raosim.export import _mesh_diagnostics, _write_stl, inspect_stl


_DEFAULT_MOTOR_PACKAGE_DENSITY = 2700.0
_DEFAULT_INVERTER_PACKAGE_DENSITY = 1200.0
_DEFAULT_BATTERY_PACKAGE_DENSITY = 750.0


def _finite(value, default=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _obj_value(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _obj_value_any(obj, names, default=None):
    """Look up the first present name (dataclass attr vs to_dict key)."""
    for name in names:
        value = _obj_value(obj, name)
        if value is not None:
            return value
    return default


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _dim(component, role, symbol, name, value, unit, source, note=""):
    return {
        "component": component,
        "role": role,
        "symbol": symbol,
        "name": name,
        "value_si": _clean(_finite(value)),
        "unit": unit,
        "source": source,
        "note": note,
    }


def _camber_wrap_deg(d1, d2, envelope) -> float | None:
    """Wrap angle of the log-spiral camber line implied by the envelope."""
    beta1 = _finite(_obj_value(envelope, "inlet_angle_deg"))
    beta2 = _finite(_obj_value(envelope, "outlet_angle_deg"))
    if beta1 is None or beta2 is None or not d1 or not d2:
        return None
    from raosim.pumps import impeller_blade_camber

    try:
        camber = impeller_blade_camber(0.5 * d1, 0.5 * d2, beta1, beta2)
    except ValueError:
        return None
    return math.degrees(camber[-1]["theta_rad"])


def _reference_stations(reference_geometry) -> dict[str, Any]:
    rows = _obj_value(reference_geometry, "meridional_profile", []) or []
    stations: dict[str, Any] = {}
    for row in rows:
        name = _obj_value(row, "station")
        if name:
            stations[str(name)] = row
    return stations


def _station_value(stations, station: str, key: str, role: str) -> float:
    value = _finite(_obj_value(stations.get(station), key))
    if value is None:
        raise ValueError(
            f"{role} pump reference geometry is missing meridional station "
            f"{station!r} ({key}); CAD layout must come from the meanline"
        )
    return value


def pump_reference_geometry(pump_result) -> dict[str, Any]:
    """Flatten solved pump hardware dimensions into a CAD package manifest.

    All dimensions are consumed from the meanline's exported
    ``PumpReferenceGeometry`` (single source of truth); nothing is re-derived
    here, so the CAD package cannot drift from the solved pump.  Lines whose
    pressure rise was never solved keep the honest ``not_sized`` status.
    """
    lines = _obj_value(pump_result, "lines", {}) or {}
    dims: list[dict[str, Any]] = []
    components: dict[str, Any] = {}

    for role, line in lines.items():
        ref = _obj_value(line, "reference_geometry")
        ind = _obj_value(line, "inducer")
        drive = _obj_value(line, "drive")
        if ref is None:
            components[role] = {
                "status": "not_sized",
                "reason": "tank/inlet pressure missing; pump rise not solved",
            }
            continue

        disk = _obj_value(ref, "impeller_disk", {}) or {}
        envelope = _obj_value(ref, "blade_envelope", {}) or {}
        helix = _obj_value(ref, "inducer_helix", {}) or {}
        vane_ring = _obj_value(ref, "diffuser_vane_ring", {}) or {}
        scroll = _obj_value(ref, "volute_scroll", {}) or {}
        shaft = _obj_value(ref, "shaft_datum", {}) or {}
        ports = _obj_value(ref, "ports", {}) or {}
        stations = _reference_stations(ref)

        d2 = _finite(_obj_value(disk, "outer_diameter_m"))
        d1 = _finite(_obj_value(disk, "eye_diameter_m"))
        b2 = _finite(_obj_value(disk, "outlet_width_m"))
        blade_thickness = _finite(
            _obj_value(envelope, "estimated_blade_thickness_m")
        )
        inlet_blade_thickness = _finite(
            _obj_value(envelope, "inlet_blade_thickness_m")
        )
        if None in (d2, d1, b2, blade_thickness):
            raise ValueError(
                f"{role} pump reference geometry is missing impeller disk or "
                "blade-envelope dimensions (D2/D1/b2/thickness)"
            )
        stages = int(_obj_value(disk, "stage_count", 1) or 1)
        blade_count = int(_obj_value(envelope, "blade_count", 0) or 0)
        # Axial layout is owned by the meanline's meridional stations.
        pump_axial_width = (
            _station_value(stations, "impeller_exit", "x_m", role)
            - _station_value(stations, "impeller_eye", "x_m", role)
        )

        dims += [
            _dim("impeller", role, "D2", "impeller outer diameter",
                 d2, "m", "NASA SP-8109 head coefficient"),
            _dim("impeller", role, "D1", "impeller inlet eye diameter",
                 d1, "m", "NASA SP-8109 flow/head sizing"),
            _dim("impeller", role, "b2", "impeller outlet width",
                 b2, "m", "NASA SP-8109 flow coefficient"),
            _dim("impeller", role, "t_b", "impeller blade thickness estimate",
                 blade_thickness, "m", "meanline blade envelope",
                 "spec thickness ratio with manufacturing floor"),
            _dim("impeller", role, "t_b1", "impeller inlet blade thickness",
                 inlet_blade_thickness, "m", "tapered meanline blade envelope",
                 "selected leading-edge manufacturing input"),
            _dim("impeller", role, "Z", "impeller blade count",
                 blade_count, "count",
                 "NASA SP-8109 fig. 16 minimum blade number (digitized)"),
            _dim("impeller", role, "Z_main", "full-length inlet blades",
                 _obj_value(envelope, "inlet_blade_count"), "count",
                 "SP-8109 inlet free-area practice"),
            _dim("impeller", role, "Z_split", "downstream splitter blades",
                 _obj_value(envelope, "splitter_blade_count"), "count",
                 "SP-8109 inlet/discharge blade-count separation"),
            _dim("impeller", role, "beta1",
                 "solved inlet blade metal angle",
                 _obj_value(envelope, "inlet_angle_deg"), "deg",
                 "meanline velocity triangle"),
            _dim("impeller", role, "beta2",
                 "solved outlet blade metal angle",
                 _obj_value(envelope, "outlet_angle_deg"), "deg",
                 "meanline velocity triangle"),
            _dim("impeller", role, "theta_c", "impeller camber wrap angle",
                 _camber_wrap_deg(d1, d2, envelope), "deg",
                 "log-spiral camber between solved velocity-triangle "
                 "angles (SP-8109 blade-geometry practice)"),
            _dim("impeller", role, "U2", "impeller tip speed",
                 _obj_value(disk, "tip_speed_m_s"), "m/s",
                 "NASA SP-8109 tip-speed screen"),
            _dim("pump", role, "N_stage", "centrifugal stage count",
                 stages, "count", "NASA SP-8109 staging screen"),
        ]

        role_components = {
            "impeller": {
                "outer_diameter_m": d2,
                "inlet_diameter_m": d1,
                "outlet_width_m": b2,
                "axial_width_m": pump_axial_width,
                "blade_thickness_m": blade_thickness,
                "inlet_blade_thickness_m": inlet_blade_thickness,
                "blade_count": blade_count,
                "inlet_blade_count": int(
                    _obj_value(envelope, "inlet_blade_count", blade_count)
                    or blade_count
                ),
                "splitter_blade_count": int(
                    _obj_value(envelope, "splitter_blade_count", 0) or 0
                ),
                "splitter_start_radius_fraction": _finite(
                    _obj_value(envelope, "splitter_start_radius_fraction")
                ),
                "inlet_blockage_fraction": _finite(
                    _obj_value(envelope, "inlet_blockage_fraction")
                ),
                "exit_blockage_fraction": _finite(
                    _obj_value(envelope, "exit_blockage_fraction")
                ),
                "inlet_blockage_limit": _finite(
                    _obj_value(envelope, "inlet_blockage_limit")
                ),
                "exit_blockage_limit": _finite(
                    _obj_value(envelope, "exit_blockage_limit")
                ),
                "legacy_screening_inlet_angle_deg": _finite(
                    _obj_value(envelope, "legacy_screening_inlet_angle_deg")
                ),
                "inlet_blade_angle_deg": _finite(
                    _obj_value(envelope, "inlet_angle_deg")
                ),
                "outlet_blade_angle_deg": _finite(
                    _obj_value(envelope, "outlet_angle_deg")
                ),
                "stages": stages,
            },
            "shaft": {
                "diameter_m": _finite(_obj_value(shaft, "diameter_m")),
                "span_m": _finite(_obj_value(shaft, "estimated_span_m")),
            },
            "ports": {
                "inlet_diameter_m": _finite(
                    _obj_value(_obj_value(ports, "inlet"), "diameter_m")
                ),
                "outlet_area_m2": _finite(
                    _obj_value(_obj_value(ports, "outlet"), "area_m2")
                ),
                "outlet_equivalent_diameter_m": _finite(
                    _obj_value(_obj_value(ports, "outlet"),
                               "equivalent_diameter_m")
                ),
            },
        }

        d_ind = _finite(_obj_value(helix, "diameter_m"))
        if d_ind is not None:
            hub_ratio = _finite(_obj_value(helix, "hub_ratio"))
            pitch = _finite(_obj_value(helix, "pitch_m"))
            wrap = _finite(_obj_value(helix, "wrap_angle_deg"))
            blade_count_i = int(_obj_value(helix, "blade_count", 0) or 0)
            if None in (hub_ratio, pitch, wrap):
                raise ValueError(
                    f"{role} pump inducer helix is missing hub ratio, pitch, "
                    "or wrap angle"
                )
            hub_d = hub_ratio * d_ind
            # Exact axial extent of a helix with the solved pitch and wrap.
            length = pitch * wrap / 360.0
            dims += [
                _dim("inducer", role, "D_ind", "inducer tip diameter",
                     d_ind, "m", "NASA SP-8052 inducer inlet eye"),
                _dim("inducer", role, "D_hub", "inducer hub diameter",
                     hub_d, "m", "NASA SP-8052 hub ratio"),
                _dim("inducer", role, "Z_ind", "inducer blade count",
                     blade_count_i, "count",
                     "NASA SP-8052 sec. 3.1.14 blade number"),
                _dim("inducer", role, "beta_t1",
                     "inducer inlet tip blade angle",
                     _obj_value(helix, "inlet_tip_blade_angle_deg"), "deg",
                     "NASA SP-8052 sec. 3.1.9 alpha/beta incidence ratio"),
                _dim("inducer", role, "beta_h1",
                     "inducer hub blade angle",
                     _obj_value(helix, "hub_blade_angle_deg"), "deg",
                     "NASA SP-8052 sec. 3.1.10 constant-lead helix"),
                _dim("inducer", role, "alpha_i",
                     "inducer incidence angle",
                     _obj_value(helix, "incidence_deg"), "deg",
                     "NASA SP-8052 sec. 3.1.9 alpha/beta incidence ratio"),
                _dim("inducer", role, "phi_1",
                     "inducer inlet tip flow coefficient",
                     _obj_value(helix, "inlet_flow_coefficient"), "ratio",
                     "meanline continuity at the inducer eye"),
                _dim("inducer", role, "t_le",
                     "inducer leading-edge thickness",
                     _obj_value(helix, "leading_edge_thickness_m"), "m",
                     "NASA SP-8052 sec. 2.1.6 J-2/F-1 edge practice"),
                _dim("inducer", role, "p_ind", "inducer helix pitch (lead)",
                     pitch, "m",
                     "NASA SP-8052 sec. 3.1.10 constant-lead helix"),
                _dim("inducer", role, "phi_wrap", "inducer helix wrap angle",
                     wrap, "deg",
                     "NASA SP-8052 sec. 3.1.15 cascade solidity"),
                _dim("inducer", role, "sigma", "inducer solidity",
                     _obj_value(ind, "solidity"), "ratio",
                     "NASA SP-8052 sec. 3.1.15 cascade solidity"),
                _dim("inducer", role, "Nss", "suction specific speed",
                     _obj_value(ind, "suction_specific_speed"), "ratio",
                     "NASA SP-8052 suction performance"),
            ]
            role_components["inducer"] = {
                "diameter_m": d_ind,
                "hub_diameter_m": hub_d,
                "length_m": length,
                "pitch_m": pitch,
                "blade_count": blade_count_i,
                "wrap_angle_deg": wrap,
                "inlet_tip_blade_angle_deg": _finite(
                    _obj_value(helix, "inlet_tip_blade_angle_deg")
                ),
                "hub_blade_angle_deg": _finite(
                    _obj_value(helix, "hub_blade_angle_deg")
                ),
                "leading_edge_thickness_m": _finite(
                    _obj_value(helix, "leading_edge_thickness_m")
                ),
                "shaft_fit_radial_clearance_m": _finite(
                    _obj_value(helix, "shaft_fit_radial_clearance_m")
                ),
                "hub_wall_thickness_m": _finite(
                    _obj_value(helix, "hub_wall_thickness_m")
                ),
            }

        channel = _obj_value(ref, "meridional_channel")
        if channel:
            role_components["meridional_channel"] = _clean(channel)
            dims.append(_dim(
                "impeller", role, "cm2/cm1",
                "meridional velocity ratio (exit/inlet)",
                _obj_value(channel, "cm_ratio"), "ratio",
                "NASA SP-8109 sec. 2.3.1.2 (1 to 1.5 x inlet)",
                _obj_value(channel, "cm_ratio_status", ""),
            ))

        balance = _obj_value(ref, "thrust_balance")
        if balance:
            role_components["thrust_balance"] = _clean(balance)
            holes = _obj_value(balance, "balance_holes", {}) or {}
            seal = _obj_value(balance, "shaft_seal_land", {}) or {}
            dims += [
                _dim("thrust_balance", role, "D_wr_hub",
                     "hub wear-ring diameter",
                     _obj_value(balance, "hub_wear_ring_diameter_m"), "m",
                     "NASA SP-8109 sec. 3.5.2.1 wear-ring thrust balance",
                     "equal-diameter neutral start, editable trim"),
                _dim("thrust_balance", role, "d_bh",
                     "impeller balance-hole diameter",
                     _obj_value(holes, "diameter_m"), "m",
                     "NASA SP-8109 sec. 3.5.2.1 (area = 4 x seal clearance)",
                     str(_obj_value(holes, "status", ""))),
                _dim("thrust_balance", role, "U_seal",
                     "shaft seal land face speed",
                     _obj_value(seal, "face_speed_m_s"), "m/s",
                     "seal face-speed screen",
                     str(_obj_value(seal, "status", ""))),
            ]

        vane_width = _finite(_obj_value(vane_ring, "vane_width_m"))
        if vane_width is not None:
            throat_area = _finite(_obj_value(vane_ring, "throat_area_m2"))
            vane_count = int(_obj_value(vane_ring, "vane_count", 0) or 0)
            volute_exit_area = _finite(_obj_value(scroll, "exit_area_m2"))
            inner = _station_value(stations, "impeller_exit", "radius_m", role)
            outer = _station_value(stations, "diffuser_exit", "radius_m", role)
            dims += [
                _dim("diffuser_volute", role, "A_th",
                     "diffuser throat area", throat_area, "m2",
                     "NASA SP-8109 diffusion system"),
                _dim("diffuser_volute", role, "b_v",
                     "diffuser vane/flow width", vane_width, "m",
                     "NASA SP-8109 diffusion system"),
                _dim("diffuser_volute", role, "Z_v",
                     "diffuser vane count", vane_count, "count",
                     "NASA SP-8109 vaned/vaneless diffuser selection"),
                _dim("diffuser_volute", role, "A_vol",
                     "volute exit area", volute_exit_area,
                     "m2", "NASA SP-8109 collecting volute"),
            ]
            role_components["diffuser_volute"] = {
                "inner_radius_m": inner,
                "outer_radius_m": outer,
                "axial_width_m": vane_width,
                "vane_count": vane_count,
                "vane_angle_deg": _finite(
                    _obj_value(vane_ring, "vane_angle_deg")
                ),
                "selection": _obj_value(vane_ring, "selection", "diffuser"),
                "volute_exit_area_m2": volute_exit_area,
                "area_schedule": _obj_value(scroll, "area_schedule"),
                "casing_inner_radius_m": _finite(
                    _obj_value(scroll, "casing_inner_radius_m")
                ),
                "casing_wall_thickness_m": _finite(
                    _obj_value(scroll, "casing_wall_thickness_m")
                ),
                "design_pressure_pa": _finite(
                    _obj_value(scroll, "design_pressure_pa")
                ),
                "wall_sizing_model": _obj_value(
                    scroll, "wall_sizing_model"
                ),
                "split_casing_joint": _clean(
                    _obj_value(scroll, "split_casing_joint", {}) or {}
                ),
            }

        if drive is not None:
            m_motor = max(_finite(
                _obj_value_any(drive, ("motor_mass", "motor_mass_kg")), 0.0
            ), 0.0)
            m_inv = max(_finite(
                _obj_value_any(drive, ("inverter_mass", "inverter_mass_kg")),
                0.0,
            ), 0.0)
            motor_d = _cylinder_diameter_from_mass(m_motor, _DEFAULT_MOTOR_PACKAGE_DENSITY)
            inverter_side = (m_inv / _DEFAULT_INVERTER_PACKAGE_DENSITY) ** (1.0 / 3.0) if m_inv > 0.0 else 0.0
            dims += [
                _dim("motor", role, "m_motor", "motor mass estimate",
                     m_motor, "kg", "electric pump power density"),
                _dim("motor", role, "D_motor",
                     "motor package reference diameter",
                     motor_d, "m", "package density placeholder",
                     "layout placeholder, not a selected motor drawing"),
                _dim("inverter", role, "m_inverter", "inverter mass estimate",
                     m_inv, "kg", "electric pump inverter power density"),
            ]
            role_components["motor"] = {
                "diameter_m": motor_d,
                "length_m": 1.2 * motor_d if motor_d > 0.0 else 0.0,
                "mass_kg": m_motor,
            }
            role_components["inverter"] = {
                "box_m": [1.4 * inverter_side, inverter_side, 0.45 * inverter_side],
                "mass_kg": m_inv,
            }

        components[role] = role_components

    batt = _obj_value(pump_result, "battery")
    battery_component = None
    if batt is not None:
        mass = max(_finite(_obj_value_any(batt, ("mass", "mass_kg")), 0.0), 0.0)
        volume = mass / _DEFAULT_BATTERY_PACKAGE_DENSITY if mass > 0.0 else 0.0
        # 2.4:1.2:0.6 package aspect ratio.
        base = (volume / max(2.4 * 1.2 * 0.6, 1e-12)) ** (1.0 / 3.0) if volume > 0.0 else 0.0
        battery_component = {
            "box_m": [2.4 * base, 1.2 * base, 0.6 * base],
            "mass_kg": mass,
        }
        dims += [
            _dim("battery", "shared", "m_pack", "battery pack mass",
                 mass, "kg", "Lee et al. electric-pump cycle mass closure"),
            _dim("battery", "shared", "V_pack", "battery bus voltage",
                 _obj_value_any(batt, ("voltage", "voltage_v")), "V",
                 "shared electric-pump bus"),
        ]

    return {
        "model": "electric_pump_reference_cad_v1",
        "qualification_status": "reference_geometry_not_hardware_qualified",
        "axes": (
            "+Z is the pump shaft axis; XY is the radial pump plane; "
            "dimensions are SI metres in JSON/CSV, CAD kernels may use mm."
        ),
        "literature_basis": {
            "NASA SP-8109": (
                "Centrifugal pump elements, specific-speed/specific-diameter "
                "classification, head/flow coefficients, impeller blade "
                "geometry, diffuser and volute selection."
            ),
            "NASA SP-8052": (
                "Inducer inlet eye, hub ratio, blade number, cascade solidity, "
                "suction-specific-speed and NPSH screening."
            ),
            "Lee et al. 2021": (
                "Electric-pump motor/battery mass and 2-D impeller layout "
                "comparison basis."
            ),
            "Spiller et al. 2013": (
                "Small electric-pump efficiency caution; use real pump curves "
                "before hardware decisions."
            ),
        },
        "components": components,
        "battery": battery_component,
        "dimensions": dims,
    }


def _cylinder_diameter_from_mass(mass: float, density: float) -> float:
    if mass <= 0.0 or density <= 0.0:
        return 0.0
    volume = mass / density
    return (4.0 * volume / (1.2 * math.pi)) ** (1.0 / 3.0)


def write_pump_parameters_json(geom: dict[str, Any], path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(_clean(geom), indent=2) + "\n", encoding="utf-8")
    return path


def write_pump_dimensions_csv(geom: dict[str, Any], path) -> Path:
    path = Path(path)
    fields = ["component", "role", "symbol", "name", "value_si", "unit",
              "source", "note"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in geom["dimensions"]:
            writer.writerow(row)
    return path


def _tri(a, b, c):
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    mag = math.sqrt(sum(v * v for v in n))
    normal = tuple(v / mag for v in n) if mag > 0.0 else (0.0, 0.0, 0.0)
    return normal, a, b, c


def _annular_cylinder(radius_outer, radius_inner, z0, z1, n=64):
    tris = []
    ro = float(radius_outer)
    ri = max(0.0, float(radius_inner))
    # One welded vertex ring per edge circle, indexed so the closing segment
    # reuses column 0 exactly — float drift at the 2π seam would otherwise
    # split vertices and leave boundary edges in the mesh gate.
    ring = [
        (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]
    outer0 = [(ro * c, ro * s, z0) for c, s in ring]
    outer1 = [(ro * c, ro * s, z1) for c, s in ring]
    inner0 = [(ri * c, ri * s, z0) for c, s in ring]
    inner1 = [(ri * c, ri * s, z1) for c, s in ring]
    center0 = (0.0, 0.0, z0)
    center1 = (0.0, 0.0, z1)
    for i in range(n):
        j = (i + 1) % n
        o00, o01 = outer0[i], outer0[j]
        o10, o11 = outer1[i], outer1[j]
        tris.extend([_tri(o00, o11, o10), _tri(o00, o01, o11)])
        if ri > 0.0:
            i00, i01 = inner0[i], inner0[j]
            i10, i11 = inner1[i], inner1[j]
            tris.extend([_tri(i00, i10, i11), _tri(i00, i11, i01)])
            tris.extend([_tri(i10, o10, o11), _tri(i10, o11, i11)])
            tris.extend([_tri(i00, o01, o00), _tri(i00, i01, o01)])
        else:
            tris.extend([_tri(center1, o10, o11), _tri(center0, o01, o00)])
    return tris


def _box(dx, dy, dz, center=(0.0, 0.0, 0.0), angle=0.0):
    cx, cy, cz = center
    hx, hy, hz = 0.5 * dx, 0.5 * dy, 0.5 * dz
    raw = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    ca, sa = math.cos(angle), math.sin(angle)

    def xform(p):
        x, y, z = p
        return (cx + ca * x - sa * y, cy + sa * x + ca * y, cz + z)

    v = [xform(p) for p in raw]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return [_tri(v[i], v[j], v[k]) for i, j, k in faces]


def _radial_blade_boxes(count, r_inner, r_outer, thickness, height, z_center, twist=0.0):
    tris = []
    r_mid = 0.5 * (r_inner + r_outer)
    length = max(r_outer - r_inner, 1e-6)
    for i in range(max(int(count), 1)):
        angle = 2.0 * math.pi * i / max(int(count), 1) + twist
        center = (r_mid * math.cos(angle), r_mid * math.sin(angle), z_center)
        tris.extend(_box(length, thickness, height, center=center, angle=angle))
    return tris


_PUMP_STEP_UNAVAILABLE = (
    "pump STEP export is unavailable: the faceted pseudo-STEP writer was "
    "removed (import-hostile, not a valid B-rep exchange body) and the "
    "CadQuery B-rep pump path has not landed yet "
    "(docs/PUMP_CAD_IMPLEMENTATION_PLAN.md Phase 1); export 'stl' instead"
)


def _write_part(path: Path, triangles: list, fmt: str, metadata: dict[str, Any],
                *, allow_open_mesh: bool = False) -> dict:
    if fmt == "step":
        raise ValueError(_PUMP_STEP_UNAVAILABLE)
    if fmt != "stl":
        raise ValueError(f"unsupported pump CAD format {fmt!r}")
    # Same gate as export_stl applies to the wall: refuse open, inconsistently
    # wound, or non-positive-volume meshes before anything reaches disk.
    gate = _mesh_diagnostics(triangles)
    if not (gate["watertight"] and gate["signed_volume_m3"] > 0.0):
        if not allow_open_mesh:
            raise RuntimeError(
                f"pump part mesh validation failed for {path.name}: "
                f"{gate['boundary_edge_count']} boundary edges, "
                f"{gate['nonmanifold_edge_count']} non-manifold edges, "
                f"{gate['inconsistent_winding_edge_count']} inconsistently "
                "wound edges, "
                f"{gate['degenerate_triangle_count']} degenerate triangles, "
                f"signed volume {gate['signed_volume_m3']:.6g} m^3 "
                "(pass --allow-open-pump-mesh to export anyway)"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_stl(path, triangles)
    diag = inspect_stl(path)
    info = {"path": str(path), "format": "stl", "diagnostics": diag}
    if not diag["watertight"]:
        info["mesh_gate"] = "waived_not_watertight"
    return info


def _impeller_mesh(comp):
    d2 = comp["outer_diameter_m"]
    d1 = min(comp["inlet_diameter_m"], 0.90 * d2)
    width = comp["axial_width_m"]
    blade_count = comp["blade_count"]
    r2 = 0.5 * d2
    r1 = max(0.5 * d1, 0.16 * r2)
    tris = _annular_cylinder(r2, r1, -0.5 * width, 0.5 * width)
    tris += _annular_cylinder(0.32 * r2, 0.0, -0.75 * width, 0.75 * width)
    blade_thickness = comp["blade_thickness_m"]
    tris += _radial_blade_boxes(
        blade_count, r1, 0.94 * r2, blade_thickness, 1.35 * width, 0.0,
        twist=math.radians(12.0),
    )
    return tris


def _inducer_mesh(comp):
    d = comp["diameter_m"]
    hub = comp["hub_diameter_m"]
    length = comp["length_m"]
    r_tip = 0.5 * d
    r_hub = 0.5 * hub
    tris = _annular_cylinder(r_hub, 0.0, -0.5 * length, 0.5 * length)
    solved_thickness = comp.get("leading_edge_thickness_m")
    if solved_thickness is None or float(solved_thickness) <= 0.0:
        raise ValueError(
            "inducer leading-edge thickness is missing from the solved "
            "pump manifest; refusing to substitute a CAD-only diameter ratio"
        )
    blade_thickness = float(solved_thickness)
    blade_radial = max(r_tip - r_hub, 1e-6)
    for i in range(max(int(comp["blade_count"]), 1)):
        angle = 2.0 * math.pi * i / max(int(comp["blade_count"]), 1)
        tris += _box(
            blade_radial,
            blade_thickness,
            length,
            center=(
                0.5 * (r_hub + r_tip) * math.cos(angle),
                0.5 * (r_hub + r_tip) * math.sin(angle),
                0.0,
            ),
            angle=angle + math.radians(25.0),
        )
    return tris


def _diffuser_volute_mesh(comp):
    inner = comp["inner_radius_m"]
    outer = comp["outer_radius_m"]
    width = comp["axial_width_m"]
    tris = _annular_cylinder(outer, inner, -0.5 * width, 0.5 * width)
    vane_count = int(comp.get("vane_count") or 0)
    if vane_count > 0:
        pitch = 2.0 * math.pi * 0.5 * (inner + outer) / vane_count
        tris += _radial_blade_boxes(
            vane_count, inner, outer, 0.22 * pitch, 1.15 * width, 0.0,
            twist=math.radians(20.0),
        )
    volute_area = comp.get("volute_exit_area_m2") or 0.0
    if volute_area > 0.0:
        side = math.sqrt(volute_area)
        tris += _box(
            2.0 * side,
            side,
            max(width, side),
            center=(outer + side, 0.0, 0.0),
            angle=0.0,
        )
    return tris


def _motor_mesh(comp):
    d = comp["diameter_m"]
    length = comp["length_m"]
    if d <= 0.0 or length <= 0.0:
        return []
    return _annular_cylinder(0.5 * d, 0.12 * d, -0.5 * length, 0.5 * length)


def _box_component_mesh(comp):
    dims = comp.get("box_m") or [0.0, 0.0, 0.0]
    if min(dims) <= 0.0:
        return []
    return _box(float(dims[0]), float(dims[1]), float(dims[2]))


def _part_meshes_for_role(role_components: dict[str, Any]) -> dict[str, list]:
    meshes = {}
    if "impeller" in role_components:
        meshes["impeller"] = _impeller_mesh(role_components["impeller"])
    if "inducer" in role_components:
        meshes["inducer"] = _inducer_mesh(role_components["inducer"])
    if "diffuser_volute" in role_components:
        meshes["diffuser_volute"] = _diffuser_volute_mesh(
            role_components["diffuser_volute"]
        )
    if "motor" in role_components:
        meshes["motor"] = _motor_mesh(role_components["motor"])
    if "inverter" in role_components:
        meshes["inverter"] = _box_component_mesh(role_components["inverter"])
    return {name: tris for name, tris in meshes.items() if tris}


def _mesh_bounds(triangles: list) -> dict[str, float]:
    vertices = [point for _normal, *points in triangles for point in points]
    if not vertices:
        raise ValueError("cannot bound an empty pump mesh")
    return {
        "xmin": min(v[0] for v in vertices),
        "xmax": max(v[0] for v in vertices),
        "ymin": min(v[1] for v in vertices),
        "ymax": max(v[1] for v in vertices),
        "zmin": min(v[2] for v in vertices),
        "zmax": max(v[2] for v in vertices),
    }


def _translate_mesh(triangles: list, dx: float, dy: float, dz: float) -> list:
    def move(point):
        return (point[0] + dx, point[1] + dy, point[2] + dz)

    return [
        _tri(move(a), move(b), move(c))
        for _normal, a, b, c in triangles
    ]


def export_pump_package(
    pump_result,
    out_dir,
    *,
    cad: str = "parts",
    cad_format: str = "stl",
    allow_open_mesh: bool = False,
) -> dict[str, Any]:
    """Write pump reference parameters and optional part CAD."""
    if cad and cad != "none":
        if cad_format in ("step", "both"):
            raise ValueError(_PUMP_STEP_UNAVAILABLE)
        if cad_format != "stl":
            raise ValueError(f"unsupported pump CAD format {cad_format!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = pump_reference_geometry(pump_result)
    files: dict[str, str] = {}
    notes: list[str] = [
        "Legacy triangle-mesh CAD is schematic_only: its blade twists and "
        "volute solids are not the solved B-rep geometry.",
        "Use pump_cad_brep export for solved beta1, main/splitter blades, "
        "annular hubs, connected flow voids, and separable casing halves.",
        "Use blade-to-blade CFD, rotordynamics, structural FEA, seals/bearings, "
        "vendor motor maps, and cold-flow tests before hardware release.",
    ]
    files["parameters_json"] = str(write_pump_parameters_json(
        geom, out_dir / "pump_parameters.json"
    ))
    files["dimensions_csv"] = str(write_pump_dimensions_csv(
        geom, out_dir / "pump_dimensions.csv"
    ))

    if not cad or cad == "none":
        return {"dir": str(out_dir), "files": files, "geometry": geom,
                "cad_fidelity": "parameters_only_no_solid",
                "cold_flow_release_ready": False,
                "notes": notes}

    cad_dir = out_dir / "pump_parts"
    assembly_mesh: list = []
    assembly_placements: list[dict[str, Any]] = []
    assembly_x_cursor = 0.0
    exploded_gap_m = 0.010
    cad_files: dict[str, str] = {}
    cad_diagnostics: dict[str, Any] = {}
    for role, components in geom["components"].items():
        if components.get("status") == "not_sized":
            notes.append(f"{role} pump CAD skipped: {components['reason']}")
            continue
        for part, triangles in _part_meshes_for_role(components).items():
            path = cad_dir / f"{role}_{part}.stl"
            info = _write_part(
                path,
                triangles,
                "stl",
                metadata={
                    "role": role,
                    "component": part,
                    "model": geom["model"],
                    "qualification_status": geom["qualification_status"],
                },
                allow_open_mesh=allow_open_mesh,
            )
            key = f"{role}_{part}_stl"
            cad_files[key] = str(path)
            cad_diagnostics[key] = info
            bounds = _mesh_bounds(triangles)
            dx = assembly_x_cursor - bounds["xmin"]
            assembly_mesh.extend(_translate_mesh(triangles, dx, 0.0, 0.0))
            placed_xmax = bounds["xmax"] + dx
            assembly_placements.append({
                "role": role,
                "component": part,
                "translation_m": [dx, 0.0, 0.0],
                "source_bounds_m": bounds,
            })
            assembly_x_cursor = placed_xmax + exploded_gap_m

    if geom.get("battery"):
        battery_mesh = _box_component_mesh(geom["battery"])
        if battery_mesh:
            path = cad_dir / "shared_battery_pack.stl"
            info = _write_part(
                path,
                battery_mesh,
                "stl",
                metadata={
                    "role": "shared",
                    "component": "battery_pack",
                    "model": geom["model"],
                    "qualification_status": geom["qualification_status"],
                },
                allow_open_mesh=allow_open_mesh,
            )
            cad_files["shared_battery_pack_stl"] = str(path)
            cad_diagnostics["shared_battery_pack_stl"] = info
            bounds = _mesh_bounds(battery_mesh)
            dx = assembly_x_cursor - bounds["xmin"]
            assembly_mesh.extend(_translate_mesh(
                battery_mesh, dx, 0.0, 0.0
            ))
            assembly_placements.append({
                "role": "shared",
                "component": "battery_pack",
                "translation_m": [dx, 0.0, 0.0],
                "source_bounds_m": bounds,
            })
            assembly_x_cursor = bounds["xmax"] + dx + exploded_gap_m

    if cad in ("reference", "parts", "auto") and assembly_mesh:
        path = out_dir / "pump_reference_assembly.stl"
        info = _write_part(
            path,
            assembly_mesh,
            "stl",
            metadata={
                "component": "pump_reference_assembly",
                "model": geom["model"],
                "qualification_status": geom["qualification_status"],
            },
            allow_open_mesh=allow_open_mesh,
        )
        cad_files["reference_assembly_stl"] = str(path)
        cad_diagnostics["reference_assembly_stl"] = info
        layout_path = out_dir / "pump_reference_assembly_layout.json"
        layout_path.write_text(json.dumps({
            "schema": "raosim.pump_exploded_layout.v1",
            "artifact": path.name,
            "status": "noninterfering_exploded_reference_not_operating_assembly",
            "numeric_coordinate_unit": "m",
            "minimum_x_gap_m": exploded_gap_m,
            "placements": assembly_placements,
            "hardware_qualified": False,
            "note": (
                "The legacy STL is an exploded inspection layout. Use the "
                "OpenCascade pump B-rep package for shaft-aligned mechanical "
                "assembly and interference/clearance gates."
            ),
        }, indent=2) + "\n", encoding="utf-8")
        cad_files["reference_assembly_layout_json"] = str(layout_path)

    files.update(cad_files)
    return {
        "dir": str(out_dir),
        "files": files,
        "geometry": geom,
        "cad_diagnostics": cad_diagnostics,
        "cad_fidelity": "schematic_only_not_meanline_faithful",
        "cold_flow_release_ready": False,
        "hardware_qualified": False,
        "notes": notes,
    }
