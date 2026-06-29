"""injector_cad.py - named-body pintle STEP/STL export.

Builds a CadQuery assembly of the sized pintle with the named bodies the design
brief calls for (faceplate, hollow pintle body, tip, axial annulus, radial slot
network, fuel/oxidizer manifolds, optional movable sleeve, igniter interface,
regen-coolant outlet).  STEP is authoritative (it carries the named assembly);
per-body STLs are emitted for printing.  Geometry is the SIZED geometry from
:mod:`raosim.injector`; it is a preliminary CAD schematic, not a drawing-ready
part.

Axes: axial = +Z (injector face at Z=0, chamber downstream Z>0, manifolds /
faceplate behind at Z<0); radial in XY.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path


def cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
    except Exception:
        return False
    return True


def _cq():
    try:
        import cadquery as cq
    except Exception as exc:
        raise RuntimeError(
            "pintle STEP export requires CadQuery/OpenCascade (pip install "
            "cadquery)"
        ) from exc
    return cq


def _f(value, default=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _positive(value, default):
    v = _f(value)
    return float(v) if v is not None and v > 0.0 else float(default)


def _count(value, default=0):
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return int(default)


def _clean_json(value):
    """Map dataclasses and non-finite floats to JSON-safe values."""
    if is_dataclass(value):
        return _clean_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _mechanical_spec(spec):
    if spec is not None and getattr(spec, "mechanical", None) is not None:
        return spec.mechanical
    from raosim.injector import PintleMechanicalSpec
    return PintleMechanicalSpec()


def _manufacturing_spec(spec):
    return getattr(spec, "manufacturing", None) if spec is not None else None


def _geometry_spec(spec):
    return getattr(spec, "geometry", None) if spec is not None else None


def _role_mechanical_inputs(mech, role):
    if role == "fuel":
        return {
            "inlet_count": mech.fuel_inlet_count,
            "inlet_diameter": mech.fuel_inlet_diameter,
            "inlet_angle": mech.fuel_inlet_angle,
            "inlet_fitting": mech.fuel_inlet_fitting,
            "manifold_width": mech.fuel_manifold_width,
            "manifold_depth": mech.fuel_manifold_depth,
        }
    return {
        "inlet_count": mech.oxidizer_inlet_count,
        "inlet_diameter": mech.oxidizer_inlet_diameter,
        "inlet_angle": mech.oxidizer_inlet_angle,
        "inlet_fitting": mech.oxidizer_inlet_fitting,
        "manifold_width": mech.oxidizer_manifold_width,
        "manifold_depth": mech.oxidizer_manifold_depth,
    }


def resolve_machined_pintle_layout(inj, spec=None) -> dict:
    """Resolve preliminary machined-pintle dimensions from hydraulic sizing.

    The calculations are intentionally first-order and traceable: hydraulic
    flow areas come from the injector model, manifold/inlet cross-sections are
    sized from velocity limits, and manufacturing details are governed by tool
    radius, land, web, and tolerance floors.  The report marks the result as
    preliminary until cold-flow validation supplies distribution data.
    """
    mech = _mechanical_spec(spec)
    mfg = _manufacturing_spec(spec)
    geo = _geometry_spec(spec)

    min_feature = _positive(getattr(mfg, "min_feature", None), 3.0e-4)
    tol = _positive(mech.tolerance, getattr(mfg, "concentricity_tolerance", 5.0e-5))
    min_tool = _positive(mech.min_tool_diameter, max(min_feature, 3.0e-4))
    tool_radius = 0.5 * min_tool
    min_corner = _positive(mech.min_corner_radius, tool_radius)

    Dp = float(inj.pintle_diameter)
    Rp = 0.5 * Dp
    gap = _positive(inj.annulus.detail.get("gap"), 0.1 * Rp)
    Ro = 0.5 * _positive(inj.annulus.detail.get("outer_diameter"), Dp + 2.0 * gap)
    slot_w = _positive(inj.slots.detail.get("slot_width"), min_feature)
    slot_h = _positive(inj.slots.detail.get("slot_height"), slot_w)
    slot_depth_source = (
        mech.slot_depth
        if mech.slot_depth is not None else
        getattr(geo, "slot_depth", None)
        if geo is not None and getattr(geo, "slot_depth", None) is not None else
        inj.slots.detail.get("slot_depth")
    )

    base_wall = max(0.25 * Rp, 1.0e-3)
    requested_slot_depth = _positive(slot_depth_source, inj.slots.hydraulic_diameter)
    pintle_wall = min(_positive(mech.pintle_wall_thickness, base_wall), 0.80 * Rp)
    # A cut-through STEP must actually breach the pintle wall.  Preserve the
    # requested value in the report and use a cut depth that reaches the bore.
    slot_cut_depth = max(requested_slot_depth, pintle_wall + 2.0 * tol)
    inner_pintle_radius = max(Rp - pintle_wall, min_feature)

    sleeve_wall = _positive(mech.sleeve_wall_thickness, max(0.4 * gap, 0.5e-3))
    Rs = Ro + sleeve_wall
    body_len = _positive(getattr(geo, "body_length", None) if geo else None,
                         3.0 * Dp)
    body_straight = max(body_len - Rp, 0.5 * body_len)
    annulus_length = _positive(mech.annulus_length, max(body_len, 4.0 * gap))

    face_t_input = _f(mech.faceplate_thickness)
    face_t_default = (
        getattr(geo, "face_thickness", None)
        if geo is not None and getattr(geo, "face_thickness", None) is not None
        else max(0.4 * Dp, 6.0 * min_tool)
    )
    face_t = _positive(mech.faceplate_thickness, face_t_default)
    face_t = max(face_t, 0.4 * Dp, 6.0 * min_tool)

    roles = {}
    for role in ("fuel", "oxidizer"):
        s = inj.streams[role]
        rmi = _role_mechanical_inputs(mech, role)
        inlet_count = _count(rmi["inlet_count"], 2)
        inlet_area_req = (
            s.mdot / max(inj.feed[role].density * mech.inlet_velocity_limit, 1e-12)
            if inlet_count > 0 else 0.0
        )
        inlet_d = _positive(
            rmi["inlet_diameter"],
            math.sqrt(4.0 * inlet_area_req / (math.pi * max(inlet_count, 1)))
            if inlet_count > 0 else min_tool,
        )
        inlet_d = max(inlet_d, min_tool)

        if inj.manifold is not None and role in inj.manifold.streams:
            plenum_area = _f(inj.manifold.streams[role].plenum_area)
        else:
            plenum_area = None
        manifold_area_req = max(
            2.0 * s.area,
            s.mdot / max(inj.feed[role].density * mech.manifold_velocity_limit, 1e-12),
            plenum_area if plenum_area is not None and plenum_area > 0.0 else 0.0,
        )
        depth = _positive(rmi["manifold_depth"], max(3.0 * min_tool,
                                                     math.sqrt(manifold_area_req)))
        width = _positive(rmi["manifold_width"], manifold_area_req / max(depth, 1e-12))
        width = max(width, 2.0 * min_tool)
        depth = max(depth, 2.0 * min_tool)
        area = width * depth
        roles[role] = {
            "feeds": s.geometry,
            "inlet_count": inlet_count,
            "inlet_diameter_m": inlet_d,
            "inlet_angle_deg": float(rmi["inlet_angle"]),
            "inlet_fitting": str(rmi["inlet_fitting"] or "layout_only"),
            "inlet_area_total_m2": (
                inlet_count * math.pi * (0.5 * inlet_d) ** 2
            ),
            "inlet_velocity_m_s": (
                s.mdot / max(inj.feed[role].density
                             * inlet_count * math.pi * (0.5 * inlet_d) ** 2,
                             1e-12)
                if inlet_count > 0 else float("inf")
            ),
            "manifold_width_m": width,
            "manifold_depth_m": depth,
            "manifold_cross_section_area_m2": area,
            "manifold_velocity_m_s": (
                s.mdot / max(inj.feed[role].density * area, 1e-12)
            ),
            "stream_area_m2": s.area,
            "stream_velocity_m_s": s.velocity,
        }

    max_manifold_depth = max(d["manifold_depth_m"] for d in roles.values())
    back_web = max(2.0 * min_tool, 4.0 * tol)
    requested_face_t = face_t_input if face_t_input is not None else face_t
    min_face_t = max(max_manifold_depth + back_web, 0.4 * Dp, 6.0 * min_tool)
    face_t = max(face_t, min_face_t)

    land = _positive(mech.gasket_land_width, max(2.5 * min_tool, 4.0 * tol))
    manifold_order = sorted(
        roles,
        key=lambda r: 0 if roles[r]["feeds"] == "annulus" else 1,
    )
    next_inner = Rs + land
    for role in manifold_order:
        width = roles[role]["manifold_width_m"]
        inner = next_inner
        outer = inner + width
        roles[role]["manifold_inner_radius_m"] = inner
        roles[role]["manifold_outer_radius_m"] = outer
        roles[role]["manifold_mean_radius_m"] = 0.5 * (inner + outer)
        roles[role]["manifold_volume_m3"] = (
            math.pi * (outer * outer - inner * inner)
            * roles[role]["manifold_depth_m"]
        )
        next_inner = outer + land

    for role, d in roles.items():
        transfer_count = max(2, d["inlet_count"], 4 if d["feeds"] == "annulus" else 2)
        transfer_d = max(
            min_tool,
            math.sqrt(4.0 * d["stream_area_m2"] / (math.pi * transfer_count)),
        )
        d["transfer_count"] = transfer_count
        d["transfer_diameter_m"] = transfer_d
        d["transfer_angle_deg"] = d["inlet_angle_deg"] + 0.5 * (
            360.0 / max(transfer_count, 1)
        )
        d["transfer_z_m"] = -face_t + 0.5 * d["manifold_depth_m"]

    bolt_count = _count(mech.bolt_count, 8)
    bolt_hole = _positive(mech.bolt_hole_diameter, max(2.5 * min_tool, 3.0e-3))
    outermost_manifold = max(d["manifold_outer_radius_m"] for d in roles.values())
    bolt_circle_requested = _f(mech.bolt_circle_diameter)
    bolt_circle_min = 2.0 * (outermost_manifold + land + 0.5 * bolt_hole)
    face_od_min = max(
        2.0 * inj.chamber_radius,
        2.0 * (next_inner + 2.0 * land + bolt_hole),
        2.0 * (Rs + 4.0 * land),
    )
    bolt_circle = _positive(
        mech.bolt_circle_diameter,
        max(bolt_circle_min, 0.78 * face_od_min),
    )
    bolt_circle = max(bolt_circle, bolt_circle_min)
    face_od_default = (
        getattr(geo, "face_od", None)
        if geo is not None and getattr(geo, "face_od", None) is not None
        else face_od_min
    )
    face_od = _positive(mech.faceplate_outer_diameter, face_od_default)
    face_od = max(face_od, face_od_min, bolt_circle + 4.0 * bolt_hole)
    face_R = 0.5 * face_od

    igniter_d = _positive(mech.igniter_port_diameter, max(0.5 * inner_pintle_radius, min_tool))
    igniter_d = min(igniter_d, max(2.0 * (inner_pintle_radius - min_feature), min_tool))
    igniter_depth = _positive(mech.igniter_port_depth, face_t + body_len)

    o_width = _positive(mech.o_ring_groove_width, max(2.0 * min_tool, 1.5e-3))
    o_depth = _positive(mech.o_ring_groove_depth, max(0.75 * min_tool, 7.5e-4))
    seal_center = min(
        max(next_inner + land + 0.5 * o_width, Rs + 2.0 * land),
        max(face_R - land - 0.5 * o_width, Rs + 2.0 * land),
    )

    slot_corner = _positive(mech.slot_corner_radius, min_corner)
    max_slot_corner = 0.49 * min(slot_w, slot_h)
    slot_corner = min(slot_corner, max_slot_corner)

    gates = []

    def gate(name, ok, detail, warn=False):
        gates.append({
            "name": name,
            "status": "pass" if ok else ("warn" if warn else "fail"),
            "detail": detail,
        })

    gate(
        "slot_cut_through",
        slot_cut_depth >= pintle_wall + tol,
        f"slot Boolean depth {slot_cut_depth*1e3:.3f} mm vs pintle wall "
        f"{pintle_wall*1e3:.3f} mm plus tolerance {tol*1e3:.3f} mm",
    )
    gate(
        "slot_corner_radius",
        slot_corner >= min_corner or max_slot_corner < min_corner,
        f"slot corner radius {slot_corner*1e3:.3f} mm; tool-rule target "
        f"{min_corner*1e3:.3f} mm; geometric max {max_slot_corner*1e3:.3f} mm",
        warn=slot_corner < min_corner,
    )
    gate(
        "minimum_tool_diameter",
        slot_w >= min_tool and gap >= min_tool,
        f"slot width {slot_w*1e3:.3f} mm and annulus gap {gap*1e3:.3f} mm vs "
        f"minimum tool {min_tool*1e3:.3f} mm",
    )
    gates.append({
        "name": "faceplate_manifold_pocket_depth",
        "status": "pass" if requested_face_t >= min_face_t else "warn",
        "detail": (
            f"resolved faceplate thickness {face_t*1e3:.3f} mm covers max "
            f"manifold pocket {max_manifold_depth*1e3:.3f} mm plus back-web "
            f"{back_web*1e3:.3f} mm"
            + (
                f"; requested {requested_face_t*1e3:.3f} mm was increased"
                if requested_face_t < min_face_t else ""
            )
        ),
    })
    if bolt_count > 0:
        bolt_land = 0.5 * (face_od - bolt_circle) - 0.5 * bolt_hole
        bolt_manifold_clearance = (
            0.5 * bolt_circle - 0.5 * bolt_hole - outermost_manifold
        )
        gate(
            "bolt_edge_land",
            bolt_land >= land,
            f"outer bolt land {bolt_land*1e3:.3f} mm vs land rule "
            f"{land*1e3:.3f} mm",
        )
        gates.append({
            "name": "bolt_manifold_clearance",
            "status": (
                "pass"
                if bolt_circle_requested is None or bolt_circle_requested >= bolt_circle_min
                else "warn"
            ),
            "detail": (
                f"bolt-hole inner edge clears outer manifold by "
                f"{bolt_manifold_clearance*1e3:.3f} mm vs land rule "
                f"{land*1e3:.3f} mm"
                + (
                    f"; requested BCD {bolt_circle_requested*1e3:.3f} mm was "
                    f"increased to {bolt_circle*1e3:.3f} mm"
                    if bolt_circle_requested is not None
                    and bolt_circle_requested < bolt_circle_min else ""
                )
            ),
        })
    else:
        gates.append({
            "name": "bolt_pattern",
            "status": "warn",
            "detail": "no bolt holes requested; sealed joint clamp model is incomplete",
        })
    for role, d in roles.items():
        gate(
            f"{role}_inlet_velocity",
            d["inlet_velocity_m_s"] <= mech.inlet_velocity_limit,
            f"{role} inlet velocity {d['inlet_velocity_m_s']:.2f} m/s vs limit "
            f"{mech.inlet_velocity_limit:.2f} m/s",
            warn=True,
        )
        gate(
            f"{role}_manifold_velocity",
            d["manifold_velocity_m_s"] <= mech.manifold_velocity_limit,
            f"{role} manifold velocity {d['manifold_velocity_m_s']:.2f} m/s vs "
            f"limit {mech.manifold_velocity_limit:.2f} m/s",
            warn=True,
        )
    if mech.seal_type == "o_ring":
        gate(
            "o_ring_groove",
            o_width >= 2.0 * min_tool and o_depth >= 0.5 * min_tool,
            f"O-ring groove {o_width*1e3:.3f} x {o_depth*1e3:.3f} mm; "
            "crush/preload must be checked against the selected seal standard",
            warn=True,
        )
    elif mech.seal_type == "gasket":
        gate(
            "gasket_land",
            land >= 2.0 * min_tool,
            f"gasket land {land*1e3:.3f} mm vs tool land rule "
            f"{2.0*min_tool*1e3:.3f} mm",
            warn=True,
        )
    else:
        gates.append({
            "name": "seal_model",
            "status": "warn",
            "detail": "seal_type=none; pressure joint sealing is not represented",
        })

    return {
        "status": "preliminary_machined_layout_requires_cold_flow_validation",
        "coordinate_system": (
            "axisymmetric injector axis is +Z; injector face/chamber side at "
            "Z=0; upstream manifolds at Z<0"
        ),
        "hydraulic_basis": {
            "annulus_area_m2": inj.annulus.area,
            "slot_area_total_m2": inj.slots.area,
            "slot_count": int(inj.slot_count),
            "radial_stream": inj.radial_stream,
            "total_momentum_ratio": inj.total_momentum_ratio,
            "blockage_factor": inj.blockage_factor,
        },
        "mechanical_spec": _clean_json(mech),
        "resolved": {
            "pintle_outer_radius_m": Rp,
            "pintle_inner_radius_m": inner_pintle_radius,
            "pintle_wall_thickness_m": pintle_wall,
            "pintle_body_length_m": body_len,
            "pintle_body_straight_length_m": body_straight,
            "annulus_inner_radius_m": Rp,
            "annulus_outer_radius_m": Ro,
            "annulus_gap_m": gap,
            "annulus_length_m": annulus_length,
            "sleeve_inner_radius_m": Ro,
            "sleeve_outer_radius_m": Rs,
            "sleeve_wall_thickness_m": sleeve_wall,
            "faceplate_radius_m": face_R,
            "faceplate_outer_diameter_m": face_od,
            "faceplate_thickness_m": face_t,
            "faceplate_input_thickness_m": face_t_input,
            "faceplate_requested_thickness_m": requested_face_t,
            "faceplate_minimum_thickness_m": min_face_t,
            "manifold_back_web_m": back_web,
            "bolt_count": bolt_count,
            "bolt_circle_diameter_m": bolt_circle,
            "bolt_circle_requested_diameter_m": bolt_circle_requested,
            "bolt_circle_minimum_diameter_m": bolt_circle_min,
            "bolt_hole_diameter_m": bolt_hole,
            "slot_width_m": slot_w,
            "slot_height_m": slot_h,
            "slot_depth_requested_m": requested_slot_depth,
            "slot_cut_depth_m": slot_cut_depth,
            "slot_corner_radius_m": slot_corner,
            "slot_end_condition": mech.slot_end_condition,
            "igniter_port_diameter_m": igniter_d,
            "igniter_port_depth_m": igniter_depth,
            "seal_type": mech.seal_type,
            "seal_center_radius_m": seal_center,
            "o_ring_groove_width_m": o_width,
            "o_ring_groove_depth_m": o_depth,
            "gasket_land_width_m": land,
            "min_tool_diameter_m": min_tool,
            "min_corner_radius_m": min_corner,
            "tolerance_m": tol,
        },
        "roles": roles,
        "manufacturing_gates": gates,
        "flow_continuity": {
            "status": "preliminary_transfer_paths_modeled",
            "features": [
                "radial faceplate transfer cuts from each manifold toward the central bore",
                "radial sleeve wall transfer holes for the annulus-fed stream",
                "radial pintle-post transfer holes for the slot-fed stream",
            ],
            "limits": (
                "Transfer paths are geometric continuity features only; detailed "
                "loss coefficients, entrance radii, drill accessibility, and "
                "3-D maldistribution still need design review and cold-flow data."
            ),
        },
        "literature_basis": [
            "NASA SP-8089: injector orifice, annulus, slot, pressure-drop and cold-flow validation ground rules.",
            "Huzel & Huang NASA SP-125 and Sutton & Biblarz: injector pressure-drop fractions and feed-pressure closure.",
            "NASA SP-8087 / annular manifold network screen: manifold velocity, plenum area, and maldistribution remain preliminary.",
            "Son/Hwang pintle literature: TMR and blockage factor are retained as the master pintle geometry/performance variables.",
            "CadQuery/OpenCascade export is geometry only; tolerances, surface finish, O-ring crush, bolt preload, and fitting standards remain drawing/analysis items.",
        ],
    }


def write_machined_pintle_report(layout: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(_clean_json(layout), indent=2) + "\n",
                    encoding="utf-8")
    return path


def _points_on_circle(radius, count, start_deg):
    return [
        (
            radius * math.cos(math.radians(start_deg + 360.0 * i / count)),
            radius * math.sin(math.radians(start_deg + 360.0 * i / count)),
        )
        for i in range(max(int(count), 1))
    ]


def _annular_cut(cq, inner, outer, z0, depth):
    return (
        cq.Workplane("XY").workplane(offset=z0)
        .circle(outer).circle(inner).extrude(depth)
    )


def _radial_cylinder(cq, radius_start, radius_end, z, diameter, angle_deg):
    length = max(radius_end - radius_start, diameter)
    cyl = (
        cq.Workplane("YZ").workplane(offset=radius_start)
        .center(0.0, z).circle(0.5 * diameter).extrude(length)
    )
    return cyl.rotate((0, 0, 0), (0, 0, 1), angle_deg)


def _workplane_solids(shape) -> list:
    vals = shape.vals() if hasattr(shape, "vals") else [shape]
    solids = []
    for val in vals:
        if hasattr(val, "Solids"):
            solids.extend(val.Solids())
    return solids


def inspect_machined_step(path) -> dict:
    """Re-import a machined STEP file and report solid validity."""
    cq = _cq()
    imported = cq.importers.importStep(str(path))
    solids = [solid for shape in imported.vals() for solid in shape.Solids()]
    volumes = [float(s.Volume()) for s in solids]
    return {
        "path": str(path),
        "solid_count": len(solids),
        "single_solid": len(solids) == 1,
        "all_solids_valid": bool(solids) and all(s.isValid() for s in solids),
        "volumes_m3": volumes,
    }


def build_pintle_assembly(inj, *, movable_sleeve=False,
                          igniter_diameter=None):
    """Return a named ``cadquery.Assembly`` for the sized pintle injector."""
    cq = _cq()
    # --- sized dimensions ------------------------------------------------
    Dp = inj.pintle_diameter
    Rp = 0.5 * Dp
    gap = max(inj.annulus.detail.get("gap", 0.1 * Rp), 1e-4)
    Ro = 0.5 * inj.annulus.detail.get("outer_diameter", Dp + 2 * gap)
    Rc = inj.chamber_radius
    slot_w = max(inj.slots.detail.get("slot_width", 0.4 * gap), 5e-5)
    slot_h = max(inj.slots.detail.get("slot_height", slot_w), 5e-5)
    n_slot = max(int(inj.slot_count), 1)
    t_wall = max(0.25 * Rp, 1e-3)               # pintle wall thickness
    body_len = 3.0 * Dp                         # protrusion into the chamber
    body_straight = body_len - Rp               # body before the rounded tip
    t_face = max(0.4 * Dp, 2 * gap)             # faceplate thickness
    R_face = max(Rc, Ro + 2e-3)                 # faceplate spans the chamber bore
    t_sleeve = max(0.4 * gap, 0.5e-3)
    R_ig = 0.5 * (igniter_diameter if igniter_diameter
                  else max(0.3 * (Rp - t_wall), 1.5e-3))

    asm = cq.Assembly(name="pintle_injector")

    # --- injector faceplate: disk with a central bore for the annulus ----
    faceplate = (
        cq.Workplane("XY").workplane(offset=-t_face)
        .circle(R_face).circle(Ro)
        .extrude(t_face)
    )
    asm.add(faceplate, name="injector_faceplate",
            color=cq.Color(0.6, 0.6, 0.65))

    # --- hollow pintle body (tube) ---------------------------------------
    pintle = (
        cq.Workplane("XY").circle(Rp).circle(max(Rp - t_wall, 1e-4))
        .extrude(body_straight)
    )
    asm.add(pintle, name="hollow_pintle_body", color=cq.Color(0.7, 0.7, 0.72))

    # --- pintle tip: hemispherical dome on the body end ------------------
    sphere = (cq.Workplane("XY").workplane(offset=body_straight)
              .sphere(Rp))
    upper = (cq.Workplane("XY").workplane(offset=body_straight)
             .box(2.2 * Rp, 2.2 * Rp, 2.2 * Rp, centered=(True, True, False)))
    tip = sphere.intersect(upper)
    asm.add(tip, name="pintle_tip", color=cq.Color(0.75, 0.72, 0.6))

    # --- axial annulus (flow passage between pintle OD and sleeve ID) ----
    annulus = (
        cq.Workplane("XY").workplane(offset=-t_face)
        .circle(Ro).circle(Rp)
        .extrude(t_face + 0.6 * body_straight)
    )
    asm.add(annulus, name="axial_annulus", color=cq.Color(0.3, 0.55, 0.85))

    # --- radial slot network (boxes through the tip wall) ----------------
    slot_assembly = cq.Assembly(name="radial_slot_network")
    z_slot = body_straight - 0.5 * slot_h
    for i in range(n_slot):
        ang = 360.0 * i / n_slot
        box = (
            cq.Workplane("XY").workplane(offset=z_slot)
            .center(Rp - 0.5 * t_wall, 0.0)
            .box(2.0 * (gap + t_wall), slot_w, slot_h,
                 centered=(True, True, False))
            .rotate((0, 0, 0), (0, 0, 1), ang)
        )
        slot_assembly.add(box, name=f"slot_{i:02d}",
                          color=cq.Color(0.9, 0.5, 0.2))
    asm.add(slot_assembly, name="radial_slot_network")

    # --- propellant manifolds (annular rings behind the face) -----------
    fuel_manifold = (
        cq.Workplane("XY").workplane(offset=-t_face - 0.6 * t_face)
        .circle(Ro + 3e-3).circle(Ro + 1e-3).extrude(0.5 * t_face)
    )
    asm.add(fuel_manifold, name="fuel_manifold", color=cq.Color(0.85, 0.3, 0.2))
    ox_manifold = (
        cq.Workplane("XY").workplane(offset=-t_face - 1.3 * t_face)
        .circle(Ro + 6e-3).circle(Ro + 4e-3).extrude(0.5 * t_face)
    )
    asm.add(ox_manifold, name="oxidizer_manifold", color=cq.Color(0.2, 0.4, 0.8))

    # --- igniter interface (central tube down the axis) -----------------
    igniter = (
        cq.Workplane("XY").workplane(offset=-1.6 * t_face)
        .circle(R_ig).extrude(1.6 * t_face + 0.5 * body_straight)
    )
    asm.add(igniter, name="igniter_interface", color=cq.Color(0.4, 0.8, 0.4))

    # --- regen-coolant outlet connection (port on the faceplate edge) ---
    regen = (
        cq.Workplane("XZ").workplane(offset=-(R_face))
        .center(0.0, -0.5 * t_face)
        .circle(max(2e-3, 0.5 * gap)).extrude(-6e-3)
    )
    asm.add(regen, name="regen_coolant_outlet", color=cq.Color(0.5, 0.8, 0.9))

    # --- optional movable sleeve ----------------------------------------
    if movable_sleeve:
        sleeve = (
            cq.Workplane("XY").circle(Ro + t_sleeve).circle(Ro)
            .extrude(0.6 * body_straight)
        )
        asm.add(sleeve, name="movable_sleeve", color=cq.Color(0.55, 0.55, 0.6))

    return asm


def export_pintle_step(inj, path, *, movable_sleeve=False,
                       igniter_diameter=None, stl_dir=None):
    """Write the named-body pintle assembly to ``path`` as STEP.

    Returns a small dict describing the export (named bodies, validity).  When
    ``stl_dir`` is given, also writes a per-body STL for printing.
    """
    cq = _cq()
    from pathlib import Path
    path = Path(path)
    asm = build_pintle_assembly(
        inj, movable_sleeve=movable_sleeve, igniter_diameter=igniter_diameter)
    # STEP (authoritative, carries the named assembly tree)
    try:
        asm.export(str(path))
    except Exception:
        asm.save(str(path))
    bodies = [c.name for c in asm.children] or ["pintle_injector"]
    result = {
        "format": "STEP_AP214_named_assembly",
        "path": str(path),
        "named_bodies": bodies,
        "movable_sleeve": bool(movable_sleeve),
        "status": "named_assembly_written",
    }
    # Optional per-body STL for printing (STL cannot carry the named tree).
    if stl_dir is not None:
        stl_dir = Path(stl_dir)
        stl_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for child in asm.children:
            try:
                shape = child.obj
                cq.exporters.export(
                    shape, str(stl_dir / f"{child.name}.stl"),
                    exportType="STL")
                written.append(f"{child.name}.stl")
            except Exception:
                continue
        result["stl_bodies"] = written
    return result


# ---------------------------------------------------------------------------
# Named reference PARTS (the four bodies the CLI deliverable names)
# ---------------------------------------------------------------------------
def build_pintle_parts(inj) -> dict:
    """Return ``{part_name: cadquery solid}`` for the four reference parts:
    ``pintle_rod`` (hollow post), ``pintle_tip`` (rounded nose),
    ``annular_sleeve`` (outer annulus wall) and ``injector_face`` (faceplate).
    Geometry matches :func:`build_pintle_assembly` and the 2-D schematic.
    """
    cq = _cq()
    Dp = inj.pintle_diameter
    Rp = 0.5 * Dp
    gap = max(inj.annulus.detail.get("gap", 0.1 * Rp), 1e-4)
    Ro = 0.5 * inj.annulus.detail.get("outer_diameter", Dp + 2 * gap)
    Rc = inj.chamber_radius
    t_sleeve = max(0.4 * gap, 0.5e-3)
    t_wall = max(0.25 * Rp, 1e-3)
    body_len = 3.0 * Dp
    body_straight = body_len - Rp
    t_face = max(0.4 * Dp, 2 * gap)
    R_face = max(Rc, Ro + 2e-3)

    rod = (cq.Workplane("XY").circle(Rp).circle(max(Rp - t_wall, 1e-4))
           .extrude(body_straight))
    sphere = cq.Workplane("XY").workplane(offset=body_straight).sphere(Rp)
    upper = (cq.Workplane("XY").workplane(offset=body_straight)
             .box(2.2 * Rp, 2.2 * Rp, 2.2 * Rp, centered=(True, True, False)))
    tip = sphere.intersect(upper)
    sleeve = (cq.Workplane("XY").workplane(offset=-t_face)
              .circle(Ro + t_sleeve).circle(Ro)
              .extrude(t_face + 0.6 * body_straight))
    face = (cq.Workplane("XY").workplane(offset=-t_face)
            .circle(R_face).circle(Ro).extrude(t_face))
    return {"pintle_rod": rod, "pintle_tip": tip,
            "annular_sleeve": sleeve, "injector_face": face}


# ---------------------------------------------------------------------------
# Machined Boolean-cut STEP bodies
# ---------------------------------------------------------------------------
def _build_machined_faceplate(cq, layout: dict, *, include_inlet_bosses=True):
    r = layout["resolved"]
    roles = layout["roles"]
    eps = max(0.2 * r["tolerance_m"], 1.0e-6)
    face_R = r["faceplate_radius_m"]
    face_t = r["faceplate_thickness_m"]
    bore_R = r["sleeve_outer_radius_m"] + r["tolerance_m"]
    min_tool = r["min_tool_diameter_m"]

    face = (
        cq.Workplane("XY").workplane(offset=-face_t)
        .circle(face_R).circle(bore_R).extrude(face_t)
    )

    side_ports = []
    # Add positive inlet bosses before cutting the manifold pockets.  The boss
    # overlaps the disk by more than its wall thickness so the later side-port
    # cut cannot remove the only fused neck and leave a separate STEP solid.
    if include_inlet_bosses:
        for role, d in roles.items():
            count = int(d["inlet_count"])
            if count <= 0:
                continue
            depth = d["manifold_depth_m"]
            z_mid = -face_t + 0.5 * depth
            for i in range(count):
                ang = d["inlet_angle_deg"] + 360.0 * i / count
                boss_od = max(2.4 * d["inlet_diameter_m"],
                              d["inlet_diameter_m"] + 2.0 * min_tool)
                boss_overlap = max(boss_od, 3.0 * min_tool)
                boss = _radial_cylinder(
                    cq, face_R - boss_overlap, face_R + 2.5 * boss_od,
                    z_mid, boss_od, ang)
                face = face.union(boss)

    for role, d in roles.items():
        depth = d["manifold_depth_m"]
        inner = d["manifold_inner_radius_m"]
        outer = d["manifold_outer_radius_m"]
        count = int(d["inlet_count"])
        if count > 0:
            hole_r = 0.5 * d["inlet_diameter_m"]
            points = _points_on_circle(
                d["manifold_mean_radius_m"], count, d["inlet_angle_deg"])
            axial = (
                cq.Workplane("XY").workplane(offset=-face_t - eps)
                .pushPoints(points).circle(hole_r).extrude(depth + 2.0 * eps)
            )
            side_ports.append(axial)

            z_mid = -face_t + 0.5 * depth
            for i in range(count):
                ang = d["inlet_angle_deg"] + 360.0 * i / count
                boss_od = max(2.4 * d["inlet_diameter_m"],
                              d["inlet_diameter_m"] + 2.0 * min_tool)
                port = _radial_cylinder(
                    cq, d["manifold_mean_radius_m"],
                    face_R + 3.0 * boss_od, z_mid,
                    d["inlet_diameter_m"], ang)
                side_ports.append(port)

        face = face.cut(_annular_cut(cq, inner, outer, -face_t - eps,
                                     depth + eps))

        for i in range(int(d["transfer_count"])):
            ang = d["transfer_angle_deg"] + 360.0 * i / int(d["transfer_count"])
            transfer = _radial_cylinder(
                cq,
                r["sleeve_outer_radius_m"] - min_tool,
                outer + 0.5 * d["transfer_diameter_m"],
                d["transfer_z_m"],
                d["transfer_diameter_m"],
                ang,
            )
            face = face.cut(transfer)

    for cutter in side_ports:
        face = face.cut(cutter)

    bolt_count = int(r["bolt_count"])
    if bolt_count > 0:
        bolt_pts = _points_on_circle(
            0.5 * r["bolt_circle_diameter_m"], bolt_count, 0.0)
        bolt = (
            cq.Workplane("XY").workplane(offset=-face_t - eps)
            .pushPoints(bolt_pts).circle(0.5 * r["bolt_hole_diameter_m"])
            .extrude(face_t + 2.0 * eps)
        )
        face = face.cut(bolt)

    if r["seal_type"] == "o_ring":
        sw = r["o_ring_groove_width_m"]
        sd = r["o_ring_groove_depth_m"]
        sr = r["seal_center_radius_m"]
        face = face.cut(_annular_cut(cq, sr - 0.5 * sw, sr + 0.5 * sw,
                                     -sd, sd + eps))

    return face


def _slot_cutter(cq, layout: dict, angle_deg: float):
    r = layout["resolved"]
    cut_depth = r["slot_cut_depth_m"] + 2.0 * r["tolerance_m"]
    Rp = r["pintle_outer_radius_m"]
    z_slot = r["pintle_body_straight_length_m"] - 0.5 * r["slot_height_m"]
    cutter = (
        cq.Workplane("XY").workplane(offset=z_slot)
        .center(Rp - 0.5 * cut_depth, 0.0)
        .box(cut_depth, r["slot_width_m"], r["slot_height_m"],
             centered=(True, True, True))
    )
    if (
        r["slot_end_condition"] in ("rounded", "drilled", "edm")
        and r["slot_corner_radius_m"] > 0.0
    ):
        try:
            cutter = cutter.edges("|X").fillet(r["slot_corner_radius_m"])
        except Exception:
            pass
    return cutter.rotate((0, 0, 0), (0, 0, 1), angle_deg)


def _build_machined_pintle_post(cq, layout: dict):
    r = layout["resolved"]
    eps = max(0.2 * r["tolerance_m"], 1.0e-6)
    Rp = r["pintle_outer_radius_m"]
    Ri = r["pintle_inner_radius_m"]
    face_t = r["faceplate_thickness_m"]
    straight = r["pintle_body_straight_length_m"]

    post = (
        cq.Workplane("XY").workplane(offset=-face_t)
        .circle(Rp).circle(Ri).extrude(face_t + straight)
    )
    sphere = cq.Workplane("XY").workplane(offset=straight).sphere(Rp)
    upper = (
        cq.Workplane("XY").workplane(offset=straight)
        .box(2.2 * Rp, 2.2 * Rp, 2.2 * Rp, centered=(True, True, False))
    )
    tip = sphere.intersect(upper)
    post = post.union(tip)

    igniter = (
        cq.Workplane("XY").workplane(offset=-face_t - eps)
        .circle(0.5 * r["igniter_port_diameter_m"])
        .extrude(r["igniter_port_depth_m"] + eps)
    )
    post = post.cut(igniter)

    n_slot = max(int(layout["hydraulic_basis"]["slot_count"]), 1)
    for i in range(n_slot):
        post = post.cut(_slot_cutter(cq, layout, 360.0 * i / n_slot))

    for role, d in layout["roles"].items():
        if d["feeds"] != "slots":
            continue
        count = int(d["transfer_count"])
        for i in range(count):
            ang = d["transfer_angle_deg"] + 360.0 * i / count
            transfer = _radial_cylinder(
                cq,
                Ri - 0.5 * d["transfer_diameter_m"],
                Rp + 0.5 * d["transfer_diameter_m"],
                d["transfer_z_m"],
                d["transfer_diameter_m"],
                ang,
            )
            post = post.cut(transfer)
    return post


def _build_machined_sleeve(cq, layout: dict):
    r = layout["resolved"]
    sleeve = (
        cq.Workplane("XY").workplane(offset=-r["faceplate_thickness_m"])
        .circle(r["sleeve_outer_radius_m"])
        .circle(r["sleeve_inner_radius_m"])
        .extrude(r["faceplate_thickness_m"] + r["annulus_length_m"])
    )
    for role, d in layout["roles"].items():
        if d["feeds"] != "annulus":
            continue
        count = int(d["transfer_count"])
        for i in range(count):
            ang = d["transfer_angle_deg"] + 360.0 * i / count
            transfer = _radial_cylinder(
                cq,
                r["sleeve_inner_radius_m"] - 0.5 * d["transfer_diameter_m"],
                r["sleeve_outer_radius_m"] + 0.5 * d["transfer_diameter_m"],
                d["transfer_z_m"],
                d["transfer_diameter_m"],
                ang,
            )
            sleeve = sleeve.cut(transfer)
    return sleeve


def build_machined_pintle_bodies(
    inj, *, spec=None, layout=None, include_inlet_bosses=True
) -> dict:
    """Return true Boolean-cut machined solids for the pintle injector."""
    cq = _cq()
    layout = layout or resolve_machined_pintle_layout(inj, spec=spec)
    return {
        "injector_face_machined": _build_machined_faceplate(
            cq, layout, include_inlet_bosses=include_inlet_bosses),
        "pintle_post_slotted": _build_machined_pintle_post(cq, layout),
        "annular_sleeve": _build_machined_sleeve(cq, layout),
    }


def export_machined_pintle_cad(inj, out_dir, *, spec=None, fmt="step") -> dict:
    """Export machined pintle STEP bodies plus a manufacturing report.

    The STEP files are true OpenCascade B-rep solids with Boolean-subtracted
    slots, annulus/sleeve passages, manifolds, feed/inlet ports, bolt holes,
    igniter bore, and seal grooves.  If CadQuery is unavailable, the sizing and
    manufacturing report are still written so the run is auditable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []
    layout = resolve_machined_pintle_layout(inj, spec=spec)

    if fmt != "step":
        notes.append(
            "machined mode is STEP-only; writing true STEP filenames and "
            f"ignoring requested format {fmt!r}")

    if not cadquery_available():
        layout["cad_export"] = {
            "status": "cadquery_unavailable",
            "representation": "report_only",
            "required_extra": "pip install -e .[cad]",
        }
        report = write_machined_pintle_report(
            layout, out_dir / "injector_manufacturing_report.json")
        files["manufacturing_report"] = str(report)
        notes.append(
            "machined STEP export requested but CadQuery/OpenCascade is not "
            "installed; manufacturing report written and STEP bodies skipped.")
        return {"files": files, "notes": notes, "layout": layout}

    cq = _cq()
    step_files = {
        "injector_face_machined": out_dir / "injector_face_machined.step",
        "pintle_post_slotted": out_dir / "pintle_post_slotted.step",
        "annular_sleeve": out_dir / "annular_sleeve.step",
    }
    try:
        bodies = build_machined_pintle_bodies(inj, spec=spec, layout=layout)
        inspections = {}
        inlet_bosses_exported = True
        for name, solid in bodies.items():
            cq.exporters.export(solid, str(step_files[name]), exportType="STEP")
            inspection = inspect_machined_step(step_files[name])
            if (
                name == "injector_face_machined"
                and not (
                    inspection["single_solid"]
                    and inspection["all_solids_valid"]
                )
            ):
                notes.append(
                    "injector_face_machined STEP re-imported as "
                    f"{inspection['solid_count']} solids with layout-only "
                    "inlet bosses; re-exporting the face as a single body "
                    "without additive fitting bosses.")
                inlet_bosses_exported = False
                solid = _build_machined_faceplate(
                    cq, layout, include_inlet_bosses=False)
                bodies[name] = solid
                cq.exporters.export(solid, str(step_files[name]), exportType="STEP")
                inspection = inspect_machined_step(step_files[name])
            if not (
                inspection["single_solid"] and inspection["all_solids_valid"]
            ):
                raise RuntimeError(
                    f"{name} STEP round-trip is not one valid solid: {inspection}"
                )
            inspections[name] = inspection
            files[name] = str(step_files[name])

        asm = cq.Assembly(name="pintle_injector_machined")
        colors = {
            "injector_face_machined": cq.Color(0.60, 0.60, 0.65),
            "pintle_post_slotted": cq.Color(0.72, 0.72, 0.70),
            "annular_sleeve": cq.Color(0.50, 0.55, 0.60),
        }
        for name, solid in bodies.items():
            asm.add(solid, name=name, color=colors.get(name))
        assembly_path = out_dir / "injector_assembly_machined.step"
        try:
            asm.export(str(assembly_path))
        except Exception:
            asm.save(str(assembly_path))
        files["machined_assembly"] = str(assembly_path)
        layout["cad_export"] = {
            "status": "step_written",
            "representation": "open_cascade_brep_boolean_cut_solids",
            "files": files.copy(),
            "inspection": inspections,
            "inlet_bosses_exported": inlet_bosses_exported,
            "boolean_features": [
                f"{int(layout['hydraulic_basis']['slot_count'])} radial pintle slot cut-throughs",
                "annular sleeve passage",
                "annular manifold cavities",
                "radial manifold transfer cuts",
                "annulus sleeve transfer holes",
                "pintle-post transfer holes for the slotted stream",
                (
                    "fuel/oxidizer feed and side inlet ports with layout-only inlet bosses"
                    if inlet_bosses_exported else
                    "fuel/oxidizer feed and side inlet ports; additive inlet bosses suppressed for single-solid STEP"
                ),
                "bolt holes on bolt circle",
                "central igniter bore",
                "O-ring groove when seal_type=o_ring",
            ],
        }
    except Exception as exc:
        layout["cad_export"] = {
            "status": "step_export_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        notes.append(f"machined STEP export failed: {type(exc).__name__}: {exc}")

    report = write_machined_pintle_report(
        layout, out_dir / "injector_manufacturing_report.json")
    files["manufacturing_report"] = str(report)
    return {"files": files, "notes": notes, "layout": layout}


def _pintle_profile_loops(inj) -> dict:
    """Closed meridional half-section loops (mm) for a 2-D DXF profile."""
    mm = 1.0e3
    Dp = float(inj.pintle_diameter); Rp = 0.5 * Dp
    gap = float(inj.annulus.detail.get("gap", 0.1 * Rp))
    Ro = 0.5 * float(inj.annulus.detail.get("outer_diameter", Dp + 2 * gap))
    t_sleeve = max(0.4 * gap, 0.5e-3)
    t_face = max(0.4 * Dp, 2 * gap)
    Rc = float(inj.chamber_radius)
    body_len = 3.0 * Dp
    body_straight = body_len - Rp
    rod = [(0.0, 0.0), (0.0, Rp), (body_straight, Rp)]
    for i in range(1, 17):                       # quarter-arc to the apex
        a = i / 16.0 * (math.pi / 2.0)
        rod.append((body_straight + Rp * math.sin(a), Rp * math.cos(a)))
    rod.append((0.0, 0.0))
    xs = 0.55 * body_len
    sleeve = [(0.0, Ro), (xs, Ro), (xs, Ro + t_sleeve), (0.0, Ro + t_sleeve),
              (0.0, Ro)]
    R_face = max(Rc, Ro + 2e-3)
    face = [(-t_face, Ro + t_sleeve), (0.0, Ro + t_sleeve), (0.0, R_face),
            (-t_face, R_face), (-t_face, Ro + t_sleeve)]
    loops = {"pintle_rod_tip": rod, "annular_sleeve": sleeve,
             "injector_face": face}
    return {k: [(x * mm, y * mm) for x, y in v] for k, v in loops.items()}


def write_pintle_profile_dxf(inj, path):
    """Write the 2-D meridional profile as DXF (ezdxf if present, else a
    hand-written R12 DXF so DXF works without any optional dependency)."""
    from pathlib import Path
    path = Path(path)
    loops = _pintle_profile_loops(inj)
    try:
        import ezdxf
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        for name, pts in loops.items():
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": name})
        doc.saveas(str(path))
        return path
    except Exception:
        pass
    out = ["0", "SECTION", "2", "ENTITIES"]
    for name, pts in loops.items():
        out += ["0", "POLYLINE", "8", name, "66", "1", "70", "1"]
        for x, y in pts:
            out += ["0", "VERTEX", "8", name,
                    "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0.0"]
        out += ["0", "SEQEND"]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def export_pintle_cad(inj, out_dir, *, mode="reference", fmt="step",
                      movable_sleeve=False) -> dict:
    """Write CAD-neutral reference geometry for the sized pintle.

    ``mode``  : ``reference`` (single assembly file) | ``parts`` (also the four
                named part files in ``pintle_parts/``) | ``machined`` (true
                Boolean-cut machined STEP solids + manufacturing report).
    ``fmt``   : ``step`` (portable B-rep, default) | ``stl`` (mesh preview) |
                ``dxf`` (2-D meridional profile).
    STEP/STL require CadQuery; DXF does not.  Returns ``{files, notes}``.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []

    if mode == "machined":
        return export_machined_pintle_cad(inj, out_dir, fmt=fmt)

    if fmt == "dxf":
        files["reference_dxf"] = str(write_pintle_profile_dxf(
            inj, out_dir / "pintle_reference.dxf"))
        notes.append("DXF is the 2-D meridional profile (revolve to recover the "
                     "solid); use --injector-cad-format step for 3-D parts.")
        return {"files": files, "notes": notes}

    cq = _cq()                                   # STEP/STL need CadQuery
    ext = "step" if fmt == "step" else "stl"
    etype = "STEP" if fmt == "step" else "STL"

    ref = out_dir / f"pintle_reference.{ext}"
    if fmt == "step":
        export_pintle_step(inj, ref, movable_sleeve=movable_sleeve)
    else:
        asm = build_pintle_assembly(inj, movable_sleeve=movable_sleeve)
        try:
            shape = asm.toCompound()
        except Exception:
            shape = asm
        cq.exporters.export(shape, str(ref), exportType="STL")
    files[f"reference_{ext}"] = str(ref)

    if mode == "parts":
        pdir = out_dir / "pintle_parts"
        pdir.mkdir(parents=True, exist_ok=True)
        for name, solid in build_pintle_parts(inj).items():
            p = pdir / f"{name}.{ext}"
            try:
                cq.exporters.export(solid, str(p), exportType=etype)
                files[f"part_{name}"] = str(p)
            except Exception as exc:
                notes.append(f"part {name} export failed: {type(exc).__name__}")
    return {"files": files, "notes": notes}
