"""pump_cad_brep.py - true B-rep CAD for the electric-pump package.

Builds named CadQuery assemblies for the sized fuel/oxidizer pumps
(shaft, inducer with helical swept blades, impeller with log-spiral camber
blades, diffuser vane ring, volute casing + collecting scroll, and the
motor/inverter/battery package placeholders).  Every dimension is consumed
from the meanline manifest (:func:`raosim.pump_cad.pump_reference_geometry`,
itself fed by ``pumps.PumpReferenceGeometry``); blade camber math lives in
:func:`raosim.pumps.impeller_blade_camber` (physics owns it, CAD sweeps it).

Conventions mirror :mod:`raosim.injector_cad` (+Z shaft axis, SI metres in,
millimetres at the CadQuery/OpenCascade kernel boundary, named
``cq.Assembly``) and :mod:`raosim.regen_cad` validation (export -> re-import
-> ``isValid``/solid-count/volume gates).  This is reference geometry for
layout and trade review, not blade-to-blade design: use pump CFD,
rotordynamics, seals/bearings, and cold-flow tests before hardware release.

Axes: +Z is the shaft axis; the impeller exit plane sits at Z = 0 (the
meridional station x = 0), flow approaches from -Z; XY is the radial plane.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from raosim.pump_cad import pump_reference_geometry
from raosim.pumps import impeller_blade_camber

_M_TO_MM = 1000.0
# Layout placeholders where the meanline solves no dimension (labeled, same
# values the Phase-0 mesh writer used); all solved dimensions come from the
# manifest.
_INDUCER_BLADE_THICKNESS_RATIO = 0.025
_INDUCER_BLADE_THICKNESS_FLOOR_M = 3.0e-4
_VANE_THICKNESS_PITCH_RATIO = 0.22


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
            "pump B-rep STEP export requires CadQuery/OpenCascade "
            "(pip install cadquery)"
        ) from exc
    return cq


def _mm(value_m: float) -> float:
    return float(value_m) * _M_TO_MM


def _revolve_profile(cq, points_rz_mm: list[tuple[float, float]]):
    """Revolve a closed (radius, z) polyline about the +Z shaft axis."""
    return (
        cq.Workplane("XZ")
        .polyline(points_rz_mm)
        .close()
        .revolve(360.0, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        .val()
    )


def _washer(cq, r_inner_mm, r_outer_mm, z0_mm, z1_mm):
    return _revolve_profile(cq, [
        (r_inner_mm, z0_mm), (r_outer_mm, z0_mm),
        (r_outer_mm, z1_mm), (r_inner_mm, z1_mm),
    ])


def _cylinder(cq, radius_mm, z0_mm, z1_mm):
    return cq.Solid.makeCylinder(
        float(radius_mm), float(z1_mm - z0_mm),
        cq.Vector(0.0, 0.0, float(z0_mm)), cq.Vector(0.0, 0.0, 1.0),
    )


def _rotate_z(cq, shape, angle_deg: float):
    return shape.rotate(
        cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), float(angle_deg)
    )


def _fuse_all(base, shapes):
    # Single n-ary fuse; no clean() afterwards — OCC's unifier corrupts the
    # thin-blade fuses this module produces (verified: valid before, invalid
    # after clean), and the re-import gate would reject the result.
    if not shapes:
        return base
    return base.fuse(*shapes)


def _camber_ribbon_xy(camber, thickness_mm: float) -> list[tuple[float, float]]:
    """Closed in-plane outline of a blade: camber line offset by +-t/2."""
    pts = [
        (
            _mm(p["radius_m"]) * math.cos(p["theta_rad"]),
            _mm(p["radius_m"]) * math.sin(p["theta_rad"]),
        )
        for p in camber
    ]
    n = len(pts)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    half = 0.5 * float(thickness_mm)
    for i, (x, y) in enumerate(pts):
        x0, y0 = pts[max(i - 1, 0)]
        x1, y1 = pts[min(i + 1, n - 1)]
        tx, ty = x1 - x0, y1 - y0
        norm = math.hypot(tx, ty)
        if norm <= 0.0:
            raise ValueError("degenerate camber line (repeated points)")
        nx, ny = -ty / norm, tx / norm
        left.append((x + half * nx, y + half * ny))
        right.append((x - half * nx, y - half * ny))
    return left + right[::-1]


def build_impeller(cq, impeller_comp, channel_comp, shaft_comp,
                   balance_comp=None, *, samples: int = 61):
    """Hub/backplate revolve + Z log-spiral camber blades, shaft bore.

    The hub/backplate is one revolve through the meanline's meridional
    channel hub curve; the blades stand on it and are trimmed by a revolve
    of the shroud curve (semi-open impeller; SP-8109 shrouded selection is
    a later option).  Both curves come from ``meridional_channel``
    (quarter-ellipse hub/shroud honoring D1, D2, b2 - SP-8109 sec. 2.3.1.2
    meridional practice).
    """
    r2 = 0.5 * _mm(impeller_comp["outer_diameter_m"])
    t_b = _mm(impeller_comp["blade_thickness_m"])
    blade_count = int(impeller_comp["blade_count"])
    beta1 = impeller_comp["inlet_blade_angle_deg"]
    beta2 = impeller_comp["outlet_blade_angle_deg"]
    if beta1 is None or beta2 is None:
        raise ValueError(
            "impeller blade angles missing from the pump manifest; re-run "
            "pump sizing (pumps.size_electric_pumps) to export them"
        )
    if not channel_comp:
        raise ValueError(
            "meridional channel missing from the pump manifest; re-run "
            "pump sizing (pumps.size_electric_pumps) to export it"
        )
    hub_pts = [(_mm(p["r_m"]), _mm(p["x_m"]))
               for p in channel_comp["hub_curve"]]
    shroud_pts = [(_mm(p["r_m"]), _mm(p["x_m"]))
                  for p in channel_comp["shroud_curve"]]
    # The channel curves are authoritative for the axial extent.
    width = -hub_pts[0][1]
    r_hub_eye = hub_pts[0][0]
    r1 = shroud_pts[0][0]

    # Hub + backplate as ONE revolve (axis -> eye hub -> hub curve -> exit
    # rim -> backplate face): no hub/backplate fuse, no coincident faces.
    body = _revolve_profile(cq, [
        (0.0, -width),
        *hub_pts,
        (r2, t_b),
        (0.0, t_b),
    ])

    camber = impeller_blade_camber(
        impeller_comp["inlet_diameter_m"] * 0.5,
        impeller_comp["outer_diameter_m"] * 0.5,
        beta1, beta2, samples=samples,
    )
    outline = _camber_ribbon_xy(camber, t_b)
    # Blades overlap half a blade thickness into the backplate: exactly-
    # coincident tangent faces make OCC fuses invalid.
    blade = (
        cq.Workplane("XY", origin=(0.0, 0.0, 0.5 * t_b))
        .polyline(outline)
        .close()
        .extrude(-(width + 0.5 * t_b))
        .val()
    )
    # Enforce the exact solved tip diameter, then trim the blade tops to
    # the shroud curve revolve (full eye height at r1 -> b2 at the exit).
    blade = blade.intersect(_cylinder(cq, r2, -1.5 * width, t_b))
    shroud_cutter = _revolve_profile(cq, [
        *shroud_pts,
        (1.05 * r2, shroud_pts[-1][1]),
        (1.05 * r2, -1.5 * width),
        (r1, -1.5 * width),
    ])
    blade = blade.cut(shroud_cutter)
    blades = [
        _rotate_z(cq, blade, 360.0 * k / blade_count)
        for k in range(blade_count)
    ]
    body = _fuse_all(body, blades)
    notes: list[str] = []

    if balance_comp:
        # Raised hub-side wear-ring land on the back face (SP-8109
        # sec. 3.5.2.1 layout; land width/height take the blade thickness
        # as the labeled placeholder scale).  Overlaps half t_b into the
        # backplate so the fuse never sees coincident faces.
        r_ring = 0.5 * _mm(balance_comp["hub_wear_ring_diameter_m"])
        land = _washer(cq, r_ring - t_b, r_ring, 0.5 * t_b, 1.5 * t_b)
        body = _fuse_all(body, [land])
        holes = balance_comp.get("balance_holes") or {}
        if holes.get("status") == "sized":
            hole_r = 0.5 * _mm(holes["diameter_m"])
            count = max(int(holes["count"]), 1)
            r_inner = r_hub_eye
            r_outer = r_ring - t_b
            room = 0.45 * (r_outer - r_inner)
            if hole_r <= room:
                # Balance holes vent the back cavity to the impeller inlet
                # (SP-8109 sec. 3.5.2.2): drilled at a diameter smaller
                # than the hub wear ring, one per blade passage.
                r_c = 0.5 * (r_inner + r_outer)
                for k in range(count):
                    ang = 2.0 * math.pi * k / count
                    hole = cq.Solid.makeCylinder(
                        hole_r, width + 3.0 * t_b,
                        cq.Vector(r_c * math.cos(ang),
                                  r_c * math.sin(ang), -width),
                        cq.Vector(0.0, 0.0, 1.0),
                    )
                    body = body.cut(hole)
            else:
                notes.append(
                    "impeller balance holes skipped: sized hole radius "
                    f"{hole_r / _M_TO_MM:.4g} m does not fit the annulus "
                    "between the eye hub and the hub wear ring"
                )

    bore_r = 0.5 * _mm(shaft_comp["diameter_m"])
    if bore_r < 0.9 * r_hub_eye:
        bore = _cylinder(cq, bore_r, -1.5 * width, 2.0 * t_b)
        return body.cut(bore), notes
    notes.append(
        "impeller shaft bore skipped: solved shaft diameter "
        f"{shaft_comp['diameter_m']:.4g} m does not fit inside the eye hub "
        f"(radius {r_hub_eye / _M_TO_MM:.4g} m) - integral shaft/rotor "
        "assumed (small-pump minimum-shaft screen vs hub sizing; see plan "
        "Phase 2)"
    )
    return body, notes


def build_inducer(cq, inducer_comp, shaft_comp):
    """Hub cylinder + helical swept blades, trimmed to the solved envelope.

    The blade section is swept along the parametric helix defined by the
    solved pitch/wrap (``inducer_helix``); an intersection with the tip
    cylinder enforces the exact D_ind and axial length.
    """
    r_tip = 0.5 * _mm(inducer_comp["diameter_m"])
    r_hub = 0.5 * _mm(inducer_comp["hub_diameter_m"])
    length = _mm(inducer_comp["length_m"])
    pitch = _mm(inducer_comp["pitch_m"])
    blade_count = int(inducer_comp["blade_count"])
    thickness = max(
        _INDUCER_BLADE_THICKNESS_RATIO * 2.0 * r_tip,
        _mm(_INDUCER_BLADE_THICKNESS_FLOOR_M),
    )

    # Hub runs past the envelope planes and the blades are swept untrimmed:
    # every fuse then sees a volumetric overlap, never an exactly-coincident
    # face (which makes OCC fuses invalid), and the solved envelope is
    # enforced afterwards with a single complement cut (OCC's common/
    # intersect silently returns empty on fuse results here; cuts are
    # reliable).
    hub = _cylinder(cq, r_hub, -0.1 * length, 1.1 * length)
    # Root the blade section at half the hub radius: the Frenet frame tilts
    # the swept rectangle, so a shallow radial overlap can detach from the
    # hub and leave disconnected solids.
    r_mid = 0.5 * (0.5 * r_hub + r_tip)
    radial_extent = r_tip - 0.5 * r_hub
    helix = cq.Wire.makeHelix(pitch=pitch, height=length, radius=r_mid)
    path = cq.Workplane("XY").add(helix)
    blade = (
        cq.Workplane("XZ", origin=(r_mid, 0.0, 0.0))
        .rect(radial_extent, thickness)
        .sweep(path, isFrenet=True)
        .val()
    )
    blades = [
        _rotate_z(cq, blade, 360.0 * k / blade_count)
        for k in range(blade_count)
    ]
    body = _fuse_all(hub, blades)
    # No radial trim: the swept blade face already rides the tip radius, and
    # a cylindrical cutter there is an OCC near-tangent boolean that returns
    # invalid negative-volume bodies (chordal corner overshoot is only
    # ~t^2/(8 r_tip), microns).  The axial overshoot is removed with two
    # planar slab cuts, which are transversal and robust.
    below = _cylinder(cq, 4.0 * r_tip, -0.5 * length, 0.0)
    above = _cylinder(cq, 4.0 * r_tip, length, 1.5 * length)
    body = body.cut(below).cut(above)
    bore_r = 0.5 * _mm(shaft_comp["diameter_m"])
    if bore_r < 0.9 * r_hub:
        bore = _cylinder(cq, bore_r, -0.2 * length, 1.2 * length)
        return body.cut(bore), None
    return body, (
        "inducer shaft bore skipped: solved shaft diameter "
        f"{shaft_comp['diameter_m']:.4g} m does not fit inside the hub "
        f"(hub diameter {inducer_comp['hub_diameter_m']:.4g} m) - integral "
        "shaft/rotor assumed (small-pump minimum-shaft screen vs SP-8052 "
        "hub ratio; see plan Phase 2)"
    )


def build_diffuser_ring(cq, ring_comp):
    """Vaned diffuser ring: side plates + vanes set at the solved flow angle."""
    inner = _mm(ring_comp["inner_radius_m"])
    outer = _mm(ring_comp["outer_radius_m"])
    b_v = _mm(ring_comp["axial_width_m"])
    t_wall = _mm(ring_comp["casing_wall_thickness_m"])
    vane_count = int(ring_comp.get("vane_count") or 0)

    back_plate = _washer(cq, inner, outer, 0.0, t_wall)
    front_plate = _washer(cq, inner, outer, -b_v - t_wall, -b_v)
    vanes = []
    if vane_count > 0:
        angle_deg = ring_comp.get("vane_angle_deg")
        if angle_deg is None:
            raise ValueError(
                "diffuser vane angle missing from the pump manifest; re-run "
                "pump sizing to export the solved flow angle"
            )
        alpha = math.radians(max(float(angle_deg), 5.0))
        r_mid = 0.5 * (inner + outer)
        pitch = 2.0 * math.pi * r_mid / vane_count
        t_vane = _VANE_THICKNESS_PITCH_RATIO * pitch
        chord = (outer - inner) / max(math.sin(alpha), 0.25)
        # Vanes overlap half a wall thickness into each side plate so the
        # fuse never sees exactly-coincident tangent faces.
        band = _washer(cq, inner, outer, -b_v - 0.5 * t_wall, 0.5 * t_wall)
        for k in range(vane_count):
            phi = 360.0 * k / vane_count
            box = cq.Solid.makeBox(
                chord, t_vane, b_v + t_wall,
                cq.Vector(-0.5 * chord, -0.5 * t_vane, -b_v - 0.5 * t_wall),
            )
            # Long axis along the local flow direction: tangential direction
            # (phi + 90 deg) swung back toward radial by the vane angle.
            box = box.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1),
                             phi + 90.0 - math.degrees(alpha))
            box = box.translate(cq.Vector(
                r_mid * math.cos(math.radians(phi)),
                r_mid * math.sin(math.radians(phi)),
                0.0,
            ))
            vanes.append(box.intersect(band))
    return _fuse_all(back_plate, [front_plate, *vanes])


def build_volute_casing(cq, ring_comp, ports_comp, *, sections: int = 24):
    """Casing shell + collecting scroll with a linear area schedule.

    A(theta) = A_exit * theta / 2pi (constant-angular-momentum first pass,
    SP-8109 collecting-volute practice); the exit port leaves tangentially
    at the outlet-port equivalent diameter.
    """
    casing_r = _mm(ring_comp["casing_inner_radius_m"])
    t_wall = _mm(ring_comp["casing_wall_thickness_m"])
    b_v = _mm(ring_comp["axial_width_m"])
    exit_area_mm2 = float(ring_comp["volute_exit_area_m2"]) * _M_TO_MM ** 2
    a_exit = math.sqrt(exit_area_mm2 / math.pi)
    z_mid = -0.5 * b_v

    shell = _washer(cq, casing_r, casing_r + t_wall,
                    -b_v - 2.0 * t_wall, 2.0 * t_wall)
    wires = []
    for i in range(1, sections + 1):
        theta = 2.0 * math.pi * i / sections
        radius = a_exit * math.sqrt(theta / (2.0 * math.pi))
        center = cq.Vector(
            casing_r * math.cos(theta), casing_r * math.sin(theta), z_mid
        )
        normal = cq.Vector(-math.sin(theta), math.cos(theta), 0.0)
        wires.append(cq.Wire.makeCircle(radius, center, normal))
    scroll = cq.Solid.makeLoft(wires, True)

    port_d = ports_comp.get("outlet_equivalent_diameter_m")
    if port_d is None or port_d <= 0.0:
        raise ValueError(
            "volute outlet port diameter missing from the pump manifest"
        )
    port = cq.Solid.makeCylinder(
        0.5 * _mm(port_d), 3.0 * a_exit,
        cq.Vector(casing_r, 0.0, z_mid), cq.Vector(0.0, 1.0, 0.0),
    )
    return _fuse_all(shell, [scroll, port])


def build_shaft(cq, shaft_comp, z_start_mm: float):
    return _cylinder(cq, 0.5 * _mm(shaft_comp["diameter_m"]),
                     z_start_mm, z_start_mm + _mm(shaft_comp["span_m"]))


def build_motor(cq, motor_comp, shaft_comp, z_start_mm: float):
    d = _mm(motor_comp["diameter_m"])
    length = _mm(motor_comp["length_m"])
    if d <= 0.0 or length <= 0.0:
        return None
    body = _cylinder(cq, 0.5 * d, z_start_mm, z_start_mm + length)
    bore_r = 0.5 * _mm(shaft_comp["diameter_m"])
    if bore_r < 0.45 * d:
        bore = _cylinder(cq, bore_r,
                         z_start_mm - 1.0, z_start_mm + length + 1.0)
        body = body.cut(bore)
    return body


def _box_solid(cq, box_m, center_mm):
    dims = [(_mm(v)) for v in (box_m or [0.0, 0.0, 0.0])]
    if min(dims) <= 0.0:
        return None
    return cq.Solid.makeBox(
        dims[0], dims[1], dims[2],
        cq.Vector(
            center_mm[0] - 0.5 * dims[0],
            center_mm[1] - 0.5 * dims[1],
            center_mm[2] - 0.5 * dims[2],
        ),
    )


def _stations_mm(reference_geometry) -> dict[str, float]:
    if reference_geometry is None:
        return {}
    profile = (
        reference_geometry.get("meridional_profile")
        if isinstance(reference_geometry, dict)
        else reference_geometry.meridional_profile
    ) or []
    return {
        str(row["station"]): _mm(row["x_m"])
        for row in profile
        if row.get("station") is not None and row.get("x_m") is not None
    }


def build_pump_parts(
    pump_result, role: str, *, samples: int = 61
) -> tuple[dict, list[str]]:
    """Build the named B-rep solids for one pump stream (global coords).

    Returns ``(parts, notes)``; notes record honest deviations such as a
    skipped shaft bore when the solved shaft cannot fit the solved hub.
    """
    cq = _cq()
    geom = pump_reference_geometry(pump_result)
    comp = geom["components"].get(role)
    if comp is None or comp.get("status") == "not_sized":
        raise ValueError(f"{role} pump is not sized; no CAD to build")

    lines = (
        pump_result.get("lines")
        if isinstance(pump_result, dict)
        else pump_result.lines
    )
    line = lines[role]
    ref = (
        line.get("reference_geometry")
        if isinstance(line, dict)
        else line.reference_geometry
    )
    stations = _stations_mm(ref)

    shaft_comp = comp["shaft"]
    impeller_comp = comp["impeller"]
    parts: dict[str, Any] = {}
    notes: list[str] = []

    impeller, imp_notes = build_impeller(
        cq, impeller_comp, comp.get("meridional_channel"), shaft_comp,
        comp.get("thrust_balance"), samples=samples,
    )
    parts["impeller"] = impeller
    notes.extend(f"{role}: {n}" for n in imp_notes)

    if "inducer" in comp:
        inducer, note = build_inducer(cq, comp["inducer"], shaft_comp)
        if note:
            notes.append(f"{role}: {note}")
        z_le = stations.get(
            "inducer_leading_edge",
            -_mm(impeller_comp["axial_width_m"])
            - _mm(comp["inducer"]["length_m"]),
        )
        parts["inducer"] = inducer.translate(cq.Vector(0, 0, z_le))

    if "diffuser_volute" in comp:
        parts["diffuser_ring"] = build_diffuser_ring(
            cq, comp["diffuser_volute"]
        )
        parts["volute_casing"] = build_volute_casing(
            cq, comp["diffuser_volute"], comp["ports"]
        )

    z_shaft_start = stations.get(
        "inlet_port", -0.55 * _mm(shaft_comp["span_m"])
    )
    parts["shaft"] = build_shaft(cq, shaft_comp, z_shaft_start)

    t_b = _mm(impeller_comp["blade_thickness_m"])
    if "motor" in comp:
        motor = build_motor(cq, comp["motor"], shaft_comp, t_b)
        if motor is not None:
            parts["motor"] = motor
    if "inverter" in comp:
        motor_d = _mm(comp.get("motor", {}).get("diameter_m") or 0.0)
        box_m = comp["inverter"].get("box_m")
        offset_x = 0.5 * motor_d + 0.7 * _mm((box_m or [0.0])[0])
        inverter = _box_solid(
            cq, box_m,
            (offset_x, 0.0, t_b + 0.5 * _mm((box_m or [0, 0, 0])[2])),
        )
        if inverter is not None:
            parts["inverter"] = inverter

    bad = [
        name for name, solid in parts.items()
        if not solid.isValid() or len(solid.Solids()) != 1
    ]
    if bad:
        raise RuntimeError(
            f"{role} pump B-rep produced invalid or disconnected solids: "
            f"{bad}"
        )
    return parts, notes


def build_pump_assembly(pump_result, role: str, *, samples: int = 61):
    """Named ``cq.Assembly`` for one pump stream."""
    cq = _cq()
    parts, _notes = build_pump_parts(pump_result, role, samples=samples)
    assembly = cq.Assembly(name=f"{role}_pump")
    for name, solid in parts.items():
        assembly.add(solid, name=name)
    return assembly


def inspect_pump_step(path: str | Path) -> dict:
    """Re-import a STEP file and report B-rep validity and dimensions."""
    cq = _cq()
    path = Path(path).expanduser().resolve()
    imported = cq.importers.importStep(str(path))
    shapes = imported.vals()
    solids = [solid for shape in shapes for solid in shape.Solids()]
    valid = bool(solids) and all(s.isValid() for s in solids)
    if solids:
        bbox = solids[0].BoundingBox()
        for solid in solids[1:]:
            bbox.add(solid.BoundingBox())
        volume = float(sum(abs(float(s.Volume())) for s in solids))
        bbox_mm = {
            "x": float(bbox.xlen), "y": float(bbox.ylen),
            "z": float(bbox.zlen),
        }
    else:
        volume = 0.0
        bbox_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "path": str(path),
        "representation": "open_cascade_brep",
        "shape_count": len(shapes),
        "solid_count": len(solids),
        "valid": valid,
        "volume_mm3": volume,
        "bounding_box_mm": bbox_mm,
    }


def export_pump_brep_package(
    pump_result,
    out_dir,
    *,
    formats: tuple[str, ...] = ("step", "stl"),
    samples: int = 61,
) -> dict[str, Any]:
    """Write per-part STEP+STL, per-role assembly STEP, and diagnostics.

    Every exported STEP is re-imported and gated on ``isValid``/volume like
    the wall path; a failed gate raises rather than shipping a bad body.
    """
    cq = _cq()
    out_dir = Path(out_dir)
    cad_dir = out_dir / "pump_brep"
    cad_dir.mkdir(parents=True, exist_ok=True)
    geom = pump_reference_geometry(pump_result)

    files: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    notes: list[str] = [
        "Pump B-rep CAD is meanline reference geometry, not production "
        "blade design.",
    ]

    def _export_solid(key: str, solid, stem: str):
        if "step" in formats:
            path = cad_dir / f"{stem}.step"
            cq.exporters.export(cq.Workplane(obj=solid), str(path))
            info = inspect_pump_step(path)
            if not (info["valid"] and info["volume_mm3"] > 0.0):
                raise RuntimeError(
                    f"pump B-rep re-import gate failed for {path.name}: "
                    f"{info['solid_count']} solids, valid={info['valid']}, "
                    f"volume={info['volume_mm3']:.6g} mm^3"
                )
            files[f"{key}_step"] = str(path)
            diagnostics[f"{key}_step"] = info
        if "stl" in formats:
            path = cad_dir / f"{stem}.stl"
            cq.exporters.export(cq.Workplane(obj=solid), str(path))
            files[f"{key}_stl"] = str(path)

    for role, comp in geom["components"].items():
        if comp.get("status") == "not_sized":
            notes.append(f"{role} pump B-rep skipped: {comp['reason']}")
            continue
        parts, part_notes = build_pump_parts(
            pump_result, role, samples=samples
        )
        notes.extend(part_notes)
        assembly = cq.Assembly(name=f"{role}_pump")
        for name, solid in parts.items():
            _export_solid(f"{role}_{name}", solid, f"{role}_{name}")
            assembly.add(solid, name=name)
        assembly_path = cad_dir / f"{role}_pump.step"
        if hasattr(assembly, "export"):
            assembly.export(str(assembly_path))
        else:  # cadquery < 2.7 keeps the old name
            assembly.save(str(assembly_path))
        info = inspect_pump_step(assembly_path)
        if not (info["valid"] and info["volume_mm3"] > 0.0):
            raise RuntimeError(
                f"pump assembly re-import gate failed for "
                f"{assembly_path.name}"
            )
        files[f"{role}_pump_assembly_step"] = str(assembly_path)
        diagnostics[f"{role}_pump_assembly_step"] = info

    battery = geom.get("battery")
    if battery:
        pack = _box_solid(_cq(), battery.get("box_m"), (0.0, 0.0, 0.0))
        if pack is not None:
            _export_solid("shared_battery_pack", pack, "shared_battery_pack")

    return {
        "dir": str(cad_dir),
        "files": files,
        "diagnostics": diagnostics,
        "step_representation": "open_cascade_brep",
        "geometry": geom,
        "notes": notes,
    }
