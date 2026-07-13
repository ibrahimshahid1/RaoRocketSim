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

import json
import math
from pathlib import Path
from typing import Any

from raosim.pump_cad import pump_reference_geometry
from raosim.pumps import impeller_blade_camber

_M_TO_MM = 1000.0
# Layout placeholders where the meanline solves no dimension (labeled, same
# values the Phase-0 mesh writer used); all solved dimensions come from the
# manifest.
_VANE_THICKNESS_PITCH_RATIO = 0.22
_SHAFT_FIT_RADIAL_CLEARANCE_MM = 0.015
_ROTOR_STATOR_RADIAL_CLEARANCE_MM = 0.15
_ROTOR_STATOR_AXIAL_CLEARANCE_MM = 0.15
_MOTOR_SHAFT_RADIAL_CLEARANCE_MM = 0.05


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


def _camber_ribbon_xy(
    camber,
    thickness_mm: float,
    inlet_thickness_mm: float | None = None,
) -> list[tuple[float, float]]:
    """Closed blade outline with a linear leading-edge-to-exit taper."""
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
    for i, (x, y) in enumerate(pts):
        x0, y0 = pts[max(i - 1, 0)]
        x1, y1 = pts[min(i + 1, n - 1)]
        tx, ty = x1 - x0, y1 - y0
        norm = math.hypot(tx, ty)
        if norm <= 0.0:
            raise ValueError("degenerate camber line (repeated points)")
        nx, ny = -ty / norm, tx / norm
        t0 = (
            float(inlet_thickness_mm)
            if inlet_thickness_mm is not None else float(thickness_mm)
        )
        thickness = t0 + (float(thickness_mm) - t0) * i / max(n - 1, 1)
        half = 0.5 * thickness
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
    t_b1 = _mm(
        impeller_comp.get("inlet_blade_thickness_m")
        or impeller_comp["blade_thickness_m"]
    )
    blade_count = int(impeller_comp["blade_count"])
    inlet_blade_count = int(
        impeller_comp.get("inlet_blade_count") or blade_count
    )
    splitter_count = int(impeller_comp.get("splitter_blade_count") or 0)
    if inlet_blade_count + splitter_count != blade_count:
        raise ValueError("main plus splitter blade count must equal exit count")
    if blade_count % inlet_blade_count != 0:
        raise ValueError(
            "full-length inlet blade count must divide the exit slot count"
        )
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
    solved_r_hub_eye = hub_pts[0][0]
    shaft_r = 0.5 * _mm(shaft_comp["diameter_m"])
    fit_clearance = _mm(
        channel_comp.get("shaft_fit_radial_clearance_m")
        or (_SHAFT_FIT_RADIAL_CLEARANCE_MM / _M_TO_MM)
    )
    hub_wall = _mm(
        channel_comp.get("impeller_hub_wall_thickness_m")
        or (max(t_b1, 0.30) / _M_TO_MM)
    )
    required_hub = shaft_r + fit_clearance + hub_wall
    if solved_r_hub_eye < required_hub - 1.0e-6:
        raise ValueError(
            "solved impeller hub is smaller than the upstream shaft/fit/wall "
            "envelope; CAD may not change hydraulic flow area"
        )
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
    def blade_solid(blade_camber):
        start_fraction = (
            (blade_camber[0]["radius_m"] - camber[0]["radius_m"])
            / max(camber[-1]["radius_m"] - camber[0]["radius_m"], 1e-12)
        )
        start_t = t_b1 + (t_b - t_b1) * start_fraction
        outline = _camber_ribbon_xy(blade_camber, t_b, start_t)
        # Blades overlap half the exit thickness into the backplate so the
        # fuse never relies on coincident tangent faces.
        return (
            cq.Workplane("XY", origin=(0.0, 0.0, 0.5 * t_b))
            .polyline(outline)
            .close()
            .extrude(-(width + 0.5 * t_b))
            .val()
        )

    main_blade = blade_solid(camber)
    # Enforce the exact solved tip diameter, then trim the blade tops to
    # the shroud curve revolve (full eye height at r1 -> b2 at the exit).
    main_blade = main_blade.intersect(
        _cylinder(cq, r2, -1.5 * width, t_b)
    )
    shroud_cutter = _revolve_profile(cq, [
        *shroud_pts,
        (1.05 * r2, shroud_pts[-1][1]),
        (1.05 * r2, -1.5 * width),
        (r1, -1.5 * width),
    ])
    main_blade = main_blade.cut(shroud_cutter)
    split_fraction = float(
        impeller_comp.get("splitter_start_radius_fraction") or 0.55
    )
    split_index = min(
        max(int(round(split_fraction * (len(camber) - 1))), 1),
        len(camber) - 2,
    )
    splitter = blade_solid(camber[split_index:]).intersect(
        _cylinder(cq, r2, -1.5 * width, t_b)
    ).cut(shroud_cutter)
    stride = blade_count // inlet_blade_count
    blades = []
    for slot in range(blade_count):
        template = main_blade if slot % stride == 0 else splitter
        blades.append(_rotate_z(cq, template, 360.0 * slot / blade_count))
    body = _fuse_all(body, blades)
    notes: list[str] = []
    notes.append(
        f"impeller uses {inlet_blade_count} full-length blades and "
        f"{splitter_count} downstream splitters; inlet/exit blockage="
        f"{impeller_comp.get('inlet_blockage_fraction')}/"
        f"{impeller_comp.get('exit_blockage_fraction')}"
    )

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

    bore_r = shaft_r + fit_clearance
    if bore_r < r_hub_eye - 0.20:
        bore = _cylinder(cq, bore_r, -1.5 * width, 2.0 * t_b)
        return body.cut(bore), notes
    raise ValueError(
        "resolved impeller hub cannot contain the solved shaft plus a "
        "machinable wall after clearance resolution"
    )


def build_inducer(cq, inducer_comp, shaft_comp):
    """Hub cylinder + helical swept blades, trimmed to the solved envelope.

    The blade section is swept along the parametric helix defined by the
    solved pitch/wrap (``inducer_helix``); an intersection with the tip
    cylinder enforces the exact D_ind and axial length.
    """
    r_tip = 0.5 * _mm(inducer_comp["diameter_m"])
    solved_r_hub = 0.5 * _mm(inducer_comp["hub_diameter_m"])
    length = _mm(inducer_comp["length_m"])
    pitch = _mm(inducer_comp["pitch_m"])
    blade_count = int(inducer_comp["blade_count"])
    solved_thickness = inducer_comp.get("leading_edge_thickness_m")
    if solved_thickness is None or float(solved_thickness) <= 0.0:
        raise ValueError(
            "inducer leading-edge thickness missing from the solved manifest"
        )
    # Do not replace the SP-8052 sizing result with a CAD-only diameter ratio.
    thickness = _mm(float(solved_thickness))
    shaft_r = 0.5 * _mm(shaft_comp["diameter_m"])
    fit_clearance = _mm(
        inducer_comp.get("shaft_fit_radial_clearance_m")
        or (_SHAFT_FIT_RADIAL_CLEARANCE_MM / _M_TO_MM)
    )
    hub_wall = _mm(
        inducer_comp.get("hub_wall_thickness_m")
        or (max(thickness, 0.20) / _M_TO_MM)
    )
    required_hub = shaft_r + fit_clearance + hub_wall
    if solved_r_hub < required_hub - 1.0e-6:
        raise ValueError(
            "solved inducer hub is smaller than the upstream shaft/fit/wall "
            "envelope; CAD may not change hydraulic inlet area"
        )
    r_hub = solved_r_hub
    if r_hub >= r_tip - thickness:
        raise ValueError(
            "solved shaft plus inducer blade-root wall does not fit inside "
            "the solved inducer tip diameter"
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
    bore_r = shaft_r + fit_clearance
    bore = _cylinder(cq, bore_r, -0.2 * length, 1.2 * length)
    return body.cut(bore), None


def build_diffuser_ring(cq, ring_comp):
    """Vaned diffuser ring: side plates + vanes set at the solved flow angle."""
    # The meanline inner radius is the impeller tip radius.  A literal use
    # produces a zero-clearance rotor/stator rub; add a documented cold-build
    # clearance while preserving the solved outer envelope.
    inner = (
        _mm(ring_comp["inner_radius_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    )
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


def build_volute_casing(
    cq, ring_comp, ports_comp, shaft_comp=None, *, sections: int = 24
):
    """Construction-only one-piece casing envelope used before planar split.

    A(theta) = A_exit * theta / 2pi is the explicitly labelled
    constant-mean-velocity collection schedule (not the distinct SP-8109
    constant-moment-of-momentum construction). The operating export always
    routes this intermediate through :func:`build_split_volute_casing`;
    this closed body is not an assemblable pump part. Earlier CAD fused that *fluid
    volume* to a material ring, so the advertised volute and outlet were solid
    metal.  This builder creates an outer scroll/boss envelope and subtracts
    the scheduled scroll, tangential outlet bore, impeller cavity, and axial
    inlet bore.  Front/back covers close the package around those flow paths.
    """
    casing_r = _mm(ring_comp["casing_inner_radius_m"])
    t_wall = _mm(ring_comp["casing_wall_thickness_m"])
    b_v = _mm(ring_comp["axial_width_m"])
    exit_area_mm2 = float(ring_comp["volute_exit_area_m2"]) * _M_TO_MM ** 2
    a_exit = math.sqrt(exit_area_mm2 / math.pi)
    z_mid = -0.5 * b_v

    shell = _washer(
        cq, casing_r, casing_r + t_wall,
        -b_v - 2.0 * t_wall, 2.0 * t_wall,
    )
    fluid_wires = []
    outer_wires = []
    for i in range(1, sections + 1):
        theta = 2.0 * math.pi * i / sections
        radius = a_exit * math.sqrt(theta / (2.0 * math.pi))
        center = cq.Vector(
            casing_r * math.cos(theta), casing_r * math.sin(theta), z_mid
        )
        normal = cq.Vector(-math.sin(theta), math.cos(theta), 0.0)
        fluid_wires.append(cq.Wire.makeCircle(radius, center, normal))
        outer_wires.append(
            cq.Wire.makeCircle(radius + t_wall, center, normal)
        )
    fluid_scroll = cq.Solid.makeLoft(fluid_wires, True)
    outer_scroll = cq.Solid.makeLoft(outer_wires, True)

    port_d = ports_comp.get("outlet_equivalent_diameter_m")
    if port_d is None or port_d <= 0.0:
        raise ValueError(
            "volute outlet port diameter missing from the pump manifest"
        )
    outlet_fluid = cq.Solid.makeCylinder(
        0.5 * _mm(port_d), 3.0 * a_exit,
        cq.Vector(casing_r, 0.0, z_mid), cq.Vector(0.0, 1.0, 0.0),
    )
    outlet_boss = cq.Solid.makeCylinder(
        0.5 * _mm(port_d) + t_wall, 3.0 * a_exit,
        cq.Vector(casing_r, 0.0, z_mid), cq.Vector(0.0, 1.0, 0.0),
    )

    inlet_d = ports_comp.get("inlet_diameter_m")
    if inlet_d is None or inlet_d <= 0.0:
        raise ValueError("pump casing inlet diameter missing from manifest")
    inlet_r = 0.5 * _mm(inlet_d)
    inlet_bore_r = inlet_r + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    outer_cover_r = casing_r + t_wall
    front_cover = _washer(
        cq, inlet_bore_r, outer_cover_r,
        -b_v - 2.0 * t_wall, -b_v - t_wall,
    )
    shaft_r = (
        0.5 * _mm(shaft_comp["diameter_m"])
        + _MOTOR_SHAFT_RADIAL_CLEARANCE_MM
        if shaft_comp else inlet_r
    )
    back_cover = _washer(
        cq, shaft_r, outer_cover_r, t_wall, 2.0 * t_wall,
    )
    inlet_boss = _washer(
        cq, inlet_bore_r, inlet_bore_r + t_wall,
        -b_v - 4.0 * t_wall, -b_v - t_wall,
    )
    inlet_fluid = _cylinder(
        cq, inlet_bore_r,
        -b_v - 4.1 * t_wall, -b_v + 0.5 * t_wall,
    )

    material = _fuse_all(
        shell,
        [outer_scroll, outlet_boss, front_cover, back_cover, inlet_boss],
    )
    # Seat the separately exported diffuser with positive radial/axial
    # clearance.  This envelope also joins the impeller cavity/inlet to the
    # collecting scroll; it is intentionally an assembly-clearance/flow
    # envelope, not a claim that the vane metal itself is fluid.
    diffuser_clearance = _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    diffuser_pocket = _washer(
        cq,
        max(_mm(ring_comp["inner_radius_m"]) - diffuser_clearance, 0.0),
        _mm(ring_comp["outer_radius_m"]) + diffuser_clearance,
        -b_v - t_wall - diffuser_clearance,
        t_wall + diffuser_clearance,
    )
    rotor_pocket = _cylinder(
        cq,
        _mm(ring_comp["inner_radius_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM,
        -b_v - 4.1 * t_wall,
        t_wall + diffuser_clearance,
    )
    flow_passage = fluid_scroll.fuse(
        outlet_fluid, inlet_fluid, rotor_pocket, diffuser_pocket
    )
    if not flow_passage.isValid() or len(flow_passage.Solids()) != 1:
        raise RuntimeError(
            "pump inlet/impeller/diffuser/volute/outlet envelope is not one "
            "connected valid passage"
        )
    casing = material.cut(flow_passage)
    # Outer-scroll loft sections can bulge across the axis above/below the
    # main cover planes.  Maintain a shaft corridor through the complete
    # scroll envelope, not just through the rear cover washer.
    shaft_corridor = _cylinder(
        cq,
        shaft_r,
        z_mid - a_exit - 2.0 * t_wall,
        z_mid + a_exit + 2.0 * t_wall,
    )
    casing = casing.cut(shaft_corridor)
    if not casing.isValid() or len(casing.Solids()) != 1:
        raise RuntimeError(
            "hollow volute casing Boolean did not produce one valid material solid"
        )
    return casing


def _volute_flow_envelope(
    cq, ring_comp, ports_comp, *, sections: int = 24
):
    """Rebuild the connected inlet-to-outlet void used by the casing."""
    casing_r = _mm(ring_comp["casing_inner_radius_m"])
    t_wall = _mm(ring_comp["casing_wall_thickness_m"])
    b_v = _mm(ring_comp["axial_width_m"])
    exit_area_mm2 = float(ring_comp["volute_exit_area_m2"]) * _M_TO_MM ** 2
    a_exit = math.sqrt(exit_area_mm2 / math.pi)
    z_mid = -0.5 * b_v
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
    outlet = cq.Solid.makeCylinder(
        0.5 * _mm(ports_comp["outlet_equivalent_diameter_m"]),
        3.0 * a_exit,
        cq.Vector(casing_r, 0.0, z_mid),
        cq.Vector(0.0, 1.0, 0.0),
    )
    inlet_bore_r = (
        0.5 * _mm(ports_comp["inlet_diameter_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    )
    inlet = _cylinder(
        cq, inlet_bore_r, -b_v - 4.1 * t_wall, -b_v + 0.5 * t_wall
    )
    inner = _mm(ring_comp["inner_radius_m"])
    outer = _mm(ring_comp["outer_radius_m"])
    clearance = _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    rotor = _cylinder(
        cq, inner + clearance,
        -b_v - 4.1 * t_wall, t_wall + clearance,
    )
    diffuser = _washer(
        cq, max(inner - clearance, 0.0), outer + clearance,
        -b_v - t_wall - clearance, t_wall + clearance,
    )
    passage = scroll.fuse(outlet, inlet, rotor, diffuser)
    return passage, {
        "scroll": scroll,
        "outlet": outlet,
        "inlet": inlet,
        "rotor": rotor,
        "diffuser": diffuser,
        "a_exit_mm": a_exit,
        "z_mid_mm": z_mid,
        "casing_radius_mm": casing_r,
        "wall_mm": t_wall,
        "vane_width_mm": b_v,
    }


def _split_joint_hole_layout(ring_comp, ports_comp) -> dict[str, Any]:
    joint = dict(ring_comp.get("split_casing_joint") or {})
    casing_r = _mm(ring_comp["casing_inner_radius_m"])
    a_exit = math.sqrt(
        float(ring_comp["volute_exit_area_m2"]) * _M_TO_MM**2 / math.pi
    )
    port_r = 0.5 * _mm(ports_comp["outlet_equivalent_diameter_m"])
    flange_r = _mm(
        joint.get("flange_outer_radius_m")
        or ((casing_r + 2.0 * a_exit + 8.0) / _M_TO_MM)
    )
    gasket_land = _mm(joint.get("gasket_land_width_m") or 2.0e-3)
    hole_r = 0.5 * _mm(joint.get("bolt_hole_diameter_m") or 3.45e-3)
    wet_r = casing_r + a_exit
    bolt_r = 0.5 * (wet_r + gasket_land + flange_r - hole_r)
    requested = int(joint.get("body_bolt_count") or 8)
    candidates = []
    for k in range(48):
        angle = 2.0 * math.pi * (k + 0.5) / 48.0
        x, y = bolt_r * math.cos(angle), bolt_r * math.sin(angle)
        y_near = min(max(y, 0.0), 3.0 * a_exit)
        outlet_distance = math.hypot(x - casing_r, y - y_near)
        if outlet_distance > port_r + gasket_land + 1.5 * hole_r:
            candidates.append((x, y))
    if len(candidates) < requested:
        raise ValueError("split casing has insufficient clear body-bolt sectors")
    stride = len(candidates) / requested
    body_holes = [candidates[int(i * stride)] for i in range(requested)]
    neck_offset = port_r + gasket_land + 2.0 * hole_r
    neck_holes: list[tuple[float, float]] = []
    outlet_end = 3.0 * a_exit + port_r
    for y in (
        outlet_end + gasket_land + 2.0 * hole_r,
        outlet_end + gasket_land + 6.0 * hole_r,
    ):
        neck_holes.extend([
            (casing_r - neck_offset, y),
            (casing_r + neck_offset, y),
        ])
    dowel_r = max(0.35 * hole_r, 0.5)
    dowels = [(-0.80 * flange_r, -0.20 * flange_r),
              (-0.80 * flange_r, 0.20 * flange_r)]
    return {
        "flange_outer_radius_mm": flange_r,
        "gasket_land_mm": gasket_land,
        "bolt_hole_radius_mm": hole_r,
        "body_bolt_centers_mm": body_holes,
        "outlet_neck_bolt_centers_mm": neck_holes,
        "dowel_radius_mm": dowel_r,
        "dowel_centers_mm": dowels,
    }


def build_split_volute_casing(
    cq, ring_comp, ports_comp, shaft_comp=None, *, sections: int = 24
):
    """Build axially separable rear body and front cover casing halves."""
    one_piece = build_volute_casing(
        cq, ring_comp, ports_comp, shaft_comp, sections=sections
    )
    passage, primitives = _volute_flow_envelope(
        cq, ring_comp, ports_comp, sections=sections
    )
    layout = _split_joint_hole_layout(ring_comp, ports_comp)
    z_mid = primitives["z_mid_mm"]
    joint = ring_comp.get("split_casing_joint") or {}
    flange_t = _mm(
        joint.get("flange_thickness_m")
        or ring_comp["casing_wall_thickness_m"]
    )
    flange_r = layout["flange_outer_radius_mm"]
    round_flange = _cylinder(
        cq, flange_r, z_mid - 0.5 * flange_t, z_mid + 0.5 * flange_t
    )
    a_exit = primitives["a_exit_mm"]
    casing_r = primitives["casing_radius_mm"]
    port_r = 0.5 * _mm(ports_comp["outlet_equivalent_diameter_m"])
    neck_half_width = (
        port_r + layout["gasket_land_mm"]
        + 4.5 * layout["bolt_hole_radius_mm"]
    )
    neck = cq.Solid.makeBox(
        2.0 * neck_half_width,
        max(
            3.0 * a_exit + neck_half_width,
            max(y for _, y in layout["outlet_neck_bolt_centers_mm"])
            + 3.0 * layout["bolt_hole_radius_mm"],
        ),
        flange_t,
        cq.Vector(
            casing_r - neck_half_width,
            0.0,
            z_mid - 0.5 * flange_t,
        ),
    )
    jointed = one_piece.fuse(round_flange, neck).cut(passage)
    all_holes = [
        *layout["body_bolt_centers_mm"],
        *layout["outlet_neck_bolt_centers_mm"],
    ]
    z0 = z_mid - flange_t
    z1 = z_mid + flange_t
    for x, y in all_holes:
        hole = cq.Solid.makeCylinder(
            layout["bolt_hole_radius_mm"], z1 - z0,
            cq.Vector(x, y, z0), cq.Vector(0.0, 0.0, 1.0),
        )
        if float(abs(hole.intersect(passage).Volume())) > 1.0e-6:
            raise ValueError("split-casing bolt pattern intersects the flow path")
        jointed = jointed.cut(hole)
    for x, y in layout["dowel_centers_mm"]:
        dowel = cq.Solid.makeCylinder(
            layout["dowel_radius_mm"], z1 - z0,
            cq.Vector(x, y, z0), cq.Vector(0.0, 0.0, 1.0),
        )
        if float(abs(dowel.intersect(passage).Volume())) > 1.0e-6:
            raise ValueError("split-casing dowel pattern intersects the flow path")
        jointed = jointed.cut(dowel)

    box = jointed.BoundingBox()
    pad = 2.0 * max(box.xlen, box.ylen, box.zlen, 1.0)
    x0, y0 = box.xmin - pad, box.ymin - pad
    dx, dy = box.xlen + 2.0 * pad, box.ylen + 2.0 * pad
    front_box = cq.Solid.makeBox(
        dx, dy, z_mid - (box.zmin - pad), cq.Vector(x0, y0, box.zmin - pad)
    )
    body_box = cq.Solid.makeBox(
        dx, dy, box.zmax + pad - z_mid, cq.Vector(x0, y0, z_mid)
    )
    cover = jointed.intersect(front_box)
    body = jointed.intersect(body_box)
    if not all(s.isValid() and len(s.Solids()) == 1 for s in (body, cover)):
        raise RuntimeError("split volute casing did not produce two valid halves")
    layout.update({
        "parting_plane_z_mm": z_mid,
        "jointed_volume_mm3": float(abs(jointed.Volume())),
        "body_volume_mm3": float(abs(body.Volume())),
        "cover_volume_mm3": float(abs(cover.Volume())),
        "minimum_scroll_section_diameter_mm": 2.0 * a_exit / math.sqrt(sections),
        "selected_machining_tool_diameter_mm": _mm(
            joint.get("selected_scroll_tool_diameter_m") or 5.0e-4
        ),
    })
    return body, cover, layout


def audit_volute_flow_passage(
    cq, ring_comp, ports_comp, shaft_comp=None, *, sections: int = 24
) -> dict[str, Any]:
    """Classify continuity of the casing's inlet-to-outlet void envelope.

    This repeats the inexpensive construction primitives used by
    :func:`build_volute_casing` and reports each required positive-volume
    handoff.  It deliberately checks the *void envelope*; impeller/diffuser
    blade-to-blade area and hydraulic loss still require CFD/cold flow.
    """
    casing_r = _mm(ring_comp["casing_inner_radius_m"])
    t_wall = _mm(ring_comp["casing_wall_thickness_m"])
    b_v = _mm(ring_comp["axial_width_m"])
    exit_area_mm2 = float(ring_comp["volute_exit_area_m2"]) * _M_TO_MM ** 2
    a_exit = math.sqrt(exit_area_mm2 / math.pi)
    z_mid = -0.5 * b_v
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
    outlet = cq.Solid.makeCylinder(
        0.5 * _mm(ports_comp["outlet_equivalent_diameter_m"]),
        3.0 * a_exit,
        cq.Vector(casing_r, 0.0, z_mid),
        cq.Vector(0.0, 1.0, 0.0),
    )
    inlet_bore_r = (
        0.5 * _mm(ports_comp["inlet_diameter_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    )
    inlet = _cylinder(
        cq, inlet_bore_r, -b_v - 4.1 * t_wall, -b_v + 0.5 * t_wall
    )
    inner = _mm(ring_comp["inner_radius_m"])
    outer = _mm(ring_comp["outer_radius_m"])
    clearance = _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    rotor = _cylinder(
        cq, inner + clearance,
        -b_v - 4.1 * t_wall, t_wall + clearance,
    )
    diffuser = _washer(
        cq, max(inner - clearance, 0.0), outer + clearance,
        -b_v - t_wall - clearance, t_wall + clearance,
    )
    handoffs = {
        "inlet_to_impeller_mm3": float(inlet.intersect(rotor).Volume()),
        "impeller_to_diffuser_mm3": float(rotor.intersect(diffuser).Volume()),
        "diffuser_to_scroll_mm3": float(diffuser.intersect(scroll).Volume()),
        "scroll_to_outlet_mm3": float(scroll.intersect(outlet).Volume()),
    }
    passage = scroll.fuse(outlet, inlet, rotor, diffuser)
    connected = (
        passage.isValid()
        and len(passage.Solids()) == 1
        and min(handoffs.values()) > 1.0e-6
    )
    return {
        "passed": bool(connected),
        "status": "pass" if connected else "fail",
        "single_connected_solid": len(passage.Solids()) == 1,
        "valid": bool(passage.isValid()),
        "void_envelope_volume_mm3": float(abs(passage.Volume())),
        "handoff_overlaps": handoffs,
        "model": "connected_casing_void_envelope_not_blade_to_blade_cfd",
    }


def audit_split_casing_manufacturability(
    body, cover, layout: dict[str, Any], ring_comp, ports_comp
) -> dict[str, Any]:
    """Gate the bounded claims made by the separable casing topology."""
    overlap = float(abs(body.intersect(cover).Volume()))
    summed = float(abs(body.Volume()) + abs(cover.Volume()))
    reference = max(float(layout["jointed_volume_mm3"]), 1e-12)
    closure = abs(summed - reference) / reference
    joint = ring_comp.get("split_casing_joint") or {}
    tool_clearance = (
        float(layout["minimum_scroll_section_diameter_mm"])
        - float(layout["selected_machining_tool_diameter_mm"])
    )
    inlet_radial_clearance = _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    diffuser_aperture = 2.0 * (
        _mm(ring_comp["outer_radius_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    )
    required_diffuser = 2.0 * _mm(ring_comp["outer_radius_m"])
    bolt_pass = bool(joint.get("bolt_screen_passed", False))
    passed = bool(
        body.isValid()
        and cover.isValid()
        and len(body.Solids()) == 1
        and len(cover.Solids()) == 1
        and overlap <= 1.0e-6
        and closure <= 1.0e-6
        and tool_clearance >= 0.0
        and inlet_radial_clearance > 0.0
        and diffuser_aperture > required_diffuser
        and bolt_pass
        and float(joint.get("gasket_land_width_m", 0.0)) > 0.0
    )
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "body_valid_single_solid": body.isValid() and len(body.Solids()) == 1,
        "cover_valid_single_solid": cover.isValid() and len(cover.Solids()) == 1,
        "material_overlap_mm3": overlap,
        "relative_volume_closure_error": closure,
        "scroll_tool_clearance_mm": tool_clearance,
        "front_cover_over_inducer_radial_clearance_mm": inlet_radial_clearance,
        "opened_diffuser_aperture_mm": diffuser_aperture,
        "required_diffuser_envelope_mm": required_diffuser,
        "bolt_clamp_screen_passed": bolt_pass,
        "gasket_land_width_m": joint.get("gasket_land_width_m"),
        "parting_plane_z_mm": layout["parting_plane_z_mm"],
        "body_bolt_count": len(layout["body_bolt_centers_mm"]),
        "outlet_neck_bolt_count": len(
            layout["outlet_neck_bolt_centers_mm"]
        ),
        "dowel_count": len(layout["dowel_centers_mm"]),
        "machining_claim": (
            "scroll halves are directly exposed at the centerplane and the "
            "selected nominal tool fits the smallest modeled section"
        ),
        "qualification": (
            "assembly/machining topology only; gasket selection, flange FEA, "
            "bolt preload/threads, dowel fits, tolerance/thermal stack, shaft "
            "retention, bearings, seals, rotordynamics, proof and cold-flow "
            "tests remain required"
        ),
    }


def build_shaft(cq, shaft_comp, z_start_mm: float):
    return _cylinder(cq, 0.5 * _mm(shaft_comp["diameter_m"]),
                     z_start_mm, z_start_mm + _mm(shaft_comp["span_m"]))


def build_motor(cq, motor_comp, shaft_comp, z_start_mm: float):
    d = _mm(motor_comp["diameter_m"])
    length = _mm(motor_comp["length_m"])
    if d <= 0.0 or length <= 0.0:
        return None
    body = _cylinder(cq, 0.5 * d, z_start_mm, z_start_mm + length)
    bore_r = (
        0.5 * _mm(shaft_comp["diameter_m"])
        + _MOTOR_SHAFT_RADIAL_CLEARANCE_MM
    )
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
    pump_result,
    role: str,
    *,
    samples: int = 61,
    build_diagnostics: dict[str, Any] | None = None,
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
        clearance_target = (
            -_mm(impeller_comp["axial_width_m"])
            - _ROTOR_STATOR_AXIAL_CLEARANCE_MM
            - _mm(comp["inducer"]["length_m"])
        )
        if z_le > clearance_target:
            notes.append(
                f"{role}: inducer moved {z_le - clearance_target:.3f} mm "
                "upstream to provide nonzero inducer/impeller axial clearance"
            )
            z_le = clearance_target
        parts["inducer"] = inducer.translate(cq.Vector(0, 0, z_le))

    if "diffuser_volute" in comp:
        parts["diffuser_ring"] = build_diffuser_ring(
            cq, comp["diffuser_volute"]
        )
        body, cover, split_layout = build_split_volute_casing(
            cq, comp["diffuser_volute"], comp["ports"], shaft_comp
        )
        split_gate = audit_split_casing_manufacturability(
            body,
            cover,
            split_layout,
            comp["diffuser_volute"],
            comp["ports"],
        )
        if not split_gate["passed"]:
            raise RuntimeError(
                f"{role} split-casing manufacturability gate failed: "
                f"{split_gate}"
            )
        parts["volute_body"] = body
        parts["volute_front_cover"] = cover
        if build_diagnostics is not None:
            build_diagnostics["split_casing_layout"] = split_layout
            build_diagnostics["split_casing_manufacturability"] = split_gate

    z_shaft_start = stations.get(
        "inlet_port", -0.55 * _mm(shaft_comp["span_m"])
    )
    parts["shaft"] = build_shaft(cq, shaft_comp, z_shaft_start)

    t_b = _mm(impeller_comp["blade_thickness_m"])
    casing_t = _mm(
        comp.get("diffuser_volute", {}).get("casing_wall_thickness_m") or 0.0
    )
    motor_z = max(
        t_b,
        2.0 * casing_t + _ROTOR_STATOR_AXIAL_CLEARANCE_MM,
    )
    casing_parts = [
        parts[name] for name in ("volute_body", "volute_front_cover")
        if name in parts
    ]
    if casing_parts:
        motor_z = max(
            motor_z,
            max(float(part.BoundingBox().zmax) for part in casing_parts)
            + _ROTOR_STATOR_AXIAL_CLEARANCE_MM,
        )
    if "motor" in comp:
        motor = build_motor(cq, comp["motor"], shaft_comp, motor_z)
        if motor is not None:
            parts["motor"] = motor
    if "inverter" in comp:
        motor_d = _mm(comp.get("motor", {}).get("diameter_m") or 0.0)
        box_m = comp["inverter"].get("box_m")
        offset_x = 0.5 * motor_d + 0.7 * _mm((box_m or [0.0])[0])
        inverter = _box_solid(
            cq, box_m,
            (offset_x, 0.0,
             motor_z + 0.5 * _mm((box_m or [0, 0, 0])[2])),
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
    interference = audit_pump_component_interference(parts)
    if not interference["passed"]:
        raise RuntimeError(
            f"{role} pump component interference gate failed: {interference}"
        )
    return parts, notes


def audit_pump_component_interference(parts: dict[str, Any]) -> dict[str, Any]:
    """Reject positive-volume collisions in a named pump assembly.

    Shaft/hub and rotor/stator fits now carry explicit clearances, so there is
    no collision allow-list: any shared material volume is an invalid package.
    Touching stationary mating faces are permitted.
    """
    names = list(parts)
    records: list[dict[str, Any]] = []
    maximum = 0.0
    tolerance = 1.0e-6  # mm^3, below machining/CAD kernel significance
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            try:
                common = parts[left].intersect(parts[right])
                volume = float(sum(abs(s.Volume()) for s in common.Solids()))
            except Exception as exc:
                return {
                    "passed": False,
                    "status": "failed_to_evaluate",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pairs": records,
                }
            maximum = max(maximum, volume)
            records.append({
                "components": [left, right],
                "overlap_mm3": volume,
                "status": "pass" if volume <= tolerance else "fail",
            })
    return {
        "passed": maximum <= tolerance,
        "status": "pass" if maximum <= tolerance else "fail",
        "tolerance_mm3": tolerance,
        "maximum_overlap_mm3": maximum,
        "pairs": records,
    }


def audit_pump_clearances(parts: dict[str, Any], comp: dict) -> dict[str, Any]:
    """Report every intentionally modeled cold-build running clearance.

    These are CAD construction clearances, not tolerance/thermal-stack
    qualifications.  Keeping them explicit prevents a zero-overlap assembly
    from being mistaken for proof that a positive running gap exists.
    """
    impeller_r = 0.5 * _mm(comp["impeller"]["outer_diameter_m"])
    shaft_fit = _mm(
        comp.get("meridional_channel", {}).get(
            "shaft_fit_radial_clearance_m",
            _SHAFT_FIT_RADIAL_CLEARANCE_MM / _M_TO_MM,
        )
    )
    inducer_fit = _mm(
        comp.get("inducer", {}).get(
            "shaft_fit_radial_clearance_m", shaft_fit / _M_TO_MM
        )
    )
    diffuser_inner = (
        _mm(comp["diffuser_volute"]["inner_radius_m"])
        + _ROTOR_STATOR_RADIAL_CLEARANCE_MM
    )
    values = {
        "shaft_to_impeller_bore_radial_mm": shaft_fit,
        "shaft_to_inducer_bore_radial_mm": inducer_fit,
        "shaft_to_motor_bore_radial_mm": _MOTOR_SHAFT_RADIAL_CLEARANCE_MM,
        "impeller_to_diffuser_radial_mm": diffuser_inner - impeller_r,
        "diffuser_to_casing_pocket_radial_mm":
            _ROTOR_STATOR_RADIAL_CLEARANCE_MM,
    }
    if "inducer" in parts:
        values["inducer_to_impeller_axial_mm"] = (
            float(parts["impeller"].BoundingBox().zmin)
            - float(parts["inducer"].BoundingBox().zmax)
        )
    casing_parts = [
        parts[name] for name in ("volute_body", "volute_front_cover")
        if name in parts
    ]
    if "motor" in parts and casing_parts:
        values["casing_to_motor_axial_mm"] = (
            float(parts["motor"].BoundingBox().zmin)
            - max(float(part.BoundingBox().zmax) for part in casing_parts)
        )
    axial_engagements: dict[str, float] = {}
    if "shaft" in parts:
        shaft_box = parts["shaft"].BoundingBox()
        for name in ("inducer", "impeller", "motor"):
            if name not in parts:
                continue
            box = parts[name].BoundingBox()
            axial_engagements[f"shaft_through_{name}_mm"] = max(
                0.0,
                min(float(shaft_box.zmax), float(box.zmax))
                - max(float(shaft_box.zmin), float(box.zmin)),
            )
    passed = bool(values) and all(
        value > 1.0e-6 for value in values.values()
    ) and bool(axial_engagements) and all(
        value > 1.0e-6 for value in axial_engagements.values()
    )
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "clearances": values,
        "axial_shaft_engagements": axial_engagements,
        "qualification": (
            "positive_nominal_cold_build_gaps_only; tolerance stack, thermal "
            "growth, deflection, wear, keys/splines/couplings, bearings, and "
            "rotordynamics remain required"
        ),
    }


def audit_meanline_geometry_fidelity(comp: dict) -> dict[str, Any]:
    """Identity-audit the upstream shaft/hub solve against CAD requirements.

    CAD no longer enlarges either hub.  Any deviation here is a regression in
    the coupled hydraulic/mechanical solve and is an exporter failure, not a
    packaging-only body that may be released.
    """
    shaft_r = 0.5 * float(comp["shaft"]["diameter_m"])
    impeller = comp["impeller"]
    channel = comp["meridional_channel"]
    imp_solved = float(channel["hub_curve"][0]["r_m"])
    fit_clearance = float(
        channel.get("shaft_fit_radial_clearance_m")
        or (_SHAFT_FIT_RADIAL_CLEARANCE_MM / _M_TO_MM)
    )
    imp_wall = float(
        channel.get("impeller_hub_wall_thickness_m")
        or max(
            float(impeller.get("inlet_blade_thickness_m") or 0.0),
            0.30 / _M_TO_MM,
        )
    )
    imp_required = max(
        imp_solved,
        shaft_r + fit_clearance + imp_wall,
    )
    imp_outer = float(channel["shroud_curve"][0]["r_m"])

    inducer = comp.get("inducer")
    cases = [
        ("impeller_eye_hub", imp_solved, imp_required, imp_outer)
    ]
    if inducer:
        ind_solved = 0.5 * float(inducer["hub_diameter_m"])
        ind_clearance = float(
            inducer.get("shaft_fit_radial_clearance_m") or fit_clearance
        )
        ind_wall = float(
            inducer.get("hub_wall_thickness_m")
            or max(
                float(inducer["leading_edge_thickness_m"]),
                0.20 / _M_TO_MM,
            )
        )
        ind_required = max(
            ind_solved,
            shaft_r + ind_clearance + ind_wall,
        )
        cases.append((
            "inducer_hub",
            ind_solved,
            ind_required,
            0.5 * float(inducer["diameter_m"]),
        ))

    records = []
    for feature, solved, cad, outer in cases:
        solved_area = math.pi * max(outer**2 - solved**2, 0.0)
        cad_area = math.pi * max(outer**2 - cad**2, 0.0)
        changed = cad > solved + 1.0e-12
        records.append({
            "feature": feature,
            "changed": changed,
            "solved_hub_radius_m": solved,
            "cad_hub_radius_m": cad,
            "hub_radius_increase_m": cad - solved,
            "solved_inlet_area_m2": solved_area,
            "cad_inlet_area_m2": cad_area,
            "inlet_area_reduction_m2": solved_area - cad_area,
            "inlet_area_reduction_fraction": (
                (solved_area - cad_area) / solved_area
                if solved_area > 0.0 else float("nan")
            ),
        })
    changed_records = [record for record in records if record["changed"]]
    return {
        "passed": not changed_records,
        "status": (
            "pass" if not changed_records else "requires_meanline_resolve"
        ),
        "deviations": changed_records,
        "all_features": records,
        "qualification": (
            "Identity check only: annular continuity, inlet velocity, blade "
            "blockage, shaft fit, and hub wall were solved upstream. Any "
            "reported change is a hard regression."
        ),
    }


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
            # CadQuery BoundBox.add returns a new box; it does not mutate.
            bbox = bbox.add(solid.BoundingBox())
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
    assembly_gates: dict[str, Any] = {}
    notes: list[str] = [
        "Pump B-rep CAD is meanline reference geometry, not production "
        "blade design.",
        "Operating volute CAD is exported as rear body and removable front "
        "cover; the one-piece hollow body is construction-only and is never "
        "written as the operating casing.",
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
        build_gates: dict[str, Any] = {}
        parts, part_notes = build_pump_parts(
            pump_result,
            role,
            samples=samples,
            build_diagnostics=build_gates,
        )
        notes.extend(part_notes)
        fidelity_gate = audit_meanline_geometry_fidelity(comp)
        if not fidelity_gate["passed"]:
            raise RuntimeError(
                f"{role} pump CAD would change solved hydraulic geometry: "
                f"{fidelity_gate}"
            )
        assembly_gates[f"{role}_meanline_geometry_fidelity"] = fidelity_gate
        assembly_gates[f"{role}_component_interference"] = (
            audit_pump_component_interference(parts)
        )
        clearance_gate = audit_pump_clearances(parts, comp)
        if not clearance_gate["passed"]:
            raise RuntimeError(
                f"{role} pump nominal-clearance gate failed: {clearance_gate}"
            )
        assembly_gates[f"{role}_nominal_clearances"] = clearance_gate
        flow_gate = audit_volute_flow_passage(
            cq, comp["diffuser_volute"], comp["ports"], comp.get("shaft")
        )
        if not flow_gate["passed"]:
            raise RuntimeError(
                f"{role} pump casing flow-passage gate failed: {flow_gate}"
            )
        assembly_gates[f"{role}_casing_flow_passage"] = flow_gate
        if "split_casing_manufacturability" in build_gates:
            assembly_gates[f"{role}_split_casing_manufacturability"] = (
                build_gates["split_casing_manufacturability"]
            )
            diagnostics[f"{role}_split_casing_layout"] = build_gates[
                "split_casing_layout"
            ]
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

    units_path = cad_dir / "pump_cad_units.json"
    units_path.write_text(json.dumps({
        "schema": "raosim.cad_units.v1",
        "public_api_linear_unit": "m",
        "neutral_file_linear_unit": "mm",
        "volume_unit": "mm^3",
        "stl_unit_policy": (
            "STL has no embedded unit; all numeric coordinates in this "
            "package are millimetres"
        ),
        "files": {
            Path(value).name: "mm"
            for value in files.values()
            if str(value).lower().endswith((".step", ".stl"))
        },
    }, indent=2) + "\n", encoding="utf-8")
    files["cad_units"] = str(units_path)

    return {
        "dir": str(cad_dir),
        "files": files,
        "diagnostics": diagnostics,
        "assembly_gates": assembly_gates,
        "step_representation": "open_cascade_brep",
        "geometry": geom,
        "cold_flow_release_ready": False,
        "hardware_qualified": False,
        "external_release_blockers": [
            "selected gasket and qualified split-flange preload/thread/dowel design",
            "bearing, seal, wear-ring, thrust-balance, torque-coupling, and lubrication design",
            "tolerance stack, thermal growth, rotordynamics, and burst/FEA evidence",
            "pump-map CFD plus cavitation/NPSH and cold-flow test evidence",
            "released manufacturing drawings, proof test, and acceptance limits",
        ],
        "notes": notes,
    }
