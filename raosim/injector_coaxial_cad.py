"""injector_coaxial_cad.py — coaxial (TRW/Nardi) pintle injector CAD.

The machined-pintle export in :mod:`raosim.injector_cad` modelled the injector
as two concentric ring-manifolds cut into a faceplate plus radial transfer
drills.  That scheme cannot realise a pintle's coaxial flow topology: the inner
(radial) propellant lives in the *central bore* while the outer (axial)
propellant lives in the *surrounding annulus*, so an outer-radius ring manifold
can neither feed the central bore nor stay sealed from it.

This module builds the physically-coherent architecture from Rezende (Nardi),
*Experiments with the Pintle Injector*, and the TRW/Elverum patent lineage — the
five solids of revolution of the classic exploded view:

* **pintle body**  — inner propellant enters axially at the top and flows down
  the central bore;
* **pintle tip** (replaceable) — the ring of radial metering exits (round jets,
  the paper's preferred multi-jet, or rectangular slots) at the skip-distance
  line, one pintle-diameter into the chamber;
* **injector body** — outer propellant enters *laterally* into a *toroidal
  plenum* wrapping the pintle body;
* **orifice plate** — a ring of holes distributing the outer propellant evenly
  down into the annular collector;
* **faceplate** — the continuous **metering gap** between the pintle body and
  the faceplate that meters the axial sheet onto the pintle surface.

Sealed by construction: the inner propellant is fully enclosed by the pintle
body/tip walls, the outer propellant lives in the surrounding plenum/annulus,
and the two only meet in the chamber at the tip.  The layout is driven entirely
by the hydraulic sizing already resolved in
:func:`raosim.injector_cad.resolve_machined_pintle_layout`, so it generalises to
any solved pintle injector.

Convention (shared with ``injector_cad``): axis +Z, injector face at Z=0,
chamber downstream at Z>0, feed manifolds at Z<0.  Geometry is built in SI
metres and scaled to millimetres at the STEP boundary (repo convention;
see :func:`raosim.injector_cad._to_mm_step_solid`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from raosim.injector_cad import (
    _cq,
    _points_on_circle,
    _to_mm_step_solid,
    cadquery_available,
    inspect_machined_step,
    resolve_machined_pintle_layout,
    write_machined_pintle_report,
)


# ----------------------------------------------------------------------
#  Layout
# ----------------------------------------------------------------------
def resolve_coaxial_layout(inj, spec=None) -> dict:
    """Extend the machined hydraulic layout with coaxial-architecture stations.

    Reuses :func:`resolve_machined_pintle_layout` for the sized radii,
    manufacturing floors and per-stream (inlet/manifold) data, then adds the
    axial stack (faceplate / orifice plate / injector body / feed stub) and the
    plenum/collector/tip stations that realise the sealed coaxial circuit.
    """
    base = resolve_machined_pintle_layout(inj, spec=spec)
    r = base["resolved"]
    roles = base["roles"]

    # Which propellant is inner (radial, down the bore) vs outer (axial, annulus).
    inner_role = next(
        (k for k, d in roles.items() if d["feeds"] in ("slots", "holes")),
        None,
    )
    outer_role = next((k for k, d in roles.items() if d["feeds"] == "annulus"), None)
    if inner_role is None or outer_role is None:
        # Fall back to the pintle radial_stream convention.
        inner_role = base["hydraulic_basis"]["radial_stream"]
        outer_role = "oxidizer" if inner_role == "fuel" else "fuel"

    Rp = r["pintle_outer_radius_m"]
    Ri = r["pintle_inner_radius_m"]
    Dp = 2.0 * Rp
    gap = r["annulus_gap_m"]
    Ro = r["annulus_outer_radius_m"]           # metering-gap outer radius
    tol = r["tolerance_m"]
    min_tool = r["min_tool_diameter_m"]
    land = r["gasket_land_width_m"]
    clear = max(tol, 1.0e-4)                    # radial running clearance
    seal_type = r["seal_type"]
    seal_w = r["o_ring_groove_width_m"]
    seal_d = r["o_ring_groove_depth_m"]

    # Radial collector just outboard of the metering gap.
    col_o = Ro + max(3.0e-3, 2.0 * gap, 3.0 * min_tool)

    # Toroidal plenum: cross-section >= 2x the outer stream area (velocity floor
    # already baked into the resolved manifold area).
    ox = roles[outer_role]
    plenum_area = max(ox["manifold_cross_section_area_m2"], 2.0 * ox["stream_area_m2"])
    plenum_h = max(4.0e-3, math.sqrt(plenum_area), ox["manifold_depth_m"])
    w_pl = max(plenum_area / plenum_h, 3.0 * min_tool)
    pl_i = Rp + clear
    pl_o = max(col_o + land, pl_i + w_pl)
    plate_R = pl_o + 2.0 * land + seal_w

    # Axial stack (in -Z from the face): faceplate, orifice plate, injector body.
    # The metering land forms the annular gap's flow length; honour the
    # hydraulic model's L/Dh so the Cd assumption holds (annulus Dh = 2*gap).
    length_over_dh = float(inj.annulus.detail.get("length_over_dh", 2.0) or 2.0)
    land_h = max(length_over_dh * 2.0 * gap, 1.5 * gap, 2.0e-3, 2.0 * min_tool)
    col_h = max(3.0e-3, 2.0 * min_tool)                 # collector recess
    face_t = max(r["faceplate_thickness_m"], land_h + col_h + 2.0e-3)
    op_t = max(2.5e-3, 3.0 * min_tool)                  # orifice plate thickness
    seal_h = max(4.0e-3, 3.0 * min_tool)                # upper seal land
    body_h = plenum_h + seal_h + max(3.0e-3, 2.0 * min_tool)
    feed_stub = max(8.0e-3, Dp)                         # pintle feed stub
    tip_engage = max(3.0e-3, 2.0 * min_tool)            # body<->tip mating length
    # Pintle-body retention: a mounting flange seating on the injector-body
    # top (feed-side) face — the Nardi/TRW pintle is fixed from the top, not
    # floating in the clearance bore.
    flange_t = max(3.0e-3, 2.0 * min_tool)
    post_seal_center = Rp + clear + land + 0.5 * seal_w
    retention_bolt_d = max(2.0 * min_tool, 2.0e-3)
    retention_bolt_c = (
        post_seal_center + 0.5 * seal_w + land + 0.5 * retention_bolt_d
    )
    retention_bolt_count = 4
    flange_r = max(
        Rp + max(6.0e-3, 4.0 * min_tool),
        retention_bolt_c + 0.5 * retention_bolt_d + land,
    )

    z_fp_top = -face_t
    z_op_top = z_fp_top - op_t
    z_body_bot = z_op_top
    z_body_top = z_body_bot - body_h
    z_pl_bot = z_body_bot
    z_pl_top = z_pl_bot - plenum_h
    z_feed_top = z_body_top - feed_stub

    # Chamber side: first hole line one pintle diameter into the chamber (skip).
    skip = Dp
    z_holes = skip
    tip_nose = Rp
    z_tip_end = z_holes + tip_nose

    # Real stepped spigot/socket envelope for the replaceable tip.  Standard
    # threads are represented by their major/minor envelopes and a drawing
    # callout rather than fragile tessellated helical faces.
    tip_thread_major = Ri + 0.55 * (Rp - Ri)
    tip_thread_socket = tip_thread_major + clear
    shoulder_span = Rp - tip_thread_socket
    if shoulder_span <= 0.0:
        raise ValueError(
            "replaceable-tip socket leaves no radial shoulder for retention/seal"
        )
    tip_seal_w = min(
        seal_w,
        max(0.25 * shoulder_span, min_tool),
        0.80 * shoulder_span,
    )
    tip_seal_d = min(seal_d, 0.30 * tip_engage)
    tip_seal_center = 0.5 * (tip_thread_socket + Rp)

    # Bolt circle / outer diameter (respect the resolved values, but keep the
    # plenum + land inside the bolt circle).
    face_od = max(
        r["faceplate_outer_diameter_m"],
        2.0 * (plate_R + land + r["bolt_hole_diameter_m"]),
    )
    bolt_r = 0.5 * r["bolt_hole_diameter_m"]
    bolt_c = max(0.5 * r["bolt_circle_diameter_m"], plate_R + land + bolt_r)
    face_od = max(face_od, 2.0 * (bolt_c + bolt_r + land))
    n_bolt = int(r["bolt_count"])

    # Inner (radial) exit sizing.
    inner_stream = inj.streams[inner_role]
    inner_area = inner_stream.area
    n_slot = max(int(base["hydraulic_basis"]["slot_count"]), 1)
    radial_exit_style = str(inner_stream.geometry)
    if radial_exit_style == "holes":
        hole_d = float(inner_stream.detail["hole_diameter"])
        hole_length = float(inner_stream.detail["hole_length"])
        physical_hole_length = Rp - Ri
        if not math.isclose(
            physical_hole_length, hole_length, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            raise RuntimeError(
                "coaxial CAD pintle wall does not reproduce the solved "
                f"round-hole length ({physical_hole_length:.6g} vs "
                f"{hole_length:.6g} m)"
            )
        radial_area_from_dimensions = n_slot * math.pi * hole_d**2 / 4.0
    elif radial_exit_style == "slots":
        hole_d = None
        hole_length = None
        physical_hole_length = None
        radial_area_from_dimensions = (
            n_slot
            * float(inner_stream.detail["slot_width"])
            * float(inner_stream.detail["slot_height"])
        )
    else:
        raise ValueError(
            f"unsupported solved radial exit geometry {radial_exit_style!r}"
        )
    radial_area_error = (
        radial_area_from_dimensions - inner_area
    ) / max(inner_area, 1.0e-30)
    if abs(radial_area_error) > 1.0e-10:
        raise RuntimeError(
            "coaxial CAD radial-opening dimensions do not reproduce the "
            f"hydraulic area (relative error {radial_area_error:.3e})"
        )

    # Outer lateral inlet(s) and distribution holes.  The annular gap remains
    # the primary metering element; the orifice plate holes are sized with
    # margin so they distribute the outer stream without becoming the bottleneck.
    inlet_d = max(ox["inlet_diameter_m"], min_tool)
    inlet_count = max(int(ox.get("inlet_count", 1) or 1), 1)
    inlet_angle = float(ox.get("inlet_angle_deg", 0.0) or 0.0)
    # Distribution-hole count follows the hydraulic exit pattern, not the
    # chamber-flange bolt count.  Coupling it to ``2*n_bolt`` made a large
    # auto-sized flange create dozens of overlapping holes on a small ring,
    # fragmenting the orifice plate into disconnected islands.  Keep enough
    # circumferential samples for distribution while preserving a machinable
    # ligament between neighboring bores.
    op_count = max(12, n_slot)
    op_area = max(1.25 * ox["stream_area_m2"],
                  op_count * math.pi * (0.5 * min_tool) ** 2)
    op_hole_d = max(
        min_tool,
        math.sqrt(4.0 * op_area / (math.pi * op_count)),
    )
    op_ring_radius = 0.5 * (Ro + col_o)
    op_pitch = 2.0 * math.pi * op_ring_radius / op_count
    op_web = op_pitch - op_hole_d
    if op_web < min_tool:
        raise RuntimeError(
            "orifice-plate distribution holes do not fit with the configured "
            f"minimum ligament: count={op_count}, diameter={op_hole_d:.6g} m, "
            f"pitch={op_pitch:.6g} m, web={op_web:.6g} m, "
            f"required={min_tool:.6g} m. Increase collector radius or reduce "
            "the distribution-hole count through a re-solved layout."
        )

    stations = {
        "inner_role": inner_role,
        "outer_role": outer_role,
        "Rp": Rp, "Ri": Ri, "Dp": Dp, "gap": gap, "Ro": Ro,
        "clear": clear, "col_o": col_o,
        "pl_i": pl_i, "pl_o": pl_o, "plate_R": plate_R,
        "plenum_h": plenum_h,
        "land_h": land_h, "col_h": col_h, "face_t": face_t, "op_t": op_t,
        "seal_h": seal_h, "body_h": body_h, "feed_stub": feed_stub,
        "tip_engage": tip_engage, "flange_t": flange_t, "flange_r": flange_r,
        "post_seal_center": post_seal_center,
        "retention_bolt_d": retention_bolt_d,
        "retention_bolt_c": retention_bolt_c,
        "retention_bolt_count": retention_bolt_count,
        "seal_type": seal_type, "seal_w": seal_w, "seal_d": seal_d,
        "joint_seal_center": r["seal_center_radius_m"],
        "plate_body_seal_center": pl_o + land + 0.5 * seal_w,
        "plate_face_seal_center": col_o + land + 0.5 * seal_w,
        "tip_thread_major": tip_thread_major,
        "tip_thread_socket": tip_thread_socket,
        "tip_seal_w": tip_seal_w, "tip_seal_d": tip_seal_d,
        "tip_seal_center": tip_seal_center,
        "z_fp_top": z_fp_top, "z_op_top": z_op_top,
        "z_body_bot": z_body_bot, "z_body_top": z_body_top,
        "z_pl_bot": z_pl_bot, "z_pl_top": z_pl_top, "z_feed_top": z_feed_top,
        "skip": skip, "z_holes": z_holes, "z_tip_end": z_tip_end, "tip_nose": tip_nose,
        "face_od": face_od, "bolt_r": bolt_r, "bolt_c": bolt_c, "n_bolt": n_bolt,
        "n_exit": n_slot, "hole_d": hole_d,
        "hole_length_m": hole_length,
        "physical_hole_length_m": physical_hole_length,
        "radial_exit_style": radial_exit_style,
        "radial_opening_area_m2": radial_area_from_dimensions,
        "radial_opening_area_error_fraction": radial_area_error,
        "slot_width_m": r["slot_width_m"], "slot_height_m": r["slot_height_m"],
        "inlet_d": inlet_d, "inlet_count": inlet_count,
        "inlet_angle_deg": inlet_angle,
        "orifice_plate_hole_count": op_count,
        "orifice_plate_hole_diameter_m": op_hole_d,
        "orifice_plate_hole_pitch_m": op_pitch,
        "orifice_plate_minimum_web_m": op_web,
        "orifice_plate_open_area_m2": op_count * math.pi * (0.5 * op_hole_d) ** 2,
        "min_tool": min_tool, "tol": tol,
    }
    base["coaxial"] = stations
    base.setdefault("manufacturing_gates", []).append({
        "name": "orifice_plate_hole_ligament",
        "status": "pass",
        "detail": (
            f"distribution-hole web {op_web * 1e3:.3f} mm vs minimum "
            f"{min_tool * 1e3:.3f} mm; {op_count} holes are independent of "
            "the chamber-flange bolt count"
        ),
    })
    base["architecture"] = "coaxial_five_part_center_bore_annular_sheet"
    base["flow_continuity"] = {
        "status": "coaxial_circuits_sealed_until_chamber_exit",
        "inner_stream": (
            f"{inner_role} enters axially from the top, flows through the "
            "central pintle bore, and exits radially at the replaceable tip"
        ),
        "outer_stream": (
            f"{outer_role} enters laterally, fills the toroidal plenum, passes "
            "through the orifice plate into the collector, and exits through "
            "the annular metering gap as an axial sheet"
        ),
        "features": [
            "central bore inside pintle body and pintle tip",
            "replaceable pintle tip with radial holes or slots",
            "pintle-body retention flange seating on the injector-body top face",
            "matched pintle-retention holes with blind body engagement",
            "post, plate, chamber-joint, and replaceable-tip seal glands",
            "stepped tip spigot/socket with standard-thread envelopes",
            "lateral outer-stream inlet(s) into a toroidal plenum",
            "orifice plate distribution holes upstream of the annular collector",
            "faceplate land forming the continuous annular metering gap at the "
            "hydraulic model's L/Dh",
        ],
        "limits": (
            "This is reference CAD generated from 1-D hydraulic sizing. "
            "Seal glands and retention envelopes are represented, but seal "
            "dash numbers/squeeze, thread class, fastener preload, thermal "
            "growth, and cold-flow distribution still need drawing review."
        ),
    }
    base["mechanical_features"] = {
        "release_status": "geometry_complete_preliminary_not_drawing_released",
        "pintle_retention": {
            "flange_modeled": True,
            "fastener_holes_modeled": True,
            "fastener_count": retention_bolt_count,
            "fastener_hole_diameter_m": retention_bolt_d,
        },
        "replaceable_tip": {
            "spigot_socket_modeled": True,
            "shoulder_seal_gland_modeled": seal_type == "o_ring",
            "thread_envelopes_modeled": True,
            "thread_form_modeled": False,
            "required_drawing_callout": (
                "thread standard/class, proof load, locking method, and "
                "anti-galling coating"
            ),
        },
        "seals": {
            "type": seal_type,
            "chamber_joint_gland_modeled": seal_type == "o_ring",
            "post_face_gland_modeled": seal_type == "o_ring",
            "plate_outer_glands_modeled": seal_type == "o_ring",
            "selected_standard_and_squeeze_verified": False,
        },
    }
    return base


# ----------------------------------------------------------------------
#  Body builders (SI metres)
# ----------------------------------------------------------------------
def _ring(cq, z0, h, r_out, r_in):
    return (cq.Workplane("XY").workplane(offset=z0)
            .circle(r_out).circle(r_in).extrude(h))


def _bolts(cq, S, z0, h):
    pts = _points_on_circle(S["bolt_c"], S["n_bolt"], 0.0)
    return (cq.Workplane("XY").workplane(offset=z0 - 1e-4)
            .pushPoints(pts).circle(S["bolt_r"]).extrude(h + 2e-4))


def _radial_cylinder(cq, radius_start, radius_end, z, diameter, angle_deg):
    length = max(radius_end - radius_start, diameter)
    cyl = (cq.Workplane("YZ").workplane(offset=radius_start)
           .center(0.0, z).circle(0.5 * diameter).extrude(length))
    return cyl.rotate((0, 0, 0), (0, 0, 1), angle_deg)


def build_pintle_body(cq, S):
    z0, z1 = S["z_feed_top"], S["z_holes"] - S["tip_engage"]
    post = _ring(cq, z0, z1 - z0, S["Rp"], S["Ri"])
    post = post.union(_ring(
        cq, z1, S["tip_engage"], S["tip_thread_major"], S["Ri"]
    ))
    # Retention flange seating on the injector-body top (feed-side) face at
    # z_body_top: fixes the pintle axially instead of floating in the bore.
    flange = _ring(cq, S["z_body_top"] - S["flange_t"], S["flange_t"],
                   S["flange_r"], S["Ri"])
    post = post.union(flange)
    if S["seal_type"] == "o_ring":
        post = post.cut(_ring(
            cq,
            S["z_body_top"] - S["seal_d"],
            S["seal_d"],
            S["post_seal_center"] + 0.5 * S["seal_w"],
            S["post_seal_center"] - 0.5 * S["seal_w"],
        ))
    pts = _points_on_circle(
        S["retention_bolt_c"], S["retention_bolt_count"], 45.0
    )
    holes = (cq.Workplane("XY")
             .workplane(offset=S["z_body_top"] - S["flange_t"] - 1.0e-4)
             .pushPoints(pts).circle(0.5 * S["retention_bolt_d"])
             .extrude(S["flange_t"] + 2.0e-4))
    return post.cut(holes)


def build_pintle_tip(cq, S, radial_style=None):
    Rp, Ri = S["Rp"], S["Ri"]
    zt = S["z_holes"]
    stub_bot = zt - S["tip_engage"]
    tip = _ring(cq, stub_bot, zt - stub_bot, Rp, S["tip_thread_socket"])
    if S["seal_type"] == "o_ring":
        # The replaceable-tip seal belongs in the tip's feed-side shoulder.
        # The prior cutter was applied to the smaller body-thread envelope,
        # outside that body's material, so metadata claimed a gland that did
        # not exist.  Cut the annular axial-face gland in its actual owning
        # part, just inside the tip shoulder.
        tip = tip.cut(_ring(
            cq,
            stub_bot,
            S["tip_seal_d"],
            S["tip_seal_center"] + 0.5 * S["tip_seal_w"],
            S["tip_seal_center"] - 0.5 * S["tip_seal_w"],
        ))
    tip = tip.union(_ring(cq, zt, 0.5e-3, Rp, Ri))      # thin cap ring
    nose = cq.Workplane("XY").workplane(offset=zt + 0.5e-3).sphere(Rp)
    upper = (cq.Workplane("XY").workplane(offset=zt + 0.5e-3)
             .box(2.2 * Rp, 2.2 * Rp, 2.2 * Rp, centered=(True, True, False)))
    tip = tip.union(nose.intersect(upper))
    n = S["n_exit"]
    radial_style = radial_style or S["radial_exit_style"]
    if radial_style == "slots":
        w, h = S["slot_width_m"], S["slot_height_m"]
        depth = (Rp - Ri) + 2.0e-3
        for i in range(n):
            ang = 360.0 * i / n
            cutter = (cq.Workplane("XY").workplane(offset=zt)
                      .center(Rp - 0.5 * depth, 0.0)
                      .box(depth, w, h, centered=(True, True, True))
                      .rotate((0, 0, 0), (0, 0, 1), ang))
            tip = tip.cut(cutter)
    elif radial_style == "holes":                      # round jets
        d = S["hole_d"]
        for i in range(n):
            ang = 360.0 * i / n
            cutter = (cq.Workplane("YZ").workplane(offset=0.0)
                      .center(0.0, zt).circle(0.5 * d).extrude(Rp + 1e-3)
                      .rotate((0, 0, 0), (0, 0, 1), ang))
            tip = tip.cut(cutter)
    else:
        raise ValueError("radial_style must be 'slots' or 'holes'")
    return tip


def build_orifice_plate(cq, S):
    z0, z1 = S["z_op_top"], S["z_fp_top"]
    plate = _ring(cq, z0, z1 - z0, S["plate_R"], S["Rp"] + S["clear"])
    n = int(S["orifice_plate_hole_count"])
    r_ring = 0.5 * (S["Ro"] + S["col_o"])
    hd = S["orifice_plate_hole_diameter_m"]
    pts = _points_on_circle(r_ring, n, 0.0)
    holes = (cq.Workplane("XY").workplane(offset=z0 - 1e-4)
             .pushPoints(pts).circle(0.5 * hd).extrude((z1 - z0) + 2e-4))
    plate = plate.cut(holes)
    if S["seal_type"] == "o_ring":
        sr = S["plate_face_seal_center"]
        plate = plate.cut(_ring(
            cq, z1 - S["seal_d"], S["seal_d"],
            sr + 0.5 * S["seal_w"], sr - 0.5 * S["seal_w"],
        ))
    return plate


def build_faceplate(cq, S):
    z0 = S["z_fp_top"]
    fp = _ring(cq, z0, -z0, 0.5 * S["face_od"], S["col_o"])
    land = _ring(cq, -S["land_h"], S["land_h"], S["col_o"], S["Ro"])
    fp = fp.union(land)
    fp = fp.cut(_bolts(cq, S, z0, -z0))
    if S["seal_type"] == "o_ring":
        sr = S["joint_seal_center"]
        fp = fp.cut(_ring(
            cq, -S["seal_d"], S["seal_d"],
            sr + 0.5 * S["seal_w"], sr - 0.5 * S["seal_w"],
        ))
    return fp


def build_injector_body(cq, S):
    z_top, z_bot = S["z_body_top"], S["z_body_bot"]
    body = _ring(cq, z_top, z_bot - z_top, 0.5 * S["face_od"], S["Rp"] + S["clear"])
    plenum = _ring(cq, S["z_pl_top"], S["z_pl_bot"] - S["z_pl_top"],
                   S["pl_o"], S["pl_i"])
    body = body.cut(plenum)
    z_mid = 0.5 * (S["z_pl_top"] + S["z_pl_bot"])
    for i in range(int(S["inlet_count"])):
        ang = S["inlet_angle_deg"] + 360.0 * i / int(S["inlet_count"])
        inlet = _radial_cylinder(
            cq, S["pl_i"], 0.5 * S["face_od"] + 2e-3,
            z_mid, S["inlet_d"], ang,
        )
        body = body.cut(inlet)
    body = body.cut(_bolts(cq, S, z_top, z_bot - z_top))
    pts = _points_on_circle(
        S["retention_bolt_c"], S["retention_bolt_count"], 45.0
    )
    engage = min(0.75 * S["seal_h"], 0.30 * (z_bot - z_top))
    ret = (cq.Workplane("XY").workplane(offset=z_top - 1.0e-5)
           .pushPoints(pts).circle(0.5 * S["retention_bolt_d"])
           .extrude(engage + 1.0e-5))
    body = body.cut(ret)
    if S["seal_type"] == "o_ring":
        sr = S["plate_body_seal_center"]
        body = body.cut(_ring(
            cq, z_bot - S["seal_d"], S["seal_d"],
            sr + 0.5 * S["seal_w"], sr - 0.5 * S["seal_w"],
        ))
    return body


def _resolved_radial_style(inj, radial_style=None) -> str:
    solved = str(inj.slots.geometry)
    style = solved if radial_style is None else str(radial_style)
    if style not in ("slots", "holes"):
        raise ValueError("radial_style must be 'slots' or 'holes'")
    if style != solved:
        raise ValueError(
            f"requested CAD radial_style={style!r} disagrees with solved "
            f"hydraulic geometry {solved!r}; re-solve the injector instead "
            "of changing the metering topology at export"
        )
    return style


def build_coaxial_bodies(inj, *, spec=None, layout=None, radial_style=None):
    """Return ``{name: cadquery solid}`` (SI metres) for the five coaxial parts."""
    cq = _cq()
    layout = layout or resolve_coaxial_layout(inj, spec=spec)
    radial_style = _resolved_radial_style(inj, radial_style)
    S = layout["coaxial"]
    return {
        "pintle_body": build_pintle_body(cq, S),
        "pintle_tip": build_pintle_tip(cq, S, radial_style=radial_style),
        "injector_body": build_injector_body(cq, S),
        "orifice_plate": build_orifice_plate(cq, S),
        "faceplate": build_faceplate(cq, S),
    }


# ----------------------------------------------------------------------
#  Flow-circuit seal audit
# ----------------------------------------------------------------------
def analytic_flow_separation(layout: dict) -> dict:
    """CadQuery-free seal invariant for the coaxial topology."""
    S = layout["coaxial"]
    radial_clearance = (S["Rp"] + S["clear"]) - S["Ri"]
    return {
        "inner_role": S["inner_role"],
        "outer_role": S["outer_role"],
        "method": "analytic_coaxial_radial_envelope",
        "inner_bore_outer_radius_m": S["Ri"],
        "outer_circuit_inner_radius_m": S["Rp"] + S["clear"],
        "minimum_radial_separation_m": radial_clearance,
        "inner_outer_overlap_m3": 0.0 if radial_clearance > 0.0 else float("nan"),
        "circuits_sealed": bool(radial_clearance > 0.0),
    }


def audit_flow_separation(layout: dict) -> dict:
    """Verify the inner-bore fluid and outer-plenum fluid volumes are disjoint.

    This is the invariant the old concentric-ring geometry violated: the two
    propellant circuits must not intersect anywhere inside the injector.
    """
    analytic = analytic_flow_separation(layout)
    if not cadquery_available():
        return analytic
    cq = _cq()
    S = layout["coaxial"]
    inner = (cq.Workplane("XY").workplane(offset=S["z_feed_top"])
             .circle(S["Ri"]).extrude(S["z_holes"] - S["z_feed_top"])).val()
    outer = (cq.Workplane("XY").workplane(offset=S["z_pl_top"])
             .circle(S["pl_o"]).circle(S["Rp"] + S["clear"])
             .extrude(S["z_fp_top"] - S["z_pl_top"])).val()
    try:
        overlap = inner.intersect(outer)
        vol = sum(s.Volume() for s in overlap.Solids())
    except Exception:
        vol = float("nan")
    sealed = math.isfinite(vol) and abs(vol) <= 1e-9
    return {
        **analytic,
        "method": "cadquery_volume_intersection",
        "inner_role": S["inner_role"],
        "outer_role": S["outer_role"],
        "inner_outer_overlap_m3": float(vol),
        "circuits_sealed": bool(sealed),
    }


def audit_flow_connectivity(layout: dict) -> dict:
    """Verify both propellant voids connect continuously to their exits.

    This complements :func:`audit_flow_separation`: two disjoint circuits can
    still each contain a blocked inlet, isolated plenum, or decorative exit.
    The audit reconstructs the exact coaxial void primitives, requires a
    positive-volume handoff at every interface, and requires each complete
    circuit to fuse into one solid.  It is a topology gate, not a pressure-loss
    or maldistribution solution.
    """
    if not cadquery_available():
        return {
            "passed": False,
            "status": "not_evaluated_cadquery_unavailable",
            "model": "void_topology_only_not_hydraulic_cfd",
        }

    cq = _cq()
    S = layout["coaxial"]
    eps = max(2.0 * float(S["tol"]), 1.0e-4)
    threshold = 1.0e-18  # m^3 in the SI-valued construction model

    # Inner circuit: feed bore -> every radial opening.
    exit_height = (
        float(S["hole_d"])
        if S["radial_exit_style"] == "holes"
        else float(S["slot_height_m"])
    )
    bore = (
        cq.Workplane("XY")
        .workplane(offset=S["z_feed_top"] - eps)
        .circle(S["Ri"])
        .extrude(S["z_holes"] - S["z_feed_top"] + 2.0 * eps)
        .val()
    )
    exits = []
    for i in range(int(S["n_exit"])):
        angle = 360.0 * i / int(S["n_exit"])
        if S["radial_exit_style"] == "holes":
            opening = _radial_cylinder(
                cq,
                S["Ri"] - eps,
                S["Rp"] + eps,
                S["z_holes"],
                S["hole_d"],
                angle,
            ).val()
        else:
            depth = S["Rp"] - S["Ri"] + 2.0 * eps
            opening = (
                cq.Workplane("XY")
                .workplane(offset=S["z_holes"])
                .center(S["Rp"] - 0.5 * depth, 0.0)
                .box(
                    depth,
                    S["slot_width_m"],
                    exit_height,
                    centered=(True, True, True),
                )
                .rotate((0, 0, 0), (0, 0, 1), angle)
                .val()
            )
        exits.append(opening)

    # Outer circuit: lateral inlet(s) -> toroidal plenum -> distribution
    # holes -> collector -> the continuous annular metering sheet.
    plenum = _ring(
        cq,
        S["z_pl_top"],
        S["z_pl_bot"] - S["z_pl_top"],
        S["pl_o"],
        S["pl_i"],
    ).val()
    z_pl_mid = 0.5 * (S["z_pl_top"] + S["z_pl_bot"])
    inlets = [
        _radial_cylinder(
            cq,
            S["pl_i"] - eps,
            0.5 * S["face_od"] + eps,
            z_pl_mid,
            S["inlet_d"],
            S["inlet_angle_deg"] + 360.0 * i / int(S["inlet_count"]),
        ).val()
        for i in range(int(S["inlet_count"]))
    ]
    r_distribution = 0.5 * (S["Ro"] + S["col_o"])
    distribution = []
    for i in range(int(S["orifice_plate_hole_count"])):
        theta = 2.0 * math.pi * i / int(S["orifice_plate_hole_count"])
        distribution.append(cq.Solid.makeCylinder(
            0.5 * S["orifice_plate_hole_diameter_m"],
            S["z_fp_top"] - S["z_op_top"] + 2.0 * eps,
            cq.Vector(
                r_distribution * math.cos(theta),
                r_distribution * math.sin(theta),
                S["z_op_top"] - eps,
            ),
            cq.Vector(0.0, 0.0, 1.0),
        ))
    collector = _ring(
        cq,
        S["z_fp_top"] - eps,
        (-S["land_h"] + eps) - (S["z_fp_top"] - eps),
        S["col_o"],
        S["Rp"] + S["clear"],
    ).val()
    metering_gap = _ring(
        cq,
        -S["land_h"] - eps,
        S["land_h"] + 2.0 * eps,
        S["Ro"],
        S["Rp"],
    ).val()

    handoffs = {
        "minimum_inner_bore_to_exit_m3": min(
            float(abs(bore.intersect(opening).Volume())) for opening in exits
        ),
        "minimum_outer_inlet_to_plenum_m3": min(
            float(abs(plenum.intersect(inlet).Volume())) for inlet in inlets
        ),
        "minimum_plenum_to_distribution_hole_m3": min(
            float(abs(plenum.intersect(hole).Volume()))
            for hole in distribution
        ),
        "minimum_distribution_hole_to_collector_m3": min(
            float(abs(collector.intersect(hole).Volume()))
            for hole in distribution
        ),
        "collector_to_metering_gap_m3": float(
            abs(collector.intersect(metering_gap).Volume())
        ),
    }
    inner = bore.fuse(*exits)
    outer = plenum.fuse(*inlets, *distribution, collector, metering_gap)
    inner_connected = inner.isValid() and len(inner.Solids()) == 1
    outer_connected = outer.isValid() and len(outer.Solids()) == 1
    passed = bool(
        inner_connected
        and outer_connected
        and min(handoffs.values()) > threshold
    )
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "inner_single_connected_void": bool(inner_connected),
        "outer_single_connected_void": bool(outer_connected),
        "handoff_overlap_volumes_m3": handoffs,
        "minimum_required_overlap_m3": threshold,
        "model": "void_topology_only_not_pressure_loss_or_maldistribution_cfd",
    }


def audit_nominal_clearances(layout: dict) -> dict:
    """Report positive nominal gaps explicitly represented in the assembly."""
    S = layout["coaxial"]
    clearances = {
        "pintle_body_to_injector_body_radial_m":
            (S["Rp"] + S["clear"]) - S["Rp"],
        "replaceable_tip_thread_envelope_radial_m":
            S["tip_thread_socket"] - S["tip_thread_major"],
        "orifice_plate_bore_to_pintle_radial_m":
            (S["Rp"] + S["clear"]) - S["Rp"],
        "annular_metering_gap_m": S["Ro"] - S["Rp"],
    }
    passed = all(value > 0.0 for value in clearances.values())
    return {
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "clearances": clearances,
        "qualification": (
            "positive_nominal_geometry_only; tolerance stack, seal squeeze, "
            "thermal growth, pressure deflection, and surface finish remain "
            "drawing/analysis requirements"
        ),
    }


def audit_component_interference(bodies: dict[str, Any]) -> dict:
    """Pairwise positive-volume interference gate for assembled hardware.

    Coincident mating faces are allowed; any positive shared material volume
    is rejected.  Bodies are supplied in SI model units, hence intersection
    volumes are numeric m^3 here (before neutral-file scaling).
    """
    names = list(bodies)
    pairs: list[dict[str, Any]] = []
    max_overlap = 0.0
    tolerance = 1.0e-15  # m^3 = 1e-6 mm^3, below kernel significance
    for i, left in enumerate(names):
        a = bodies[left].val() if hasattr(bodies[left], "val") else bodies[left]
        for right in names[i + 1:]:
            b = bodies[right].val() if hasattr(bodies[right], "val") else bodies[right]
            try:
                overlap = a.intersect(b)
                volume = float(sum(abs(s.Volume()) for s in overlap.Solids()))
            except Exception as exc:
                return {
                    "status": "failed_to_evaluate",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pairs": pairs,
                    "passed": False,
                }
            max_overlap = max(max_overlap, volume)
            pairs.append({
                "components": [left, right],
                "overlap_m3": volume,
                "status": "pass" if volume <= tolerance else "fail",
            })
    return {
        "status": "pass" if max_overlap <= tolerance else "fail",
        "tolerance_m3": tolerance,
        "maximum_overlap_m3": max_overlap,
        "pairs": pairs,
        "passed": max_overlap <= tolerance,
    }


# ----------------------------------------------------------------------
#  Export
# ----------------------------------------------------------------------
_COLORS = {
    "pintle_body": (0.72, 0.30, 0.30),
    "pintle_tip": (0.30, 0.62, 0.32),
    "injector_body": (0.78, 0.74, 0.60),
    "orifice_plate": (0.32, 0.55, 0.72),
    "faceplate": (0.34, 0.36, 0.68),
}


def export_coaxial_pintle_cad(inj, out_dir, *, spec=None, fmt="step",
                              radial_style=None) -> dict:
    """Export the five coaxial STEP bodies + a named assembly + a report.

    Bodies are built in SI metres and scaled to millimetres at the STEP
    boundary.  Each body is re-imported and gated as a single valid solid, and
    the inner/outer flow circuits are audited for separation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []
    radial_style = _resolved_radial_style(inj, radial_style)
    layout = resolve_coaxial_layout(inj, spec=spec)
    seal = analytic_flow_separation(layout)

    if fmt != "step":
        notes.append(f"coaxial mode is STEP-only; ignoring requested format {fmt!r}")

    if not cadquery_available():
        layout["cad_export"] = {
            "status": "cadquery_unavailable",
            "architecture": layout["architecture"],
            "radial_exit_style": radial_style,
            "representation": "report_only",
            "flow_separation_audit": seal,
            "required_extra": "pip install -e .[cad]",
        }
        notes.append("coaxial STEP export requested but CadQuery/OpenCascade is "
                     "not installed; layout written, STEP skipped.")
        report = write_machined_pintle_report(
            layout, out_dir / "injector_manufacturing_report.json")
        files["manufacturing_report"] = str(report)
        return {"files": files, "notes": notes, "layout": layout}

    cq = _cq()
    failure: Exception | None = None
    try:
        bodies_si = build_coaxial_bodies(
            inj, spec=spec, layout=layout, radial_style=radial_style
        )
        interference = audit_component_interference(bodies_si)
        if not interference["passed"]:
            raise RuntimeError(
                "coaxial injector component interference gate failed: "
                f"{interference}"
            )
        connectivity = audit_flow_connectivity(layout)
        if not connectivity["passed"]:
            raise RuntimeError(
                "coaxial injector flow-connectivity gate failed: "
                f"{connectivity}"
            )
        clearances = audit_nominal_clearances(layout)
        if not clearances["passed"]:
            raise RuntimeError(
                "coaxial injector nominal-clearance gate failed: "
                f"{clearances}"
            )
        bodies = {n: _to_mm_step_solid(s) for n, s in bodies_si.items()}

        inspections: dict[str, Any] = {}
        asm = cq.Assembly(name="pintle_injector_coaxial")
        for name, solid in bodies.items():
            path = out_dir / f"{name}.step"
            cq.exporters.export(solid, str(path), exportType="STEP")
            insp = inspect_machined_step(path)
            if not (insp["single_solid"] and insp["all_solids_valid"]):
                raise RuntimeError(
                    f"{name} STEP round-trip is not one valid solid: {insp}")
            inspections[name] = insp
            files[name] = str(path)
            asm.add(solid, name=name, color=cq.Color(*_COLORS[name]))

        # Keep the ecosystem's "machined assembly" filename (the engine-assembly
        # child list and CLI look for it); the architecture is now coaxial.
        assembly_path = out_dir / "injector_assembly_machined.step"
        try:
            asm.export(str(assembly_path))
        except Exception:
            asm.save(str(assembly_path))
        assembly_insp = inspect_machined_step(assembly_path)
        if not (
            assembly_insp["all_solids_valid"]
            and assembly_insp["solid_count"] == len(bodies)
        ):
            raise RuntimeError(
                "coaxial assembly STEP round-trip failed: "
                f"{assembly_insp}"
            )
        files["machined_assembly"] = str(assembly_path)

        units_path = out_dir / "injector_cad_units.json"
        units_path.write_text(
            json.dumps({
                "schema": "raosim.cad_units.v1",
                "public_api_linear_unit": "m",
                "neutral_file_linear_unit": "mm",
                "volume_unit": "mm^3",
                "stl_unit_policy": (
                    "STL has no embedded unit; RaoRocketSim neutral CAD uses "
                    "millimetre numeric coordinates"
                ),
                "files": {
                    Path(value).name: "mm"
                    for value in files.values()
                    if str(value).lower().endswith(".step")
                },
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        files["cad_units"] = str(units_path)

        seal = audit_flow_separation(layout)
        if not seal["circuits_sealed"]:
            notes.append(
                f"WARNING: inner/outer flow circuits overlap by "
                f"{seal['inner_outer_overlap_m3']:.3e} m^3 — not sealed.")
        layout["cad_export"] = {
            "status": "step_written",
            "architecture": layout["architecture"],
            "radial_exit_style": radial_style,
            "representation": "coaxial_five_part_solids_of_revolution",
            "files": files.copy(),
            "inspection": inspections,
            "assembly_inspection": assembly_insp,
            "component_interference_audit": interference,
            "flow_separation_audit": seal,
            "flow_connectivity_audit": connectivity,
            "nominal_clearance_audit": clearances,
            "geometry_topology_ready_for_cold_flow_build": not any(
                gate.get("status") == "fail"
                for gate in layout.get("manufacturing_gates", [])
            ),
            "cold_flow_release_ready": False,
            "hot_fire_release_ready": False,
            "hardware_qualified": False,
            "external_release_blockers": [
                "released toleranced drawings and process plan",
                "selected seal standards, squeeze, and pressure qualification",
                "fastener/thread preload, proof, and locking substantiation",
                "proof/leak test and cold-flow distribution evidence",
                "thermal/structural/stability analysis and hot-fire evidence",
            ],
            "parts": [
                "pintle body (inner propellant down the central bore)",
                "pintle tip (replaceable; radial %s at skip = 1 pintle diameter)"
                % ("jets" if radial_style == "holes" else "slots"),
                "injector body (lateral inlet + toroidal plenum for the outer stream)",
                "orifice plate (distributes the outer stream into the collector)",
                "faceplate (continuous metering gap -> axial sheet on the pintle)",
            ],
        }
    except Exception as exc:
        failure = exc
        layout["cad_export"] = {
            "status": "step_export_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        notes.append(f"coaxial STEP export failed: {type(exc).__name__}: {exc}")

    report = write_machined_pintle_report(
        layout, out_dir / "injector_manufacturing_report.json")
    files["manufacturing_report"] = str(report)
    if failure is not None:
        raise RuntimeError(
            "coaxial injector CAD failed a required geometry/export gate; "
            f"diagnostics were written to {report}: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    return {"files": files, "notes": notes, "layout": layout}
