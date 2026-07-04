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
    inner_role = next((k for k, d in roles.items() if d["feeds"] == "slots"), None)
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

    # Axial stack (in -Z from the face): faceplate, orifice plate, injector body.
    land_h = max(2.0e-3, 1.5 * gap, 2.0 * min_tool)     # metering land near face
    col_h = max(3.0e-3, 2.0 * min_tool)                 # collector recess
    face_t = max(r["faceplate_thickness_m"], land_h + col_h + 2.0e-3)
    op_t = max(2.5e-3, 3.0 * min_tool)                  # orifice plate thickness
    seal_h = max(4.0e-3, 3.0 * min_tool)                # upper seal land
    body_h = plenum_h + seal_h + max(3.0e-3, 2.0 * min_tool)
    feed_stub = max(8.0e-3, Dp)                         # pintle feed stub
    tip_engage = max(3.0e-3, 2.0 * min_tool)            # body<->tip mating length

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

    # Bolt circle / outer diameter (respect the resolved values, but keep the
    # plenum + land inside the bolt circle).
    face_od = max(r["faceplate_outer_diameter_m"], 2.0 * (pl_o + land + r["bolt_hole_diameter_m"]))
    bolt_r = 0.5 * r["bolt_hole_diameter_m"]
    bolt_c = max(0.5 * r["bolt_circle_diameter_m"], pl_o + land + bolt_r)
    n_bolt = int(r["bolt_count"])

    # Inner (radial) exit sizing.
    inner_area = inj.streams[inner_role].area
    n_slot = max(int(base["hydraulic_basis"]["slot_count"]), 1)
    hole_d = math.sqrt(4.0 * inner_area / (math.pi * max(n_slot, 1)))
    hole_d = max(hole_d, min_tool)

    # Outer lateral inlet(s) and distribution holes.  The annular gap remains
    # the primary metering element; the orifice plate holes are sized with
    # margin so they distribute the outer stream without becoming the bottleneck.
    inlet_d = max(ox["inlet_diameter_m"], min_tool)
    inlet_count = max(int(ox.get("inlet_count", 1) or 1), 1)
    inlet_angle = float(ox.get("inlet_angle_deg", 0.0) or 0.0)
    op_count = max(12, 2 * n_bolt, n_slot)
    op_area = max(1.25 * ox["stream_area_m2"],
                  op_count * math.pi * (0.5 * min_tool) ** 2)
    op_hole_d = max(
        min_tool,
        math.sqrt(4.0 * op_area / (math.pi * op_count)),
    )

    stations = {
        "inner_role": inner_role,
        "outer_role": outer_role,
        "Rp": Rp, "Ri": Ri, "Dp": Dp, "gap": gap, "Ro": Ro,
        "clear": clear, "col_o": col_o,
        "pl_i": pl_i, "pl_o": pl_o, "plenum_h": plenum_h,
        "land_h": land_h, "col_h": col_h, "face_t": face_t, "op_t": op_t,
        "seal_h": seal_h, "body_h": body_h, "feed_stub": feed_stub,
        "tip_engage": tip_engage,
        "z_fp_top": z_fp_top, "z_op_top": z_op_top,
        "z_body_bot": z_body_bot, "z_body_top": z_body_top,
        "z_pl_bot": z_pl_bot, "z_pl_top": z_pl_top, "z_feed_top": z_feed_top,
        "skip": skip, "z_holes": z_holes, "z_tip_end": z_tip_end, "tip_nose": tip_nose,
        "face_od": face_od, "bolt_r": bolt_r, "bolt_c": bolt_c, "n_bolt": n_bolt,
        "n_exit": n_slot, "hole_d": hole_d,
        "slot_width_m": r["slot_width_m"], "slot_height_m": r["slot_height_m"],
        "inlet_d": inlet_d, "inlet_count": inlet_count,
        "inlet_angle_deg": inlet_angle,
        "orifice_plate_hole_count": op_count,
        "orifice_plate_hole_diameter_m": op_hole_d,
        "orifice_plate_open_area_m2": op_count * math.pi * (0.5 * op_hole_d) ** 2,
        "min_tool": min_tool, "tol": tol,
    }
    base["coaxial"] = stations
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
            "lateral outer-stream inlet(s) into a toroidal plenum",
            "orifice plate distribution holes upstream of the annular collector",
            "faceplate land forming the continuous annular metering gap",
        ],
        "limits": (
            "This is reference CAD generated from 1-D hydraulic sizing. "
            "Detailed loss coefficients, seals, fastener preload, thermal "
            "growth, and cold-flow distribution still need design review."
        ),
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
    return _ring(cq, z0, z1 - z0, S["Rp"], S["Ri"])


def build_pintle_tip(cq, S, radial_style="holes"):
    Rp, Ri = S["Rp"], S["Ri"]
    zt = S["z_holes"]
    stub_bot = zt - S["tip_engage"]
    tip = _ring(cq, stub_bot, zt - stub_bot, Rp, Ri)
    tip = tip.union(_ring(cq, zt, 0.5e-3, Rp, Ri))      # thin cap ring
    nose = cq.Workplane("XY").workplane(offset=zt + 0.5e-3).sphere(Rp)
    upper = (cq.Workplane("XY").workplane(offset=zt + 0.5e-3)
             .box(2.2 * Rp, 2.2 * Rp, 2.2 * Rp, centered=(True, True, False)))
    tip = tip.union(nose.intersect(upper))
    n = S["n_exit"]
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
    else:                                               # round jets
        d = S["hole_d"]
        for i in range(n):
            ang = 360.0 * i / n
            cutter = (cq.Workplane("YZ").workplane(offset=0.0)
                      .center(0.0, zt).circle(0.5 * d).extrude(Rp + 1e-3)
                      .rotate((0, 0, 0), (0, 0, 1), ang))
            tip = tip.cut(cutter)
    return tip


def build_orifice_plate(cq, S):
    z0, z1 = S["z_op_top"], S["z_fp_top"]
    plate = _ring(cq, z0, z1 - z0, S["pl_o"], S["Rp"] + S["clear"])
    n = int(S["orifice_plate_hole_count"])
    r_ring = 0.5 * (S["Ro"] + S["col_o"])
    hd = S["orifice_plate_hole_diameter_m"]
    pts = _points_on_circle(r_ring, n, 0.0)
    holes = (cq.Workplane("XY").workplane(offset=z0 - 1e-4)
             .pushPoints(pts).circle(0.5 * hd).extrude((z1 - z0) + 2e-4))
    return plate.cut(holes)


def build_faceplate(cq, S):
    z0 = S["z_fp_top"]
    fp = _ring(cq, z0, -z0, 0.5 * S["face_od"], S["col_o"])
    land = _ring(cq, -S["land_h"], S["land_h"], S["col_o"], S["Ro"])
    fp = fp.union(land)
    return fp.cut(_bolts(cq, S, z0, -z0))


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
    return body.cut(_bolts(cq, S, z_top, z_bot - z_top))


def build_coaxial_bodies(inj, *, spec=None, layout=None, radial_style="holes"):
    """Return ``{name: cadquery solid}`` (SI metres) for the five coaxial parts."""
    cq = _cq()
    layout = layout or resolve_coaxial_layout(inj, spec=spec)
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
                              radial_style="holes") -> dict:
    """Export the five coaxial STEP bodies + a named assembly + a report.

    Bodies are built in SI metres and scaled to millimetres at the STEP
    boundary.  Each body is re-imported and gated as a single valid solid, and
    the inner/outer flow circuits are audited for separation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []
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
    try:
        bodies = build_coaxial_bodies(inj, spec=spec, layout=layout,
                                      radial_style=radial_style)
        bodies = {n: _to_mm_step_solid(s) for n, s in bodies.items()}

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
        files["machined_assembly"] = str(assembly_path)

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
            "flow_separation_audit": seal,
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
        layout["cad_export"] = {
            "status": "step_export_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        notes.append(f"coaxial STEP export failed: {type(exc).__name__}: {exc}")

    report = write_machined_pintle_report(
        layout, out_dir / "injector_manufacturing_report.json")
    files["manufacturing_report"] = str(report)
    return {"files": files, "notes": notes, "layout": layout}
