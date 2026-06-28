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

import math


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
                named part files in ``pintle_parts/``).
    ``fmt``   : ``step`` (portable B-rep, default) | ``stl`` (mesh preview) |
                ``dxf`` (2-D meridional profile).
    STEP/STL require CadQuery; DXF does not.  Returns ``{files, notes}``.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []

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
