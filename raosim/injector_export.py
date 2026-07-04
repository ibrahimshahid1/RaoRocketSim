"""injector_export.py - pintle reference-geometry resolution + output package.

Turns a solved :class:`raosim.injector.InjectorDesignResult` into the CLI
deliverable folder::

    pintle/
      pintle_parameters.json     # full labeled geometry + operating point
      pintle_dimensions.csv      # flat dimension table (symbol, value, unit)
      pintle_schematic.svg       # mandatory labeled 2-D schematic (vector)
      pintle_cross_section.png   # mandatory labeled 2-D schematic (raster)
      pintle_reference.step       # optional, --injector-cad reference
      pintle_parts/               # optional, --injector-cad parts
        pintle_rod.step  pintle_tip.step  annular_sleeve.step  injector_face.step
      pintle_body.step            # optional, --injector-cad machined
      pintle_tip.step
      injector_body.step
      orifice_plate.step
      faceplate.step
      injector_assembly_machined.step
      injector_manufacturing_report.json

Naming follows the repo's FIXED annulus + radial-slot pintle and, where they
correspond, Son et al. (*Design Procedure of a Movable Pintle Injector*, J.
Propulsion & Power 33-4, 2017) symbols.  Dimensions that exist only on a
movable/center-gap pintle (``L_open``, ``D_cg``) are reported as
``not_applicable`` for this architecture rather than invented.  Schematic-only
construction values (wall/sleeve thicknesses, body length) are flagged ``kind:
schematic`` so they are never mistaken for solved results.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

# Schematic construction constants shared with injector_cad / injector_plots so
# the drawing, the CAD bodies and the dimension table all agree.
def _t_wall(Rp):
    return max(0.25 * Rp, 1.0e-3)        # pintle wall thickness


def _t_sleeve(gap):
    return max(0.4 * gap, 0.5e-3)        # annular sleeve wall


def _body_len(Dp):
    return 3.0 * Dp                      # protrusion into chamber


def _tip_len(Rp):
    return max(0.35 * Rp, 0.5e-3)        # short blunt/chamfered tip cap


def _f(x) -> float | None:
    """Coerce to float, mapping NaN/None to None for clean JSON/CSV."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _dim(group, symbol, name, value_m, unit="mm", kind="solved", note=""):
    """One dimension record.  Linear values are stored in SI and mm."""
    v = _f(value_m)
    rec = {
        "group": group, "symbol": symbol, "name": name,
        "kind": kind, "unit": unit, "note": note, "value_si": v,
    }
    if unit == "mm":
        rec["value_mm"] = (v * 1.0e3) if v is not None else None
    else:
        rec["value"] = v
    return rec


def pintle_reference_geometry(inj, spec=None) -> dict[str, Any]:
    """Resolve the labeled reference geometry from a solved injector result.

    ``spec`` (the :class:`InjectorSpec`) is optional; when supplied it provides
    the input-side anchors that the result does not echo (tip radius, deflection
    angle, impingement distance, face OD).  Returns ``{architecture, axes,
    dimensions:[...], operating_point:{...}, notes:[...]}``; every linear
    dimension carries SI + mm values and a ``kind`` tag (``solved`` |
    ``schematic`` | ``input`` | ``not_applicable``).
    """
    g = getattr(spec, "geometry", None)
    Dp = float(inj.pintle_diameter)
    Rp = 0.5 * Dp
    ann = inj.annulus.detail
    slot = inj.slots.detail
    gap = float(ann.get("gap", 0.1 * Rp))
    Do = float(ann.get("outer_diameter", Dp + 2.0 * gap))
    t_sleeve = _t_sleeve(gap)
    Rc = float(inj.chamber_radius)
    Lc = float(inj.chamber_length)

    tip_radius = _f(getattr(g, "tip_radius", None)) if g is not None else None
    tip_d = 2.0 * tip_radius if tip_radius else 2.0 * Rp
    theta_pt = _f(getattr(g, "deflector_angle", 0.0)) if g is not None else 0.0
    x_imp = _f(getattr(g, "impingement_distance", None)) if g is not None else None
    face_od = _f(getattr(g, "face_od", None)) if g is not None else None
    if not face_od:
        face_od = 2.0 * Rc

    dims = [
        # --- pintle post / rod -----------------------------------------
        _dim("pintle_post", "D_pr", "pintle rod (post) outer diameter", Dp),
        _dim("pintle_post", "t_wall", "pintle wall thickness", _t_wall(Rp),
             kind="schematic", note="schematic hollow-post wall for CAD/drawing"),
        _dim("pintle_post", "L_body", "pintle protrusion length", _body_len(Dp),
             kind="schematic", note="schematic protrusion into chamber (3*D_pr)"),
        # --- pintle tip -------------------------------------------------
        _dim("pintle_tip", "D_tip", "pintle tip diameter", tip_d,
             note="short blunt/chamfered cap unless a tip radius is sized"),
        _dim("pintle_tip", "L_tip", "pintle tip cap length", _tip_len(Rp),
             kind="schematic", note="schematic blunt-tip length for CAD/drawing"),
        _dim("pintle_tip", "theta_pt", "tip / radial deflection angle",
             theta_pt if theta_pt is not None else 0.0, unit="deg", kind="input",
             note="radial-stream deflection; not a Son conical theta_pt unless "
                  "a cone is specified"),
        # --- annulus (axial stream) ------------------------------------
        _dim("annulus", "D_ann_i", "annulus inner diameter (= pintle OD)", Dp),
        _dim("annulus", "D_ann_o", "annulus outer diameter (sleeve ID)", Do),
        _dim("annulus", "h_ann", "annular gap", gap),
        _dim("annulus", "D_ob", "outer body (sleeve) outer diameter",
             Do + 2.0 * t_sleeve, kind="schematic",
             note="schematic: sleeve ID + 2*sleeve wall"),
        # --- radial slots (radial stream) ------------------------------
        _dim("slots", "N_slot", "radial slot count", int(inj.slot_count),
             unit="count"),
        _dim("slots", "w_slot", "slot width", _f(slot.get("slot_width"))),
        _dim("slots", "h_slot", "slot height", _f(slot.get("slot_height"))),
        _dim("slots", "L_slot", "slot depth", _f(slot.get("slot_depth")),
             note="NaN/blank when not sized in auto mode"),
        _dim("slots", "Dh_slot", "slot hydraulic diameter",
             _f(slot.get("hydraulic_diameter"))),
        _dim("slots", "w_web", "inter-slot web (ligament)", _f(slot.get("web"))),
        _dim("slots", "BF", "blockage factor (N*w/(pi*D_pr))",
             float(inj.blockage_factor), unit="fraction"),
        # --- spray / chamber interaction -------------------------------
        _dim("spray", "theta_s", "spray half-angle",
             float(inj.spray_half_angle_deg), unit="deg",
             note="arctan(radial/axial momentum), leading-order"),
        _dim("spray", "x_imp", "openings -> interaction (impingement) distance",
             x_imp, kind="input"),
        _dim("spray", "x_wall", "spray -> chamber-wall intercept (axial)",
             _f(inj.spray_wall_axial_distance),
             note="inf when the cone does not reach the wall within L_c"),
        _dim("chamber", "D_c", "chamber diameter", 2.0 * Rc, kind="input"),
        _dim("chamber", "L_c", "chamber length", Lc, kind="input"),
        _dim("chamber", "D_face", "injector face outer diameter", face_od,
             kind="input"),
        # --- movable-pintle-only (not applicable to fixed architecture) -
        _dim("movable", "L_open", "pintle opening distance", None,
             kind="not_applicable",
             note="movable/center-gap pintle only; fixed annulus+slots here"),
        _dim("movable", "D_cg", "center-gap diameter", None,
             kind="not_applicable", note="movable/center-gap pintle only"),
    ]

    def _stream(s):
        return {
            "role": s.role, "geometry": s.geometry,
            "mdot_kg_s": _f(s.mdot), "dp_pa": _f(s.dp), "cd": _f(s.cd),
            "area_m2": _f(s.area), "velocity_m_s": _f(s.velocity),
            "reynolds": _f(s.reynolds), "weber": _f(s.weber),
            "ohnesorge": _f(s.ohnesorge),
        }

    op = {
        "total_momentum_ratio": _f(inj.total_momentum_ratio),
        "blockage_factor": _f(inj.blockage_factor),
        "spray_half_angle_deg": _f(inj.spray_half_angle_deg),
        "spray_wall_axial_distance_m": _f(inj.spray_wall_axial_distance),
        "radial_stream": inj.radial_stream,
        "annulus": _stream(inj.annulus),
        "slots": _stream(inj.slots),
    }
    if getattr(inj, "feed_system", None) is not None:
        op["feed_system"] = inj.feed_system.to_dict()

    return {
        "architecture": "fixed_annulus_radial_slots",
        "axes": "axial +x from injector face (x=0); radius r about the axis",
        "dimensions": dims,
        "operating_point": op,
        "notes": [
            "Names follow the repo's fixed annulus+radial-slot pintle and Son "
            "et al. (JPP 33-4, 2017) where they correspond.",
            "kind=schematic values are construction aids (wall/sleeve/body), "
            "not solved results; kind=not_applicable marks movable-pintle-only "
            "dimensions absent from this fixed architecture.",
            "Spray angle is the leading-order momentum resultant; SP-8089 "
            "requires cold-flow testing for an authoritative spray distribution.",
        ],
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_pintle_parameters_json(geom: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(geom, indent=2) + "\n", encoding="utf-8")
    return path


def write_pintle_dimensions_csv(geom: dict, path) -> Path:
    path = Path(path)
    fields = ["group", "symbol", "name", "kind", "unit",
              "value_si", "value_mm", "value", "note"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for d in geom["dimensions"]:
            w.writerow(d)
    return path


# ---------------------------------------------------------------------------
# Package orchestrator
# ---------------------------------------------------------------------------
def export_pintle_package(
    inj,
    out_dir,
    *,
    spec=None,
    cad: str = "none",            # "none" | "reference" | "parts" | "machined"
    cad_format: str = "step",     # "step" | "stl" | "dxf"
    movable_sleeve: bool = False,
    radial_style: str = "holes",  # coaxial tip exit: "holes" | "slots"
) -> dict[str, Any]:
    """Write the full ``pintle/`` deliverable folder for a solved injector.

    The labeled schematic (SVG + PNG), parameters JSON and dimensions CSV are
    ALWAYS written.  CAD (STEP/STL/DXF) is optional and degrades gracefully when
    CadQuery is unavailable.  Machined mode always writes the manufacturing
    report and writes true STEP bodies when OpenCascade is installed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    notes: list[str] = []
    cad = (cad or "none").lower()
    cad_format = (cad_format or "step").lower()
    if cad == "auto":
        cad = "machined"
        cad_format = "step"
        notes.append(
            "auto pintle CAD selected the machined STEP package "
            "(injector_assembly_machined.step plus per-body STEP files)."
        )
    elif cad == "step":
        cad = "machined"
        cad_format = "step"
        notes.append(
            "legacy pintle CAD mode 'step' is treated as machined STEP output."
        )

    geom = pintle_reference_geometry(inj, spec=spec)

    # --- mandatory: parameters + dimensions ----------------------------
    files["parameters_json"] = str(write_pintle_parameters_json(
        geom, out_dir / "pintle_parameters.json"))
    files["dimensions_csv"] = str(write_pintle_dimensions_csv(
        geom, out_dir / "pintle_dimensions.csv"))

    # --- mandatory: labeled 2-D schematic (SVG + PNG) ------------------
    from raosim.injector_plots import plot_pintle_schematic
    fig = plot_pintle_schematic(inj, geom=geom, spec=spec)
    svg = out_dir / "pintle_schematic.svg"
    png = out_dir / "pintle_cross_section.png"
    fig.savefig(str(svg))                       # vector
    fig.savefig(str(png), dpi=200)              # raster
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    files["schematic_svg"] = str(svg)
    files["cross_section_png"] = str(png)

    # --- optional: CAD-neutral reference geometry ----------------------
    if cad and cad != "none":
        from raosim.injector_cad import (
            export_machined_pintle_cad,
            export_pintle_cad,
            cadquery_available,
        )
        if cad == "machined":
            # Machined mode builds the physically-coherent coaxial (TRW/Nardi)
            # 5-part injector with sealed inner/outer circuits (see
            # export_machined_pintle_cad -> raosim.injector_coaxial_cad).
            cad_files = export_machined_pintle_cad(
                inj, out_dir, spec=spec, fmt=cad_format,
                radial_style=radial_style)
            files.update(cad_files.get("files", {}))
            notes.extend(cad_files.get("notes", []))
        elif cad_format in ("step", "stl") and not cadquery_available():
            notes.append(
                "CAD export requested but CadQuery/OpenCascade is not installed; "
                "skipped (pip install cadquery). JSON/CSV/SVG/PNG still written.")
        else:
            cad_files = export_pintle_cad(
                inj, out_dir, mode=cad, fmt=cad_format,
                movable_sleeve=movable_sleeve)
            files.update(cad_files.get("files", {}))
            notes.extend(cad_files.get("notes", []))

    return {"dir": str(out_dir), "files": files, "geometry": geom,
            "notes": notes}
