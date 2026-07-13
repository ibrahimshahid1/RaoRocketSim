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

Fixed discrete exits retain the repository's existing slot/hole report.  A
Son et al. (*Design Procedure of a Movable Pintle Injector*, J. Propulsion &
Power 33-4, 2017) continuous-gap result instead reports its resolved internal
metering geometry, hard stops, calibration, actuator ledger, and independent
sheet-thickness evidence.  Those reports deliberately do not dispatch into the
fixed-geometry CAD exporters: a swept moving assembly is not implemented.
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


def _area_dim(group, symbol, name, value_m2, *, kind="solved", note=""):
    """One area record with SI authority and a convenient mm^2 value."""
    v = _f(value_m2)
    return {
        "group": group,
        "symbol": symbol,
        "name": name,
        "kind": kind,
        "unit": "mm^2",
        "note": note,
        "value_si": v,
        "value": (v * 1.0e6) if v is not None else None,
    }


def _stream_record(stream) -> dict[str, Any]:
    return {
        "role": stream.role,
        "geometry": stream.geometry,
        "mdot_kg_s": _f(stream.mdot),
        "dp_pa": _f(stream.dp),
        "cd": _f(stream.cd),
        "area_m2": _f(stream.area),
        "velocity_m_s": _f(stream.velocity),
        "reynolds": _f(stream.reynolds),
        "weber": _f(stream.weber),
        "ohnesorge": _f(stream.ohnesorge),
    }


def _movable_reference_geometry(inj, spec=None) -> dict[str, Any]:
    """Resolve a report-only Son-2017 continuous movable-pintle record."""
    act = getattr(inj, "actuation", None)
    if act is None:
        raise ValueError(
            "son_continuous_movable export requires the resolved actuation "
            "and metering ledger"
        )
    if str(getattr(act, "model_id", "")) != "son2017_continuous_radial_gap":
        raise ValueError(
            "son_continuous_movable export requires the Son-2017 continuous-"
            "gap actuation model"
        )

    g = getattr(spec, "geometry", None)
    movable = getattr(spec, "movable_pintle", None)
    detail = inj.slots.detail
    ann = inj.annulus.detail

    def _first(*values):
        for value in values:
            parsed = _f(value)
            if parsed is not None:
                return parsed
        return None

    d_post = _first(
        detail.get("post_diameter"),
        getattr(movable, "post_diameter", None),
        inj.pintle_diameter,
    )
    t_post = _first(
        detail.get("post_thickness"),
        getattr(movable, "post_thickness", None),
    )
    d_cg = _first(
        detail.get("center_gap_diameter"),
        getattr(movable, "center_gap_diameter", None),
    )
    d_pr = _first(
        detail.get("pintle_rod_diameter"),
        getattr(movable, "pintle_rod_diameter", None),
    )
    if any(value is None for value in (d_post, t_post, d_cg, d_pr)):
        raise ValueError(
            "son_continuous_movable report is missing D_post, t_post, D_cg, "
            "or D_pr"
        )
    if not (d_post > 0.0 and 0.0 < t_post < 0.5 * d_post):
        raise ValueError(
            "son_continuous_movable report requires 0 < t_post < D_post/2"
        )
    if not (d_cg > d_pr > 0.0):
        raise ValueError(
            "son_continuous_movable report requires D_cg > D_pr > 0"
        )
    if not (
        0.0 < act.opening_distance <= act.maximum_opening
        < act.transition_opening
    ):
        raise ValueError(
            "son_continuous_movable report requires the opening within hard "
            "stops and the open stop below the center-gap transition"
        )
    controlling_area = min(act.tip_minimum_area, act.center_gap_area)
    if not (
        act.tip_minimum_area > 0.0
        and act.tip_minimum_area < act.center_gap_area
        and math.isclose(
            act.effective_metering_area,
            controlling_area,
            rel_tol=1.0e-10,
            abs_tol=1.0e-15,
        )
    ):
        raise ValueError(
            "son_continuous_movable report requires a positive, tip-controlled "
            "A_eff=min(A_tip,A_cg) ledger"
        )

    gap = float(ann.get("gap", 0.0))
    ann_outer = float(ann.get("outer_diameter", d_post + 2.0 * gap))
    if not (gap > 0.0 and ann_outer > d_post):
        raise ValueError(
            "son_continuous_movable report requires a resolved positive fixed "
            "axial-annulus gap"
        )
    theta = _first(
        detail.get("tip_angle_deg"),
        getattr(g, "deflector_angle", None),
    )
    sheet = _f(act.sheet_thickness)
    sheet_source = (
        str(act.sheet_thickness_source)
        if act.sheet_thickness_source is not None else None
    )
    sheet_method = (
        str(act.sheet_thickness_method)
        if act.sheet_thickness_method is not None else None
    )
    sheet_sha = (
        str(act.sheet_thickness_artifact_sha256)
        if act.sheet_thickness_artifact_sha256 is not None else None
    )
    resolved_fingerprint = str(act.resolved_geometry_fingerprint_sha256)
    if (
        act.discharge_coefficient_model
        == "linear_calibrated_cd_vs_opening_fraction"
        and act.discharge_coefficient_geometry_fingerprint_sha256
        != resolved_fingerprint
    ):
        raise ValueError(
            "movable Cd calibration geometry fingerprint does not match the "
            "resolved Son geometry"
        )
    if (
        sheet is not None
        and act.sheet_thickness_geometry_fingerprint_sha256
        != resolved_fingerprint
    ):
        raise ValueError(
            "movable sheet-thickness geometry fingerprint does not match the "
            "resolved Son geometry"
        )
    equivalent_sheet = _f(detail.get("equivalent_exit_sheet_thickness"))

    dims = [
        _dim("movable_geometry", "D_post", "pintle post outer diameter", d_post),
        _dim("movable_geometry", "t_post", "pintle post thickness", t_post),
        _dim("movable_geometry", "D_pr", "pintle rod diameter", d_pr),
        _dim("movable_geometry", "D_cg", "center-gap diameter", d_cg),
        _dim("movable_geometry", "R_f", "post flow radius", 0.5 * d_post - t_post,
             note="R_f=D_post/2-t_post (Son 2017 Eq. 1)"),
        _dim("movable_geometry", "theta_pt", "pintle tip angle", theta,
             unit="deg", kind="input"),
        _dim("movable_kinematics", "L_open", "resolved opening distance",
             act.opening_distance),
        _dim("movable_kinematics", "L_min", "minimum normal opening distance",
             act.minimum_opening_distance,
             note="L_min=L_open*cos(theta_pt)"),
        _dim("movable_kinematics", "L_open_max", "physical open-stop distance",
             act.maximum_opening),
        _dim("movable_kinematics", "L_transition",
             "center-gap transition opening", act.transition_opening,
             note="A_tip=A_cg; commanded travel must remain below this value"),
        _area_dim("movable_metering", "A_tip", "Son tip/post minimum area",
                  act.tip_minimum_area,
                  note="Son 2017 Eq. 1"),
        _area_dim("movable_metering", "A_cg", "center-gap limiting area",
                  act.center_gap_area,
                  note="pi/4*(D_cg^2-D_pr^2)"),
        _area_dim("movable_metering", "A_eff", "effective metering area",
                  act.effective_metering_area,
                  note="min(A_tip,A_cg); design remains tip-controlled"),
        _area_dim("movable_metering", "A_sheet,ext",
                  "external 360-degree geometric opening area",
                  detail.get("external_sheet_inlet_area_360"),
                  note="2*pi*(D_post/2)*L_open; not the internal metering area"),
        _dim("sheet_handoff", "delta_eq",
             "continuity-equivalent external sheet thickness", equivalent_sheet,
             kind="derived",
             note="A_eff/(2*pi*R_exit); explicitly not VOF/measured sheet truth"),
        _dim("sheet_handoff", "delta_sheet",
             "measured/VOF liquid-sheet thickness", sheet,
             kind="evidence" if sheet is not None else "unresolved",
             note="independent evidence; never inferred from L_open or delta_eq"),
        _dim("fixed_axial_annulus", "D_ann_i", "annulus inner diameter", d_post),
        _dim("fixed_axial_annulus", "D_ann_o", "annulus outer diameter",
             ann_outer),
        _dim("fixed_axial_annulus", "h_ann", "fixed annular gap", gap),
        _dim("chamber", "D_c", "chamber diameter", 2.0 * inj.chamber_radius,
             kind="input"),
        _dim("chamber", "L_c", "chamber length", inj.chamber_length,
             kind="input"),
    ]

    actuation = act.to_dict()
    op = {
        "total_momentum_ratio": _f(inj.total_momentum_ratio),
        "blockage_factor": _f(inj.blockage_factor),
        "spray_half_angle_deg": _f(inj.spray_half_angle_deg),
        "spray_wall_axial_distance_m": _f(inj.spray_wall_axial_distance),
        "radial_stream": inj.radial_stream,
        "radial_exit_style": "continuous_radial_gap",
        "annulus": _stream_record(inj.annulus),
        "radial_openings": _stream_record(inj.slots),
        "continuous_radial_gap": _stream_record(inj.slots),
        "movable_actuation": actuation,
    }
    if getattr(inj, "feed_system", None) is not None:
        op["feed_system"] = inj.feed_system.to_dict()

    return {
        "architecture": "son_continuous_movable",
        "model_id": act.model_id,
        "axes": "axial +x from injector face (x=0); radius r about the axis",
        "dimensions": dims,
        "operating_point": op,
        "evidence": {
            "discharge_coefficient": {
                "model": act.discharge_coefficient_model,
                "source": act.discharge_coefficient_source,
                "artifact_sha256": (
                    act.discharge_coefficient_artifact_sha256
                ),
                "geometry_fingerprint_sha256": (
                    act.discharge_coefficient_geometry_fingerprint_sha256
                ),
                "resolved_geometry_fingerprint_sha256": (
                    act.resolved_geometry_fingerprint_sha256
                ),
                "calibration_reynolds_range": (
                    list(act.calibration_reynolds_range)
                    if act.calibration_reynolds_range is not None else None
                ),
                "calibration_pressure_drop_range_pa": (
                    list(act.calibration_pressure_drop_range)
                    if act.calibration_pressure_drop_range is not None else None
                ),
                "calibration_temperature_range_k": (
                    list(act.calibration_temperature_range)
                    if act.calibration_temperature_range is not None else None
                ),
                "calibration_cavitation_number_range": (
                    list(act.calibration_cavitation_number_range)
                    if act.calibration_cavitation_number_range is not None else None
                ),
                "calibration_fluid_name": act.calibration_fluid_name,
            },
            "sheet_thickness": {
                "value_m": sheet,
                "method": sheet_method,
                "source": sheet_source,
                "artifact_sha256": sheet_sha,
                "geometry_fingerprint_sha256": (
                    act.sheet_thickness_geometry_fingerprint_sha256
                ),
                "fluid_name": act.sheet_thickness_fluid_name,
                "opening_range_m": (
                    list(act.sheet_thickness_opening_range)
                    if act.sheet_thickness_opening_range is not None else None
                ),
                "pressure_drop_range_pa": (
                    list(act.sheet_thickness_pressure_drop_range)
                    if act.sheet_thickness_pressure_drop_range is not None else None
                ),
                "mass_flow_range_kg_s": (
                    list(act.sheet_thickness_mass_flow_range)
                    if act.sheet_thickness_mass_flow_range is not None else None
                ),
                "basis": "independent_measured_or_vof_evidence",
            },
            "position_metrology": {
                "source": act.metrology_source,
                "artifact_sha256": act.metrology_artifact_sha256,
            },
            "closed_stop_leakage": {
                "source": act.leakage_source,
                "artifact_sha256": act.leakage_artifact_sha256,
            },
            "actuator_and_material": {
                "source": act.actuator_source,
                "artifact_sha256": act.actuator_artifact_sha256,
            },
            "hardware_qualified": False,
        },
        "cad_status": {
            "available": False,
            "reason": (
                "no swept moving assembly with closed/open stops, running "
                "clearances, seals, and collision verification is implemented"
            ),
        },
        "notes": [
            "A_tip is the Son-2017 internal tip/post minimum area; A_cg is a "
            "hard transition cap, not a second area command.",
            "The fixed axial annulus does not move with pintle stroke; a "
            "separate upstream controller is required for its throttle schedule.",
            "Mechanical opening L_open, continuity-equivalent delta_eq, and "
            "measured/VOF delta_sheet are distinct quantities.",
            "This is a hydraulic/actuation report and schematic only. It is "
            "not manufacturing CAD or hardware qualification.",
        ],
    }


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
    radial_style = str(getattr(inj.slots, "geometry", "slots")).lower()
    declared_architecture = str(
        getattr(inj, "architecture", "fixed_discrete")
    ).lower()
    if radial_style == "continuous_radial_gap":
        if declared_architecture != "son_continuous_movable":
            raise ValueError(
                "continuous_radial_gap report requires architecture="
                "'son_continuous_movable'"
            )
        return _movable_reference_geometry(inj, spec=spec)
    if declared_architecture == "son_continuous_movable":
        raise ValueError(
            "son_continuous_movable report requires solved radial geometry "
            "'continuous_radial_gap'"
        )
    if radial_style not in {"slots", "holes"}:
        raise ValueError(
            f"unsupported solved radial exit geometry {radial_style!r}"
        )
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

    if radial_style == "holes":
        radial_dims = [
            _dim("holes", "N_hole", "radial hole count", int(inj.slot_count),
                 unit="count"),
            _dim("holes", "d_hole", "radial hole diameter",
                 _f(slot.get("hole_diameter"))),
            _dim("holes", "L_hole", "radial hole metering length",
                 _f(slot.get("hole_length"))),
            _dim("holes", "Dh_hole", "hole hydraulic diameter",
                 _f(slot.get("hydraulic_diameter"))),
            _dim("holes", "w_web", "inter-hole circumferential ligament",
                 _f(slot.get("web"))),
            _dim("holes", "BF", "blockage factor (d_hole/pitch)",
                 float(inj.blockage_factor), unit="fraction"),
        ]
    else:
        radial_dims = [
            _dim("slots", "N_slot", "radial slot count", int(inj.slot_count),
                 unit="count"),
            _dim("slots", "w_slot", "slot width",
                 _f(slot.get("slot_width"))),
            _dim("slots", "h_slot", "slot height",
                 _f(slot.get("slot_height"))),
            _dim("slots", "L_slot", "slot depth",
                 _f(slot.get("slot_depth")),
                 note="NaN/blank when not sized in auto mode"),
            _dim("slots", "Dh_slot", "slot hydraulic diameter",
                 _f(slot.get("hydraulic_diameter"))),
            _dim("slots", "w_web", "inter-slot web (ligament)",
                 _f(slot.get("web"))),
            _dim("slots", "BF", "blockage factor (N*w/(pi*D_pr))",
                 float(inj.blockage_factor), unit="fraction"),
        ]

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
        # --- radial metering openings (solved holes or slots) ----------
        *radial_dims,
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
             note="movable/center-gap pintle only; fixed annulus and radial "
                  f"{radial_style} here"),
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
        "radial_exit_style": radial_style,
        "annulus": _stream(inj.annulus),
        "radial_openings": _stream(inj.slots),
        radial_style: _stream(inj.slots),
    }
    if getattr(inj, "feed_system", None) is not None:
        op["feed_system"] = inj.feed_system.to_dict()

    return {
        "architecture": f"fixed_annulus_radial_{radial_style}",
        "axes": "axial +x from injector face (x=0); radius r about the axis",
        "dimensions": dims,
        "operating_point": op,
        "notes": [
            "Radial-opening names and dimensions are taken from the solved "
            f"{radial_style} hydraulic geometry; hole and slot fields are not "
            "interchanged.",
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


def _plot_movable_report_schematic(geom: dict):
    """Create an explicitly non-CAD control-area schematic for the report."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    by_symbol = {item["symbol"]: item for item in geom["dimensions"]}

    def _value(symbol):
        return by_symbol[symbol]["value_si"]

    l_open = _value("L_open") * 1.0e3
    l_max = _value("L_open_max") * 1.0e3
    l_transition = _value("L_transition") * 1.0e3
    areas = [
        _value("A_tip") * 1.0e6,
        _value("A_cg") * 1.0e6,
        _value("A_eff") * 1.0e6,
    ]

    fig, (ax, area_ax) = plt.subplots(1, 2, figsize=(11.0, 6.0))
    ax.axis("off")
    ax.set_title("Son-2017 movable-pintle control-area schematic", fontsize=11)
    ax.text(
        0.02,
        0.92,
        "Hydraulic/kinematic report only — not a manufacturing cross-section",
        transform=ax.transAxes,
        fontsize=9,
        color="#b2182b",
        weight="bold",
    )
    # A compact cause-and-limit diagram avoids implying that the repository
    # owns a swept solid model that does not exist.
    boxes = [
        (0.06, 0.62, f"commanded travel\nL_open = {l_open:.4f} mm"),
        (0.39, 0.62, f"Son Eq. 1\nA_tip = {areas[0]:.4f} mm²"),
        (0.72, 0.62, f"A_eff = min(A_tip,A_cg)\n{areas[2]:.4f} mm²"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7fbff", ec="#3182bd"),
        )
    for x0, x1 in ((0.30, 0.39), (0.63, 0.72)):
        ax.annotate(
            "",
            xy=(x1, 0.62),
            xytext=(x0, 0.62),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.3),
        )
    ax.text(
        0.39,
        0.36,
        f"center-gap cap A_cg = {areas[1]:.4f} mm²\n"
        f"open stop = {l_max:.4f} mm < transition = {l_transition:.4f} mm",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff7bc", ec="#d95f0e"),
    )
    ax.annotate(
        "hard cap",
        xy=(0.77, 0.53),
        xytext=(0.62, 0.40),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color="#d95f0e"),
        color="#d95f0e",
        fontsize=8,
    )
    sheet = geom["evidence"]["sheet_thickness"]
    sheet_text = (
        f"independent sheet evidence: {sheet['value_m'] * 1.0e3:.4f} mm\n"
        f"source: {sheet['source']}"
        if sheet["value_m"] is not None
        else "independent sheet thickness: unresolved"
    )
    ax.text(
        0.06,
        0.12,
        sheet_text + "\nL_open is not liquid-sheet thickness.",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="#edf8e9", ec="#31a354"),
        wrap=True,
    )

    colors = ["#3182bd", "#d95f0e", "#31a354"]
    bars = area_ax.bar(["A_tip", "A_cg", "A_eff"], areas, color=colors)
    area_ax.set_ylabel("area [mm²]")
    area_ax.set_title("Resolved controlling-area ledger")
    for bar, value in zip(bars, areas):
        area_ax.text(
            bar.get_x() + 0.5 * bar.get_width(),
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    area_ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Continuous movable pintle — report-only evidence view "
        "(not hardware-qualified)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


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
    radial_style: str | None = None,  # None -> solved hydraulic exit geometry
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
    cad_audit: dict[str, Any] | None = None
    cad = (cad or "none").lower()
    cad_format = (cad_format or "step").lower()
    solved_radial_style = str(
        getattr(inj.slots, "geometry", "slots")
    ).lower()
    if radial_style is None:
        radial_style = solved_radial_style
    else:
        radial_style = str(radial_style).lower()
        if radial_style != solved_radial_style:
            raise ValueError(
                "requested CAD radial_style does not match the solved "
                f"hydraulic exit: {radial_style!r} != "
                f"{solved_radial_style!r}"
            )
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
    if geom["architecture"] == "son_continuous_movable" and cad != "none":
        raise NotImplementedError(
            "CAD export is blocked for son_continuous_movable: the current "
            "fixed-geometry exporters do not implement a swept moving-pintle "
            "assembly with closed/open stops, running clearances, seals, and "
            "collision checks. Use cad='none' for the report-only package."
        )

    # --- mandatory: parameters + dimensions ----------------------------
    files["parameters_json"] = str(write_pintle_parameters_json(
        geom, out_dir / "pintle_parameters.json"))
    files["dimensions_csv"] = str(write_pintle_dimensions_csv(
        geom, out_dir / "pintle_dimensions.csv"))

    # --- mandatory: labeled 2-D schematic (SVG + PNG) ------------------
    if geom["architecture"] == "son_continuous_movable":
        fig = _plot_movable_report_schematic(geom)
    else:
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
            cad_audit = (
                cad_files.get("layout", {}).get("cad_export")
            )
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

    return {
        "dir": str(out_dir),
        "files": files,
        "geometry": geom,
        "cad_audit": cad_audit,
        "notes": notes,
    }
