"""Pump CAD package gates (docs/PUMP_CAD_IMPLEMENTATION_PLAN.md Phase 0).

Every emitted pump part mesh must pass the repo's own STL gates
(watertight, positive volume), CAD dimensions must be consumed from the
meanline's ``PumpReferenceGeometry`` rather than re-derived, and the
removed faceted pseudo-STEP must be impossible to produce.
"""

import math
from pathlib import Path

import pytest

from raosim.export import inspect_stl
from raosim.injector import FeedLineLedger, FeedSystemLedger
from raosim.pump_cad import (
    _write_part,
    export_pump_package,
    pump_reference_geometry,
)
from raosim.pumps import impeller_blade_camber, size_electric_pumps

_GRAVITY = 9.80665


def _feed_line(role: str, rho: float, mdot: float,
               tank_pressure: float | None = 4.0e5) -> FeedLineLedger:
    required = 3.2e6
    Q = mdot / rho
    if tank_pressure is None:
        rise = head = power = npsh = None
    else:
        rise = max(required - tank_pressure, 0.0)
        head = rise / (rho * _GRAVITY)
        power = Q * rise / 0.6
        npsh = tank_pressure - 2.0e3
    return FeedLineLedger(
        role=role,
        chamber_pressure=2.0e6,
        injector_dp=6.0e5,
        manifold_loss=1.0e5,
        manifold_screen_loss=0.0,
        regen_loss=2.0e5 if role == "fuel" else 0.0,
        line_valve_loss=1.5e5,
        control_margin=1.5e5,
        required_outlet_pressure=required,
        available_outlet_pressure=None,
        pressure_margin=None,
        density=rho,
        viscosity=1.7e-3,
        vapor_pressure=2.0e3,
        volumetric_flow=Q,
        required_pressure_rise=rise,
        required_pump_head=head,
        ideal_pump_power=power,
        flow_capacity=None,
        capacity_margin=None,
        npsh_available=npsh,
        npsh_required=None,
        npsh_margin=None,
        status="info",
    )


def _pump_result(oxidizer_tank_pressure: float | None = 4.0e5, spec=None):
    ledger = FeedSystemLedger(
        architecture="pump_fed",
        lines={
            "fuel": _feed_line("fuel", 810.0, 0.35),
            "oxidizer": _feed_line("oxidizer", 1141.0, 0.80,
                                   tank_pressure=oxidizer_tank_pressure),
        },
        governing_required_pressure=3.2e6,
        notes=[],
    )
    return size_electric_pumps(ledger, spec)


def _pump_result_with_spec(spec):
    return _pump_result(spec=spec)


def test_every_emitted_pump_part_passes_stl_gates(tmp_path):
    pump = _pump_result()
    pkg = export_pump_package(pump, tmp_path / "pump")

    stl_paths = [Path(p) for p in pkg["files"].values()
                 if str(p).endswith(".stl")]
    assert len(stl_paths) >= 11, sorted(pkg["files"])
    names = {p.name for p in stl_paths}
    for expected in ("fuel_impeller.stl", "fuel_inducer.stl",
                     "fuel_diffuser_volute.stl", "oxidizer_impeller.stl",
                     "shared_battery_pack.stl",
                     "pump_reference_assembly.stl"):
        assert expected in names

    for path in stl_paths:
        diag = inspect_stl(path)
        assert diag["watertight"], (path.name, diag)
        assert diag["volume_m3"] > 0.0, (path.name, diag)
        assert diag["signed_volume_m3"] > 0.0, (path.name, diag)

    for key, info in pkg["cad_diagnostics"].items():
        assert info["diagnostics"]["watertight"], key
        assert "mesh_gate" not in info, key


def test_pump_step_request_is_refused(tmp_path):
    pump = _pump_result()
    for fmt in ("step", "both"):
        with pytest.raises(ValueError, match="STEP"):
            export_pump_package(pump, tmp_path / "pump", cad_format=fmt)
    out = tmp_path / "pump"
    assert not out.exists() or not list(out.rglob("*.step"))
    # cad="none" writes no CAD, so the format is irrelevant by construction.
    pkg = export_pump_package(pump, tmp_path / "nocad", cad="none")
    assert not list((tmp_path / "nocad").rglob("*.stl"))
    assert pkg["files"]


def test_open_mesh_gate_blocks_and_waives(tmp_path):
    open_mesh = [((0.0, 0.0, 1.0),
                  (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
    path = tmp_path / "open.stl"
    with pytest.raises(RuntimeError, match="boundary edges"):
        _write_part(path, open_mesh, "stl", metadata={})
    assert not path.exists()
    info = _write_part(path, open_mesh, "stl", metadata={},
                       allow_open_mesh=True)
    assert path.exists()
    assert info["mesh_gate"] == "waived_not_watertight"
    assert not info["diagnostics"]["watertight"]


def test_cad_dimensions_come_from_reference_geometry():
    pump = _pump_result()
    geom = pump_reference_geometry(pump)

    for role, line in pump.lines.items():
        ref = line.reference_geometry
        assert ref is not None
        comp = geom["components"][role]

        disk = ref.impeller_disk
        impeller = comp["impeller"]
        assert impeller["outer_diameter_m"] == disk["outer_diameter_m"]
        assert impeller["inlet_diameter_m"] == disk["eye_diameter_m"]
        assert impeller["outlet_width_m"] == disk["outlet_width_m"]
        assert (impeller["blade_thickness_m"]
                == ref.blade_envelope["estimated_blade_thickness_m"])
        assert impeller["blade_count"] == ref.blade_envelope["blade_count"]

        stations = {row["station"]: row for row in ref.meridional_profile}
        assert impeller["axial_width_m"] == pytest.approx(
            stations["impeller_exit"]["x_m"] - stations["impeller_eye"]["x_m"],
            rel=1e-12,
        )

        helix = ref.inducer_helix
        inducer = comp["inducer"]
        assert inducer["diameter_m"] == helix["diameter_m"]
        assert inducer["pitch_m"] == helix["pitch_m"]
        assert inducer["wrap_angle_deg"] == helix["wrap_angle_deg"]
        assert inducer["hub_diameter_m"] == pytest.approx(
            helix["hub_ratio"] * helix["diameter_m"], rel=1e-12)
        assert inducer["length_m"] == pytest.approx(
            helix["pitch_m"] * helix["wrap_angle_deg"] / 360.0, rel=1e-12)

        ring = comp["diffuser_volute"]
        assert ring["axial_width_m"] == ref.diffuser_vane_ring["vane_width_m"]
        assert ring["vane_count"] == ref.diffuser_vane_ring["vane_count"]
        assert ring["inner_radius_m"] == stations["impeller_exit"]["radius_m"]
        assert ring["outer_radius_m"] == stations["diffuser_exit"]["radius_m"]
        assert ring["volute_exit_area_m2"] == ref.volute_scroll["exit_area_m2"]
        assert (ring["casing_inner_radius_m"]
                == ref.volute_scroll["casing_inner_radius_m"])

        shaft = comp["shaft"]
        assert shaft["diameter_m"] == ref.shaft_datum["diameter_m"]
        assert shaft["span_m"] == ref.shaft_datum["estimated_span_m"]

    # The dict form (pump.json round-trip) must resolve identically.
    geom_from_dict = pump_reference_geometry(pump.to_dict())
    assert geom_from_dict["components"] == geom["components"]


def test_pump_mount_pressure_screen_survives_json_shape_roundtrip():
    from raosim.engine_cad import pump_mount_flange_screen

    pump = _pump_result()
    object_screen = pump_mount_flange_screen(pump)
    dict_screen = pump_mount_flange_screen(pump.to_dict())

    assert set(dict_screen) == set(object_screen) == {"fuel", "oxidizer"}
    for role in object_screen:
        assert dict_screen[role] == object_screen[role]


def test_unsized_line_keeps_honest_not_sized_status(tmp_path):
    pump = _pump_result(oxidizer_tank_pressure=None)
    geom = pump_reference_geometry(pump)
    assert geom["components"]["oxidizer"]["status"] == "not_sized"
    assert "impeller" not in geom["components"]["oxidizer"]

    pkg = export_pump_package(pump, tmp_path / "pump")
    assert any("oxidizer pump CAD skipped" in note for note in pkg["notes"])
    written = {Path(p).name for p in pkg["files"].values()}
    assert "fuel_impeller.stl" in written
    assert not any(name.startswith("oxidizer_") for name in written)


def test_impeller_blade_camber_matches_log_spiral_closed_form():
    # Constant blade angle -> exact log spiral theta = ln(r2/r1)/tan(beta).
    pts = impeller_blade_camber(0.01, 0.03, 25.0, 25.0, samples=4001)
    exact = math.log(3.0) / math.tan(math.radians(25.0))
    assert pts[0]["theta_rad"] == 0.0
    assert pts[0]["radius_m"] == 0.01
    assert pts[-1]["radius_m"] == 0.03
    assert pts[-1]["theta_rad"] == pytest.approx(exact, rel=1e-6)
    # Monotonic wrap (backswept blades never reverse).
    thetas = [p["theta_rad"] for p in pts]
    assert all(b > a for a, b in zip(thetas, thetas[1:]))
    with pytest.raises(ValueError):
        impeller_blade_camber(0.03, 0.01, 20.0, 25.0)
    with pytest.raises(ValueError):
        impeller_blade_camber(0.01, 0.03, 0.0, 25.0)


class TestInducerBladeAnglesSP8052:
    """SP-8052 flat-plate inducer blade geometry (19710025474.pdf).

    Secs. 2.1.9/3.1.9: incidence-to-blade-angle ratio alpha/beta is the
    cavitation design parameter, 0.35 (thin) .. 0.50 (thick), mean 0.425
    preferred.  Secs. 2.1.10/3.1.10: constant-lead helix r*tan(beta) =
    const, lead = 2*pi*r*tan(beta).  Sec. 3.1.15: solidity 2.5.
    """

    def test_incidence_to_blade_angle_ratio_is_the_design_value(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ind = line.inducer
            assert ind.incidence_to_blade_ratio == pytest.approx(0.425)
            assert ind.incidence_deg == pytest.approx(
                0.425 * ind.inlet_tip_blade_angle_deg, rel=1e-9)
            assert (ind.inlet_flow_angle_deg + ind.incidence_deg
                    == pytest.approx(ind.inlet_tip_blade_angle_deg, rel=1e-9))

    def test_flow_angle_from_eye_continuity(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ind = line.inducer
            imp = line.impeller
            eye_area = (0.25 * math.pi * ind.diameter**2
                        * (1.0 - ind.hub_ratio**2))
            cm1 = line.volumetric_flow / eye_area
            u_tip = (2.0 * math.pi * imp.rpm / 60.0) * 0.5 * ind.diameter
            assert ind.inlet_flow_coefficient == pytest.approx(
                cm1 / u_tip, rel=1e-9)
            assert ind.inlet_flow_angle_deg == pytest.approx(
                math.degrees(math.atan(cm1 / u_tip)), rel=1e-9)

    def test_constant_lead_helix(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ind = line.inducer
            r_tip = 0.5 * ind.diameter
            beta_tip = math.radians(ind.inlet_tip_blade_angle_deg)
            beta_hub = math.radians(ind.hub_blade_angle_deg)
            assert ind.pitch == pytest.approx(
                2.0 * math.pi * r_tip * math.tan(beta_tip), rel=1e-9)
            # r * tan(beta) constant from tip to hub.
            assert r_tip * math.tan(beta_tip) == pytest.approx(
                ind.hub_ratio * r_tip * math.tan(beta_hub), rel=1e-9)

    def test_wrap_angle_from_developed_chord_solidity(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ind = line.inducer
            beta_tip = math.radians(ind.inlet_tip_blade_angle_deg)
            expected = math.degrees(
                2.0 * math.pi * ind.solidity * math.cos(beta_tip)
                / ind.blade_count
            )
            assert ind.wrap_angle_deg == pytest.approx(expected, rel=1e-9)
            # SP-8052 sec. 3.1.15 (2.0..2.5, Hong 2012 flew 2.6).
            assert 2.0 <= ind.solidity <= 2.6
            # SP-8052 sec. 3.1.14: three blades preferred (odd count).
            assert ind.blade_count == 3

    def test_hong2012_rocket_inducer_consistency(self):
        """Hong et al. 2012 (hong2012.pdf): 3-blade rocket turbopump
        inducer, inlet tip blade angle 10.4 deg, tip solidity 2.6.

        Cross-source check: for the SP-8052 alpha/beta band, Hong's blade
        angle implies a design tip flow coefficient inside SP-8052 inducer
        practice, and the preferred 0.425 ratio reproduces the angle.
        """
        for ratio in (0.35, 0.425, 0.50):
            phi_implied = math.tan(math.radians((1.0 - ratio) * 10.4))
            assert 0.06 <= phi_implied <= 0.13
        phi_mean = math.tan(math.radians(0.575 * 10.4))
        beta = math.degrees(math.atan(phi_mean)) / 0.575
        assert beta == pytest.approx(10.4, abs=1e-9)


class TestSP8109Fig16BladeCount:
    """SP-8109 fig. 16 minimum-blade-number chart (19740020848.pdf p.30).

    Digitized carpet: psi vs phi2 for Z = 3..24, zero prewhirl, shrouded,
    delta = 0.65; chart read-off ~ +/-0.02 in psi.
    """

    def test_chart_anchor_points(self):
        from raosim.pumps import sp8109_min_blade_count
        # Directly on digitized nodes.
        assert sp8109_min_blade_count(0.35, 0.095)["chart_minimum"] == 3
        assert sp8109_min_blade_count(0.50, 0.070)["chart_minimum"] == 6
        assert sp8109_min_blade_count(0.625, 0.050)["chart_minimum"] == 12
        # Between curves: needs the next count up.
        assert sp8109_min_blade_count(0.52, 0.070)["chart_minimum"] == 8

    def test_monotonic_in_head_and_flow(self):
        from raosim.pumps import sp8109_min_blade_count
        counts_psi = [
            sp8109_min_blade_count(psi, 0.08)["chart_minimum"]
            for psi in (0.35, 0.45, 0.55, 0.65)
        ]
        assert counts_psi == sorted(counts_psi)
        counts_phi = [
            sp8109_min_blade_count(0.50, phi)["chart_minimum"]
            for phi in (0.05, 0.10, 0.20, 0.28)
        ]
        assert counts_phi == sorted(counts_phi)

    def test_beyond_chart_flags_status(self):
        from raosim.pumps import sp8109_min_blade_count
        out = sp8109_min_blade_count(0.75, 0.20)
        assert out["status"] == "head_coefficient_beyond_digitized_chart"

    def test_snap_to_inducer_multiple(self):
        from raosim.pumps import sp8109_min_blade_count
        out = sp8109_min_blade_count(0.52, 0.07, multiple_of=3)
        assert out["chart_minimum"] == 8
        assert out["blade_count"] == 9
        assert "snapped_to_multiple_of_3" in out["basis"]

    def test_sized_pump_uses_chart_blade_count(self):
        pump = _pump_result()
        for line in pump.lines.values():
            imp = line.impeller
            # psi=0.55, phi2=0.08 defaults: the digitized Z=8 curve reads
            # psi 0.5485 there (just under target within the +/-0.02 chart
            # accuracy), so the conservative minimum is 10, snapped to 12
            # (multiple of the 3-blade inducer, SP-8052 sec. 3.1.14).
            assert imp.blade_count == 12
            assert imp.blade_count_source is not None
            assert "sp8109_fig16" in imp.blade_count_source
            # SP-8109 sec. 2.3.1.2: psi 0.55 sits in the many-blade regime
            # (3..5 blades only reach psi ~0.35..0.47).
            assert imp.blade_count > 5


class TestMeridionalChannel:
    """Quarter-ellipse hub/shroud channel (SP-8109 sec. 2.3.1.2)."""

    def test_endpoints_honor_solved_dimensions(self):
        pump = _pump_result()
        for line in pump.lines.values():
            imp = line.impeller
            ch = line.reference_geometry.meridional_channel
            hub, shroud = ch["hub_curve"], ch["shroud_curve"]
            r1 = 0.5 * imp.inlet_diameter
            r2 = 0.5 * imp.impeller_diameter
            assert shroud[0]["r_m"] == pytest.approx(r1, rel=1e-12)
            assert shroud[-1]["r_m"] == pytest.approx(r2, rel=1e-12)
            assert shroud[-1]["x_m"] == pytest.approx(
                -imp.outlet_width, rel=1e-12)
            assert hub[-1]["r_m"] == pytest.approx(r2, rel=1e-12)
            assert hub[-1]["x_m"] == pytest.approx(0.0, abs=1e-15)
            assert hub[0]["r_m"] == pytest.approx(
                ch["eye_hub_radius_m"], rel=1e-12)
            # Same eye plane for both curves.
            assert hub[0]["x_m"] == pytest.approx(
                shroud[0]["x_m"], rel=1e-12)

    def test_exact_inlet_and_exit_areas(self):
        pump = _pump_result()
        for line in pump.lines.values():
            imp = line.impeller
            ch = line.reference_geometry.meridional_channel
            r1 = 0.5 * imp.inlet_diameter
            r2 = 0.5 * imp.impeller_diameter
            r_hub = ch["eye_hub_radius_m"]
            assert ch["inlet_area_m2"] == pytest.approx(
                math.pi * (r1**2 - r_hub**2), rel=1e-9)
            assert ch["exit_area_m2"] == pytest.approx(
                2.0 * math.pi * r2 * imp.outlet_width, rel=1e-9)

    def test_meridional_velocity_screen(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ch = line.reference_geometry.meridional_channel
            cm1 = ch["inlet_meridional_velocity_m_s"]
            cm2 = ch["exit_meridional_velocity_m_s"]
            assert ch["cm_ratio"] == pytest.approx(cm2 / cm1, rel=1e-12)
            expected = "pass" if 1.0 <= ch["cm_ratio"] <= 1.5 else "warn"
            assert ch["cm_ratio_status"] == expected
            profile = ch["effective_area_profile_m2"]
            expected_contracting = all(
                b <= a * (1.0 + 1e-9)
                for a, b in zip(profile, profile[1:])
            )
            assert ch["area_progression_contracting"] == expected_contracting

    def test_annular_eye_velocity_triangle_and_blockage_close(self):
        pump = _pump_result()
        for line in pump.lines.values():
            imp = line.impeller
            tri = line.hydraulic_meanline.velocity_triangle
            ch = line.reference_geometry.meridional_channel
            eye = ch["eye_solve"]
            assert line.volumetric_flow == pytest.approx(
                ch["effective_inlet_area_m2"]
                * ch["inlet_meridional_velocity_m_s"], rel=1e-11
            )
            assert tri.inlet_meridional_velocity == pytest.approx(
                ch["inlet_meridional_velocity_m_s"], rel=1e-12
            )
            assert eye["inlet_flow_coefficient"] == pytest.approx(
                eye["target_inlet_flow_coefficient"], rel=1e-10
            )
            assert imp.inlet_blockage_fraction <= 0.20
            assert imp.exit_blockage_fraction <= 0.15 + 1e-12
            assert imp.inlet_blade_count + imp.splitter_blade_count == imp.blade_count
            assert imp.blade_count % imp.inlet_blade_count == 0

    def test_beta1_is_converged_metal_angle_not_legacy_placeholder(self):
        pump = _pump_result()
        geom = pump_reference_geometry(pump)
        for role, line in pump.lines.items():
            tri = line.hydraulic_meanline.velocity_triangle
            comp = geom["components"][role]["impeller"]
            assert tri.inlet_incidence_deg == pytest.approx(0.0, abs=1e-15)
            assert line.impeller.inlet_blade_angle_deg == pytest.approx(
                tri.inlet_blade_metal_angle_deg, rel=1e-12
            )
            assert comp["inlet_blade_angle_deg"] == pytest.approx(
                tri.inlet_blade_metal_angle_deg, rel=1e-12
            )
            assert comp["legacy_screening_inlet_angle_deg"] != pytest.approx(
                tri.inlet_blade_metal_angle_deg
            )

    def test_shaft_fit_is_upstream_of_cad(self):
        pump = _pump_result()
        for line in pump.lines.values():
            ch = line.reference_geometry.meridional_channel
            shaft_r = 0.5 * line.reference_geometry.shaft_datum["diameter_m"]
            required = (
                shaft_r + ch["shaft_fit_radial_clearance_m"]
                + ch["impeller_hub_wall_thickness_m"]
            )
            assert ch["eye_hub_radius_m"] >= required - 1e-12

    def test_impossible_blade_thickness_is_rejected(self):
        from raosim.pumps import PumpSizingSpec
        with pytest.raises(ValueError, match="free-area gate"):
            _pump_result_with_spec(PumpSizingSpec(blade_thickness_ratio=0.04))

    def test_blade_root_stress_closes_back_into_thickness_and_free_area(self):
        from raosim.pumps import PumpSizingSpec

        pump = _pump_result_with_spec(PumpSizingSpec(
            rotor_yield_strength=100.0e6,
            structural_fos=2.0,
        ))
        assert pump.feasibility.feasible
        for line in pump.lines.values():
            imp = line.impeller
            margin = line.thermal_stress.margins["blade_root_bending"]
            assert imp.blade_thickness_source == "blade_root_structural_closure"
            assert imp.blade_thickness >= (
                imp.blade_root_structural_minimum_thickness
                * (1.0 - 2.0e-9)
            )
            assert margin >= 1.0
            assert imp.exit_blockage_fraction <= 0.15 + 1.0e-12

    def test_fixed_rpm_reports_when_structural_root_cannot_fit_free_area(self):
        from raosim.pumps import ElectricDriveSpec, PumpSizingSpec

        pump = _pump_result_with_spec(PumpSizingSpec(
            drive=ElectricDriveSpec(rpm=100_000.0),
            rotor_yield_strength=100.0e6,
            structural_fos=2.0,
        ))
        assert not pump.feasibility.feasible
        failed = {
            gate.name for gate in pump.feasibility.gates
            if gate.status == "fail"
        }
        assert "impeller_exit_free_area_fuel" in failed
        assert "impeller_exit_free_area_oxidizer" in failed

    def test_channel_reaches_cad_manifest(self):
        pump = _pump_result()
        geom = pump_reference_geometry(pump)
        for role, line in pump.lines.items():
            ch = geom["components"][role]["meridional_channel"]
            assert ch["hub_curve"] == [
                {"x_m": p["x_m"], "r_m": p["r_m"]}
                for p in line.reference_geometry.meridional_channel[
                    "hub_curve"]
            ]
            symbols = {d["symbol"] for d in geom["dimensions"]
                       if d["role"] == role}
            assert "cm2/cm1" in symbols


class TestThrustBalanceHooks:
    """SP-8109 secs. 2.5.2/3.5.2 wear rings, seal land, balance holes."""

    def test_wear_rings_recommended_at_eye_diameter(self):
        pump = _pump_result()
        for line in pump.lines.values():
            tb = line.reference_geometry.thrust_balance
            assert tb["selection"] == "impeller_wear_rings"
            assert tb["hub_wear_ring_diameter_m"] == pytest.approx(
                line.impeller.inlet_diameter, rel=1e-12)
            assert tb["balance_holes"]["status"] == "clearance_not_specified"
            assert tb["balance_holes"]["diameter_m"] is None

    def test_seal_land_face_speed_is_solved(self):
        pump = _pump_result()
        for line in pump.lines.values():
            tb = line.reference_geometry.thrust_balance
            seal = tb["shaft_seal_land"]
            omega = 2.0 * math.pi * line.impeller.rpm / 60.0
            assert seal["face_speed_m_s"] == pytest.approx(
                omega * 0.5 * seal["diameter_m"], rel=1e-12)
            expected = ("pass" if seal["face_speed_m_s"]
                        <= seal["face_speed_limit_m_s"] else "warn")
            assert seal["status"] == expected

    def test_balance_holes_follow_four_times_clearance_rule(self):
        from raosim.pumps import PumpSizingSpec
        clearance = 1.2e-4
        pump = _pump_result_with_spec(
            PumpSizingSpec(wear_ring_radial_clearance=clearance))
        for line in pump.lines.values():
            tb = line.reference_geometry.thrust_balance
            holes = tb["balance_holes"]
            assert holes["status"] == "sized"
            d_ring = tb["hub_wear_ring_diameter_m"]
            clearance_area = math.pi * d_ring * clearance
            assert holes["seal_clearance_area_m2"] == pytest.approx(
                clearance_area, rel=1e-12)
            # SP-8109 sec. 3.5.2.1: flow area ~= 4 x seal-clearance area.
            assert holes["total_area_m2"] == pytest.approx(
                4.0 * clearance_area, rel=1e-12)
            area_from_holes = (holes["count"] * math.pi
                               * (0.5 * holes["diameter_m"])**2)
            assert area_from_holes == pytest.approx(
                holes["total_area_m2"], rel=1e-12)

    def test_hooks_reach_cad_manifest(self):
        pump = _pump_result()
        geom = pump_reference_geometry(pump)
        for role in pump.lines:
            tb = geom["components"][role]["thrust_balance"]
            assert tb["selection"] == "impeller_wear_rings"
            symbols = {d["symbol"] for d in geom["dimensions"]
                       if d["role"] == role}
            assert {"D_wr_hub", "d_bh", "U_seal"} <= symbols


class TestLiteratureBenchmarks:
    """Package-closure and envelope benchmarks against the pump corpus."""

    def test_lee2021_electric_pump_package_mass_closure(self):
        """Lee et al. 2021 (s42405-020-00325-z.pdf) 500 N / 20 bar / 600 s
        LOX/RP-1 electric-pump case, Tables 2 and 4 (transcribed from the
        paper text): motor 0.875 kW/kg at 87%, inverter 60 kW/kg at 85%,
        battery 325 Wh/kg / 0.650 kW/kg at 92.5% with 20% structure margin
        -> motor mass 451.2 g, battery mass 985.6 g (power-limited).
        """
        from raosim.pumps import (
            BatterySpec,
            ElectricDriveSpec,
            PumpSizingSpec,
            _battery_sizing,
            _drive_sizing,
        )
        spec = PumpSizingSpec(
            drive=ElectricDriveSpec(
                motor_efficiency=0.87,
                inverter_efficiency=0.85,
                motor_power_density=875.0,
                inverter_power_density=60.0e3,
            ),
            battery=BatterySpec(
                energy_density=325.0 * 3600.0,
                power_density=650.0,
                discharge_efficiency=0.925,
                structural_margin=1.20,
            ),
            burn_time=600.0,
        )
        # Table 4 motor mass 451.2 g at 0.875 kW/kg implies the total pump
        # shaft power of the case.
        shaft_power = 0.4512 * 875.0
        drive = _drive_sizing("fuel", shaft_power, 57940.0, spec,
                              bus_voltage=48.0, voltage_source="benchmark")
        assert drive.motor_mass == pytest.approx(0.4512, rel=1e-6)
        battery = _battery_sizing(drive.electric_power, spec, 48.0)
        assert battery.limiting == "power"
        assert battery.mass == pytest.approx(0.9856, rel=2e-3)

    def test_sp8109_specific_speed_envelope(self):
        """SP-8109 flight-proven stage specific speeds span 450..2100 US
        units (verified verbatim in the audit); Ns_US = 2733 * dimensionless
        omega*sqrt(Q)/(gH)^0.75."""
        pump = _pump_result()
        for line in pump.lines.values():
            ns_us = 2733.0 * line.impeller.specific_speed
            assert 450.0 <= ns_us <= 2100.0, ns_us

    def test_hong2012_defaults_within_practice(self):
        """Hong et al. 2012 rocket turbopump: 3-blade inducer (odd,
        SP-8052 preferred), tip solidity 2.6; our defaults sit in the same
        practice band."""
        from raosim.pumps import PumpSizingSpec
        spec = PumpSizingSpec()
        assert spec.inducer_blade_count == 3
        assert 2.0 <= spec.inducer_solidity <= 2.6


def test_cli_rejects_pump_step_format_without_cadquery(
    tmp_path, capsys, monkeypatch,
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import raosim.pump_cad_brep as pump_cad_brep
    from raosim import run_nozzle

    monkeypatch.setattr(pump_cad_brep, "cadquery_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        run_nozzle.main([
            "--no-banner",
            "--injector", "pintle",
            "--electric-pump",
            "--pump-cad-format", "step",
            "--out", str(tmp_path / "out"),
        ])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "CadQuery" in err and "--pump-cad-format" in err


# --------------------------------------------------------------------------- #
# Split-casing bolt layout must FIT the flange (2026-07-13 FINDING 2)          #
# --------------------------------------------------------------------------- #

def _joint_and_cad_layout(ring_like, ports_like):
    from raosim.pump_cad_brep import _split_joint_hole_layout
    return _split_joint_hole_layout(ring_like, ports_like)


def test_split_casing_bolts_fit_clear_arc_at_13kn_lox_scale():
    """Regression: 13 kN LOX volute demanded 56 x M3 body bolts, more than
    the flange circumference carries; sizing must grow bolt diameter until
    the clamp-driven count fits the clear arc at >= 2.5 d pitch, and the CAD
    hole layout must then place exactly that count at >= the minimum pitch.
    Numbers captured from builds/tests/13kn_sl_sandbox_20260713 stage P."""
    from raosim.pumps import PumpSizingSpec, _split_casing_joint_layout

    spec = PumpSizingSpec()
    joint = _split_casing_joint_layout(
        0.031313,          # casing inner radius [m]
        0.001612,          # casing wall [m]
        0.001133,          # volute exit area [m^2]
        3.87e6,            # design pressure [Pa]
        spec,
    )
    assert joint["bolt_layout_fits_clear_arc"] is True
    assert joint["bolt_screen_passed"] is True
    assert joint["bolt_nominal_diameter_m"] >= 3.0e-3
    assert joint["body_bolt_count"] >= 8
    assert (
        (joint["body_bolt_count"] + 1) * joint["bolt_min_pitch_m"]
        <= joint["clear_bolt_circumference_m"] + 1e-9
    )

    ring = {
        "casing_inner_radius_m": 0.031313,
        "volute_exit_area_m2": 0.001133,
        "split_casing_joint": joint,
    }
    ports = {"outlet_equivalent_diameter_m": 0.037980842896640735}
    layout = _joint_and_cad_layout(ring, ports)
    pts = layout["body_bolt_centers_mm"]
    assert len(pts) == joint["body_bolt_count"]
    min_pitch_mm = joint["bolt_min_pitch_m"] * 1e3
    spacing = [
        math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))
    ]
    assert min(spacing) >= min_pitch_mm - 1e-6, (min(spacing), min_pitch_mm)


def test_split_casing_bolt_layout_fits_for_both_solved_roles():
    pump = _pump_result()
    for role in ("fuel", "oxidizer"):
        geo = pump.lines[role].reference_geometry
        assert geo is not None
        scroll = geo.volute_scroll
        joint = scroll["split_casing_joint"]
        assert joint["bolt_layout_fits_clear_arc"] is True, (role, joint)
        ring = {
            "casing_inner_radius_m": scroll["casing_inner_radius_m"],
            "volute_exit_area_m2": scroll["exit_area_m2"],
            "split_casing_joint": joint,
        }
        ports = {
            "outlet_equivalent_diameter_m":
                geo.ports["outlet"]["equivalent_diameter_m"],
        }
        layout = _joint_and_cad_layout(ring, ports)
        assert len(layout["body_bolt_centers_mm"]) == joint["body_bolt_count"]
