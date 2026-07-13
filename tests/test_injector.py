"""Tests for raosim.injector — pintle hydraulic sizing and gates."""

import math

import pytest

from raosim.injector import (
    FeedLineSpec,
    FeedSystemSpec,
    InjectorManufacturingSpec,
    InjectorSpec,
    InjectorSpecError,
    InjectorUnsupportedState,
    PintleGeometrySpec,
    MovablePintleSpec,
    PropellantFeedSpec,
    evaluate_pintle_injector,
    resolve_feed_state,
    size_pintle_injector,
)
from raosim.pumps import (
    BatterySpec,
    ElectricDriveSpec,
    PumpSizingSpec,
    size_electric_pumps,
)
from raosim.movable_pintle import movable_geometry_fingerprint


# ---- representative LOX/RP-1 operating point ---------------------------
PC = 7.0e6
MR = 2.6
MDOT = 1.8116689741766580
MDOT_F = MDOT / (1.0 + MR)
MDOT_O = MR * MDOT_F


def _spec(**kw):
    geom = kw.pop("geometry", None) or PintleGeometrySpec(
        pintle_diameter=0.02, slot_count=24, radial_stream="fuel"
    )
    return InjectorSpec(
        type="pintle",
        sizing=kw.pop("sizing", "auto"),
        fuel_dp_fraction=kw.pop("fuel_dp", 0.2),
        oxidizer_dp_fraction=kw.pop("ox_dp", 0.2),
        fuel=PropellantFeedSpec(role="fuel", name="RP-1",
                                inlet_temperature=298.0),
        oxidizer=PropellantFeedSpec(role="oxidizer", name="LOX"),
        geometry=geom,
        manufacturing=kw.pop("manufacturing", InjectorManufacturingSpec()),
        **kw,
    )


def _size(spec=None, **kw):
    pc = kw.pop("Pc", PC)
    return size_pintle_injector(
        spec or _spec(), mdot_fuel=kw.pop("mdot_fuel", MDOT_F),
        mdot_oxidizer=kw.pop("mdot_oxidizer", MDOT_O), Pc=pc,
        mixture_ratio=kw.pop("mixture_ratio", MR),
        chamber_radius=kw.pop("chamber_radius", 0.0339),
        chamber_length=kw.pop("chamber_length", 0.10),
        gamma=1.24, Tc=3571.0, R_gas=379.6, **kw,
    )


def _movable_spec(*, calibrated=True, sheet_evidence=True, **overrides):
    movable = MovablePintleSpec(
        post_diameter=0.020,
        post_thickness=0.001,
        center_gap_diameter=0.012,
        pintle_rod_diameter=0.008,
        cd_vs_opening_fraction=(
            ((0.0, 0.62), (0.5, 0.70), (1.0, 0.76))
            if calibrated else ()
        ),
        cd_calibration_source=(
            "configuration-controlled unit-test cold-flow map"
            if calibrated else None
        ),
        cd_calibration_artifact_sha256=("b" * 64 if calibrated else None),
        cd_reynolds_range=((1.0, 1.0e9) if calibrated else None),
        cd_pressure_drop_range=((1.0, 1.0e9) if calibrated else None),
        cd_temperature_range=((200.0, 400.0) if calibrated else None),
        cd_cavitation_number_range=((0.0, 100.0) if calibrated else None),
        cd_fluid_name=("RP-1" if calibrated else None),
        position_tolerance=1.0e-6,
        position_feedback_resolution=1.0e-6,
        backlash=1.0e-6,
        closed_leakage_area=0.0,
        metrology_source="configuration-controlled metrology fixture",
        metrology_artifact_sha256="c" * 64,
        leakage_source="configuration-controlled leakage fixture",
        leakage_artifact_sha256="d" * 64,
        unbalanced_pressure_area=2.0e-5,
        spring_preload_force=5.0,
        seal_friction_force=4.0,
        moving_mass=0.2,
        maximum_acceleration=50.0,
        actuator_force_capacity=500.0,
        stem_diameter=0.006,
        stem_allowable_stress=200.0e6,
        actuator_source="configuration-controlled actuator/material fixture",
        actuator_artifact_sha256="e" * 64,
        sheet_thickness=(0.125e-3 if sheet_evidence else None),
        sheet_thickness_method=("vof" if sheet_evidence else None),
        sheet_thickness_source=(
            "configuration-controlled VOF unit-test fixture"
            if sheet_evidence else None
        ),
        sheet_thickness_artifact_sha256=("a" * 64 if sheet_evidence else None),
        sheet_thickness_fluid_name=("RP-1" if sheet_evidence else None),
        sheet_thickness_opening_range=(
            (1.0e-6, 2.0e-3) if sheet_evidence else None
        ),
        sheet_thickness_pressure_drop_range=(
            (1.0, 1.0e9) if sheet_evidence else None
        ),
        sheet_thickness_mass_flow_range=(
            (1.0e-9, 10.0) if sheet_evidence else None
        ),
    )
    for key, value in overrides.items():
        setattr(movable, key, value)
    geometry_digest = movable_geometry_fingerprint(
        movable, tip_angle_deg=20.0
    )
    if calibrated and movable.cd_geometry_fingerprint_sha256 is None:
        movable.cd_geometry_fingerprint_sha256 = geometry_digest
    if (
        sheet_evidence
        and movable.sheet_thickness_geometry_fingerprint_sha256 is None
    ):
        movable.sheet_thickness_geometry_fingerprint_sha256 = geometry_digest
    return InjectorSpec(
        type="pintle",
        architecture="son_continuous_movable",
        sizing="auto",
        fuel_dp_fraction=0.2,
        oxidizer_dp_fraction=0.2,
        fuel=PropellantFeedSpec(
            role="fuel", name="RP-1", inlet_temperature=298.0
        ),
        oxidizer=PropellantFeedSpec(role="oxidizer", name="LOX"),
        geometry=PintleGeometrySpec(
            pintle_diameter=0.020,
            radial_stream="fuel",
            radial_exit_style="continuous_radial_gap",
            deflector_angle=20.0,
        ),
        movable_pintle=movable,
    )


# ---- feed-state resolution ---------------------------------------------
class TestFeedState:
    def test_rp1_literature_constants(self):
        fs = resolve_feed_state(
            PropellantFeedSpec(role="fuel", name="RP-1",
                               inlet_temperature=298.0),
            default_pressure=8.4e6,
        )
        assert fs.liquid_ok
        assert fs.density == pytest.approx(810.0)
        assert fs.surface_tension == pytest.approx(0.023)
        assert "RP-1" in fs.source or "Sutton" in fs.source

    def test_lox_coolprop(self):
        fs = resolve_feed_state(
            PropellantFeedSpec(role="oxidizer", name="LOX"),
            default_pressure=8.4e6,
        )
        assert fs.liquid_ok
        # liquid oxygen near 90 K is ~1100-1150 kg/m^3
        assert 1050.0 < fs.density < 1250.0
        assert fs.surface_tension > 0.0

    def test_overrides_win(self):
        fs = resolve_feed_state(
            PropellantFeedSpec(
                role="fuel", name="mystery", inlet_temperature=300.0,
                density=900.0, viscosity=1e-3, surface_tension=0.02,
                vapor_pressure=1e3,
            ),
            default_pressure=8e6,
        )
        assert fs.density == 900.0 and fs.liquid_ok

    def test_unknown_propellant_raises(self):
        with pytest.raises(InjectorUnsupportedState):
            resolve_feed_state(
                PropellantFeedSpec(role="fuel", name="unobtainium"),
                default_pressure=8e6,
            )

    def test_forced_gas_not_liquid(self):
        fs = resolve_feed_state(
            PropellantFeedSpec(role="fuel", name="RP-1", phase="gas",
                               inlet_temperature=298.0),
            default_pressure=8e6,
        )
        assert not fs.liquid_ok


# ---- hydraulic closure --------------------------------------------------
class TestHydraulics:
    def test_area_mass_flow_round_trip(self):
        """A = mdot/(Cd sqrt(2 rho dp)) must reproduce mdot exactly."""
        r = _size()
        for role, s in r.streams.items():
            req = MDOT_F if role == "fuel" else MDOT_O
            assert s.mdot == pytest.approx(req, rel=1e-9)

    def test_velocity_definition(self):
        r = _size()
        for s in r.streams.values():
            rho = r.feed[s.role].density
            assert s.velocity == pytest.approx(s.mdot / (rho * s.area), rel=1e-9)

    def test_dimensionless_groups(self):
        r = _size()
        s = r.slots
        rho = r.feed[s.role].density
        mu = r.feed[s.role].viscosity
        sigma = r.feed[s.role].surface_tension
        assert s.reynolds == pytest.approx(
            rho * s.velocity * s.hydraulic_diameter / mu, rel=1e-9)
        assert s.weber == pytest.approx(
            rho * s.velocity**2 * s.hydraulic_diameter / sigma, rel=1e-9)
        assert s.ohnesorge == pytest.approx(
            math.sqrt(s.weber) / s.reynolds, rel=1e-6)

    def test_tmr_definition(self):
        r = _size()
        a, sl = r.annulus, r.slots
        assert r.total_momentum_ratio == pytest.approx(
            (sl.mdot * sl.velocity) / (a.mdot * a.velocity), rel=1e-9)

    def test_annulus_area_identity(self):
        """Annulus area equals pi/4 (Do^2 - Di^2)."""
        r = _size()
        d = r.annulus.detail
        area = math.pi / 4.0 * (d["outer_diameter"]**2 - d["inner_diameter"]**2)
        assert area == pytest.approx(r.annulus.area, rel=1e-9)

    def test_slot_area_identity(self):
        r = _size()
        d = r.slots.detail
        assert r.slot_count * d["slot_width"] * d["slot_height"] == \
            pytest.approx(r.slots.area, rel=1e-9)

    def test_higher_dp_smaller_area(self):
        low = _size(_spec(fuel_dp=0.1, ox_dp=0.1))
        high = _size(_spec(fuel_dp=0.4, ox_dp=0.4))
        assert high.slots.area < low.slots.area
        assert high.slots.velocity > low.slots.velocity

    def test_radial_stream_assignment(self):
        r_fuel = _size(_spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel")))
        r_ox = _size(_spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="oxidizer")))
        assert r_fuel.slots.role == "fuel" and r_fuel.annulus.role == "oxidizer"
        assert r_ox.slots.role == "oxidizer" and r_ox.annulus.role == "fuel"


class TestContinuousMovablePintle:
    def test_auto_solve_closes_son_area_mass_flow_and_actuator_ledger(self):
        r = _size(_movable_spec())
        act = r.actuation

        assert r.architecture == "son_continuous_movable"
        assert r.slots.geometry == "continuous_radial_gap"
        assert r.slot_count == 1
        assert act is not None
        assert r.slots.mdot == pytest.approx(MDOT_F, rel=1e-10)
        assert r.slots.area == pytest.approx(act.tip_minimum_area)
        assert act.effective_metering_area < act.center_gap_area
        assert 0.0 < act.opening_distance < act.maximum_opening
        assert act.maximum_opening < act.transition_opening
        assert r.slots.detail["external_sheet_inlet_area_360"] != pytest.approx(
            r.slots.area
        )
        assert r.slots.detail["equivalent_exit_sheet_thickness"] != pytest.approx(
            act.opening_distance
        )
        assert act.sheet_thickness != pytest.approx(act.opening_distance)
        assert act.actuator_force_margin > 1.0
        assert act.stem_stress_margin > 1.0
        assert _gate(r, "movable_center_gap_transition").status in ("pass", "warn")
        assert _gate(r, "movable_cd_calibration").status == "pass"
        assert _gate(r, "movable_actuator_force_margin").status == "pass"
        assert _gate(r, "movable_sheet_thickness_handoff").status == "pass"
        assert r.feasible

    def test_fixed_opening_reproduces_auto_delivered_flow(self):
        auto_spec = _movable_spec()
        auto = _size(auto_spec)
        fixed_spec = _movable_spec()
        fixed_spec.sizing = "fixed"
        fixed_spec.geometry.annulus_gap = auto.annulus.detail["gap"]
        fixed_spec.movable_pintle.maximum_opening = auto.actuation.maximum_opening
        fixed_spec.movable_pintle.commanded_opening = auto.actuation.opening_distance

        fixed = _size(fixed_spec)

        assert fixed.slots.mdot == pytest.approx(auto.slots.mdot, rel=2e-10)
        assert fixed.annulus.mdot == pytest.approx(auto.annulus.mdot, rel=2e-10)
        assert fixed.actuation.opening_fraction == pytest.approx(
            auto.actuation.opening_fraction
        )

    def test_uncalibrated_cd_and_incomplete_actuator_fail_closed(self):
        spec = _movable_spec(calibrated=False)
        spec.movable_pintle.unbalanced_pressure_area = None
        result = _size(spec)

        assert _gate(result, "movable_cd_calibration").status == "fail"
        assert _gate(result, "movable_actuator_force_margin").status == "fail"
        assert not result.feasible

    def test_cd_calibration_is_bound_to_fluid_and_operating_domain(self):
        wrong_fluid = _movable_spec()
        wrong_fluid.movable_pintle.cd_fluid_name = "water"
        result = _size(wrong_fluid)
        assert _gate(result, "movable_cd_calibration").status == "fail"
        assert not result.feasible

        wrong_dp = _movable_spec()
        wrong_dp.movable_pintle.cd_pressure_drop_range = (1.0, 1.0e5)
        result = _size(wrong_dp)
        assert _gate(result, "movable_cd_calibration").status == "fail"
        assert not result.feasible

        wrong_geometry = _movable_spec()
        wrong_geometry.movable_pintle.cd_geometry_fingerprint_sha256 = "d" * 64
        result = _size(wrong_geometry)
        assert _gate(result, "movable_cd_calibration").status == "fail"
        assert not result.feasible

    def test_sheet_evidence_outside_solved_state_is_rejected(self):
        spec = _movable_spec()
        spec.movable_pintle.sheet_thickness_opening_range = (
            1.0e-6,
            2.0e-6,
        )
        result = _size(spec)

        assert _gate(result, "movable_sheet_thickness_handoff").status == "fail"
        assert not result.feasible

        geometry_mismatch = _movable_spec()
        geometry_mismatch.movable_pintle.sheet_thickness_geometry_fingerprint_sha256 = (
            "d" * 64
        )
        result = _size(geometry_mismatch)
        assert _gate(result, "movable_sheet_thickness_handoff").status == "fail"
        assert not result.feasible

    @pytest.mark.parametrize(
        ("source_field", "digest_field", "gate_name"),
        (
            (
                "metrology_source",
                "metrology_artifact_sha256",
                "movable_position_authority",
            ),
            (
                "leakage_source",
                "leakage_artifact_sha256",
                "movable_closed_stop_leakage",
            ),
            (
                "actuator_source",
                "actuator_artifact_sha256",
                "movable_actuator_force_margin",
            ),
        ),
    )
    def test_hardware_ledgers_require_source_hash_evidence(
        self, source_field, digest_field, gate_name
    ):
        spec = _movable_spec()
        setattr(spec.movable_pintle, source_field, None)
        setattr(spec.movable_pintle, digest_field, None)
        result = _size(spec)

        assert _gate(result, gate_name).status == "fail"
        assert not result.feasible

    def test_architecture_and_geometry_must_be_selected_together(self):
        wrong_architecture = _movable_spec()
        wrong_architecture.architecture = "fixed_discrete"
        with pytest.raises(InjectorSpecError, match="requires architecture"):
            _size(wrong_architecture)

        wrong_geometry = _spec()
        wrong_geometry.architecture = "son_continuous_movable"
        with pytest.raises(InjectorSpecError, match="requires radial_exit_style"):
            _size(wrong_geometry)

        ignored_movable_input = _spec()
        ignored_movable_input.movable_pintle.post_diameter = 0.020
        with pytest.raises(InjectorSpecError, match="not ignored"):
            _size(ignored_movable_input)

    def test_open_stop_at_center_gap_transition_is_rejected(self):
        from raosim.movable_pintle import transition_opening

        spec = _movable_spec()
        transition = transition_opening(
            spec.movable_pintle,
            tip_angle_deg=spec.geometry.deflector_angle,
        )
        spec.movable_pintle.maximum_opening = transition
        with pytest.raises(ValueError, match="reaches/exceeds"):
            _size(spec)

    def test_partial_sheet_evidence_is_rejected(self):
        spec = _movable_spec(sheet_evidence=False)
        spec.movable_pintle.sheet_thickness = 0.1e-3
        with pytest.raises(InjectorSpecError, match="thickness, method, source"):
            _size(spec)


# ---- gates --------------------------------------------------------------
def _gate(r, name):
    return next(g for g in r.gates if g.name == name)


class TestGates:
    def test_auto_closure_passes(self):
        r = _size()
        assert _gate(r, "mass_flow_mixture_ratio_closure").status == "pass"
        assert r.feasible

    def test_blockage_fail_when_too_many_slots(self):
        # 400 slots on a 20 mm pintle overflow the circumference.
        r = _size(_spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=400, radial_stream="fuel")))
        assert r.blockage_factor >= 1.0
        assert _gate(r, "slot_blockage").status == "fail"
        assert not r.feasible

    def test_min_feature_fail(self):
        # A huge min-feature floor cannot be met by sub-mm slots.
        r = _size(_spec(manufacturing=InjectorManufacturingSpec(
            min_feature=5e-3)))
        assert _gate(r, "min_slot_width").status == "fail"

    def test_cavitation_gate_present(self):
        r = _size()
        g = _gate(r, "cavitation_oxidizer")
        assert g.status in ("pass", "warn", "fail")

    def test_validation_status_always_warns(self):
        r = _size()
        assert _gate(r, "validation_status").status == "warn"

    def test_stability_gates_report(self):
        r = _size()
        assert _gate(r, "chamber_acoustic_modes").status == "info"
        assert _gate(r, "feed_system_chug").status in ("pass", "warn", "fail")
        assert _gate(r, "ntau_coupling").status in ("info", "warn")

    def test_insufficient_supplied_feed_pressure_fails(self):
        spec = _spec()
        spec.fuel.inlet_pressure = 7.1e6
        spec.oxidizer.inlet_pressure = 7.1e6
        r = _size(spec)
        assert _gate(r, "upstream_pressure_fuel").status == "fail"
        assert _gate(r, "upstream_pressure_oxidizer").status == "fail"
        assert not r.feasible


# ---- phase guard + compressible (gas) branch ----------------------------
class TestPhaseGuard:
    def test_rejects_gas_without_gamma_R(self):
        # RP-1 forced to gas: the literature table has no gas gamma/R.
        spec = _spec()
        spec.fuel.phase = "gas"
        with pytest.raises(InjectorUnsupportedState):
            _size(spec)

    def test_rejects_subcritical_low_pressure_flashing(self):
        # N2O4 near its vapor pressure is two-phase: neither branch applies.
        spec = InjectorSpec(
            type="pintle",
            fuel=PropellantFeedSpec(role="fuel", name="MMH"),
            oxidizer=PropellantFeedSpec(
                role="oxidizer", name="N2O4", inlet_temperature=293.0,
                inlet_pressure=1.0e5),  # ~ vapor pressure
        )
        with pytest.raises(InjectorUnsupportedState):
            size_pintle_injector(
                spec, mdot_fuel=0.5, mdot_oxidizer=1.0, Pc=7e6,
                mixture_ratio=2.0, chamber_radius=0.03, chamber_length=0.1,
                gamma=1.23, Tc=3122.0, R_gas=386.0)

    def test_accepts_gaseous_oxygen_compressible(self):
        # GOX through the annulus: compressible branch, larger area, subsonic.
        spec = _spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel"))
        spec.oxidizer = PropellantFeedSpec(
            role="oxidizer", name="oxygen", inlet_temperature=280.0,
            phase="gas")
        r = _size(spec)
        assert r.annulus.detail["injection"]["branch"].startswith("compressible")
        # gas is far less dense -> needs much more area than liquid LOX (~33 mm^2)
        assert r.annulus.area > 5e-5
        assert any(g.name == "injection_state_oxidizer" for g in r.gates)
        # A gas hydraulic branch is never interpreted as a droplet stream.
        ox = r.atomization.streams["oxidizer"]
        assert not ox.applicable
        assert math.isnan(ox.sauter_mean_diameter)
        assert "not a liquid droplet phase" in ox.validity_reason

    def test_choked_gas_reports_choke(self):
        # A large oxidizer dp-fraction (P0 >> 2 Pc) chokes the gas stream.
        spec = _spec(ox_dp=1.5, geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel"))
        spec.oxidizer = PropellantFeedSpec(
            role="oxidizer", name="oxygen", inlet_temperature=280.0,
            phase="gas")
        r = _size(spec)
        assert r.annulus.detail["injection"]["choked"] is True
        g = _gate(r, "injection_state_oxidizer")
        assert "CHOKED" in g.detail


# ---- fixed mode ---------------------------------------------------------
class TestFixedMode:
    def test_fixed_geometry_not_resized_and_large_drift_fails(self):
        geom = PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            annulus_gap=3e-4, slot_width=5e-4, slot_height=5e-4, slot_depth=1e-3)
        r = _size(_spec(sizing="fixed", geometry=geom))
        # supplied dimensions are preserved exactly (no silent resize)
        assert r.slots.detail["slot_width"] == pytest.approx(5e-4)
        assert r.annulus.detail["gap"] == pytest.approx(3e-4)
        # a large delivered-flow drift FAILS regardless of sizing mode
        assert _gate(r, "mass_flow_mixture_ratio_closure").status == "fail"
        assert not r.feasible

    def test_fixed_roundtrip_auto_geometry_closes(self):
        """Feeding the auto-sized geometry back in fixed mode closes the flow."""
        auto = _size()
        geom = PintleGeometrySpec(
            pintle_diameter=auto.pintle_diameter,
            slot_count=auto.slot_count, radial_stream=auto.radial_stream,
            annulus_gap=auto.annulus.detail["gap"],
            slot_width=auto.slots.detail["slot_width"],
            slot_height=auto.slots.detail["slot_height"], slot_depth=1e-3)
        r = _size(_spec(sizing="fixed", geometry=geom))
        assert _gate(r, "mass_flow_mixture_ratio_closure").status == "pass"

    def test_fixed_without_dimensions_raises(self):
        from raosim.injector import InjectorSpecError
        with pytest.raises(InjectorSpecError):
            _size(_spec(sizing="fixed"))  # no annulus_gap / slot_width supplied


# ---- spec validation front gate -----------------------------------------
class TestValidation:
    def test_zero_cd_raises(self):
        from raosim.injector import InjectorSpecError
        s = _spec()
        s.fuel_cd = 0.0
        with pytest.raises(InjectorSpecError):
            _size(s)

    def test_zero_slot_count_raises(self):
        from raosim.injector import InjectorSpecError
        with pytest.raises(InjectorSpecError):
            _size(_spec(geometry=PintleGeometrySpec(
                pintle_diameter=0.02, slot_count=0, radial_stream="fuel")))

    def test_zero_dp_raises(self):
        from raosim.injector import InjectorSpecError
        s = _spec()
        s.fuel_dp_fraction = 0.0
        with pytest.raises(InjectorSpecError):
            _size(s)

    def test_nonpositive_flow_raises(self):
        from raosim.injector import InjectorSpecError
        with pytest.raises(InjectorSpecError):
            _size(mdot_fuel=0.0)

    @pytest.mark.parametrize(
        "geometry",
        [
            PintleGeometrySpec(
                pintle_diameter=0.02, slot_count=24, tip_radius=-1e-3
            ),
            PintleGeometrySpec(
                pintle_diameter=0.02, slot_count=24,
                impingement_distance=-1e-3,
            ),
            PintleGeometrySpec(
                pintle_diameter=0.02, slot_count=24, deflector_angle=180.0
            ),
        ],
    )
    def test_invalid_geometry_raises(self, geometry):
        from raosim.injector import InjectorSpecError
        with pytest.raises(InjectorSpecError):
            _size(_spec(geometry=geometry))


# ---- activated geometry parameters --------------------------------------
class TestActiveParameters:
    def test_unreachable_target_momentum_ratio_is_rejected(self):
        s = _spec()
        s.target_momentum_ratio = 5.0  # far from the ~0.5 achieved
        from raosim.injector import InjectorSpecError
        with pytest.raises(InjectorSpecError, match="achievable TMR"):
            _size(s)

    def test_target_momentum_ratio_actively_solves_radial_dp(self):
        s = _spec()
        s.target_momentum_ratio = 0.50
        r = _size(s)
        assert r.total_momentum_ratio == pytest.approx(0.50, rel=1e-7)
        assert r.momentum_targeting["active"]
        assert r.momentum_targeting["solved_role"] == "fuel"
        assert _gate(r, "target_momentum_ratio").status == "pass"

    def test_tip_radius_and_impingement_used(self):
        geom = PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            tip_radius=0.008, impingement_distance=0.01,
            face_od=0.08)
        base = _size()
        r = _size(_spec(geometry=geom,
                        manufacturing=InjectorManufacturingSpec(
                            min_feature=3e-4, edge_distance_min=4e-4)))
        names = {g.name for g in r.gates}
        assert {"pintle_tip_radius", "impingement_distance",
                "injector_face_od", "edge_distance"} <= names
        # impingement distance pushes the wall interception further downstream
        assert r.spray_wall_axial_distance > base.spray_wall_axial_distance

    def test_tip_radius_exceeding_pintle_fails(self):
        r = _size(_spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            tip_radius=0.05)))  # 50 mm tip on a 20 mm pintle
        assert _gate(r, "pintle_tip_radius").status == "fail"


class TestIntegratedCoupling:
    class _Cooling:
        method = "regenerative"
        coolant = "rp1"

        def __init__(self, mdot):
            self.coolant_mass_flow = mdot

    def test_regen_flow_mismatch_fails_and_does_not_use_outlet_state(self):
        r = evaluate_pintle_injector(
            _spec(),
            mdot_fuel=MDOT_F,
            mdot_oxidizer=MDOT_O,
            Pc=PC,
            mixture_ratio=MR,
            chamber_radius=0.0339,
            chamber_length=0.10,
            gamma=1.24,
            Tc=3571.0,
            R_gas=379.6,
            cooling=self._Cooling(10.0),
            cooling_result={
                "coolant_outlet_temperature": 500.0,
                "coolant_outlet_pressure": 8.4e6,
            },
        )
        assert _gate(r, "regen_fuel_flow_closure").status == "fail"
        assert r.feed["fuel"].temperature == pytest.approx(298.0)
        assert not r.feasible

    def test_closed_regen_flow_hands_outlet_state_to_injector(self):
        spec = _spec()
        spec.fuel.inlet_temperature = None
        r = evaluate_pintle_injector(
            spec,
            mdot_fuel=MDOT_F,
            mdot_oxidizer=MDOT_O,
            Pc=PC,
            mixture_ratio=MR,
            chamber_radius=0.0339,
            chamber_length=0.10,
            gamma=1.24,
            Tc=3571.0,
            R_gas=379.6,
            cooling=self._Cooling(MDOT_F),
            cooling_result={
                "coolant_outlet_temperature": 410.0,
                "coolant_outlet_pressure": 8.4e6,
            },
        )
        assert _gate(r, "regen_fuel_flow_closure").status == "pass"
        assert r.feed["fuel"].temperature == pytest.approx(410.0)
        assert r.feed["fuel"].pressure == pytest.approx(8.4e6)

    def test_regen_pressure_drop_charged_to_feed_ledger(self):
        r = evaluate_pintle_injector(
            _spec(),
            mdot_fuel=MDOT_F,
            mdot_oxidizer=MDOT_O,
            Pc=PC,
            mixture_ratio=MR,
            chamber_radius=0.0339,
            chamber_length=0.10,
            gamma=1.24,
            Tc=3571.0,
            R_gas=379.6,
            cooling=self._Cooling(MDOT_F),
            cooling_result={
                "coolant_outlet_temperature": 410.0,
                "coolant_outlet_pressure": 8.4e6,
                "coolant_pressure_drop": 2.0e6,
            },
        )
        ln = r.feed_system.lines["fuel"]
        assert r.feed["fuel"].temperature == pytest.approx(410.0)
        assert _gate(r, "regen_fuel_flow_closure").status == "fail"
        assert "differs from jacket outlet" in _gate(
            r, "regen_fuel_flow_closure"
        ).detail
        assert ln.regen_loss == pytest.approx(2.0e6, rel=1e-12)
        assert ln.required_outlet_pressure == pytest.approx(
            ln.chamber_pressure + ln.injector_dp + ln.regen_loss, rel=1e-12)


# ---- spray atomization / vaporization screen ----------------------------
class TestAtomization:
    def test_smd_in_micron_range(self):
        r = _size(Pc=1.5e6)
        at = r.atomization
        assert at is not None
        for role, s in at.streams.items():
            # The screen is bounded by its passage hydraulic diameter; it is
            # not asserted to be a universal "tens of microns" prediction.
            assert 0.0 < s.sauter_mean_diameter <= r.streams[role].hydraulic_diameter
            assert s.aerodynamic_weber > 0

    def test_chamber_gas_density(self):
        r = _size(Pc=1.5e6)
        # rho_g = Pc/(R Tc)
        assert r.atomization.chamber_gas_density == pytest.approx(
            1.5e6 / (379.6 * 3571.0), rel=1e-6)

    def test_hinze_reitz_and_d2_equations_are_independently_recomputed(self):
        """Verify the three screen equations from primitive result values."""
        pc = 1.5e6
        chamber_length = 0.10
        r = _size(Pc=pc, chamber_length=chamber_length)
        rho_g = pc / (379.6 * 3571.0)
        K = 1.0e-6
        for role, item in r.atomization.streams.items():
            stream = r.streams[role]
            sigma = r.feed[role].surface_tension
            expected_we = (
                rho_g * stream.velocity**2 * stream.hydraulic_diameter / sigma
            )
            expected_d32 = min(
                13.0 * sigma / (rho_g * stream.velocity**2),
                stream.hydraulic_diameter,
            )
            expected_breakup = 15.0 * stream.hydraulic_diameter
            expected_t99 = (
                expected_d32**2 * (1.0 - 0.01 ** (2.0 / 3.0)) / K
            )
            expected_vap_length = stream.velocity * expected_t99
            residence = max(0.0, chamber_length - expected_breakup) / stream.velocity
            remaining_d2 = max(0.0, expected_d32**2 - K * residence)
            expected_vaporized = 1.0 - (
                remaining_d2 / expected_d32**2
            ) ** 1.5
            assert item.aerodynamic_weber == pytest.approx(expected_we)
            assert item.sauter_mean_diameter == pytest.approx(expected_d32)
            assert item.breakup_length == pytest.approx(expected_breakup)
            assert item.vaporization_length == pytest.approx(expected_vap_length)
            assert item.vaporized_fraction == pytest.approx(expected_vaporized)

    def test_short_chamber_warns_and_drops_efficiency(self):
        """A too-short chamber cannot develop combustion: warn + η_c* < 1."""
        short = _size(Pc=1.5e6, chamber_length=0.01)
        long = _size(Pc=1.5e6, chamber_length=0.5)
        assert (short.atomization.predicted_cstar_efficiency
                <= long.atomization.predicted_cstar_efficiency)
        assert short.atomization.development_margin < long.atomization.development_margin
        g = _gate(short, "combustion_development_length")
        assert g.status in ("warn", "pass")  # surrogate never hard-fails
        # the surrogate must not by itself make the design infeasible
        assert g.status != "fail"

    def test_efficiency_bounded(self):
        r = _size(Pc=1.5e6)
        assert 0.0 <= r.atomization.eta_vaporization <= 1.0
        assert r.atomization.eta_mixing is None
        assert r.atomization.eta_combustion is None
        assert r.atomization.eta_cstar is None
        # Compatibility alias is explicitly only the vaporization screen.
        assert (r.atomization.predicted_cstar_efficiency
                == r.atomization.eta_vaporization)

    def test_atomization_in_to_dict(self):
        d = _size(Pc=1.5e6).to_dict()
        assert d["atomization"]["model"].startswith("hinze")
        assert "fuel" in d["atomization"]["streams"]
        assert d["atomization"]["eta_cstar"] is None

    def test_transcritical_pressure_disables_droplet_screen(self):
        r = _size(Pc=7.0e6)
        assert r.atomization.limiting_role is None
        assert math.isnan(r.atomization.eta_vaporization)
        assert all(not s.applicable for s in r.atomization.streams.values())
        assert _gate(r, "atomization_applicability_oxidizer").status == "warn"

    def test_round_holes_have_their_own_area_and_hydraulic_diameter(self):
        s = _spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            radial_exit_style="holes",
        ))
        r = _size(s, Pc=1.5e6)
        radial = r.slots
        d = radial.detail["hole_diameter"]
        assert radial.geometry == "holes"
        assert radial.detail["injection_form"] == "round_hole_jet"
        assert radial.area == pytest.approx(24 * math.pi * d**2 / 4.0)
        assert radial.hydraulic_diameter == pytest.approx(d)
        assert radial.detail["length_over_dh"] == pytest.approx(2.0)
        pitch = math.pi * r.pintle_diameter / 24
        assert radial.detail["web"] == pytest.approx(pitch - d)
        assert _gate(r, "min_hole_diameter").name == "min_hole_diameter"

    def test_fixed_round_hole_diameter_sets_delivered_area(self):
        d = 8.0e-4
        s = _spec(sizing="fixed", geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            radial_exit_style="holes", radial_hole_diameter=d,
            radial_hole_length=2.0e-3, annulus_gap=3.0e-4,
        ))
        r = _size(s, Pc=1.5e6)
        assert r.slots.area == pytest.approx(24 * math.pi * d**2 / 4.0)
        assert r.slots.detail["length_over_dh"] == pytest.approx(2.5)

    def test_continuous_radial_sheet_requires_separate_model(self):
        from raosim.injector import InjectorSpecError
        s = _spec(geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_stream="fuel",
            radial_exit_style="continuous_sheet",
        ))
        with pytest.raises(InjectorSpecError, match="radial_exit_style"):
            _size(s, Pc=1.5e6)

    def test_explicit_cstar_coupling_requires_other_efficiencies(self):
        from raosim.injector import couple_atomization_to_performance
        r = _size(Pc=1.5e6)
        coupled = couple_atomization_to_performance(
            r.atomization, ideal_cstar=1700.0, chamber_pressure=1.5e6,
            throat_area=1.0e-3, eta_mixing=0.97, eta_combustion=0.99,
        )
        expected_eta = r.atomization.eta_vaporization * 0.97 * 0.99
        assert coupled.eta_cstar == pytest.approx(expected_eta)
        assert coupled.required_mass_flow == pytest.approx(
            1.5e6 * 1.0e-3 / (1700.0 * expected_eta)
        )


# ---- manifold distribution ----------------------------------------------
class TestManifold:
    def test_both_manifolds_present(self):
        r = _size()
        assert r.manifold is not None
        assert set(r.manifold.streams) == {"fuel", "oxidizer"}
        # the slotted stream feeds slots; the other feeds the annulus
        assert r.manifold.streams[r.radial_stream].feeds == "slots"

    def test_manifold_gates_per_stream(self):
        r = _size()
        names = {g.name for g in r.gates}
        assert {"manifold_maldistribution_fuel",
                "manifold_maldistribution_oxidizer"} <= names

    def test_port_count_flows_through(self):
        s = _spec()
        s.fuel_manifold_ports = 5
        r = _size(s)
        assert r.manifold.streams["fuel"].port_count == 5

    def test_manifold_in_to_dict(self):
        d = _size().to_dict()
        assert d["manifold"] is not None
        assert "fuel" in d["manifold"]["streams"]


# ---- face / pintle-tip thermal coupling ---------------------------------
class TestFaceTipThermal:
    def test_thermal_present_and_bounded(self):
        s = _spec()
        s.pintle_material = "GRCop-84"
        r = _size(s)
        t = r.thermal
        assert t is not None
        assert t.recovery_temperature == pytest.approx(0.8 * 3571.0)
        # propellant-cooled wall stays well below the recovery temperature
        assert t.tip_wall_temperature < t.recovery_temperature
        assert t.governing_margin > 0

    def test_thermal_gate_is_real_margin(self):
        s = _spec()
        s.pintle_material = "GRCop-84"
        r = _size(s)
        g = _gate(r, "face_tip_thermal_margin")
        assert g.status in ("pass", "warn")     # no longer info-only
        assert "margin" in g.detail

    def test_thermal_in_to_dict(self):
        d = _size().to_dict()
        assert d["thermal"] is not None
        assert "tip" in d["thermal"] and "face" in d["thermal"]


# ---- stability screen ---------------------------------------------------
class TestStability:
    def test_chug_fails_at_low_dp_fraction(self):
        s = _spec(fuel_dp=0.05, ox_dp=0.05)
        r = _size(s)
        assert _gate(r, "feed_system_chug").status == "fail"
        assert r.stability.injector_decoupling_fraction == pytest.approx(0.05)

    def test_chug_passes_at_high_dp_fraction(self):
        s = _spec(fuel_dp=0.25, ox_dp=0.25)
        r = _size(s)
        assert _gate(r, "feed_system_chug").status == "pass"

    def test_acoustic_modes_ordered(self):
        st = _size().stability
        # tangential/radial transverse modes are higher than the first long.
        assert st.f_R1 > st.f_T1
        assert st.f_L2 == pytest.approx(2 * st.f_L1)

    def test_stability_in_to_dict(self):
        assert _size().to_dict()["stability"]["chug_status"] is not None


# ---- architecture-dispatched throttle map -------------------------------
class TestThrottleMap:
    def _map(self, **kw):
        from raosim.injector import throttle_map
        pc_full = kw.pop("Pc_full", 1.5e6)
        return throttle_map(
            _spec(), mdot_fuel_full=MDOT_F, mdot_oxidizer_full=MDOT_O,
            Pc_full=pc_full, mixture_ratio=MR, chamber_radius=0.0339,
            chamber_length=0.10, gamma=1.24, Tc=3571.0, R_gas=379.6, **kw)

    def test_preserves_of_and_tmr(self):
        tm = self._map(levels=(0.4, 0.7, 1.0))
        assert tm.preserved["mixture_ratio"]
        assert tm.preserved["dp_fraction"]
        assert tm.preserved["total_momentum_ratio"]

    def test_velocity_and_atomization_fall_at_low_throttle(self):
        tm = self._map(levels=(0.3, 1.0), pc_exponent=1.0)
        low, high = tm.points[0], tm.points[-1]
        assert low.throttle < high.throttle
        # deep throttle: lower injection velocity and coarser atomization
        assert low.v_annulus < high.v_annulus
        assert low.smd_limiting > high.smd_limiting
        assert low.sleeve_stroke_fraction < high.sleeve_stroke_fraction
        assert low.actuator_stroke_fraction is None
        assert low.annulus_area_command_fraction == low.sleeve_stroke_fraction

    def test_constant_pc_holds_velocity(self):
        # pc_exponent=0 -> constant Pc, so injection velocity is held
        tm = self._map(levels=(0.4, 1.0), pc_exponent=0.0)
        assert tm.points[0].v_annulus == pytest.approx(
            tm.points[-1].v_annulus, rel=1e-2)

    def test_to_dict_serializable(self):
        import json
        json.dumps(self._map(levels=(0.5, 1.0)).to_dict())

    def test_son_map_holds_hardware_and_solves_separate_axial_controller(self):
        from raosim.injector import throttle_map

        tm = throttle_map(
            _movable_spec(),
            mdot_fuel_full=MDOT_F,
            mdot_oxidizer_full=MDOT_O,
            Pc_full=PC,
            mixture_ratio=MR,
            chamber_radius=0.0339,
            chamber_length=0.10,
            gamma=1.24,
            Tc=3571.0,
            R_gas=379.6,
            levels=(0.2, 0.6, 1.0),
            pc_exponent=1.0,
        )

        assert tm.architecture == "son_continuous_movable"
        assert tm.schedule_semantics == (
            "fixed_hardware_center_pintle_plus_upstream_annulus_controller"
        )
        assert tm.kinematic_model == "son2017_continuous_radial_gap"
        assert tm.preserved["mixture_ratio"]
        assert tm.preserved["requested_mass_flow"]
        assert tm.preserved["fixed_hardware"]
        assert tm.preserved["upstream_controller_schedule"]
        assert not tm.preserved["dp_fraction"]

        gaps = [point.annulus_gap for point in tm.points]
        assert gaps == pytest.approx([gaps[-1]] * len(gaps), rel=1e-12)
        assert all(
            point.annulus_area_command_fraction == pytest.approx(1.0)
            for point in tm.points
        )
        assert all(point.upstream_controller_required for point in tm.points)
        assert all(point.axial_controller_role == "oxidizer" for point in tm.points)
        assert [point.actuator_stroke_fraction for point in tm.points] == sorted(
            point.actuator_stroke_fraction for point in tm.points
        )
        assert [
            point.required_axial_controller_dp_fraction for point in tm.points
        ] == sorted(
            point.required_axial_controller_dp_fraction for point in tm.points
        )
        for point in tm.points:
            assert point.radial_opening == pytest.approx(point.slot_width)
            assert point.mdot_total == pytest.approx(
                point.throttle * MDOT, rel=1e-8
            )
            assert point.mixture_ratio == pytest.approx(MR, rel=1e-8)
        assert tm.points[0].slot_area_command_fraction != pytest.approx(
            tm.points[0].actuator_stroke_fraction
        )

    def test_son_map_fails_when_controller_envelope_cannot_reach_full_power(self):
        from raosim.injector import throttle_map

        spec = _movable_spec()
        spec.movable_axial_controller_dp_fraction_bounds = (0.25, 0.50)
        with pytest.raises(InjectorSpecError, match="full-power oxidizer"):
            throttle_map(
                spec,
                mdot_fuel_full=MDOT_F,
                mdot_oxidizer_full=MDOT_O,
                Pc_full=PC,
                mixture_ratio=MR,
                chamber_radius=0.0339,
                chamber_length=0.10,
                gamma=1.24,
                Tc=3571.0,
                R_gas=379.6,
                levels=(0.2, 1.0),
            )


# ---- figures ------------------------------------------------------------
class TestPlots:
    def test_full_diagnostic_set_renders(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from raosim.injector_plots import export_all_injector_figures
        r = _size()
        written = export_all_injector_figures(r, tmp_path)
        # the eight core diagnostics (no throttle map passed)
        for name in ("cross_section", "spray", "hydraulics", "atomization",
                     "thermal", "stability", "manifold", "gates"):
            f = f"injector_{name}.png"
            assert f in written
            assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0

    def test_throttle_map_figure_added(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from raosim.injector import throttle_map
        from raosim.injector_plots import export_all_injector_figures
        tm = throttle_map(
            _spec(), mdot_fuel_full=MDOT_F, mdot_oxidizer_full=MDOT_O,
            Pc_full=PC, mixture_ratio=MR, chamber_radius=0.0339,
            chamber_length=0.10, gamma=1.24, Tc=3571.0, R_gas=379.6,
            levels=(0.5, 1.0))
        written = export_all_injector_figures(_size(), tmp_path, throttle=tm)
        assert "injector_throttle_map.png" in written


# ---- named-body STEP CAD (CadQuery-gated) -------------------------------
class TestPintleCad:
    def test_named_step_round_trips(self, tmp_path):
        from raosim.injector_cad import (
            export_pintle_step, cadquery_available)
        if not cadquery_available():
            pytest.skip("CadQuery not installed")
        r = _size()
        path = tmp_path / "pintle.step"
        res = export_pintle_step(r, path, movable_sleeve=True,
                                 stl_dir=tmp_path / "stl")
        assert path.exists() and path.stat().st_size > 1000
        # the required named bodies are present
        for body in ("injector_faceplate", "hollow_pintle_body", "pintle_tip",
                     "axial_annulus", "radial_slot_network", "fuel_manifold",
                     "oxidizer_manifold", "igniter_interface",
                     "regen_coolant_outlet", "movable_sleeve"):
            assert body in res["named_bodies"]
        # re-import and confirm valid B-rep solids
        import cadquery as cq
        imp = cq.importers.importStep(str(path))
        solids = imp.objects[0].Solids()
        assert len(solids) >= 9
        assert all(s.isValid() for s in solids)


# ---- serialization ------------------------------------------------------
def test_to_dict_round_trips_json():
    import json
    r = _size()
    d = r.to_dict()
    json.dumps(d)  # must be JSON-serializable
    assert d["feasible"] is True
    assert d["slots"]["role"] == "fuel"
    assert len(d["gates"]) > 10
    assert d["atomization"] is not None
    # feed_system is attached by evaluate_pintle_injector (regen-aware path),
    # not by bare size_pintle_injector, so it is legitimately None here.
    assert d["feed_system"] is None


# ---- feed-system pressure ledger ---------------------------------------
def _eval(spec=None, **kw):
    return evaluate_pintle_injector(
        spec or _spec(),
        mdot_fuel=kw.pop("mdot_fuel", MDOT_F),
        mdot_oxidizer=kw.pop("mdot_oxidizer", MDOT_O),
        Pc=PC,
        mixture_ratio=kw.pop("mixture_ratio", MR),
        chamber_radius=kw.pop("chamber_radius", 0.0339),
        chamber_length=kw.pop("chamber_length", 0.10),
        gamma=1.24, Tc=3571.0, R_gas=379.6,
        fuel_name="RP-1", oxidizer_name="LOX", **kw,
    )


def _gate_status(r, name):
    for g in r.gates:
        if g.name == name:
            return g.status
    return None


class TestFeedSystemLedger:
    def test_attached_and_serializes(self):
        import json
        r = _eval()
        assert r.feed_system is not None
        d = r.to_dict()["feed_system"]
        assert set(d["lines"]) == {"fuel", "oxidizer"}
        json.dumps(d)

    def test_minimal_budget_is_pc_plus_injector_only(self):
        # No pump data and no allowances -> required is just Pc + metering drop.
        r = _eval()
        ln = r.feed_system.lines["fuel"]
        assert ln.injector_dp == pytest.approx(0.2 * PC, rel=1e-6)
        assert ln.manifold_loss == 0.0
        assert ln.line_valve_loss == 0.0
        assert ln.control_margin == 0.0
        assert ln.regen_loss == 0.0
        assert ln.required_outlet_pressure == pytest.approx(
            PC + ln.injector_dp, rel=1e-6)

    def test_manifold_screen_reported_but_not_charged(self):
        # The maldistribution network produces a nonzero estimate that must NOT
        # be charged to the pump budget automatically.
        r = _eval()
        ln = r.feed_system.lines["fuel"]
        assert ln.manifold_screen_loss > 0.0
        assert ln.manifold_loss == 0.0
        assert ln.required_outlet_pressure == pytest.approx(
            ln.chamber_pressure + ln.injector_dp + ln.manifold_loss
            + ln.regen_loss + ln.line_valve_loss + ln.control_margin, rel=1e-9)

    def test_budget_reconciles_with_allowances(self):
        fs = FeedSystemSpec(
            fuel=FeedLineSpec(line_loss_fraction=0.05,
                              control_margin_fraction=0.05,
                              manifold_loss_fraction=0.03),
            oxidizer=FeedLineSpec(line_loss_fraction=0.05,
                                  control_margin_fraction=0.05,
                                  manifold_loss_fraction=0.03))
        r = _eval(_spec(feed_system=fs))
        for ln in r.feed_system.lines.values():
            assert ln.line_valve_loss == pytest.approx(0.05 * PC, rel=1e-6)
            assert ln.manifold_loss == pytest.approx(0.03 * PC, rel=1e-6)
            assert ln.required_outlet_pressure == pytest.approx(
                ln.chamber_pressure + ln.injector_dp + ln.manifold_loss
                + ln.regen_loss + ln.line_valve_loss + ln.control_margin,
                rel=1e-9)

    def test_no_pump_data_is_info(self):
        r = _eval()
        assert _gate_status(r, "feed_pump_pressure_fuel") == "info"
        assert _gate_status(r, "feed_pump_pressure_oxidizer") == "info"
        assert r.feed_system.lines["fuel"].status == "info"

    def test_pump_pressure_gate_pass_and_fail(self):
        ok = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5))))
        assert _gate_status(ok, "feed_pump_pressure_fuel") == "pass"

        bad = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=PC),   # below Pc + dp
            oxidizer=FeedLineSpec(supply_pressure=200e5))))
        assert _gate_status(bad, "feed_pump_pressure_fuel") == "fail"
        assert bad.feasible is False

    def test_pump_head_and_flow(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=4e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=4e5))))
        ln = r.feed_system.lines["fuel"]
        rho = r.feed["fuel"].density
        assert ln.volumetric_flow == pytest.approx(
            r.streams["fuel"].mdot / rho, rel=1e-9)
        rise = ln.required_outlet_pressure - 4e5
        assert ln.required_pressure_rise == pytest.approx(rise, rel=1e-9)
        assert ln.required_pump_head == pytest.approx(
            rise / (rho * 9.80665), rel=1e-9)

    def test_pump_head_is_zero_when_tank_pressure_exceeds_requirement(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=200e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=200e5))))
        ln = r.feed_system.lines["fuel"]
        assert ln.required_pressure_rise == pytest.approx(0.0, abs=1e-12)
        assert ln.required_pump_head == pytest.approx(0.0, abs=1e-12)
        assert ln.ideal_pump_power == pytest.approx(0.0, abs=1e-12)

    def test_capacity_gate_fail(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, flow_capacity=MDOT_F * 0.5),
            oxidizer=FeedLineSpec(supply_pressure=200e5))))
        assert _gate_status(r, "feed_pump_capacity_fuel") == "fail"
        assert r.feed_system.lines["fuel"].capacity_margin < 0.0

    def test_npsh_gate_fail_and_pass(self):
        bad = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=3e5,
                              npsh_required=50e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5))))
        assert _gate_status(bad, "feed_npsh_fuel") == "fail"

        ok = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=10e5,
                              npsh_required=1e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5))))
        assert _gate_status(ok, "feed_npsh_fuel") == "pass"


# ---- electric pump sizing ------------------------------------------------
class TestElectricPumpSizing:
    def test_sizes_drive_battery_and_geometry_from_feed_ledger(self):
        import json
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=5e5,
                              npsh_required=1e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=6e5,
                                  npsh_required=1e5))))
        pump = size_electric_pumps(r.feed_system, PumpSizingSpec(
            burn_time=12.0,
            drive=ElectricDriveSpec(voltage=96.0, rpm=60000.0),
            battery=BatterySpec(voltage=96.0, vehicle_mass=80.0),
            pump_efficiency={"fuel": 0.55, "oxidizer": 0.60},
        ))
        d = pump.to_dict()
        json.dumps(d)
        assert pump.battery.mass > 0.0
        assert pump.lines["fuel"].shaft_power > 0.0
        assert pump.lines["fuel"].efficiency_source == "user"
        assert pump.lines["fuel"].impeller.impeller_diameter > 0.0
        assert pump.lines["fuel"].hydraulic_meanline is not None
        assert pump.lines["fuel"].performance_curve is not None
        assert pump.lines["oxidizer"].inducer.diameter > 0.0
        assert any(g.name == "impeller_tip_speed_fuel"
                   for g in pump.feasibility.gates)

    def test_default_electric_pump_path_solves_rpm_and_efficiency(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=5e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=6e5))))
        pump = size_electric_pumps(r.feed_system)
        fuel = pump.lines["fuel"]
        assert fuel.rpm_source.startswith("auto_")
        assert fuel.efficiency_source.startswith("meanline_loss_model")
        assert fuel.hydraulic_meanline is not None
        assert fuel.efficiency == pytest.approx(
            fuel.hydraulic_meanline.hydraulic_efficiency)
        assert fuel.hydraulic_meanline.velocity_triangle.slip_factor < 1.0
        assert fuel.hydraulic_meanline.losses.total_loss_head > 0.0
        curve = fuel.performance_curve
        assert curve is not None
        design = next(p for p in curve.points if abs(p.flow_ratio - 1.0) < 1e-12)
        assert design.head == pytest.approx(fuel.head)
        assert curve.points[0].head > curve.points[-1].head
        assert fuel.drive.rpm > 0.0
        assert fuel.drive.voltage > 0.0
        assert pump.lines["oxidizer"].drive.voltage == pytest.approx(
            fuel.drive.voltage
        )
        assert pump.battery.voltage == pytest.approx(fuel.drive.voltage)
        assert pump.assumptions["pump_rpm"] == "auto"
        assert pump.assumptions["electric_bus_architecture"] == "shared_pack_bus"
        assert pump.assumptions["selected_bus_voltage_source"].startswith("shared_")
        assert any(
            g.name == "pump_efficiency_screen_fuel" and g.status == "warn"
            for g in pump.feasibility.gates
        )

    def test_pump_exports_architecture_bom_reference_geometry_and_screens(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=5e5,
                              npsh_required=1e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=6e5,
                                  npsh_required=1e5))))
        pump = size_electric_pumps(r.feed_system)
        d = pump.to_dict()
        fuel = d["lines"]["fuel"]

        assert fuel["architecture"]["primary_type"]
        assert "electric_motor_driven" in fuel["architecture"]["candidate_types"]
        assert fuel["reference_geometry"]["editable"] is True
        assert fuel["reference_geometry"]["impeller_disk"]["outer_diameter_m"] > 0.0
        assert fuel["system_curve"]["points"]
        assert fuel["system_curve"]["supported_throttle_range"] is not None
        assert fuel["thermal_stress"]["thermal"]["estimated_propellant_temperature_rise_k"] >= 0.0
        assert fuel["thermal_stress"]["stress"]["impeller_rotating_hoop_stress_pa"] > 0.0
        components = {
            item["component"]
            for item in d["hardware_bom"]
            if item["role"] in {"fuel", "shared"}
        }
        for component in (
            "axial inducer", "centrifugal impeller", "shaft and coupling",
            "motor", "inverter/controller", "bearings",
            "dynamic shaft seals", "pump casing", "battery pack / DC bus",
        ):
            assert component in components
        assert any(g.name == "system_curve_throttle_margin_fuel"
                   for g in pump.feasibility.gates)
        assert any(g.name == "shaft_torsion_fuel"
                   for g in pump.feasibility.gates)
        assert any(g.name == "seal_face_speed_fuel"
                   for g in pump.feasibility.gates)

    def test_shared_bus_flags_inconsistent_explicit_voltages(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=5e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=6e5))))
        pump = size_electric_pumps(r.feed_system, PumpSizingSpec(
            drive=ElectricDriveSpec(voltage=96.0),
            battery=BatterySpec(voltage=270.0),
        ))

        assert pump.feasible is False
        gate = next(g for g in pump.feasibility.gates
                    if g.name == "electric_bus_voltage_consistency")
        assert gate.status == "fail"

    def test_missing_tank_pressure_reports_requirement_without_geometry(self):
        r = _eval()
        pump = size_electric_pumps(r.feed_system)
        assert pump.lines["fuel"].shaft_power is None
        assert pump.lines["fuel"].impeller is None
        gate = next(g for g in pump.feasibility.gates
                    if g.name == "electric_pump_pressure_rise_fuel")
        assert gate.status == "info"
        assert "needs" in gate.detail

    def test_envelope_failures_are_direct_and_actionable(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=2e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=2e5))))
        pump = size_electric_pumps(r.feed_system, PumpSizingSpec(
            drive=ElectricDriveSpec(
                voltage=24.0, rpm=180000.0, max_rpm=60000.0,
                max_motor_power=50.0, max_current=1.0,
            ),
            battery=BatterySpec(voltage=24.0, max_current=1.0),
            material_tip_speed_limit=40.0,
        ))
        assert pump.feasible is False
        assert any(g.status == "fail" for g in pump.feasibility.gates)
        assert pump.feasibility.suggestions[0].startswith(
            "Electric pump feed is infeasible for this Pc"
        )

    def test_pump_thermal_stress_screens_can_fail(self):
        r = _eval(_spec(feed_system=FeedSystemSpec(
            fuel=FeedLineSpec(supply_pressure=200e5, tank_pressure=2e5),
            oxidizer=FeedLineSpec(supply_pressure=200e5, tank_pressure=2e5))))
        pump = size_electric_pumps(r.feed_system, PumpSizingSpec(
            rotor_yield_strength=5.0e6,
            casing_yield_strength=5.0e6,
            bearing_dn_limit=1000.0,
            seal_face_speed_limit=0.5,
            max_propellant_temperature_rise=0.01,
        ))

        assert pump.feasible is False
        failed = {g.name for g in pump.feasibility.gates if g.status == "fail"}
        assert "impeller_rotating_stress_fuel" in failed
        assert "bearing_dn_fuel" in failed
        assert "seal_face_speed_fuel" in failed
        assert "propellant_heating_fuel" in failed
