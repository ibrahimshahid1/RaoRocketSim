"""Tests for raosim.injector — pintle hydraulic sizing and gates."""

import math

import pytest

from raosim.injector import (
    FeedLineSpec,
    FeedSystemSpec,
    InjectorManufacturingSpec,
    InjectorSpec,
    InjectorUnsupportedState,
    PintleGeometrySpec,
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
    return size_pintle_injector(
        spec or _spec(), mdot_fuel=kw.pop("mdot_fuel", MDOT_F),
        mdot_oxidizer=kw.pop("mdot_oxidizer", MDOT_O), Pc=PC,
        mixture_ratio=kw.pop("mixture_ratio", MR),
        chamber_radius=kw.pop("chamber_radius", 0.0339),
        chamber_length=kw.pop("chamber_length", 0.10),
        gamma=1.24, Tc=3571.0, R_gas=379.6, **kw,
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
    def test_target_momentum_ratio_gates(self):
        s = _spec()
        s.target_momentum_ratio = 5.0  # far from the ~0.5 achieved
        r = _size(s)
        g = _gate(r, "target_momentum_ratio")
        assert g.status == "fail"

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
        assert ln.regen_loss == pytest.approx(2.0e6, rel=1e-12)
        assert ln.required_outlet_pressure == pytest.approx(
            ln.chamber_pressure + ln.injector_dp + ln.regen_loss, rel=1e-12)


# ---- spray atomization / vaporization screen ----------------------------
class TestAtomization:
    def test_smd_in_micron_range(self):
        r = _size()
        at = r.atomization
        assert at is not None
        for s in at.streams.values():
            # rocket SMD is tens of microns; never larger than the jet
            assert 1e-6 < s.sauter_mean_diameter < 5e-4
            assert s.aerodynamic_weber > 0

    def test_chamber_gas_density(self):
        r = _size()
        # rho_g = Pc/(R Tc) = 7e6/(379.6*3571)
        assert r.atomization.chamber_gas_density == pytest.approx(
            7e6 / (379.6 * 3571.0), rel=1e-6)

    def test_short_chamber_warns_and_drops_efficiency(self):
        """A too-short chamber cannot develop combustion: warn + η_c* < 1."""
        short = _size(chamber_length=0.01)
        long = _size(chamber_length=0.5)
        assert (short.atomization.predicted_cstar_efficiency
                <= long.atomization.predicted_cstar_efficiency)
        assert short.atomization.development_margin < long.atomization.development_margin
        g = _gate(short, "combustion_development_length")
        assert g.status in ("warn", "pass")  # surrogate never hard-fails
        # the surrogate must not by itself make the design infeasible
        assert g.status != "fail"

    def test_efficiency_bounded(self):
        r = _size()
        assert 0.0 <= r.atomization.predicted_cstar_efficiency <= 1.0

    def test_atomization_in_to_dict(self):
        d = _size().to_dict()
        assert d["atomization"]["model"].startswith("hinze")
        assert "fuel" in d["atomization"]["streams"]


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


# ---- movable-sleeve throttle map ----------------------------------------
class TestThrottleMap:
    def _map(self, **kw):
        from raosim.injector import throttle_map
        return throttle_map(
            _spec(), mdot_fuel_full=MDOT_F, mdot_oxidizer_full=MDOT_O,
            Pc_full=PC, mixture_ratio=MR, chamber_radius=0.0339,
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

    def test_constant_pc_holds_velocity(self):
        # pc_exponent=0 -> constant Pc, so injection velocity is held
        tm = self._map(levels=(0.4, 1.0), pc_exponent=0.0)
        assert tm.points[0].v_annulus == pytest.approx(
            tm.points[-1].v_annulus, rel=1e-2)

    def test_to_dict_serializable(self):
        import json
        json.dumps(self._map(levels=(0.5, 1.0)).to_dict())


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
