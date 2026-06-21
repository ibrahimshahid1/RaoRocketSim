"""Tests for raosim.injector — pintle hydraulic sizing and gates."""

import math

import pytest

from raosim.injector import (
    InjectorManufacturingSpec,
    InjectorSpec,
    InjectorUnsupportedState,
    PintleGeometrySpec,
    PropellantFeedSpec,
    resolve_feed_state,
    size_pintle_injector,
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

    def test_acoustic_screen_reports(self):
        r = _size()
        assert _gate(r, "acoustic_screen").status == "info"


# ---- liquid-only guard --------------------------------------------------
class TestLiquidOnlyGuard:
    def test_rejects_gas_feed(self):
        spec = _spec()
        spec.fuel.phase = "gas"
        with pytest.raises(InjectorUnsupportedState):
            _size(spec)

    def test_rejects_subcritical_low_pressure_flashing(self):
        # N2O4 is volatile; a near-vapor-pressure feed must be rejected.
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


# ---- serialization ------------------------------------------------------
def test_to_dict_round_trips_json():
    import json
    r = _size()
    d = r.to_dict()
    json.dumps(d)  # must be JSON-serializable
    assert d["feasible"] is True
    assert d["slots"]["role"] == "fuel"
    assert len(d["gates"]) > 10
