import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg.quantities import resolve_symbol
from kg import store, variable_links


def test_gravity_does_not_become_gamma():
    assert resolve_symbol("g")["state"] == "AMBIGUOUS_SYMBOL"
    assert resolve_symbol("g", units="m/s²")["candidates"][0]["qid"] == "QTY_GRAVITY"
    assert resolve_symbol("g", units="1")["candidates"][0]["qid"] == "QTY_GAMMA"
    assert resolve_symbol("k", units="W/m/K")["candidates"][0]["qid"] == "QTY_KCOND"
    assert resolve_symbol("g", units="unknown")["state"] == "UNRESOLVED_ENTITY"


def test_equal_units_do_not_merge_thrust_friction_and_drag_discharge():
    assert resolve_symbol("c_f", units="1")["state"] == "AMBIGUOUS_SYMBOL"
    assert resolve_symbol("c_f", units="1", meaning="skin friction coefficient")["candidates"][0]["qid"] == "QTY_CF_SKIN"
    assert resolve_symbol("C_F", units="1", meaning="thrust coefficient")["candidates"][0]["qid"] == "QTY_CF"
    assert resolve_symbol("C_D", units="1")["state"] == "AMBIGUOUS_SYMBOL"
    assert resolve_symbol("C_D", units="1", meaning="drag coefficient")["candidates"][0]["qid"] == "QTY_CD_DRAG"


def test_conflicting_meaning_and_units_refuse_resolution():
    assert resolve_symbol("g", units="1", meaning="gravitational acceleration")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("k", units="1", meaning="thermal conductivity")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("g", units="m/s^2", meaning="specific-heat ratio")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("h", units="m", meaning="inlet outer-streamline height")["candidates"][0]["qid"] == "QTY_STREAMLINE_HEIGHT"
    assert resolve_symbol("h", units="m", meaning="pump head rise")["candidates"][0]["qid"] == "QTY_HEAD"
    assert resolve_symbol("T_0", units="K", meaning="stagnation temperature")["candidates"][0]["qid"] == "QTY_T0"
    assert resolve_symbol("T", units="K", meaning="local static temperature")["candidates"][0]["qid"] == "QTY_T_STATIC"
    assert resolve_symbol("A", units="m^2", meaning="local duct area")["candidates"][0]["qid"] == "QTY_LOCAL_FLOW_AREA"
    assert resolve_symbol("A*", units="m^2")["state"] == "AMBIGUOUS_SYMBOL"
    assert resolve_symbol("A*", units="m^2", meaning="sonic reference area")["candidates"][0]["qid"] == "QTY_SONIC_REFERENCE_AREA"
    assert resolve_symbol("A_star", units="m^2", meaning="sonic throat area")["candidates"][0]["qid"] == "QTY_AT"
    assert resolve_symbol("p_0", "performance", units="Pa", meaning="local isentropic stagnation pressure")["candidates"][0]["qid"] == "QTY_P0"


def test_specific_thermodynamic_quantities_keep_overloaded_glyphs_distinct():
    assert resolve_symbol("h")["state"] == "AMBIGUOUS_SYMBOL"
    for symbol, meaning in (("h", "specific enthalpy"), ("h_i", "species specific enthalpy"),
                            ("h_0", "stagnation specific enthalpy"), ("h_2", "local specific enthalpy")):
        result = resolve_symbol(symbol, units="J/kg", meaning=meaning)
        assert result["state"] == "RESOLVED"
        assert result["candidates"][0]["qid"] == "QTY_H_SPECIFIC"
        assert result["candidates"][0]["dimension"] == "L^2 T^-2"
    assert resolve_symbol("h", units="m", meaning="inlet outer-streamline height")["candidates"][0]["qid"] == "QTY_STREAMLINE_HEIGHT"
    assert resolve_symbol("h", units="W/m^2/K", meaning="gas-side heat-transfer coefficient")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("h_i", units="J/kg", meaning="specific internal energy")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("e_i", units="J/kg", meaning="species specific internal energy")["candidates"][0]["qid"] == "QTY_E_SPECIFIC"
    assert resolve_symbol("e", units="J/kg", meaning="specific enthalpy")["state"] == "UNRESOLVED_ENTITY"


def test_specific_heat_and_composition_meanings_do_not_merge_equal_units():
    for symbol in ("c_v", "cv", "c_vf", "c_vi"):
        result = resolve_symbol(symbol, units="J/kg/K", meaning="specific heat at constant volume")
        assert result["state"] == "RESOLVED"
        assert result["candidates"][0]["qid"] == "QTY_CV"
    for symbol, meaning in (("c_p", "mixture specific heat at constant pressure"),
                            ("cp_f", "frozen specific heat at constant pressure"),
                            ("c_pi", "species specific heat at constant pressure")):
        result = resolve_symbol(symbol, units="J/kg/K", meaning=meaning)
        assert result["state"] == "RESOLVED"
        assert result["candidates"][0]["qid"] == "QTY_CP"
    assert resolve_symbol("c_v", units="J/kg/K", meaning="specific heat at constant pressure")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("c_p", units="J/kg/K", meaning="specific heat at constant volume")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("c_i", units="1", meaning="species mass fraction")["candidates"][0]["qid"] == "QTY_MASS_FRACTION"
    assert resolve_symbol("c_i", units="m/s", meaning="speed of sound")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("c", units="1", meaning="species mass fraction")["state"] == "UNRESOLVED_ENTITY"


def test_local_speed_and_specific_volume_are_separated_by_units_and_context():
    assert resolve_symbol("V")["state"] == "AMBIGUOUS_SYMBOL"
    assert resolve_symbol("v", units="m^3/kg", meaning="specific volume")["candidates"][0]["qid"] == "QTY_SPECIFIC_VOLUME"
    for symbol, meaning in (("u", "local flow velocity"), ("V", "flow velocity")):
        assert resolve_symbol(symbol, units="m/s", meaning=meaning)["candidates"][0]["qid"] == "QTY_FLOW_SPEED"
    assert resolve_symbol("u", units="m/s", meaning="friction velocity")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("u_star", units="m/s", meaning="friction velocity")["candidates"][0]["qid"] == "QTY_FRICTION_VELOCITY"
    assert resolve_symbol("V", units="m/s", meaning="specific volume")["state"] == "UNRESOLVED_ENTITY"


def test_sonic_and_friction_speed_require_source_meaning_despite_equal_units():
    assert resolve_symbol("u_star", units="m/s")["state"] == "AMBIGUOUS_SYMBOL"
    for meaning in ("sonic flow speed", "sonic critical velocity"):
        result = resolve_symbol("u_star", units="m/s", meaning=meaning)
        assert result["state"] == "RESOLVED"
        assert result["candidates"][0]["qid"] == "QTY_FLOW_SPEED"
    assert resolve_symbol("u_star", units="m/s", meaning="friction velocity")["candidates"][0]["qid"] == "QTY_FRICTION_VELOCITY"
    for symbol, meaning in (("a_f", "frozen speed of sound"), ("a_e", "equilibrium speed of sound"),
                            ("a", "local speed of sound")):
        result = resolve_symbol(symbol, units="m/s", meaning=meaning)
        assert result["state"] == "RESOLVED"
        assert result["candidates"][0]["qid"] == "QTY_SOUND"
    assert resolve_symbol("a_f", units="m/s", meaning="sonic flow speed")["state"] == "UNRESOLVED_ENTITY"
    assert resolve_symbol("u_star", units="m/s", meaning="speed of sound")["state"] == "UNRESOLVED_ENTITY"


def test_reconcile_retains_source_and_history_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", tmp_path)
    store.save("equations", [{"id": "EQ_T", "form": "N_s = omega sqrt(Q)/(gH)^(3/4)",
        "quote": "Source anchor", "variables": [{"symbol": "g", "meaning": "gravity", "units": "m/s^2",
            "resolution_state": "RESOLVED", "quantity_candidates": ["QTY_GAMMA"]}]}])
    assert variable_links.reconcile()["changed_variable_links"] == 1
    record = store.load("equations")[0]
    assert record["form"] == "N_s = omega sqrt(Q)/(gH)^(3/4)"
    assert record["quote"] == "Source anchor"
    assert record["variables"][0]["quantity_candidates"] == ["QTY_GRAVITY"]
    assert record["variables"][0]["normalization_history"][0]["quantity_candidates"] == ["QTY_GAMMA"]
    assert variable_links.reconcile()["changed_variable_links"] == 0
