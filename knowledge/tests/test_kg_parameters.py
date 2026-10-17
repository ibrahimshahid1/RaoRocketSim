"""Repository defaults stay distinct, inspectable and free of source support."""

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import harvest_parameters as parameters, harvest_survey, store
from kg.schema import node_id


def _fixture(root, *, mdo_density="1300.0"):
    pumps = root / "raosim/pumps.py"
    pumps.parent.mkdir(parents=True, exist_ok=True)
    pumps.write_text('''SCREENING_DEFAULTS = {
    "motor_efficiency": 0.90,
    "battery_energy_density": 250.0 * 3600.0,
    "battery_power_density": 3000.0,
}
class ElectricDriveSpec:
    motor_efficiency: float = SCREENING_DEFAULTS["motor_efficiency"]
def _drive_sizing(drive):
    return drive.motor_efficiency
''')
    schema = root / "raosim/mdo/schema.py"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(f'''class MissionSpec:
    eta_motor: float = 0.90
    battery_energy_density: float = 6.5e5
    battery_power_density: float = {mdo_density}
    pump_specific_mass: float = 2500.0  # W/kg hydraulic (screening)
''')
    consumer = root / "raosim/mdo/pump.py"
    consumer.write_text('''def electric_feed(mission, P_shaft):
    return P_shaft / mission.pump_specific_mass, mission.battery_power_density
''')
    return pumps, schema


def _sandbox(root, monkeypatch):
    monkeypatch.setattr(store, "KNOWLEDGE_ROOT", root / "knowledge")
    store.save("software_links", [
        {"id": "CURATED", "type": "EmpiricalConstant", "value": "independent", "basis_class": "UNKNOWN_BASIS"},
        {"id": node_id("SW", "raosim/mdo/schema.py:MissionSpec"), "type": "SoftwareComponent"},
        {"id": node_id("SW", "raosim/mdo/pump.py:electric_feed"), "type": "SoftwareComponent"},
    ])
    store.save("models", [{"id": parameters.MODEL_ID, "registry_id": "pump.electric_drive"}])
    store.save("provenance", [{"id": "PROV_CURATED", "kind": "SOURCE_STATES"}])


def test_numeric_ast_definitions_keep_both_sets_and_exact_locators(tmp_path):
    pumps, schema = _fixture(tmp_path)
    result = parameters.collect(repo_root=tmp_path)
    assert result == parameters.collect(repo_root=tmp_path)
    records = {(r["default_set"], r["name"]): r for r in result["records"]}
    assert records["traditional_screening", "battery_energy_density"]["value"] == 900000.0
    assert records["mdo_mission", "battery_energy_density"]["value"] == 650000.0
    assert records["traditional_screening", "motor_efficiency"]["id"] != records["mdo_mission", "eta_motor"]["id"]
    for row in result["records"]:
        source = tmp_path / row["file"]
        assert row["code_quote"] == "\n".join(source.read_text().splitlines()[row["line"] - 1:row["end_line"]])
        assert row["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert row["basis_class"] == "UNKNOWN_BASIS" and row["active"]
        assert row["source_document_id"] is None and row["hardware_qualified"] is False
    power = next(c for c in result["report"]["comparisons"] if c["traditional_name"] == "battery_power_density")
    assert power["state"] == "DIFFERENT_ASSUMPTIONS"
    assert (power["traditional_value"], power["mdo_value"]) == (3000.0, 1300.0)
    assert any(i["state"] == "DECLARATION_CONSUMER_DISCREPANCY" for i in result["report"]["issues"])
    refs = records["traditional_screening", "motor_efficiency"]["consumers"]
    assert any(r["qualname"] == "_drive_sizing" and r["reference"] == "drive.motor_efficiency" for r in refs)
    assert all(p["kind"] == "CODE_STATES" for p in result["provenance"])


def test_expressions_never_execute_and_nonfinite_values_are_unavailable(tmp_path):
    _, schema = _fixture(tmp_path)
    marker = tmp_path / "executed"
    schema.write_text(f'''class MissionSpec:
    eta_motor: float = True
    battery_power_density: float = __import__("pathlib").Path({str(marker)!r}).touch()
    battery_energy_density: float = 1e999
''')
    result = parameters.collect(repo_root=tmp_path)
    assert not marker.exists()
    assert not any(r["default_set"] == "mdo_mission" for r in result["records"])
    assert len([i for i in result["report"]["issues"] if i["state"] == "UNSUPPORTED_DEFAULT_EXPRESSION"]) == 3
    assert all(c["state"] == "UNAVAILABLE" for c in result["report"]["comparisons"])


def test_merge_is_idempotent_preserves_surveys_and_invalidates_changes(tmp_path, monkeypatch):
    _, schema = _fixture(tmp_path)
    _sandbox(tmp_path, monkeypatch)
    parameters.harvest(repo_root=tmp_path)
    original = {r["id"]: r for r in store.load("software_links")}
    first_bytes = store.store_path("software_links").read_bytes()
    prov_bytes = store.store_path("provenance").read_bytes()
    parameters.harvest(repo_root=tmp_path)
    assert store.store_path("software_links").read_bytes() == first_bytes
    assert store.store_path("provenance").read_bytes() == prov_bytes
    old = next(r for r in original.values() if r.get("logical_key") == "mdo_mission:battery_power_density")
    schema.write_text(schema.read_text().replace("1300.0", "1700.0"))
    parameters.harvest(repo_root=tmp_path)
    rows = {r["id"]: r for r in store.load("software_links")}
    assert rows["CURATED"] == original["CURATED"]
    assert rows[old["id"]]["active"] is False
    assert rows[old["id"]]["source_state"] == "STALE_DEFINITION"
    current = [r for r in rows.values() if r.get("logical_key") == old["logical_key"] and r["active"]]
    assert len(current) == 1 and current[0]["value"] == 1700.0
    assert all(not r["active"] for r in rows.values() if r.get("from") == old["id"] or r.get("to") == old["id"])
    assert any(p["id"] == old["provenance_id"] for p in store.load("provenance"))
    schema.unlink()
    parameters.harvest(repo_root=tmp_path)
    assert all(not r["active"] for r in store.load("software_links") if r.get("default_set") == "mdo_mission")
    assert any(p["id"] == "PROV_CURATED" for p in store.load("provenance"))


def test_line_shift_refreshes_locator_without_duplicating_definition(tmp_path, monkeypatch):
    _, schema = _fixture(tmp_path)
    _sandbox(tmp_path, monkeypatch)
    parameters.harvest(repo_root=tmp_path)
    old = next(r for r in store.load("software_links") if r.get("logical_key") == "mdo_mission:eta_motor")
    schema.write_text("# unrelated leading comment\n" + schema.read_text())
    parameters.harvest(repo_root=tmp_path)
    rows = [r for r in store.load("software_links") if r.get("logical_key") == old["logical_key"]]
    assert len(rows) == 1 and rows[0]["id"] == old["id"]
    assert rows[0]["line"] == old["line"] + 1
    assert rows[0]["source_sha256"] != old["source_sha256"]
    assert rows[0]["provenance_id"] != old["provenance_id"]


def test_survey_refresh_preserves_parameter_owned_history(tmp_path, monkeypatch):
    _fixture(tmp_path)
    _sandbox(tmp_path, monkeypatch)
    parameters.harvest(repo_root=tmp_path)
    owned = {r["id"]: r for r in store.load("software_links") if r.get("harvest_owner") == parameters.HARVESTER}
    harvest_survey.harvest([], set())
    after = {r["id"]: r for r in store.load("software_links")}
    assert all(after[rid] == row for rid, row in owned.items())


def test_definition_symlink_outside_repository_is_not_read(tmp_path):
    root = tmp_path / "repo"
    _, schema = _fixture(root)
    external = tmp_path / "external.py"
    external.write_text("class MissionSpec:\n    eta_motor: float = 0.123456789\n")
    schema.unlink()
    schema.symlink_to(external)
    result = parameters.collect(repo_root=root)
    assert not any(r["default_set"] == "mdo_mission" for r in result["records"])
    assert any(i.get("reason") == "OUTSIDE_REPOSITORY" for i in result["report"]["issues"])


def test_registry_references_resolve_to_one_shared_set_without_execution(tmp_path):
    """Both default sets may read their electric values from the technology
    registry; the AST harvester resolves ``alias.field`` to the registry
    literal, records the registry set itself, and still creates no support."""
    _fixture(tmp_path)
    marker = tmp_path / "executed"
    registry = tmp_path / "raosim/electric_technology.py"
    registry.write_text(f'''SCREEN = ElectricTechnologySet(
    id="screen_v1",
    basis_class="repository_screening_assumption",
    motor_efficiency=0.90,
    battery_energy_density=250.0 * 3600.0,
    battery_power_density=3000.0,
    side_effect=__import__("pathlib").Path({str(marker)!r}).touch(),
)
PUBLISHED = ElectricTechnologySet(
    id="paper_set",
    basis_class="literature",
    motor_efficiency=0.87,
    battery_energy_density=325.0 * 3600.0,
    battery_power_density=650.0,
    sources=(TechnologySource("paper", "propulsion_texts/paper.pdf", 11, "Table 2"),),
)
DEFAULT = SCREEN
''')
    (tmp_path / "raosim/pumps.py").write_text('''from raosim.electric_technology import DEFAULT
SCREENING_DEFAULTS = {
    "motor_efficiency": DEFAULT.motor_efficiency,
    "battery_energy_density": DEFAULT.battery_energy_density,
    "battery_power_density": (DEFAULT.battery_power_density),
}
''')
    (tmp_path / "raosim/mdo/schema.py").write_text('''from raosim.electric_technology import DEFAULT as _E
class MissionSpec:
    eta_motor: float = _E.motor_efficiency
    battery_energy_density: float = _E.battery_energy_density
    battery_power_density: float = _E.battery_power_density
''')
    result = parameters.collect(repo_root=tmp_path)
    assert not marker.exists()
    records = {(r["default_set"], r["name"]): r for r in result["records"]}
    trad = records["traditional_screening", "battery_power_density"]
    mdo = records["mdo_mission", "battery_power_density"]
    assert trad["value"] == mdo["value"] == 3000.0
    assert trad["derived_from"]["technology_id"] == "screen_v1"
    assert mdo["derived_from"]["technology_id"] == "screen_v1"
    assert records["electric_technology:paper_set", "battery_power_density"]["value"] == 650.0
    published = records["electric_technology:paper_set", "motor_efficiency"]
    assert published["declared_basis_class"] == "literature"
    assert published["declared_source_pages"] == [11]
    assert published["basis_class"] == "UNKNOWN_BASIS"
    assert result["report"]["scientific_support_created"] is False
    for comparison in result["report"]["comparisons"]:
        if comparison["state"] != "UNAVAILABLE":
            assert comparison["state"] == "EQUAL"
            assert comparison["shared_registry_set"] == "screen_v1"
    assert not any(i["state"] == "UNSUPPORTED_DEFAULT_EXPRESSION"
                   for i in result["report"]["issues"])
