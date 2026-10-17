"""Observe electric-feed defaults from Python AST, without importing models.

These are separate repository technology assumptions, not literature constants.
Exact definitions, static references and history are retained; disagreement does
not choose a preferred default set or create scientific source support.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
from pathlib import Path

from . import REPO_ROOT
from .schema import node_id
from .store import load, load_meta, save, store_path

HARVESTER = "kg.harvest_parameters@1"
MODEL_ID = node_id("MODEL", "pump.electric_drive")
DEFAULT_SETS = {
    "traditional_screening": ("raosim/pumps.py", "SCREENING_DEFAULTS"),
    "mdo_mission": ("raosim/mdo/schema.py", "MissionSpec"),
}
# Both default sets above take their electric-drive values from one registry
# of ``ElectricTechnologySet(...)`` literals.  The registry is read by AST too:
# a field written as ``<alias>.<field>`` is resolved to the literal in the
# registry call it names, never by importing or executing anything.
REGISTRY_FILE = "raosim/electric_technology.py"
REGISTRY_MODULE = "raosim.electric_technology"
REGISTRY_CLASS = "ElectricTechnologySet"
CONSUMER_FILES = (
    "raosim/electric_technology.py",
    "raosim/pumps.py", "raosim/run_nozzle.py", "raosim/mdo/schema.py",
    "raosim/mdo/pump.py", "raosim/mdo/assembly.py", "raosim/mdo/state.py",
    "raosim/mdo/engine.py", "raosim/mdo/postprocess.py", "raosim/mdo/snapshot.py",
)
# Repository field semantics, not source-document assertions. Values are never
# stored here: only the actual AST definition supplies a numeric default.
SEMANTICS = {
    "pump_margin": ("1", "Minimum pump flow-capacity fractional margin"),
    "motor_efficiency": ("1", "Motor shaft output / electrical input"),
    "inverter_efficiency": ("1", "Inverter AC output / DC input"),
    "motor_power_density": ("W/kg", "Motor shaft-output power per motor mass"),
    "inverter_power_density": ("W/kg", "Inverter AC-output power per inverter mass"),
    "battery_energy_density": ("J/kg", "Battery energy per pack mass before repository margin"),
    "battery_power_density": ("W/kg", "Battery DC output power per pack mass before repository margin"),
    "battery_discharge_efficiency": ("1", "Battery discharge efficiency in energy sizing"),
    "battery_structural_margin": ("1", "Battery mass structural/package multiplier"),
    "auto_current_target": ("A", "Current target used in automatic DC bus selection"),
    "rotor_material_density": ("kg/m^3", "Screening rotor material density"),
    "rotor_yield_strength": ("Pa", "Screening rotor yield strength"),
    "casing_yield_strength": ("Pa", "Screening casing yield strength"),
    "structural_fos": ("1", "Pump structural screening factor of safety"),
    "bearing_dn_limit": ("mm rpm", "Bearing diameter-speed product screening limit"),
    "seal_face_speed_limit": ("m/s", "Seal face-speed screening limit"),
    "eta_pump": ("1", "Fixed hydraulic-to-shaft efficiency in skeleton assembly"),
    "eta_motor": ("1", "Motor shaft output / electrical input"),
    "eta_inverter": ("1", "Inverter AC output / DC input"),
    "eta_discharge": ("1", "Battery discharge efficiency in energy sizing"),
    "pump_specific_mass": ("W/kg", "Pump power-per-mass denominator; basis requires checking declaration and consumers"),
    "pump_eta_peak": ("1", "Peak efficiency of repository smooth pump surrogate"),
    "pump_ns_opt": ("1", "Dimensionless specific speed at surrogate efficiency peak"),
    "pump_eta_ns_width": ("1", "Log-specific-speed width of repository efficiency surrogate"),
    "pump_speed_rpm": ("rpm", "Screening pump rotational speed"),
    "pump_head_coefficient": ("1", "Pump head coefficient g0 H / U^2"),
    "pump_tip_speed_max": ("m/s", "Pump tip-speed screening limit"),
    "pump_nss_max": ("1", "Dimensionless suction-specific-speed screening limit"),
}
PAIRS = {
    "motor_efficiency": "eta_motor", "inverter_efficiency": "eta_inverter",
    "battery_discharge_efficiency": "eta_discharge",
    **{key: key for key in ("motor_power_density", "inverter_power_density",
                           "battery_energy_density", "battery_power_density",
                           "battery_structural_margin")},
}
DRIVE_FIELDS = set(PAIRS) | set(PAIRS.values()) | {"auto_current_target", "eta_pump", "pump_specific_mass"}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _number(node: ast.AST):
    """Evaluate finite numeric literals and basic arithmetic only; never eval."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
        operation = {ast.Add: operator.add, ast.Sub: operator.sub,
                     ast.Mult: operator.mul, ast.Div: operator.truediv}[type(node.op)]
        value = operation(_number(node.left), _number(node.right))
    else:
        raise ValueError("not a literal numeric arithmetic expression")
    if not math.isfinite(value):
        raise ValueError("nonfinite default")
    return value


def _registry(tree: ast.Module) -> dict:
    """Module-level ``NAME = ElectricTechnologySet(...)`` literals by NAME.

    ``ALIAS = NAME`` assignments in the registry module resolve to the same
    entry.  Only literal keywords are read; nothing is executed.
    """
    entries: dict[str, dict] = {}
    for statement in tree.body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            continue
        target, value = statement.targets[0].id, statement.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == REGISTRY_CLASS):
            entry = {"variable": target, "values": {}, "nodes": {}, "id": None,
                     "basis_class": None, "line": statement.lineno,
                     "end_line": statement.end_lineno, "sources": []}
            for keyword in value.keywords:
                if keyword.arg in ("id", "basis_class") and isinstance(keyword.value, ast.Constant):
                    entry[keyword.arg] = keyword.value.value
                elif keyword.arg in SEMANTICS:
                    try:
                        entry["values"][keyword.arg] = _number(keyword.value)
                        entry["nodes"][keyword.arg] = keyword
                    except (ValueError, ArithmeticError, OverflowError):
                        pass
                elif keyword.arg == "sources" and isinstance(keyword.value, (ast.Tuple, ast.List)):
                    for call in keyword.value.elts:
                        if isinstance(call, ast.Call) and len(call.args) >= 3:
                            page = call.args[2]
                            entry["sources"].append({
                                "pdf_page": page.value if isinstance(page, ast.Constant) else None,
                                "statement_line": call.lineno,
                            })
            if entry["id"]:
                entries[target] = entry
        elif isinstance(value, ast.Name) and value.id in entries:
            entries[target] = entries[value.id]
    return entries


def _registry_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> registry variable, from ``from raosim.electric_technology import``."""
    aliases = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == REGISTRY_MODULE:
            for alias in statement.names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _resolve(expression: ast.AST, aliases: dict[str, str], registry: dict):
    """(value, derived_from) for a literal or a registry field reference."""
    try:
        return _number(expression), None
    except ValueError:
        if (isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name)
                and expression.value.id in aliases):
            entry = registry.get(aliases[expression.value.id])
            if entry and expression.attr in entry["values"]:
                node = entry["nodes"][expression.attr]
                return entry["values"][expression.attr], {
                    "file": REGISTRY_FILE, "technology_id": entry["id"],
                    "field": expression.attr, "line": node.lineno,
                    "declared_basis_class": entry["basis_class"]}
        raise


def _quote(source: str, node: ast.AST) -> str:
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def _definitions(tree: ast.Module, container: str):
    for statement in tree.body:
        if container == "SCREENING_DEFAULTS" and isinstance(statement, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == container for t in statement.targets):
                if isinstance(statement.value, ast.Dict):
                    for key, value in zip(statement.value.keys, statement.value.values):
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            yield key.value, value, key
        elif isinstance(statement, ast.ClassDef) and statement.name == container:
            for field in statement.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.value:
                    if field.target.id in SEMANTICS:
                        yield field.target.id, field.value, field


def _references(tree: ast.Module, source: str, path: str):
    """Static syntactic references only; these do not establish runtime reach."""
    found = []
    # Follow literal dictionary defaults into their dataclass fields. Receiver
    # names are explicit repository conventions, not arbitrary attribute matches.
    aliases = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ("ElectricDriveSpec", "BatterySpec", "PumpSizingSpec"):
            prefixes = {"ElectricDriveSpec": ("drive", "spec.drive"),
                        "BatterySpec": ("batt", "spec.battery"), "PumpSizingSpec": ("spec",)}[cls.name]
            for field in cls.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name):
                    value = field.value
                    if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
                            and value.value.id == "SCREENING_DEFAULTS" and isinstance(value.slice, ast.Constant)):
                        for prefix in prefixes:
                            aliases[prefix + "." + field.target.id] = value.slice.value

    def visit(node, scope=()):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
        default_set = name = None
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id in ("SCREENING_DEFAULTS", "PUMP_DEFAULTS")
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            default_set, name = "traditional_screening", node.slice.value
        elif isinstance(node, ast.Attribute):
            reference = ast.unparse(node)
            if isinstance(node.value, ast.Name) and node.value.id == "mission":
                default_set, name = "mdo_mission", node.attr
            elif reference in aliases:
                default_set, name = "traditional_screening", aliases[reference]
        if name in SEMANTICS:
            found.append({"default_set": default_set, "name": name, "file": path,
                          "line": node.lineno, "column": node.col_offset,
                          "qualname": ".".join(scope) or "<module>",
                          "component_id": node_id("SW", f"{path}:{scope[-1]}") if scope else None,
                          "reference": ast.get_source_segment(source, node), "code_quote": _quote(source, node),
                          "verification": "static_named_reference_only_not_runtime_reach"})
        for child in ast.iter_child_nodes(node):
            visit(child, scope)
    visit(tree)
    return found


def collect(*, repo_root: Path = REPO_ROOT) -> dict:
    """Return deterministic current defaults, provenance and comparisons."""
    root = Path(repo_root).resolve()
    parsed, sources, issues = {}, {}, []
    for path in sorted(set(CONSUMER_FILES) | {p for p, _ in DEFAULT_SETS.values()}):
        file = root / path
        try:
            if not file.resolve().is_relative_to(root):
                raise ValueError("OUTSIDE_REPOSITORY")
            raw = file.read_bytes()
            source = raw.decode("utf-8")
            parsed[path] = source, ast.parse(source, filename=path)
            sources[path] = {"path": path, "state": "CURRENT", "sha256": hashlib.sha256(raw).hexdigest()}
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            sources[path] = {"path": path, "state": "SOURCE_UNAVAILABLE", "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
    references = [ref for path, (source, tree) in parsed.items() for ref in _references(tree, source, path)]
    registry = _registry(parsed[REGISTRY_FILE][1]) if REGISTRY_FILE in parsed else {}
    records, provenance = [], []
    # Each registered technology set is its own observed default set.
    set_definitions = {}
    for entry in {id(e): e for e in registry.values()}.values():
        set_name = "electric_technology:" + entry["id"]
        set_definitions[set_name] = (REGISTRY_FILE, entry["variable"], [
            (name, entry["nodes"][name].value, entry["nodes"][name])
            for name in entry["values"]])
    for default_set, (path, container) in [*DEFAULT_SETS.items(), *(
            (name, (path, container)) for name, (path, container, _) in sorted(set_definitions.items()))]:
        if path not in parsed:
            issues.append({"default_set": default_set, **sources[path]})
            continue
        source, tree = parsed[path]
        aliases = _registry_aliases(tree)
        definitions = (set_definitions[default_set][2] if default_set in set_definitions
                       else list(_definitions(tree, container)))
        if not definitions:
            issues.append({"default_set": default_set, "path": path, "state": "CONTAINER_NOT_FOUND"})
        for name, expression, locator in definitions:
            if name not in SEMANTICS:
                issues.append({"default_set": default_set, "path": path, "name": name, "state": "UNREVIEWED_PARAMETER_SEMANTICS"})
                continue
            try:
                value, derived_from = _resolve(expression, aliases, registry)
            except (ValueError, ArithmeticError, OverflowError) as exc:
                issues.append({"default_set": default_set, "path": path, "name": name,
                               "state": "UNSUPPORTED_DEFAULT_EXPRESSION", "reason": str(exc)})
                continue
            line, end_line = locator.lineno, expression.end_lineno
            code = "\n".join(source.splitlines()[line - 1:end_line])
            definition_sha = _hash((default_set, path, name, code))
            rid = "CONST_DEFAULT_" + definition_sha
            pid = "PROV_DEFAULT_" + _hash((rid, sources[path]["sha256"], line))
            units, definition = SEMANTICS[name]
            record = {"id": rid, "type": "EmpiricalConstant", "harvest_owner": HARVESTER,
                      "default_set": default_set, "logical_key": default_set + ":" + name,
                      "name": name, "value": value, "units": units, "definition": definition,
                      "file": path, "line": line, "end_line": end_line, "container": container,
                      "code_quote": code, "raw_expression": ast.get_source_segment(source, expression),
                      "source_sha256": sources[path]["sha256"], "definition_sha256": definition_sha,
                      "component_id": node_id("SW", f"{path}:{container}"),
                      "model_ids": [MODEL_ID] if name in DRIVE_FIELDS else [],
                      "consumers": sorted((r for r in references if r["default_set"] == default_set and r["name"] == name),
                                          key=lambda r: (r["file"], r["line"], r["column"])),
                      "basis_class": "UNKNOWN_BASIS", "stated_basis": "Repository screening default; numerical scientific basis not established by AST extraction.",
                      "metadata_basis": "repository parameter semantics; not a literature claim",
                      "verified": True, "verification": "python_ast_numeric_definition",
                      "source_state": "CURRENT", "active": True, "provenance_id": pid,
                      "source_document_id": None, "hardware_qualified": False}
            if derived_from:
                # The default is a reference into the registry; its value is
                # the registry literal, and the declared basis is the
                # registry's label, never source support inferred here.
                record["derived_from"] = derived_from
            if default_set in set_definitions:
                entry = registry[set_definitions[default_set][1]]
                record["declared_basis_class"] = entry["basis_class"]
                record["declared_source_pages"] = sorted(
                    {s["pdf_page"] for s in entry["sources"] if s["pdf_page"] is not None})
            records.append(record)
            provenance.append({"id": pid, "type": "ProvenanceRecord", "kind": "CODE_STATES",
                               "subject": rid, "harvest_owner": HARVESTER, "asserted_by": path,
                               "file": path, "line": line, "end_line": end_line, "code_quote": code,
                               "source_sha256": sources[path]["sha256"], "definition_sha256": definition_sha,
                               "statement": f"{container} defines {name} = {value!r}; no literature support inferred.",
                               "verification": "python_ast_numeric_definition", "physical_validation": False})
    by_key = {(r["default_set"], r["name"]): r for r in records}
    comparisons = []
    for traditional, mdo in PAIRS.items():
        a, b = by_key.get(("traditional_screening", traditional)), by_key.get(("mdo_mission", mdo))
        comparisons.append({"traditional_name": traditional, "mdo_name": mdo,
                            "traditional_id": a["id"] if a else None, "mdo_id": b["id"] if b else None,
                            "traditional_value": a["value"] if a else None, "mdo_value": b["value"] if b else None,
                            "units": a["units"] if a else (b["units"] if b else None),
                            "state": "UNAVAILABLE" if not a or not b else ("EQUAL" if a["value"] == b["value"] else "DIFFERENT_ASSUMPTIONS"),
                            "shared_registry_set": (
                                a["derived_from"]["technology_id"]
                                if a and b and a.get("derived_from") and b.get("derived_from")
                                and a["derived_from"]["technology_id"] == b["derived_from"]["technology_id"]
                                else None),
                            "resolution": "Preserve separate technology assumptions; neither value is source-validated here."})
    pump_mass = by_key.get(("mdo_mission", "pump_specific_mass"))
    if pump_mass:
        shaft_consumers = [c for c in pump_mass["consumers"] if "P_shaft / mission.pump_specific_mass" in c["code_quote"]]
        if shaft_consumers and "hydraulic" in pump_mass["code_quote"].lower():
            issues.append({"state": "DECLARATION_CONSUMER_DISCREPANCY", "parameter_id": pump_mass["id"],
                           "note": "Definition comment says hydraulic power; consumers divide shaft power. Requires explicit convention review.",
                           "consumers": shaft_consumers})
    report = {"harvest_owner": HARVESTER, "scope": "bounded repository defaults and static references",
              "records": len(records), "default_sets": {s: sum(r["default_set"] == s for r in records)
                                                        for s in [*DEFAULT_SETS, *sorted(set_definitions)]},
              "sources": [sources[p] for p, _ in DEFAULT_SETS.values()], "comparisons": comparisons,
              "unpaired_parameters": [{"id": r["id"], "default_set": r["default_set"], "name": r["name"],
                                        "state": "NO_COUNTERPART_IN_INSPECTED_DEFAULT_SET"} for r in records
                                       if r["default_set"] in DEFAULT_SETS
                                       and r["name"] not in (PAIRS if r["default_set"] == "traditional_screening" else PAIRS.values())],
              "issues": issues, "scientific_support_created": False, "numerical_defaults_changed": False}
    return {"records": sorted(records, key=lambda r: r["id"]),
            "provenance": sorted(provenance, key=lambda r: r["id"]), "report": report}


def harvest(*, repo_root: Path = REPO_ROOT) -> dict:
    """Merge own records; preserve unrelated surveys and stale definition history."""
    result = collect(repo_root=repo_root)
    previous = {r["id"]: r for r in load("software_links")}
    current = {r["id"]: r for r in result["records"]}
    components = {r["id"]: r for r in previous.values() if r.get("type") == "SoftwareComponent"}
    models = {r["id"] for r in load("models")}
    for record in result["records"]:
        consumer_models = {mid for c in record["consumers"]
                           for mid in components.get(c.get("component_id"), {}).get("model_ids", [])}
        record["model_ids"] = sorted(set(record["model_ids"]) | (consumer_models & models))
        record["model_ids_basis"] = "repository parameter role and surveyed models of current static consumers; not source support"
        endpoints = {(record["component_id"], "DEFINES")}
        endpoints.update((c["component_id"], "USES") for c in record["consumers"] if c.get("component_id"))
        for source, edge in sorted(endpoints):
            if source not in components:
                continue  # no invented software node or dangling edge
            lid = "LINK_DEFAULT_" + _hash((source, record["id"], edge))
            current[lid] = {"id": lid, "type": "SoftwareLink", "from": source, "to": record["id"], "edge": edge,
                            "harvest_owner": HARVESTER, "active": True, "source_state": "CURRENT",
                            "provenance": record["provenance_id"], "note": "AST definition/reference only; not scientific validation."}
        for mid in record["model_ids"]:
            if mid not in models:
                continue
            lid = "LINK_DEFAULT_" + _hash((record["id"], mid, "PARAMETER_OF"))
            current[lid] = {"id": lid, "type": "SoftwareLink", "from": record["id"], "to": mid, "edge": "PARAMETER_OF",
                            "harvest_owner": HARVESTER, "active": True, "source_state": "CURRENT",
                            "provenance": record["provenance_id"], "note": "Repository parameter association; no source support asserted."}
    logical_keys = {r["logical_key"] for r in result["records"]}
    for rid, record in previous.items():
        if record.get("harvest_owner") != HARVESTER or rid in current:
            continue
        state = "STALE_DEFINITION" if record.get("logical_key") in logical_keys else "DEFINITION_UNAVAILABLE"
        if record.get("type") == "SoftwareLink":
            state = "STALE_PARAMETER_ASSOCIATION"
        current[rid] = dict(record, active=False, source_state=state)
    merged = {**previous, **current}
    save("software_links", merged.values(), meta={**load_meta("software_links"), "parameter_harvester": HARVESTER})
    provenance = {p["id"]: p for p in load("provenance")}
    provenance.update({p["id"]: p for p in result["provenance"]})
    save("provenance", provenance.values(), meta=load_meta("provenance"))
    report_path = store_path("software_links").parent / "reports" / "electric_drive_defaults.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result["report"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return result["report"]
