"""Refresh exact Python AST locators without certifying implementation science.

This reads source; it never imports the inspected modules or infers runtime
reach. Existing model/equation associations are retained, not invented. Missing
or ambiguous definitions remain inspectable as inactive records. A structural
definition change requires a new implementation-versus-literature comparison;
model source-support verdicts are owned by the source review workflow.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import REPO_ROOT
from . import store

OWNER = "kg.implementation_freshness@1"
SCOPE = "Exact file and qualified Python AST symbol only; not scientific correctness, execution reach, or physical validation."


class _Definitions(ast.NodeVisitor):
    def __init__(self):
        self.scope: list[str] = []
        self.nodes: dict[str, list[ast.AST]] = defaultdict(list)

    def _definition(self, node):
        self.nodes[".".join([*self.scope, node.name])].append(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _definition
    visit_AsyncFunctionDef = _definition
    visit_ClassDef = _definition

    def _assignment(self, node):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        def target_names(target):
            if isinstance(target, ast.Name):
                return [target.id]
            if isinstance(target, (ast.Tuple, ast.List)):
                return [name for child in target.elts for name in target_names(child)]
            return []  # obj.attr and array[index] do not define scoped symbols
        names = {name for target in targets for name in target_names(target)}
        for name in names:
            self.nodes[".".join([*self.scope, name])].append(node)
        # Do not confuse names in the assigned expression with definitions.

    visit_Assign = _assignment
    visit_AnnAssign = _assignment


def _read(path: Path) -> dict:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return {"state": "SOURCE_UNAVAILABLE", "reason": str(exc)}
    sha = hashlib.sha256(payload).hexdigest()
    try:
        tree = ast.parse(payload, filename=str(path))
    except (SyntaxError, ValueError, UnicodeError) as exc:
        return {"state": "SOURCE_UNREADABLE", "reason": str(exc), "source_sha256": sha}
    visitor = _Definitions()
    visitor.visit(tree)
    visitor.nodes["<module>"] = [tree]
    return {"state": "CURRENT", "nodes": visitor.nodes, "source_sha256": sha,
            "line_count": max(1, len(payload.splitlines()))}


def _symbol(record: dict, relative: Path) -> tuple[str | None, str]:
    module = ".".join(relative.with_suffix("").parts)
    if module.endswith(".__init__"):
        module = module[:-len(".__init__")]
    qualname = record.get("qualname") or ""
    if ":" in qualname:
        prefix, symbol = qualname.split(":", 1)
        if prefix not in {module, relative.as_posix()}:
            return None, "Qualified module does not match the recorded file."
    elif qualname.startswith(module + "."):
        symbol = qualname[len(module) + 1:]
    else:
        return None, "An exact module-qualified symbol is required."
    symbol = symbol.replace(".<locals>.", ".").strip()
    if not symbol:
        return None, "The qualified symbol is empty."
    return symbol, ""


def _snapshot(record: dict) -> dict:
    fields = ("file", "line", "end_line", "source_sha256", "definition_sha256", "ast_qualname",
              "implementation_locator_state", "implementation_scientific_comparison_state", "active", "source_state", "verified")
    return {key: record[key] for key in fields if key in record}


def reconcile(*, repo_root: Path | None = None, write: bool = True) -> dict:
    """Reconcile stored locators; ``write=False`` performs a read-only audit.

    First observations have no historical body fingerprint and therefore cannot
    prove an existing survey still matches the implementation. Unchanged ASTs
    survive line/comment-only edits without losing an already recorded comparison
    state. Inactive records owned elsewhere are never automatically reactivated.
    """
    root = Path(REPO_ROOT if repo_root is None else repo_root).resolve()
    records = store.load("software_links")
    provenance = {p["id"]: p for p in store.load("provenance")}
    cache: dict[Path, dict] = {}
    now = datetime.now(timezone.utc).isoformat()
    stats = Counter()
    unavailable = []
    changed_bodies = []
    updated = []
    provenance_dirty = False
    for original in records:
        if original.get("type") != "SoftwareComponent":
            updated.append(original)
            continue
        if ((not original.get("active", True) or original.get("source_state", "CURRENT") != "CURRENT")
                and original.get("implementation_locator_owner") != OWNER):
            stats["skipped_inactive"] += 1
            updated.append(original)
            continue
        stats["inspected"] += 1
        record = copy.deepcopy(original)
        old = _snapshot(original)
        definition_sha = None
        candidates = []
        state, reason = "SYMBOL_NOT_FOUND", "No exact qualified AST definition was found."
        try:
            path = (root / (record.get("file") or "")).resolve()
            relative = path.relative_to(root)
        except ValueError:
            path = None
            state, reason = "SOURCE_UNAVAILABLE", "Recorded file is outside the repository."
        if path is not None:
            if path not in cache:
                cache[path] = _read(path)
            source = cache[path]
            if source["state"] != "CURRENT":
                state, reason = source["state"], source["reason"]
            else:
                symbol, error = _symbol(record, relative)
                if error:
                    state, reason = "QUALNAME_FILE_MISMATCH", error
                else:
                    matches = source["nodes"].get(symbol, [])
                    if len(matches) != 1:
                        candidates = sorted(name for name in source["nodes"] if name.split(".")[-1] == symbol.split(".")[-1])
                        if len(matches) > 1 or (not matches and len(candidates) > 1):
                            state, reason = "AMBIGUOUS_SYMBOL", "Multiple definitions exist; no nearest-line or terminal-name guess is permitted."
                    else:
                        node, = matches
                        definition_sha = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                        state, reason = "VERIFIED", "Exact qualified symbol located in current Python AST."
                        record.update(line=getattr(node, "lineno", 1),
                                      end_line=getattr(node, "end_lineno", source["line_count"]),
                                      source_sha256=source["source_sha256"], definition_sha256=definition_sha,
                                      ast_qualname=symbol, verified=True, active=True, source_state="CURRENT")

        record.update(implementation_locator_state=state, implementation_locator_owner=OWNER,
                      implementation_locator_scope=SCOPE, implementation_locator_reason=reason)
        if state == "VERIFIED":
            stats["verified"] += 1
            previous_sha = original.get("definition_sha256")
            changed = bool(previous_sha and previous_sha != definition_sha)
            recovered = original.get("implementation_locator_state") not in (None, "VERIFIED")
            if changed or not previous_sha or recovered:
                why = ("Definition AST changed." if changed else "Previously unavailable locator restored; implementation comparison requires review."
                       if recovered else "No earlier definition fingerprint; prior survey/body comparison cannot be confirmed.")
                record.setdefault("implementation_scientific_comparison_history", []).append({
                    "state": original.get("implementation_scientific_comparison_state", "UNKNOWN"),
                    "definition_sha256": previous_sha, "source_sha256": original.get("source_sha256"),
                    "recorded_at": now, "reason": why,
                })
                record["implementation_scientific_comparison_state"] = "NOT_RECHECKED"
                record["implementation_scientific_comparison_reason"] = why
                stats["changed_definitions" if changed else "locators_restored" if recovered else "fingerprints_baselined"] += 1
                if changed:
                    changed_bodies.append(record["id"])
            elif "implementation_scientific_comparison_state" not in record:
                record["implementation_scientific_comparison_state"] = "NOT_RECHECKED"
                record["implementation_scientific_comparison_reason"] = "Locator verification does not establish an implementation-versus-literature comparison."
            if original.get("line") != record["line"] or original.get("end_line") != record["end_line"]:
                stats["locators_corrected"] += 1
            if original.get("line") != record["line"]:
                stats["line_corrections"] += 1
            if original.get("end_line") != record["end_line"]:
                stats["end_line_updates"] += 1
        else:
            if original.get("implementation_scientific_comparison_state") != "UNAVAILABLE":
                record.setdefault("implementation_scientific_comparison_history", []).append({
                    "state": original.get("implementation_scientific_comparison_state", "UNKNOWN"),
                    "definition_sha256": original.get("definition_sha256"), "recorded_at": now, "reason": reason,
                })
            record.update(active=False, verified=False, source_state="IMPLEMENTATION_UNAVAILABLE",
                          implementation_scientific_comparison_state="UNAVAILABLE")
            unavailable.append({"id": record["id"], "state": state, "reason": reason, "candidates": candidates})
            stats["unavailable"] += 1

        if _snapshot(record) != old:
            record.setdefault("implementation_locator_history", []).append({**old, "recorded_at": now,
                                                                            "reason": reason})
            record["implementation_locator_checked_at"] = now
        if record != original:
            stats["records_updated"] += 1
        updated.append(record)
        # Refresh existing CODE_STATES locator evidence only. Source statements
        # and model support are owned elsewhere and must remain untouched.
        pid = record.get("provenance_id")
        if pid in provenance and provenance[pid].get("kind") == "CODE_STATES":
            prior = provenance[pid]
            new = copy.deepcopy(prior)
            new.update(implementation_locator_state=state, implementation_locator_scope=SCOPE,
                       implementation_scientific_comparison_state=record["implementation_scientific_comparison_state"])
            if state == "VERIFIED":
                for key in ("file", "line", "end_line", "source_sha256", "definition_sha256"):
                    new[key] = record[key]
            if new != prior:
                new.setdefault("implementation_locator_history", []).append({**_snapshot(prior), "recorded_at": now})
                provenance[pid] = new
                provenance_dirty = True
        elif pid and pid not in provenance:
            stats["missing_provenance"] += 1
            if state == "VERIFIED":
                provenance[pid] = {"id": pid, "type": "ProvenanceRecord", "kind": "CODE_STATES",
                    "subject": record["id"], "asserted_by": OWNER,
                    "file": record["file"], "line": record["line"], "end_line": record["end_line"],
                    "source_sha256": record["source_sha256"], "definition_sha256": record["definition_sha256"],
                    "statement": "Exact qualified AST locator observed in current source; historical scientific assertions are not revalidated.",
                    "implementation_scientific_comparison_state": "NOT_RECHECKED", "recorded_at": now}
                provenance_dirty = True
                stats["provenance_repaired"] += 1

    if write:
        if updated != records:
            store.save("software_links", updated, meta=store.load_meta("software_links"))
        if provenance_dirty:
            store.save("provenance", provenance.values(), meta=store.load_meta("provenance"))
    return {"scope": SCOPE, "write": write, "files_inspected": len(cache),
            **{key: stats[key] for key in ("inspected", "verified", "unavailable", "skipped_inactive",
                                           "locators_corrected", "fingerprints_baselined", "changed_definitions",
                                           "line_corrections", "end_line_updates", "locators_restored",
                                           "records_updated", "missing_provenance", "provenance_repaired")},
            "changed_definition_ids": changed_bodies, "unavailable_components": unavailable}


def definition_fingerprint(file: str, symbol: str, *, repo_root: Path | None = None) -> dict:
    """Fingerprint one exact qualified definition for a model-versus-source review.

    The body fingerprint is the AST dump without positions, so comment/line-only
    edits keep it while any structural code change invalidates a comparison.
    """
    root = Path(REPO_ROOT if repo_root is None else repo_root).resolve()
    try:
        path = (root / file).resolve()
        path.relative_to(root)
    except ValueError:
        return {"file": file, "symbol": symbol, "state": "SOURCE_UNAVAILABLE"}
    source = _read(path)
    if source["state"] != "CURRENT":
        return {"file": file, "symbol": symbol, "state": source["state"]}
    matches = source["nodes"].get(symbol, [])
    if len(matches) != 1:
        return {"file": file, "symbol": symbol,
                "state": "AMBIGUOUS_SYMBOL" if len(matches) > 1 else "SYMBOL_NOT_FOUND"}
    node, = matches
    return {"file": file, "symbol": symbol, "state": "VERIFIED", "line": getattr(node, "lineno", 1),
            "end_line": getattr(node, "end_lineno", source["line_count"]), "file_sha256": source["source_sha256"],
            "definition_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
