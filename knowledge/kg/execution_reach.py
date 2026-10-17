"""Bounded static call-site evidence; never a runtime trace or physics validation.

Explicit model-to-symbol mappings below are project-maintained implementation
routes. Unmapped models and unresolved object/dynamic calls remain UNKNOWN.
"""
from __future__ import annotations

import ast
from collections import defaultdict, deque
import hashlib
import importlib.util
from pathlib import Path


MODEL_SYMBOLS = {
    "thermal.bartz": ("raosim.jax.thermal.bartz_hg",),
    "thermal.sieder_tate": ("raosim.jax.thermal.sieder_tate_hc",),
    "injector.spray_cstar_fixed_point": ("raosim.spray_coupling.solve_spray_cstar_fixed_point",),
    "spray.schiller_naumann_drag": ("raosim.spray.drag.drag_acceleration",),
    "spray.seeded_discrete_random_walk": ("raosim.spray.dispersion.DiscreteRandomWalk.advance",),
    "spray.rayleigh_taylor_openfoam": ("raosim.spray.breakup.rayleigh_taylor_metrics",),
    "spray.spalding_evaporation_eq16": ("raosim.spray.evaporation.evaporation_rate_2021",),
    "spray.weighted_smd_rosin_rammler": ("raosim.spray.statistics.summarize_spray",),
    "spray.deterministic_parcel_march": ("raosim.spray.solver.march_parcels",),
    "spray.typed_cycle_handoff": ("raosim.spray.handoff.build_cycle_handoff",),
}
ROOTS = {
    "optimizer": ("raosim.mdo.nlp._make_callables.solve",),
    "traditional_analysis": ("raosim.design.design_nozzle_v2",),
    "explicit_analysis_api": ("raosim.spray.solver.march_parcels", "raosim.spray.handoff.build_cycle_handoff"),
}


class StaticReach:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.definitions = {}
        self.edges = defaultdict(list)
        # Deliberately bounded to the cooling/optimizer and opt-in spray paths.
        files = {repo_root / p for p in (
            "raosim/mdo/nlp.py", "raosim/mdo/engine.py", "raosim/mdo/cooling.py",
            "raosim/jax/thermal.py", "raosim/design.py", "raosim/injector.py", "raosim/physics.py",
            "raosim/spray_coupling.py")}
        files.update((repo_root / "raosim/spray").glob("*.py"))
        parsed = []
        for path in sorted(files):
            if not path.is_file():
                continue
            data = path.read_bytes()
            try:
                tree = ast.parse(data)
            except SyntaxError:
                continue
            module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            aliases = {}
            # Function-local/dynamic imports are deliberately unresolved rather
            # than leaking an alias into unrelated function scopes.
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for item in node.names:
                        aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom):
                    package = module.rsplit(".", 1)[0]
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else (node.module or "")
                    for item in node.names:
                        if item.name != "*":
                            aliases[item.asname or item.name] = base + "." + item.name

            def register(body, parents=()):
                for node in body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        key = module + "." + ".".join((*parents, node.name))
                        self.definitions[key] = {"file": str(path.relative_to(repo_root)), "line": node.lineno,
                                                 "source_sha256": hashlib.sha256(data).hexdigest()}
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            parsed.append((module, key, node, aliases))
                        register(node.body, (*parents, node.name))
            register(tree.body)

        for module, caller, node, aliases in parsed:
            def resolve(expr):
                if isinstance(expr, ast.Name):
                    parts = caller[len(module) + 1:].split(".")
                    for count in range(len(parts), -1, -1):
                        candidate = module + "." + ".".join((*parts[:count], expr.id))
                        if candidate in self.definitions:
                            return candidate
                    return aliases.get(expr.id)
                if isinstance(expr, ast.Attribute):
                    base = resolve(expr.value)
                    return base + "." + expr.attr if base else None
                return None

            index = self
            class Calls(ast.NodeVisitor):
                def visit_FunctionDef(self, child):
                    pass
                visit_AsyncFunctionDef = visit_FunctionDef
                def visit_ClassDef(self, child):
                    pass
                def visit_Call(self, call):
                    target = resolve(call.func)
                    if target in index.definitions:
                        index.edges[caller].append({"caller": caller, "callee": target,
                                                    **index.definitions[caller], "line": call.lineno})
                    self.generic_visit(call)
            visitor = Calls()
            for statement in node.body:
                visitor.visit(statement)

    def _path(self, start: str, targets: set[str]) -> list[dict] | None:
        if start not in self.definitions:
            return None
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            current, path = queue.popleft()
            if current in targets:
                return path
            for edge in self.edges[current]:
                if edge["callee"] not in seen:
                    seen.add(edge["callee"])
                    queue.append((edge["callee"], [*path, edge]))
        return None

    def model(self, registry_id: str) -> dict:
        targets = set(MODEL_SYMBOLS.get(registry_id, ()))
        available = targets & self.definitions.keys()
        routes = []
        for scope, roots in ROOTS.items():
            for root in roots:
                path = self._path(root, available)
                if path is not None:
                    routes.append({"scope": scope, "root": root, "root_locator": self.definitions[root],
                                   "call_sites": path, "state": "STATIC_POSSIBLE_CALL_PATH"})
        return {"state": "CODE_INSPECTED" if routes else "UNKNOWN", "routes": routes,
                "mapped_symbols": sorted(targets), "located_symbols": sorted(available),
                "optimizer_reach_state": ("STATIC_PATH_FOUND" if any(r["scope"] == "optimizer" for r in routes)
                                          else "NO_STATIC_PATH_FOUND" if available else "UNKNOWN"),
                "runtime_trace_state": "NOT_RUN", "dynamic_dispatch_state": "UNKNOWN",
                "optimizer_leverage_state": "NOT_MEASURED",
                "basis": "bounded AST imports and direct call sites; conditional execution and dynamic calls are not proven"}
