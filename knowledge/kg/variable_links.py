"""Rebuild conservative quantity aliases without rewriting source mathematics."""
from .quantities import resolve_symbol
from .store import load, load_meta, save


def reconcile() -> dict:
    rows = load("equations")
    changed = 0
    for e in rows:
        for v in e.get("variables", []):
            result = resolve_symbol(v.get("symbol", ""), e.get("subsystem") or None, units=v.get("units"), meaning=v.get("meaning"))
            current = {"resolution_state": result["state"],
                       "quantity_candidates": [c["qid"] for c in result["candidates"]][:3]}
            old = {k: v.get(k) for k in current}
            if old != current:
                history = v.setdefault("normalization_history", [])
                if old not in history:
                    history.append(old)
                v.update(current)
                v["normalization_basis"] = "SYSTEM_INFERS: symbol, exact canonical units and explicit meaning aliases; no unit conversion or physical validity inferred"
                changed += 1
    if changed:
        save("equations", rows, meta=load_meta("equations"))
    return {"changed_variable_links": changed, "source_forms_rewritten": 0}
