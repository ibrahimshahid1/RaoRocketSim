"""Keep legacy survey labels separate from independently reviewed source basis."""
from pathlib import Path
import hashlib

from . import REPO_ROOT, store


NUMERICAL_TUNING = {
    "CONST_RAOSIM_RAO_VARIATIONAL_PY_RAOSOLVERCONFIG_JAX_CONSTRAINT_WEIGHT_LADDER":
        ("jax_constraint_weight_ladder", "Default (1, 10, 30, 100) is the J4-gate ladder."),
    "CONST_RAOSIM_RAO_VARIATIONAL_PY_RAO_BVP_RESIDUAL_GROUPS_PHYSICS_WEIGHT":
        ("PHYSICS_WEIGHT = 0.05", "weight = 0.05"),
}


def reconcile(*, repo_root: Path | None = None):
    root = Path(repo_root or REPO_ROOT)
    records = store.load("software_links")
    annotated = corrected = 0
    for row in records:
        if row.get("type") != "EmpiricalConstant" or row.get("harvest_owner"):
            continue
        row.setdefault("reported_basis_class", row.get("basis_class", "UNKNOWN_BASIS"))
        row.setdefault("basis_class_basis", "Legacy agent-survey classification; not independently confirmed source evidence")
        row.setdefault("scientific_basis_review_state", "NOT_CHECKED")
        row.setdefault("value_freshness_state", "NOT_RECHECKED")
        row["verified_scope"] = "Historical code-literal proximity check only; not source basis, current value identity, calibration, or physical validation"
        annotated += 1
        probes = NUMERICAL_TUNING.get(row["id"])
        if not probes:
            continue
        path = root / row.get("file", "")
        if not path.is_file():
            continue
        data = path.read_bytes()
        text = data.decode("utf-8")
        if not all(probe in text for probe in probes):
            continue
        row.update(basis_class="IMPLEMENTATION_CHOICE",
                   basis_class_basis="Source-code comments describe numerical residual weights/continuation, not a fit to named physical measurements",
                   classification_evidence={"kind": "CODE_STATES", "file": row["file"],
                       "source_sha256": hashlib.sha256(data).hexdigest(), "anchors": list(probes),
                       "scope": "Classification as numerical solver tuning only; values and solver quality not independently validated"})
        corrected += 1
    store.save("software_links", records, meta=store.load_meta("software_links"))
    return {"legacy_basis_labels_annotated": annotated, "numerical_tuning_classifications": corrected}
