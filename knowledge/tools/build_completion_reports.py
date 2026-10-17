"""Write CORPUS_COMPLETENESS, EQUATION_FIDELITY_AUDIT and MODEL_SOURCE_AUDIT reports.

    .venv-jax/bin/python knowledge/tools/build_completion_reports.py

Deterministic views over stores, journals and the packet ledger; read-only
with respect to scientific stores.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kg.completion import write_all  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(write_all(), indent=1, default=str))
