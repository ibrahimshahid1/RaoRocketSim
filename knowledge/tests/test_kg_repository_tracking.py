"""A checkout must contain every knowledge module, test, tool and reviewed journal.

Regression for 2026-10-16: a blanket ``knowledge/`` ignore rule left tracked
modules importing untracked ones (``kg.execution_reach``), so a clean export
of HEAD failed at import and the corpus-free CI job could not run.
"""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

KNOWLEDGE = Path(__file__).resolve().parents[1]
ROOT = KNOWLEDGE.parent


def _git_available() -> bool:
    if not shutil.which("git"):
        return False
    probe = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    return probe.returncode == 0 and probe.stdout.strip() == "true"


pytestmark = pytest.mark.skipif(not _git_available(), reason="requires a git work tree")


def _ignored(paths):
    if not paths:
        return set()
    out = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
                         input="\n".join(str(p.relative_to(ROOT)) for p in paths),
                         capture_output=True, text=True)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def test_knowledge_code_tests_tools_and_reviews_are_not_ignored():
    paths = [p for pattern in ("kg/*.py", "tools/*.py", "tests/*.py", "workflows/*.json",
                               "workflows/*.js", "workflows/results/*", "*.md", "kg.sh", "schema.json")
             for p in KNOWLEDGE.glob(pattern) if p.is_file()]
    assert paths
    assert _ignored(paths) == set()


def test_authoritative_stores_are_not_ignored():
    import sys
    sys.path.insert(0, str(KNOWLEDGE))
    from kg.store import STORE_FILES
    paths = [KNOWLEDGE / name for name in STORE_FILES.values() if (KNOWLEDGE / name).exists()]
    assert _ignored(paths) == set()


def test_every_internal_kg_import_resolves_to_a_shipped_module():
    available = {p.stem for p in (KNOWLEDGE / "kg").glob("*.py")}
    missing = []
    for path in (KNOWLEDGE / "kg").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    names = [node.module] if node.module else [a.name for a in node.names]
                elif node.module == "kg":
                    names = [a.name for a in node.names]
                elif node.module and node.module.startswith("kg."):
                    names = [node.module.split(".")[1]]
            for name in names:
                head = (name or "").split(".")[0]
                # names imported from the package itself may be attributes of kg/__init__
                if head and head not in available and head not in {"KNOWLEDGE_ROOT", "REPO_ROOT", "CORPUS_ROOT",
                                                                   "AGENT_CORPUS_ROOT", "MARKDOWN_ROOT"}:
                    missing.append((path.name, head))
    assert missing == []
    assert _ignored([KNOWLEDGE / "kg" / f"{name}.py" for name in sorted(available)]) == set()
