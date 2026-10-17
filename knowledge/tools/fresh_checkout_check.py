"""Prove that a commit of the current knowledge changes yields a working checkout.

    .venv-jax/bin/python knowledge/tools/fresh_checkout_check.py [--output report.json]

Exports ``HEAD`` with ``git archive`` into a temporary directory (no refs,
index or objects are written), overlays exactly the working-tree files a commit
would add or change (tracked modifications plus untracked, non-ignored files),
and runs the corpus-free knowledge suite there. It also runs the same suite on
the bare ``HEAD`` export, so the before/after difference is explicit. The
listed files are the ones the owner must ``git add``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def changed_files() -> dict:
    modified, deleted, untracked = [], [], []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        code, path = line[:2], line[3:]
        if code == "??":
            untracked.append(path)
        elif "D" in code:
            deleted.append(path)
        else:
            modified.append(path)
    return {"modified_tracked": sorted(modified), "deleted_tracked": sorted(deleted),
            "untracked_to_add": sorted(untracked)}


def _export_head(target: Path) -> None:
    archive = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "archive", "HEAD"],
                             check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)


def _pytest(where: Path) -> dict:
    result = subprocess.run([sys.executable, "-m", "pytest", "knowledge/tests", "-q", "-p", "no:cacheprovider"],
                            cwd=where, capture_output=True, text=True, timeout=1200)
    tail = result.stdout.strip().splitlines()[-1:] or [result.stderr.strip()[-300:]]
    errors = [line for line in result.stdout.splitlines() if line.startswith(("ERROR", "E   "))][:10]
    return {"exit_code": result.returncode, "summary": tail[0], "first_errors": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    ns = ap.parse_args(argv)
    files = changed_files()
    report = {"head": _git("rev-parse", "HEAD").strip(), **files}
    with tempfile.TemporaryDirectory(prefix="kg-fresh-") as tmp:
        bare, overlaid = Path(tmp, "head"), Path(tmp, "head_plus_changes")
        bare.mkdir()
        _export_head(bare)
        shutil.copytree(bare, overlaid)
        for rel in files["modified_tracked"] + files["untracked_to_add"]:
            src, dst = ROOT / rel, overlaid / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for rel in files["deleted_tracked"]:
            (overlaid / rel).unlink(missing_ok=True)
        report["head_only_knowledge_tests"] = _pytest(bare)
        report["head_plus_changes_knowledge_tests"] = _pytest(overlaid)
    report["proof"] = ("PASS" if report["head_plus_changes_knowledge_tests"]["exit_code"] == 0 else "FAIL")
    text = json.dumps(report, indent=1)
    if ns.output:
        ns.output.write_text(text + "\n")
    print(text)
    return 0 if report["proof"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
