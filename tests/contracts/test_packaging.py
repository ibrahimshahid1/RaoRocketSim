"""The ``src/`` layout installs as a wheel that works outside the checkout.

Builds from a temporary copy of the project so setuptools never writes
``build/`` or ``*.egg-info`` into the working tree, then installs the wheel into
a fresh virtual environment.  Runtime dependencies are borrowed from the running
interpreter through a plain path entry, which deliberately does not process that
environment's editable-install ``.pth`` hooks: only the wheel provides
``raosim``/``lrekit``.
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from lrekit.paths import package_root


pytestmark = pytest.mark.slow
ROOT = package_root().parents[1]

REQUIRED_DATA = [
    "lrekit/data/cea_tables/catalog_v1.json",
    "lrekit/benchmark_data/cuffel_back_massier_1969.json",
]
REQUIRED_GLOBS = [
    "lrekit/data/cea_tables/*.npz",
    "lrekit/data/rao_domains/*.json",
    "lrekit/benchmark_data/cases/*.json",
    "lrekit/benchmark_data/curves/*.csv",
]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    pytest.importorskip("build")
    work = tmp_path_factory.mktemp("project")
    shutil.copytree(ROOT / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, work / name)
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(out), str(work)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    wheels = list(out.glob("lrekit-*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_ships_runtime_data_and_nothing_from_the_research_corpus(wheel):
    names = zipfile.ZipFile(wheel).namelist()
    tops = {n.split("/")[0] for n in names}
    assert {"raosim", "lrekit"} <= tops
    assert not tops & {"tests", "archive", "scripts", "knowledge", "docs", "main.py"}
    for path in REQUIRED_DATA:
        assert path in names, path
    for pattern in REQUIRED_GLOBS:
        directory, _, suffix = pattern.rpartition("/*")
        assert any(n.startswith(directory + "/") and n.endswith(suffix) for n in names), pattern
    assert sorted(n for n in names if n.startswith("raosim/")) == ["raosim/__init__.py", "raosim/__main__.py"]
    for forbidden in ("lrekit/data/raw/", "lrekit/data/derived/", "lrekit/data/catalog/"):
        assert not any(n.startswith(forbidden) for n in names), forbidden
    assert not any(n.endswith(".pdf") for n in names)
    assert sum(info.file_size for info in zipfile.ZipFile(wheel).infolist()) < 25e6


def test_clean_install_imports_and_runs_entry_points_outside_the_checkout(wheel, tmp_path):
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    py = env_dir / "bin" / "python"
    subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", "--no-index", str(wheel)],
                   check=True, capture_output=True, timeout=300)
    purelib = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             check=True, capture_output=True, text=True).stdout.strip()
    deps = [p for p in site.getsitepackages() if Path(p).is_dir()]
    Path(purelib, "zz_borrowed_dependencies.pth").write_text("\n".join(deps) + "\n")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    env = {"PATH": f"{env_dir / 'bin'}:/usr/bin:/bin", "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "MPLBACKEND": "Agg", "HOME": str(tmp_path)}
    probe = subprocess.run([py, "-c", """
import sys, pathlib
import raosim, lrekit
from lrekit.paths import repository_root
from importlib import resources
import raosim.mdo.job as legacy, lrekit.mdo.job as flat, lrekit.optimization.job as canonical
assert legacy is flat is canonical
assert "site-packages" in lrekit.__file__ and "site-packages" in raosim.__file__, lrekit.__file__
assert repository_root() == pathlib.Path.cwd()
assert resources.files("lrekit").joinpath("data/cea_tables/catalog_v1.json").is_file()
from lrekit.verification.benchmarks import DATA_ROOT
assert (DATA_ROOT / "cuffel_back_massier_1969.json").is_file()
import lrekit.application.runner, lrekit.legacy.toolbox
print("ok")
"""], cwd=outside, env=env, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0 and probe.stdout.strip().endswith("ok"), probe.stderr[-3000:]

    for command in (["lrekit", "--help"], ["raosim", "--help"], ["RaoRocketSim", "--help"],
                    ["RaoRocketSimLegacy", "--help"], ["python", "-m", "lrekit", "--help"],
                    ["python", "-m", "raosim", "--help"]):
        result = subprocess.run(command, cwd=outside, env=env, capture_output=True, text=True,
                                timeout=300, stdin=subprocess.DEVNULL)
        assert result.returncode == 0, (command, result.stderr[-2000:])
        assert "usage:" in result.stdout, command
    assert os.listdir(outside) == [], "running --help must not write into the working directory"
