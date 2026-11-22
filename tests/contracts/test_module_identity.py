"""One implementation, one module identity, whatever the import route.

The implementation lives once, under ``lrekit``. Two older spellings must keep
importing without creating copies:

* ``raosim.<x>``: the package's name until 2026-11-21 (``raosim`` is now a
  compatibility package with only ``__init__`` and ``__main__``);
* the flat module names from before the component packages existed
  (``raosim.cea``, ``raosim.mdo``, ``lrekit.engine`` ...).

Before 2026-11-21 the ``lrekit`` facade extended its ``__path__`` with
``raosim.__path__``, so ``lrekit.engine`` executed ``raosim/engine.py`` a second
time and every dataclass, enum, registry and cache existed twice.  These
contracts pin the replacement, :mod:`lrekit._compat`.
"""

from __future__ import annotations

import ast
import importlib
import json
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import lrekit
from lrekit._compat import FLAT_ALIASES, canonical_name
from lrekit.paths import package_root


SRC = package_root().parent
FLAT_PAIRS = sorted(FLAT_ALIASES.items())


def _is_real_module(name: str) -> bool:
    """True when *name* is a file or package that really exists under ``src/``."""
    path = SRC.joinpath(*name.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _through_alias(name: str) -> bool:
    """True when *name* only resolves via the compatibility aliases."""
    parts = name.split(".")
    if parts[0] == "raosim":
        return len(parts) > 1 and not _is_real_module(".".join(parts[:2]))
    if parts[0] != "lrekit" or len(parts) < 2:
        return False
    # The longest importable prefix decides: lrekit.cea is an alias, while
    # lrekit.optimization.job.EngineJobSpec is a real module plus an attribute.
    for end in range(len(parts), 1, -1):
        if _is_real_module(".".join(parts[:end])):
            return False
    return canonical_name(name) != name


def _run_isolated(code: str) -> dict:
    """Run *code* in a fresh interpreter (clean ``sys.modules``) and return its JSON."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(SRC), "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("flat,canonical", FLAT_PAIRS, ids=[a for a, _ in FLAT_PAIRS])
def test_every_old_spelling_is_the_canonical_module(flat, canonical):
    target = importlib.import_module(f"lrekit.{canonical}")
    assert target.__name__ == f"lrekit.{canonical}" == target.__spec__.name
    assert importlib.import_module(f"raosim.{flat}") is target
    assert importlib.import_module(f"raosim.{canonical}") is target
    if _is_real_module(f"lrekit.{flat}"):  # lrekit.cli is its own module, not application.cli
        assert importlib.import_module(f"lrekit.{flat}").__name__ == f"lrekit.{flat}"
    else:
        assert importlib.import_module(f"lrekit.{flat}") is target


def test_from_import_of_an_old_name_returns_the_canonical_module():
    from raosim import cea  # noqa: PLC0415 - the old spelling is the point
    from lrekit import interface  # noqa: PLC0415
    from raosim.optimization import job  # noqa: PLC0415

    assert cea is importlib.import_module("lrekit.thermochemistry.cea")
    assert interface is importlib.import_module("lrekit.assembly.interfaces")
    assert job is importlib.import_module("lrekit.optimization.job")


def test_children_of_an_aliased_package_follow_the_parent_identity():
    assert importlib.import_module("raosim.mdo.job") is importlib.import_module("lrekit.optimization.job")
    assert importlib.import_module("lrekit.mdo.job") is importlib.import_module("lrekit.optimization.job")
    assert (importlib.import_module("raosim.spray.primary")
            is importlib.import_module("lrekit.components.injector.spray.primary"))
    assert (importlib.import_module("raosim.components.nozzle.solvers.moc")
            is importlib.import_module("lrekit.components.nozzle.solvers.moc"))


def test_dataclasses_are_one_type_and_pickle_through_their_canonical_module():
    from lrekit.mdo.job import EngineJobSpec as via_flat  # noqa: PLC0415
    from lrekit.optimization.job import EngineJobSpec as canonical  # noqa: PLC0415
    from raosim.mdo.job import EngineJobSpec as via_old_flat  # noqa: PLC0415
    from raosim.optimization.job import EngineJobSpec as via_old_name  # noqa: PLC0415
    from lrekit.assembly.interfaces import InterfaceGate  # noqa: PLC0415
    from raosim.interface import InterfaceGate as old_gate  # noqa: PLC0415

    assert via_flat is via_old_flat is via_old_name is canonical
    assert canonical.__module__ == "lrekit.optimization.job"
    assert old_gate is InterfaceGate
    gate = InterfaceGate(name="throat", status="pass", detail="shared radius", value=1.0, limit=2.0)
    restored = pickle.loads(pickle.dumps(gate))
    assert type(restored) is InterfaceGate and restored == gate


def test_objects_pickled_under_the_old_names_still_load():
    """Pickles written before the renames name ``raosim.interface`` / ``raosim.assembly.interfaces``."""
    from lrekit.assembly.interfaces import InterfaceGate  # noqa: PLC0415

    gate = InterfaceGate(name="throat", status="pass", detail="shared radius", value=1.0, limit=2.0)
    # Protocol 2 writes the class as a text GLOBAL record ("module\nname\n"), so the
    # module name can be swapped without re-framing; this mimics an older pickle.
    current = pickle.dumps(gate, protocol=2)
    assert b"lrekit.assembly.interfaces\nInterfaceGate\n" in current
    for old_module in (b"raosim.interface", b"raosim.assembly.interfaces"):
        legacy = current.replace(b"lrekit.assembly.interfaces\n", old_module + b"\n")
        assert old_module + b"\nInterfaceGate\n" in legacy
        restored = pickle.loads(legacy)
        assert type(restored) is InterfaceGate and restored == gate


def test_canonical_name_is_the_static_map():
    assert canonical_name("raosim.moc") == "lrekit.components.nozzle.solvers.moc"
    assert canonical_name("raosim.mdo.job") == "lrekit.optimization.job"
    assert canonical_name("lrekit.cea") == "lrekit.thermochemistry.cea"
    assert canonical_name("raosim.optimization.job") == "lrekit.optimization.job"
    assert canonical_name("lrekit.optimization.job") == "lrekit.optimization.job"


def test_the_compatibility_package_holds_no_implementation():
    files = sorted(p.name for p in (SRC / "raosim").iterdir() if p.name != "__pycache__")
    assert files == ["__init__.py", "__main__.py"]


def test_no_source_file_is_executed_as_two_modules():
    report = _run_isolated("""
        import importlib, json, pkgutil, sys
        import lrekit, raosim
        from lrekit._compat import FLAT_ALIASES
        failures = {}
        names = [m.name for m in pkgutil.walk_packages(lrekit.__path__, "lrekit.")]
        names += [f"raosim.{n.removeprefix('lrekit.')}" for n in names]
        names += [f"{p}.{flat}" for p in ("raosim", "lrekit") for flat in FLAT_ALIASES]
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:  # report, do not hide, import failures
                failures[name] = f"{type(exc).__name__}: {exc}"
        by_file = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if path and ("raosim" in path or "lrekit" in path):
                by_file.setdefault(path, set()).add(id(module))
        print(json.dumps({
            "imported": len(names),
            "failures": failures,
            "duplicated_files": sorted(p for p, ids in by_file.items() if len(ids) > 1),
            "spec_mismatch": sorted(n for n, m in sys.modules.items()
                                    if n.startswith(("raosim", "lrekit")) and getattr(m, "__spec__", None)
                                    and m.__spec__.name != m.__name__),
            "archive_modules": sorted(n for n, m in sys.modules.items()
                                      if "/archive/" in (getattr(m, "__file__", "") or "")),
        }))
    """)
    assert report["imported"] > 300
    assert report["failures"] == {}
    assert report["duplicated_files"] == []
    assert report["spec_mismatch"] == []
    assert report["archive_modules"] == []


def test_a_real_module_is_never_redirected(tmp_path):
    """A module that really exists under a name wins over the alias for that name."""
    (tmp_path / "components").mkdir()
    (tmp_path / "components" / "__init__.py").write_text("REAL = True\n")
    (tmp_path / "components" / "nozzle.py").write_text("REAL = True\n")
    (tmp_path / "cea.py").write_text("REAL = True\n")
    report = _run_isolated(f"""
        import importlib, json
        import lrekit, raosim
        raosim.__path__.append({str(tmp_path)!r})
        lrekit.__path__.append({str(tmp_path)!r})
        found = {{}}
        for name in ("lrekit.cli", "raosim.__main__", "raosim.components", "raosim.components.nozzle", "lrekit.cea"):
            module = importlib.import_module(name)
            found[name] = [module.__name__, getattr(module, "REAL", False)]
        print(json.dumps(found))
    """)
    assert report == {
        "lrekit.cli": ["lrekit.cli", False],
        "raosim.__main__": ["raosim.__main__", False],
        "raosim.components": ["raosim.components", True],
        "raosim.components.nozzle": ["raosim.components.nozzle", True],
        "lrekit.cea": ["lrekit.cea", True],
    }


def test_importing_the_packages_stays_lazy():
    report = _run_isolated("""
        import json, sys
        import lrekit, raosim
        heavy = ["jax", "cadquery", "matplotlib", "scipy", "lrekit.performance.engine", "lrekit.application.runner"]
        print(json.dumps(sorted(m for m in heavy if m in sys.modules)))
    """)
    assert report == []


def _imported_names(path: Path, module: str) -> list[tuple[int, str]]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                full = ".".join(base + ([node.module] if node.module else []))
            else:
                full = node.module or ""
            found += [(node.lineno, full)] + [(node.lineno, f"{full}.{a.name}") for a in node.names]
    return found


def test_implementation_never_imports_through_a_compatibility_alias():
    """Keeps the aliases removable once external callers have migrated."""
    offenders = []
    for path in sorted((SRC / "lrekit").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = ".".join(path.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")
        for line, name in _imported_names(path, module):
            if name.split(".")[0] == "raosim" or _through_alias(name):
                offenders.append(f"{path.relative_to(SRC)}:{line} imports {name}")
    assert offenders == []


def test_console_entry_points_resolve_to_one_runner():
    import tomllib  # noqa: PLC0415

    from lrekit import cli as lrekit_cli  # noqa: PLC0415
    from lrekit.application import cli as app_cli  # noqa: PLC0415

    scripts = tomllib.loads((SRC.parent / "pyproject.toml").read_text())["project"]["scripts"]
    assert set(scripts) == {"lrekit", "raosim", "RaoRocketSim", "RaoRocketSimLegacy"}
    for name, target in scripts.items():
        module, _, attr = target.partition(":")
        assert module.startswith("lrekit.") and not _through_alias(module), f"{name} -> {module}"
        assert callable(getattr(importlib.import_module(module), attr)), name
    # lrekit falls through to the same packaged runner entry as raosim/RaoRocketSim.
    assert lrekit_cli._runner_main is app_cli.main
    assert lrekit_cli.legacy_main is app_cli.legacy_main


def test_version_has_one_source():
    import tomllib  # noqa: PLC0415

    import raosim  # noqa: PLC0415

    project = tomllib.loads((SRC.parent / "pyproject.toml").read_text())
    assert "version" not in project["project"] and "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "lrekit.__version__"}
    assert raosim.__version__ == lrekit.__version__
