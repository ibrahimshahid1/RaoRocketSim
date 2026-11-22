"""python -m raosim must preserve the CLI's process status."""

from lrekit.paths import repository_root as _repository_root

from pathlib import Path
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize("status", [0, 2, 7])
def test_module_entrypoint_propagates_cli_return_code(monkeypatch, status):
    cli = types.ModuleType("lrekit.application.cli")
    cli.main = lambda: status
    monkeypatch.setitem(sys.modules, "lrekit.application.cli", cli)
    entrypoint = _repository_root() / "src" / "raosim" / "__main__.py"
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(entrypoint), run_name="__main__")
    assert result.value.code == status
