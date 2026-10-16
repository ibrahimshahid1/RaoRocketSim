"""python -m raosim must preserve the CLI's process status."""

from pathlib import Path
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize("status", [0, 2, 7])
def test_module_entrypoint_propagates_cli_return_code(monkeypatch, status):
    cli = types.ModuleType("raosim.cli")
    cli.main = lambda: status
    monkeypatch.setitem(sys.modules, "raosim.cli", cli)
    entrypoint = Path(__file__).resolve().parents[1] / "raosim" / "__main__.py"
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(entrypoint), run_name="__main__")
    assert result.value.code == status
