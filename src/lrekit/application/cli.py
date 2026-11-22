"""Console entry points for LREKit and the legacy RaoRocketSim alias."""

from __future__ import annotations

from importlib import import_module
import sys


def main() -> int:
    """Run the current packaged nozzle/chamber/injector design CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "export-openfoam-spray":
        from lrekit.simulation.openfoam import cli_main as openfoam_cli_main

        return int(openfoam_cli_main(sys.argv[2:]))

    from lrekit.application.runner import main as runner_main

    return int(runner_main())


def legacy_main() -> int:
    """Run the older top-level toolbox CLI kept for compatibility."""
    result = import_module("lrekit.legacy.toolbox").main()
    return int(result or 0)
