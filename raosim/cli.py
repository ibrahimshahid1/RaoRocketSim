"""Console entry points for RaoRocketSim."""

from __future__ import annotations

from importlib import import_module


def main() -> int:
    """Run the current packaged nozzle/chamber/injector design CLI."""
    from raosim.run_nozzle import main as runner_main

    return int(runner_main())


def legacy_main() -> int:
    """Run the older top-level toolbox CLI kept for compatibility."""
    result = import_module("main").main()
    return int(result or 0)
