#!/usr/bin/env python3
"""Compatibility wrapper for the packaged RaoRocketSim runner.

The maintained implementation lives in :mod:`lrekit.application.runner` so editable
installs, console scripts, ``python -m raosim``, and this historical script path
all execute the same CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ is None:
    # A checkout without an editable install still runs this script directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrekit.application import runner as _runner  # noqa: E402


main = _runner.main
_default_coolant_inlet_temperature = _runner._default_coolant_inlet_temperature


def __getattr__(name: str):
    return getattr(_runner, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_runner)))


if __name__ == "__main__":
    raise SystemExit(main())
