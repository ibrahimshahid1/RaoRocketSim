"""Allow `python -m raosim` to launch the design CLI (same as the `raosim` command)."""

from __future__ import annotations

from lrekit.application.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
