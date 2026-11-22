"""Allow `python -m lrekit` to launch the main CLI."""

from __future__ import annotations

from lrekit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
