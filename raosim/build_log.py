"""
build_log.py – Versioned build output with metadata logging.

Every run of main.py stores its artefacts (CSV, STL, sweep data) in a
monotonically increasing versioned directory under ``builds/``.

Directory naming scheme::

    builds/v001_20260302_121700/
    builds/v002_20260302_130015/
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDS_DIR = _REPO_ROOT / "builds"


def _next_version() -> int:
    """Scan ``builds/`` for the highest existing vNNN prefix and return N+1."""
    if not BUILDS_DIR.exists():
        return 1
    pattern = re.compile(r"^v(\d{3})_")
    max_v = 0
    for child in BUILDS_DIR.iterdir():
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def create_build_dir() -> tuple[Path, int]:
    """Create and return the next versioned build directory.

    Returns
    -------
    (build_dir, version)
        The resolved Path of the new directory and its version number.
    """
    version = _next_version()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"v{version:03d}_{stamp}"
    build_dir = BUILDS_DIR / dirname
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir, version


def write_metadata(
    build_dir: Path,
    *,
    version: int,
    mode: str,
    params: dict[str, Any],
    performance: dict[str, Any] | None = None,
    files: list[str] | None = None,
) -> Path:
    """Write a human-readable ``metadata.txt`` into *build_dir*.

    Parameters
    ----------
    build_dir : Path
        Directory created by :func:`create_build_dir`.
    version : int
        Build version number.
    mode : str
        One of ``"batch"``, ``"interactive"``, or ``"sweep"``.
    params : dict
        Input parameters (propellant, Pc, Pa, Rt, epsilon, …).
    performance : dict, optional
        Engine performance results to log.
    files : list[str], optional
        Basenames of output files written into this build directory.

    Returns
    -------
    Path to the metadata file.
    """
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append(f"  Rao Bell Nozzle — Build v{version:03d}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Timestamp : {datetime.now().isoformat()}")
    lines.append(f"Mode      : {mode}")
    lines.append("")


    lines.append("── Input Parameters " + "─" * 40)
    _max_key = max(len(k) for k in params) if params else 0
    for key, val in params.items():
        lines.append(f"  {key:<{_max_key}} : {val}")
    lines.append("")


    if performance:
        lines.append("── Engine Performance " + "─" * 38)
        _max_key = max(len(k) for k in performance)
        for key, val in performance.items():
            lines.append(f"  {key:<{_max_key}} : {val}")
        lines.append("")


    if files:
        lines.append("── Output Files " + "─" * 43)
        for f in files:
            lines.append(f"  • {f}")
        lines.append("")

    meta_path = build_dir / "metadata.txt"
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta_path
