"""Parsers for NASA/JHU MOC_Grid_BDE reference outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LegacyTable:
    """Simple columnar representation of a NASA/JHU whitespace table."""

    columns: tuple[str, ...]
    data: np.ndarray
    provenance: dict = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]


@dataclass(frozen=True)
class SummaryReport:
    """Parsed scalar fields and initial-data table from ``summary.out``."""

    fields: dict[str, str]
    initial_data: LegacyTable | None
    provenance: dict = field(default_factory=dict)


def _normalise_column(name: str) -> str:
    return (
        name.strip()
        .replace("(deg)", "_deg")
        .replace(".", "")
        .replace("/", "_over_")
        .replace("*", "star")
        .replace("(", "_")
        .replace(")", "")
        .replace("'", "")
        .replace(" ", "_")
    )


def _read_numeric_table(path: str | Path, *, skip_prefixes: tuple[str, ...] = ()) -> LegacyTable:
    p = Path(path)
    lines = [line.rstrip() for line in p.read_text().splitlines() if line.strip()]
    header: tuple[str, ...] | None = None
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if skip_prefixes and stripped.startswith(skip_prefixes):
            continue
        parts = stripped.split()
        try:
            row = [float(part) for part in parts]
        except ValueError:
            # NASA output files use tabs between columns; column names like
            # "Pressure, psia" contain spaces and split() would oversplit.
            tab_parts = [p for p in line.split("\t") if p.strip()]
            if len(tab_parts) >= 2:
                header = tuple(_normalise_column(part) for part in tab_parts)
            else:
                header = tuple(_normalise_column(part) for part in parts)
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No numeric rows found in {p}")
    width = len(rows[0])
    if header is None or len(header) != width:
        header = tuple(f"col{i}" for i in range(width))
    return LegacyTable(
        columns=header,
        data=np.asarray(rows, dtype=float),
        provenance={"path": str(p), "rows": len(rows)},
    )


def parse_wall_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``wall.out``."""
    return _read_numeric_table(path)


def parse_center_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``center.out``."""
    return _read_numeric_table(path, skip_prefixes=("Centerline",))


def parse_kernel_out(path: str | Path) -> LegacyTable:
    """Parse NASA kernel files such as ``TT'BF_Kernel.out``."""
    return _read_numeric_table(path)


def parse_rao_dat(path: str | Path) -> LegacyTable:
    """Parse NASA ``rao.dat`` three-column contour output."""
    table = _read_numeric_table(path)
    return LegacyTable(
        columns=("R_over_Rstar", "X_over_Rstar", "theta_deg"),
        data=table.data,
        provenance=table.provenance,
    )


def parse_summary_out(path: str | Path) -> SummaryReport:
    """Parse scalar fields and the initial throat table from ``summary.out``."""
    p = Path(path)
    lines = p.read_text().splitlines()
    fields: dict[str, str] = {}
    table_header: tuple[str, ...] | None = None
    table_rows: list[list[float]] = []
    in_table = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("I\tX/R*"):
            table_header = tuple(_normalise_column(part) for part in line.split("\t"))
            in_table = True
            continue
        if in_table:
            parts = line.split()
            try:
                table_rows.append([float(part) for part in parts])
                continue
            except ValueError:
                in_table = False
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()

    initial = None
    if table_header and table_rows:
        initial = LegacyTable(
            columns=table_header,
            data=np.asarray(table_rows, dtype=float),
            provenance={"path": str(p), "section": "Initial Data Line"},
        )
    return SummaryReport(
        fields=fields,
        initial_data=initial,
        provenance={"path": str(p)},
    )
