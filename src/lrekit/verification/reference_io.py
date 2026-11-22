"""Parsers for NASA/JHU MOC_Grid_BDE reference outputs.

The legacy code writes a mix of whitespace tables and Tecplot multi-zone
``.plt`` files.  These parsers intentionally preserve the raw numeric data
while giving the known files stable, normalized column names so tests and
comparison scripts can use the NASA/JHU outputs as an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable

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
class TecplotZone:
    """One numeric ``zone`` from a Tecplot ASCII file."""

    name: str
    columns: tuple[str, ...]
    data: np.ndarray
    metadata: dict = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]


@dataclass(frozen=True)
class TecplotFile:
    """Parsed Tecplot ASCII file with variables and zones."""

    columns: tuple[str, ...]
    zones: tuple[TecplotZone, ...]
    title: str | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def data(self) -> np.ndarray:
        if not self.zones:
            return np.empty((0, len(self.columns)), dtype=float)
        populated = [zone.data for zone in self.zones if zone.data.size]
        if not populated:
            return np.empty((0, len(self.columns)), dtype=float)
        return np.vstack(populated)

    def zone(self, name: str) -> TecplotZone:
        for zone in self.zones:
            if zone.name == name:
                return zone
        raise KeyError(name)


@dataclass(frozen=True)
class SummaryReport:
    """Parsed scalar fields and initial-data table from ``summary.out``."""

    fields: dict[str, str]
    initial_data: LegacyTable | None
    provenance: dict = field(default_factory=dict)


WALL_COLUMNS = (
    "i", "j", "X_over_Rstar", "R_over_Rstar", "mach", "theta_deg",
    "Pressure_psia",
)
CENTER_COLUMNS = (
    "J", "X_over_Rstar", "R_over_Rstar", "Mach", "Pres", "Temp", "Rho",
    "Theta", "Gamma", "MassFlow",
)
POINT_TRACE_COLUMNS = (
    "i", "j", "x", "r", "mach", "theta", "press", "temp", "massflow",
)
KERNEL_COLUMNS = (
    "i", "j", "x", "r", "mach", "theta", "press", "temp",
)
KERNEL_WITH_MASS_COLUMNS = (*KERNEL_COLUMNS, "Massflow")
UNCROPPED_KERNEL_COLUMNS = (
    "i", "j", "x_in", "r_in", "mach", "theta", "pres_psi", "temp_R",
    "Density_slug_over_ft3", "massflow",
)
TT_PRIME_COLUMNS = ("i", "X", "R", "MACH", "THETA", "massflow")
THETA_B_COLUMNS = ("I", "ThetaB", "xArcMax", "LastKernelJ", "paramErr")
RAO_COLUMNS = ("R_over_Rstar", "X_over_Rstar", "theta_deg")


def _normalise_column(name: str) -> str:
    return (
        name.strip()
        .strip('"')
        .replace("(deg)", "_deg")
        .replace(".", "")
        .replace(",", "_")
        .replace("/", "_over_")
        .replace("*", "star")
        .replace("(", "_")
        .replace(")", "")
        .replace("'", "")
        .replace("%", "pct")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _coerce_columns(table: LegacyTable, columns: tuple[str, ...]) -> LegacyTable:
    if table.data.shape[1] != len(columns):
        raise ValueError(
            f"{table.provenance.get('path', '<table>')} has "
            f"{table.data.shape[1]} columns, expected {len(columns)}"
        )
    return LegacyTable(columns=columns, data=table.data, provenance=table.provenance)


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


_QUOTED_VALUE_RE = re.compile(r'"([^"]+)"')
_ZONE_DIM_RE = re.compile(r"\b([IJK])\s*=\s*(\d+)", re.IGNORECASE)


def _parse_tecplot_variables(line: str) -> tuple[str, ...]:
    variables = tuple(_normalise_column(match) for match in _QUOTED_VALUE_RE.findall(line))
    if not variables:
        raise ValueError(f"Could not parse Tecplot VARIABLES line: {line!r}")
    return variables


def _parse_zone_line(line: str) -> tuple[str, dict]:
    name_match = re.search(r'\bt\s*=\s*"([^"]*)"', line, flags=re.IGNORECASE)
    name = name_match.group(1) if name_match else ""
    metadata: dict[str, int | str] = {}
    for key, value in _ZONE_DIM_RE.findall(line):
        metadata[key.upper()] = int(value)
    return name, metadata


def parse_tecplot_file(path: str | Path) -> TecplotFile:
    """Parse a NASA/JHU Tecplot ASCII file into zones.

    Supported files include ``MOC_Grid.plt``, ``MOC_SL.plt``, and
    ``Summary.plt``.  The parser follows quoted variable names instead of
    comma splitting because ``MOC_SL.plt`` writes ``"Massflow""J"`` without
    a separating comma.
    """
    p = Path(path)
    columns: tuple[str, ...] | None = None
    title: str | None = None
    zones: list[TecplotZone] = []
    zone_name: str | None = None
    zone_metadata: dict = {}
    zone_rows: list[list[float]] = []

    def flush_zone() -> None:
        nonlocal zone_name, zone_metadata, zone_rows
        if zone_name is None:
            return
        if columns is None:
            raise ValueError(f"Tecplot zone before VARIABLES in {p}")
        data = np.asarray(zone_rows, dtype=float)
        if data.size == 0:
            data = np.empty((0, len(columns)), dtype=float)
        elif data.shape[1] != len(columns):
            raise ValueError(
                f"{p} zone {zone_name!r} has width {data.shape[1]}, "
                f"expected {len(columns)}"
            )
        zones.append(TecplotZone(
            name=zone_name,
            columns=columns,
            data=data,
            metadata=dict(zone_metadata),
        ))
        zone_name = None
        zone_metadata = {}
        zone_rows = []

    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("VARIABLES"):
            columns = _parse_tecplot_variables(line)
            continue
        if upper.startswith("TITLE"):
            quoted = _QUOTED_VALUE_RE.findall(line)
            title = quoted[0] if quoted else None
            continue
        if upper.startswith("TEXT"):
            continue
        if upper.startswith("ZONE"):
            flush_zone()
            zone_name, zone_metadata = _parse_zone_line(line)
            zone_rows = []
            continue
        if columns is None:
            continue
        parts = line.split()
        try:
            row = [float(part) for part in parts]
        except ValueError:
            continue
        if len(row) != len(columns):
            raise ValueError(
                f"{p} numeric row has width {len(row)}, expected {len(columns)}: {line!r}"
            )
        zone_rows.append(row)

    flush_zone()
    if columns is None:
        raise ValueError(f"No Tecplot VARIABLES found in {p}")
    if not zones:
        raise ValueError(f"No Tecplot zones found in {p}")
    return TecplotFile(
        columns=columns,
        zones=tuple(zones),
        title=title,
        provenance={"path": str(p), "zones": len(zones)},
    )


def parse_wall_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``wall.out``."""
    return _coerce_columns(_read_numeric_table(path), WALL_COLUMNS)


def parse_center_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``center.out``."""
    return _coerce_columns(
        _read_numeric_table(path, skip_prefixes=("Centerline",)),
        CENTER_COLUMNS,
    )


def parse_kernel_out(path: str | Path) -> LegacyTable:
    """Parse NASA kernel files such as ``TT'BF_Kernel.out``."""
    table = _read_numeric_table(path)
    if table.data.shape[1] == len(KERNEL_COLUMNS):
        return _coerce_columns(table, KERNEL_COLUMNS)
    if table.data.shape[1] == len(KERNEL_WITH_MASS_COLUMNS):
        return _coerce_columns(table, KERNEL_WITH_MASS_COLUMNS)
    if table.data.shape[1] == len(UNCROPPED_KERNEL_COLUMNS):
        return _coerce_columns(table, UNCROPPED_KERNEL_COLUMNS)
    return table


def parse_last_kernel_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``LastKernel.out``."""
    return _coerce_columns(_read_numeric_table(path), KERNEL_WITH_MASS_COLUMNS)


def parse_uncropped_kernel_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``UncroppedKernel.out``."""
    return _coerce_columns(_read_numeric_table(path), UNCROPPED_KERNEL_COLUMNS)


def parse_tt_prime_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``TT'.out`` initial throat line."""
    return _coerce_columns(_read_numeric_table(path), TT_PRIME_COLUMNS)


def parse_theta_b_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``ThetaB.out`` secant trace."""
    return _coerce_columns(_read_numeric_table(path), THETA_B_COLUMNS)


def parse_axis_i_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``axis_i.out`` initial-kernel axis trace."""
    return _coerce_columns(_read_numeric_table(path), POINT_TRACE_COLUMNS)


def parse_wall_i_out(path: str | Path) -> LegacyTable:
    """Parse NASA ``wall_i.out`` initial-kernel wall trace."""
    return _coerce_columns(_read_numeric_table(path), POINT_TRACE_COLUMNS)


def parse_moc_grid_plt(path: str | Path) -> TecplotFile:
    """Parse NASA ``MOC_Grid.plt`` RRC contour zones."""
    return parse_tecplot_file(path)


def parse_moc_sl_plt(path: str | Path) -> TecplotFile:
    """Parse NASA ``MOC_SL.plt`` streamline zones."""
    return parse_tecplot_file(path)


def parse_summary_plt(path: str | Path) -> TecplotFile:
    """Parse NASA ``Summary.plt`` primary contour zones."""
    return parse_tecplot_file(path)


def parse_rao_dat(path: str | Path) -> LegacyTable:
    """Parse NASA ``rao.dat`` three-column contour output."""
    table = _read_numeric_table(path)
    return LegacyTable(
        columns=RAO_COLUMNS,
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


NASA_OUTPUT_PARSERS: dict[str, Callable[[str | Path], object]] = {
    "wall.out": parse_wall_out,
    "center.out": parse_center_out,
    "rao.dat": parse_rao_dat,
    "summary.out": parse_summary_out,
    "moc_grid.plt": parse_moc_grid_plt,
    "moc_sl.plt": parse_moc_sl_plt,
    "summary.plt": parse_summary_plt,
    "thetab.out": parse_theta_b_out,
    "axis_i.out": parse_axis_i_out,
    "wall_i.out": parse_wall_i_out,
    "tt'.out": parse_tt_prime_out,
    "tt'bf_kernel.out": parse_kernel_out,
    "bfe_kernel.out": parse_kernel_out,
    "lastkernel.out": parse_last_kernel_out,
    "uncroppedkernel.out": parse_uncropped_kernel_out,
}


def parse_nasa_output(path: str | Path) -> object:
    """Parse any known file from ``MOC_Grid_BDE/outputs_M3.5Perf``."""
    p = Path(path)
    parser = NASA_OUTPUT_PARSERS.get(p.name.lower())
    if parser is None:
        raise ValueError(f"No NASA/JHU parser registered for {p.name}")
    return parser(p)
