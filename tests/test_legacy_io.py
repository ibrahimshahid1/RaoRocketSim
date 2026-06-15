from pathlib import Path

import pytest

from raosim.legacy_io import (
    parse_center_out,
    parse_kernel_out,
    parse_rao_dat,
    parse_summary_out,
    parse_wall_out,
)


NASA_OUT = (
    Path(__file__).resolve().parents[1]
    / "Three-Dimensional-Nozzle-Design-Code-master"
    / "MOC_Grid_BDE"
    / "outputs_M3.5Perf"
)


def test_parse_nasa_wall_out_sample():
    wall = parse_wall_out(NASA_OUT / "wall.out")

    assert wall.data.shape[0] > 100
    assert wall.column("X_over_Rstar")[0] == pytest.approx(0.0)
    assert wall.column("R_over_Rstar")[0] == pytest.approx(1.0)
    assert wall.column("mach")[0] == pytest.approx(1.17779)


def test_parse_nasa_kernel_and_rao_outputs():
    kernel = parse_kernel_out(NASA_OUT / "BFE_Kernel.out")
    rao = parse_rao_dat(NASA_OUT / "rao.dat")
    center = parse_center_out(NASA_OUT / "center.out")

    assert kernel.data.shape[1] == 8
    assert rao.columns == ("R_over_Rstar", "X_over_Rstar", "theta_deg")
    assert rao.data[0, 2] == pytest.approx(15.2196)
    assert center.column("R_over_Rstar")[0] == pytest.approx(0.0)


def test_parse_nasa_summary_initial_data():
    summary = parse_summary_out(NASA_OUT / "summary.out")

    assert summary.fields["Nozzle Type"] == "Perfect Nozzle"
    assert summary.fields["Gamma"] == "1.4"
    assert summary.initial_data is not None
    assert summary.initial_data.column("Mach")[0] == pytest.approx(1.17779)
