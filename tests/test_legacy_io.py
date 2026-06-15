from pathlib import Path

import pytest

from raosim.legacy_io import (
    LegacyTable,
    SummaryReport,
    TecplotFile,
    parse_center_out,
    parse_kernel_out,
    parse_last_kernel_out,
    parse_moc_grid_plt,
    parse_moc_sl_plt,
    parse_nasa_output,
    parse_rao_dat,
    parse_summary_out,
    parse_summary_plt,
    parse_theta_b_out,
    parse_tt_prime_out,
    parse_uncropped_kernel_out,
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


def test_every_m35perf_reference_file_parses():
    parsed = {path.name: parse_nasa_output(path) for path in sorted(NASA_OUT.iterdir())}

    assert set(parsed) == {
        "BFE_Kernel.out",
        "LastKernel.out",
        "MOC_Grid.plt",
        "MOC_SL.plt",
        "Summary.plt",
        "TT'.out",
        "TT'BF_Kernel.out",
        "ThetaB.out",
        "UncroppedKernel.out",
        "axis_i.out",
        "center.out",
        "rao.dat",
        "summary.out",
        "wall.out",
        "wall_i.out",
    }
    for name, obj in parsed.items():
        if isinstance(obj, LegacyTable):
            assert obj.data.size > 0, name
            assert len(obj.columns) == obj.data.shape[1], name
        elif isinstance(obj, TecplotFile):
            assert obj.zones, name
            assert obj.data.size > 0, name
            for zone in obj.zones:
                assert len(zone.columns) == zone.data.shape[1], (name, zone.name)
        elif isinstance(obj, SummaryReport):
            assert obj.fields, name
            assert obj.initial_data is not None, name
        else:
            raise AssertionError(f"Unexpected parser result for {name}: {type(obj)!r}")


def test_parse_nasa_tecplot_zone_files():
    grid = parse_moc_grid_plt(NASA_OUT / "MOC_Grid.plt")
    streamlines = parse_moc_sl_plt(NASA_OUT / "MOC_SL.plt")
    summary = parse_summary_plt(NASA_OUT / "Summary.plt")

    assert grid.columns == ("X_over_R", "R_over_R", "Mach", "Theta", "Massflow", "I")
    assert len(grid.zones) == 158
    assert grid.zones[0].name == "J = 0"
    assert grid.zones[-1].name == "J = 157"

    assert streamlines.columns == (
        "X_over_R", "Y_over_R", "Z_over_R", "R_over_R", "Mach", "Pres",
        "Temp", "Rho", "Theta", "Gamma", "Massflow", "J",
    )
    assert len(streamlines.zones) >= 10
    assert streamlines.zones[0].metadata["I"] == 37

    assert [zone.name for zone in summary.zones] == [
        "Wall contour", "RRC BD", "LRC DE",
    ]


def test_parse_nasa_iteration_and_kernel_tables():
    theta_b = parse_theta_b_out(NASA_OUT / "ThetaB.out")
    last = parse_last_kernel_out(NASA_OUT / "LastKernel.out")
    uncropped = parse_uncropped_kernel_out(NASA_OUT / "UncroppedKernel.out")
    tt_prime = parse_tt_prime_out(NASA_OUT / "TT'.out")

    assert theta_b.column("ThetaB")[-1] == pytest.approx(15.2196)
    assert theta_b.column("paramErr")[-1] == pytest.approx(9.46895e-10)

    assert last.columns[-1] == "Massflow"
    assert last.column("j")[0] == pytest.approx(57.0)
    assert uncropped.columns == (
        "i", "j", "x_in", "r_in", "mach", "theta", "pres_psi",
        "temp_R", "Density_slug_over_ft3", "massflow",
    )
    assert tt_prime.columns == ("i", "X", "R", "MACH", "THETA", "massflow")
