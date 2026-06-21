"""Phase 13 smoke tests (REWRITE_PLAN §12 / test plan row 13).

Headless (Agg) smoke coverage: every plot function must produce a figure
from representative inputs without a solver run.  plot_topology gets a
real Phase-12.6 topology (cheap: one fixed-end construction); the
solution-based plots get a minimal duck-typed solution.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")  # noqa: E402  (before pyplot import via raosim.plotting)

import numpy as np
import pytest

from raosim.moc import CharPoint, FlowNode
from raosim.gas_dynamics import mach_angle, prandtl_meyer
from raosim.plotting import (plot_characteristic_net, plot_flowfield_mach,
                             plot_net_diagnostics, plot_topology)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opens so the headless run never trips the
    matplotlib 'more than 20 figures' retention warning."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _cp(x, r, theta, M, gamma=1.4):
    nu = prandtl_meyer(max(M, 1.000001), gamma)
    return CharPoint(x=x, r=r, theta=theta, M=max(M, 1.000001), nu=nu,
                     mu=mach_angle(max(M, 1.000001)),
                     compat_plus=theta + nu, compat_minus=theta - nu)


class _Row:
    def __init__(self, pts):
        self._pts = pts

    def all_points(self):
        return self._pts


class _FakeSolution:
    """Duck-typed minimal RaoSolution for plotting smoke tests."""

    def __init__(self):
        x = np.linspace(0.0, 0.12, 30)
        r = 0.02 + 0.04 * np.sqrt(x / 0.12)
        self.wall_raw = np.column_stack([x, r])
        self.wall_export = self.wall_raw.copy()
        self.kernel_points = [_cp(0.0, ri, 0.0, 1.2)
                              for ri in np.linspace(0.0, 0.02, 6)]
        self.characteristic_net = [
            _Row([_cp(xi, ri, 0.1, 2.0)
                  for xi, ri in zip(np.linspace(0.01 * j, 0.05 + 0.01 * j, 8),
                                    np.linspace(0.0, 0.03, 8))])
            for j in range(4)
        ]

        class _CE:
            x = np.linspace(0.01, 0.12, 10)
            r = np.linspace(0.005, 0.063, 10)

        self.control_surface = _CE()


def test_plot_characteristic_net_smoke():
    fig = plot_characteristic_net(_FakeSolution(), geometry="raw")
    assert fig is not None
    assert fig.axes, "no axes produced"


def test_plot_characteristic_net_export_mode():
    fig = plot_characteristic_net(_FakeSolution(), geometry="export")
    assert fig is not None


def test_plot_flowfield_mach_smoke():
    fig = plot_flowfield_mach(_FakeSolution())
    assert fig is not None


def test_plot_topology_smoke_with_real_construction():
    from raosim.moc_topology import build_topology
    from raosim.nasa_moc import calc_bde_region, set_theta_b
    from raosim.rao_variational import _target_length

    Rt = 0.020
    Ln = _target_length(Rt, 10.0, 80.0)
    nasa_topo, kernel = set_theta_b(
        Rt, 10.0, 80.0, 1.4, 0.01,
        theta_b_init_deg=21.87, n_kernel=24, n_de_points=24,
        starting_line_method="kliegel_levine", L_target=Ln,
        Ru=1.5 * Rt, end_condition="fixed_end", max_iter=30,
    )
    topo = build_topology(kernel, nasa_topo, calc_bde_region(kernel, nasa_topo))
    fig = plot_topology(topo)
    assert fig is not None
    labels = {t.get_text() for t in fig.axes[0].get_legend().get_texts()}
    assert "streamline BE (bell wall)" in labels
    assert "BD (mass-flow curve)" in labels


def test_plot_topology_duck_typed_minimal():
    class _Mini:
        B = _cp(0.003, 0.0207, math.radians(25.5), 2.16)
        D = _cp(0.05, 0.012, math.radians(18.0), 3.4)
        E = _cp(0.129, 0.0632, math.radians(8.0), 3.9)
        theta_B = math.radians(25.5)

    fig = plot_topology(_Mini())
    assert fig is not None


# ---------------------------------------------------------------------
#  Phase 13 batch (§12.1 plots #1, #4, #5, #6, #7, #10)
# ---------------------------------------------------------------------


def test_plot_nozzle_geometry_smoke():
    from raosim.plotting import plot_nozzle_geometry

    fig = plot_nozzle_geometry(_FakeSolution())
    assert fig is not None


def test_plot_flowfield_pressure_and_theta_smoke():
    from raosim.plotting import plot_flowfield_pressure, plot_flowfield_theta

    sol = _FakeSolution()
    assert plot_flowfield_pressure(sol, 1.4) is not None
    assert plot_flowfield_theta(sol) is not None


def test_plot_wall_distributions_smoke():
    from raosim.plotting import plot_wall_distributions

    fig = plot_wall_distributions(_FakeSolution(), 1.4)
    assert fig is not None
    assert len(fig.axes) == 3


def test_plot_wall_distributions_requires_net():
    from raosim.plotting import plot_wall_distributions

    sol = _FakeSolution()
    sol.characteristic_net = []
    with pytest.raises(ValueError, match="characteristic_net"):
        plot_wall_distributions(sol, 1.4)


def test_plot_exit_plane_smoke():
    from raosim.plotting import plot_exit_plane

    fig = plot_exit_plane(_FakeSolution(), 1.4, x_band=0.5)
    assert fig is not None
    assert len(fig.axes) == 3


def test_plot_nasa_overlay_smoke():
    from pathlib import Path

    from raosim.plotting import plot_nasa_overlay

    nasa_dir = (Path(__file__).resolve().parent.parent
                / "Three-Dimensional-Nozzle-Design-Code-master"
                / "MOC_Grid_BDE" / "outputs_M3.5Perf")
    if not nasa_dir.exists():
        pytest.skip("NASA M3.5Perf reference outputs not present")
    fig = plot_nasa_overlay(_FakeSolution(), nasa_dir, Rt=0.02)
    assert fig is not None


# ---------------------------------------------------------------------
#  Spec plot #8 — plot_net_diagnostics (REWRITE_PLAN §12.1 / §12.5).
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def misconverged_solution():
    """The §12.5 gate fixture: a deliberately mis-converged run
    (max_nfev=0 initial-residual-only) with a real forward-audit net
    (evaluate_moc=True + the BDE seed wall).  NumPy-only path."""
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=8, n_kernel=24,
        max_nfev=0, evaluate_moc=True, wall_method="bde",
    )
    return solve_rao_bvp(cfg)


def test_plot_net_diagnostics_flags_bad_links(misconverged_solution,
                                              capsys):
    """§12.5 gate: at least one bad link is highlighted on the
    mis-converged run, and the offending indices go to stdout."""
    fig = plot_net_diagnostics(misconverged_solution)
    assert fig is not None
    diag = fig.net_diagnostics
    assert diag["n_links"] > 0
    assert len(diag["flagged"]) >= 1
    out = capsys.readouterr().out
    assert "flagged links" in out


def test_plot_net_diagnostics_ce_fallback():
    """Without a forward-audit net (evaluate_moc=False) the CE's own
    C+ chain provides the link set."""
    from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp

    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01,
        length_pct=80.0, n_control=8, n_kernel=24,
        max_nfev=0, evaluate_moc=False,
    )
    sol = solve_rao_bvp(cfg)
    fig = plot_net_diagnostics(sol)
    assert fig is not None
    assert fig.net_diagnostics["n_links"] >= sol.control_surface.r.size - 1


# --------------------------------------------------------------------- #
#  Thermal visualisation: the regen wall solution (heat flux + wall and
#  coolant temperatures) that the analysis computes but nothing plotted.
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cooling_case():
    """A real regen cooling solution on a small contour (no MOC solve)."""
    from raosim.design import CoolingSpec, MaterialSpec
    from raosim.nozzle_geometry import bell_nozzle_contour
    from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
    from raosim.propellants import custom_propellant

    contour = bell_nozzle_contour(Rt=0.040, epsilon=10.0, gamma=1.24,
                                  length_pct=80.0)
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    hf = bartz_heat_flux(contour, 5.0e6, prop, wall_temperature=900.0)
    spec = CoolingSpec(method="regenerative", coolant="rp1",
                       channel_count=120, channel_width=0.0015,
                       channel_height=0.003, coolant_mass_flow=6.0,
                       coolant_inlet_temperature=300.0)
    mat = MaterialSpec.from_catalog("grcop-84")
    cooling = regenerative_cooling_analysis(hf, contour, spec, mat,
                                            0.001, prop, 5.0e6)
    return contour, cooling, mat


def test_plot_cooling_profile_stacks_flux_and_temperatures(cooling_case):
    """Three stacked panels (flux / temperatures / contour) come straight
    off the real cooling return dict, and the RP-1 coking limit is drawn."""
    from raosim.plotting import plot_cooling_profile
    contour, cooling, mat = cooling_case
    fig = plot_cooling_profile(cooling, contour=contour,
                               max_wall_temperature=mat.max_temperature)
    assert fig is not None
    assert len(fig.axes) == 3
    assert "MW/m" in fig.axes[0].get_ylabel()
    assert "[K]" in fig.axes[1].get_ylabel()
    legend = fig.axes[1].get_legend()
    assert any("coking" in t.get_text() for t in legend.get_texts())


def test_plot_cooling_profile_without_contour_drops_a_panel(cooling_case):
    from raosim.plotting import plot_cooling_profile
    _contour, cooling, _mat = cooling_case
    fig = plot_cooling_profile(cooling)        # no contour → two panels
    assert len(fig.axes) == 2


def test_plot_wall_field_on_contour_paints_the_wall(cooling_case):
    """The field-on-contour hero image colours the wall by T_wg (and by q),
    and rejects a non-existent field key."""
    from raosim.plotting import plot_wall_field_on_contour
    contour, cooling, _mat = cooling_case
    assert plot_wall_field_on_contour(cooling, contour).axes
    assert plot_wall_field_on_contour(
        cooling, contour, field="convective_heat_flux").axes
    with pytest.raises(ValueError):
        plot_wall_field_on_contour(cooling, contour, field="not_a_key")


def test_plot_coolant_channel_march_temp_pressure_velocity(cooling_case):
    """Three coolant-side panels (temperature / pressure / velocity) with
    the 61 m/s recommendation drawn on the velocity axis."""
    from raosim.plotting import plot_coolant_channel_march
    _contour, cooling, _mat = cooling_case
    fig = plot_coolant_channel_march(cooling)
    assert len(fig.axes) == 3
    assert "bar" in fig.axes[1].get_ylabel()
    assert "m/s" in fig.axes[2].get_ylabel()
    vlegend = fig.axes[2].get_legend()
    assert any("61" in t.get_text() for t in vlegend.get_texts())


def _stress_profile_for(cooling, contour, mat):
    from raosim.physics import coaxial_shell_wall_stress_profile
    return coaxial_shell_wall_stress_profile(
        pressure_differential=cooling["liner_pressure_differential"],
        inner_radius=contour["y"], wall_thickness=0.001,
        heat_flux=cooling["q"], elastic_modulus=mat.elastic_modulus,
        thermal_expansion=mat.thermal_expansion, poisson_ratio=mat.poisson_ratio,
        conductivity=mat.conductivity, yield_strength=mat.yield_strength)


def test_structural_life_dashboard_gives_sourced_Nf(cooling_case):
    """The dashboard shows eq. 4-31 stress vs yield and a real sourced
    N_f(x) for GRCop-84 (the Lerch-Ellis total-strain-life fit)."""
    from raosim.plotting import plot_structural_life_dashboard
    contour, cooling, mat = cooling_case
    stress = _stress_profile_for(cooling, contour, mat)
    fig = plot_structural_life_dashboard(cooling, stress, material=mat,
                                         required_cycles=100)
    assert len(fig.axes) == 3
    assert "MPa" in fig.axes[0].get_ylabel()
    # Panel 3 carries a real N_f(x) line (GRCop-84 has sourced curves), so a
    # finite-valued log curve was plotted.
    nlines = fig.axes[2].get_lines()
    assert any(np.all(np.isfinite(ln.get_ydata())) and ln.get_ydata().size > 2
               for ln in nlines)


def test_structural_life_dashboard_without_fatigue_data_is_graceful(cooling_case):
    """A material with no sourced fatigue coefficients still renders the
    stress panels and just annotates the missing life."""
    from raosim.design import MaterialSpec
    from raosim.plotting import plot_structural_life_dashboard
    contour, cooling, _mat = cooling_case
    bare = MaterialSpec.from_catalog("ofhc")        # no fatigue curves
    stress = _stress_profile_for(cooling, contour, bare)
    fig = plot_structural_life_dashboard(cooling, stress, material=bare)
    assert len(fig.axes) == 3                        # renders, no exception


def test_plot_channel_cross_section_shows_land_hot_spot(cooling_case):
    """The 2-D cell map solves and renders, and the land hot spot is hotter
    than the channel centre (the spread the 1-D circuit averages away)."""
    from raosim.design import CoolingSpec
    from raosim.physics import bartz_heat_flux, wall_cross_section_field
    from raosim.plotting import plot_channel_cross_section
    from raosim.propellants import custom_propellant
    contour, cooling, mat = cooling_case

    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    hf = bartz_heat_flux(contour, 5.0e6, prop, wall_temperature=900.0)
    spec = CoolingSpec(method="regenerative", coolant="rp1", channel_count=120,
                       channel_width=0.0015, channel_height=0.003,
                       coolant_mass_flow=6.0, coolant_inlet_temperature=300.0)
    xs = wall_cross_section_field(cooling, hf, contour, spec, mat,
                                  0.001, prop, 5.0e6, station="peak")
    assert xs["T_land"] >= xs["T_channel"] - 1e-6      # land is the hot spot
    fig = plot_channel_cross_section(xs)
    assert fig.axes and "[K]" in fig.axes[-1].get_ylabel()  # colourbar in K


def test_plot_separation_on_contour_marks_detachment():
    """A high-ε nozzle is over-expanded at sea level, so the wall detachment
    point is marked and the legend names the separation."""
    from raosim.nozzle_geometry import bell_nozzle_contour
    from raosim.plotting import plot_separation_on_contour
    from raosim.propellants import custom_propellant
    contour = bell_nozzle_contour(Rt=0.05, epsilon=40.0, gamma=1.2,
                                  length_pct=80.0)
    prop = custom_propellant(gamma=1.2, Mw=0.022, Tc=3500.0)
    fig = plot_separation_on_contour(contour, 5.0e6, prop,
                                     ambient_pressures=[101325.0, 20000.0])
    assert fig.axes
    leg = fig.axes[0].get_legend()
    assert any("separat" in t.get_text() for t in leg.get_texts())
