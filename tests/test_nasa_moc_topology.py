import math

import pytest

from raosim.nasa_moc import (
    calc_lrc_de,
    calc_mdot_bd,
    curve_mass_flux,
)
from raosim.rao_residuals import residual_left_mach_geometry
from raosim.rao_variational import (
    RaoSolverConfig,
    _initial_ce_from_kernel,
    _target_length,
)


def test_calc_mdot_bd_matches_curve_segment_flux():
    cfg = RaoSolverConfig(Rt=0.020, epsilon=10.0, n_control=8, n_kernel=8)
    _ce, kernel_bd, _topology, _kernel = _initial_ce_from_kernel(cfg)

    mdot, segment = calc_mdot_bd(kernel_bd, 0.5, cfg.gamma)

    assert mdot == pytest.approx(curve_mass_flux(segment, cfg.gamma), rel=1e-12)
    assert segment[0].x == pytest.approx(kernel_bd[0].x)


def test_calc_lrc_de_closes_d_to_e_and_left_mach_segments():
    cfg = RaoSolverConfig(
        Rt=0.020,
        epsilon=10.0,
        gamma=1.4,
        pa_over_p0=0.01,
        length_pct=80.0,
        n_control=8,
        n_kernel=8,
    )
    _ce, kernel_bd, _topology, _kernel = _initial_ce_from_kernel(cfg)

    topology = calc_lrc_de(
        kernel_bd,
        x_E=_target_length(cfg.Rt, cfg.epsilon, cfg.length_pct),
        r_E=math.sqrt(cfg.epsilon) * cfg.Rt,
        gamma=cfg.gamma,
        Rt=cfg.Rt,
        epsilon=cfg.epsilon,
        pa_over_p0=cfg.pa_over_p0,
        n_points=cfg.n_control,
    )

    # NASA-style DE starts at D and is integrated forward via the C+
    # derivative system until cumulative mass matches mass_BD; it does
    # not have to land at (L, Re) directly — the wall MOC march
    # downstream of the kernel reaches the exit.
    assert topology.DE[0].x == pytest.approx(topology.D.x)
    assert topology.DE[0].r == pytest.approx(topology.D.r)
    assert topology.DE[0].theta == pytest.approx(topology.D.theta)
    assert topology.mass_DE == pytest.approx(topology.mass_BD, rel=1e-6)
    # E lands strictly downstream of D along the C+ family.
    assert topology.E.r >= topology.D.r - 1e-9
    # Left-Mach geometry on each DE segment: by the NASA derivative system
    # `dx/dr = 1/tan(theta + mu)`, so `dr - dx*tan(theta + mu) = 0` to
    # the RK4 truncation tolerance.
    for p0, p1 in zip(topology.DE[:-1], topology.DE[1:]):
        assert abs(residual_left_mach_geometry(p0, p1)) < 5e-3
