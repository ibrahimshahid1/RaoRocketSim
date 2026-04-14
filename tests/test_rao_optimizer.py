import math

import numpy as np
import pytest

import raosim.rao_optimizer as ro


def _run_opt():
    return ro.optimize_wall(
        Rt=0.02,
        epsilon=10.0,
        gamma=1.4,
        length_pct=80.0,
        n_control=3,
        n_char=10,
        max_iter=120,
    )


@pytest.mark.skipif(not ro.SCIPY_AVAILABLE, reason='scipy unavailable')
def test_optimization_converged_and_constraints():
    opt = _run_opt()

    assert isinstance(opt['converged'], bool)
    assert opt['optimizer'] == 'scipy-SLSQP'

    wall = opt['wall']
    _, r, _ = wall.sample(80)
    assert np.all(np.diff(r) >= -1e-8)
    theta_knots = np.array([wall.theta(xi) for xi in wall.x_knots])
    assert np.all(np.diff(theta_knots) <= 1e-6)

    expected_re = math.sqrt(10.0) * 0.02
    expected_ln = 0.8 * ((expected_re - 0.02) / math.tan(math.radians(15.0)))
    assert wall.x_end == pytest.approx(expected_ln, rel=1e-10, abs=1e-10)
    assert wall.r(wall.x_end) == pytest.approx(expected_re, rel=1e-10, abs=1e-10)

    cps = opt['control_points']
    assert np.all(np.diff(cps) >= -1e-10)


@pytest.mark.skipif(not ro.SCIPY_AVAILABLE, reason='scipy unavailable')
def test_reproducibility():
    a = _run_opt()
    b = _run_opt()

    assert a['theta_n'] == pytest.approx(b['theta_n'], rel=1e-6, abs=1e-6)
    assert np.allclose(a['control_points'], b['control_points'], rtol=1e-6, atol=1e-8)


def test_nelder_mead_fallback(monkeypatch):
    monkeypatch.setattr(ro, 'SCIPY_AVAILABLE', False)
    opt = ro.optimize_wall(
        Rt=0.02,
        epsilon=8.0,
        gamma=1.4,
        length_pct=80.0,
        n_control=3,
        n_char=8,
        max_iter=10,
    )
    assert opt['optimizer'] == 'nelder-mead'
