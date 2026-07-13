import pytest

from raosim.trajectory import simulate_stage


def test_actual_delta_v_is_burnout_velocity_change():
    initial_velocity = 125.0
    result = simulate_stage(
        m0=100.0,
        m_prop=10.0,
        thrust=5_000.0,
        Isp=250.0,
        Cd=0.0,
        A_ref=0.1,
        v0=initial_velocity,
        dt=0.01,
        coast_to_apogee=False,
    )

    assert result.actual_dv == pytest.approx(
        result.burnout_vel - initial_velocity
    )
    assert result.actual_dv == pytest.approx(
        result.ideal_dv - result.gravity_loss - result.drag_loss,
        rel=2e-2,
    )


def test_coast_does_not_get_counted_as_propulsive_loss():
    kwargs = dict(
        m0=100.0,
        m_prop=10.0,
        thrust=5_000.0,
        Isp=250.0,
        Cd=0.2,
        A_ref=0.1,
        v0=0.0,
        dt=0.03,
    )
    burnout = simulate_stage(**kwargs, coast_to_apogee=False)
    apogee = simulate_stage(**kwargs, coast_to_apogee=True)

    assert apogee.gravity_loss == pytest.approx(burnout.gravity_loss)
    assert apogee.drag_loss == pytest.approx(burnout.drag_loss)
    assert apogee.actual_dv == pytest.approx(burnout.actual_dv)
    assert apogee.final_mass == pytest.approx(90.0, abs=1e-10)
