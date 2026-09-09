"""D26: hydraulic k_s is a length; legacy mission roughness remains k_s/D_h.

Exercise the actual cooling marches and handoff, rather than only checking
field names. Correlations and their existing transition differences are not
changed by this unit-contract remediation.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raosim import physics
from raosim.mdo.cooling import cooling_march
from raosim.mdo.grid import build_station_grid
from raosim.mdo.postprocess import to_design_input
from raosim.mdo.resolved_inputs import (
    crosscheck_design_input,
    resolve_channel_roughness_input,
    resolve_engine_inputs,
)
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import custom_propellant


WIDTHS = (0.0005, 0.0009, 0.0012)
HEIGHT = 0.002
KS = 20e-6
LEGACY_RELATIVE = 0.015
COOLANT_MDOT = 1.02


def _design(width):
    return DesignVector(
        Pc=3e6, eps=8.0, dp_f_frac=0.2, dp_o_frac=0.2,
        channel_width=width, channel_height=HEIGHT, film_frac=0.0,
    ).as_dict()


def _resolved(mission, design):
    return resolve_engine_inputs(
        design, mission, effective_of=mission.OF,
        of_source="mission_nominal", total_mass_flow=COOLANT_MDOT * (1 + mission.OF),
    )


@pytest.fixture(scope="module")
def grid():
    return build_station_grid(jnp.asarray(0.025), jnp.asarray(8.0), MissionSpec())


def _march(mission, grid, width):
    return cooling_march(
        jnp.full(grid.r.shape, 900.0), grid,
        Pc=3e6, gamma=1.24, Tc=3550.0, c_star_del=1750.0,
        mdot_cool=COOLANT_MDOT, mission=mission,
        channel_width=width, channel_height=HEIGHT,
    )


@pytest.mark.parametrize("basis", ["absolute", "legacy"])
@pytest.mark.parametrize("width", WIDTHS)
def test_nonzero_roughness_actual_march_and_handoff(grid, basis, width):
    mission = MissionSpec(
        n_channels=100, channel_width=0.0007, channel_height=0.003,
        **({"channel_hydraulic_roughness_m": KS} if basis == "absolute"
           else {"channel_roughness": LEGACY_RELATIVE}),
    )
    design = _design(width)
    dh = 2 * width * HEIGHT / (width + HEIGHT)
    expected_ks = KS if basis == "absolute" else LEGACY_RELATIVE * dh
    expected_relative = expected_ks / dh
    resolved = _resolved(mission, design)
    thermal = resolved.thermal
    assert thermal.channel_hydraulic_roughness_m == pytest.approx(expected_ks)
    assert thermal.channel_roughness_input_basis == (
        "absolute_hydraulic_m" if basis == "absolute" else "legacy_relative"
    )
    assert thermal.channel_roughness_legacy_relative == (
        None if basis == "absolute" else LEGACY_RELATIVE
    )
    # The exported design dimensions override the deliberately different
    # mission nominal dimensions when migrating the old relative value.
    di = to_design_input(
        design, mission, mdot_total=resolved.total_mass_flow,
    )
    assert di.cooling.channel_roughness == pytest.approx(expected_ks)
    assert crosscheck_design_input(resolved, di) == ()

    out = _march(mission, grid, width)
    np.testing.assert_allclose(out.hydraulic_diameter, dh)
    np.testing.assert_allclose(out.hydraulic_roughness_m, expected_ks)
    np.testing.assert_allclose(out.relative_roughness, expected_relative)
    velocity = COOLANT_MDOT / (mission.rho_cool * mission.n_channels * width * HEIGHT)
    re = mission.rho_cool * velocity * dh / mission.mu_cool
    assert re > 4000  # compare rough turbulent branches, outside transition
    expected_f = float(physics.darcy_friction_factor(re, expected_relative))
    np.testing.assert_allclose(out.darcy_factor, expected_f, rtol=1e-12)
    expected_dp = expected_f * np.sum(grid.dseg) / dh * 0.5 * mission.rho_cool * velocity**2
    assert float(out.dp_total) == pytest.approx(float(expected_dp), rel=1e-12)
    smooth = _march(replace(mission, channel_roughness=0.0,
                            channel_hydraulic_roughness_m=None), grid, width)
    assert float(out.dp_total) > float(smooth.dp_total) * 1.2

    # Run the traditional cooling path with the converted specification.
    # Its thermal properties/discretization differ; the unit invariant and
    # rough friction correlation, not total thermal/pressure parity, are tested.
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3550.0)
    contour = bell_nozzle_contour(Rt=0.025, epsilon=8.0, gamma=1.24, length_pct=80.0)
    heat = physics.bartz_heat_flux(contour, 3e6, prop, wall_temperature=900.0)
    traditional = physics.regenerative_cooling_analysis(
        heat, contour, di.cooling, di.material, design["t_wall"], prop, 3e6,
        n_iter=3, curvature_correction=False,
    )
    np.testing.assert_allclose(traditional["hydraulic_diameter_profile"], dh)
    np.testing.assert_allclose(traditional["relative_roughness_profile"], expected_relative)
    assert traditional["channel_roughness"] == pytest.approx(expected_ks)
    assert "swamee_jain" in traditional["pressure_drop_correlation"]
    assert np.isfinite(traditional["coolant_pressure_drop"])


def test_absolute_hydraulic_roughness_uses_local_diameter_in_traditional_profile():
    mission = MissionSpec(n_channels=100, channel_hydraulic_roughness_m=KS)
    di = to_design_input(_design(WIDTHS[0]), mission,
                         mdot_total=COOLANT_MDOT * (1 + mission.OF))
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3550.0)
    contour = bell_nozzle_contour(Rt=0.025, epsilon=8.0, gamma=1.24, length_pct=80.0)
    n = len(contour["x"])
    widths = np.linspace(WIDTHS[0], WIDTHS[-1], n)
    profile = SimpleNamespace(
        channel_width=widths, channel_height=np.full(n, HEIGHT),
        t_hot=np.full(n, 0.0008), channel_count=100,
        helix_turns=0.0, land_width=np.full(n, 0.0005),
    )
    heat = physics.bartz_heat_flux(contour, 3e6, prop, wall_temperature=900.0)
    out = physics.regenerative_cooling_analysis(
        heat, contour, di.cooling, di.material, 0.0008, prop, 3e6,
        n_iter=3, curvature_correction=False, wall_profile=profile,
    )
    dh = 2 * widths * HEIGHT / (widths + HEIGHT)
    np.testing.assert_allclose(out["relative_roughness_profile"], KS / dh)
    assert np.ptp(out["relative_roughness_profile"]) > 0.01
    assert np.isfinite(out["coolant_pressure_drop"])


def test_absolute_roughness_channel_size_is_jittable_and_differentiable(grid):
    mission = MissionSpec(n_channels=100, channel_hydraulic_roughness_m=KS)
    pressure_loss = jax.jit(lambda width: _march(mission, grid, width).dp_total)
    width, step = 0.0009, 1e-7
    ad = float(jax.grad(pressure_loss)(width))
    fd = float((pressure_loss(width + step) - pressure_loss(width - step)) / (2 * step))
    assert np.isfinite(ad) and ad < 0
    assert ad == pytest.approx(fd, rel=1e-6)


def test_smooth_zero_compatibility(grid):
    legacy = MissionSpec(n_channels=100)
    absolute = replace(legacy, channel_hydraulic_roughness_m=0.0)
    a, b = _march(legacy, grid, WIDTHS[1]), _march(absolute, grid, WIDTHS[1])
    np.testing.assert_array_equal(a.darcy_factor, b.darcy_factor)
    assert float(a.dp_total) == float(b.dp_total)
    assert np.count_nonzero(a.hydraulic_roughness_m) == 0
    assert np.count_nonzero(b.hydraulic_roughness_m) == 0


@pytest.mark.parametrize("field", ["channel_roughness", "channel_hydraulic_roughness_m"])
@pytest.mark.parametrize("value", [-1e-5, float("nan"), float("inf"), -float("inf")])
def test_invalid_roughness_is_rejected_at_input(field, value):
    with pytest.raises(ValueError, match="roughness"):
        MissionSpec(**{field: value})
    # The resolver also rejects a mission-like object bypassing the dataclass.
    raw = {"channel_roughness": 0.0, "channel_hydraulic_roughness_m": None, field: value}
    with pytest.raises(ValueError, match="roughness"):
        resolve_channel_roughness_input(SimpleNamespace(**raw))


def test_ambiguous_roughness_is_rejected_even_if_values_could_match():
    with pytest.raises(ValueError, match="roughness"):
        MissionSpec(channel_roughness=LEGACY_RELATIVE, channel_hydraulic_roughness_m=KS)
    with pytest.raises(ValueError, match="roughness"):
        resolve_channel_roughness_input(SimpleNamespace(
            channel_roughness=LEGACY_RELATIVE, channel_hydraulic_roughness_m=0.0,
        ))


def test_resolved_roughness_basis_and_actual_geometry_are_part_of_identity():
    relative_mission = MissionSpec(channel_roughness=LEGACY_RELATIVE)
    d = _design(WIDTHS[1])
    relative = _resolved(relative_mission, d)
    absolute = _resolved(MissionSpec(
        channel_hydraulic_roughness_m=relative.thermal.channel_hydraulic_roughness_m,
    ), d)
    assert relative.digest() != absolute.digest()
    assert "channel_roughness" not in relative.to_dict()["thermal"]
    with pytest.raises(ValueError, match="conversion"):
        replace(relative.thermal, channel_width=WIDTHS[0])
    with pytest.raises(ValueError, match="basis"):
        replace(absolute.thermal, channel_roughness_input_basis="Ra")
    with pytest.raises(ValueError, match="channel_width"):
        _resolved(relative_mission, {**d, "channel_width": 0.0})
