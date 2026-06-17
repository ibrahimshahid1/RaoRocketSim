"""
3-D regen nozzle geometry (raosim.regen_geometry): wall surface of
revolution + the N cooling channels (axial or helical), STL/PNG export,
and optional wall-temperature colouring tied to the cooling analysis.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from raosim.design import CoolingSpec, MaterialSpec
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
from raosim.propellants import custom_propellant
from raosim.regen_geometry import (
    generate_regen_nozzle,
    nozzle_wall_surface,
    regen_channel_rails,
)


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=0.020, epsilon=10.0, gamma=1.24,
                               length_pct=80.0)


def _cool(N=80, w=0.0008, h=0.0025):
    return CoolingSpec(method="regenerative", coolant="rp1",
                       channel_count=N, channel_width=w, channel_height=h)


def test_wall_surface_is_revolution(contour):
    x = np.asarray(contour["x"]); r = np.asarray(contour["y"])
    verts = nozzle_wall_surface(x, r, n_theta=64)
    assert verts.shape == (len(x), 64, 3)
    # Every ring has radius r(x) about the x-axis.
    rad = np.hypot(verts[..., 1], verts[..., 2])
    np.testing.assert_allclose(
        rad, np.broadcast_to(np.abs(r)[:, None], rad.shape), atol=1e-9)


def test_axial_channels_fit_and_count(contour):
    res = generate_regen_nozzle(contour, _cool(80), 0.001, helix_turns=0.0)
    assert res["summary"]["channels_fit"] is True
    assert len(res["channel_rails"]) == 80
    # Each channel rail set is (n_x, 4 corners, 3).
    assert res["channel_rails"][0].shape[1:] == (4, 3)


def test_channels_seated_above_inner_wall(contour):
    """Channel floor sits at r_inner + t_wall (outside the gas wall)."""
    t_w = 0.0012
    rails = regen_channel_rails(
        np.asarray(contour["x"]), np.asarray(contour["y"]),
        n_channels=80, channel_width=0.0008, channel_height=0.0025,
        wall_thickness=t_w, helix_turns=0.0)
    r_floor = np.hypot(rails[0][:, 0, 1], rails[0][:, 0, 2])
    assert np.all(r_floor >= np.asarray(contour["y"]) + t_w - 1e-9)


def test_helix_turns_advance_angle(contour):
    """A helical channel's angular position advances by 2π·turns end to end."""
    rails = regen_channel_rails(
        np.asarray(contour["x"]), np.asarray(contour["y"]),
        n_channels=24, channel_width=0.0018, channel_height=0.003,
        wall_thickness=0.001, helix_turns=2.0)
    ch = rails[0]
    a0 = math.atan2(ch[0, 0, 2], ch[0, 0, 1])
    # Unwrap the angle along the channel and check the net winding.
    ang = np.unwrap(np.arctan2(ch[:, 0, 2], ch[:, 0, 1]))
    assert abs((ang[-1] - ang[0]) - 2.0 * math.pi * 2.0) < 0.3


def test_too_many_channels_flagged(contour):
    # 400 × 0.8 mm = 320 mm > 126 mm throat circumference.
    res = generate_regen_nozzle(contour, _cool(400), 0.001)
    assert res["summary"]["channels_fit"] is False


def test_stl_and_png_export(contour, tmp_path):
    stl = tmp_path / "n.stl"
    png = tmp_path / "n.png"
    res = generate_regen_nozzle(contour, _cool(40), 0.001,
                                stl_path=stl, png_path=png)
    assert stl.exists() and stl.stat().st_size > 84   # header + count + tris
    assert png.exists()
    assert res["n_triangles"] > 0


def test_thermal_colouring_runs(contour, tmp_path):
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    hf = bartz_heat_flux(contour, 7.0e6, prop, wall_temperature=900.0)
    cool_res = regenerative_cooling_analysis(
        hf, contour, _cool(80), MaterialSpec(conductivity=350.0),
        0.001, prop, 7.0e6)
    res = generate_regen_nozzle(contour, _cool(80), 0.001,
                                png_path=tmp_path / "t.png",
                                cooling_result=cool_res)
    assert (tmp_path / "t.png").exists()
    assert res["summary"]["channels_fit"]
