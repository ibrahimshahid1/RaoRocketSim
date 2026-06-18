"""Variable hot-wall + jacket profile sizing (``size_wall_profile``) and
its use in the regen geometry — the manufacturable variable wall the
verdict flagged as missing over a single uniform ``t_hot``.

Grounded in SP-125 (``propulsion_texts/19710019929.pdf``): the inner-shell
eq. 4-31 combined stress sets the hot-wall band station by station, and the
OUTER shell carries the coolant-pressure hoop (jacket sizing).
"""
from __future__ import annotations

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import custom_propellant
from raosim.regen_geometry import generate_regen_nozzle
from raosim.regen_profile import RegenWallProfile
from raosim.thermal_design import size_wall_profile


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=0.08, epsilon=10.0, gamma=1.24, length_pct=80.0)


_KW = dict(channel_count=300, channel_width=0.001, channel_height=0.005,
           mixture_ratio=2.0, n_outer=3, n_iter=12)


def test_wall_profile_is_variable_not_uniform(prop, contour):
    r = size_wall_profile(contour, prop, 1.0e6, material="grcop-84",
                          thermal_margin=1.05, structural_fos=1.0,
                          t_hot_min=0.0005, t_hot_max=0.003, **_KW)
    p = r["profile"]
    assert isinstance(p, RegenWallProfile)
    assert not p.is_uniform                         # genuinely varies along x
    assert r["t_hot_max_mm"] > r["t_hot_min_mm"] + 0.05
    # Thicker toward the large-radius exit (pressure stress ∝ p_diff·R/t),
    # thinnest near the throat (held at/above the manufacturing floor).
    ti = int(np.argmin(np.asarray(contour["y"])))
    assert p.t_hot[-1] > p.t_hot[ti]
    assert r["t_hot_min_mm"] >= 0.5 - 1e-6          # honors the mfg floor


def test_wall_profile_meets_structural_and_thermal(prop, contour):
    r = size_wall_profile(contour, prop, 1.0e6, material="grcop-84",
                          thermal_margin=1.05, structural_fos=1.0,
                          t_hot_min=0.0005, t_hot_max=0.003, **_KW)
    # Sized to the structural lower bound, so the min margin sits at ~FoS.
    assert r["min_structural_margin"] >= 1.0 - 0.05
    assert r["min_jacket_margin"] >= 1.5 - 1e-3
    assert r["thermal_feasible"] is True
    assert r["peak_wall_T"] <= 1000.0               # GRCop-84 service limit
    assert r["feasible"] is True


def test_jacket_sized_from_coolant_hoop_scales_with_yield(prop, contour):
    """SP-125: the OUTER shell carries the coolant-pressure hoop, so
    ``t_jacket = p_co·R_j·FoS/S_y``.  A stronger jacket alloy (Inconel over
    a copper liner) therefore needs proportionally less thickness."""
    cu = size_wall_profile(contour, prop, 3.0e6, material="grcop-84",
                           jacket_fos=1.5, t_jacket_min=1e-9, **_KW)
    inc = size_wall_profile(contour, prop, 3.0e6, material="grcop-84",
                            jacket_material="inconel718",
                            jacket_fos=1.5, t_jacket_min=1e-9, **_KW)
    assert inc["jacket_material"] == "Inconel 718"
    assert cu["jacket_material"] == "GRCop-84"
    # The liner sizing is independent of the jacket alloy, so p_co and R_j
    # match → the jacket thickness ratio is the inverse yield ratio.
    ratio = cu["t_jacket_max_mm"] / inc["t_jacket_max_mm"]
    assert ratio == pytest.approx(1035.0 / 186.0, rel=0.2)


def test_higher_pc_thickens_the_jacket(prop, contour):
    lo = size_wall_profile(contour, prop, 1.0e6, material="grcop-84",
                           jacket_material="inconel718", t_jacket_min=1e-9, **_KW)
    hi = size_wall_profile(contour, prop, 4.0e6, material="grcop-84",
                           jacket_material="inconel718", t_jacket_min=1e-9, **_KW)
    assert hi["t_jacket_max_mm"] > lo["t_jacket_max_mm"]


def test_geometry_consumes_the_variable_profile(prop, contour, tmp_path):
    """generate_regen_nozzle, given the variable profile, builds a
    station-wise wall + an outer jacket visualization mesh."""
    r = size_wall_profile(contour, prop, 1.0e6, material="grcop-84",
                          jacket_material="inconel718",
                          channel_count=200, channel_width=0.001,
                          channel_height=0.005, mixture_ratio=2.0,
                          n_outer=3, n_iter=12)
    reg = generate_regen_nozzle(contour, None, 0.001, wall_profile=r["profile"],
                                stl_path=tmp_path / "regen.stl")
    s = reg["summary"]
    assert s["variable_profile"] is True
    assert s["t_hot_range_mm"][1] > s["t_hot_range_mm"][0]    # carried through
    assert "t_jacket_range_mm" in s
    assert reg.get("jacket_verts") is not None                # jacket surface built
    assert (tmp_path / "regen.stl").stat().st_size > 1000
    assert reg["summary"]["representation"] == (
        "visualization_surfaces_not_manufacturing_solid"
    )


def test_wall_profile_rejects_invalid_geometry(prop, contour):
    with pytest.raises(ValueError, match="channel count"):
        size_wall_profile(
            contour, prop, 1.0e6, material="grcop-84",
            channel_count=0, channel_width=0.001, channel_height=0.005,
        )
