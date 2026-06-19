"""SP-125 longitudinal inelastic buckling (eq 4-28/4-29) + the distinct
external-pressure rib-supported-liner screen, and the wall-arc-length heat
load.

eq 4-29 (``S_c = 4 E_t E_c t / [(√E_t+√E_c)² √(3(1−ν²)) r]``) reduces to the
classic elastic cylinder axial-buckling stress ``E t / (r √(3(1−ν²)))`` when
``E_t = E_c = E``; the tangent moduli supply the inelastic knockdown.  The
rib screen is a SEPARATE plate-buckling check under the ``|Δp|R/t``
compression — not eq 4-29.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import (
    bartz_heat_flux,
    regenerative_cooling_analysis,
    rib_supported_liner_buckling_profile,
    sp125_inelastic_buckling_critical_stress,
)
from raosim.propellants import custom_propellant
from raosim.regen_profile import helix_passage_lengths


# --------------------------------------------------------------------- #
#  eq 4-29 longitudinal inelastic buckling
# --------------------------------------------------------------------- #
def test_eq429_reduces_to_elastic_cylinder_buckling():
    E, t, r, nu = 140e9, 1.0e-3, 0.05, 0.33
    Sc = float(sp125_inelastic_buckling_critical_stress(
        wall_thickness=t, local_radius=r, tangent_modulus_tension=E,
        tangent_modulus_compression=E, poisson_ratio=nu))
    expected = E * t / (r * math.sqrt(3.0 * (1.0 - nu * nu)))
    assert Sc == pytest.approx(expected, rel=1e-9)


def test_eq429_inelastic_knockdown_lowers_critical_stress():
    kw = dict(wall_thickness=1.0e-3, local_radius=0.05, poisson_ratio=0.33)
    elastic = float(sp125_inelastic_buckling_critical_stress(
        tangent_modulus_tension=140e9, tangent_modulus_compression=140e9, **kw))
    inelastic = float(sp125_inelastic_buckling_critical_stress(
        tangent_modulus_tension=14e9, tangent_modulus_compression=14e9, **kw))
    assert inelastic < elastic
    # With E_t = E_c the equivalent modulus is just E_t, so S_c scales linearly.
    assert inelastic == pytest.approx(0.1 * elastic, rel=1e-9)


def test_eq429_scales_with_thickness_and_inverse_radius():
    kw = dict(tangent_modulus_tension=140e9, tangent_modulus_compression=140e9,
              poisson_ratio=0.33)
    base = float(sp125_inelastic_buckling_critical_stress(
        wall_thickness=1.0e-3, local_radius=0.05, **kw))
    thinner = float(sp125_inelastic_buckling_critical_stress(
        wall_thickness=0.5e-3, local_radius=0.05, **kw))
    bigger_r = float(sp125_inelastic_buckling_critical_stress(
        wall_thickness=1.0e-3, local_radius=0.10, **kw))
    assert thinner == pytest.approx(0.5 * base, rel=1e-9)    # S_c ∝ t
    assert bigger_r == pytest.approx(0.5 * base, rel=1e-9)   # S_c ∝ 1/r


# --------------------------------------------------------------------- #
#  external-pressure rib-supported-liner screen (distinct from eq 4-29)
# --------------------------------------------------------------------- #
def _rib(**over):
    n = 5  # it is a per-station profile function → pass arrays
    arr = ("pressure_differential", "inner_radius", "wall_thickness", "unsupported_span")
    kw = dict(pressure_differential=2.0e6, inner_radius=0.05, wall_thickness=1.0e-3,
              unsupported_span=1.0e-3, elastic_modulus=140e9, poisson_ratio=0.33)
    kw.update(over)
    kw = {k: (np.full(n, v) if k in arr else v) for k, v in kw.items()}
    return rib_supported_liner_buckling_profile(**kw)


def test_rib_screen_is_a_distinct_model():
    out = _rib()
    assert "rib_supported_liner_plate_buckling" in out["model"]
    assert "eq_4_29" not in out["model"]            # NOT the longitudinal buckling
    assert out["margin"] > 0.0


def test_rib_screen_margin_falls_with_span_and_pressure():
    base = _rib()
    wider = _rib(unsupported_span=2.0e-3)            # σ_cr ∝ (t/b)² ⇒ worse
    hotter = _rib(pressure_differential=4.0e6)       # more compression ⇒ worse
    thinner = _rib(wall_thickness=0.5e-3)            # less σ_cr AND more σ ⇒ worse
    assert wider["margin"] < base["margin"]
    assert hotter["margin"] < base["margin"]
    assert thinner["margin"] < base["margin"]


# --------------------------------------------------------------------- #
#  heat load follows the wall arc length, not the axial spacing
# --------------------------------------------------------------------- #
def test_total_heat_uses_wall_arc_length():
    prop = custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)
    contour = bell_nozzle_contour(Rt=0.05, epsilon=10.0, gamma=1.24, length_pct=80.0)
    spec = SimpleNamespace(
        method="regenerative", coolant="rp1", channel_count=120,
        channel_width=0.0012, channel_height=0.003, coolant_mass_flow=12.0,
        coolant_cp=None, coolant_inlet_temperature=300.0,
        max_wall_temperature=1000.0, coolant_density=None,
        coolant_viscosity=None, coolant_conductivity=None)
    mat = SimpleNamespace(conductivity=285.0)
    hf = bartz_heat_flux(contour, 7.0e6, prop, wall_temperature=900.0)
    res = regenerative_cooling_analysis(hf, contour, spec, mat, 0.001, prop, 7.0e6)

    x = np.asarray(res["x"]); y = np.asarray(contour["y"]); q = np.asarray(res["q"])
    _, ds_wall = helix_passage_lengths(x, y)
    arc = float(np.sum(q * 2.0 * np.pi * np.maximum(y, 1e-9) * ds_wall))
    axial = float(np.trapezoid(q * 2.0 * np.pi * np.maximum(y, 1e-9), x))
    # The reported total heat integrates over the wall ARC LENGTH ...
    assert res["total_heat_load"] == pytest.approx(arc, rel=1e-6)
    # ... which exceeds the axial-spacing integral on a sloped/curved wall
    # (so the old axial form under-predicted the heat the coolant must absorb).
    assert arc > axial * 1.02
