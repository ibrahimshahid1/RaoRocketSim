"""
tests/test_mdo_pintle_component.py — the pintle as an optimizable MDO component.

Gates (plan §12.1 rule 4): NumPy parity against the host post-mass ledger and
the host Son inverse; AD-vs-central-difference on every owned variable;
architecture applicability (plan rules 4 and 8); live, non-dead design
variables in the NLP; and the post-wall handoff to the injector CAD inputs.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp

from raosim import movable_pintle as host_movable
from raosim.mass_ledger import injector_mass_ledger
from raosim.mdo.constraints import (
    ENGINE_CONSTRAINT_SPECS,
    constraint_metadata,
)
from raosim.mdo.engine import chamber_surfaces_for, solve_engine
from raosim.mdo.injector import injector_readouts
from raosim.mdo.nlp import _OptimizationSpace
from raosim.mdo.pintle import (
    INERT_MARGIN,
    pintle_component,
    pintle_density,
)
from raosim.mdo.schema import DesignVector, MissionSpec, default_design_space

_SLOTTED = MissionSpec()
_MOVABLE = dataclasses.replace(
    MissionSpec(), pintle_architecture="son_movable",
    pintle_throttle_min_fraction=0.4,
)
_MDOT = 3.37
_PC, _CHI_F, _CHI_O, _RT = 3.0e6, 0.22, 0.18, 0.035


def _component(mission, D=0.020, t=1.2e-3, D_cg=0.012, Pc=_PC):
    of = mission.OF
    inj = injector_readouts(
        Pc=jnp.asarray(Pc), chi_f=jnp.asarray(_CHI_F),
        chi_o=jnp.asarray(_CHI_O), D_pintle=jnp.asarray(D),
        mdot_fuel=jnp.asarray(_MDOT / (1.0 + of)),
        mdot_ox=jnp.asarray(_MDOT * of / (1.0 + of)),
        mission=mission, t_post=jnp.asarray(t), D_cg=jnp.asarray(D_cg),
    )
    comp = pintle_component(
        D_pintle=jnp.asarray(D), t_post=jnp.asarray(t),
        D_cg=jnp.asarray(D_cg), Pc=jnp.asarray(Pc),
        chi_f=jnp.asarray(_CHI_F), Rt=jnp.asarray(_RT),
        injector=inj, mission=mission,
    )
    return inj, comp


# --------------------------------------------------------------------------- #
# NumPy parity                                                                 #
# --------------------------------------------------------------------------- #
def test_slotted_post_mass_matches_host_ledger():
    D, t = 0.020, 1.2e-3
    inj, comp = _component(_SLOTTED, D=D, t=t)
    w = float(inj.slot_width)
    layout = {
        "resolved": {
            "pintle_outer_radius_m": 0.5 * D,
            "pintle_inner_radius_m": 0.5 * D - t,
            "pintle_body_length_m": _SLOTTED.pintle_body_length_factor * D,
            "radial_exit_style": "slots",
            "slot_width_m": w,
            "slot_height_m": _SLOTTED.pintle_slot_aspect_ratio * w,
            "pintle_wall_thickness_m": t,
        },
        "hydraulic_basis": {
            "radial_opening_count": _SLOTTED.pintle_slot_count,
        },
    }
    material = SimpleNamespace(name="mdo closeout alloy",
                               density=pintle_density(_SLOTTED))
    ledger = injector_mass_ledger(layout, body_material=material)
    post = next(item for item in ledger.items
                if item.component == "pintle post")
    assert post.mass_kg is not None
    assert float(comp.post_mass) == pytest.approx(post.mass_kg, rel=1e-12)
    assert float(comp.rod_mass) == 0.0


@pytest.mark.parametrize("tip_angle", [0.0, 20.0])
def test_throttle_opening_matches_host_son_inverse(tip_angle):
    mission = dataclasses.replace(_MOVABLE, pintle_tip_angle_deg=tip_angle)
    D, t = 0.020, 1.2e-3
    inj, comp = _component(mission, D=D, t=t)
    f = mission.pintle_throttle_min_fraction
    for area, opening in (
        (float(inj.area_tip_branch), comp.opening_full),
        (float(inj.area_tip_branch) * np.sqrt(f), comp.opening_throttle_min),
    ):
        host = host_movable.opening_for_tip_area(
            area, post_diameter=D, post_thickness=t,
            tip_angle_deg=tip_angle,
        )
        assert float(opening) == pytest.approx(host, rel=1e-10)
    assert float(comp.min_opening_margin) == pytest.approx(
        float(comp.opening_throttle_min) - mission.pintle_min_opening
    )


@pytest.mark.parametrize("t", [4.0e-4, 1.0e-3, 2.5e-3, 5.0e-3, 7.9e-3])
def test_outer_radius_hoop_bounds_lame_bore_stress(t):
    """The Barlow outer-radius form is conservative for any wall thickness."""
    D = 0.020
    _, comp = _component(_SLOTTED, D=D, t=t)
    p = _PC * (1.0 + _CHI_F) - _SLOTTED.Pa
    ro, ri = 0.5 * D, 0.5 * D - t
    lame = p * (ro ** 2 + ri ** 2) / (ro ** 2 - ri ** 2)
    assert float(comp.post_hoop_stress) >= lame


# --------------------------------------------------------------------------- #
# AD vs central differences on the owned variables                            #
# --------------------------------------------------------------------------- #
def _outputs(mission, D, t, D_cg):
    _, comp = _component(mission, D=D, t=t, D_cg=D_cg)
    return jnp.stack([
        comp.mass, comp.post_stress_margin, comp.wall_fit_margin,
        comp.min_opening_margin, comp.center_gap_fit_margin,
        comp.module_fit_margin,
    ])


@pytest.mark.parametrize("mission", [_SLOTTED, _MOVABLE],
                         ids=["fixed_slotted", "son_movable"])
@pytest.mark.parametrize("index, x0", [(0, 0.020), (1, 1.2e-3), (2, 0.012)])
def test_ad_matches_central_difference(mission, index, x0):
    base = [0.020, 1.2e-3, 0.012]

    def fn(value):
        args = list(base)
        args[index] = value
        return _outputs(mission, *args)

    ad = np.asarray(jax.jacfwd(fn)(jnp.asarray(x0)))
    assert np.allclose(ad, np.asarray(jax.jacrev(fn)(jnp.asarray(x0))))
    h = 1.0e-6 * abs(x0)
    fd = (np.asarray(fn(jnp.asarray(x0 + h)))
          - np.asarray(fn(jnp.asarray(x0 - h)))) / (2.0 * h)
    scale = np.maximum(np.abs(fd), 1.0e-6 * np.abs(np.asarray(fn(x0))) + 1e-12)
    assert np.max(np.abs(ad - fd) / scale) < 1e-4, f"ad={ad} fd={fd}"


def test_mass_and_post_wall_are_live():
    """The variables reach the objective, so the optimum is no longer inert."""
    dmass_dD = jax.grad(lambda D: _outputs(_MOVABLE, D, 1.2e-3, 0.012)[0])
    dmass_dt = jax.grad(lambda t: _outputs(_MOVABLE, 0.020, t, 0.012)[0])
    assert float(dmass_dD(jnp.asarray(0.020))) > 0.0
    assert float(dmass_dt(jnp.asarray(1.2e-3))) > 0.0
    # The valve's centre gap trades the transition area against the bore fit.
    _, comp_small = _component(_MOVABLE, D_cg=0.011)
    _, comp_large = _component(_MOVABLE, D_cg=0.016)
    assert float(comp_large.center_gap_fit_margin) < float(
        comp_small.center_gap_fit_margin)


# --------------------------------------------------------------------------- #
# Architecture applicability (plan rules 4 and 8)                              #
# --------------------------------------------------------------------------- #
_SLOTTED_ONLY = {"blockage_lo_margin", "blockage_hi_margin"}
_MOVABLE_ONLY = {"pintle_transition_margin", "pintle_min_opening_margin",
                 "pintle_center_gap_fit_margin"}
_BOTH = {"pintle_post_stress_margin", "pintle_wall_fit_margin",
         "pintle_module_fit_margin"}


@pytest.mark.parametrize("mission, applies, inert", [
    (_SLOTTED, _SLOTTED_ONLY | _BOTH, _MOVABLE_ONLY),
    (_MOVABLE, _MOVABLE_ONLY | _BOTH, _SLOTTED_ONLY),
], ids=["fixed_slotted", "son_movable"])
def test_rows_follow_the_selected_architecture(mission, applies, inert):
    surfaces = chamber_surfaces_for(mission)
    applicable, _, required, _ = constraint_metadata(mission, surfaces)
    by_key = {spec.engine_key: i for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS)}
    for key in applies:
        assert applicable[by_key[key]] and required[by_key[key]], key
    for key in inert:
        assert not applicable[by_key[key]], key

    space = default_design_space(mission)
    x = DesignVector.from_active_array(
        jnp.asarray([s.ref() for s in space]), mission.design_layout(),
        fixed_of=mission.OF,
    )
    r = solve_engine(x, mission)
    for key in inert:
        assert float(r.constraints[key]) == INERT_MARGIN
    assert bool(r.finite)
    # The pintle is inside the partial dry-mass subtotal it is optimized in.
    assert float(r.dry_mass_partial_exact_mass) == pytest.approx(
        float(r.electric_package_exact_mass + r.chamber_mass.total
              + r.pintle.mass), rel=1e-14)
    assert float(r.mass_ledger["pintle_component"]) > 0.0


def test_slotted_nlp_pins_center_gap_and_frees_pintle_wall():
    slotted = _OptimizationSpace.for_mission(_SLOTTED)
    free = {spec.name for spec in slotted.free_specs}
    assert "t_post" in free and "D_pintle" in free
    assert "D_cg" not in free
    assert dict(slotted.fixed_values)["D_cg"] > _SLOTTED.pintle_rod_diameter
    with pytest.raises(ValueError, match="son_movable"):
        _OptimizationSpace.for_mission(_SLOTTED, bounds={"D_cg": (0.011, 0.015)})

    movable = _OptimizationSpace.for_mission(_MOVABLE)
    assert {"t_post", "D_cg", "D_pintle"} <= {
        spec.name for spec in movable.free_specs}


def test_reference_center_gap_fits_inside_reference_bore():
    for mission in (_MOVABLE, MissionSpec.for_thrust(500.0),
                    MissionSpec.for_thrust(130.0e3)):
        specs = {s.name: s for s in default_design_space(mission)}
        bore = specs["D_pintle"].ref() - 2.0 * specs["t_post"].ref()
        assert specs["D_cg"].lower <= specs["D_cg"].ref() <= specs["D_cg"].upper
        assert specs["D_cg"].ref() <= bore or (
            specs["D_cg"].ref() == specs["D_cg"].lower)


# --------------------------------------------------------------------------- #
# Contract and validation                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("overrides, match", [
    ({"pintle_architecture": "sleeve"}, "pintle_architecture"),
    ({"pintle_throttle_min_fraction": 0.0}, "throttle"),
    ({"pintle_throttle_min_fraction": 1.5}, "throttle"),
    ({"pintle_min_opening": -1.0e-4}, "pintle_min_opening"),
])
def test_mission_rejects_invalid_pintle_settings(overrides, match):
    with pytest.raises(ValueError, match=match):
        MissionSpec(**overrides)


def test_legacy_eleven_value_array_keeps_its_meaning():
    legacy = jnp.asarray([3.0e6, 8.0, 0.2, 0.2, 0.02, 3.0e4, 5e-4, 1.5e-3,
                          0.1, 8e-4, 2.4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        x = DesignVector.from_array(legacy)
    assert float(x.OF) == pytest.approx(2.4)
    assert float(x.t_post) == pytest.approx(1.5e-3)
    assert x.layout.of_is_variable
    assert x.to_contract_array().shape == (13,)


def test_post_wall_reaches_the_injector_cad_inputs():
    from raosim.mdo.postprocess import to_design_input

    space = default_design_space(_SLOTTED)
    design = {s.name: s.ref() for s in space}
    design["t_post"] = 7.5e-4
    design["OF"] = _SLOTTED.OF
    di = to_design_input(design, _SLOTTED, effective_of=_SLOTTED.OF)
    assert di.injector.mechanical.pintle_wall_thickness == pytest.approx(7.5e-4)
    assert di.injector.architecture == "fixed_discrete"


def test_movable_valve_refuses_the_slotted_cad_handoff():
    from raosim.mdo.postprocess import to_design_input

    design = {s.name: s.ref() for s in default_design_space(_MOVABLE)}
    design["OF"] = _MOVABLE.OF
    with pytest.raises(ValueError, match="son_movable"):
        to_design_input(design, _MOVABLE, effective_of=_MOVABLE.OF)


# --------------------------------------------------------------------------- #
# Solver robustness exposed by the pintle variables                           #
# --------------------------------------------------------------------------- #
def test_zero_length_barrel_segment_has_a_finite_derivative():
    """With the L* closure active the barrel length can be exactly 0, so two
    stations coincide.  The plain norm's reverse-mode derivative there is
    0 * inf = NaN, which reached the cooling IFT linear solve and aborted the
    default 13 kN NLP once the pintle variables moved its path onto it."""
    from raosim.mdo.grid import station_segment_lengths

    def total_length(L):
        x = jnp.asarray([0.0, 0.0, 0.0]) + jnp.asarray([0.0, 1.0, 2.0]) * L
        x = jnp.concatenate([x, jnp.asarray([0.5])])
        r = jnp.asarray([1.0, 1.0, 1.0, 1.2])
        return jnp.sum(station_segment_lengths(x, r))

    g = jax.grad(total_length)(jnp.asarray(0.0))
    assert np.isfinite(float(g))
    # Forward values are the plain Euclidean lengths, bit for bit.
    x = jnp.asarray([0.0, 0.3, 0.3, 1.1])
    r = jnp.asarray([1.0, 1.0, 1.0, 1.4])
    np.testing.assert_array_equal(
        np.asarray(station_segment_lengths(x, r)),
        np.asarray(jnp.sqrt(jnp.diff(x) ** 2 + jnp.diff(r) ** 2)),
    )
