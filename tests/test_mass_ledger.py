"""Tests for raosim.mass_ledger and its differentiable mirror raosim.mdo.mass.

The point of these tests is that hardware mass is *integrated geometry*, not a
correlation: every assertion below is checkable by hand from the shapes the CAD
exporters build.  Where a quantity cannot be resolved, the ledger must say so
with a reason rather than emit a zero.
"""

import math

import numpy as np
import pytest

from raosim.interface import resolve_bolted_interface_geometry
from raosim.mass_ledger import (
    combine_ledgers,
    flange_bolt_mass_ledger,
    injector_mass_ledger,
    thrust_chamber_mass_ledger,
)
from raosim.materials import get_material
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.regen_profile import RegenWallProfile


RT = 0.025
EPS = 8.0


@pytest.fixture(scope="module")
def contour():
    return bell_nozzle_contour(Rt=RT, epsilon=EPS, length_pct=80.0, n_pts=400)


@pytest.fixture(scope="module")
def profile(contour):
    return RegenWallProfile.uniform(
        contour,
        channel_count=60,
        channel_width=1.5e-3,
        channel_height=2.5e-3,
        t_hot=1.0e-3,
        t_jacket=2.0e-3,
    )


# --------------------------------------------------------------------------- #
# thrust chamber
# --------------------------------------------------------------------------- #
def test_liner_mass_matches_pappus_shell_integral(profile):
    """SP-125 eq. 8-32 shell mass uses the MID-SURFACE radius, not the bore."""

    led = thrust_chamber_mass_ledger(profile, liner_material="GRCop-84")
    liner = next(i for i in led.items if i.component == "hot-gas liner")
    rho = get_material("GRCop-84").density

    x, r, t = profile.x, profile.r_inner, profile.t_hot
    seg = np.hypot(np.diff(x), np.diff(r))
    w = np.empty(len(seg) + 1)
    w[0], w[-1] = 0.5 * seg[0], 0.5 * seg[-1]
    w[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    expected_v = float(np.sum(2.0 * np.pi * (r + 0.5 * t) * t * w))

    assert liner.volume_m3 == pytest.approx(expected_v, rel=1e-12)
    assert liner.mass_kg == pytest.approx(expected_v * rho, rel=1e-12)
    # The mid-surface form must exceed the bore-radius form it replaces.
    bore_v = float(np.sum(2.0 * np.pi * r * t * w))
    assert liner.volume_m3 > bore_v


def test_land_area_fraction_form_equals_discrete_rib_count(profile):
    """b/(b+w) x annulus and N*b*h must agree for radial ribs.

    They are not bit-identical on a curved wall: ``RegenWallProfile`` measures
    the rib pitch on the *normal*-offset mid-surface, while the annulus
    ``pi(r_o^2 - r_i^2)`` is taken on radial offsets from the contour.  On a
    converging or diverging section the normal offset tilts, so the two forms
    differ by a fraction of a percent.  The annulus form is the correct
    solid-of-revolution and is what the ledger reports; this test pins the
    agreement so a real error in either form cannot hide inside that gap.
    """

    led = thrust_chamber_mass_ledger(profile, liner_material="GRCop-84")
    land = next(i for i in led.items if i.component == "regen channel lands")

    x, r, h = profile.x, profile.r_inner, profile.channel_height
    b = profile.land_width
    seg = np.hypot(np.diff(x), np.diff(r))
    wt = np.empty(len(seg) + 1)
    wt[0], wt[-1] = 0.5 * seg[0], 0.5 * seg[-1]
    wt[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    discrete = float(np.sum(profile.channel_count * b * h * wt))

    assert land.volume_m3 == pytest.approx(discrete, rel=1e-2)
    assert land.volume_m3 > 0.0


def test_ledger_scales_linearly_with_density_and_allowance(profile):
    base = thrust_chamber_mass_ledger(profile, liner_material="GRCop-84")
    allowed = thrust_chamber_mass_ledger(
        profile, liner_material="GRCop-84", joint_allowance=1.05
    )
    assert allowed.total_mass == pytest.approx(1.05 * base.total_mass, rel=1e-12)

    steel = thrust_chamber_mass_ledger(profile, liner_material="Stainless 316L")
    ratio = get_material("Stainless 316L").density / get_material("GRCop-84").density
    assert steel.total_mass == pytest.approx(base.total_mass * ratio, rel=1e-12)


def test_two_material_wall_prices_liner_and_closeout_separately(profile):
    """A copper liner inside a superalloy jacket is the SP-8087 norm."""

    led = thrust_chamber_mass_ledger(
        profile, liner_material="NARloy-Z", closeout_material="Inconel 718"
    )
    liner = next(i for i in led.items if i.component == "hot-gas liner")
    close = next(i for i in led.items if "closeout" in i.component)
    assert liner.density_kg_m3 == pytest.approx(get_material("NARloy-Z").density)
    assert close.density_kg_m3 == pytest.approx(get_material("Inconel 718").density)
    assert led.complete


def test_missing_density_is_unavailable_not_zero(profile):
    """The whole point of the availability contract."""

    class _NoDensity:
        name = "mystery alloy"
        density = None

    led = thrust_chamber_mass_ledger(profile, liner_material=_NoDensity())
    assert not led.complete
    assert led.total_mass is None
    assert led.resolved_mass == 0.0
    assert led.unavailable_reason and "density" in led.unavailable_reason
    for item in led.items:
        assert item.mass_kg is None
        assert item.unavailable_reason


def test_quadrature_does_not_overcount_the_meridian(profile):
    """The nodal weights must sum to the arc length exactly.

    Summing ``hypot(gradient(x), gradient(r))`` -- the legacy private helper's
    quadrature -- gives each end node a full segment and over-counts by one
    grid interval, inflating every mass in proportion.
    """

    x, r = profile.x, profile.r_inner
    seg = np.hypot(np.diff(x), np.diff(r))
    led = thrust_chamber_mass_ledger(profile, liner_material="GRCop-84")
    reported = led.items[0].key_parameters["meridional_length_m"]
    assert reported == pytest.approx(float(np.sum(seg)), rel=1e-12)

    naive = float(np.sum(np.hypot(np.gradient(x), np.gradient(r))))
    assert naive > reported


# --------------------------------------------------------------------------- #
# bolted interface
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def resolution():
    return resolve_bolted_interface_geometry(
        chamber_radius=0.05,
        chamber_pressure=3.0e6,
        wall_thickness=1.0e-3,
        material_yield_strength=900e6,
    )


def test_flange_ring_is_an_annulus_less_its_bolt_holes(resolution):
    led = flange_bolt_mass_ledger(resolution, flange_material="Inconel 718")
    ring = next(i for i in led.items if i.component == "chamber flange ring")

    gross = 0.25 * math.pi * (
        resolution.flange_outer_diameter ** 2
        - resolution.chamber_outer_diameter ** 2
    ) * resolution.flange_length
    holes = (
        resolution.bolt_count * 0.25 * math.pi
        * resolution.bolt_hole_diameter ** 2 * resolution.flange_length
    )
    assert ring.volume_m3 == pytest.approx(gross - holes, rel=1e-12)
    assert 0.0 < ring.volume_m3 < gross


def test_fastener_uses_the_same_tensile_area_as_the_joint_screen(resolution):
    led = flange_bolt_mass_ledger(resolution, flange_material="Inconel 718")
    bolt = next(i for i in led.items if "bolt" in i.component)
    assert bolt.quantity == resolution.bolt_count
    assert bolt.status == "screening_sized"
    # raosim.interface uses 0.75 * pi d^2 / 4 and infers d = 0.9 * hole.
    d = 0.9 * resolution.bolt_hole_diameter
    assert bolt.key_parameters["bolt_diameter_m"] == pytest.approx(d)
    assert bolt.key_parameters["thread_tensile_area_m2"] == pytest.approx(
        0.75 * 0.25 * math.pi * d ** 2
    )
    # Envelope must exceed the bare shank (head + nut are real metal).
    shank = bolt.key_parameters["thread_tensile_area_m2"] * bolt.key_parameters[
        "grip_length_m"
    ]
    assert bolt.volume_m3 > shank


def test_unresolved_interface_reports_a_reason(resolution):
    led = flange_bolt_mass_ledger({}, flange_material="Inconel 718")
    assert not led.complete
    assert led.total_mass is None
    assert all(i.mass_kg is None for i in led.items)
    assert "bolted interface" in led.unavailable_reason


# --------------------------------------------------------------------------- #
# injector
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pintle_layout():
    from raosim.injector import (
        InjectorSpec,
        PintleGeometrySpec,
        PropellantFeedSpec,
        evaluate_pintle_injector,
    )
    from raosim.injector_cad import resolve_machined_pintle_layout

    fuel = PropellantFeedSpec(
        role="fuel", name="rp-1", inlet_temperature=298.0, density=810.0,
        viscosity=1.6e-3, surface_tension=2.3e-2, vapor_pressure=2.0e3,
    )
    ox = PropellantFeedSpec(
        role="oxidizer", name="lox", inlet_temperature=90.0, density=1140.0,
        viscosity=1.9e-4, surface_tension=1.3e-2, vapor_pressure=1.0e5,
    )
    spec = InjectorSpec(
        type="pintle", sizing="auto", fuel=fuel, oxidizer=ox,
        geometry=PintleGeometrySpec(
            pintle_diameter=0.02, slot_count=24, radial_exit_style="slots",
            radial_stream="fuel", deflector_angle=15.0, face_od=0.10,
        ),
    )
    mdot, mr = 1.8116689741766580, 2.6
    inj = evaluate_pintle_injector(
        spec, mdot_fuel=mdot / (1.0 + mr), mdot_oxidizer=mr * mdot / (1.0 + mr),
        Pc=7.0e6, mixture_ratio=mr, chamber_radius=0.035, chamber_length=0.13,
        gamma=1.2, Tc=3500.0, R_gas=350.0, fuel_name="rp-1",
        oxidizer_name="lox",
    )
    return resolve_machined_pintle_layout(inj, spec=spec)


def test_faceplate_subtracts_every_resolved_cutout(pintle_layout):
    led = injector_mass_ledger(pintle_layout, body_material="Inconel 718")
    face = next(i for i in led.items if "faceplate" in i.component)
    r = pintle_layout["resolved"]

    t = r["faceplate_thickness_m"]
    gross = math.pi * r["faceplate_radius_m"] ** 2 * t
    bore = math.pi * (r["sleeve_outer_radius_m"] + r["tolerance_m"]) ** 2 * t
    holes = (
        r["bolt_count"] * 0.25 * math.pi * r["bolt_hole_diameter_m"] ** 2 * t
    )
    manifolds = sum(
        v["manifold_volume_m3"] for v in pintle_layout["roles"].values()
    )
    kp = face.key_parameters
    expected = (
        gross
        + kp["added_inlet_boss_volume_m3"]
        - bore - holes - manifolds
        - kp["subtracted_port_volume_m3"]
        - kp["subtracted_transfer_volume_m3"]
        - kp["subtracted_seal_groove_volume_m3"]
        - kp["subtracted_igniter_port_volume_m3"]
    )
    assert face.volume_m3 == pytest.approx(expected, rel=1e-12)
    assert kp["subtracted_manifold_volume_m3"] == pytest.approx(manifolds)
    assert kp["manifold_volume_complete"] is True
    assert 0.0 < face.volume_m3 < gross


def test_pintle_post_is_a_tube_less_its_metering_openings(pintle_layout):
    led = injector_mass_ledger(pintle_layout, body_material="Inconel 718")
    post = next(i for i in led.items if i.component == "pintle post")
    r = pintle_layout["resolved"]
    tube = math.pi * (
        r["pintle_outer_radius_m"] ** 2 - r["pintle_inner_radius_m"] ** 2
    ) * r["pintle_body_length_m"]
    assert 0.0 < post.volume_m3 < tube
    assert post.key_parameters["removed_opening_volume_m3"] > 0.0
    assert post.key_parameters["radial_opening_count"] == int(
        pintle_layout["hydraulic_basis"]["radial_opening_count"]
    )


def test_injector_ledger_tracks_the_cad_layout(pintle_layout):
    led = injector_mass_ledger(pintle_layout, body_material="Inconel 718")
    assert led.complete and led.total_mass > 0.0
    assert "resolve_machined_pintle_layout" in led.provenance["geometry_source"]
    # Every priced part must be one the CAD builder actually cuts.
    assert {i.component for i in led.items} == {
        "faceplate / manifold body", "pintle post", "annulus sleeve",
    }


def test_layout_without_resolved_section_is_rejected():
    with pytest.raises(ValueError, match="resolve_machined_pintle_layout"):
        injector_mass_ledger({}, body_material="Inconel 718")


# --------------------------------------------------------------------------- #
# combination + availability propagation
# --------------------------------------------------------------------------- #
def test_combined_ledger_withholds_the_total_if_any_part_is_unknown(
    profile, resolution, pintle_layout
):
    good = combine_ledgers(
        [
            thrust_chamber_mass_ledger(profile, liner_material="GRCop-84"),
            flange_bolt_mass_ledger(resolution, flange_material="Inconel 718"),
            injector_mass_ledger(pintle_layout, body_material="Inconel 718"),
        ],
        scope="engine_hardware",
    )
    assert good.complete
    assert good.total_mass == pytest.approx(good.resolved_mass, rel=1e-12)
    assert set(good.by_subsystem()) == {
        "thrust_chamber", "chamber_interface", "injector",
    }

    partial = combine_ledgers(
        [
            thrust_chamber_mass_ledger(profile, liner_material="GRCop-84"),
            flange_bolt_mass_ledger({}, flange_material="Inconel 718"),
        ],
        scope="engine_hardware",
    )
    assert not partial.complete
    assert partial.total_mass is None
    # The partial rollup still exists but is explicitly labelled.
    assert partial.resolved_mass > 0.0
    assert partial.to_dict()["resolved_mass_is_partial"] is True
    assert partial.by_subsystem()["chamber_interface"] is None


# --------------------------------------------------------------------------- #
# differentiable mirror
# --------------------------------------------------------------------------- #
def test_mdo_chamber_mass_matches_a_numpy_re_derivation():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from raosim.mdo.grid import build_station_grid
    from raosim.mdo.mass import chamber_mass
    from raosim.mdo.schema import MissionSpec

    from dataclasses import replace

    # Pin the legacy fixed-ratio jacket so this test checks the SHELL
    # integrals; the hoop-sized jacket has its own test below.
    mission = replace(MissionSpec(), closeout_sizing="ratio", rho_closeout=None)
    grid = build_station_grid(jnp.asarray(0.0314), jnp.asarray(8.0), mission)
    t_w, w, h = 8.0e-4, 5.0e-4, 1.5e-3
    out = chamber_mass(
        grid, mission, t_wall=t_w, channel_width=w, channel_height=h
    )

    r = np.asarray(grid.r)
    seg = np.asarray(grid.dseg)
    ds = np.empty(len(seg) + 1)
    ds[0], ds[-1] = 0.5 * seg[0], 0.5 * seg[-1]
    ds[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    t_j = t_w * mission.closeout_thickness_ratio
    pitch = 2.0 * np.pi * (r + t_w + 0.5 * h) / mission.n_channels
    b = pitch - w
    lf = b / (b + w)
    liner = mission.rho_wall * np.sum(2 * np.pi * (r + 0.5 * t_w) * t_w * ds)
    land = mission.rho_wall * np.sum(
        np.pi * ((r + t_w + h) ** 2 - (r + t_w) ** 2) * lf * ds
    )
    close = mission.rho_wall * np.sum(
        2 * np.pi * (r + t_w + h + 0.5 * t_j) * t_j * ds
    )

    assert float(out.liner) == pytest.approx(liner, rel=1e-12)
    assert float(out.lands) == pytest.approx(land, rel=1e-12)
    assert float(out.closeout) == pytest.approx(close, rel=1e-12)
    assert float(out.total) == pytest.approx(liner + land + close, rel=1e-12)

    # Land area fraction and the discrete N*b*h rib form must agree.
    discrete = mission.rho_wall * np.sum(mission.n_channels * b * h * ds)
    assert float(out.lands) == pytest.approx(discrete, rel=2e-3)


def test_mdo_chamber_mass_is_differentiable_and_monotone():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from raosim.mdo.grid import build_station_grid
    from raosim.mdo.mass import chamber_mass
    from raosim.mdo.schema import MissionSpec

    from dataclasses import replace

    mission = replace(MissionSpec(), closeout_sizing="ratio", rho_closeout=None)

    def total(t_wall, w, h, Rt):
        grid = build_station_grid(Rt, jnp.asarray(8.0), mission)
        return chamber_mass(
            grid, mission, t_wall=t_wall, channel_width=w, channel_height=h
        ).total

    args = (
        jnp.asarray(8.0e-4), jnp.asarray(5.0e-4),
        jnp.asarray(1.5e-3), jnp.asarray(0.0314),
    )
    grads = jax.grad(total, argnums=(0, 1, 2, 3))(*args)
    assert all(np.isfinite(float(g)) for g in grads)

    # A thicker wall, a bigger throat and a taller channel all add metal;
    # a wider channel removes land metal.
    d_t, d_w, d_h, d_Rt = (float(g) for g in grads)
    assert d_t > 0.0 and d_h > 0.0 and d_Rt > 0.0
    assert d_w < 0.0

    # Finite-difference agreement on the wall-thickness sensitivity.
    eps = 1.0e-8
    fd = (
        float(total(args[0] + eps, *args[1:]))
        - float(total(args[0] - eps, *args[1:]))
    ) / (2.0 * eps)
    assert d_t == pytest.approx(fd, rel=1e-5)


# --------------------------------------------------------------------------- #
# structural jacket (R1): SP-125 p.109 outer-shell hoop screen
# --------------------------------------------------------------------------- #
def test_jacket_is_hoop_sized_and_tapered_not_a_thickness_ratio():
    """SP-125 p.109: the outer shell carries only the coolant hoop stress."""

    jnp = pytest.importorskip("jax.numpy")
    from dataclasses import replace

    from raosim.mdo.mass import closeout_thickness
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()
    r_outer = jnp.asarray([0.040, 0.060, 0.090])
    p_cool = jnp.asarray([4.0e6, 5.0e6, 5.2e6])
    t = np.asarray(closeout_thickness(r_outer, p_cool, mission))

    expected = (
        mission.closeout_structural_fos
        * np.asarray(p_cool) * np.asarray(r_outer)
        / mission.closeout_sigma_yield
    )
    floored = np.maximum(expected, mission.closeout_thickness_min)
    # The smooth floor is conservative: never below the hard max, and tight.
    assert np.all(t >= floored - 1e-12)
    assert np.all(t <= floored * 1.02 + 1e-9)
    # Tapered: thicker where p*r is larger.
    assert t[-1] > t[0]


def test_jacket_floor_binds_smoothly_and_stays_differentiable():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from raosim.mdo.mass import closeout_thickness
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()

    def t_of_p(p):
        return closeout_thickness(
            jnp.asarray(0.09), jnp.asarray(p), mission
        )

    # Sweep the coolant pressure straight through the point where the
    # manufacturing floor stops binding; the derivative must stay finite and
    # monotone rather than jumping.
    pressures = np.linspace(1.0e5, 2.0e7, 60)
    grads = np.array([float(jax.grad(t_of_p)(float(p))) for p in pressures])
    assert np.all(np.isfinite(grads))
    assert np.all(grads >= -1e-15)
    assert grads[-1] > grads[0]


def test_thin_shell_margin_rejects_a_jacket_outside_the_hoop_model():
    """SP-125 p.336 limits the membrane treatment to t/r <= ~1/15."""

    jnp = pytest.importorskip("jax.numpy")
    from dataclasses import replace

    from raosim.mdo.grid import build_station_grid
    from raosim.mdo.mass import chamber_mass
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()
    grid = build_station_grid(jnp.asarray(0.0314), jnp.asarray(8.0), mission)
    p_cool = jnp.full_like(grid.r, 5.0e6)
    kw = dict(t_wall=8.0e-4, channel_width=5.0e-4, channel_height=1.5e-3)

    strong = chamber_mass(grid, mission, coolant_pressure=p_cool, **kw)
    assert float(strong.closeout_thin_shell_margin) > 0.0

    # A soft jacket alloy needs a wall the thin-shell formula cannot describe.
    soft = replace(mission, closeout_sigma_yield=60.0e6)
    weak = chamber_mass(grid, soft, coolant_pressure=p_cool, **kw)
    assert float(weak.closeout_thin_shell_margin) < 0.0
    assert float(weak.closeout) > float(strong.closeout)


def test_hoop_sized_jacket_is_lighter_than_the_legacy_ratio_assumption():
    """The whole point of R1: the 2x-t_wall copper jacket was both
    structurally unjustified and heavier than a sized superalloy one."""

    jnp = pytest.importorskip("jax.numpy")
    from dataclasses import replace

    from raosim.mdo.grid import build_station_grid
    from raosim.mdo.mass import chamber_mass
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()
    grid = build_station_grid(jnp.asarray(0.0314), jnp.asarray(8.0), mission)
    p_cool = jnp.full_like(grid.r, 5.0e6)
    kw = dict(t_wall=8.0e-4, channel_width=5.0e-4, channel_height=1.5e-3)

    sized = chamber_mass(grid, mission, coolant_pressure=p_cool, **kw)
    legacy = chamber_mass(
        grid,
        replace(mission, closeout_sizing="ratio", rho_closeout=None),
        **kw,
    )
    assert float(sized.closeout) < float(legacy.closeout)
    # Liner and lands are untouched by the jacket change.
    assert float(sized.liner) == pytest.approx(float(legacy.liner), rel=1e-12)
    assert float(sized.lands) == pytest.approx(float(legacy.lands), rel=1e-12)


def test_jacket_mass_is_differentiable_through_the_hoop_screen():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from raosim.mdo.grid import build_station_grid
    from raosim.mdo.mass import chamber_mass
    from raosim.mdo.schema import MissionSpec

    mission = MissionSpec()

    def closeout(p_scale, Rt):
        grid = build_station_grid(Rt, jnp.asarray(8.0), mission)
        p_cool = jnp.full_like(grid.r, 1.0) * p_scale
        return chamber_mass(
            grid, mission, t_wall=8.0e-4, channel_width=5.0e-4,
            channel_height=1.5e-3, coolant_pressure=p_cool,
        ).closeout

    args = (jnp.asarray(1.5e7), jnp.asarray(0.0314))
    d_p, d_Rt = jax.grad(closeout, argnums=(0, 1))(*args)
    assert np.isfinite(float(d_p)) and np.isfinite(float(d_Rt))
    # Above the manufacturing floor a higher jacket pressure and a bigger
    # throat both demand more jacket metal.
    assert float(d_p) > 0.0
    assert float(d_Rt) > 0.0


def test_injector_ledger_accounts_for_every_cad_feature(pintle_layout):
    """The ledger must describe the solid the exporter actually writes.

    ``injector_cad._build_machined_faceplate`` fuses inlet bosses and cuts
    manifold pockets, inlet holes, side ports, transfer passages, bolt holes,
    an o-ring groove and the igniter port.  Before this test the ledger carried
    only the bore, bolt holes and manifolds, so it overstated the removed metal
    and ignored the added metal entirely.
    """

    led = injector_mass_ledger(pintle_layout, body_material="Inconel 718")
    face = next(i for i in led.items if "faceplate" in i.component)
    kp = face.key_parameters

    for key in (
        "added_inlet_boss_volume_m3",
        "subtracted_manifold_volume_m3",
        "subtracted_port_volume_m3",
        "subtracted_transfer_volume_m3",
        "subtracted_igniter_port_volume_m3",
    ):
        assert key in kp, key
        assert kp[key] >= 0.0

    # The bosses are real added metal, and the ports bored through them are
    # real removed metal; neither may be silently dropped.
    assert kp["added_inlet_boss_volume_m3"] > 0.0
    assert kp["subtracted_port_volume_m3"] > 0.0
    assert kp["subtracted_igniter_port_volume_m3"] > 0.0

    r = pintle_layout["resolved"]
    plain_disc = (
        math.pi * r["faceplate_radius_m"] ** 2 * r["faceplate_thickness_m"]
    )
    assert 0.0 < face.volume_m3 < plain_disc


def test_faceplate_ledger_matches_the_exported_solid(pintle_layout):
    """Cross-check the ledger against the CAD volume when CadQuery is present.

    This is the assertion that actually enforces "mass is integrated from the
    same geometry the CAD builds".  The ledger uses closed-form primitives
    while the exporter does real booleans, so overlaps at the boss/disc
    junction make them agree closely rather than exactly.
    """

    from raosim.injector_cad import cadquery_available

    if not cadquery_available():
        pytest.skip("CadQuery not installed")

    import cadquery as cq

    from raosim.injector_cad import _build_machined_faceplate

    solid = _build_machined_faceplate(cq, pintle_layout)
    cad_volume = float(
        sum(abs(s.Volume()) for v in solid.vals() for s in v.Solids())
    )
    led = injector_mass_ledger(pintle_layout, body_material="Inconel 718")
    face = next(i for i in led.items if "faceplate" in i.component)
    assert face.volume_m3 == pytest.approx(cad_volume, rel=0.10)
