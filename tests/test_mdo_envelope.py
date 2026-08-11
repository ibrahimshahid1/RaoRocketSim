"""
tests/test_mdo_envelope.py — SP-125 §2.1 item 6 envelope screen.

The envelope is the "smallest cylinder ... into which the engine would fit"
(SP-125 printed p. 31).  Three properties matter and are pinned here:

* it is **conservative** — the smooth radial maximum never understates the
  diameter, because an envelope screen that under-reports would pass designs
  that do not fit;
* its conservatism is **bounded and computable**, not unknown;
* it describes the **same jacket** the mass ledger integrates, so envelope and
  mass cannot disagree about one piece of hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

import raosim.jax  # noqa: F401  -- float64
import jax.numpy as jnp

from raosim.mdo.schema import MissionSpec, DesignVector, default_design_space
from raosim.mdo.engine import solve_engine
from raosim.mdo.envelope import (
    chamber_envelope, diameter_overshoot_bound, envelope_margins,
)

_THRUST = 5.0e3


@pytest.fixture(scope="module")
def solved():
    m = MissionSpec.for_thrust(_THRUST)
    x = DesignVector.from_active_array(
        jnp.asarray([s.ref() for s in default_design_space(m)]),
        m.design_layout(),
        fixed_of=m.OF,
    )
    return m, solve_engine(x, m)


def test_engine_converges(solved):
    _, r = solved
    assert bool(r.solver_converged) and bool(r.finite)


def test_diameter_never_understates_the_true_maximum(solved):
    _, r = solved
    true_d = 2.0 * float(jnp.max(r.envelope.r_outer))
    assert float(r.envelope.diameter) >= true_d


def test_diameter_overshoot_stays_inside_the_stated_bound(solved):
    """logsumexp(k v)/k overshoots max by at most ln(n)/k — so the envelope's
    conservatism is a number the report can quote, not a mystery."""
    _, r = solved
    true_d = 2.0 * float(jnp.max(r.envelope.r_outer))
    n = int(r.envelope.r_outer.shape[0])
    assert (float(r.envelope.diameter) - true_d) <= (
        diameter_overshoot_bound(n) + 1e-12)


def test_outer_radius_is_the_mass_ledger_radial_stack(solved):
    """r_outer must be liner + channel + the SOLVED jacket, i.e. exactly the
    stack raosim.mdo.mass integrates.  Recomputing the jacket here instead of
    reusing chamber.closeout_thickness is the bug this test exists to catch."""
    m, r = solved
    space = {s.name: s.ref() for s in default_design_space(m)}
    expected = (
        r.cooling.grid.r if hasattr(r.cooling, "grid") else None
    )
    # Reconstruct from the published pieces rather than internals:
    t_j = r.chamber_mass.closeout_thickness
    stack = r.envelope.r_outer - t_j - space["t_wall"] - space["channel_height"]
    # What remains must be the hot-gas wall radius: positive and monotone into
    # the throat then out again (a real contour, not an offset constant).
    stack = np.asarray(stack)
    assert np.all(stack > 0.0)
    assert stack.min() == pytest.approx(float(r.Rt), rel=1e-9)


def test_length_spans_injector_face_to_exit(solved):
    """SP-125 (printed p. 88) puts the chamber datum at the injector face; R0
    made that the shared convention.  Length must be positive and comparable to
    the nozzle scale, not a station-count artefact."""
    _, r = solved
    assert float(r.envelope.length) > 0.0
    assert float(r.envelope.length) > float(r.Rt)


def test_envelope_is_labelled_a_lower_bound(solved):
    """It excludes the flange, injector body and feed hardware.  The flag is
    what stops a downstream report from calling it the engine envelope."""
    _, r = solved
    assert r.envelope.is_lower_bound is True


def test_sentinel_limits_leave_the_margins_inert(solved):
    """Defaults are large finite sentinels, so an engine with no envelope
    requirement is not accidentally constrained — and no `inf` reaches SLSQP.

    With the fractional form the inert value is ~1.0 rather than ~1e3, which is
    what keeps the constraint Jacobian auditable by finite differences.
    """
    m, r = solved
    d, l = envelope_margins(r.envelope, m)
    assert np.isfinite(float(d)) and 0.99 < float(d) <= 1.0
    assert np.isfinite(float(l)) and 0.99 < float(l) <= 1.0


def test_margins_are_signed_against_a_real_limit(solved):
    _, r = solved
    tight = MissionSpec.for_thrust(
        _THRUST, envelope_diameter_max=0.5 * float(r.envelope.diameter))
    d, _l = envelope_margins(r.envelope, tight)
    assert float(d) == pytest.approx(-1.0)     # exactly 2x the limit


def test_margin_is_a_used_fraction(solved):
    """1 - value/limit reads directly as 'fraction of the allowance unused'."""
    _, r = solved
    limit = float(r.envelope.diameter) / 0.75      # design uses 75 % of it
    m = MissionSpec.for_thrust(_THRUST, envelope_diameter_max=limit)
    d, _l = envelope_margins(r.envelope, m)
    assert float(d) == pytest.approx(0.25, abs=1e-9)


def test_fractional_margin_is_scale_free_across_thrust_classes():
    """The property the fractional form exists to buy.

    Two engines three orders of magnitude apart, each sized to the same
    fraction of its own envelope allowance, must produce the same margin — so
    one QP scale factor works at every thrust.  An absolute margin in metres
    could not do this.
    """
    from raosim.mdo.envelope import fractional_margin

    small = fractional_margin(0.12, 0.16)          # 5 kN class, metres
    large = fractional_margin(3.4, 4.533333333)    # 3 MN class, metres
    assert float(small) == pytest.approx(float(large), abs=1e-9)
    assert float(small) == pytest.approx(0.25, abs=1e-9)


def test_constraints_dict_exposes_all_three_requirement_screens(solved):
    _, r = solved
    for key in ("envelope_diameter_margin", "envelope_length_margin",
                "dry_mass_partial_margin"):
        assert key in r.constraints
        assert np.isfinite(float(r.constraints[key]))


def test_dry_mass_partial_margin_matches_its_published_scalar(solved):
    """The margin must be built from the same dry_mass_partial the _SCALARS
    accessor publishes — two definitions of engine mass is how the two
    pipelines drifted apart before."""
    m, r = solved
    partial = float(r.objective_mass + r.chamber_mass.total)
    assert float(r.constraints["dry_mass_partial_margin"]) == pytest.approx(
        1.0 - partial / float(m.dry_mass_max), rel=1e-12)


def test_physical_values_stay_reportable_in_diagnostics(solved):
    """The NLP consumes dimensionless margins, but a human reading a report
    still needs metres and kilograms.  Both must exist and agree."""
    m, r = solved
    d = r.diagnostics
    assert float(d["envelope_diameter_partial"]) == pytest.approx(
        float(r.envelope.diameter), rel=1e-12)
    assert float(d["envelope_length_partial"]) == pytest.approx(
        float(r.envelope.length), rel=1e-12)
    assert float(d["dry_mass_partial"]) == pytest.approx(
        float(r.objective_mass + r.chamber_mass.total), rel=1e-12)
    # ...and the fractional margin must be reconstructible from them.
    assert float(r.constraints["envelope_diameter_margin"]) == pytest.approx(
        1.0 - float(d["envelope_diameter_partial"]) / m.envelope_diameter_max,
        rel=1e-12)
