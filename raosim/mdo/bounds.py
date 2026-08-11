"""mdo/bounds.py — design-variable bounds derived from the architecture.

Why this module exists
----------------------
``MissionSpec.scaled_design_space`` used to hard-code

    VariableSpec("Pc",  1.5e6, 6.0e6, 3.0e6)
    VariableSpec("eps", 3.0,   40.0,  8.0)

with a docstring admitting they "bracket the 13 kN LOX/RP-1 baseline".  Those
two boxes were then applied unchanged to LOX/LH2, N2O4/MMH and every thrust
class, which is exactly the defect
``docs/GENERALIZATION_PLAN_THRUST_PROPELLANT_FEED.md`` §2 is about.  Neither
limit is actually set by thrust or by kerosene:

* **Chamber-pressure guidance depends on the engine cycle.**  Parsley and
  Zhang (2004) describe different limiting or optimizing mechanisms for the
  common pump-fed cycles.  Their approximate optima and onset-of-limitation
  values are not universal hard-validity endpoints.
* **Expansion ratio is set by the model's own validity domain.**  The analytic
  TOP wall comes from the Rao/NASA chart, which is tabulated over a specific
  box; outside it ``rao_chart_domain_violation`` already declares the design
  inadmissible.

One source of truth
-------------------
Both bounds are taken from the *same* place as the constraint that screens
them.  ``eps`` is bounded by the chart grid that ``chart_domain_margin`` uses;
``OF`` (in ``MissionSpec.of_design_space``) is bounded by the CEA grid that
``property_domain_margin`` uses.  A bound and a constraint that disagree about
the same limit is how the two pipelines drifted apart before R0.

Sources
-------
Parsley, R. C. and Zhang, B., *Thermodynamic Power Cycles for Pump-Fed Liquid
Rocket Engines*, 2004, Chapter 18, DOI
10.2514/5.9781600866760.0621.0648 —
``propulsion_texts/fuel_pump_design/thermodynamic-power-cycles-for-pumpfed-liquid-rocket-engines-2004.pdf``.

Huzel & Huang, *Design of Liquid Propellant Rocket Engines*, NASA SP-125, 1967,
ch. V and ch. III — ``propulsion_texts/19710019929.pdf``.

NASA MSFC, *Liquid Propulsion: Propellant Feed System Design*, 2010 —
``propulsion_texts/fuel_pump_design/20100035254.pdf``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover
    from raosim.mdo.schema import MissionSpec

__all__ = [
    "ArchitecturePressureLimit",
    "ArchitecturePressureWindow",
    "PRESSURE_LIMITS",
    "chamber_pressure_bounds",
    "chamber_pressure_search_window",
    "chamber_pressure_hard_domain",
    "expansion_ratio_bounds",
]


class ArchitecturePressureLimit(NamedTuple):
    """Numerical search window for one feed architecture, with its basis.

    ``hard_validity`` is intentionally explicit.  A value can be sourced from
    literature and still be a performance optimum, approximate scaling limit,
    or initialization recommendation rather than a physical validity bound.
    """

    lower: float          # Pa
    upper: float          # Pa
    mechanism: str        # what sets the ceiling
    source: str           # where the number comes from
    literature: bool      # False => repository finding, not a published limit
    hard_validity: bool = False


# Clearer public spelling; retain the old type name for compatibility.
ArchitecturePressureWindow = ArchitecturePressureLimit


#: Search guidance is architecture-dependent, but these endpoints are not
#: universal hard domains.  Live component constraints and validated property
#: surfaces determine admissibility.
#:
#: Only ``electric_pump`` is implemented in ``raosim.mdo`` today; the rest are
#: carried so that adding a cycle picks up its own ceiling rather than
#: inheriting the electric-pump box.  ``raosim.requirements`` still refuses an
#: unimplemented architecture outright — this table does not enable anything on
#: its own.
PRESSURE_LIMITS: dict[str, ArchitecturePressureLimit] = {
    # SP-125 ch. V: "Tank pressures range from 100 to 400 psia" for
    # pressurized-gas-feed upper stages.  Chamber pressure is what survives the
    # injector and line drops: at SP-125's own 15-20 % injector stability rule
    # plus line/valve losses, roughly 0.6-0.7 of tank pressure.  The worked A-4
    # engine (Table 3-5) confirms the scale -- 110 psia injector-end, 100 psia
    # nozzle stagnation, from a 165 psia oxidizer tank.  400 psia x ~0.65 is
    # about 1.8 MPa.
    "pressure_fed": ArchitecturePressureLimit(
        lower=3.0e5, upper=1.8e6,
        mechanism="tank pressure, which drives tank mass",
        source="NASA SP-125 ch. V (tank 100-400 psia); Table 3-5 A-4 engine",
        literature=True,
    ),
    # No published thermodynamic ceiling: an electric feed has no turbine to
    # balance, so the limiter is battery mass and the chamber's own thermal
    # capability, both of which the optimiser already models.  The window below
    # is a REPOSITORY finding, not a literature limit: TEST_PROMPT.md records
    # Pc 3.0 MPa as thermally feasible and 7 MPa as infeasible for catalog
    # liners at the 13 kN baseline.  Labelled ``literature=False`` so it cannot
    # be quoted as sourced.
    "electric_pump": ArchitecturePressureLimit(
        lower=1.5e6, upper=6.0e6,
        mechanism="battery mass and liner thermal capability",
        source="repository finding (TEST_PROMPT.md), not a published limit",
        literature=False,
    ),
    # Parsley & Zhang 2004 sec. I: "Chamber pressure for a gas generator cycle is selected
    # to optimize total engine performance ... This performance optimum
    # generally occurs at 10-15 MPa of chamber pressure, depending on
    # propellant selection, with an overboard flow generally less than 4 % of
    # the total engine flow."
    "gas_generator": ArchitecturePressureLimit(
        lower=3.0e6, upper=15.0e6,
        mechanism="total-engine performance optimum including overboard flow",
        source=("Parsley & Zhang 2004, Chapter 18 sec. I, PDF p. 3, "
                "DOI 10.2514/5.9781600866760.0621.0648"),
        literature=True,
    ),
    # Parsley & Zhang 2004 sec. I: "The energy available for the expander cycle is limited
    # by the thrust chamber and nozzle heat transfer, which limits potential
    # chamber pressure to ~10 MPa."  Note also sec. II's propellant
    # restriction: expander fuels are "limited to hydrogen, methane, or
    # propane" -- that is a compatibility screen, not a bound, and belongs in
    # the architecture enumeration.
    "expander": ArchitecturePressureLimit(
        lower=2.0e6, upper=10.0e6,
        mechanism="thrust chamber and nozzle heat transfer available to the turbine",
        source=("Parsley & Zhang 2004, Chapter 18 sec. I, PDF p. 3, "
                "DOI 10.2514/5.9781600866760.0621.0648"),
        literature=True,
    ),
    # Parsley & Zhang 2004 sec. I: "The performance of a staged combustion cycle generally
    # begins to become hardware limited between 20 and 25 MPa chamber
    # pressure."
    "staged_combustion": ArchitecturePressureLimit(
        lower=5.0e6, upper=25.0e6,
        mechanism="pump discharge pressure and turbine temperature hardware limits",
        source=("Parsley & Zhang 2004, Chapter 18 sec. I, PDF p. 3, "
                "DOI 10.2514/5.9781600866760.0621.0648"),
        literature=True,
    ),
}


def chamber_pressure_bounds(
    feed_architecture: str = "electric_pump",
) -> ArchitecturePressureLimit:
    """Compatibility alias for :func:`chamber_pressure_search_window`.

    The returned endpoints condition the numerical search; they are not a
    declaration that pressures outside them are physically invalid.

    Raises
    ------
    KeyError
        For an unknown architecture.  Silently falling back to the
        electric-pump box would reintroduce exactly the defect this module
        exists to remove.
    """

    return chamber_pressure_search_window(feed_architecture)


def chamber_pressure_search_window(
    feed_architecture: str = "electric_pump",
) -> ArchitecturePressureWindow:
    """Evidence-labelled numerical search guidance for an architecture."""

    key = str(feed_architecture).strip().lower()
    if key not in PRESSURE_LIMITS:
        raise KeyError(
            f"no chamber-pressure basis for feed architecture {feed_architecture!r}; "
            f"known: {sorted(PRESSURE_LIMITS)}"
        )
    return PRESSURE_LIMITS[key]


def chamber_pressure_hard_domain(
    feed_architecture: str = "electric_pump",
) -> tuple[float, float] | None:
    """Return a true architecture hard domain, if one is actually encoded.

    None of the present architecture windows qualifies.  In particular, the
    historical 1.5--6 MPa electric-pump box is a repository search window, not
    a universal physical domain.  Sampled property tables and live thermal,
    structural, pump, and power constraints provide hard limits elsewhere.
    """

    window = chamber_pressure_search_window(feed_architecture)
    if not window.hard_validity:
        return None
    return window.lower, window.upper


def expansion_ratio_bounds(mission: "MissionSpec | None" = None
                           ) -> tuple[float, float]:
    """Expansion-ratio window, taken from the Rao/TOP chart's tabulated box.

    The analytic wall in :mod:`raosim.mdo.grid` interpolates the Rao/NASA chart
    for the wall angles, and ``rao_chart_domain_violation`` makes any design
    outside the tabulated box infeasible.  Bounding ``eps`` anywhere else means
    the optimiser either wastes iterations in a region the constraint will
    reject, or is capped below what the model actually supports.

    Reading the grid rather than restating it also means the bound tracks the
    chart data: extend the table and the design space widens automatically.

    Notes
    -----
    This is a *model-validity* bound, not a physical one.  A high-area-ratio
    vacuum engine is not unphysical; it is outside the chart, and the honest
    routes past it are extending the tabulated chart or landing the exact
    implicit Rao solver (``docs/IMPLICIT_RAO_JAX_MDO_ARCHITECTURE.md``), not
    widening a bound past the data behind it.

    Nozzle separation is deliberately *not* folded in here: it is already the
    ``separation_margin`` constraint, evaluated at the mission's own ambient
    pressure.  Encoding it a second time as a bound would be two sources of
    truth for one limit.
    """

    from raosim.mdo.grid import _EPS_GRID

    return float(_EPS_GRID[0]), float(_EPS_GRID[-1])


def expansion_ratio_reference(mission: "MissionSpec") -> float:
    """A starting ``eps`` inside the chart box, biased by ambient pressure.

    Sea level wants a modest expansion (an overexpanded bell loses thrust and
    trips the separation screen); near-vacuum wants as much as the chart
    allows.  SP-125 ch. III makes the same distinction for the Alpha vehicle:
    the upper stages "operate in the vacuum and can use the largest practical
    expansion area ratio for best performance", while the sea-level A-1 does
    not.  This only seeds the optimiser -- it constrains nothing.
    """

    lo, hi = expansion_ratio_bounds(mission)
    Pa = float(getattr(mission, "Pa", 101325.0))
    if Pa >= 5.0e4:                 # near sea level
        return min(max(8.0, lo), hi)
    if Pa >= 5.0e3:                 # mid altitude
        return min(max(15.0, lo), hi)
    return min(max(0.6 * hi, lo), hi)
