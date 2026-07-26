"""
raosim.mdo — multidisciplinary-feasible (MDF) differentiable engine MDO layer.

Implements docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md (§10 architecture, §12 build
order).  This package is a *pure-numerical* layer: no CAD, no file I/O in the
differentiated path, no host callbacks inside jitted code (plan rule 10 and
Phase-1 gate).  CAD/reporting stays downstream in the existing LREKit workflow.

Build state (see the plan's §11 tracker):

* Phase 1 — ``schema.py`` + ``scaling.py``: design-variable pytrees, bounds,
  affine scaling, jit/jacfwd-clean evaluate.                      [this commit]
* Phase 2 — ``properties.py``: C¹ shape-preserving property surfaces with an
  offline CEA sampler (``scripts/sample_cea_surface.py``).        [this commit]
* Phases 3–7 — ``assembly.py``: the walking-skeleton coupled residual
  R(y, x) = 0 (analytic nozzle + throat thermal + injector/pump/battery
  algebra) solved with a square Newton root-find and differentiated via the
  implicit function theorem.
* Phase 4a — ``grid.py`` + ``cooling.py``: fixed-topology station grid and the
  1-D+fin regen residual (Bartz/Sieder-Tate/land-fin, counterflow upwind march)
  solved for the stationwise wall temperatures; exposes the channel-fit and
  SP-8087 RP-1 coking (T_wc ≤ 728 K) constraints.
* Phase 5 — ``injector.py``: closed-form pintle block (orifice/TMR/blockage/
  spray) with the Son-2017 movable-pintle two-branch minimum area and the chug
  screen.
* Phase 6 — ``pump.py``: pump/electric-feed block with a C¹ efficiency-vs-
  specific-speed surrogate (replacing the binned ``pumps.py`` estimator),
  meanline duty/NPSH/tip-speed screens, and the Lee-2021 battery epigraph.
* Phase 7 — ``engine.py``: the coupled whole-engine solve ``solve_engine`` —
  all four blocks in one differentiable evaluation with the cooling Δp → pump
  hydraulic edge closed and an optional spray→η_c* fixed point (default frozen +
  ablation).  Surfaced via the ``--engine-mdo`` CLI flag.
* Phase 8/9 — ``nlp.py``: the ε-constraint hard-constrained NLP ``solve_min_mass``
  / ``pareto_frontier`` — min package mass s.t. Isp ≥ floor and every enforced
  margin ≥ 0, over the 6-var unit-box design vector, exact JAX Jacobians → SLSQP.
  Surfaced via ``--engine-mdo-optimize``.

Hard rules inherited from the plan (§0.1): no ``max()`` in the differentiable
core; discrete choices enumerated outside; never differentiate through solver
iterations; every block keeps a NumPy oracle + parity test.
"""

from raosim.mdo.schema import (  # noqa: F401
    DesignVector,
    MissionSpec,
    VariableSpec,
    default_design_space,
)
from raosim.mdo.scaling import ScaledSpace  # noqa: F401
from raosim.mdo.grid import build_station_grid, GridTopology  # noqa: F401
from raosim.mdo.cooling import cooling_march, solve_cooling  # noqa: F401
from raosim.mdo.injector import injector_readouts  # noqa: F401
from raosim.mdo.pump import (  # noqa: F401
    pump_stream,
    pump_efficiency,
    electric_feed,
    battery_masses,
)
from raosim.mdo.engine import (  # noqa: F401
    solve_engine,
    engine_outputs,
    ablation_delta,
    EngineResult,
)
from raosim.mdo.nlp import (  # noqa: F401
    solve_min_mass,
    pareto_frontier,
    NLPResult,
    DEFAULT_ENFORCED,
)
