"""
raosim.jax.reference — canonical M3.5Perf reference constants.

Per JAX_DIFFERENTIABLE_PLAN.md decision (4): static array shapes for the JAX
march are sized from the NASA/JHU ``outputs_M3.5Perf`` reference case rather than
made configurable.  Values are taken from
``docs/rice_jhu_moc_topology_map.md`` (parsed from ``summary.out``).

The march runs as a ``lax.scan`` over a padded grid with a validity mask;
``MAX_KERNEL_ROWS`` must comfortably exceed the reference ``lastKernelJ`` so no
real row is ever truncated.  Padded (masked) rows contribute zero residual.
"""

from __future__ import annotations

# --- From outputs_M3.5Perf/summary.out (docs/rice_jhu_moc_topology_map.md) --- #
R_STAR = 1.0            # throat radius, in
RWTD_OVER_RSTAR = 1.0   # downstream throat-wall radius ratio (Rd/Rt)
GAMMA = 1.4
M_EXIT = 3.5
THETA_B_DEG = 15.2196   # converged initial expansion angle
LAST_KERNEL_J = 57      # number of kernel rows in the reference solve
L_OVER_RSTAR = 12.5363  # nozzle length

# --- Static padded-grid sizing (lax.scan shapes) --------------------------- #
#: Max kernel rows the JAX march will allocate.  57 reference + headroom.
MAX_KERNEL_ROWS = 64
#: Max nodes per characteristic row (kernel grows ~1 node/row from the axis).
MAX_NODES_PER_ROW = 80
#: Max control-surface (DE) + wall nodes carried in the BVP unknown vector.
MAX_CE_NODES = 80
MAX_WALL_NODES = 80

__all__ = [
    "R_STAR", "RWTD_OVER_RSTAR", "GAMMA", "M_EXIT", "THETA_B_DEG",
    "LAST_KERNEL_J", "L_OVER_RSTAR",
    "MAX_KERNEL_ROWS", "MAX_NODES_PER_ROW", "MAX_CE_NODES", "MAX_WALL_NODES",
]
