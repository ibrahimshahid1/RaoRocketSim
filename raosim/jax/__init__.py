"""
raosim.jax — differentiable (JAX) Rao nozzle core.

This subpackage is a *separate solver backend* living inside the same repo.
It re-expresses the throat-to-contour numerics of ``raosim.rao_variational``
in JAX so the BVP residual is end-to-end autodifferentiable.  The NumPy/SciPy
path in ``raosim.rao_variational`` remains the default and the regression
oracle; this backend is opt-in via ``solver_backend="jax"`` and does not earn
the default until the J5 gate passes (see ``JAX_DIFFERENTIABLE_PLAN.md``).

Run tag for outputs/builds produced by this backend: ``"jax"``.

IMPORTANT: float64 is required for bit-parity with the NumPy primitives.  JAX
defaults to float32, so x64 is enabled here at import time, before any array is
created.  Import ``raosim.jax`` (or anything under it) before constructing JAX
arrays elsewhere if you need x64 globally.
"""

from __future__ import annotations

import jax

# Must run before any jnp array is allocated anywhere in the process.
jax.config.update("jax_enable_x64", True)

#: Backend identifier, used as the run tag in build/output directories.
BACKEND = "jax"

__all__ = ["BACKEND"]
