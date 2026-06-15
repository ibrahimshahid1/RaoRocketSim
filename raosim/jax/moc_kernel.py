"""
raosim.jax.moc_kernel — differentiable transonic start line + characteristic march.

Lands in **J2/J3** (see JAX_DIFFERENTIABLE_PLAN.md §4.2).

Ports the Kliegel-Levine throat start line (``raosim.transonic_kernel``, with the
documented NASA ``KLThroat`` typo correction) and the axisymmetric unit processes
from ``raosim.moc`` — the Anderson Q± source terms (``moc.py:154-160``) are the
literature ground truth and port verbatim.  The march is the only data-dependent
sequential piece; it runs as a ``lax.scan`` over a padded grid (shapes from
``raosim.jax.reference``) with a validity mask.
"""

from __future__ import annotations

import raosim.jax  # noqa: F401


def build_start_line(*args, **kwargs):
    raise NotImplementedError("J2: transonic start line — see JAX_DIFFERENTIABLE_PLAN.md")


def march_kernel(*args, **kwargs):
    raise NotImplementedError("J2/J3: characteristic march — see JAX_DIFFERENTIABLE_PLAN.md")
