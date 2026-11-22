"""LREKit: liquid rocket engine design, analysis and optimization toolkit.

The implementation is organized by responsibility: ``lrekit.components`` (nozzle,
chamber, cooling, injector, pump, electrical), ``lrekit.optimization`` (engine MDO,
job specifications), ``lrekit.numerics`` (JAX kernels), ``lrekit.assembly``,
``lrekit.thermochemistry``, ``lrekit.application`` (runner and console entry
points) and the shared ``core``/``geometry``/``verification`` packages.

The former package name ``raosim`` and the former flat module names keep
importing as aliases of these modules (see :mod:`lrekit._compat`).
"""

from __future__ import annotations

__version__ = "0.1.1"

from ._compat import install as _install_compatibility

_install_compatibility()
del _install_compatibility

__all__ = ["__version__"]
