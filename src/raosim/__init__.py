"""Compatibility package for ``raosim``, the package's name before 2026-11-21.

``raosim.<x>`` and the former flat module names (``raosim.cea``, ``raosim.mdo``,
``raosim.run_nozzle`` ...) resolve to the single implementation under
:mod:`lrekit`; nothing is executed twice, and objects pickled under the old
names still load. New code should import :mod:`lrekit`.
"""

from __future__ import annotations

from lrekit import __version__  # importing lrekit installs the aliases

__all__ = ["__version__"]
