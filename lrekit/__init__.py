"""LREKit public package facade.

The original solver modules currently live under :mod:`raosim` for
compatibility. New user-facing entry points should prefer :mod:`lrekit`.
"""

from __future__ import annotations

from pathlib import Path

import raosim
from raosim import __version__

# Let `lrekit.engine`, `lrekit.design`, etc. resolve to the existing raosim
# modules while the internal package layout migrates gradually.
__path__ = [str(Path(__file__).parent), *raosim.__path__]

__all__ = ["__version__"]
