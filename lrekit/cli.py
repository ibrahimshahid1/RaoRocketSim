"""Console entry points for LREKit.

The job verbs (``run``/``validate``/``catalog``) are handled first; everything
else falls through to the existing CLI unchanged.  That ordering is the
compatibility translator: no existing invocation changes meaning, and the
unified workflow becomes reachable without a flag-day migration.
"""

from __future__ import annotations

import sys

from raosim.cli import legacy_main, main as _runner_main

__all__ = ["main", "legacy_main"]


def main() -> int:
    """Run a job verb if one was given, otherwise the packaged design CLI."""

    from lrekit.job_cli import JOB_VERBS, main as job_main

    if len(sys.argv) > 1 and sys.argv[1] in JOB_VERBS:
        return int(job_main(sys.argv[1:]))
    return int(_runner_main())
