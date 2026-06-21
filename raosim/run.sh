#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" scripts/run_nozzle.py "$@"
