#!/usr/bin/env bash
# kg wrapper: uses the repo's .venv (numpy/scipy) unless KG_PYTHON is set.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${KG_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
cd "$ROOT" && PYTHONPATH="$HERE:$ROOT" exec "$PY" -m kg "$@"
