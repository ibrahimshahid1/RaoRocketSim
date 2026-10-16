#!/usr/bin/env bash
# Prefer a repository environment with the KG dependencies.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${KG_PYTHON:-}"
if [ -z "$PY" ]; then
  PY=python3
  for CANDIDATE in "$ROOT/.venv/bin/python" "$ROOT/.venv-jax/bin/python" "$ROOT/.venv-kg/bin/python"; do
    if [ -x "$CANDIDATE" ]; then
      PY="$CANDIDATE"
      break
    fi
  done
fi
cd "$ROOT" && PYTHONPATH="$HERE:$ROOT" exec "$PY" -m kg "$@"
