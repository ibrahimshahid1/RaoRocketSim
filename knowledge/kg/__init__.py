"""LREKit local scientific memory and engineering provenance layer.

Public entry points live in :mod:`kg.api` (programmatic) and :mod:`kg.cli`
(``kg`` command).  Storage layout is documented in ``knowledge/AGENTS.md``.
"""

from __future__ import annotations

from pathlib import Path

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KNOWLEDGE_ROOT.parent
CORPUS_ROOT = REPO_ROOT / "propulsion_texts"
AGENT_CORPUS_ROOT = CORPUS_ROOT / "propulsion_texts_for_agents"
MARKDOWN_ROOT = AGENT_CORPUS_ROOT / "markdown"

__all__ = ["KNOWLEDGE_ROOT", "REPO_ROOT", "CORPUS_ROOT", "AGENT_CORPUS_ROOT", "MARKDOWN_ROOT"]
