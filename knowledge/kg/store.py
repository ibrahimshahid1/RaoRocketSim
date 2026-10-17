"""Inspectable on-disk stores for the knowledge system.

Curated knowledge objects (documents, equations, claims, models, ...) are
plain JSON files under ``knowledge/`` so a human or agent can open them.
Chunks are numerous (tens of thousands) and live in SQLite with an FTS5
index under ``knowledge/vector_index/``.

Every JSON store is a list of records with a stable ``id`` field; writes are
atomic (temp file + rename) and sorted by id so diffs stay reviewable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Iterable

from . import KNOWLEDGE_ROOT

STORE_FILES = {
    "manifest": "manifest.json",
    "entities": "entities.json",
    "quantities": "quantities.json",
    "equations": "equations.json",
    "claims": "claims.json",
    "models": "models.json",
    "requirements": "requirements.json",
    "validity": "validity.json",
    "software_links": "software_links.json",
    "relationships": "relationships.json",
    "provenance": "provenance.json",
    "repo_documents": "repo_documents.json",
    "tests": "tests.json",
    "capabilities": "capabilities.json",
    "ingestion_state": "ingestion_state.json",
    "artifacts": "artifacts.json",
    "enrichments": "enrichments.json",
    "standards": "standards.json",
}

VECTOR_DIR = KNOWLEDGE_ROOT / "vector_index"
GRAPH_DIR = KNOWLEDGE_ROOT / "graph"
REPORT_DIR = KNOWLEDGE_ROOT / "reports"
CHUNK_DB = VECTOR_DIR / "chunks.sqlite"


def store_path(name: str) -> Path:
    return KNOWLEDGE_ROOT / STORE_FILES[name]


def load(name: str) -> list[dict]:
    path = store_path(name)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data


def load_meta(name: str) -> dict:
    path = store_path(name)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("meta", {}) if isinstance(data, dict) else {}


class DuplicateRecordIdError(ValueError):
    """Conflicting records must be consolidated or resolved before persistence."""

    def __init__(self, key: str, value, records: list[dict]):
        self.key, self.value, self.records = key, value, records
        super().__init__(f"Conflicting duplicate {key}={value!r}; resolve identity or preserve survey facets before saving.")


def _unique_records(records: Iterable[dict], key: str) -> list[dict]:
    unique, seen = [], {}
    for record in records:
        if key not in record:
            unique.append(record)
            continue
        value = record[key]
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if value in seen:
            previous, previous_encoded = seen[value]
            if encoded != previous_encoded:
                raise DuplicateRecordIdError(key, value, [previous, record])
            continue  # byte-equivalent JSON records add no distinct evidence
        seen[value] = (record, encoded)
        unique.append(record)
    return unique


# Process umask, read once: os.umask can only be queried by setting it.
_UMASK = os.umask(0)
os.umask(_UMASK)


def _published_mode(path: Path) -> int:
    """Keep an existing file's permission bits; a new file gets the umask default."""
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return 0o666 & ~_UMASK


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish a complete file; a private temp name cannot collide with another writer.

    The private temporary file is created 0600; its mode is set to the published
    file's (or the umask default) before the replace, so atomic publication never
    narrows who can read a store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _published_mode(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as fh:
        temporary = Path(fh.name)
        try:
            json.dump(payload, fh, indent=1, ensure_ascii=False, sort_keys=False)
            fh.flush()
            os.fchmod(fh.fileno(), mode)
            os.fsync(fh.fileno())
        except BaseException:
            fh.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save(name: str, records: Iterable[dict], meta: dict | None = None, key: str = "id") -> Path:
    # Validate before creating/replacing files; never silently pick a facet.
    records = sorted(_unique_records(records, key), key=lambda r: str(r.get(key, "")))
    path = store_path(name)
    from . import locks  # local import: locks resolves its directory through this module
    with locks.writer(purpose=f"save {name}"):
        _atomic_write_json(path, {"meta": meta or {}, "records": records})
    return path


def upsert(name: str, new_records: Iterable[dict], key: str = "id", meta: dict | None = None) -> int:
    """Insert-or-replace by id.  Returns number of records written."""
    from . import locks
    with locks.writer(purpose=f"upsert {name}"):  # read-modify-write must not interleave
        existing = {r[key]: r for r in _unique_records(load(name), key) if key in r}
        n = 0
        for rec in _unique_records(new_records, key):
            existing[rec[key]] = rec
            n += 1
        save(name, existing.values(), meta=meta or load_meta(name), key=key)
    return n


def index_by(name: str, key: str = "id") -> dict[str, dict]:
    return {r[key]: r for r in load(name) if key in r}


# --------------------------------------------------------------------------
# Chunk database
# --------------------------------------------------------------------------

CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section TEXT,
    kind TEXT,
    char_start INTEGER,
    char_end INTEGER,
    n_chars INTEGER,
    equation_signals INTEGER DEFAULT 0,
    has_table INTEGER DEFAULT 0,
    extraction_version TEXT,
    source_sha256 TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section, doc_id UNINDEXED, chunk_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 0 tokenchars ''_*'''
);
CREATE TABLE IF NOT EXISTS doc_state (
    doc_id TEXT PRIMARY KEY,
    source_sha256 TEXT,
    markdown_sha256 TEXT,
    extraction_version TEXT,
    n_chunks INTEGER,
    ingested_at TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or CHUNK_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(CHUNK_SCHEMA)
    return conn


def replace_doc_chunks(conn: sqlite3.Connection, doc_id: str, chunks: list[dict], state: dict) -> None:
    with conn:
        conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            """INSERT INTO chunks (chunk_id, doc_id, seq, page_start, page_end, section, kind,
                                   char_start, char_end, n_chars, equation_signals, has_table,
                                   extraction_version, source_sha256, text)
               VALUES (:chunk_id, :doc_id, :seq, :page_start, :page_end, :section, :kind,
                       :char_start, :char_end, :n_chars, :equation_signals, :has_table,
                       :extraction_version, :source_sha256, :text)""",
            chunks,
        )
        conn.executemany(
            "INSERT INTO chunks_fts (text, section, doc_id, chunk_id) VALUES (:text, :section, :doc_id, :chunk_id)",
            chunks,
        )
        conn.execute(
            """INSERT OR REPLACE INTO doc_state (doc_id, source_sha256, markdown_sha256, extraction_version,
                                                 n_chunks, ingested_at)
               VALUES (:doc_id, :source_sha256, :markdown_sha256, :extraction_version, :n_chunks, :ingested_at)""",
            state,
        )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return dict(row) if row else None


def doc_chunks(conn: sqlite3.Connection, doc_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY seq", (doc_id,))]


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.seq, c.page_start, c.page_end, c.section, c.kind,
                  c.equation_signals, c.has_table, bm25(chunks_fts) AS score
           FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
           WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]
