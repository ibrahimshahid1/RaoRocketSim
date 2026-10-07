"""Hybrid retrieval index: FTS5 (lexical, BM25) + LSA (latent semantic, dense).

Why not a neural embedding model?  The knowledge system must run offline in any
of the repository's virtual environments with nothing beyond NumPy/SciPy.  A
TF-IDF -> truncated-SVD (LSA) space over the whole corpus gives useful
semantic neighbourhoods for a single technical domain and is fully
inspectable: every dimension is a linear combination of vocabulary terms
stored on disk.  The lexical channel keeps exact symbols (``c*``, ``T_wg``)
findable; the dense channel catches paraphrase.  Results are fused with
reciprocal-rank fusion.

Artifacts (``knowledge/vector_index/``):
    chunks.sqlite        chunk text + FTS5 (see store.py)
    vocab.json           term -> column index
    idf.npy              inverse document frequency per term
    components.npy       V x K projection (float16)
    chunk_vectors.npy    N x K unit-normalised chunk vectors (float16)
    chunk_ids.json       row -> chunk_id
    index_meta.json      build parameters, corpus fingerprint
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .store import VECTOR_DIR, connect, fts_search

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'*]*|[α-ωΑ-Ω]|\d+(?:\.\d+)?")
STOP = set("""a an and are as at be by for from has have if in into is it its of on or
that the this to was were will with we our which these those than then there their
can may also such into over under between within without both each been being
""".split())

GREEK_NAMES = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta",
    "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu",
    "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
    "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi", "Ψ": "psi", "Ω": "omega",
}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    out: list[str] = []
    for tok in TOKEN_RE.findall(text):
        if tok in GREEK_NAMES:
            out.append(GREEK_NAMES[tok])
            continue
        t = tok.lower().strip("_'")
        if not t or t in STOP or (t.isdigit() and len(t) > 4):
            continue
        if len(t) == 1 and not t.isalpha():
            continue
        out.append(t)
    return out


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:]) if len(a) > 2 and len(b) > 2]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def corpus_fingerprint(conn) -> str:
    """Identify indexed content, including changes that preserve chunk count."""
    digest = hashlib.sha256()
    for row in conn.execute(
            "SELECT chunk_id, doc_id, seq, page_start, page_end, section, kind, source_sha256, text "
            "FROM chunks ORDER BY chunk_id"):
        digest.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(k: int = 256, max_vocab: int = 40000, min_df: int = 3, max_df_frac: float = 0.5,
          seed: int = 0, verbose: bool = True) -> dict:
    from scipy import sparse
    from scipy.sparse.linalg import svds

    if k < 1:
        raise ValueError("k must be positive")
    conn = connect()
    try:
        rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY doc_id, seq").fetchall()
        fingerprint = corpus_fingerprint(conn)
    finally:
        conn.close()
    n = len(rows)
    if verbose:
        print(f"[index] {n} chunks")
    df: Counter[str] = Counter()
    docs_tokens: list[Counter[str]] = []
    for r in rows:
        toks = tokenize(r["text"])
        toks = toks + _bigrams(toks)
        c = Counter(toks)
        docs_tokens.append(c)
        df.update(c.keys())
    max_df = int(max_df_frac * n)
    vocab_terms = [t for t, d in df.items() if min_df <= d <= max_df]
    vocab_terms.sort(key=lambda t: (-df[t], t))
    vocab_terms = vocab_terms[:max_vocab]
    vocab = {t: i for i, t in enumerate(vocab_terms)}
    idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab_terms], dtype=np.float32)
    if verbose:
        print(f"[index] vocab {len(vocab)} (from {len(df)} candidate terms)")

    data, indices, indptr = [], [], [0]
    for c in docs_tokens:
        row_items = [(vocab[t], cnt) for t, cnt in c.items() if t in vocab]
        norm_acc = 0.0
        vals = []
        for j, cnt in row_items:
            v = (1.0 + math.log(cnt)) * idf[j]
            vals.append((j, v))
            norm_acc += v * v
        norm = math.sqrt(norm_acc) or 1.0
        for j, v in vals:
            indices.append(j)
            data.append(v / norm)
        indptr.append(len(indices))
    X = sparse.csr_matrix((np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
                          shape=(n, len(vocab)))
    rank_bound = min(X.shape)
    if rank_bound == 0:
        # Persist an explicit empty index, replacing any previous vectors. FTS
        # still works when frequency filters leave no semantic vocabulary.
        k_eff = 0
        S = np.empty(0, dtype=np.float32)
        components = np.empty((len(vocab), 0), dtype=np.float32)
        chunk_vecs = np.empty((n, 0), dtype=np.float32)
    else:
        k_eff = min(k, max(1, rank_bound - 1))
        if verbose:
            print(f"[index] SVD k={k_eff} ...")
        if rank_bound == 1:
            # ARPACK requires k < min(shape); a single row/column is small
            # enough for an exact SVD and still has one useful direction.
            U, S, Vt = np.linalg.svd(X.toarray(), full_matrices=False)
        else:
            rng = np.random.default_rng(seed)
            v0 = rng.standard_normal(rank_bound).astype(np.float32)
            U, S, Vt = svds(X.asfptype(), k=k_eff, v0=v0)
        order = np.argsort(-S)
        U, S, Vt = U[:, order], S[order], Vt[order, :]
        components = (Vt.T).astype(np.float32)             # V x K
        chunk_vecs = (U * S).astype(np.float32)             # N x K
    norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    chunk_vecs = chunk_vecs / norms

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    (VECTOR_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    np.save(VECTOR_DIR / "idf.npy", idf)
    np.save(VECTOR_DIR / "components.npy", components.astype(np.float16))
    np.save(VECTOR_DIR / "chunk_vectors.npy", chunk_vecs.astype(np.float16))
    (VECTOR_DIR / "chunk_ids.json").write_text(json.dumps([r["chunk_id"] for r in rows]))
    meta = {
        "method": "tfidf(sublinear, l2) + truncated SVD (LSA)",
        "n_chunks": n, "vocab_size": len(vocab), "k": int(k_eff),
        "min_df": min_df, "max_df_frac": max_df_frac, "seed": seed,
        "explained_singular_values": [float(s) for s in S[:10]],
        "corpus_fingerprint": fingerprint,
        "state": "OK" if k_eff else "UNAVAILABLE",
    }
    if not k_eff:
        meta["reason"] = "EMPTY_CORPUS" if not n else "NO_INDEXABLE_TERMS"
    (VECTOR_DIR / "index_meta.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print("[index] done")
    return meta


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------

class VectorIndex:
    def __init__(self):
        try:
            self.vocab = json.loads((VECTOR_DIR / "vocab.json").read_text())
            self.idf = np.load(VECTOR_DIR / "idf.npy")
            self.components = np.load(VECTOR_DIR / "components.npy").astype(np.float32)
            self.vectors = np.load(VECTOR_DIR / "chunk_vectors.npy").astype(np.float32)
            self.chunk_ids = json.loads((VECTOR_DIR / "chunk_ids.json").read_text())
            self.meta = json.loads((VECTOR_DIR / "index_meta.json").read_text())
            if (self.components.ndim != 2 or self.vectors.ndim != 2
                    or self.idf.shape != (len(self.vocab),)
                    or self.components.shape[0] != len(self.vocab)
                    or self.vectors.shape != (len(self.chunk_ids), self.components.shape[1])
                    or self.meta.get("n_chunks") != len(self.chunk_ids)):
                raise ValueError("Inconsistent vector artifacts; rebuild was interrupted")
        except (OSError, ValueError) as exc:
            self.vocab, self.chunk_ids = {}, []
            self.idf = np.empty(0, dtype=np.float32)
            self.components = np.empty((0, 0), dtype=np.float32)
            self.vectors = np.empty((0, 0), dtype=np.float32)
            self.meta = {"state": "UNAVAILABLE", "reason": "VECTOR_INDEX_NOT_BUILT"
                         if isinstance(exc, FileNotFoundError) else "VECTOR_INDEX_INCONSISTENT"}
        available = bool(self.vocab and self.chunk_ids and self.components.shape[1])
        self.status = {"state": "OK" if available else "UNAVAILABLE"}
        if not available:
            self.status["reason"] = self.meta.get("reason", "NO_INDEXABLE_TERMS")
            self.status["recovery"] = "Run knowledge/kg.sh rebuild, then reset API caches or restart the process."

    def embed(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        toks = toks + _bigrams(toks)
        c = Counter(toks)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t, cnt in c.items():
            j = self.vocab.get(t)
            if j is not None:
                vec[j] = (1.0 + math.log(cnt)) * self.idf[j]
        nrm = np.linalg.norm(vec)
        if nrm:
            vec /= nrm
        q = vec @ self.components
        qn = np.linalg.norm(q)
        return q / qn if qn else q

    def search(self, text: str, limit: int = 50) -> list[tuple[str, float]]:
        if limit <= 0 or not self.chunk_ids:
            return []
        q = self.embed(text)
        if not q.any():
            return []
        scores = self.vectors @ q
        # A nonpositive cosine is not positive evidence of relevance. Exclude
        # it before rank fusion, which otherwise turns it into a positive hit.
        candidates = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        count = min(limit, len(candidates))
        if not count:
            return []
        top = candidates[np.argpartition(-scores[candidates], count - 1)[:count]]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]


def _fts_query(text: str) -> str:
    toks = [t for t in tokenize(text) if len(t) > 1]
    if not toks:
        return '""'
    # OR semantics with quoting so tokens containing * or _ survive FTS syntax
    return " OR ".join('"' + t.replace('"', '') + '"' for t in dict.fromkeys(toks))


def hybrid_search(text: str, limit: int = 20, expand_terms: list[str] | None = None,
                  vector_index: VectorIndex | None = None) -> list[dict]:
    """Reciprocal-rank fusion of BM25 and LSA hits."""
    if limit <= 0:
        return []
    conn = connect()
    query_text = text if not expand_terms else text + " " + " ".join(expand_terms)
    lex = fts_search(conn, _fts_query(query_text), limit=limit * 4)
    vi = vector_index or VectorIndex()
    sem = vi.search(query_text, limit=limit * 4)

    K = 60.0
    fused: dict[str, float] = {}
    ranks: dict[str, dict] = {}
    for r, row in enumerate(lex):
        fused[row["chunk_id"]] = fused.get(row["chunk_id"], 0.0) + 1.0 / (K + r)
        ranks.setdefault(row["chunk_id"], {})["bm25_rank"] = r + 1
    for r, (cid, s) in enumerate(sem):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (K + r)
        ranks.setdefault(cid, {})["lsa_rank"] = r + 1
        ranks[cid]["lsa_cos"] = round(s, 4)
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
    out = []
    for cid, score in top:
        row = conn.execute(
            "SELECT chunk_id, doc_id, seq, page_start, page_end, section, kind, equation_signals, has_table, text "
            "FROM chunks WHERE chunk_id = ?", (cid,)).fetchone()
        if not row:
            continue
        d = dict(row)
        d["score"] = round(score, 5)
        d.update(ranks.get(cid, {}))
        out.append(d)
    return out
