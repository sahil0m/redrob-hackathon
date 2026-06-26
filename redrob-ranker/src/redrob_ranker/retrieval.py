"""
retrieval.py — Hybrid semantic + lexical matching of the JD against candidates.

Two complementary signals, both computed against the JD query:

  * DENSE (e5-small-v2 embeddings, cosine): captures *intent*. A profile that
    describes building a recommender without ever writing "RAG" still lands near
    the JD in embedding space. This is what surfaces plain-language Tier-5s.
  * SPARSE (BM25 over the lowercased profile blob): captures *exact term
    overlap*. Rewards profiles that genuinely use the JD's vocabulary
    (embeddings, FAISS, NDCG) where it actually appears in their work.

Hybrid score = convex blend of min-max-normalised dense and sparse scores. The
blend weight favours dense (intent) because the whole challenge is built to
punish pure lexical matching — but sparse keeps us honest on the rare profiles
that legitimately name the right tools.

The expensive part — embedding 100k profiles — is done ONCE, offline, by
scripts/precompute.py (the spec permits pre-computation outside the 5-minute
ranking budget). At rank time we only load the cached matrix and embed the
single JD query, which is milliseconds.

CPU-ONLY is enforced explicitly (device="cpu"); the spec forbids GPU at rank
time, and forcing CPU also avoids depending on any particular CUDA build.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Make the run hermetic and CPU-only regardless of host setup. Set before torch
# is imported by sentence-transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

EMBED_MODEL = "intfloat/e5-small-v2"
EMBED_DIM = 384
# Cap the transformer sequence length. Our dense_text is front-loaded to ~256
# words; e5's default 512-token window means short profiles still pay for 512
# positions. Measured on this pool: capping to 128 tokens lifts CPU throughput
# from ~28/s to ~90/s (100k: ~60min -> ~18min) with negligible quality loss,
# because the JD-decisive signal (title, summary, recent role) lives in the head.
MAX_SEQ_LENGTH = 128


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_embedder():
    """Load e5-small-v2 on CPU from the local HF cache (no network)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed_passages(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed candidate profile blobs. e5 expects a 'passage: ' prefix."""
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def embed_query(model, query: str) -> np.ndarray:
    """Embed the JD query. e5 expects a 'query: ' prefix."""
    emb = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)[0]


def dense_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity (vectors are L2-normalised, so this is a dot product)."""
    return doc_matrix @ query_vec


# --------------------------------------------------------------------------- #
# BM25 sparse retrieval
# --------------------------------------------------------------------------- #

_TOKEN_SPLIT = None


def _tokenize(text: str) -> list[str]:
    global _TOKEN_SPLIT
    if _TOKEN_SPLIT is None:
        import re
        _TOKEN_SPLIT = re.compile(r"[a-z0-9+#.]+")
    return _TOKEN_SPLIT.findall(text.lower())


def build_bm25(corpus_texts: list[str]):
    """Build a BM25 index over the lowercased sparse profile blobs."""
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def bm25_scores(bm25, query: str) -> np.ndarray:
    return np.asarray(bm25.get_scores(_tokenize(query)), dtype=np.float32)


def hybrid_score(
    dense: np.ndarray,
    sparse: np.ndarray,
    dense_weight: float = 0.7,
) -> np.ndarray:
    """
    Convex blend of min-max-normalised dense and sparse scores, in [0,1].
    dense_weight favours semantic intent over lexical overlap because the
    challenge explicitly penalises keyword matching.
    """
    d = _minmax(dense)
    s = _minmax(sparse)
    return dense_weight * d + (1.0 - dense_weight) * s
