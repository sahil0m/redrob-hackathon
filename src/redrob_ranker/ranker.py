"""
ranker.py — Orchestrates the full ranking pipeline end-to-end.

    load candidates
      -> extract interpretable features (+ integrity report) for each
      -> attach hybrid-retrieval semantic similarity (cached dense + live BM25)
      -> [optional] train/apply XGBoost LTR over the features
      -> score (linear or LTR base) x behavioral x penalties, honeypot gate
      -> sort, take top-100, generate reasoning, emit rows

Designed to run within the 5-minute CPU budget: the expensive embeddings are
loaded from the precomputed cache; only the single JD query is embedded live.
BM25 over 100k short blobs builds in a few seconds. The LTR model is tiny and
trains in seconds on the proxy labels (or is loaded if pre-trained).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import jd_spec, ltr, reasoning, retrieval, scoring
from .data import CandidateView, build_view, stream_raw
from .features import FeatureBundle, extract
from .integrity import assess


@dataclass
class RankRow:
    candidate_id: str
    rank: int
    score: float
    reasoning: str


def _attach_semantic(
    views: list[CandidateView],
    fbs: list[FeatureBundle],
    embeddings: np.ndarray | None,
    emb_ids: np.ndarray | None,
    use_bm25: bool,
    log,
    embed_live: bool = False,
) -> None:
    """Compute hybrid (dense+BM25) JD similarity and write it into each FeatureBundle."""
    n = len(views)
    dense = np.zeros(n, dtype=np.float32)
    have_dense = False

    if embeddings is not None and emb_ids is not None:
        # Align cached embeddings to the current candidate order by id.
        id_to_row = {cid: i for i, cid in enumerate(emb_ids.tolist())}
        model = retrieval.load_embedder()
        qvec = retrieval.embed_query(model, jd_spec.JD_QUERY_TEXT)
        sims_all = retrieval.dense_similarity(qvec, embeddings)
        for i, v in enumerate(views):
            row = id_to_row.get(v.candidate_id)
            if row is not None:
                dense[i] = sims_all[row]
        have_dense = True
        log(f"  dense similarity attached for {sum(1 for v in views if v.candidate_id in id_to_row)}/{n}")
    elif embed_live:
        # No cache: embed the current (small) set on the fly. Intended for the
        # sandbox demo on <=100 candidates; fast at that size.
        model = retrieval.load_embedder()
        doc_mat = retrieval.embed_passages(model, [v.dense_text for v in views])
        qvec = retrieval.embed_query(model, jd_spec.JD_QUERY_TEXT)
        dense = retrieval.dense_similarity(qvec, doc_mat)
        have_dense = True
        log(f"  dense similarity embedded live for {n} candidates")
    else:
        log("  no precomputed embeddings found; dense similarity disabled (BM25-only)")

    if use_bm25:
        t = time.time()
        bm25 = retrieval.build_bm25([v.sparse_text for v in views])
        sparse = retrieval.bm25_scores(bm25, jd_spec.JD_QUERY_TEXT)
        log(f"  BM25 built + scored in {time.time()-t:.1f}s")
    else:
        sparse = np.zeros(n, dtype=np.float32)

    if have_dense:
        hybrid = retrieval.hybrid_score(dense, sparse, dense_weight=0.7)
    else:
        # BM25 only
        from .retrieval import _minmax
        hybrid = _minmax(sparse)

    for i, fb in enumerate(fbs):
        fb.semantic_sim = float(hybrid[i])


def rank(
    candidates_path: str | Path,
    artifacts_dir: str | Path | None = None,
    use_ltr: bool = True,
    use_bm25: bool = True,
    top_k: int = 100,
    limit: int | None = None,
    verbose: bool = True,
    embed_live: bool = False,
) -> list[RankRow]:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t_start = time.time()

    # 1. Load candidates + per-candidate features and integrity.
    log("Loading candidates and extracting features...")
    views: list[CandidateView] = []
    fbs: list[FeatureBundle] = []
    reports = []
    for i, raw in enumerate(stream_raw(candidates_path)):
        if limit and i >= limit:
            break
        v = build_view(raw)
        views.append(v)
        fbs.append(extract(v))
        reports.append(assess(v))
    log(f"  {len(views)} candidates in {time.time()-t_start:.1f}s")

    # 2. Hybrid semantic similarity.
    embeddings = emb_ids = None
    if artifacts_dir:
        adir = Path(artifacts_dir)
        emb_file, id_file = adir / "embeddings.npy", adir / "candidate_ids.npy"
        if emb_file.exists() and id_file.exists():
            embeddings = np.load(emb_file)
            emb_ids = np.load(id_file, allow_pickle=True)
    log("Attaching hybrid (dense + BM25) JD similarity...")
    _attach_semantic(views, fbs, embeddings, emb_ids, use_bm25, log, embed_live=embed_live)

    # 3. Optional LTR over interpretable features + JD-rubric proxy labels.
    ltr_bases: list[float | None] = [None] * len(views)
    # XGBoost needs a reasonable sample to be stable; on tiny sets (sandbox demo)
    # fall back to the interpretable linear blend.
    MIN_LTR_SAMPLES = 500
    if use_ltr and len(views) < MIN_LTR_SAMPLES:
        log(f"  only {len(views)} candidates (<{MIN_LTR_SAMPLES}); using linear blend instead of LTR")
        use_ltr = False
    if use_ltr:
        log("Training XGBoost LTR on JD-rubric proxy labels...")
        penalties = [scoring._disqualifier_penalty(r) for r in reports]
        labels = np.array([
            ltr.proxy_label(fbs[i], penalties[i], reports[i].is_honeypot)
            for i in range(len(views))
        ], dtype=np.float32)
        X = np.array([fb.vector() for fb in fbs], dtype=np.float32)
        t = time.time()
        model = ltr.train(X, labels)
        preds = ltr.predict(model, X)
        ltr_bases = [float(p) for p in preds]
        log(f"  LTR trained + applied in {time.time()-t:.1f}s")
        log(f"  feature importance (gain): {ltr.feature_importance(model)}")
        rank.last_ltr_model = model  # type: ignore[attr-defined]

    # 4. Score everyone.
    log("Scoring candidates...")
    scored = [
        scoring.score_candidate(views[i], fbs[i], reports[i], ltr_base=ltr_bases[i])
        for i in range(len(views))
    ]

    # 5. Sort. We sort by the SAME rounded score that gets written to the CSV,
    #    then by candidate_id ascending. The validator checks its tie-break rule
    #    (equal scores => candidate_id ascending) against the written 6-dp values,
    #    so sorting on raw floats could place two ids that round to equal scores
    #    out of ascending order. Rounding first makes the written file satisfy the
    #    rule by construction.
    for sc in scored:
        sc.score = round(sc.score, 6)
    scored.sort(key=lambda s: (-s.score, s.candidate_id))
    top = scored[:top_k]

    # 6. Reasoning + assemble rows (scores already rounded + non-increasing).
    rows: list[RankRow] = []
    for idx, sc in enumerate(top):
        rows.append(RankRow(
            candidate_id=sc.candidate_id,
            rank=idx + 1,
            score=sc.score,
            reasoning=reasoning.generate(sc, idx + 1),
        ))

    log(f"Done in {time.time()-t_start:.1f}s. Honeypots in top-{top_k}: "
        f"{sum(1 for s in top if s.integrity.is_honeypot)}")
    return rows
