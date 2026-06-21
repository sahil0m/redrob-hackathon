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

from . import candidate_generation, jd_spec, ltr, reasoning, retrieval, scoring
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
    query_vec: np.ndarray | None = None,
) -> None:
    """Compute hybrid (dense+BM25) JD similarity and write it into each FeatureBundle.

    The dense path NEVER loads the embedding model at rank time when the JD query
    vector is cached (query_vec): it is just a matrix-vector dot product against
    the precomputed candidate matrix. The model is only loaded for the live-embed
    sandbox path, and even there a load failure degrades gracefully to BM25 — so a
    broken torch/transformers environment can never break the submission."""
    n = len(views)
    dense = np.zeros(n, dtype=np.float32)
    have_dense = False

    if embeddings is not None and emb_ids is not None:
        # Align cached embeddings to the current candidate order by id.
        id_to_row = {cid: i for i, cid in enumerate(emb_ids.tolist())}
        qvec = query_vec
        if qvec is None:
            # No cached query vector — embed the single JD query once. Wrapped so a
            # broken embedding environment falls back to BM25 rather than aborting.
            try:
                model = retrieval.load_embedder()
                qvec = retrieval.embed_query(model, jd_spec.JD_QUERY_TEXT)
            except Exception as e:  # noqa: BLE001 — any import/runtime failure
                log(f"  embedder unavailable ({type(e).__name__}); dense disabled, using BM25 only")
                qvec = None
        if qvec is not None:
            sims_all = retrieval.dense_similarity(qvec, embeddings)
            for i, v in enumerate(views):
                row = id_to_row.get(v.candidate_id)
                if row is not None:
                    dense[i] = sims_all[row]
            have_dense = True
            log(f"  dense similarity attached for {sum(1 for v in views if v.candidate_id in id_to_row)}/{n}")
    elif embed_live:
        # No cache: embed the current (small) set on the fly (sandbox demo).
        try:
            model = retrieval.load_embedder()
            doc_mat = retrieval.embed_passages(model, [v.dense_text for v in views])
            qvec = retrieval.embed_query(model, jd_spec.JD_QUERY_TEXT)
            dense = retrieval.dense_similarity(qvec, doc_mat)
            have_dense = True
            log(f"  dense similarity embedded live for {n} candidates")
        except Exception as e:  # noqa: BLE001
            log(f"  embedder unavailable ({type(e).__name__}); dense disabled, using BM25 only")
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

    # 2. Hybrid semantic similarity. We load the precomputed candidate matrix AND
    #    the precomputed JD query vector, so the rank step does a pure dot product
    #    with no embedding-model load (faster, and robust to a broken torch env).
    embeddings = emb_ids = query_vec = None
    if artifacts_dir:
        adir = Path(artifacts_dir)
        emb_file, id_file = adir / "embeddings.npy", adir / "candidate_ids.npy"
        q_file = adir / "jd_query.npy"
        if emb_file.exists() and id_file.exists():
            embeddings = np.load(emb_file)
            emb_ids = np.load(id_file, allow_pickle=True)
        if q_file.exists():
            query_vec = np.load(q_file)
    log("Attaching hybrid (dense + BM25) JD similarity...")
    _attach_semantic(views, fbs, embeddings, emb_ids, use_bm25, log,
                     embed_live=embed_live, query_vec=query_vec)

    # 3. STAGE 1 — candidate generation. Gate the pool down to the eligible set
    #    (on-role / strong-adjacent, non-honeypot). Measured lossless recall of
    #    relevant candidates while shrinking the re-ranking problem ~9x.
    gates = [candidate_generation.passes_gate(views[i], fbs[i], reports[i])
             for i in range(len(views))]
    eligible_idx = [i for i, g in enumerate(gates) if g.eligible]
    log(f"Stage 1 (candidate generation): {len(eligible_idx)} eligible of {len(views)}")

    # 4. STAGE 2 — re-rank the eligible set. The LTR trains on the eligible
    #    distribution (not 90k off-role noise), which is both faster and better
    #    ML practice. role_fit is excluded from the re-rank features (it's the
    #    gate; see rerank.py).
    ltr_bases: list[float | None] = [None] * len(views)
    MIN_LTR_SAMPLES = 300
    do_ltr = use_ltr and len(eligible_idx) >= MIN_LTR_SAMPLES
    if use_ltr and not do_ltr:
        log(f"  only {len(eligible_idx)} eligible (<{MIN_LTR_SAMPLES}); using linear re-rank blend")
    if do_ltr:
        log("Stage 2: training XGBoost LTR on the eligible set (JD-rubric proxy labels)...")
        penalties = {i: scoring._disqualifier_penalty(reports[i]) for i in eligible_idx}
        labels = np.array([
            ltr.proxy_label(fbs[i], penalties[i], reports[i].is_honeypot)
            for i in eligible_idx
        ], dtype=np.float32)
        X = np.array([fbs[i].rerank_vector() for i in eligible_idx], dtype=np.float32)
        t = time.time()
        model = ltr.train(X, labels)
        preds = ltr.predict(model, X)
        for j, i in enumerate(eligible_idx):
            ltr_bases[i] = float(preds[j])
        log(f"  LTR trained + applied in {time.time()-t:.1f}s")
        rank.last_ltr_model = model  # type: ignore[attr-defined]

    # 5. Score: eligible candidates get a real score; gated-out get 0.
    log("Scoring candidates...")
    scored = [
        scoring.score_candidate(views[i], fbs[i], reports[i],
                                ltr_base=ltr_bases[i], eligible=gates[i].eligible)
        for i in range(len(views))
    ]

    # 6. Sort. We sort by the SAME rounded score that gets written to the CSV,
    #    then by candidate_id ascending. The validator checks its tie-break rule
    #    (equal scores => candidate_id ascending) against the written 6-dp values,
    #    so sorting on raw floats could place two ids that round to equal scores
    #    out of ascending order. Rounding first makes the written file satisfy the
    #    rule by construction.
    for sc in scored:
        sc.score = round(sc.score, 6)
    scored.sort(key=lambda s: (-s.score, s.candidate_id))
    top = scored[:top_k]

    # 7. Reasoning + assemble rows (scores already rounded + non-increasing).
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
