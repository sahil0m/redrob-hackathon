"""
derive_weights.py — Empirically derive Stage-2 re-rank weights from ablation.

Instead of hand-picking weights, we measure each differentiator's marginal
contribution to the composite ON THE GATED POOL and set the weight proportional
to that contribution. This is the analysis that lets us say, at interview, "the
weights reflect measured marginal impact, not judgement."

Method:
  1. Gate the pool (Stage 1) -> eligible candidates only.
  2. With a uniform-weight linear re-ranker as the reference, zero each feature
     in turn and measure the drop in composite (vs the independent eval labels).
  3. Normalise the non-negative drops to sum to 1 -> the derived weights.

We report the derived weights; paste them into rerank.RERANK_WEIGHTS (or confirm
they match). Run on the full pool for a stable estimate.

    python scripts/derive_weights.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from redrob_ranker import metrics, scoring  # noqa: E402
from redrob_ranker.candidate_generation import passes_gate  # noqa: E402
from redrob_ranker.data import build_view, stream_raw  # noqa: E402
from redrob_ranker.eval_labels import relevance_tier  # noqa: E402
from redrob_ranker.features import extract  # noqa: E402
from redrob_ranker.integrity import assess  # noqa: E402
from redrob_ranker.ranker import _attach_semantic  # noqa: E402
from redrob_ranker.rerank import RERANK_FEATURES, SEMANTIC_WEIGHT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    views, fbs, reports, grades = [], [], [], {}
    for i, raw in enumerate(stream_raw(args.candidates)):
        if args.limit and i >= args.limit:
            break
        v = build_view(raw)
        views.append(v); fbs.append(extract(v)); reports.append(assess(v))
        grades[v.candidate_id] = relevance_tier(v)
    all_grades = list(grades.values())

    adir = Path(args.artifacts)
    emb = ids = None
    if (adir / "embeddings.npy").exists():
        emb = np.load(adir / "embeddings.npy")
        ids = np.load(adir / "candidate_ids.npy", allow_pickle=True)
    _attach_semantic(views, fbs, emb, ids, True, lambda *_: None)

    # Gate.
    elig = [(v, fb, r) for v, fb, r in zip(views, fbs, reports) if passes_gate(v, fb, r).eligible]
    print(f"eligible after gate: {len(elig)} / {len(views)}")

    uniform = {f: 1.0 / len(RERANK_FEATURES) for f in RERANK_FEATURES}

    def composite_with(weights, zero=None):
        scored = []
        for v, fb, rep in elig:
            feat = sum((0.0 if f == zero else weights[f]) * fb.values[f] for f in weights)
            base = (feat + SEMANTIC_WEIGHT * (0.0 if zero == "semantic_sim" else fb.semantic_sim)) / (1 + SEMANTIC_WEIGHT)
            base = max(0.0, min(1.0, base))
            penalty = scoring._disqualifier_penalty(rep)
            score = base * fb.behavioral_mult * penalty
            scored.append((score, v.candidate_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        ranked = [grades.get(cid, 0) for _, cid in scored[:100]]
        return metrics.composite(ranked, all_grades)["composite"]

    base = composite_with(uniform)
    print(f"uniform-weight composite: {base:.4f}\n")
    drops = {}
    for f in RERANK_FEATURES:
        c = composite_with(uniform, zero=f)
        drops[f] = max(0.0, base - c)
        print(f"  drop without {f:18s}: {base - c:+.4f}")

    total = sum(drops.values())
    print("\nDERIVED WEIGHTS (normalised marginal contribution):")
    if total <= 0:
        print("  all drops ~0 on these labels; falling back to equal weights")
        derived = {f: round(1 / len(RERANK_FEATURES), 3) for f in RERANK_FEATURES}
    else:
        derived = {f: round(drops[f] / total, 3) for f in RERANK_FEATURES}
    for f, w in sorted(derived.items(), key=lambda x: -x[1]):
        print(f"  {f:18s} {w}")


if __name__ == "__main__":
    main()
