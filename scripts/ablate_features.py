"""
ablate_features.py — Leave-one-out feature ablation on the full pool.

Turns "we chose these weights by judgement" into "we measured each feature's
marginal contribution." For each feature we zero it out (set its value to 0 for
every candidate) and re-rank, then report the drop in composite vs the full
model. A feature whose removal hurts a lot is load-bearing; one whose removal
does nothing is a candidate for pruning.

This is the analysis that lets us defend the design at interview: the weight
ORDERING is backed by measured importance, not vibes.

    python scripts/ablate_features.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from redrob_ranker import jd_spec, ltr, metrics, scoring  # noqa: E402
from redrob_ranker.data import build_view, stream_raw  # noqa: E402
from redrob_ranker.eval_labels import relevance_tier  # noqa: E402
from redrob_ranker.features import FEATURE_NAMES, extract  # noqa: E402
from redrob_ranker.integrity import assess  # noqa: E402
from redrob_ranker import retrieval  # noqa: E402


def rank_with_overrides(views, fbs, reports, zero_feature=None, zero_semantic=False,
                        zero_behavioral=False):
    """Re-score with one signal ablated; return ordered candidate_ids.

    Applies the Stage-1 gate exactly like the real ranker, so the ablation
    reflects the two-stage pipeline (off-role candidates are forced to 0)."""
    import copy
    from redrob_ranker.candidate_generation import passes_gate
    scored = []
    for v, fb0, rep in zip(views, fbs, reports):
        fb = copy.copy(fb0)
        fb.values = dict(fb0.values)
        if zero_feature:
            fb.values[zero_feature] = 0.0
        if zero_semantic:
            fb.semantic_sim = 0.0
        bmult = 1.0 if zero_behavioral else fb.behavioral_mult
        fb.behavioral_mult = bmult
        eligible = passes_gate(v, fb0, rep).eligible
        sc = scoring.score_candidate(v, fb, rep, eligible=eligible)
        scored.append(sc)
    scored.sort(key=lambda s: (-s.score, s.candidate_id))
    return scored


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
        views.append(v)
        fbs.append(extract(v))
        reports.append(assess(v))
        grades[v.candidate_id] = relevance_tier(v)
    all_grades = list(grades.values())

    # attach dense + bm25 (same as the ranker)
    adir = Path(args.artifacts)
    emb = ids = None
    if (adir / "embeddings.npy").exists():
        emb = np.load(adir / "embeddings.npy")
        ids = np.load(adir / "candidate_ids.npy", allow_pickle=True)
    from redrob_ranker.ranker import _attach_semantic
    _attach_semantic(views, fbs, emb, ids, True, lambda *_: None, embed_live=False)

    def score_of(scored):
        ranked = [grades.get(s.candidate_id, 0) for s in scored[:100]]
        return metrics.composite(ranked, all_grades)["composite"]

    base = score_of(rank_with_overrides(views, fbs, reports))
    print(f"FULL MODEL composite: {base:.4f}\n")
    print(f"{'ablated signal':<22} {'composite':>10} {'drop':>9}")
    print("-" * 44)
    rows = []
    for f in FEATURE_NAMES:
        c = score_of(rank_with_overrides(views, fbs, reports, zero_feature=f))
        rows.append((f, c, base - c))
    rows.append(("semantic_sim", (s := score_of(rank_with_overrides(views, fbs, reports, zero_semantic=True))), base - s))
    rows.append(("behavioral_mult", (s := score_of(rank_with_overrides(views, fbs, reports, zero_behavioral=True))), base - s))
    for name, comp, drop in sorted(rows, key=lambda x: -x[2]):
        print(f"{name:<22} {comp:>10.4f} {drop:>+9.4f}")


if __name__ == "__main__":
    main()
