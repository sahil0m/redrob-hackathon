"""
evaluate.py — Measure ranking quality against the independent synthetic labels.

This is the harness that turns "we think it's good" into numbers. It runs the
full ranker over a candidate set, grades every candidate with the independent
JD-rule labels (eval_labels.py), and reports the competition metrics plus the
honeypot rate in the top-100 (the Stage-3 disqualifier).

    python scripts/evaluate.py --candidates ./candidates.jsonl [--limit N] [--no-ltr] [--no-bm25]

Use --limit for fast iteration; run on the full pool before trusting a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import metrics  # noqa: E402
from redrob_ranker.data import build_view, stream_raw  # noqa: E402
from redrob_ranker.eval_labels import relevance_tier  # noqa: E402
from redrob_ranker.integrity import detect_honeypot  # noqa: E402
from redrob_ranker.ranker import rank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ltr", action="store_true")
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--embed-live", action="store_true")
    args = ap.parse_args()

    # 1. Grade every candidate with the independent labels + flag honeypots.
    grades: dict[str, int] = {}
    honeypots: set[str] = set()
    grade_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for i, raw in enumerate(stream_raw(args.candidates)):
        if args.limit and i >= args.limit:
            break
        v = build_view(raw)
        g = relevance_tier(v)
        grades[v.candidate_id] = g
        grade_hist[g] += 1
        if detect_honeypot(v)[0]:
            honeypots.add(v.candidate_id)
    all_grades = list(grades.values())
    print(f"Graded {len(grades)} candidates. Tier histogram: {grade_hist}")
    print(f"  relevant (tier>=3): {sum(1 for g in all_grades if g>=3)} | honeypots: {len(honeypots)}")

    # 2. Rank.
    rows = rank(
        candidates_path=args.candidates,
        artifacts_dir=args.artifacts,
        use_ltr=not args.no_ltr,
        use_bm25=not args.no_bm25,
        top_k=100,
        limit=args.limit,
        verbose=True,
        embed_live=args.embed_live,
    )

    # 3. Map our ranking to the true grades and compute metrics.
    ranked_grades = [grades.get(r.candidate_id, 0) for r in sorted(rows, key=lambda x: x.rank)]
    m = metrics.composite(ranked_grades, all_grades)

    hp_in_top100 = sum(1 for r in rows if r.candidate_id in honeypots)
    hp_in_top10 = sum(1 for r in sorted(rows, key=lambda x: x.rank)[:10] if r.candidate_id in honeypots)

    print("\n================ EVALUATION ================")
    for k, v in m.items():
        print(f"  {k:10s} {v:.4f}")
    print(f"  honeypots in top-100: {hp_in_top100}  (DISQUALIFY if >10)")
    print(f"  honeypots in top-10:  {hp_in_top10}")
    print("  top-10 true grades:  ", ranked_grades[:10])
    print("============================================")


if __name__ == "__main__":
    main()
