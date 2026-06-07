"""
robustness_check.py — Is the ranker genuinely good, or overfit to our own labels?

Our evaluate.py grades with eval_labels.relevance_tier and the ranker also encodes
JD logic, so a perfect score could be self-congratulation. This check stresses
that by re-grading under a DIFFERENT, stricter labeling policy and seeing whether
the ranker still scores well. If quality holds under a label definition we did
NOT design the ranker around, the result is trustworthy; if it collapses, we were
overfit.

Alternative policy ("strict recruiter"):
  * requires BOTH on-role title AND ml-systems evidence for any tier>=3
    (no "benefit of the doubt" for on-role-but-thin profiles),
  * demands in-band experience for tier 4,
  * treats ANY disqualifier as a hard tier-0 (not a soft knock-down).

    python scripts/robustness_check.py --candidates ./candidates.jsonl --limit 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import jd_spec, metrics  # noqa: E402
from redrob_ranker.data import CandidateView, build_view, stream_raw  # noqa: E402
from redrob_ranker.eval_labels import (_ADJACENT_TITLES, _CORE_AI_TITLES,  # noqa: E402
                                       _has_ml_systems_evidence, _in_band, _reachable)
from redrob_ranker.integrity import detect_disqualifiers, detect_honeypot  # noqa: E402
from redrob_ranker.ranker import rank  # noqa: E402


def strict_tier(c: CandidateView) -> int:
    if detect_honeypot(c)[0]:
        return 0
    if detect_disqualifiers(c):
        return 0  # strict: any disqualifier => irrelevant
    title = c.current_title.lower()
    is_core = any(t in title for t in _CORE_AI_TITLES)
    is_adj = any(t in title for t in _ADJACENT_TITLES)
    sysev = _has_ml_systems_evidence(c)
    if is_core and sysev and _in_band(c) and _reachable(c):
        return 4
    if (is_core or is_adj) and sysev:
        return 3
    if is_core or is_adj:
        return 2
    return 1 if (is_core or is_adj) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    grades, hp = {}, set()
    for i, raw in enumerate(stream_raw(args.candidates)):
        if args.limit and i >= args.limit:
            break
        v = build_view(raw)
        grades[v.candidate_id] = strict_tier(v)
        if detect_honeypot(v)[0]:
            hp.add(v.candidate_id)
    allg = list(grades.values())
    print(f"strict labels: tier>=3 = {sum(1 for g in allg if g>=3)}, honeypots = {len(hp)}")

    rows = rank(candidates_path=args.candidates, artifacts_dir=args.artifacts,
                use_ltr=True, use_bm25=True, top_k=100, limit=args.limit, verbose=False)
    ranked = [grades.get(r.candidate_id, 0) for r in sorted(rows, key=lambda x: x.rank)]
    m = metrics.composite(ranked, allg)
    print("\n=== ROBUSTNESS (strict-recruiter labels) ===")
    for k, v in m.items():
        print(f"  {k:10s} {v:.4f}")
    print(f"  honeypots in top-100: {sum(1 for r in rows if r.candidate_id in hp)}")
    print("  top-10 strict grades:", ranked[:10])


if __name__ == "__main__":
    main()
