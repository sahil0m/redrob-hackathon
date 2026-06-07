"""
submission.py — Write and self-validate the top-100 CSV.

The CSV must satisfy the bundled validate_submission.py exactly: header
`candidate_id,rank,score,reasoning`, 100 data rows, ranks 1..100 unique, unique
ids, score non-increasing by rank, ties broken by candidate_id ascending. We
enforce all of that here before writing, and re-check after, so we never upload
an invalid file.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .ranker import RankRow

HEADER = ["candidate_id", "rank", "score", "reasoning"]
_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


def validate_rows(rows: list[RankRow]) -> list[str]:
    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"expected 100 rows, got {len(rows)}")

    seen_ids, seen_ranks = set(), set()
    prev_score = None
    for r in sorted(rows, key=lambda x: x.rank):
        if not _ID_RE.match(r.candidate_id):
            errors.append(f"bad candidate_id: {r.candidate_id}")
        if r.candidate_id in seen_ids:
            errors.append(f"duplicate candidate_id: {r.candidate_id}")
        seen_ids.add(r.candidate_id)
        if not (1 <= r.rank <= 100):
            errors.append(f"rank out of range: {r.rank}")
        if r.rank in seen_ranks:
            errors.append(f"duplicate rank: {r.rank}")
        seen_ranks.add(r.rank)
        if prev_score is not None and r.score > prev_score + 1e-12:
            errors.append(f"score increases at rank {r.rank} ({r.score} > {prev_score})")
        prev_score = r.score

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"missing ranks: {sorted(missing)}")
    return errors


def write_csv(rows: list[RankRow], out_path: str | Path) -> None:
    errors = validate_rows(rows)
    if errors:
        raise ValueError("Refusing to write invalid submission:\n  " + "\n  ".join(errors))

    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in sorted(rows, key=lambda x: x.rank):
            # Format score with fixed precision so the non-increasing check is
            # unambiguous and the file is stable across runs.
            w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])
