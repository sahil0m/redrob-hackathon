#!/usr/bin/env python3
"""
rank.py — Single reproduce command for the Redrob candidate ranker.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces the top-100 ranking CSV. Loads precomputed embeddings from ./artifacts
if present (run scripts/precompute.py once first); otherwise falls back to a
BM25-only semantic signal so the pipeline still runs end-to-end with no setup.

CPU-only and offline by construction (see redrob_ranker.retrieval); designed to
finish within the spec's 5-minute / 16 GB / no-network budget on the full 100k
pool once embeddings are precomputed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Keep the run hermetic + CPU-only regardless of host (spec: no GPU, no network).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.ranker import rank  # noqa: E402
from redrob_ranker.submission import write_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank top-100 candidates for the Redrob JD.")
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl(.gz)")
    ap.add_argument("--out", default="submission.csv", help="output CSV path")
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parent / "artifacts"),
                    help="dir with precomputed embeddings.npy + candidate_ids.npy")
    ap.add_argument("--no-ltr", action="store_true", help="use the interpretable linear blend only")
    ap.add_argument("--no-bm25", action="store_true", help="disable the BM25 sparse signal")
    ap.add_argument("--limit", type=int, default=None, help="rank only the first N (testing)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows = rank(
        candidates_path=args.candidates,
        artifacts_dir=args.artifacts,
        use_ltr=not args.no_ltr,
        use_bm25=not args.no_bm25,
        limit=args.limit,
        verbose=not args.quiet,
    )
    write_csv(rows, args.out)
    if not args.quiet:
        print(f"\nWrote {len(rows)} rows to {args.out} in {time.time()-t0:.1f}s total.")
        print("Top 5:")
        for r in rows[:5]:
            print(f"  #{r.rank} {r.candidate_id} ({r.score:.4f}) — {r.reasoning}")


if __name__ == "__main__":
    main()
