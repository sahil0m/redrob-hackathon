"""
precompute.py — Offline pre-computation of candidate embeddings.

The submission spec permits pre-computation outside the 5-minute ranking budget,
as long as the *ranking step* that produces submission.csv stays within it.
Embedding 100k profiles with e5-small on CPU takes a few minutes, so we do it
once here and cache the result; rank.py then loads the cached matrix and only
embeds the single JD query at rank time (milliseconds).

Outputs (to --out-dir, default ./artifacts):
    embeddings.npy   float32 [N, 384]  L2-normalised e5 passage embeddings
    candidate_ids.npy   the candidate_id for each row, in the same order

Run:
    python scripts/precompute.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker import retrieval  # noqa: E402
from redrob_ranker.data import build_view, stream_raw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="for quick local tests")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading candidate profiles...")
    t0 = time.time()
    ids: list[str] = []
    texts: list[str] = []
    for i, raw in enumerate(stream_raw(args.candidates)):
        if args.limit and i >= args.limit:
            break
        v = build_view(raw)
        ids.append(v.candidate_id)
        texts.append(v.dense_text)
    print(f"  {len(ids)} profiles in {time.time() - t0:.1f}s")

    print("Loading e5-small-v2 (CPU)...")
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    model = retrieval.load_embedder()

    print("Embedding profiles...")
    t0 = time.time()
    emb = retrieval.embed_passages(model, texts, batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"  embedded {len(ids)} in {dt:.1f}s ({len(ids)/dt:.0f}/s), shape {emb.shape}")

    # Also cache the JD query embedding so the ranking step is a pure dot product
    # with NO model load at rank time (faster, and robust to a broken torch env).
    from redrob_ranker import jd_spec
    qvec = retrieval.embed_query(model, jd_spec.JD_QUERY_TEXT)

    np.save(out_dir / "embeddings.npy", emb)
    np.save(out_dir / "candidate_ids.npy", np.array(ids))
    np.save(out_dir / "jd_query.npy", qvec)
    print(f"Saved embeddings.npy ({emb.nbytes/1e6:.0f} MB), candidate_ids.npy, "
          f"and jd_query.npy to {out_dir}")


if __name__ == "__main__":
    main()
