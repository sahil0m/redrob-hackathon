"""Print the feature breakdown for specific candidate IDs. Debug tool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker.data import stream_raw, build_view  # noqa: E402
from redrob_ranker.features import extract  # noqa: E402
from redrob_ranker.integrity import assess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    args = ap.parse_args()
    want = set(args.ids)

    for raw in stream_raw(args.candidates):
        cid = raw.get("candidate_id")
        if cid not in want:
            continue
        v = build_view(raw)
        fb = extract(v)
        rep = assess(v)
        print(f"\n=== {cid} | {v.current_title} | {v.years_of_experience}y | {v.location}, {v.country} ===")
        for k, val in fb.values.items():
            print(f"  {k:18s} {val:.3f}")
        print(f"  behavioral_mult    {fb.behavioral_mult:.3f}  {fb.behavioral_components}")
        print(f"  honeypot: {rep.is_honeypot} {rep.honeypot_reasons}")
        print(f"  disqualifiers: {list(rep.disqualifier_flags.keys())}")
        want.discard(cid)
        if not want:
            break


if __name__ == "__main__":
    main()
