"""
diagnose.py — Sanity-check the data + integrity layers against the real pool.

Not part of the ranking pipeline; a developer tool to print honeypot/disqualifier
counts and spot-check a few flagged profiles. Run:
    python scripts/diagnose.py --candidates path/to/candidates.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker.data import stream_raw, build_view  # noqa: E402
from redrob_ranker.integrity import assess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    hp = 0
    flags: dict[str, int] = {}
    hp_examples: list[tuple[str, list[str]]] = []
    n = 0
    for raw in stream_raw(args.candidates):
        if args.limit and n >= args.limit:
            break
        v = build_view(raw)
        r = assess(v)
        n += 1
        if r.is_honeypot:
            hp += 1
            if len(hp_examples) < 5:
                hp_examples.append((v.candidate_id, r.honeypot_reasons))
        for k in r.disqualifier_flags:
            flags[k] = flags.get(k, 0) + 1

    print(f"processed: {n}")
    print(f"honeypots flagged (hard gate): {hp}")
    print("disqualifier flag counts:")
    for k, c in sorted(flags.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {c}")
    print("\nhoneypot examples:")
    for cid, reasons in hp_examples:
        print(f"  {cid}: {reasons}")


if __name__ == "__main__":
    main()
