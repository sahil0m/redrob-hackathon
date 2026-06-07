"""
app.py — Sandbox demo for the Redrob ranker (Streamlit / HuggingFace Spaces).

Satisfies the spec's sandbox requirement (Section 10.5): a hosted environment
that accepts a small candidate sample (<=100), runs the ranking system
end-to-end on CPU within budget, and returns a ranked CSV.

Deploy on HuggingFace Spaces (Streamlit SDK) or Streamlit Cloud:
  - this file as the entrypoint
  - requirements.txt + the src/ package + scripts/ alongside it
For a small sample the demo embeds on the fly (no precomputed artifacts needed),
which is fast enough well within the budget for <=100 candidates.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # Spaces may fetch the model once
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from redrob_ranker.ranker import rank  # noqa: E402
from redrob_ranker.submission import HEADER, write_csv  # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — sandbox demo")
st.caption(
    "Intent-aware ranking for the Senior AI Engineer JD. Upload a small "
    "candidates.jsonl sample (≤100 rows) and the system ranks them best-fit first. "
    "CPU-only, no external API calls."
)

SAMPLE_HINT = (
    'Each line is one candidate JSON object matching candidate_schema.json. '
    'Use sample_candidates.json from the bundle (paste as JSONL or upload a .jsonl).'
)

uploaded = st.file_uploader("candidates.jsonl (≤100 lines)", type=["jsonl", "json", "txt"])
top_k = st.slider("How many to rank", min_value=5, max_value=100, value=20, step=5)
use_ltr = st.checkbox("Use XGBoost LTR layer (else interpretable linear blend)", value=True)

if uploaded is not None and st.button("Rank candidates"):
    raw = uploaded.read().decode("utf-8")
    # Accept either JSONL or a JSON array.
    lines = []
    stripped = raw.strip()
    if stripped.startswith("["):
        for obj in json.loads(stripped):
            lines.append(json.dumps(obj))
    else:
        lines = [ln for ln in raw.splitlines() if ln.strip()]
    lines = lines[:100]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(lines))
        tmp_path = tf.name

    with st.spinner(f"Ranking {len(lines)} candidates on CPU..."):
        rows = rank(
            candidates_path=tmp_path,
            artifacts_dir=None,          # no cache in the sandbox
            use_ltr=use_ltr,
            use_bm25=True,
            top_k=min(top_k, len(lines)),
            verbose=False,
            embed_live=True,             # embed the small sample on the fly
        )

    st.success(f"Ranked {len(rows)} candidates.")
    table = [{"rank": r.rank, "candidate_id": r.candidate_id,
              "score": round(r.score, 4), "reasoning": r.reasoning} for r in rows]
    st.dataframe(table, use_container_width=True)

    buf = io.StringIO()
    import csv
    w = csv.writer(buf)
    w.writerow(HEADER)
    for r in sorted(rows, key=lambda x: x.rank):
        w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])
    st.download_button("Download ranked CSV", buf.getvalue(), file_name="ranked_sample.csv",
                       mime="text/csv")
else:
    st.info(SAMPLE_HINT)
