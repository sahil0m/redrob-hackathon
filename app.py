"""
app.py — Redrob Candidate Ranker, Streamlit demo.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from redrob_ranker.ranker import rank
from redrob_ranker.submission import HEADER

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Ranker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal, clean CSS — no heavy frameworks ─────────────────────────────────
st.markdown("""
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header strip */
.header-strip {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 20px 28px 16px;
    margin-bottom: 20px;
}
.header-strip h1 { color: #e2e8f0; margin: 0; font-size: 1.6rem; font-weight: 600; }
.header-strip p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.85rem; }

/* Candidate card */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.card-title { font-size: 1rem; font-weight: 600; color: #1e293b; margin: 0 0 2px; }
.card-meta  { font-size: 0.8rem; color: #64748b; margin-bottom: 10px; }
.card-reasoning { font-size: 0.82rem; color: #374151; line-height: 1.5; }

/* Score bar */
.score-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.score-label { font-size: 0.72rem; color: #6b7280; width: 120px; flex-shrink: 0; }
.score-bar-bg { background: #f1f5f9; border-radius: 4px; height: 7px; flex: 1; }
.score-bar-fill { height: 7px; border-radius: 4px; background: #6366f1; }
.score-val { font-size: 0.72rem; color: #374151; width: 36px; text-align: right; flex-shrink: 0; }

/* Tag badges */
.badge {
    display: inline-block;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 9999px;
    margin: 2px 2px 0 0;
    font-weight: 500;
}
.badge-green  { background: #dcfce7; color: #166534; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-purple { background: #ede9fe; color: #5b21b6; }
.badge-orange { background: #ffedd5; color: #9a3412; }
.badge-gray   { background: #f1f5f9; color: #475569; }

/* Rank number */
.rank-num { font-size: 1.3rem; font-weight: 700; color: #6366f1; }
.rank-num.gold   { color: #d97706; }
.rank-num.silver { color: #6b7280; }
.rank-num.bronze { color: #b45309; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-strip">
    <h1>Redrob Candidate Ranker</h1>
    <p>Intent-aware ranking engine &nbsp;&middot;&nbsp; CPU-only, no API calls &nbsp;&middot;&nbsp;
       Two-stage pipeline (heuristic gate &rarr; XGBoost LTR + e5 embeddings)</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    uploaded = st.file_uploader(
        "Candidates file (.jsonl or .json)",
        type=["jsonl", "json", "txt"],
        help="Each line is one candidate JSON object. Max 100 rows in sandbox mode."
    )

    top_k = st.slider("Candidates to rank", min_value=5, max_value=50, value=20, step=5)
    use_ltr = st.checkbox("XGBoost LTR layer", value=True,
                          help="Off = interpretable linear blend only")
    use_bm25 = st.checkbox("BM25 sparse signal", value=True)

    st.markdown("---")
    st.markdown("""
    **How it works**

    1. Heuristic role-fit gate eliminates off-role candidates
    2. Hybrid BM25 + dense (e5-small-v2) retrieval computes JD similarity
    3. XGBoost LTR re-ranks the eligible set on structured features
    4. Behavioral multiplier adjusts for platform activity signals

    **New features**
    - Chronological time-decay on career history
    - Indian tier-1 institute + unicorn pedigree signal
    - Cold-start skill inference with authenticity gating
    """)


# ── Helper: render one candidate card ────────────────────────────────────────
def _badge(label: str, kind: str = "gray") -> str:
    return f'<span class="badge badge-{kind}">{label}</span>'


def _score_bar(label: str, value: float) -> str:
    pct = int(value * 100)
    return f"""
    <div class="score-row">
        <span class="score-label">{label}</span>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%"></div>
        </div>
        <span class="score-val">{value:.2f}</span>
    </div>"""


def _rank_class(r: int) -> str:
    if r == 1: return "rank-num gold"
    if r == 2: return "rank-num silver"
    if r == 3: return "rank-num bronze"
    return "rank-num"


def _parse_reasoning_tags(reasoning_text: str) -> list[tuple[str, str]]:
    """Extract key signals from the reasoning string as (label, badge_kind) tuples."""
    tags = []
    r = reasoning_text.lower()
    if "pedigree" in r or "iit" in r or "nit" in r or "bits" in r:
        tags.append(("Tier-1 Pedigree", "purple"))
    if "unicorn" in r or "flipkart" in r or "swiggy" in r or "razorpay" in r:
        tags.append(("Unicorn Alumni", "purple"))
    if "decay" in r or "recent" in r or "current role" in r:
        tags.append(("Recent ML work", "green"))
    if "honeypot" not in r and "ineligible" not in r:
        if "notice" in r and "30" in r:
            tags.append(("Short notice", "green"))
        if "active" in r or "open to work" in r:
            tags.append(("Actively looking", "blue"))
    if "retrieval" in r or "ranking" in r or "search" in r or "recsys" in r:
        tags.append(("Ranking/Search systems", "blue"))
    if "ltr" in r or "learning to rank" in r or "ndcg" in r:
        tags.append(("LTR / Eval expertise", "blue"))
    if "llm" in r or "rag" in r or "fine-tun" in r:
        tags.append(("LLM / Fine-tuning", "orange"))
    if "github" in r or "open-source" in r:
        tags.append(("Open source", "gray"))
    return tags


def render_card(row, feature_vals: dict | None = None):
    """Render a candidate card using native Streamlit components."""
    tags = _parse_reasoning_tags(row.reasoning)
    tags_html = " ".join(_badge(t, k) for t, k in tags) if tags else ""

    rank_colors = {1: "#d97706", 2: "#6b7280", 3: "#b45309"}
    rank_color  = rank_colors.get(row.rank, "#6366f1")

    with st.container():
        # Top row: rank number + ID + score
        col_rank, col_body = st.columns([1, 11])

        with col_rank:
            st.markdown(
                f'<div style="font-size:1.5rem; font-weight:700; color:{rank_color}; '
                f'text-align:center; padding-top:6px;">#{row.rank}</div>',
                unsafe_allow_html=True,
            )

        with col_body:
            st.markdown(f"**{row.candidate_id}**")
            st.caption(f"Score: {row.score:.4f}")

            # Tags
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

            # Feature bars (if available)
            if feature_vals:
                bars = ""
                for label, key in [
                    ("Role fit",        "role_fit"),
                    ("System evidence", "system_evidence"),
                    ("Must-have skills","must_have_skills"),
                    ("Experience",      "experience_fit"),
                    ("Pedigree",        "premium_pedigree"),
                    ("Location",        "location_fit"),
                ]:
                    bars += _score_bar(label, feature_vals.get(key, 0.0))
                st.markdown(bars, unsafe_allow_html=True)

            # Reasoning — plain markdown, no HTML wrapping
            st.markdown(
                f'<p style="font-size:0.82rem; color:#374151; margin-top:6px;">'
                f'<span style="font-size:0.7rem; text-transform:uppercase; '
                f'letter-spacing:.05em; color:#6b7280; font-weight:600;">Why ranked here</span>'
                f'<br>{row.reasoning}</p>',
                unsafe_allow_html=True,
            )

        st.divider()


# ── Main area ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("Upload a `candidates.jsonl` file in the sidebar to get started.")
    st.markdown("""
    **What to upload:** Each line should be a JSON object matching the candidate schema.
    You can use the sample file from the challenge bundle, or the synthetic profiles
    from `scripts/manual_test.py`.

    **Expected fields per candidate:**
    ```
    candidate_id, profile (with current_title, years_of_experience, location…),
    career_history, education, skills, redrob_signals
    ```
    """)
    st.stop()

# Parse uploaded file
raw_text = uploaded.read().decode("utf-8")
lines = []
stripped = raw_text.strip()
if stripped.startswith("["):
    try:
        for obj in json.loads(stripped):
            lines.append(json.dumps(obj))
    except json.JSONDecodeError:
        st.error("Could not parse the file as a JSON array. Try a .jsonl file instead.")
        st.stop()
else:
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]

lines = lines[:100]
st.caption(f"Loaded **{len(lines)}** candidates from `{uploaded.name}`.")

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run = st.button("Rank candidates", type="primary", use_container_width=True)

if not run:
    st.stop()

# Write to temp file and rank
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
    tf.write("\n".join(lines))
    tmp_path = tf.name

with st.spinner(f"Running pipeline on {len(lines)} candidates (CPU-only)..."):
    rows = rank(
        candidates_path=tmp_path,
        artifacts_dir=None,
        use_ltr=use_ltr,
        use_bm25=use_bm25,
        top_k=min(top_k, len(lines)),
        verbose=False,
        embed_live=True,
    )

# ── Results header ────────────────────────────────────────────────────────────
st.success(f"Done. Showing top **{len(rows)}** candidates.")

tab_cards, tab_table = st.tabs(["Ranked cards", "Raw table"])

# ── Tab 1: Rich candidate cards ───────────────────────────────────────────────
with tab_cards:
    for row in rows:
        render_card(row)

# ── Tab 2: Plain dataframe + CSV download ────────────────────────────────────
with tab_table:
    table = [
        {
            "rank": r.rank,
            "candidate_id": r.candidate_id,
            "score": round(r.score, 4),
            "reasoning": r.reasoning,
        }
        for r in rows
    ]
    st.dataframe(table, use_container_width=True)

    buf = io.StringIO()
    import csv
    w = csv.writer(buf)
    w.writerow(HEADER)
    for r in sorted(rows, key=lambda x: x.rank):
        w.writerow([r.candidate_id, r.rank, f"{r.score:.6f}", r.reasoning])
    st.download_button(
        "Download CSV",
        buf.getvalue(),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )
