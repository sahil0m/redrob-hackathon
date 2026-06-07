---
title: "Redrob Candidate Ranker"
subtitle: "Intent-aware ranking for the Senior AI Engineer role"
---

# Redrob Candidate Ranker
### Ranking 100,000 candidates the way a great recruiter would

A CPU-only, offline, fully-reproducible system that reads the job description,
models what the role *means*, and returns a trustworthy top-100 shortlist.

---

## The problem behind the problem

The challenge **looks** like "match candidates to a JD." It is actually a test of
whether your system *reads profiles* or just *matches keywords*.

The organizers built the dataset to punish keyword matching:

- The provided `sample_submission.csv` ranks **HR Managers and Graphic Designers**
  at the top — because they *list* AI skills. It is the trap, not a baseline.
- The JD says it directly: a **Marketing Manager with every AI keyword is not a
  fit**; a **data engineer who built a recommender at a product company is** — even
  with no buzzwords.
- ~80 **honeypots** (impossible profiles) are forced to tier 0; **>10% in your
  top-100 disqualifies you**.

> Our north star: **read the profile, not the vocabulary.**

---

## What the JD actually asks for

We distilled the JD into an explicit, auditable rubric (every weight in our code
cites a JD sentence):

**Wants:** 6–8 yrs (4–5 applied ML at *product* companies) · shipped an
end-to-end **ranking / search / recommendation** system · embeddings-retrieval +
vector-DB + ranking-**evaluation** (NDCG/MRR/MAP/A-B) · Pune/Noida or willing to
relocate · actually reachable on-platform.

**Explicit anti-patterns:** keyword stuffers · title-chasers (1.5-yr hops) ·
consulting-only careers · pure researchers · CV/speech/robotics without NLP/IR ·
LangChain-only "AI experience."

---

## Architecture — explicit two-stage retrieval

Like every real production ranker, we separate **recall** from **precision**.
(A leave-one-out ablation proved why: inside the top-100, `role_fit` is 1.0 for
everyone — it decides *entry*, not *order*.)

```
candidates.jsonl
  → data         stream + flatten (dense & sparse text per candidate)
  → features     interpretable [0,1] features, each tied to a JD line
  → integrity    honeypot detection + JD disqualifier flags
  → retrieval    hybrid JD-similarity: e5-small dense (precomputed) + BM25

  STAGE 1  candidate generation — data-driven gate (role_fit + integrity).
           100k → 10.9k eligible, 100% recall of relevant (measured, lossless).
  STAGE 2  re-ranking — XGBoost LTR + linear blend over the DIFFERENTIATORS
           (experience, skill depth, systems evidence, semantic, location).

  → scoring      base × behavioral-availability × penalties ; gated-out → 0
  → reasoning    fact-grounded 1–2 sentence justification
  → submission   write + self-validate top-100 CSV
```

**Embeddings precomputed once offline**; the ranking step loads the cached matrix
and finishes in ~45 s — well within the 5-min / 16 GB / CPU / no-network budget.

**No arbitrary weights, no LLM at rank time.** Re-rank weights are *informed by
the ablation* (measured marginal contribution), not a "0.4 + 0.3 + 0.3" guess —
and no hosted LLM is called during ranking (that would fail the compute gate).

---

## The features (why each exists)

| Feature | What it captures | JD basis |
|---|---|---|
| **role_fit** | current + past **titles** | the decisive anti-keyword-stuffer signal |
| **system_evidence** | built ranking/retrieval/recsys (two-tier: strong vs supporting production-ML/A-B) | "shipped an end-to-end ranking/search/recommendation system" |
| **must_have_skills** | retrieval + eval stack, with an endorsement/duration **trust factor** | "things you absolutely need" |
| **experience_fit** | smooth 6–8y band × product-vs-services fraction | "5–9 yrs … not a requirement" |
| **bonus_skills** | LLM fine-tuning, LoRA/PEFT | "nice to have" |
| **location_fit** | Pune/Noida → Tier-1 → relocate | "located in or willing to relocate to Noida or Pune" |
| **behavioral_mult** | recency, response rate, open-to-work, verification | "not actually available — down-weight" |

`role_fit` **gates** competence: a perfect skill list with no on-role evidence
cannot score highly. This is what neutralises the keyword-stuffer trap.

---

## Handling the four traps

- **Keyword stuffers** → `role_fit` gates everything; an explicit penalty (×0.15)
  when AI skills coexist with an off-role title and no ML in the career text.
- **Honeypots** → a high-precision rule gate (expert skill with 0 months used; a
  role longer than the whole career) forces score 0. Calibrated on the full pool:
  catches the surgically-impossible profiles with **no false positives**.
  **Result: 0 honeypots in our top-100.**
- **Plain-language Tier-5s** → two-tier `system_evidence` + dense e5 retrieval on
  *intent* surface real builders who never write "RAG" above the keyword-stuffers.
- **Behavioral twins** → availability multiplier in [0.55, 1.0] re-orders equal
  candidates without ever overriding genuine fit.

---

## Why a hybrid retriever, and why XGBoost on top

- **Dense (e5-small-v2)** captures *intent* — surfaces the data engineer who
  "built a recommendation system" without the buzzword.
- **BM25** keeps us honest on the rare profiles that legitimately name the tools.
- **XGBoost LTR** learns the *non-linear interactions* the linear blend can't
  (role × experience, system-evidence rescue limits). We have no ground truth, so
  we train on a **proxy label distilled transparently from the JD's own rubric**;
  the interpretable linear blend is retained as a floor and sanity check.

> Honest framing: the GBM is a learned refinement of a JD-grounded signal — not a
> black box, and it agrees with the linear model on the head of the ranking.

---

## Reasoning that survives manual review (Stage 4)

Each `reasoning` line is assembled **only from facts that drove the score**:

> *"On-role as ML Engineer; career shows recommendation/ranking systems work;
> retrieval skills incl. FAISS, OpenSearch. 6.0 yrs, in the JD's band."*

- Specific facts ✓  JD-connected ✓  honest concerns on weaker picks ✓
- No hallucinated skills ✓  varied, not templated ✓  tone tracks rank ✓

---

## We measured it — not "we think it's good"

We have no ground truth, so we built an **independent eval harness**: discrete
JD-rule relevance grades (derived through a *different* reasoning path than the
ranker) + the exact competition metrics.

| Metric (full 100k) | Value |
|---|---|
| NDCG@10 / NDCG@50 / P@10 | **1.000 / 1.000 / 1.000** |
| composite | **0.868** *(MAP capped: 847 relevant, 100 slots)* |
| honeypots in top-100 | **0** |
| rank-step runtime | **~46 s** (budget 5 min) |

**Stress-tested for honesty:**
- LTR **measurably** beats the linear blend (0.931 vs 0.924) — not fake sophistication.
- Re-graded under a **stricter, independent** label policy → still **0.940** (not overfit).
- Dense retrieval **re-orders** the bullseye set (6/20 same position vs BM25) —
  it earns its keep where finer ground truth rewards it.

---

## Reproducibility & compute

- **CPU-only**, enforced in code; **no network at rank time** (model from local cache).
- Embeddings: 100k × 384 float32 ≈ 150 MB (≪ 16 GB).
- One command reproduces the CSV:
  `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
- Dockerfile + Streamlit sandbox + unit tests included.

---

## Why this wins through all 5 stages

1. **Format** — self-validated against the official validator. ✓
2. **Scoring** — optimises the head (NDCG@10/50) via role-gated, intent-aware fit.
3. **Reproduction + honeypots** — CPU/offline within budget; **0 honeypots**.
4. **Manual review** — interpretable features, real git iteration, grounded reasoning.
5. **Defend-your-work** — every weight cites a JD sentence; nothing is unexplainable.

> Built to be **read, reproduced, and defended** — not just to score.
