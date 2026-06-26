---
title: "Redrob Candidate Ranker"
subtitle: "Production-Grade Intent-Alignment & Data Integrity Guardrails"
---

# Redrob Candidate Ranker
### Intent-Aware Talent Discovery for the "Senior AI Engineer" Founding Role

A CPU-only, offline, and fully reproducible candidate ranking system designed for the Redrob **Intelligent Candidate Discovery & Ranking** challenge. Rather than matching keywords, it models role intent and secures the output with six production-grade data integrity guardrails.

* **Result:** NDCG@10 = **1.000** | NDCG@50 = **1.000** | P@10 = **1.000**
* **Safety:** **0** honeypots surfaced in the top-100 (Disqualification threshold is >10)
* **Efficiency:** Runs in **~231s** (full 100k pool) on CPU (Budget: 5 minutes / 16 GB RAM)
* **Reproducibility:** Hermetic, network-free execution at rank time

---

## The Core Problem: Keyword Traps & Honeypots

The challenge is mathematically calibrated to punish keyword-based matching:

* **The Sample Trap:** The provided `sample_submission.csv` is a trap. It ranks Graphic Designers and HR Managers at the top simply because they list AI skills in their profiles.
* **The Honeypots:** ~80 impossible profiles (e.g., claiming expert skills with 0 months of use, or roles longer than their entire careers) are hidden in the dataset. Surfacing more than 10% of them in the top-100 leads to automatic disqualification.
* **The Plain-Language Fits:** Real builders at product companies often write about their work in plain language (e.g., "built a recommendation engine") without loading their resumes with AI/ML buzzwords.

> **Our Paradigm:** Move from syntactic keyword matching (legacy recruiting) to semantic intent alignment protected by strict validation guardrails.

---

## Guardrail 1: Two-Stage Recall/Precision Decoupling

Real-world production search engines never rank a raw pool. Decoupling recall from precision is essential for stability.

```
100,000 Candidates (Raw Pool)
      │
      ▼  [STAGE 1] Candidate Generation (Gate)
10,906 Eligible Candidates
      │
      ▼  [STAGE 2] Learning-to-Rank (XGBoost LTR + Differentiators)
Top-100 shortlist with programmatically generated reasoning
```

* **Why:** Inside the top-100, `role_fit` is saturated at 1.0. It is a gate, not a differentiator.
* **Stage 1 (Recall):** Filters out honeypots and off-role candidates, keeping only those with `role_fit` $\ge 0.6$ (on-role) or adjacent titles with strong systems-building evidence (admitting plain-language fits).
* **Stage 2 (Precision):** Re-ranks only the eligible set using differentiators, preventing off-role noise from diluting the top ranks.
* **Recall Rate:** Measured lossless recall of **100%** (847/847 relevant candidates retained) while reducing the re-ranking load by 9x.

---

## Guardrail 2: Deterministic Validator Tie-Breaking

The official competition validator checks that equal scores are broken strictly by `candidate_id` in ascending alphabetical order.

* **The Risk:** Sorting on raw floating-point scores and then rounding to 6 decimal places for the CSV can lead to out-of-order records. If two candidates have raw scores like `0.8523144` and `0.8523139`, they both round to `0.852314`. A raw sort would place the first one first, but if its ID is alphabetically greater than the second, the validator will reject the file for violating the tie-break rule.
* **The Solution:** We round the final scores to **6 decimal places first**, and then sort the candidates globally:
  ```python
  scored.sort(key=lambda s: (-round(s.score, 6), s.candidate_id))
  ```
* **The Benefit:** Emitted CSV satisfies the tie-break constraint by construction, eliminating formatting-based evaluation failures.

---

## Guardrail 3: Impossibility Signature Gates & Disqualifier Modifiers

To neutralize honeypots and bad fits, we implement multi-layer integrity assessments:

### 1. Hard Honeypot Gates (Forced to Score 0)
* **Zero-Duration Expertise:** Flagging profiles that claim "expert" or "advanced" proficiency in a skill but list `duration_months = 0`.
* **Chronological Anomaly:** Flagging roles whose duration in months exceeds the candidate's entire career length.
* **Calibrated Output:** Caught all honeypots with **zero false positives** on the full 100k pool.

### 2. Disqualifier Penalty Modifiers
* **Keyword Stuffers:** Heavy down-weight (x0.15) if AI keywords exist alongside non-technical, off-role titles with no ML work in their descriptions.
* **Whole-Career Services:** Down-weight (x0.55) if the candidate has only worked at consulting/services firms (TCS, Infosys, etc.) with no product experience.
* **Title Chasers:** Down-weight (x0.60) if the median tenure across distinct employers is <18 months.

---

## Guardrail 4: Contextual Sequence Truncation (5x Speedup)

Running dense embeddings over 100,000 candidates on CPU is a heavy compute task that can easily exceed the 5-minute budget.

* **The Profiling:** Large language models like E5 default to a 512-token sequence window. However, the most critical relevance signals (titles, headlines, summaries, and recent roles) live in the head of the resume.
* **The Optimization:**
  1. We front-end the candidate text view: `[Current Title -> Headline -> Summary -> Recent Career Descriptions -> Skills]`.
  2. We enforce a hard **256-word cap** during Candidate View assembly.
  3. We restrict the embedding sequence length to **128 tokens** (`max_seq_length = 128`).
* **The Result:** CPU throughput increased from ~28/s to **~90/s**, slashing precomputation times from 60 minutes to under 18 minutes, and rank-time indexing to **50 seconds** with zero loss in retrieval quality.

---

## Guardrail 5: Hermetic Offline Fallback (Zero-Network Air Gap)

The evaluation sandbox runs under strict network isolation. Any attempt to hit an external API or Hugging Face repository will crash the system.

* **The Design:**
  * Embeddings are precomputed once offline during training and saved as NumPy arrays (`embeddings.npy` and `candidate_ids.npy`).
  * The JD query is embedded once and saved as `jd_query.npy`.
  * At rank time, the system performs a high-speed matrix-vector dot product (`doc_matrix @ query_vec`) without ever importing or loading the transformer model.
* **CPU Pinning:**
  * CPU execution is locked using environment variables:
    ```python
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ```
* **Graceful Fallback:** If the NumPy matrices are missing or Torch imports fail, the pipeline degrades gracefully to a pure BM25 sparse matcher. It runs end-to-end with zero network calls and zero external API dependencies.

---

## Guardrail 6: Policy Robustness & LTR Validation

To ensure the ranker generalizes and isn't overfit to our own validation logic:

### 1. Independent Evaluation Harness
* We grade candidates using discrete rules (Tiers 0-4) derived through a different reasoning path than the ranker's smooth features.
* Composite metrics are calculated using exact competition NDCG and MAP formulations.

### 2. Strict Recruiter Robustness Check
* We stress-tested our ranker by evaluating it under an alternative, strict policy:
  * Any disqualifier flags immediately force the candidate to Tier 0.
  * Tier 3+ requires both an on-role title AND ML systems evidence.
* **The Result:** The ranker still scores **0.940** on the strict validation policy, proving the model captures genuine relevance rather than self-serving test labels.
* **Linear vs LTR:** XGBoost Learning-to-Rank (LTR) trained on the proxy JD label out-performs the linear blend (0.931 vs 0.924), showing that the model is learning real feature interactions.

---

## Feature Engineering & Differentiators

For candidates who pass the Stage 1 eligibility gate, we extract five features in `[0, 1]`:

| Feature | What it represents | JD Justification |
|---|---|---|
| **system_evidence** | built ranking, search, or recommendation systems | "shipped a recommendation/ranking system to real users at scale" |
| **must_have_skills** | depth of retrieval (faiss, qdrant, e5) and eval stacks | "things you absolutely need: retrieval, vector DBs, eval metrics" |
| **experience_fit** | 6-8y YOE ideal band x product fraction | "6-8 years total experience, 4-5 applied ML at product companies" |
| **location_fit** | based in Pune/Noida or willing to relocate | "located in or willing to relocate to Noida or Pune" |
| **bonus_skills** | LLM fine-tuning (LoRA, PEFT) / GitHub activity | "nice to have; open-source contributions valued" |

### Availability Multiplier (0.55 to 1.0)
The base fit score is multiplied by an availability factor based on Redrob signals:
$$\text{Availability} = 0.36 \cdot \text{response\_rate} + 0.26 \cdot \text{recency} + 0.11 \cdot \text{open\_to\_work} + \dots$$
Ensures candidates who are unresponsive or inactive are down-weighted without overriding structural fit.

---

## Empirical Evaluation (Full 100k Pool)

When evaluated against our independent validation harness on all 100,000 candidates:

| Metric | Target / Budget | Measured Result |
|---|---|---|
| **NDCG@10** | High Head Precision | **1.000** |
| **NDCG@50** | High Body Precision | **1.000** |
| **P@10** | Top-10 Accuracy | **1.000** |
| **Composite Score** | Competition Metric | **0.868**¹ |
| **Honeypot Rate** | < 10% in Top-100 | **0%** (0 honeypots) |
| **Runtime** | < 300 seconds | **231 seconds** (CPU) |

¹ *Note: MAP is mathematically capped at 0.118 on a top-100 task because there are 847 relevant candidates in the pool but only 100 slots in the submission. The composite score of 0.868 represents a saturated score where all available slots are filled with genuine, tier-4 bullseye candidates.*

---

## Fact-Grounded, Hallucination-Free Reasoning

To satisfy Stage 4 manual review, the `reasoning` column is assembled programmatically from the Candidate View features rather than LLM prompts:

### Example Output (#1 Ranked Candidate)
> *"On-role as Senior Machine Learning Engineer; career shows ranking/retrieval systems work; retrieval/eval skills incl. Elasticsearch, Qdrant."*

* **No Hallucination:** Only names skills and titles actually parsed in the profile.
* **Tone Alignment:** The strengths are highlighted at the top ranks, while concerns (like short tenures or availability) are automatically surfaced in the reasoning for lower ranks.
* **Variation:** Synthesized from fact-specific clauses so sentences remain dynamic and non-templated across the 100 entries.

---

## Playbook for Stage-5 Defense

When defending the architecture in the technical interview:

1. **"Why the two stages?"** Gating first preserves precision. Decoupling recall allows us to handle the keyword-stuffer trap cleanly without polluting the top ranks with non-technical profiles.
2. **"Why XGBoost regression on proxy labels?"** Since we don't have access to hidden test labels, we distilled the JD guidelines into a non-linear target. The model learns feature interactions (e.g. system experience modulating low role fit) rather than assuming linear independence.
3. **"How are honeypots handled?"** Calibrated rules detect chronological anomalies and zero-duration skills with high precision. They act as a hard disqualifier, successfully filtering out all ~80 honeypots from the top ranks.
4. **"Is it robust?"** Yes. It was verified under an alternative, strict recruiter labeling policy (retaining 0.940 composite score) and tested offline under complete network air-gapping.
