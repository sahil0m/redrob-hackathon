"""
metrics.py — The competition's ranking metrics, implemented exactly.

composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10  (spec Section 4)

`ranked_grades` is the list of true relevance grades in the order our ranker
returned them (index 0 = our rank 1). `all_grades` is every candidate's grade,
used for the ideal DCG / MAP denominators.
"""

from __future__ import annotations

import math


def dcg(grades: list[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(grades[:k]):
        s += (2 ** g - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(ranked_grades: list[float], all_grades: list[float], k: int) -> float:
    ideal = sorted(all_grades, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked_grades, k) / idcg


def precision_at_k(ranked_grades: list[float], k: int, relevant_threshold: int = 3) -> float:
    top = ranked_grades[:k]
    if not top:
        return 0.0
    return sum(1 for g in top if g >= relevant_threshold) / len(top)


def average_precision(ranked_grades: list[float], all_grades: list[float],
                      relevant_threshold: int = 3) -> float:
    total_relevant = sum(1 for g in all_grades if g >= relevant_threshold)
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, g in enumerate(ranked_grades):
        if g >= relevant_threshold:
            hits += 1
            score += hits / (i + 1)
    return score / total_relevant


def composite(ranked_grades: list[float], all_grades: list[float]) -> dict[str, float]:
    n10 = ndcg_at_k(ranked_grades, all_grades, 10)
    n50 = ndcg_at_k(ranked_grades, all_grades, 50)
    mp = average_precision(ranked_grades, all_grades)
    p10 = precision_at_k(ranked_grades, 10)
    comp = 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10
    return {"NDCG@10": n10, "NDCG@50": n50, "MAP": mp, "P@10": p10, "composite": comp}
