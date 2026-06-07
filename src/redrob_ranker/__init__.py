"""Redrob candidate ranker — intent-aware, CPU-only, reproducible top-100 ranking.

Pipeline (see README.md):
    data -> integrity gate -> hybrid retrieval -> feature scoring (+ XGBoost LTR)
         -> reasoning -> top-100 CSV
"""

__version__ = "0.1.0"
