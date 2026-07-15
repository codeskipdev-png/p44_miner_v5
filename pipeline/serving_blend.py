"""ServingBlend: the deployed 3-way model (v1 + live-var + ks0.90).

Rationale (2026-07-15 evaluation, see README):
- labeled live-size proxy (80-100 hand pooled held-out): 0.8821 vs uid227 0.8513
- benchmark held-out: 0.9226 (uid227: 0.9362) - we trade a little in-distribution
  sharpness for live-regime robustness, which is what the validator scores.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .features import feature_names, rows_to_matrix


class ServingBlend:
    def __init__(self, members: List[Tuple[str, object]]):
        self.members = members  # [(tag, RankBlend)]
        self.all_cols = feature_names()
        self._col_idx = {
            tag: [self.all_cols.index(c) for c in blend.cols]
            for tag, blend in members
        }
        self.meta: Dict = {}

    @staticmethod
    def _rank01(p: np.ndarray) -> np.ndarray:
        if len(p) <= 1:
            return np.full(len(p), 0.5)
        return np.argsort(np.argsort(p, kind="mergesort")) / (len(p) - 1)

    def member_probs(self, X_all: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            tag: blend.score_prob(X_all[:, self._col_idx[tag]])
            for tag, blend in self.members
        }

    def score_prob(self, X_all: np.ndarray) -> np.ndarray:
        probs = self.member_probs(X_all)
        return sum(probs.values()) / len(probs)

    def score_rank(self, X_all: np.ndarray) -> np.ndarray:
        probs = self.member_probs(X_all)
        return sum(self._rank01(p) for p in probs.values()) / len(probs)

    def featurize(self, rows: List[Dict[str, float]]) -> np.ndarray:
        return rows_to_matrix(rows, self.all_cols)

    @classmethod
    def build(cls, artifact_dir: Path) -> "ServingBlend":
        members = []
        for tag, fname in [
            ("v1", "rankblend_v1.pkl"),
            ("live-var", "sweep_live-var_only.pkl"),
            ("ks0.90", "sweep_ks0.90.pkl"),
        ]:
            with (artifact_dir / fname).open("rb") as f:
                members.append((tag, pickle.load(f)))
        return cls(members)
