"""Rank-blended ensemble for Poker44 chunk classification.

Three decorrelated members fused by in-batch rank (calibration-agnostic,
directly optimizes the AP/recall ranking terms that dominate the reward):
  A. stacked GBDT: LightGBM + XGBoost + CatBoost + ExtraTrees -> LogisticRegression
  B. monotone LightGBM trio (sign constraints mined from cross-date Spearman)
  C. PCA -> MLP trio (different model family for decorrelation)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def mine_monotone_signs(
    X: np.ndarray,
    y: np.ndarray,
    dates: List[str],
    *,
    min_abs_rho: float = 0.15,
) -> List[int]:
    """Per-feature sign constraints that hold across every training date."""
    unique_dates = sorted(set(dates))
    dates_arr = np.asarray(dates)
    signs = np.zeros(X.shape[1], dtype=int)
    for j in range(X.shape[1]):
        rhos = []
        for d in unique_dates:
            mask = dates_arr == d
            if mask.sum() < 8 or len(set(y[mask])) < 2:
                continue
            rho = spearmanr(X[mask, j], y[mask]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
        if len(rhos) >= 3:
            arr = np.asarray(rhos)
            if abs(arr.mean()) >= min_abs_rho and (np.sign(arr) == np.sign(arr.mean())).mean() >= 0.8:
                signs[j] = int(np.sign(arr.mean()))
    return signs.tolist()


def build_stack(seed: int = 0) -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(
                n_estimators=500, num_leaves=96, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                random_state=seed, verbose=-1, n_jobs=4)),
            ("xgb", XGBClassifier(
                n_estimators=500, max_leaves=64, grow_policy="lossguide",
                learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                random_state=seed, n_jobs=4, eval_metric="logloss")),
            ("cat", CatBoostClassifier(
                iterations=600, depth=7, learning_rate=0.04,
                random_seed=seed, verbose=0, thread_count=4)),
            ("et", ExtraTreesClassifier(
                n_estimators=400, max_depth=18, min_samples_leaf=3,
                random_state=seed, n_jobs=4)),
        ],
        final_estimator=LogisticRegression(C=0.5, max_iter=2000),
        stack_method="predict_proba",
        cv=4,
        n_jobs=1,
    )


def build_mono(signs: List[int]) -> VotingClassifier:
    members = [
        (f"mlgb{s}", LGBMClassifier(
            n_estimators=450, num_leaves=63, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.8, min_child_samples=12,
            monotone_constraints=signs, random_state=100 + s,
            verbose=-1, n_jobs=4))
        for s in range(3)
    ]
    return VotingClassifier(members, voting="soft", n_jobs=1)


def build_mlp() -> VotingClassifier:
    members = [
        (f"mlp{s}", Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=56, random_state=200 + s)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(80,), alpha=1e-3, max_iter=600,
                early_stopping=True, random_state=200 + s)),
        ]))
        for s in range(3)
    ]
    return VotingClassifier(members, voting="soft", n_jobs=3)


class RankBlend:
    """Weighted rank fusion of the three members."""

    WEIGHTS = {"stack": 0.35, "mono": 0.30, "mlp": 0.35}

    def __init__(self):
        self.stack: Optional[StackingClassifier] = None
        self.mono: Optional[VotingClassifier] = None
        self.mlp: Optional[VotingClassifier] = None
        self.cols: List[str] = []
        self.meta: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray, dates: List[str], cols: List[str]):
        self.cols = list(cols)
        signs = mine_monotone_signs(X, y, dates)
        self.meta["n_monotone"] = int(np.count_nonzero(signs))
        self.stack = build_stack().fit(X, y)
        self.mono = build_mono(signs).fit(X, y)
        self.mlp = build_mlp().fit(X, y)
        return self

    @staticmethod
    def _rank01(p: np.ndarray) -> np.ndarray:
        if len(p) <= 1:
            return np.full_like(p, 0.5, dtype=float)
        order = np.argsort(np.argsort(p, kind="mergesort"))
        return order / (len(p) - 1)

    def member_probs(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "stack": self.stack.predict_proba(X)[:, 1],
            "mono": self.mono.predict_proba(X)[:, 1],
            "mlp": self.mlp.predict_proba(X)[:, 1],
        }

    def score(self, X: np.ndarray) -> np.ndarray:
        probs = self.member_probs(X)
        w = self.WEIGHTS
        num = sum(w[k] * self._rank01(p) for k, p in probs.items())
        return num / sum(w.values())

    def score_prob(self, X: np.ndarray) -> np.ndarray:
        """Probability-scale blend (batch-size independent, used for serving
        alongside rank fusion)."""
        probs = self.member_probs(X)
        w = self.WEIGHTS
        return sum(w[k] * p for k, p in probs.items()) / sum(w.values())
