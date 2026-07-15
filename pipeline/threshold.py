"""Threshold shaping: convert ranking quality into validator reward.

The reward has a hard gate at 0.5 (see 01_subnet_understanding/VALIDATOR_ANALYSIS.md):
  - zero reward if no bot chunk is scored >= 0.5;
  - human-safety decays once human FPR@0.5 exceeds 10%.
`shape_scores` applies ONE strictly monotone piecewise-linear map per batch, so
AP and recall@FPR (rank metrics) are untouched; only threshold placement moves.
"""
from __future__ import annotations

import numpy as np


def shape_adaptive(
    prob_scores: np.ndarray,
    rank_scores: np.ndarray,
    *,
    min_score: float = 0.5,
    max_pos_frac: float = 0.45,
    min_pos: int = 1,
) -> np.ndarray:
    """v5 ADAPTIVE-BUDGET ARM. Let the MODEL decide how many chunks are bots.

    Every other arm (v3/v4/v6/v7) forces exactly k = round(n * pos_frac) positives
    regardless of what the batch looks like: a 100-chunk request with 2 bots and one
    with 40 bots both get exactly 16 flags. That throws away the model's own opinion
    about prevalence -- and it is why calibration is a no-op for those arms
    (shape_gate_safe reads only rank order, and calibration is monotone).

    This is the discipline the live #1 miner (uid99, 30% emission) publishes under MIT
    in `poker44_ml/inference.py::_apply_batch_safety_budget` + `training/apply_safety_budget.py`:
      - flag whatever clears a calibrated `min_positive_score` (0.5),
      - CAP that at `max_positive_fraction` (their default: 0.45, vs our fixed 0.16),
      - compress positives into a narrow band just above 0.5 (theirs: 0.501-0.509).
    Their docstring is explicit that the budget "caps positives, it never [creates] them".

    We deviate from them in ONE deliberate way: `min_pos >= 1`. A pure probability
    threshold is exactly what zeroed our earlier v4 -- when live scores shift below
    0.5, NOTHING is flagged, true_positives == 0, and the whole reward is 0. The floor
    makes the downside a bad batch instead of a zeroed round, and costs nothing when
    the threshold already fires.

    WHO gets flagged is still decided by `rank_scores` (the strongest ordering we
    have); `prob_scores` only decides HOW MANY. So this arm tests exactly one thing:
    should the flag count adapt to the batch instead of being fixed?
    """
    p = np.asarray(prob_scores, dtype=float)
    r = np.asarray(rank_scores, dtype=float)
    n = len(r)
    if n == 0:
        return r
    cap = int(np.floor(n * max_pos_frac))
    k = int((p >= min_score).sum())
    k = min(k, cap)
    k = max(k, min_pos)
    k = min(k, n)
    order = np.argsort(-r, kind="mergesort")  # rank position 0 = most bot-like
    out = np.empty(n, dtype=float)
    for pos, i in enumerate(order):
        if pos < k:  # positives -> (0.5, 0.6], rank-preserving
            out[i] = 0.501 + 0.098 * (k - 1 - pos) / max(k - 1, 1)
        else:        # negatives -> [0, 0.5), rank-preserving
            out[i] = 0.499 * (n - 1 - pos) / max(n - 1 - k, 1)
    return out


def shape_gate_safe(rank_scores: np.ndarray, *, pos_frac: float = 0.16) -> np.ndarray:
    """Gate-safe shaping: flag exactly the top `pos_frac` of chunks (by rank) as
    positive (just above 0.5), the rest below 0.5, preserving rank order.

    Why a FIXED fraction (not a probability threshold): the reward's AP and
    recall@FPR are rank metrics, unaffected by where scores sit; only the 0.5
    crossing matters, via the human-safety gate. That gate is 0 (whole reward 0)
    if NONE of the >=0.5 chunks is a real bot (`true_positives == 0`), and decays
    only once human FPR@0.5 > 10%. The failure that zeroed v4 was true_positives=0
    (top-k caught no bot), so k must be large enough to reliably include a bot;
    tying k to a live-shifted probability threshold is exactly what made it
    unpredictable. Fixing k = round(n * pos_frac) and guaranteeing k >= 1 removes
    the catastrophic-0 risk; pos_frac trades bot-coverage against FPR.
    """
    r = np.asarray(rank_scores, dtype=float)
    n = len(r)
    if n == 0:
        return r
    k = int(max(1, min(n, round(n * pos_frac))))
    order = np.argsort(-r, kind="mergesort")  # rank position 0 = most bot-like
    out = np.empty(n, dtype=float)
    for pos, i in enumerate(order):
        if pos < k:  # positives -> (0.5, 0.6], rank-preserving
            out[i] = 0.501 + 0.098 * (k - 1 - pos) / max(k - 1, 1)
        else:        # negatives -> [0, 0.5), rank-preserving
            out[i] = 0.499 * (n - 1 - pos) / max(n - 1 - k, 1)
    return out


def fit_deploy_threshold(
    human_scores: np.ndarray,
    *,
    target_fpr: float = 0.04,
) -> float:
    """Score value above which only `target_fpr` of validation humans fall."""
    if len(human_scores) == 0:
        return 0.5
    return float(np.quantile(np.asarray(human_scores, float), 1.0 - target_fpr))


def _remap(scores: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear map sending threshold `t` to 0.5."""
    s = np.asarray(scores, dtype=float)
    t = float(min(max(t, 1e-9), 1 - 1e-9))
    lo = 0.5 * s / t
    hi = 0.5 + 0.5 * (s - t) / (1.0 - t)
    return np.clip(np.where(s < t, lo, hi), 0.0, 1.0)


def shape_scores(
    raw: np.ndarray,
    *,
    deploy_threshold: float,
    max_pos_frac: float = 0.16,
) -> np.ndarray:
    """Single monotone remap enforcing all three gate constraints:

    1. scores >= deploy_threshold land at >= 0.5 (model's tuned operating point);
    2. at most floor(n * max_pos_frac) chunks cross 0.5 (FPR-cliff insurance);
    3. at least one chunk crosses 0.5 (zero-positive gate insurance).
    """
    s = np.asarray(raw, dtype=float)
    n = len(s)
    if n == 0:
        return s

    t_eff = float(deploy_threshold)

    budget = max(1, int(np.floor(n * max_pos_frac)))
    if n > budget:
        desc = np.sort(s)[::-1]
        # Midpoint between the budget-th and (budget+1)-th highest raw scores:
        # at most `budget` items sit above it (barring exact ties).
        t_budget = 0.5 * (desc[budget - 1] + desc[budget])
        t_eff = max(t_eff, t_budget)

    s_max = float(s.max())
    if s_max < t_eff:
        t_eff = s_max - 1e-9  # guarantee one positive; gate must never zero us

    out = _remap(s, t_eff)

    # Exact-tie degenerate case (e.g. constant scores): more than `budget`
    # values sit exactly at/above 0.5. Demote surplus ties just below the gate;
    # ties carry no ranking information, so rank metrics are unaffected.
    pos = np.flatnonzero(out >= 0.5)
    if len(pos) > budget:
        order = pos[np.argsort(-out[pos], kind="mergesort")]
        for k, i in enumerate(order[budget:]):
            out[i] = 0.5 - 1e-9 * (k + 1)
    return out


def shape_hybrid(
    rank_scores: np.ndarray,
    prob_scores: np.ndarray,
    *,
    deploy_threshold: float,
    max_pos_frac: float = 0.16,
) -> np.ndarray:
    """Ordering from rank fusion; positive COUNT from calibrated probabilities.

    Rank fusion gives the best in-request ordering (the validator's scoring
    window is exactly one request's chunks), but rank values carry no absolute
    meaning for the 0.5 gate. So: k = how many chunks the calibrated prob blend
    puts above the deploy threshold (clipped to [1, budget]), and the top-k of
    the FUSED ordering are mapped above 0.5, the rest below. Output is a
    monotone function of `rank_scores`, so AP/recall follow the fused ranking.
    """
    r = np.asarray(rank_scores, dtype=float)
    p = np.asarray(prob_scores, dtype=float)
    n = len(r)
    if n == 0:
        return r
    budget = max(1, int(np.floor(n * max_pos_frac)))
    k = int(np.clip((p >= deploy_threshold).sum(), 1, budget))

    order = np.argsort(-r, kind="mergesort")  # stable: ties broken by index
    out = np.empty(n, dtype=float)
    for pos, i in enumerate(order):
        if pos < k:
            out[i] = 0.5 + 0.5 * (k - pos) / (k + 1)
        else:
            out[i] = 0.5 * (n - pos) / (n - k + 1)
    return out
