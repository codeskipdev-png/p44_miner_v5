"""Chunk-level feature extraction for Poker44 bot detection.

Design constraints (from 03_data/analysis + 01_subnet_understanding):
- Only the anonymized action sequence survives redaction: type, aliased actor,
  bucketed bb sizing, pot trajectory, street progression, stacks, hero seat.
- Live traffic is 41% 7-9-max while the benchmark is 6-max only, so every
  count-like signal is also emitted per-active-player (seat-count invariant).
- Bet amounts are re-quantized to the validator's exact 16-bucket bb grid to
  cancel the injected bucket noise.
- Bots are stationary policies: cross-hand signature-repeat and n-gram
  regularity features are the highest-signal family.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, List

import numpy as np

from .payload_view import _VISIBLE_BB_BUCKETS

_ACTIONS = ("fold", "check", "call", "bet", "raise")
_AGG_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
_NGRAM_DIM = 64
_VISIBLE_BB = 0.02


def _entropy(items) -> float:
    if not items:
        return 0.0
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _bucket_index(amount_bb: float) -> int:
    if amount_bb <= 0:
        return 0
    diffs = [abs(b - amount_bb) for b in _VISIBLE_BB_BUCKETS]
    return int(np.argmin(diffs))


def _max_run_share(seq) -> float:
    if not seq:
        return 0.0
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best / len(seq)


def hand_features(hand: dict) -> Dict[str, float]:
    meta = hand.get("metadata") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    f: Dict[str, float] = {}
    types = [a.get("action_type", "") for a in actions]
    actors = [int(a.get("actor_seat") or 0) for a in actions]
    amounts = [float(a.get("normalized_amount_bb") or 0.0) for a in actions]
    pots_after = [float(a.get("pot_after") or 0.0) / _VISIBLE_BB for a in actions]
    pots_before = [float(a.get("pot_before") or 0.0) / _VISIBLE_BB for a in actions]

    n_act = max(len(actions), 1)
    counts = Counter(types)
    for t in _ACTIONS:
        f[f"share_{t}"] = counts.get(t, 0) / n_act
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)
    f["share_aggressive"] = aggressive / n_act
    f["ratio_agg_passive"] = aggressive / max(passive, 1)
    f["n_actions"] = float(len(actions))

    street_of = [str(a.get("street") or "preflop") for a in actions]
    f["share_preflop"] = sum(s == "preflop" for s in street_of) / n_act
    f["n_streets"] = float(len(streets))
    max_street = max((_STREET_ORDER.get(s.get("street", ""), 0) for s in streets), default=0)
    f["reached_flop"] = float(max_street >= 1)
    f["reached_river"] = float(max_street >= 3)

    f["entropy_action"] = _entropy(types)
    f["entropy_actor"] = _entropy(actors)
    f["entropy_street"] = _entropy(street_of)
    f["actor_switch_rate"] = (
        sum(a != b for a, b in zip(actors, actors[1:])) / max(len(actors) - 1, 1)
    )
    f["actor_max_run_share"] = _max_run_share(actors)

    nz = [a for a in amounts if a > 0]
    f["amount_mean_bb"] = float(np.mean(nz)) if nz else 0.0
    f["amount_std_bb"] = float(np.std(nz)) if nz else 0.0
    f["amount_max_bb"] = float(np.max(nz)) if nz else 0.0
    f["amount_zero_share"] = 1.0 - len(nz) / n_act
    buckets = [_bucket_index(a) for a in nz]
    f["bucket_mean"] = float(np.mean(buckets)) if buckets else 0.0
    f["bucket_std"] = float(np.std(buckets)) if buckets else 0.0
    f["bucket_top_share"] = (
        max(Counter(buckets).values()) / len(buckets) if buckets else 0.0
    )

    f["pot_final_bb"] = pots_after[-1] if pots_after else 0.0
    f["pot_growth_bb"] = (
        (max(pots_after) - min(pots_before)) if pots_after and pots_before else 0.0
    )
    deltas = [a - b for a, b in zip(pots_after, pots_before)]
    f["pot_delta_mean_bb"] = float(np.mean(deltas)) if deltas else 0.0
    f["pot_monotonic_rate"] = (
        sum(b >= a for a, b in zip(pots_after, pots_after[1:])) / max(len(pots_after) - 1, 1)
    )
    f["raise_to_share"] = sum(a.get("raise_to") is not None for a in actions) / n_act
    f["call_to_share"] = sum(a.get("call_to") is not None for a in actions) / n_act

    stacks = [float(p.get("starting_stack") or 0.0) / _VISIBLE_BB for p in players]
    f["stack_mean_bb"] = float(np.mean(stacks)) if stacks else 0.0
    f["stack_std_bb"] = float(np.std(stacks)) if stacks else 0.0
    f["stack_min_bb"] = float(np.min(stacks)) if stacks else 0.0

    # Shift-robust sizing: bets relative to the pot they enter, not absolute bb.
    # Live pots are ~20x smaller than benchmark pots; ratios share support.
    ratios = [
        a / max(pb, 0.5)
        for a, pb in zip(amounts, pots_before)
        if a > 0 and pb > 0
    ]
    f["amt_pot_ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
    f["amt_pot_ratio_std"] = float(np.std(ratios)) if ratios else 0.0
    f["amt_pot_ratio_max"] = float(np.max(ratios)) if ratios else 0.0
    rel_pot_growth = (
        (max(pots_after) / max(min(pots_before), 0.5)) if pots_after and pots_before else 0.0
    )
    f["pot_growth_rel"] = float(min(rel_pot_growth, 50.0))

    # Seat-count invariance: live traffic includes 7-9-max tables absent from
    # the 6-max-only benchmark, so raw counts must be normalized per player.
    n_players = max(len(players), 1)
    active = max(len(set(a for a in actors if a > 0)), 1)
    max_seats = float(meta.get("max_seats") or n_players)
    f["n_players"] = float(n_players)
    f["max_seats"] = max_seats
    f["occupancy"] = n_players / max(max_seats, 1.0)
    f["actions_per_player"] = len(actions) / n_players
    f["actions_per_active"] = len(actions) / active
    f["folds_per_player"] = counts.get("fold", 0) / n_players
    f["aggr_per_active"] = aggressive / active
    f["active_share"] = active / n_players

    hero = int(meta.get("hero_seat") or 0)
    hero_actions = [a for a, s in zip(actions, actors) if s == hero]
    f["hero_action_share"] = len(hero_actions) / n_act
    f["hero_zero_actions"] = float(not hero_actions)
    hero_aggr = sum(a.get("action_type") in ("bet", "raise") for a in hero_actions)
    f["hero_aggr_share"] = hero_aggr / max(len(hero_actions), 1)

    return f


def _signatures(hand: dict) -> Dict[str, str]:
    actions = hand.get("actions") or []
    return {
        "action_sig": "|".join(a.get("action_type", "") for a in actions),
        "actor_sig": "|".join(str(a.get("actor_seat") or 0) for a in actions),
        "street_sig": "|".join(str(a.get("street") or "") for a in actions),
        "bucket_sig": "|".join(
            str(_bucket_index(float(a.get("normalized_amount_bb") or 0.0)))
            for a in actions
        ),
    }


def _hand_tokens(hand: dict, *, sized: bool) -> List[str]:
    toks = []
    for a in hand.get("actions") or []:
        street = str(a.get("street") or "p")[0]
        act = str(a.get("action_type") or "?")[0]
        if sized:
            size = _bucket_index(float(a.get("normalized_amount_bb") or 0.0))
            toks.append(f"{street}{act}{min(size, 9)}")
        else:
            toks.append(f"{street}{act}")  # size-free: robust to bet-scale shift
    return toks


def _hashed_ngrams(hands: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for prefix, sized in (("ngram", True), ("ngram_t", False)):
        vec = np.zeros(2 * _NGRAM_DIM)
        total = 0
        for hand in hands:
            toks = _hand_tokens(hand, sized=sized)
            for i in range(len(toks) - 1):
                for order, off in ((2, 0), (3, _NGRAM_DIM)):
                    if i + order <= len(toks):
                        gram = "_".join(toks[i : i + order])
                        h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
                        vec[off + h % _NGRAM_DIM] += 1
                        total += 1
        if total:
            vec /= total
        out.update({f"{prefix}_{i}": float(v) for i, v in enumerate(vec)})
    return out


def chunk_features(hands: List[dict]) -> Dict[str, float]:
    """One feature row per chunk (= one player's hands = one label)."""
    hands = hands or []
    per_hand = [hand_features(h) for h in hands]
    out: Dict[str, float] = {}

    keys = per_hand[0].keys() if per_hand else []
    for k in keys:
        vals = np.asarray([f[k] for f in per_hand], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
        out[f"{k}_min"] = float(vals.min())
        out[f"{k}_max"] = float(vals.max())
        out[f"{k}_q10"] = float(np.quantile(vals, 0.10))
        out[f"{k}_q50"] = float(np.quantile(vals, 0.50))
        out[f"{k}_q90"] = float(np.quantile(vals, 0.90))

    n = max(len(hands), 1)
    sigs = [_signatures(h) for h in hands]
    for name in ("action_sig", "actor_sig", "street_sig", "bucket_sig"):
        c = Counter(s[name] for s in sigs)
        out[f"{name}_top_share"] = max(c.values()) / n if c else 0.0
        out[f"{name}_unique_share"] = len(c) / n if c else 0.0
        out[f"{name}_entropy"] = _entropy([s[name] for s in sigs])

    out.update(_hashed_ngrams(hands))

    out["rate_low_action_entropy"] = sum(
        f["entropy_action"] < 0.8 for f in per_hand
    ) / n
    out["rate_high_aggression"] = sum(
        f["share_aggressive"] > 0.5 for f in per_hand
    ) / n
    out["rate_zero_hero"] = sum(f["hero_zero_actions"] > 0 for f in per_hand) / n
    out["rate_single_street"] = sum(f["n_streets"] <= 1 for f in per_hand) / n

    out["hand_count"] = float(len(hands))
    out["hand_count_log"] = math.log1p(len(hands))
    return out


def feature_names() -> List[str]:
    """Stable column order derived from a synthetic probe chunk."""
    probe_hand = {
        "metadata": {"max_seats": 6, "hero_seat": 1},
        "players": [{"seat": 1, "starting_stack": 2.0}],
        "streets": [{"street": "preflop"}],
        "actions": [
            {
                "street": "preflop",
                "actor_seat": 1,
                "action_type": "raise",
                "normalized_amount_bb": 2.5,
                "raise_to": 0.05,
                "call_to": None,
                "pot_before": 0.03,
                "pot_after": 0.08,
            }
        ]
        * 3,
    }
    return sorted(chunk_features([probe_hand, probe_hand]).keys())


def rows_to_matrix(rows: List[Dict[str, float]], cols: List[str]) -> np.ndarray:
    mat = np.zeros((len(rows), len(cols)), dtype=float)
    for i, row in enumerate(rows):
        for j, c in enumerate(cols):
            v = row.get(c, 0.0)
            if np.isfinite(v):
                mat[i, j] = float(v)
    return mat
