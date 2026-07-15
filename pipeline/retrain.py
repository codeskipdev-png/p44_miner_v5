"""Guarded daily retrain: fetch -> train candidate -> gate vs champion -> promote.

Usage:
    python -m pipeline.retrain --once            # one full cycle now
    python -m pipeline.retrain --daemon          # run daily at RETRAIN_UTC_HOUR
    python -m pipeline.retrain --once --dry-run  # evaluate but never promote

Gate (candidate must satisfy ALL, vs champion re-scored on the SAME data):
  1. live-size pooled reward >= champion - MAX_REGRESSION
  2. standard held-out reward >= champion - MAX_REGRESSION
  3. shaped output passes the official gated reward with human_safety == 1.0
Promotion is an atomic swap of artifacts/serving_blend_v3.pkl (+meta), with a
timestamped archive. The miner hot-reloads on file change (see serve/scorer.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import random
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from .dataset import available_dates, iter_batches
from .features import chunk_features, feature_names, rows_to_matrix
from .fetch_benchmark import sync
from .model import RankBlend
from .serving_blend import ServingBlend
from .threshold import shape_gate_safe
from .train import (
    MAX_KS,
    featurize_dates,
    featurize_real_captures,
    reward,
    select_robust_features,
)

ROOT = Path("/root/Skip/poker/SN126/04_our_miner_v5")
ARTIFACTS = ROOT / "artifacts"
ARCHIVE = ARTIFACTS / "archive"
SERVING = ARTIFACTS / "serving_blend_v3.pkl"
STATE = ARTIFACTS / "retrain_state.json"

HOLDOUT_DAYS = 2  # newest N dates are the gate set
MAX_REGRESSION = 0.02
TARGET_FPR = 0.04
MAX_POS_FRAC = 0.16
RETRAIN_UTC_HOUR = 3.5  # 03:30 UTC, after the ~20:00 UTC window closes + buffer
ARCHIVE_DEPTH = 14

MEMBER_SPECS = [
    ("v1", "all"),          # every feature (in-distribution sharp)
    ("live-var", "livevar"),  # drop live-constant features
    ("ks0.90", "ks"),       # drop KS>0.90 features
]


def log(msg: str) -> None:
    print(f"[{dt.datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}", flush=True)


def _member_cols(spec: str, Xtr: np.ndarray, Xreal: np.ndarray, cols: List[str]) -> List[str]:
    tr_std = Xtr.std(axis=0)
    if spec == "all":
        return [c for c, s in zip(cols, tr_std) if s > 0]
    if spec == "livevar":
        live_std = Xreal.std(axis=0)
        return [c for c, s, ls in zip(cols, tr_std, live_std) if s > 0 and ls > 1e-9]
    if spec == "ks":
        return select_robust_features(Xtr, Xreal, cols, max_ks=0.90)
    raise ValueError(spec)


def _pooled_livesize(dates: List[str], seed: int = 42) -> Tuple[List[List[dict]], np.ndarray]:
    rng = random.Random(seed)
    by_key: Dict[Tuple[str, int], List[List[dict]]] = {}
    for date, hands, label in iter_batches(dates):
        by_key.setdefault((date, label), []).append(hands)
    pooled, labels = [], []
    for (_, label), groups in sorted(by_key.items()):
        rng.shuffle(groups)
        i = 0
        while i < len(groups):
            acc: List[dict] = []
            target = rng.randint(80, 100)
            while i < len(groups) and len(acc) < target:
                acc.extend(groups[i])
                i += 1
            if len(acc) >= 60:
                pooled.append(acc[:target])
                labels.append(label)
    return pooled, np.asarray(labels)


def _evaluate(blend: ServingBlend, X_std, y_std, X_pool, y_pool) -> Dict:
    out: Dict = {}
    p_std = blend.score_prob(X_std)
    p_pool = blend.score_prob(X_pool)
    out["standard"] = reward(p_std, y_std)
    out["livesize"] = reward(p_pool, y_pool)
    # evaluate the SAME gate-safe shaping we serve (fixed top-k%, no threshold)
    shaped = shape_gate_safe(blend.score_rank(X_pool), pos_frac=MAX_POS_FRAC)
    sys.path.insert(0, "/root/Skip/poker/SN126/00_external/owner_repo")
    from poker44.score.scoring import reward as official  # noqa: PLC0415

    rew, res = official(shaped, y_pool)
    out["official_reward"] = float(rew)
    out["human_safety"] = float(res["human_safety_penalty"])
    return out


def _refuse_if_shared_artifact() -> None:
    """v5 SHARES v3's model artifact by symlink -- retraining here is always wrong.

    v5 is a serving-discipline arm: identical code and identical model to v3, differing
    only in HOW MANY chunks it flags (adaptive budget vs v3's fixed top-k). That
    comparison is only meaningful while the two serve the SAME model, which is why
    artifacts/serving_blend_v3.pkl is a symlink into 04_our_miner and why v5 is
    deliberately absent from retrain_all.py's MINERS.

    Promotion does `tmp.replace(SERVING)`, which would replace the SYMLINK with a real
    file -- silently ending the sharing and letting v5's model drift from v3's, destroying
    the experiment with no error. Fail loudly instead. Retrain v3; v5 follows for free
    (and re-calibrates its threshold automatically on the new model's mtime).
    """
    if SERVING.is_symlink():
        raise SystemExit(
            "REFUSING to retrain v5: artifacts/serving_blend_v3.pkl is a symlink to v3's "
            "artifact. v5 shares v3's model on purpose (only the flag-count discipline "
            "differs). Retrain v3 instead -- v5 picks the new model up via the symlink and "
            "re-calibrates min_score on load."
        )


def run_cycle(*, dry_run: bool = False, skip_fetch: bool = False) -> Dict:
    _refuse_if_shared_artifact()
    summary: Dict = {"started_at": dt.datetime.utcnow().isoformat() + "Z"}

    if skip_fetch:
        # data already downloaded by the unified orchestrator (scrape script)
        log("skip_fetch: using benchmark data already on disk")
        new_dates = []
    else:
        log("fetching new benchmark releases...")
        try:
            new_dates = sync(log=log)
        except Exception as e:  # noqa: BLE001
            log(f"fetch failed ({e}); proceeding with existing data")
            new_dates = []
    summary["new_dates"] = new_dates

    dates = available_dates()
    holdout = dates[-HOLDOUT_DAYS:]
    train_dates = dates[: -HOLDOUT_DAYS]
    summary["train_span"] = [train_dates[0], train_dates[-1]]
    summary["holdout"] = holdout
    log(f"train {train_dates[0]}..{train_dates[-1]} ({len(train_dates)} dates), gate on {holdout}")

    cols = feature_names()
    tr_d, tr_rows, tr_y = featurize_dates(train_dates, augment=True, refresh=False)
    te_d, te_rows, te_y = featurize_dates(holdout, augment=False, refresh=False)
    real_rows = featurize_real_captures(refresh=False)
    Xtr_all = rows_to_matrix(tr_rows, cols)
    Xte_all = rows_to_matrix(te_rows, cols)
    Xreal_all = rows_to_matrix(real_rows, cols)
    ytr = np.asarray(tr_y)
    yte = np.asarray(te_y)

    pooled, ypool = _pooled_livesize(holdout)
    Xpool_all = rows_to_matrix([chunk_features(c) for c in pooled], cols)
    log(f"gate sets: standard n={len(yte)}, live-size n={len(ypool)}")

    members = []
    for tag, spec in MEMBER_SPECS:
        kept = _member_cols(spec, Xtr_all, Xreal_all, cols)
        idx = [cols.index(c) for c in kept]
        t0 = time.time()
        blend = RankBlend().fit(Xtr_all[:, idx], ytr, tr_d, kept)
        members.append((tag, blend))
        log(f"member {tag}: {len(kept)} features, trained in {time.time()-t0:.0f}s")

    candidate = ServingBlend(members)
    cand = _evaluate(candidate, Xte_all, yte, Xpool_all, ypool)
    log(
        f"candidate: standard {cand['standard']['reward']:.4f} "
        f"livesize {cand['livesize']['reward']:.4f} official {cand['official_reward']:.4f} "
        f"hsp {cand['human_safety']:.2f}"
    )
    summary["candidate"] = cand

    champ = None
    if SERVING.exists():
        with SERVING.open("rb") as f:
            champion = pickle.load(f)
        champ = _evaluate(champion, Xte_all, yte, Xpool_all, ypool)
        log(
            f"champion : standard {champ['standard']['reward']:.4f} "
            f"livesize {champ['livesize']['reward']:.4f} official {champ['official_reward']:.4f}"
        )
    summary["champion"] = champ

    ok_livesize = champ is None or (
        cand["livesize"]["reward"] >= champ["livesize"]["reward"] - MAX_REGRESSION
    )
    ok_standard = champ is None or (
        cand["standard"]["reward"] >= champ["standard"]["reward"] - MAX_REGRESSION
    )
    ok_gate = cand["human_safety"] >= 1.0
    promote = ok_livesize and ok_standard and ok_gate
    summary["gate"] = {
        "livesize_ok": ok_livesize,
        "standard_ok": ok_standard,
        "official_gate_ok": ok_gate,
        "promote": promote,
    }

    if promote and not dry_run:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if SERVING.exists():
            shutil.copy2(SERVING, ARCHIVE / f"serving_blend_{stamp}.pkl")
            archives = sorted(ARCHIVE.glob("serving_blend_*.pkl"))
            for old in archives[:-ARCHIVE_DEPTH]:
                old.unlink()
        tmp = SERVING.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(candidate, f)
        meta = {
            "promoted_at": stamp,
            "train_span": summary["train_span"],
            "holdout": holdout,
            "pos_frac": MAX_POS_FRAC,
            "metrics": cand,
        }
        SERVING.with_name(SERVING.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
        tmp.replace(SERVING)  # atomic swap; miner hot-reloads on mtime change
        log(f"PROMOTED candidate -> {SERVING.name} (archived {ARCHIVE_DEPTH}-deep)")
    elif promote:
        log("dry-run: candidate passed the gate but was NOT promoted")
    else:
        log("candidate failed the gate; champion retained")

    summary["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    history = []
    if STATE.exists():
        try:
            history = json.loads(STATE.read_text()).get("history", [])
        except Exception:  # noqa: BLE001
            history = []
    history.append(summary)
    STATE.write_text(json.dumps({"history": history[-60:]}, indent=2, default=str))
    return summary


def daemon() -> None:
    log(f"retrain daemon up; daily at {RETRAIN_UTC_HOUR:.1f}h UTC")
    while True:
        now = dt.datetime.utcnow()
        target = now.replace(
            hour=int(RETRAIN_UTC_HOUR),
            minute=int((RETRAIN_UTC_HOUR % 1) * 60),
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += dt.timedelta(days=1)
        wait = (target - now).total_seconds()
        log(f"next cycle at {target.isoformat()}Z (in {wait/3600:.1f}h)")
        time.sleep(wait)
        try:
            run_cycle()
        except Exception:  # noqa: BLE001
            log(f"cycle failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip internal fetch (orchestrator scraped already)")
    args = parser.parse_args()
    if args.once:
        run_cycle(dry_run=args.dry_run, skip_fetch=args.no_fetch)
    elif args.daemon:
        daemon()
    else:
        parser.print_help()
