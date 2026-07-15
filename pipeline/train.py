"""Train the RankBlend on sanitized benchmark data and evaluate held-out.

Usage:
    python -m pipeline.train                 # train <= SPLIT, eval > SPLIT
    python -m pipeline.train --refresh-cache # refeaturize from raw JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from .dataset import BENCH_DIR, available_dates, iter_batches, live_size_augment
from .features import chunk_features, feature_names, rows_to_matrix


def _sig(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]


def _bench_sig(date: str) -> str:
    """Content signature of one benchmark day (size+mtime) so the feature cache
    refreshes automatically if that day is ever re-scraped with new data."""
    p = BENCH_DIR / f"benchmark_{date}.json"
    if not p.exists():
        return "nofile"
    st = p.stat()
    return _sig(st.st_size, int(st.st_mtime))


def _captures_sig(real_dir: Path) -> str:
    """Signature of the whole live-capture set, so KS selection refreshes when
    new validator payloads are captured."""
    files = sorted(real_dir.glob("*.json"))
    if not files:
        return "empty"
    return _sig(*[(f.name, f.stat().st_size) for f in files])

ROOT = Path("/root/Skip/poker/SN126/04_our_miner_v5")
CACHE = ROOT / "artifacts" / "feature_cache"
SPLIT = "2026-07-10"  # train <= SPLIT, eval > SPLIT (same as uid227 comparison)
CACHE_VERSION = "v2"  # bump when features.py changes
REAL_DIR = Path("/root/Skip/poker/SN126/03_data/real_challenge/raw")
MAX_KS = 0.55  # drop features whose benchmark-vs-live KS exceeds this


def recall_at_fpr(y_score, y_true, max_fpr=0.05) -> float:
    labels = np.asarray(y_true, int)
    scores = np.asarray(y_score, float)
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    rec = np.cumsum(sl == 1) / max((labels == 1).sum(), 1)
    fpr = np.cumsum(sl == 0) / max((labels == 0).sum(), 1)
    ok = fpr <= max_fpr
    return float(rec[ok].max()) if ok.any() else 0.0


def reward(y_score, y_true) -> Dict[str, float]:
    ap = average_precision_score(y_true, y_score)
    rec = recall_at_fpr(y_score, y_true)
    return {"ap": ap, "recall5": rec, "reward": 0.75 * ap + 0.25 * rec}


def _featurize_one(args: Tuple[str, List[dict], int]) -> Tuple[str, Dict[str, float], int]:
    date, hands, label = args
    return date, chunk_features(hands), label


def featurize_dates(dates: List[str], *, augment: bool, refresh: bool) -> Tuple[List[str], List[Dict], List[int]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    all_dates: List[str] = []
    all_rows: List[Dict] = []
    all_labels: List[int] = []
    for date in dates:
        tag = ("aug" if augment else "base") + "_" + CACHE_VERSION
        cache_file = CACHE / f"{date}_{tag}_{_bench_sig(date)}.pkl"
        if cache_file.exists() and not refresh:
            with cache_file.open("rb") as f:
                rows, labels = pickle.load(f)
        else:
            batches = list(iter_batches([date]))
            if augment:
                batches = batches + live_size_augment(batches)
            with ProcessPoolExecutor(max_workers=10) as ex:
                results = list(ex.map(_featurize_one, batches, chunksize=8))
            rows = [r for _, r, _ in results]
            labels = [l for _, _, l in results]
            # drop stale-signature caches for this date+tag (content changed)
            for old in CACHE.glob(f"{date}_{tag}_*.pkl"):
                if old != cache_file:
                    old.unlink()
            with cache_file.open("wb") as f:
                pickle.dump((rows, labels), f)
        all_dates.extend([date] * len(rows))
        all_rows.extend(rows)
        all_labels.extend(labels)
    return all_dates, all_rows, all_labels


def featurize_real_captures(*, refresh: bool) -> List[Dict]:
    """Feature rows for the captured live validator requests (unlabeled)."""
    cache_file = CACHE / f"real_captures_{CACHE_VERSION}_{_captures_sig(REAL_DIR)}.pkl"
    if cache_file.exists() and not refresh:
        with cache_file.open("rb") as f:
            return pickle.load(f)
    chunks = []
    for path in sorted(REAL_DIR.glob("*.json")):
        with path.open() as f:
            req = json.load(f)
        chunks.extend([("live", c, -1) for c in req.get("chunks", []) if c])
    with ProcessPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_featurize_one, chunks, chunksize=8))
    rows = [r for _, r, _ in results]
    for old in CACHE.glob(f"real_captures_{CACHE_VERSION}_*.pkl"):
        if old != cache_file:
            old.unlink()
    with cache_file.open("wb") as f:
        pickle.dump(rows, f)
    return rows


def select_robust_features(
    Xtr: np.ndarray, Xreal: np.ndarray, cols: List[str], *, max_ks: float = MAX_KS
) -> List[str]:
    """Keep features whose benchmark and live marginals overlap (KS <= max_ks).

    Uses only unlabeled live captures - no label information can leak. Features
    with disjoint support (e.g. absolute stack sizes: live is constant 100bb)
    invite the GBDTs to split on values never seen at serve time.
    """
    from scipy.stats import ks_2samp

    kept = []
    for j, c in enumerate(cols):
        if np.std(Xtr[:, j]) == 0:
            continue
        ks = ks_2samp(Xtr[:, j], Xreal[:, j]).statistic
        if ks <= max_ks:
            kept.append(c)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-robust", action="store_true", help="skip KS feature filtering")
    args = parser.parse_args()

    from .model import RankBlend

    dates = available_dates()
    train_dates = [d for d in dates if d <= SPLIT]
    test_dates = [d for d in dates if d > SPLIT]
    cols = feature_names()
    print(f"features={len(cols)} train_dates={len(train_dates)} test_dates={test_dates}")

    t0 = time.time()
    tr_dates, tr_rows, tr_y = featurize_dates(
        train_dates, augment=not args.no_augment, refresh=args.refresh_cache
    )
    te_dates, te_rows, te_y = featurize_dates(
        test_dates, augment=False, refresh=args.refresh_cache
    )
    print(f"featurized: train={len(tr_rows)} test={len(te_rows)} ({time.time()-t0:.0f}s)")

    real_rows = featurize_real_captures(refresh=args.refresh_cache)
    if not args.no_robust:
        Xtr_full = rows_to_matrix(tr_rows, cols)
        Xreal_full = rows_to_matrix(real_rows, cols)
        cols = select_robust_features(Xtr_full, Xreal_full, cols)
        print(f"robust feature selection (KS<={MAX_KS}): kept {len(cols)}")

    Xtr = rows_to_matrix(tr_rows, cols)
    Xte = rows_to_matrix(te_rows, cols)
    Xreal = rows_to_matrix(real_rows, cols)
    ytr = np.asarray(tr_y)
    yte = np.asarray(te_y)

    t0 = time.time()
    blend = RankBlend().fit(Xtr, ytr, tr_dates, cols)
    print(f"trained in {time.time()-t0:.0f}s | monotone features: {blend.meta['n_monotone']}")

    # Per-member and blended held-out performance, overall and per-date.
    member = blend.member_probs(Xte)
    for name, p in member.items():
        m = reward(p, yte)
        print(f"  member {name:6s}: AP {m['ap']:.4f}  rec@5% {m['recall5']:.4f}  reward {m['reward']:.4f}")

    te_dates_arr = np.asarray(te_dates)
    blended_all = np.zeros(len(yte))
    for d in sorted(set(te_dates)):
        mask = te_dates_arr == d
        blended_all[mask] = blend.score(Xte[mask])  # rank within a date's batch
        m = reward(blended_all[mask], yte[mask])
        print(f"  {d}: n={mask.sum():4d}  AP {m['ap']:.4f}  rec@5% {m['recall5']:.4f}  reward {m['reward']:.4f}")
    m = reward(blended_all, yte)
    print(f"BLEND ALL: AP {m['ap']:.4f}  rec@5% {m['recall5']:.4f}  reward {m['reward']:.4f}")
    print("uid227 same protocol:  AP 0.9684  rec@5% 0.8397  reward 0.9362")

    # Live-transfer proxies (unlabeled): score spread must not collapse.
    p_real = blend.score_prob(Xreal)
    p_bench = blend.score_prob(Xte)
    spread = lambda p: float(np.quantile(p, 0.9) - np.quantile(p, 0.1))  # noqa: E731
    print(
        f"live-transfer proxy: score spread benchmark={spread(p_bench):.3f} "
        f"live={spread(p_real):.3f}  (uid227 live: 0.608; v1 collapsed to 0.119)"
    )

    out = ROOT / "artifacts" / "rankblend_v2.pkl"
    with out.open("wb") as f:
        pickle.dump(blend, f)
    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_dates": [train_dates[0], train_dates[-1]],
        "test_dates": test_dates,
        "n_train": len(ytr),
        "n_features": len(cols),
        "n_monotone": blend.meta["n_monotone"],
        "heldout": m,
        "uid227_heldout": {"ap": 0.9684, "recall5": 0.8397, "reward": 0.9362},
        "live_spread": spread(p_real),
        "bench_spread": spread(p_bench),
    }
    (ROOT / "artifacts" / "rankblend_v2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
