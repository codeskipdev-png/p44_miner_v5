"""Production chunk scorer with a strict reliability contract.

Contract (one violation = the whole evaluation window scores zero):
  1. Always return exactly len(chunks) floats in [0, 1].
  2. Never raise, whatever the payload looks like.
  3. Stay far inside the validator's 180s timeout.

Scoring path: features -> ServingBlend -> in-request rank fusion ->
gate-safe shaping (flag the top P44_POS_FRAC of chunks by rank, guaranteed >=1,
rank-preserving). See pipeline/threshold.shape_gate_safe for why a fixed
fraction (not a probability threshold) is the correct gate strategy.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

# Members were fit with sklearn's positional default names (Column_0, ...); we
# predict on numpy arrays built in the exact same column order, so predictions
# are identical (verified: max diff 3e-16). Silence the benign name-mismatch spam.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pipeline.features import chunk_features
from pipeline.threshold import shape_adaptive

log = logging.getLogger("scorer")

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FALLBACK_SCORE = 0.1  # benign low-risk score for unusable chunks
MAX_HANDS_PER_CHUNK = 120  # runtime cap; live chunks are 80-100
CAPTURE_DIR = Path(
    os.getenv("P44_CAPTURE_DIR", "/root/Skip/poker/SN126/03_data/real_challenge/raw")
)
CALIB_MAX_FILES = 10  # newest N captured requests to calibrate on (bounds startup cost)


class ChunkScorer:
    def __init__(
        self,
        model_path: Path | str = ARTIFACTS / "serving_blend_v3.pkl",
    ):
        self.model_path = Path(model_path)
        # v5 ADAPTIVE BUDGET (see pipeline.threshold.shape_adaptive). Unlike the other
        # arms there is no fixed pos_frac: the calibrated probability decides how many
        # chunks clear min_score, capped at max_pos_frac and floored at min_pos.
        # Defaults mirror uid99's published budget (0.5 / 0.45); min_pos is our own
        # guard against the true_positives==0 zero that a bare threshold risks.
        # min_score is normally auto-calibrated per model (see _calibrate_min_score);
        # setting P44_MIN_SCORE pins it explicitly and disables calibration.
        pinned = os.getenv("P44_MIN_SCORE", "").strip()
        self._auto_calibrate = not pinned
        self.min_score = float(pinned) if pinned else 0.5
        self.max_pos_frac = float(os.getenv("P44_MAX_POS_FRAC", "0.45"))
        self.min_pos = int(os.getenv("P44_MIN_POS", "1"))
        self._mtime = 0.0
        self._load()

    def _calibrate_min_score(self) -> float:
        """Score threshold that makes the AVERAGE live flag rate == TARGET_FLAG_RATE.

        Must be recomputed whenever the model changes: v5 shares v3's artifact, v3
        retrains daily, and a threshold in score space goes stale the moment the score
        distribution moves. Cached against the model's mtime so a restart is cheap and
        a retrain automatically triggers one recalibration.

        Measured 2026-07-15: the live score_prob distribution is compressed into
        [0.338, 0.801] with mean 0.576 -- 80.6% of live chunks clear a naive 0.5. So a
        literal 0.5 threshold (uid99's default) would flag ~80% here and torch FPR.
        Calibrating on the UNLABELED captures fixes the rate without touching labels.
        """
        import glob  # noqa: PLC0415

        target = float(os.getenv("P44_TARGET_FLAG_RATE", "0.16"))
        cache = self.model_path.parent / "v5_calibration.json"
        try:
            if cache.exists():
                c = json.loads(cache.read_text())
                if c.get("model_mtime") == self._mtime and c.get("target") == target:
                    return float(c["min_score"])
        except Exception:  # noqa: BLE001
            log.exception("calibration cache unreadable; recomputing")

        files = sorted(glob.glob(str(CAPTURE_DIR / "*.json")))[-CALIB_MAX_FILES:]
        probs = []
        for path in files:
            try:
                with open(path) as f:
                    chunks = [c for c in (json.load(f).get("chunks") or []) if c]
                if not chunks:
                    continue
                rows = [chunk_features(c[:MAX_HANDS_PER_CHUNK]) for c in chunks]
                probs.append(self.blend.score_prob(self.blend.featurize(rows)))
            except Exception:  # noqa: BLE001
                log.exception("calibration: skipping %s", path)
        if not probs:
            log.warning("calibration: no live captures; falling back to min_score=0.5")
            return 0.5
        thr = float(np.quantile(np.concatenate(probs), 1.0 - target))
        try:
            cache.write_text(json.dumps(
                {"model_mtime": self._mtime, "target": target, "min_score": thr,
                 "n_files": len(files)}, indent=2))
        except Exception:  # noqa: BLE001
            log.exception("could not write calibration cache")
        log.info("calibrated min_score=%.4f for target flag rate %.2f", thr, target)
        return thr

    def _load(self) -> None:
        with self.model_path.open("rb") as f:
            self.blend = pickle.load(f)
        self._mtime = self.model_path.stat().st_mtime
        if self._auto_calibrate:
            self.min_score = self._calibrate_min_score()
        log.info(
            "model loaded (mtime=%s, adaptive: min_score=%.2f max_pos_frac=%.2f min_pos=%d)",
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._mtime)),
            self.min_score,
            self.max_pos_frac,
            self.min_pos,
        )

    def _maybe_reload(self) -> None:
        """Hot-reload when the retrain daemon atomically swaps the artifact."""
        try:
            if self.model_path.stat().st_mtime != self._mtime:
                self._load()
        except Exception:
            log.exception("model hot-reload failed; keeping current model")

    @staticmethod
    def _valid_hand(h: Any) -> bool:
        return isinstance(h, dict) and isinstance(h.get("actions"), list)

    def score_chunks(self, chunks: Sequence[Any]) -> List[float]:
        t0 = time.time()
        n = len(chunks or [])
        if n == 0:
            return []
        self._maybe_reload()

        rows, usable_idx = [], []
        for i, chunk in enumerate(chunks):
            try:
                hands = [h for h in (chunk or []) if self._valid_hand(h)]
                if not hands:
                    continue
                rows.append(chunk_features(hands[:MAX_HANDS_PER_CHUNK]))
                usable_idx.append(i)
            except Exception:
                log.exception("featurization failed for chunk %d", i)

        scores = np.full(n, FALLBACK_SCORE, dtype=float)
        if rows:
            try:
                X = self.blend.featurize(rows)
                # v5: the model's calibrated probability decides HOW MANY chunks are
                # flagged; the rank decides WHICH. See pipeline.threshold.shape_adaptive.
                shaped = shape_adaptive(
                    self.blend.score_prob(X),
                    self.blend.score_rank(X),
                    min_score=self.min_score,
                    max_pos_frac=self.max_pos_frac,
                    min_pos=self.min_pos,
                )
                for j, i in enumerate(usable_idx):
                    scores[i] = shaped[j]
            except Exception:
                log.exception("model scoring failed; serving fallback scores")

        out = [round(float(min(max(s, 0.0), 1.0)), 6) for s in scores]
        log.info(
            "scored %d chunks (%d usable) in %.2fs, positives=%d",
            n, len(usable_idx), time.time() - t0, sum(s >= 0.5 for s in out),
        )
        return out
