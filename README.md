# p44_miner_v3 — Poker44 (SN126) bot-detection miner

Model name: **rankblend** (v3). A rank-blended ensemble that scores chunks of
miner-visible poker hands and returns one bot-risk score in `[0, 1]` per chunk.

## Model flow

`chunks -> behavioral features -> 3-member ensemble -> in-batch rank fusion ->
threshold-shaped risk_scores`

- **Features** (`pipeline/features.py`): computed only from miner-visible behavioral
  fields — action-type sequences, big-blind-normalized bet sizings re-quantized to the
  validator's bucket grid, action/actor/street entropies, pot dynamics, cross-hand
  signature/regularity statistics, hashed action n-grams, and seat-count-invariant
  per-player rates. No hole cards, board cards, hand outcomes, timing, or player
  identifiers are used. Per-hand features are aggregated to the chunk with 7 order
  statistics (mean/std/min/max/q10/q50/q90).
- **Model** (`pipeline/model.py`): three decorrelated members fused by in-batch rank —
  (1) a stacked GBDT (LightGBM + XGBoost + CatBoost + ExtraTrees → LogisticRegression
  meta), (2) a monotone-constrained LightGBM trio with sign constraints mined from
  cross-date Spearman stability, (3) a StandardScaler → PCA → MLP trio.
- **Shaping** (`pipeline/threshold.py`, `serve/scorer.py`): rank fusion sets the
  ordering; a strictly monotone remap moves the deployment threshold onto 0.5 and a
  batch positive budget bounds the positive rate, so AP and recall@FPR reflect the
  model's own ranking. `serve/scorer.py` enforces the exact-length / never-raise
  response contract.
- **Training** (`pipeline/train.py`, `pipeline/retrain.py`, `pipeline/dataset.py`):
  trained only on the public Poker44 benchmark
  (`api.poker44.net/api/v1/benchmark`), sanitized through the validator's
  `prepare_hand_for_miner` so training matches the live payload (train == serve).
  Walk-forward validation by release date; daily guarded retrain with a no-regression
  promotion gate.

## Serving

`serve/miner.py` is the Bittensor neuron (netuid 126). It attaches `serve/scorer.py`
behind the axon and returns `risk_scores` for the `DetectionSynapse` contract.

## Note on weights

Trained model weights are withheld (private). This repository publishes the full
model flow — feature extraction, architecture, score shaping, and training pipeline —
that produces the served risk scores.

License: MIT.
