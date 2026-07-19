"""Poker44 (SN126) production miner neuron.

Self-contained: defines the DetectionSynapse wire contract locally and serves
ChunkScorer behind a bittensor axon with allowlist/permit blacklisting.

Run:
    python -m serve.miner --netuid 126 --wallet.name <cold> --wallet.hotkey <hot> \
        --subtensor.network finney --axon.port 8091
"""
# NOTE: do NOT add `from __future__ import annotations` — bittensor's
# axon.attach() introspects forward()'s synapse annotation via issubclass(),
# which breaks if PEP 563 stringizes it.

import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import bittensor as bt

# bittensor 10.x (the v431 network upgrade requires SDK >= 10.3.0) renamed the
# lowercase factory names to CapCase and dropped the old aliases. Restore the
# lowercase names to the new classes so our call sites keep working on 9.x and 10.x.
for _old, _new in (("subtensor", "Subtensor"), ("wallet", "Wallet"),
                   ("axon", "Axon"), ("config", "Config")):
    if not hasattr(bt, _old) and hasattr(bt, _new):
        setattr(bt, _old, getattr(bt, _new))

from dotenv import load_dotenv
from pydantic import ConfigDict, Field

from serve.scorer import ChunkScorer

REPO_ROOT = Path(__file__).resolve().parent.parent
# Load P44_* vars from the project .env regardless of how we're launched
# (pm2's Node-side dotenv only covers the wallet args passed on the CLI).
load_dotenv(REPO_ROOT / ".env")


class DetectionSynapse(bt.Synapse):
    """Must match the validator's synapse (name + fields) exactly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: List[List[dict]] = Field(default_factory=list)
    risk_scores: Optional[List[float]] = None
    predictions: Optional[List[bool]] = None
    model_manifest: Optional[Dict[str, Any]] = None

    required_hash_fields: ClassVar[List[str]] = ["chunks"]


def _file_sha256(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.read_bytes())
    return digest.hexdigest()


def build_manifest() -> Dict[str, Any]:
    impl = [
        REPO_ROOT / "serve" / "scorer.py",
        REPO_ROOT / "pipeline" / "features.py",
        REPO_ROOT / "pipeline" / "model.py",
        REPO_ROOT / "pipeline" / "threshold.py",
    ]
    return {
        "schema_version": "1",
        "open_source": os.getenv("P44_OPEN_SOURCE", "true").lower() == "true",
        "model_name": os.getenv("P44_MODEL_NAME", "rankblend"),
        "model_version": os.getenv("P44_MODEL_VERSION", "1"),
        "framework": "sklearn+lightgbm+xgboost+catboost rank-blend",
        "license": "MIT",
        "repo_url": os.getenv("P44_REPO_URL", ""),
        "repo_commit": os.getenv("P44_REPO_COMMIT", ""),
        "training_data_statement": (
            "Trained exclusively on the public Poker44 benchmark "
            "(api.poker44.net/api/v1/benchmark), sanitized with the reference "
            "prepare_hand_for_miner so training matches the live payload."
        ),
        "training_data_sources": ["poker44-public-benchmark"],
        "private_data_attestation": (
            "No validator-private or label-bearing live data was used for training."
        ),
        "inference_mode": "local",
        "implementation_files": [str(p.relative_to(REPO_ROOT)) for p in impl],
        "implementation_sha256": _file_sha256(impl),
    }


class Miner:
    def __init__(self, config: bt.config):
        self.config = config
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.scorer = ChunkScorer()
        self.manifest = build_manifest()
        allow = os.getenv("P44_ALLOWED_VALIDATOR_HOTKEYS", "").split()
        self.allowed_hotkeys = set(h for h in allow if h)
        self.min_stake = float(os.getenv("P44_MIN_VALIDATOR_STAKE", "1000"))

        self.axon = bt.Axon(wallet=self.wallet, config=config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Miner uid={self.uid} manifest={self.manifest['implementation_sha256'][:12]}")

    # -- live capture --------------------------------------------------------
    @staticmethod
    def _challenge_digest(chunks: List[List[dict]]) -> str:
        """Order-invariant fingerprint of a challenge.

        The same canonical challenge can reach us from multiple validators; we
        want to store it once regardless of the order chunks/hands arrive in.
        - hands are hashed with sort_keys (dict keys sorted) so identical hand
          CONTENT hashes equally, while the action LIST stays in sequence
          (sort_keys does not reorder lists) — action order is meaningful;
        - a chunk = the multiset of its hand-hashes (hand order within a chunk
          is not label-relevant), so we sort them;
        - a challenge = the multiset of its chunk-hashes, so we sort those too.
        """
        chunk_hashes = []
        for chunk in chunks:
            hand_hashes = sorted(
                hashlib.sha256(
                    json.dumps(h, sort_keys=True, default=str).encode()
                ).hexdigest()
                for h in (chunk or [])
            )
            chunk_hashes.append(hashlib.sha256("".join(hand_hashes).encode()).hexdigest())
        return hashlib.sha256("".join(sorted(chunk_hashes)).encode()).hexdigest()[:16]

    def _capture(self, synapse: DetectionSynapse, chunks: List[List[dict]]) -> None:
        """Kick off a background, deduped capture — never blocks the response."""
        if not chunks:
            return
        hotkey = synapse.dendrite.hotkey if synapse.dendrite else ""
        threading.Thread(
            target=self._capture_worker, args=(list(chunks), hotkey), daemon=True
        ).start()

    def _capture_worker(self, chunks: List[List[dict]], hotkey: str) -> None:
        """Persist an incoming validator payload once, order-invariantly deduped.

        These unlabeled captures drive KS feature selection and drift tracking
        in the retrain pipeline, so a challenge repeated across validators must
        not be double-counted.
        """
        try:
            cap_dir = Path(os.getenv(
                "P44_CAPTURE_DIR",
                "/root/Skip/poker/SN126/03_data/real_challenge/raw",
            ))
            cap_dir.mkdir(parents=True, exist_ok=True)
            digest = self._challenge_digest(chunks)
            if any(cap_dir.glob(f"*_{digest}.json")):
                bt.logging.info(f"capture: duplicate {digest} already stored, skipped")
                return  # this exact challenge (any order, any validator) is stored
            cap_max = int(os.getenv("P44_CAPTURE_MAX_FILES", "200"))
            if len(list(cap_dir.glob("request_*.json"))) > cap_max:
                bt.logging.info(f"capture: file cap ({cap_max}) reached, skipped")
                return
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            total_hands = sum(len(c or []) for c in chunks)
            record = {
                "saved_at": stamp,
                "validator_hotkey": hotkey,
                "capture_digest": digest,
                "chunk_count": len(chunks),
                "chunk_sizes": [len(c or []) for c in chunks],
                "total_hands": total_hands,
                "chunks": chunks,
            }
            # atomic: write to a temp name, then rename, so the retrain daemon
            # never reads a half-written file.
            tmp = cap_dir / f".request_{stamp}_{digest}.tmp"
            tmp.write_text(json.dumps(record))
            tmp.rename(cap_dir / f"request_{stamp}_{digest}.json")
            bt.logging.info(
                f"capture: SAVED new challenge {digest} "
                f"({len(chunks)} chunks / {total_hands} hands from {hotkey[:8]})"
            )
        except Exception:
            bt.logging.warning(f"live capture failed (non-fatal): {traceback.format_exc(limit=2)}")

    # -- axon handlers ------------------------------------------------------
    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        t0 = time.time()
        chunks = synapse.chunks or []
        if os.getenv("P44_CAPTURE", "1") == "1" and chunks:
            self._capture(synapse, chunks)
        try:
            scores = self.scorer.score_chunks(chunks)
        except Exception:
            bt.logging.error(f"scorer raised (must not happen):\n{traceback.format_exc()}")
            scores = [0.1] * len(chunks)
        if len(scores) != len(chunks):  # absolute contract: length must match
            scores = (scores + [0.1] * len(chunks))[: len(chunks)]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.manifest)
        bt.logging.info(
            f"served {len(chunks)} chunks in {time.time()-t0:.2f}s "
            f"(positives={sum(synapse.predictions)})"
        )
        if os.getenv("P44_LOG_SCORES", "1") == "1":
            bt.logging.info(f"risk_scores={[round(float(s), 4) for s in scores]}")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not hotkey:
            return True, "no hotkey"
        if self.allowed_hotkeys:
            if hotkey in self.allowed_hotkeys:
                return False, "allowlisted"
            return True, "not in allowlist"
        if hotkey not in self.metagraph.hotkeys:
            return True, "unregistered"
        uid = self.metagraph.hotkeys.index(hotkey)
        if not bool(self.metagraph.validator_permit[uid]):
            return True, "no validator permit"
        if float(self.metagraph.S[uid]) < self.min_stake:
            return True, "stake below minimum"
        return False, "ok"

    async def priority(self, synapse: DetectionSynapse) -> float:
        hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if hotkey and hotkey in self.metagraph.hotkeys:
            return float(self.metagraph.S[self.metagraph.hotkeys.index(hotkey)])
        return 0.0

    # -- main loop -----------------------------------------------------------
    def run(self):
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        bt.logging.info(f"Axon serving on port {self.config.axon.port}")
        last_sync = 0.0
        while True:
            if time.time() - last_sync > 300:
                try:
                    self.metagraph.sync(subtensor=self.subtensor)
                    last_sync = time.time()
                    bt.logging.info(
                        f"metagraph synced | uid={self.uid} "
                        f"incentive={float(self.metagraph.I[self.uid]):.6f}"
                    )
                except Exception as e:
                    bt.logging.warning(f"metagraph sync failed: {e}")
            time.sleep(15)


def parse_config() -> bt.config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netuid", type=int, default=int(os.getenv("P44_NETUID", "126")))
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.axon.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)
    # bittensor 10.x's Config drops bare top-level args (e.g. --netuid) that aren't
    # registered by a component add_args(); 9.x kept them. Restore netuid from the raw
    # parse (respects an explicit --netuid and the default) so config.netuid is set.
    if config.get("netuid") is None:
        config.netuid = parser.parse_known_args()[0].netuid
    # Env-driven defaults (from .env) so pm2 can run the module with no CLI args.
    # Explicit CLI flags still win (argparse defaults are only applied when unset).
    if os.getenv("WALLET_NAME") and config.wallet.name == "default":
        config.wallet.name = os.getenv("WALLET_NAME")
    if os.getenv("HOTKEY") and config.wallet.hotkey == "default":
        config.wallet.hotkey = os.getenv("HOTKEY")
    if os.getenv("AXON_PORT"):
        config.axon.port = int(os.getenv("AXON_PORT"))
    if os.getenv("SUBTENSOR_NETWORK"):
        config.subtensor.network = os.getenv("SUBTENSOR_NETWORK")
        config.subtensor.chain_endpoint = None
    return config


if __name__ == "__main__":
    config = parse_config()
    bt.logging(config=config)
    # bittensor 10.x: bt.logging(config=...) alone no longer emits INFO to stdout
    # (9.x did). Explicitly enable so pm2 captures serve counts / uid / incentive.
    bt.logging.enable_info()
    Miner(config).run()
