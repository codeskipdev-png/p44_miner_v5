// PM2 deployment for the v5 arm (runs ALONGSIDE the other arms, own UID/port).
// Usage:  pm2 start serve/ecosystem.config.js   (from 04_our_miner_v5/)
// Config lives in 04_our_miner_v5/.env (loaded by python-dotenv in-process).
//
// Retrain is driven by the unified ../retrain_all.py orchestrator (one shared
// benchmark scrape, then each arm retrains into its OWN artifact), so there is
// deliberately no per-miner retrain app here.

const ROOT = "/root/Skip/poker/SN126/04_our_miner_v5";
const PY = ROOT + "/.venv/bin/python";

module.exports = {
  apps: [
    {
      name: "p44_miner_v5",
      cwd: ROOT,
      script: PY,
      interpreter: "none",
      args: ["-m", "serve.miner", "--logging.info"],
      autorestart: true,
      max_restarts: 50,
      restart_delay: 15000,
      env: { PYTHONPATH: ROOT },
    },
  ],
};
