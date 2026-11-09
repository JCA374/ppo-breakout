# PPO Breakout

**🌐 Live site: https://jca374.github.io/ppo-breakout/**

A PPO agent that learns Atari Breakout from raw pixels on an 8-core laptop CPU (no GPU), tracked in MLflow, with a
live progress site and narrated explainer videos about how PPO works.

**Result:** after 40 M frames (9.6 h of CPU training) the agent averages **391.6** on the 10-game goal check
(full games, 5 lives, real score; goal ≥ 300) and **407.6 ± 6.7** on 100 fresh games. Model: release v1.0.

- `ppo/train.py`: PPO (40 M frames = 10 M agent steps), based on CleanRL's `ppo_atari_envpool.py` (Nature CNN, 8 envs × 128 steps, 4 epochs ×
  4 minibatches, clip 0.1, lr 2.5e-4 annealed, GAE λ 0.95, γ 0.99), with MLflow logging, checkpoints every 500k frames
  and automatic resume from `checkpoints/<run>/latest.pt`.
- `supervise.sh`: keeps a run alive (restarts after a crash).
- `ppo/evaluate.py`: the goal check: 10 full games (5 lives, unclipped), seeds never used in training.
- `ppo/record.py`: records a game with colour frames for the site and the videos.
- `ppo/export_mlflow.py`: MLflow file store → `docs/data/mlflow.json` for the site.
- `publish.py` + `site_hooks.py`: rebuilds and pushes the site every 15 minutes (`publish_loop.sh`).
- `video/`: the explainer videos (animations drawn frame by frame from real games and run data + Kokoro narration, Whisper-checked).

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install envpool "gymnasium[atari]" ale-py mlflow opencv-python-headless
setsid nohup ./supervise.sh main --torch-threads 4 --total-frames 40000000 > /dev/null 2>&1 < /dev/null &
.venv/bin/python ppo/evaluate.py checkpoints/main/model_039997k.pt --episodes 10 --seed 12345
.venv/bin/python ppo/final_eval.py checkpoints/main/model_039997k.pt   # 100 fresh games, both protocols
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns
```

`mlruns/` and checkpoints are not in the repo; the best model is attached to a GitHub release.
