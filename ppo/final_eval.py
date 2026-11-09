"""The pre-registered final evaluation (see the journal entry of 2026-11-09 ~01:10).

    python ppo/final_eval.py CKPT [CKPT ...]
For each checkpoint: 100 full games, strict protocol, seeds 900000-900099 -> results/final100_<frames>.json
(also the training-wrapper protocol on the same seeds). Logged to MLflow (experiment ppo-breakout-eval) by evaluate.py's
conventions here too.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_agent, evaluate, ROOT

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

torch.set_num_threads(4)
mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}"); mlflow.set_experiment("ppo-breakout-eval")
for ck in sys.argv[1:]:
    agent, st = load_agent(ck)
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": os.path.relpath(ck, ROOT), "frames": st.get("frames"), "seed": 900000}
    for proto in ("full", "wrappers"):
        torch.manual_seed(0)
        s, L, tr = evaluate(agent, 100, 900000, protocol=proto, threads=8)
        s = np.array(s); se = float(s.std(ddof=1) / math.sqrt(len(s)))
        out[proto] = dict(n=len(s), mean=float(s.mean()), se=se, ci=[float(s.mean() - 1.96 * se), float(s.mean() + 1.96 * se)],
                          median=float(np.median(s)), min=float(s.min()), max=float(s.max()), share_300=float((s >= 300).mean()),
                          scores=s.tolist(), truncated=int(sum(tr)))
        with mlflow.start_run(run_name=f"final100 {Path(ck).stem} [{proto}]"):
            mlflow.log_params(dict(policy=out["policy"], episodes=100, seed=900000, protocol=proto, frames=out["frames"]))
            mlflow.log_metrics(dict(eval_mean=out[proto]["mean"], eval_se=se, eval_min=out[proto]["min"], eval_max=out[proto]["max"],
                                    eval_share_300=out[proto]["share_300"]))
        print(Path(ck).stem, proto, f"mean {s.mean():.1f} ± {se:.1f}  median {np.median(s):.0f}  >=300: {100 * (s >= 300).mean():.0f}%", flush=True)
    json.dump(out, open(ROOT / "results" / f"final100_{Path(ck).stem.split('_')[1]}.json", "w"), indent=1)
