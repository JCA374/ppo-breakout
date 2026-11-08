"""Export the MLflow file store (./mlruns) to one JSON file for the site: every run of the training and evaluation
experiments with its parameters, status and (thinned) metric histories.

    python ppo/export_mlflow.py docs/data/mlflow.json
"""
import json, os, sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ["ppo-breakout", "ppo-breakout-eval", "smoke"]
CURVES = ["ep_return_mean", "episodic_return", "value_loss", "policy_loss", "entropy", "approx_kl", "old_approx_kl",
          "clipfrac", "explained_variance", "sps", "lr", "rollout_seconds", "update_seconds", "eval_episode_return"]


def thin(points, n=500):
    if len(points) <= n:
        return points
    # bucket means keep the shape of noisy curves better than picking every k-th point
    k = len(points) / n
    out = []
    for i in range(n):
        b = points[int(i * k):int((i + 1) * k)] or points[-1:]
        out.append([b[-1][0], round(sum(p[1] for p in b) / len(b), 5)])
    return out


def export(out):
    c = MlflowClient(tracking_uri=f"file:{ROOT / 'mlruns'}")
    data = {"experiments": []}
    for name in EXPERIMENTS:
        exp = c.get_experiment_by_name(name)
        if exp is None:
            continue
        e = {"name": name, "id": exp.experiment_id, "runs": []}
        for r in c.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"], max_results=200):
            run = {"id": r.info.run_id, "name": r.info.run_name, "status": r.info.status,
                   "start": r.info.start_time, "end": r.info.end_time, "params": r.data.params,
                   "last": {k: round(v, 5) for k, v in r.data.metrics.items()}, "curves": {}}
            for m in CURVES:
                if m in r.data.metrics:
                    h = sorted(c.get_metric_history(r.info.run_id, m), key=lambda x: (x.step, x.timestamp))
                    run["curves"][m] = thin([[x.step, x.value] for x in h])
            try:
                run["artifacts"] = [a.path for a in c.list_artifacts(r.info.run_id, "checkpoints")]
            except Exception:
                run["artifacts"] = []
            e["runs"].append(run)
        data["experiments"].append(e)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w"), separators=(",", ":"))
    return data


if __name__ == "__main__":
    d = export(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "docs/data/mlflow.json"))
    print({e["name"]: len(e["runs"]) for e in d["experiments"]})
