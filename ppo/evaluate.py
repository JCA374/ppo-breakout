"""Evaluate a checkpoint on full Breakout games (all 5 lives, unclipped score).

Standard Atari evaluation setup (envpool Breakout-v5 == BreakoutNoFrameskip-v4 + standard wrappers): frame skip 4
with max-pool, 84x84 grayscale, 4-frame stack, up to 30 no-op starts, no sticky actions, minimal action set,
108k-frame cap. Unlike training, no life-loss terminals and no reward clipping. Actions are sampled from the
policy (the same stochastic policy that was trained).

    python ppo/evaluate.py checkpoints/main/model_010000k.pt --episodes 10 --seed 12345 [--out results/goal_x.json]
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Agent, ROOT
import envpool


def load_agent(path):
    st = torch.load(path, weights_only=False, map_location="cpu")
    agent = Agent(); agent.load_state_dict(st["model"]); agent.eval()
    return agent, st


def evaluate(agent, episodes=10, seed=12345, greedy=False, threads=None, protocol="full"):
    """Run `episodes` full games (all 5 lives), one per env in parallel, each with its own seed; the score is the real
    (unclipped) score of the whole game in both protocols.
    protocol "full":     no wrapper acts at a lost life; the agent must relaunch the ball itself (strict).
    protocol "wrappers": the standard training wrappers (episodic life + fire reset: a lost life is an episode
                         boundary with a soft reset), exactly as in training; the game and its score continue."""
    episodic = protocol == "wrappers"
    env = envpool.make("Breakout-v5", env_type="gym", num_envs=episodes, batch_size=episodes, seed=seed,
                       num_threads=threads or min(episodes, 8), episodic_life=episodic, reward_clip=False,
                       repeat_action_probability=0.0, noop_max=30, full_action_space=False,
                       max_episode_steps=int(108000 / 4))
    obs, _ = env.reset()
    score = np.zeros(episodes); length = np.zeros(episodes, int); done = np.zeros(episodes, bool)
    truncated = np.zeros(episodes, bool)
    with torch.no_grad():
        while not done.all():
            x = torch.from_numpy(obs).float()
            if greedy:
                logits = agent.actor(agent.network(x / 255.0)); a = logits.argmax(1)
            else:
                a, _, _, _ = agent.get_action_and_value(x)
            obs, r, term, trunc, info = env.step(a.numpy().astype(np.int32))
            live = ~done
            score[live] += info["reward"][live]; length[live] += 1
            cap = length >= 27000                              # 108k frames per GAME (in the wrapper protocol envpool's
            truncated |= live & (trunc | cap)                  # own limit counts per life, so it is enforced here)
            done |= info["terminated"].astype(bool) | trunc | cap   # game over (all lives) or the cap
    env.close()
    return score.tolist(), length.tolist(), truncated.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--label", default=""); ap.add_argument("--note", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--protocol", choices=["full", "wrappers"], default="full")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    agent, st = load_agent(a.ckpt)
    t = time.time()
    scores, lengths, trunc = evaluate(agent, a.episodes, a.seed, a.greedy, protocol=a.protocol)
    m, sd = float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    half = 2.262 * sd / math.sqrt(len(scores)) if len(scores) == 10 else 1.96 * sd / math.sqrt(max(1, len(scores)))
    res = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), policy=os.path.relpath(a.ckpt, ROOT), frames=st.get("frames"),
               n=len(scores), score=m, std=sd, ci=[m - half, m + half], min=min(scores), max=max(scores),
               scores=scores, lengths=lengths, truncated=trunc, seed=a.seed, greedy=a.greedy, protocol=a.protocol,
               label=a.label or os.path.basename(a.ckpt), note=a.note, seconds=round(time.time() - t, 1))
    print(json.dumps({k: v for k, v in res.items()}, indent=None))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    # log to MLflow too, as a separate evaluation run
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}"); mlflow.set_experiment("ppo-breakout-eval")
    with mlflow.start_run(run_name=f"eval {res['label']} [{a.protocol}]"):
        mlflow.log_params(dict(policy=res["policy"], episodes=a.episodes, seed=a.seed, greedy=a.greedy, protocol=a.protocol,
                               frames=res["frames"]))
        mlflow.log_metrics(dict(eval_mean=m, eval_std=sd, eval_min=res["min"], eval_max=res["max"]))
        for i, s in enumerate(scores):
            mlflow.log_metric("eval_episode_return", s, step=i)


if __name__ == "__main__":
    main()
