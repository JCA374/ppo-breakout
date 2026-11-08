"""PPO on Atari Breakout from pixels, CPU only.

Based on CleanRL's ppo_atari_envpool.py (Nature CNN, clip 0.1, GAE 0.95, 4 epochs x 4 minibatches,
lr 2.5e-4 annealed), with:
  * MLflow tracking (local file store ./mlruns): params, every-iteration metrics, checkpoints
  * the site files runs/<run>/stats.csv + events.jsonl
  * checkpoints every --ckpt-every frames and automatic resume from checkpoints/<run>/latest.pt
"""
import argparse, csv, json, os, random, time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import envpool
import mlflow

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-id", default="Breakout-v5")
    p.add_argument("--total-frames", type=int, default=10_000_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--torch-threads", type=int, default=6)
    p.add_argument("--env-threads", type=int, default=0, help="0 = num_envs (capped at 8)")
    p.add_argument("--ckpt-every", type=int, default=500_000, help="frames")
    p.add_argument("--max-minutes", type=float, default=0, help="stop after this wall time (smoke test)")
    p.add_argument("--experiment", default="ppo-breakout")
    a = p.parse_args(argv)
    a.batch_size = a.num_envs * a.num_steps
    a.minibatch_size = a.batch_size // a.num_minibatches
    a.total_steps = a.total_frames // 4  # agent steps (frame skip 4)
    a.num_iterations = a.total_steps // a.batch_size
    return a


def make_env(env_id, n, seed, threads=0, **kw):
    return envpool.make(env_id, env_type="gym", num_envs=n, batch_size=n, seed=seed,
                        num_threads=threads or min(n, 8),
                        episodic_life=True, reward_clip=True,   # training-time wrappers (standard)
                        repeat_action_probability=0.0, noop_max=30, full_action_space=False,
                        max_episode_steps=int(108000 / 4), **kw)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Nature CNN torso, separate linear actor and critic heads."""
    def __init__(self, n_actions=4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class RecordEpisodeStatistics:
    """Tracks real (all-lives, unclipped) game returns from envpool's info dict, like CleanRL."""
    def __init__(self, n):
        self.ret = np.zeros(n, np.float32); self.len = np.zeros(n, np.int32)
        self.ret_r = np.zeros(n, np.float32); self.len_r = np.zeros(n, np.int32)

    def step(self, info, trunc):
        self.ret += info["reward"]; self.len += 1
        done = info["terminated"].astype(bool) | trunc  # real game over (all lives) or time limit
        finished = []
        for i in np.where(done)[0]:
            finished.append((float(self.ret[i]), int(self.len[i])))
        self.ret[done] = 0; self.len[done] = 0
        return finished


def event(run_dir, kind, **kw):
    with open(run_dir / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **kw}) + "\n")


STATS_COLS = ["iteration", "global_step", "frames", "time", "sps", "ep_return_mean", "ep_return_max",
              "ep_len_mean", "episodes", "lr", "value_loss", "policy_loss", "entropy", "old_approx_kl",
              "approx_kl", "clipfrac", "explained_variance"]


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    run_dir = ROOT / "runs" / args.run; run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / "checkpoints" / args.run; ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    agent = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    start_iter, global_step, mlflow_run_id, elapsed0 = 1, 0, None, 0.0
    if latest.exists():
        st = torch.load(latest, weights_only=False)
        agent.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        start_iter, global_step = st["iteration"] + 1, st["global_step"]
        mlflow_run_id, elapsed0 = st["mlflow_run_id"], st.get("elapsed", 0.0)
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["np_rng"])
        print(f"resumed from {latest}: iteration {st['iteration']}, frames {4*global_step}", flush=True)

    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment(args.experiment)
    mlrun = mlflow.start_run(run_id=mlflow_run_id, run_name=None if mlflow_run_id else args.run)
    if mlflow_run_id is None:
        mlflow.log_params({k: v for k, v in vars(args).items()})
        event(run_dir, "run_start", run=args.run, mlflow_run_id=mlrun.info.run_id, args=vars(args))
    else:
        event(run_dir, "resume", iteration=start_iter - 1, frames=4 * global_step)
        old_total = st["args"]["total_frames"]
        if old_total != args.total_frames:   # a deliberate change of the schedule: params are immutable, so a tag
            note = (f"total_frames changed from {old_total} to {args.total_frames} at {4 * global_step} frames; "
                    f"the linear learning-rate schedule now runs to the new total")
            mlflow.set_tag(f"change_at_{4 * global_step}", note)
            event(run_dir, "override", what=f"total frames {old_total / 1e6:g} M -> {args.total_frames / 1e6:g} M", reason=note)
    mlflow_run_id = mlrun.info.run_id

    envs = make_env(args.env_id, args.num_envs, args.seed + start_iter, args.env_threads)
    stats_ep = RecordEpisodeStatistics(args.num_envs)
    obs = torch.zeros((args.num_steps, args.num_envs, 4, 84, 84), dtype=torch.uint8)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    logprobs, rewards, dones, values = (torch.zeros((args.num_steps, args.num_envs)) for _ in range(4))

    next_obs = torch.from_numpy(envs.reset()[0])
    next_done = torch.zeros(args.num_envs)
    new_file = not (run_dir / "stats.csv").exists()
    stats_f = open(run_dir / "stats.csv", "a", newline="")
    writer = csv.DictWriter(stats_f, fieldnames=STATS_COLS)
    if new_file:
        writer.writeheader()
    t_start = time.time()
    step_at_start = global_step
    next_ckpt = (4 * global_step // args.ckpt_every + 1) * args.ckpt_every
    recent = []  # finished episode returns this iteration

    for iteration in range(start_iter, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        t_roll = time.time()
        finished = []
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.float())
            values[step] = value.flatten()
            actions[step] = action; logprobs[step] = logprob
            o, r, term, trunc, info = envs.step(action.numpy().astype(np.int32))
            rewards[step] = torch.from_numpy(r.astype(np.float32))
            next_obs = torch.from_numpy(o)
            next_done = torch.from_numpy((term | trunc).astype(np.float32))
            finished += stats_ep.step(info, trunc)
        t_roll = time.time() - t_roll

        t_upd = time.time()
        with torch.no_grad():
            next_value = agent.get_value(next_obs.float()).reshape(1, -1)
            advantages = torch.zeros_like(rewards); lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done; nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]; nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, 4, 84, 84)); b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1); b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1); b_values = values.reshape(-1)
        b_inds = np.arange(args.batch_size); clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb].float(), b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]; ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_unclipped = (newvalue - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        t_upd = time.time() - t_upd
        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        elapsed = elapsed0 + time.time() - t_start
        sps = (global_step - step_at_start) / (time.time() - t_start)
        rets = [f[0] for f in finished]
        row = dict(iteration=iteration, global_step=global_step, frames=4 * global_step, time=round(elapsed, 1),
                   sps=round(sps, 1),
                   ep_return_mean=round(float(np.mean(rets)), 2) if rets else "",
                   ep_return_max=max(rets) if rets else "",
                   ep_len_mean=round(float(np.mean([f[1] for f in finished])), 1) if finished else "",
                   episodes=len(finished), lr=optimizer.param_groups[0]["lr"],
                   value_loss=round(v_loss.item(), 5), policy_loss=round(pg_loss.item(), 5),
                   entropy=round(entropy_loss.item(), 5), old_approx_kl=round(old_approx_kl.item(), 6),
                   approx_kl=round(approx_kl.item(), 6), clipfrac=round(float(np.mean(clipfracs)), 5),
                   explained_variance=round(float(explained_var), 4))
        writer.writerow(row); stats_f.flush()
        m = {k: float(v) for k, v in row.items() if v != "" and k not in ("iteration", "global_step")}
        m["rollout_seconds"] = t_roll; m["update_seconds"] = t_upd
        mlflow.log_metrics(m, step=global_step)
        for r_, l_ in finished:
            mlflow.log_metric("episodic_return", r_, step=global_step)
        if iteration % 10 == 0:
            print(f"it {iteration}/{args.num_iterations} frames {4*global_step} sps {sps:.0f} "
                  f"ret {row['ep_return_mean']} kl {row['approx_kl']} clip {row['clipfrac']} "
                  f"roll {t_roll:.2f}s upd {t_upd:.2f}s", flush=True)

        stop = args.max_minutes and (time.time() - t_start) > 60 * args.max_minutes
        if 4 * global_step >= next_ckpt or iteration == args.num_iterations or stop:
            st = dict(model=agent.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration,
                      global_step=global_step, mlflow_run_id=mlflow_run_id, elapsed=elapsed, args=vars(args),
                      torch_rng=torch.get_rng_state(), np_rng=np.random.get_state())
            tmp = ckpt_dir / "latest.tmp"; torch.save(st, tmp); os.replace(tmp, latest)
            if 4 * global_step >= next_ckpt or iteration == args.num_iterations:
                frames_m = 4 * global_step
                snap = ckpt_dir / f"model_{frames_m // 1000:06d}k.pt"
                torch.save({"model": agent.state_dict(), "frames": frames_m, "iteration": iteration}, snap)
                mlflow.log_artifact(str(snap), artifact_path="checkpoints")
                event(run_dir, "checkpoint", frames=frames_m, path=str(snap.relative_to(ROOT)),
                      score=row["ep_return_mean"],
                      **({"champion": str(snap.relative_to(ROOT))} if (frames_m // args.ckpt_every) % 4 == 0 else {}))
                while next_ckpt <= 4 * global_step:
                    next_ckpt += args.ckpt_every
        if stop:
            print("max-minutes reached, stopping", flush=True)
            break
    else:
        event(run_dir, "run_end", frames=4 * global_step)
        (run_dir / "DONE").write_text(str(4 * global_step))
    envs.close(); stats_f.close(); mlflow.end_run()


if __name__ == "__main__":
    main()
