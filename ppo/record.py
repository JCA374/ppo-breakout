"""Record one full Breakout game with a trained agent, keeping everything the site and the videos need:
the colour frames, the agent's own 4x84x84 observation, actions, action probabilities, value estimates and rewards.

envpool has no colour render, so this uses gymnasium's ALE BreakoutNoFrameskip-v4 with the standard wrappers
(AtariPreprocessing: frame skip 4 + max-pool, 84x84 grayscale, up to 30 no-ops; FrameStack 4), the same
preprocessing the agent was trained on through envpool.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import Agent

gym.register_envs(ale_py)
ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT"]


def load(path):
    st = torch.load(path, weights_only=False, map_location="cpu")
    a = Agent(); a.load_state_dict(st["model"]); a.eval()
    return a


def record_episode(agent, seed, max_steps=27000, keep_rgb=True, keep_obs=True, greedy=False, rgb_every=1):
    """agent=None plays uniformly random. Returns a dict of per-step arrays (step t: obs seen, action taken, ...)."""
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array", repeat_action_probability=0.0,
                   full_action_space=False, max_episode_steps=108000)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
                             grayscale_obs=True, scale_obs=False)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    o, info = env.reset(seed=int(seed))
    stack = np.repeat(o[None], 4, axis=0)  # envpool starts with a zero-padded stack; the difference is one frame
    stack[:3] = 0
    out = dict(rgb=[], obs=[], action=[], probs=[], value=[], reward=[], lives=[])
    score, steps = 0.0, 0
    while True:
        x = torch.from_numpy(stack[None]).float()
        if agent is None:
            a, p, v = int(rng.integers(4)), np.full(4, .25), 0.0
        else:
            with torch.no_grad():
                h = agent.network(x / 255.0)
                logits = agent.actor(h); v = float(agent.critic(h))
                p = torch.softmax(logits, 1)[0].numpy()
            a = int(p.argmax()) if greedy else int(rng.choice(4, p=p / p.sum()))
        if keep_rgb and steps % rgb_every == 0:
            out["rgb"].append(env.unwrapped.render())
        if keep_obs:
            out["obs"].append(stack.copy())
        o, r, term, trunc, info = env.step(a)
        out["action"].append(a); out["probs"].append(p); out["value"].append(v)
        out["reward"].append(float(r)); out["lives"].append(int(info.get("lives", 0)))
        score += r; steps += 1
        stack = np.concatenate([stack[1:], o[None]], 0)
        if term or trunc or steps >= max_steps:
            if keep_rgb:
                out["rgb"].append(env.unwrapped.render())
            break
    env.close()
    res = {k: np.array(v) for k, v in out.items()}
    res["score"] = score; res["steps"] = steps; res["truncated"] = bool(trunc or steps >= max_steps)
    return res


if __name__ == "__main__":
    ag = load(sys.argv[1]) if len(sys.argv) > 1 else None
    r = record_episode(ag, 7, keep_rgb=False, keep_obs=False)
    print("score", r["score"], "steps", r["steps"])
