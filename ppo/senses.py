"""'How it sees': the agent's real input (4 stacked 84x84 frames) and saliency maps at a few moments of a real game.

Saliency is perturbation-based (Greydanus et al. 2017, "Visualizing and Understanding Atari Agents"): blur a small
patch of the input and measure how much the output changes. Two maps: the actor (how much the action
probabilities change, 0.5 * squared L2 distance of the logits) and the critic (how much the value estimate changes).

    python ppo/senses.py checkpoints/main/model_010000k.pt docs/senses
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record import load, record_episode, ACTIONS


def saliency(agent, stack, stride=4, sigma=5):
    """stack: 4x84x84 uint8. Returns (actor_map, critic_map), each 84x84 >= 0."""
    x = stack.astype(np.float32)
    blurred = np.stack([gaussian_filter(f, sigma=3) for f in x])
    ys, xs = np.mgrid[0:84, 0:84]
    centers = [(i, j) for i in range(0, 84, stride) for j in range(0, 84, stride)]
    batch = []
    for (i, j) in centers:
        m = np.exp(-((ys - i) ** 2 + (xs - j) ** 2) / (2 * sigma ** 2))[None]
        batch.append(x * (1 - m) + blurred * m)
    with torch.no_grad():
        def out(b):
            h = agent.network(torch.from_numpy(np.asarray(b, np.float32)) / 255.0)
            return agent.actor(h), agent.critic(h)[:, 0]
        l0, v0 = out(x[None])
        L, V = [], []
        for k in range(0, len(batch), 128):
            l, v = out(batch[k:k + 128]); L.append(l); V.append(v)
        L, V = torch.cat(L), torch.cat(V)
    a = (0.5 * ((L - l0) ** 2).sum(1)).numpy(); c = (0.5 * (V - v0) ** 2).numpy()
    n = 84 // stride
    A = np.zeros((84, 84)); C = np.zeros((84, 84))
    small_a = a.reshape(n, n); small_c = c.reshape(n, n)
    A = np.array(Image.fromarray(small_a.astype(np.float32)).resize((84, 84), Image.BILINEAR))
    C = np.array(Image.fromarray(small_c.astype(np.float32)).resize((84, 84), Image.BILINEAR))
    return A, C


def overlay(gray84, smap, color):
    """Saliency drawn over the newest frame, upscaled 4x."""
    g = np.repeat(gray84[..., None].astype(np.float32), 3, 2) * 0.6
    s = smap / (smap.max() + 1e-12)
    s = np.clip(s / 0.6, 0, 1)[..., None]
    img = g * (1 - s) + np.array(color, np.float32) * s
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((336, 336), Image.NEAREST)


def build(ckpt, out_dir, seed=4242, moments=4):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    agent = load(ckpt)
    ep = record_episode(agent, seed, max_steps=3000)
    n = len(ep["action"])
    # pick moments spread over the game where something is happening (the ball is in play: value changes)
    # the ball moves between the last two frames below the bricks (rows 45-76 of 84; the paddle is excluded)
    moving = [t for t in range(n) if np.abs(ep["obs"][t][3, 45:76].astype(int) - ep["obs"][t][2, 45:76]).sum() > 150]
    pool = moving if len(moving) >= moments else list(range(n))
    picks = [pool[int(len(pool) * f)] for f in np.linspace(0.1, 0.85, moments)]
    items = []
    for k, t in enumerate(picks):
        st = ep["obs"][t]
        A, C = saliency(agent, st)
        for i in range(4):
            Image.fromarray(st[i]).resize((168, 168), Image.NEAREST).save(out / f"m{k}_f{i}.png")
        Image.fromarray(ep["rgb"][t]).resize((320, 420), Image.NEAREST).save(out / f"m{k}_rgb.png")
        overlay(st[-1], A, (255, 80, 40)).save(out / f"m{k}_actor.png")
        overlay(st[-1], C, (60, 160, 255)).save(out / f"m{k}_critic.png")
        p = ep["probs"][t]
        items.append(dict(k=k, step=t, action=ACTIONS[int(ep["action"][t])], probs=[round(float(x), 3) for x in p],
                          value=round(float(ep["value"][t]), 3)))
    meta = dict(ckpt=str(ckpt).split("checkpoints/")[-1], seed=seed, score=ep["score"], steps=ep["steps"], moments=items)
    json.dump(meta, open(out / "senses.json", "w"), indent=1)
    return meta


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(build(sys.argv[1], sys.argv[2]))
