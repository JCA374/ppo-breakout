"""Explainer 4: "Watching it learn": checkpoints from early to late, side by side on the same game.
Scenes match video/ppo4/script.md: one scene per checkpoint (that model alone, large), then all four together.

    python video/build_ppo4.py CKPT1 CKPT2 CKPT3 CKPT4 [--analyse]
--analyse records the games and writes video/ppo4/facts.json (scores, steps) for the script.
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs=4); ap.add_argument("--analyse", action="store_true")
ap.add_argument("--seed", type=int, default=31); ap.add_argument("--only", default="")
A = ap.parse_args()
only = {int(x) for x in A.only.split(",") if x}
CACHE = ROOT / "video" / "build" / "ppo4_games.pkl"
fr_of = lambda p: int(Path(p).stem.split("_")[1][:-1]) * 1000

if A.analyse or not CACHE.exists():
    import torch; torch.set_num_threads(2)
    from record import load, record_episode
    G = [record_episode(load(c), A.seed, max_steps=6000, keep_obs=False, rgb_every=3) for c in A.ckpts]
    CACHE.parent.mkdir(parents=True, exist_ok=True); pickle.dump(G, open(CACHE, "wb"))
    facts = [dict(ckpt=c.split("checkpoints/")[-1], frames=fr_of(c), score=g["score"], steps=int(g["steps"]),
                  truncated=g["truncated"]) for c, g in zip(A.ckpts, G)]
    json.dump(dict(seed=A.seed, games=facts), open(ROOT / "video" / "ppo4" / "facts.json", "w"), indent=1)
    print(json.dumps(facts, indent=1))
    if A.analyse:
        sys.exit(0)
G = pickle.load(open(CACHE, "rb"))


def panel(g, k, w, label):
    """k = agent step. rgb is stored every 3rd step."""
    i = min(k // 3, len(g["rgb"]) - 1)
    im = Image.fromarray(g["rgb"][i]).resize((w, int(w * 210 / 160)), Image.NEAREST)
    sc = float(np.sum(g["reward"][:min(k, len(g["reward"]))]))
    over = k >= len(g["reward"])
    return im, sc, over


def solo(j):
    g = G[j]; f = fr_of(A.ckpts[j])

    def s(t, T):
        im, d = canvas(); header(d, f"After {f / 1e6:.1f} M frames", f"one full game (seed {A.seed}), sped up to fit")
        speed = max(1.0, len(g["reward"]) / ((T - 1.0) * 15))
        k = int(t * 15 * speed)
        p, sc, over = panel(g, k, 380, "")
        im.paste(p, (450, 110))
        text(d, (870, 200), f"score {sc:.0f}", 40, INK, bold=True)
        text(d, (870, 260), f"{speed:.1f}× speed", 20, MUTED)
        if over:
            text(d, (870, 320), f"game over: {g['score']:.0f}" + (" (cut at 6,000 steps)" if g["truncated"] else ""), 26, WARN, bold=True)
        return im
    return s


def together(t, T):
    im, d = canvas(); header(d, "Side by side", f"the same game seed for all four, {A.seed}; each runs until its own game ends")
    longest = max(len(g["reward"]) for g in G)
    speed = max(1.0, longest / ((T - 1.0) * 15))
    k = int(t * 15 * speed)
    for j, g in enumerate(G):
        p, sc, over = panel(g, k, 280, "")
        x = 30 + j * 310
        im.paste(p, (x, 120))
        text(d, (x, 495), f"{fr_of(A.ckpts[j]) / 1e6:.1f} M frames", 20, INK, bold=True)
        text(d, (x, 525), f"score {sc:.0f}" + ("  · over" if over else ""), 20, WARN if over else MUTED)
    text(d, (30, 600), f"{speed:.1f}× speed", 18, MUTED)
    return im


if __name__ == "__main__":
    render_film([solo(0), solo(1), solo(2), solo(3), together], "voice/ppo4", ROOT / "docs" / "video" / "ppo4.mp4",
                ROOT / "video" / "build" / "ppo4", only)
