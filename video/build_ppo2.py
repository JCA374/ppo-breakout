"""Explainer 2: "Advantages and GAE", on a real Breakout game from this run. Scenes match video/ppo2/script.md.

    python video/build_ppo2.py CKPT --analyse     # record the game, write video/ppo2/facts.json (numbers for the script)
    python video/build_ppo2.py CKPT [--only 2]    # render docs/video/ppo2.mp4 (needs the narration exported)
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--analyse", action="store_true")
ap.add_argument("--only", default=""); ap.add_argument("--seed", type=int, default=21)
A = ap.parse_args()
only = {int(x) for x in A.only.split(",") if x}
GAMMA, LAM = 0.99, 0.95
CACHE = ROOT / "video" / "build" / "ppo2_episode.pkl"
FACTS = ROOT / "video" / "ppo2" / "facts.json"


def advantages(r, v, done, lam):
    """GAE as in train.py: done[t] = the life ended after step t (no bootstrap past it)."""
    n = len(r); adv = np.zeros(n); last = 0.0
    for t in range(n - 1, -1, -1):
        nv = 0.0 if (done[t] or t == n - 1) else v[t + 1]
        nonterm = 0.0 if done[t] else 1.0
        delta = r[t] + GAMMA * nv - v[t]
        last = delta + GAMMA * lam * nonterm * last
        adv[t] = last
    return adv


if A.analyse or not CACHE.exists():
    import torch; torch.set_num_threads(2)
    from record import load, record_episode
    ep = record_episode(load(A.ckpt), A.seed, max_steps=4000)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(ep, open(CACHE, "wb"))
EP = pickle.load(open(CACHE, "rb"))
R = np.sign(EP["reward"]).astype(float)                        # clipped, as in training
V = EP["value"].astype(float)
L = EP["lives"]; prev = np.concatenate([[5], L[:-1]])
DONE = (L < prev) | (np.arange(len(L)) == len(L) - 1)          # a life lost after this step
DELTA = np.array([R[t] + GAMMA * (0 if DONE[t] else V[min(t + 1, len(V) - 1)]) - V[t] for t in range(len(R))])
ADV = {lam: advantages(R, V, DONE, lam) for lam in (0.0, LAM, 1.0)}
hits = np.where(R > 0)[0]; losses = np.where(L < prev)[0]
# the window: 160 steps around the first life loss that comes after at least 3 bricks (so we see both kinds of moment)
cand = [l for l in losses if (hits < l).sum() >= 3] or list(losses) or [len(R) - 1]
W1 = min(len(R), cand[0] + 12); W0 = max(0, W1 - 160)

if A.analyse:
    w = slice(W0, W1)
    facts = dict(ckpt=str(A.ckpt).split("checkpoints/")[-1], seed=A.seed, score=EP["score"], steps=int(EP["steps"]),
                 window=[int(W0), int(W1)], bricks_in_window=int(((hits >= W0) & (hits < W1)).sum()),
                 life_lost_at=int(cand[0]) if len(losses) else None,
                 value_range=[round(float(V[w].min()), 2), round(float(V[w].max()), 2)],
                 value_before_loss=round(float(V[max(W0, cand[0] - 3)]), 2) if len(losses) else None,
                 delta_at_loss=round(float(DELTA[cand[0]]), 2) if len(losses) else None,
                 std={str(l): round(float(ADV[l][w].std()), 3) for l in ADV},
                 weight_at={str(k): round((GAMMA * LAM) ** k, 3) for k in (1, 5, 10, 20, 50, 100)},
                 half_life_steps=int(np.ceil(np.log(0.5) / np.log(GAMMA * LAM))),
                 adv_before_loss={str(l): [round(float(x), 2) for x in ADV[l][cand[0] - 10:cand[0] + 1]] for l in ADV} if len(losses) else {},
                 positive_share={str(l): round(float((ADV[l][w] > 0).mean()), 2) for l in ADV})
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(facts, open(FACTS, "w"), indent=1)
    print(json.dumps(facts, indent=1))
    sys.exit(0)

FR = int(Path(A.ckpt).stem.split("_")[1][:-1]) * 1000


def frame(i, w=280):
    i = int(min(max(i, 0), len(EP["rgb"]) - 1))
    return Image.fromarray(EP["rgb"][i]).resize((w, int(w * 210 / 160)), Image.NEAREST)


def plot(d, x0, y0, w, h, series, cursor=None, bars=None, lo=None, hi=None, n=None, label=""):
    """series: [(array over window, colour, width)], bars: (array, ) drawn green/red. Shows window W0..W0+n."""
    n = n or (W1 - W0)
    vals = [s[0][:n] for s in series] + ([bars[:n]] if bars is not None else [])
    lo = lo if lo is not None else min(min(v.min() for v in vals), 0) - 0.05
    hi = hi if hi is not None else max(v.max() for v in vals) + 0.05
    X = lambda k: x0 + k * w / (W1 - W0); Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    d.line([x0, Y(0), x0 + w, Y(0)], fill=(60, 64, 76))
    if bars is not None:
        for k in range(n):
            d.line([X(k), Y(0), X(k), Y(bars[k])], fill=GOOD if bars[k] > 0 else BAD, width=3)
    for arr, col, wd in series:
        if n > 1:
            d.line([(X(k), Y(arr[k])) for k in range(n)], fill=col, width=wd)
    for k in range(n):
        g = W0 + k
        if R[g] > 0:
            d.ellipse([X(k) - 5, y0 + h + 6, X(k) + 5, y0 + h + 16], fill=GOOD)
        if g in losses:
            d.line([X(k), y0, X(k), y0 + h], fill=BAD, width=2)
    if cursor is not None:
        d.line([X(cursor), y0, X(cursor), y0 + h], fill=WARN, width=2)
    text(d, (x0, y0 - 26), label, 17, MUTED)
    return X, Y


def win(a):
    return a[W0:W1]


def legend(d, y=665):
    d.ellipse([90, y + 5, 100, y + 15], fill=GOOD); text(d, (108, y), "brick broken (+1)", 16, MUTED)
    d.line([300, y, 300, y + 20], fill=BAD, width=3); text(d, (310, y), "life lost (the training episode ends)", 16, MUTED)
    text(d, (700, y), f"real game: model at {FR / 1e6:.1f} M frames, steps {W0}–{W1}", 16, MUTED)


def s1(t, T):
    im, d = canvas(); header(d, "Which move earned the points?", "the reward arrives many steps after the move that made it possible")
    k = int(min(W1 - W0 - 1, t * 15 * 0.9 + 0))
    im.paste(frame(W0 + k, 300), (60, 120))
    plot(d, 420, 160, 800, 140, [], n=k + 1, lo=-0.1, hi=1.1, bars=win(R), label="reward per step (clipped)")
    text(d, (420, 360), "A brick breaks here, but the move that sent the ball", 22, INK)
    text(d, (420, 392), "up there happened a dozen or more steps earlier.", 22, INK)
    text(d, (420, 450), "The advantage gives every step its fair share.", 22, WARN, bold=True)
    legend(d)
    return im


def s2(t, T):
    im, d = canvas(); header(d, "The critic's yardstick", "V(s): the critic's guess of the discounted reward still to come")
    k = int(min(W1 - W0 - 1, t / (T * 0.9) * (W1 - W0)))
    im.paste(frame(W0 + k, 300), (60, 120))
    plot(d, 420, 160, 800, 360, [(win(V), ACC, 3)], cursor=k, n=k + 1, label="critic value V")
    legend(d)
    return im


def s3(t, T):
    im, d = canvas(); header(d, "One step of surprise", "δ = r + γ·V(next) − V(now)      (γ = 0.99)")
    k = int(min(W1 - W0, 2 + t / (T * 0.8) * (W1 - W0)))
    plot(d, 90, 150, 1100, 200, [(win(V), ACC, 2)], n=k, label="critic value V")
    plot(d, 90, 420, 1100, 200, [], bars=win(DELTA), n=k, label="TD error δ: green = better than expected, red = worse")
    legend(d)
    return im


def s4(t, T):
    im, d = canvas(); header(d, "Looking all the way ahead", "λ = 1: all real rewards until the life ends, minus V")
    a = ADV[1.0]; k = int(min(W1 - W0, 2 + t / (T * 0.8) * (W1 - W0)))
    plot(d, 90, 150, 1100, 420, [], bars=win(a), n=k, label="advantage with λ = 1 (Monte Carlo)")
    legend(d)
    return im


def s5(t, T):
    im, d = canvas(); header(d, "Generalized advantage estimation", "A = δ₀ + (γλ)·δ₁ + (γλ)²·δ₂ + …      γλ = 0.99 × 0.95 ≈ 0.94")
    ks = np.arange(0, 101)
    wts = (GAMMA * LAM) ** ks
    n = int(len(ks) * ease(t / (T * 0.4)))
    x0, y0, w, h = 90, 150, 520, 380
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    for k in range(n):
        x = x0 + k * w / 101; d.line([x, y0 + h, x, y0 + h - wts[k] * h], fill=ACC, width=3)
    text(d, (x0, y0 - 26), "weight of the surprise k steps ahead", 17, MUTED)
    for k in (0, 25, 50, 100):
        text(d, (x0 + k * w / 101, y0 + h + 16), str(k), 15, MUTED, anchor="mm")
    hl = int(np.ceil(np.log(0.5) / np.log(GAMMA * LAM)))
    if t > T * 0.3:
        text(d, (x0, 580), f"half the weight is gone after {hl} steps (under a second of play)", 19, INK)
    if t > T * 0.45:
        k = int(min(W1 - W0, 2 + (t - T * 0.45) / (T * 0.45) * (W1 - W0)))
        plot(d, 680, 150, 520, 380, [], bars=win(ADV[LAM]), n=k, label="advantage with λ = 0.95 (what training uses)")
    legend(d)
    return im


def s6(t, T):
    im, d = canvas(); header(d, "Three grades for the same game", "same steps, λ = 0, 0.95 and 1 (each scaled to its own range)")
    for j, lam in enumerate((0.0, LAM, 1.0)):
        if t > j * T * 0.18:
            a = win(ADV[lam]); m = max(abs(a).max(), 1e-6)
            plot(d, 90, 140 + j * 170, 1100, 120, [], bars=a, lo=-m * 1.05, hi=m * 1.05,
                 label=f"λ = {lam:g}" + ["  (one step: steady but trusts the critic)", "  (the blend)", "  (all real rewards: unbiased but noisy)"][j])
    legend(d)
    return im


if __name__ == "__main__":
    render_film([s1, s2, s3, s4, s5, s6], "voice/ppo2", ROOT / "docs" / "video" / "ppo2.mp4", ROOT / "video" / "build" / "ppo2", only)
