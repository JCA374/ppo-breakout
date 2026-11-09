"""Explainer 1: "What PPO does". Scenes match video/ppo1/script.md.

    python video/build_ppo1.py CKPT [--only 2,3]
Output docs/video/ppo1.mp4. Uses a recorded game of CKPT for the visuals and runs/main/stats.csv for the curve.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
from record import load, record_episode, ACTIONS

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--only", default="")
A = ap.parse_args()
only = {int(x) for x in A.only.split(",") if x}
import torch; torch.set_num_threads(2)
agent = load(A.ckpt)
frames_trained = int(Path(A.ckpt).stem.split("_")[1][:-1]) * 1000
EP = record_episode(agent, 11, max_steps=1500)
EPS = [record_episode(agent, 100 + i, max_steps=400, keep_obs=False) for i in range(8)]
CHIP = ["NOOP", "FIRE", "RIGHT", "LEFT"]


def gray(a, s):
    return Image.fromarray(a).resize((s, s), Image.NEAREST).convert("RGB")


def game(i, w=None):
    i = min(i, len(EP["rgb"]) - 1)
    im = Image.fromarray(EP["rgb"][i])
    return im.resize((w, int(w * 210 / 160)), Image.NEAREST) if w else im


def chips(d, x, y, act, probs=None):
    for k, n in enumerate(CHIP):
        on = k == act
        d.rounded_rectangle([x + k * 120, y, x + k * 120 + 108, y + 40], 10, fill=ACC if on else PANEL)
        text(d, (x + k * 120 + 54, y + 20), n, 18, BG if on else INK, bold=True, anchor="mm")
        if probs is not None:
            text(d, (x + k * 120 + 54, y + 58), f"{100 * probs[k]:.0f}%", 18, MUTED, anchor="mm")


def s1(t, T):
    im, d = canvas(); header(d, "The task", "the agent's only input: the screen")
    i = int(t * 15)
    im.paste(game(i, 300), (60, 120))
    text(d, (60, 520), "what you see (210×160 colour)", 18, MUTED)
    st = EP["obs"][min(i, len(EP["obs"]) - 1)]
    a = ease((t - 2) / 2)
    for k in range(4):
        x, y = 460 + (k % 2) * 200, 120 + (k // 2) * 220
        g = gray(st[k], 180)
        if a < 1:
            g = Image.blend(Image.new("RGB", g.size, BG), g, a)
        im.paste(g, (x, y)); text(d, (x, y + 184), ["t−3", "t−2", "t−1", "now"][k], 16, MUTED)
    text(d, (460, 560), "what the agent sees: 4 × 84×84 gray", 18, MUTED)
    if t > T * 0.35:
        text(d, (900, 140), "Each 1/15 s it presses one button:", 20, INK, width=340)
        for k, n in enumerate(CHIP):
            on = k == EP["action"][min(i, len(EP["action"]) - 1)]
            d.rounded_rectangle([900, 190 + k * 56, 1100, 234 + k * 56], 10, fill=ACC if on else PANEL)
            text(d, (1000, 212 + k * 56), n, 18, BG if on else INK, bold=True, anchor="mm")
    if t > T * 0.6:
        text(d, (900, 440), "+1 to +7 points per brick.", 20, GOOD, width=340)
        text(d, (900, 480), "No one tells it what a ball or a paddle is.", 20, WARN, width=340)
    return im


def s2(t, T):
    im, d = canvas(); header(d, "One network, two heads", "actor-critic: Nature CNN, 1.7 million weights")
    i = 200 + int(t * 15)
    st = EP["obs"][min(i, len(EP["obs"]) - 1)]
    p = EP["probs"][min(i, len(EP["probs"]) - 1)]; v = EP["value"][min(i, len(EP["value"]) - 1)]
    for k in range(4):
        im.paste(gray(st[k], 110), (40 + k * 12, 230 + k * 12))
    text(d, (40, 390), "input 4×84×84", 16, MUTED)
    boxes = [("conv 8×8", "32 × 20×20", 210), ("conv 4×4", "64 × 9×9", 360), ("conv 3×3", "64 × 7×7", 510), ("dense", "512", 660)]
    flow = (t * 1.2) % 5
    for k, (n, s, x) in enumerate(boxes):
        lit = int(flow) == k
        d.rounded_rectangle([x, 250, x + 120, 340], 12, fill=ACC if lit else PANEL)
        text(d, (x + 60, 280), n, 17, BG if lit else INK, bold=True, anchor="mm")
        text(d, (x + 60, 310), s, 15, BG if lit else MUTED, anchor="mm")
        d.line([x - 30, 295, x, 295], fill=MUTED, width=3)
    d.line([780, 295, 830, 200], fill=MUTED, width=3); d.line([780, 295, 830, 430], fill=MUTED, width=3)
    a1 = ease((t - T * 0.25) / 1.5)
    if a1 > 0:
        text(d, (840, 120), "ACTOR: chance of each button", 20, WARN, bold=True)
        for k in range(4):
            w = int(300 * p[k] * a1)
            d.rectangle([930, 160 + k * 34, 930 + w, 186 + k * 34], fill=WARN)
            text(d, (840, 162 + k * 34), CHIP[k], 17, INK)
            text(d, (940 + w, 162 + k * 34), f"{100 * p[k]:.0f}%", 16, MUTED)
    a2 = ease((t - T * 0.45) / 1.5)
    if a2 > 0:
        text(d, (840, 400), "CRITIC: reward still to come?", 20, ACC, bold=True)
        text(d, (840, 440), f"{v:+.2f}", 44, ACC, bold=True)
        text(d, (840, 500), "(in clipped points, discounted by 0.99 per step)", 16, MUTED)
    if t > T * 0.8:
        text(d, (60, 620), "The actor decides. The critic judges.", 30, INK, bold=True)
    return im


def s3(t, T):
    im, d = canvas(); header(d, "Collecting a rollout", "8 games side by side × 128 steps = 1,024 moments")
    fill = min(1.0, t / (T * 0.8)); n = int(fill * 128)
    for g in range(8):
        e = EPS[g]; j = min(n, len(e["rgb"]) - 1)
        fr = Image.fromarray(e["rgb"][j]).resize((120, 158), Image.NEAREST)
        im.paste(fr, (40 + (g % 4) * 130, 130 + (g // 4) * 175))
    text(d, (40, 490), f"step {n} of 128", 22, INK, bold=True)
    x0, y0, cw, ch = 590, 140, 5, 30
    for g in range(8):
        e = EPS[g]
        for s in range(n):
            r = e["reward"][s] if s < len(e["reward"]) else 0
            col = GOOD if r > 0 else (60, 70, 95)
            d.rectangle([x0 + s * cw, y0 + g * (ch + 6), x0 + s * cw + cw - 1, y0 + g * (ch + 6) + ch], fill=col)
        text(d, (x0 - 30, y0 + g * (ch + 6) + 5), f"{g + 1}", 16, MUTED)
    text(d, (x0, 450), f"{8 * n:,} / 1,024 moments recorded", 22, INK, bold=True)
    text(d, (x0, 490), "each: the 4 frames · the button · its probability · the critic's guess · the reward", 17, MUTED, width=640)
    d.rectangle([x0, 540, x0 + 18, 558], fill=GOOD); text(d, (x0 + 26, 537), "a brick was hit (reward)", 17, MUTED)
    return im


# advantage scene: real values and real discounted returns of the recorded game
R = np.clip(EP["reward"], 0, 1); V = EP["value"]
G = np.zeros_like(R, dtype=float); acc = 0.0
for k in range(len(R) - 1, -1, -1):
    lost = EP["lives"][k] < (EP["lives"][k - 1] if k else 5)        # a life ended at step k: no future after it
    acc = R[k] + 0.99 * acc * (0 if lost else 1); G[k] = acc
# the 220-step window (after the first 100 steps) where better- and worse-than-expected are most balanced
_cands = [w for w in range(100, max(101, len(R) - 220), 10)]
W0 = min(_cands, key=lambda w: abs((G[w:w + 220] > V[w:w + 220]).mean() - 0.5)) if _cands else 0
W1 = min(len(R), W0 + 220)


def s4(t, T):
    im, d = canvas(); header(d, "Was it better than expected?", "advantage = what actually came − what the critic expected")
    x0, y0, w, h = 90, 140, 1100, 380
    lo, hi = min(G[W0:W1].min(), V[W0:W1].min()) - 0.1, max(G[W0:W1].max(), V[W0:W1].max()) + 0.1
    X = lambda k: x0 + (k - W0) * w / (W1 - W0); Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    m = W0 + int((W1 - W0) * min(1, t / (T * 0.7)))
    for k in range(W0, m):
        col = GOOD if G[k] > V[k] else BAD
        d.line([X(k), Y(V[k]), X(k), Y(G[k])], fill=col, width=4)
    pts_v = [(X(k), Y(V[k])) for k in range(W0, m)]; pts_g = [(X(k), Y(G[k])) for k in range(W0, m)]
    if len(pts_v) > 1:
        d.line(pts_v, fill=ACC, width=3); d.line(pts_g, fill=INK, width=3)
    for k in range(W0, m):
        if R[k] > 0:
            d.ellipse([X(k) - 5, y0 + h + 8, X(k) + 5, y0 + h + 18], fill=GOOD)
    text(d, (x0, 560), "blue: critic's guess   white: the reward that really followed (discounted)   ● brick hit", 18, MUTED)
    text(d, (x0, 600), "green gap: better than expected → do it more    red gap: worse → do it less", 20, INK)
    text(d, (x0, 640), f"{W1 - W0} real steps of one game, model at {frames_trained / 1e6:.1f} M frames", 16, MUTED)
    return im


def s5(t, T):
    im, d = canvas(); header(d, "The careful update", "PPO clips the change: the new/old probability ratio stays within 0.9 – 1.1")
    p_old = np.array([0.20, 0.10, 0.45, 0.25]); grow = ease(t / (T * 0.45))
    want = p_old[2] * (1 + 0.6 * grow); clip = p_old[2] * 1.1
    pn = min(want, clip)
    text(d, (80, 130), "one moment with a positive advantage: RIGHT", 22, INK, bold=True)
    for k in range(4):
        x = 100 + k * 130
        hh = int(320 * p_old[k]); d.rectangle([x, 520 - hh, x + 45, 520], fill=(80, 90, 120))
        val = pn if k == 2 else p_old[k] - (pn - p_old[2]) / 3
        hn = int(320 * val); d.rectangle([x + 50, 520 - hn, x + 95, 520], fill=WARN if k == 2 else ACC)
        text(d, (x + 48, 545), CHIP[k], 17, INK, anchor="mm")
    yc = 520 - int(320 * clip)
    d.line([350, yc, 480, yc], fill=BAD, width=3); text(d, (488, yc - 12), "clip: ratio 1.1", 16, BAD)
    if want > clip:
        yw = 520 - int(320 * want); d.line([360, yw, 470, yw], fill=(120, 60, 60), width=2)
        text(d, (488, yw - 12), "what the gradient wanted", 15, MUTED)
    text(d, (100, 580), "grey: old policy   colour: new policy", 16, MUTED)
    ep = ease((t - T * 0.45) / (T * 0.45)); done = int(ep * 16)
    text(d, (760, 130), "4 epochs × 4 minibatches of 256", 22, INK, bold=True)
    for e in range(4):
        text(d, (760, 190 + e * 70), f"epoch {e + 1}", 18, MUTED)
        for b in range(4):
            on = e * 4 + b < done
            d.rounded_rectangle([880 + b * 80, 180 + e * 70, 950 + b * 80, 225 + e * 70], 8, fill=GOOD if on else PANEL)
    if done >= 16:
        text(d, (760, 480), "then the batch is thrown away", 22, WARN, bold=True)
    return im


ROWS = [r for r in csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")) if r["ep_return_mean"]]


def s6(t, T):
    im, d = canvas(); header(d, "Round after round", "this run's real training curve")
    x0, y0, w, h = 110, 130, 1080, 420
    fx = [int(r["frames"]) for r in ROWS]; fy = [float(r["ep_return_mean"]) for r in ROWS]
    # smooth over 20 iterations
    sy = [np.mean(fy[max(0, k - 20):k + 1]) for k in range(len(fy))]
    ymax = max(50, max(sy) * 1.1) if sy else 50; xmax = 40e6
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    for v in np.linspace(0, ymax, 5):
        yy = y0 + h - v / ymax * h; d.line([x0, yy, x0 + w, yy], fill=(36, 40, 48)); text(d, (x0 - 60, yy - 10), f"{v:.0f}", 15, MUTED)
    for f in range(0, 41, 10):
        xx = x0 + f * 1e6 / xmax * w; text(d, (xx - 8, y0 + h + 8), f"{f}M", 15, MUTED)
    k = int(len(sy) * min(1, t / (T * 0.8)))
    pts = [(x0 + fx[j] / xmax * w, y0 + h - sy[j] / ymax * h) for j in range(k)]
    if len(pts) > 1:
        d.line(pts, fill=ACC, width=3)
    it = int(ROWS[k - 1]["iteration"]) if k else 0
    text(d, (x0, 600), f"round {it:,} of 9,765 · score per game (mean of training games, smoothed)", 20, INK)
    text(d, (x0, 640), "40 M frames ≈ 185 hours of play at 60 fps", 18, MUTED)
    return im


if __name__ == "__main__":
    render_film([s1, s2, s3, s4, s5, s6], "voice/ppo1", ROOT / "docs" / "video" / "ppo1.mp4", ROOT / "video" / "build" / "ppo1", only)
