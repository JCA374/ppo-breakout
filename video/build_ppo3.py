"""Explainer 3: "The clipped objective". Scenes match video/ppo3/script.md.

    python video/build_ppo3.py [--only 3]
Output docs/video/ppo3.mp4. Scene 5 draws this run's real approx-KL and clip fraction from runs/main/stats.csv.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
A = ap.parse_args()
only = {int(x) for x in A.only.split(",") if x}
EPS_ = 0.1
ROWS = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))
FACTS = json.load(open(ROOT / "video" / "ppo3" / "facts.json"))


def axes(d, x0, y0, w, h, xr, yr, xl="", yl="", ticks_x=(), ticks_y=()):
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w
    Y = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    for v in ticks_x:
        d.line([X(v), y0 + h, X(v), y0 + h + 6], fill=MUTED); text(d, (X(v), y0 + h + 18), f"{v:g}", 15, MUTED, anchor="mm")
    for v in ticks_y:
        d.line([x0 - 6, Y(v), x0, Y(v)], fill=MUTED); text(d, (x0 - 10, Y(v)), f"{v:g}", 15, MUTED, anchor="rm")
    text(d, (x0 + w / 2, y0 + h + 42), xl, 17, MUTED, anchor="mm")
    text(d, (x0, y0 - 26), yl, 17, MUTED)
    return X, Y


def s1(t, T):
    im, d = canvas(); header(d, "Why be careful", "one step too far, and the data no longer describes the player")
    # an action's probability jumping in one unclipped update vs small clipped steps
    text(d, (80, 140), "probability of RIGHT in one situation", 20, INK, bold=True)
    steps = 8; k = min(steps, int(t / (T * 0.8) * steps) + 1)
    safe = [0.30 * (1.1 ** i) for i in range(steps)]
    wild = [0.30, 0.95, 0.02, 0.6, 0.05, 0.9, 0.1, 0.4]
    for j, (lab, seq, col, y) in enumerate([("unclipped, big steps", wild, BAD, 200), ("PPO: at most ±10% per update", safe, GOOD, 430)]):
        text(d, (80, y), lab, 18, col, bold=True)
        for i in range(k):
            h = int(160 * min(1, seq[i]))
            d.rectangle([100 + i * 130, y + 190 - h, 160 + i * 130, y + 190], fill=col)
            text(d, (130 + i * 130, y + 205), f"{seq[i]:.2f}", 14, MUTED, anchor="mm")
    text(d, (80, 680), "illustration (not data): what a collapse looks like vs small, steady steps", 15, MUTED)
    return im


def s2(t, T):
    im, d = canvas(); header(d, "The ratio", "ρ = π_new(action | screen) / π_old(action | screen)")
    old = 0.40; a = ease(t / (T * 0.7)); new = old * (1 + 0.2 * np.sin(a * np.pi * 1.5))
    for i, (lab, p, col) in enumerate([("old policy", old, (90, 100, 130)), ("new policy", new, WARN)]):
        x = 200 + i * 220; h = int(400 * p)
        d.rectangle([x, 600 - h, x + 140, 600], fill=col)
        text(d, (x + 70, 620), lab, 18, INK, anchor="mm"); text(d, (x + 70, 590 - h), f"{p:.3f}", 18, INK, anchor="mm")
    r = new / old
    text(d, (720, 280), "ratio", 26, MUTED)
    text(d, (720, 320), f"{r:.2f}", 90, WARN if abs(r - 1) > EPS_ else INK, bold=True)
    text(d, (720, 450), "1.00 = unchanged", 20, MUTED)
    text(d, (720, 485), "1.20 = 20% more likely", 20, MUTED)
    text(d, (720, 520), "0.80 = 20% less likely", 20, MUTED)
    return im


def clip_obj(r, adv):
    return np.minimum(r * adv, np.clip(r, 1 - EPS_, 1 + EPS_) * adv)


def s3(t, T):
    im, d = canvas(); header(d, "The clip", "objective = min( ρ·A ,  clip(ρ, 0.9, 1.1)·A )")
    rs = np.linspace(0.6, 1.4, 200)
    for j, adv in enumerate([1.0, -1.0]):
        x0 = 90 + j * 600
        show = ease((t - j * T * 0.4) / (T * 0.3))
        if show <= 0:
            continue
        X, Y = axes(d, x0, 170, 500, 380, (0.6, 1.4), (-1.5, 1.5), "ratio ρ", f"A {'> 0: a good action' if adv > 0 else '< 0: a bad action'}",
                    ticks_x=(0.6, 0.9, 1.0, 1.1, 1.4), ticks_y=(-1, 0, 1))
        d.rectangle([X(0.9), 170, X(1.1), 550], fill=(26, 32, 44))
        un = [(X(r), Y(r * adv)) for r in rs]
        d.line(un, fill=(80, 84, 96), width=2)
        n = max(2, int(len(rs) * show))
        pts = [(X(r), Y(clip_obj(r, adv))) for r in rs[:n]]
        d.line(pts, fill=GOOD if adv > 0 else BAD, width=5)
        text(d, (x0 + 10, 600), "grey: without the clip", 16, MUTED)
        text(d, (x0 + 10, 625), "flat part: no more gain, so no more push", 16, INK)
    return im


def s4(t, T):
    im, d = canvas(); header(d, "Why flat means safe", "the gradient of the objective with respect to ρ")
    X, Y = axes(d, 140, 170, 1000, 360, (0.6, 1.4), (-1.3, 1.3), "ratio ρ", "push on the action's probability (A > 0)",
                ticks_x=(0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4), ticks_y=(-1, 0, 1))
    d.line([X(0.6), Y(0), X(1.4), Y(0)], fill=(60, 64, 76))
    d.line([X(0.6), Y(1), X(1.1), Y(1)], fill=GOOD, width=5); d.line([X(1.1), Y(0), X(1.4), Y(0)], fill=GOOD, width=5)
    # a dot = one sample's ratio moving up over the 16 gradient steps of one update
    r = 1.0 + 0.16 * ease(t / (T * 0.7))
    g = 1.0 if r < 1.1 else 0.0
    d.ellipse([X(r) - 10, Y(g) - 10, X(r) + 10, Y(g) + 10], fill=WARN)
    text(d, (X(r), Y(g) - 30), f"ρ = {r:.2f}", 18, WARN, anchor="mm")
    text(d, (140, 600), "Below 1.1 the sample pushes its action up. Past 1.1 it stops pushing.", 20, INK)
    text(d, (140, 635), "If an update made a good action LESS likely (ρ < 1), the min keeps the full penalty: mistakes are never hidden.", 18, MUTED)
    return im


def s5(t, T):
    im, d = canvas(); header(d, "This run's numbers", "approx KL and clip fraction of every update (MLflow), smoothed")
    xs = np.array([int(r["frames"]) for r in ROWS]) / 1e6
    kl = np.array([float(r["approx_kl"]) for r in ROWS]); cf = np.array([float(r["clipfrac"]) for r in ROWS])
    sm = lambda a: np.convolve(a, np.ones(25) / 25, mode="same")
    n = max(2, int(len(xs) * ease(t / (T * 0.6))))
    for j, (arr, lab, col, top) in enumerate([(sm(kl), "approx KL", ACC, max(0.03, float(np.percentile(sm(kl), 99)) * 1.15)),
                                              (sm(cf), "clip fraction", WARN, max(0.3, float(np.percentile(sm(cf), 99)) * 1.15))]):
        x0 = 90 + j * 600
        X, Y = axes(d, x0, 170, 500, 340, (0, 10), (0, top), "million frames", lab, ticks_x=(0, 2, 4, 6, 8, 10),
                    ticks_y=tuple(np.round(np.linspace(0, top, 4), 3)))
        d.line([(X(xs[i]), Y(min(top, arr[i]))) for i in range(n)], fill=col, width=3)
    y = 580
    for line in FACTS["lines"]:
        if t > T * 0.35:
            text(d, (90, y), line, 19, INK); y += 32
    return im


def s6(t, T):
    im, d = canvas(); header(d, "Small, safe steps add up", "")
    y = 170
    for i, line in enumerate(FACTS["summary"]):
        if t > i * T / (len(FACTS["summary"]) + 1):
            y = text(d, (90, y), line, 26, INK if i else WARN, bold=(i == 0), width=1100) + 18
    return im


if __name__ == "__main__":
    render_film([s1, s2, s3, s4, s5, s6], "voice/ppo3", ROOT / "docs" / "video" / "ppo3.mp4", ROOT / "video" / "build" / "ppo3", only)
