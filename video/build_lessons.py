"""Lesson videos for the Lessons tab: how the project is set up and its harder parts.
Scenes match video/<lesson>/script.md one to one.

    python video/build_lessons.py lesson1|lesson2|lesson4
Output docs/video/<lesson>.mp4.
"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

LESSON = sys.argv[1]


def box(d, x, y, w, h, title, sub="", col=PANEL, on=True, tcol=INK):
    if not on:
        return
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=col)
    text(d, (x + w / 2, y + (h / 2 - 12 if sub else h / 2)), title, 21, tcol, bold=True, anchor="mm")
    if sub:
        for i, line in enumerate(sub.split("\n")):
            text(d, (x + w / 2, y + h / 2 + 14 + 20 * i), line, 15, MUTED if col == PANEL else tcol, anchor="mm")


def arrow(d, a, b, col=MUTED, w=3, on=True):
    if not on:
        return
    d.line([a, b], fill=col, width=w)
    v = np.array(b, float) - np.array(a, float); v /= (np.linalg.norm(v) + 1e-9); n = np.array([-v[1], v[0]])
    p = np.array(b, float)
    d.polygon([tuple(p), tuple(p - 14 * v + 7 * n), tuple(p - 14 * v - 7 * n)], fill=col)


def appear(t, T, k, n, start=0.05, end=0.75):
    return t >= T * (start + (end - start) * k / max(1, n))


# ------------------------------------------------------------------ lesson 1: how the project is set up
def l1_s1(t, T):
    im, d = canvas(); header(d, "The pieces", "four parts that only share files on disk")
    box(d, 470, 280, 340, 150, "files on disk", "checkpoints/  mlruns/  runs/*.csv\nresults/  docs/", (36, 40, 52))
    parts = [("Trainer", "plays + learns", 80, 140), ("Evaluator", "tests on new games", 860, 140),
             ("Publisher", "builds this site", 80, 470), ("Studio", "videos + podcasts", 860, 470)]
    for k, (n, s, x, y) in enumerate(parts):
        on = appear(t, T, k, 4)
        box(d, x, y, 340, 110, n, s, ACC if on else PANEL, on, BG)
        if on:
            arrow(d, (x + 170, y + (110 if y < 300 else 0)), (640 + (x - 470) * 0.3, 280 if y < 300 else 430), MUTED)
    return im


def l1_s2(t, T):
    im, d = canvas(); header(d, "The trainer", "ppo/train.py, kept alive by supervise.sh")
    box(d, 60, 200, 300, 160, "envpool", "8 Breakout games\nC++ threads", (36, 40, 52))
    box(d, 490, 200, 300, 160, "PyTorch", "Nature CNN on the CPU\nPPO update", (36, 40, 52))
    arrow(d, (360, 250), (490, 250), ACC); text(d, (372, 215), "screens, rewards", 15, MUTED)
    arrow(d, (490, 310), (360, 310), WARN); text(d, (392, 318), "buttons", 15, MUTED)
    outs = [("MLflow  (mlruns/)", "every update: losses, KL, clip, entropy, speed", 160),
            ("runs/main/stats.csv", "the same numbers as a plain table", 300),
            ("checkpoints/main/", "every 500,000 frames + latest.pt", 440)]
    for k, (n, s, y) in enumerate(outs):
        if appear(t, T, k, 3, 0.25, 0.8):
            arrow(d, (790, 280), (900, y + 35), MUTED)
            d.rounded_rectangle([900, y, 1240, y + 70], 10, fill=PANEL)
            text(d, (915, y + 10), n, 19, INK, bold=True); text(d, (915, y + 40), s, 14, MUTED)
    rows = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))
    text(d, (60, 600), f"right now: iteration {int(rows[-1]['iteration']):,} · {int(rows[-1]['frames']) / 1e6:.1f} M frames · "
                       f"{float(rows[-1]['sps']):.0f} agent steps/s", 20, INK)
    return im


EVENTS = [json.loads(l) for l in open(ROOT / "runs" / "main" / "events.jsonl")]


def l1_s3(t, T):
    im, d = canvas(); header(d, "Surviving a crash", "supervise.sh restarts train.py from checkpoints/main/latest.pt")
    x0, x1, y = 80, 1200, 230
    d.line([x0, y, x1, y], fill=MUTED, width=4)
    for k in range(9):
        xx = x0 + k * (x1 - x0) / 8; d.line([xx, y - 14, xx, y + 14], fill=GOOD, width=4)
        text(d, (xx, y + 34), f"{k * 0.5:g} M", 15, MUTED, anchor="mm")
    text(d, (x0, y - 60), "a checkpoint every 500,000 frames", 18, GOOD)
    if t > T * 0.2:
        cx = x0 + 6.6 * (x1 - x0) / 16
        text(d, (cx, y), "✕", 40, BAD, bold=True, anchor="mm"); text(d, (cx, y - 50), "crash", 18, BAD, anchor="mm")
    if t > T * 0.35:
        xs = x0 + 6 * (x1 - x0) / 16
        d.arc([xs - 2, y - 60, x0 + 6.6 * (x1 - x0) / 16 + 2, y + 60], 0, 180, fill=WARN, width=3)
        text(d, (xs, y + 80), "resume here", 17, WARN, anchor="mm")
    if t > T * 0.5:
        text(d, (80, 380), "latest.pt holds:", 22, INK, bold=True)
        for k, s in enumerate(["the network's weights", "the optimizer's state (Adam moments)", "iteration → the learning-rate schedule continues",
                               "random number states", "the MLflow run id → curves continue in the same run"]):
            if t > T * (0.5 + 0.08 * k):
                text(d, (110, 420 + 36 * k), "• " + s, 19, INK)
    ov = [e for e in EVENTS if e["kind"] in ("resume", "override")]
    if ov and t > T * 0.85:
        text(d, (700, 380), f"in this run: {len(ov)} resume/override events,", 17, MUTED)
        text(d, (700, 405), "e.g. the 10 M → 40 M frames change at 1.5 M", 17, MUTED)
    return im


def l1_s4(t, T):
    im, d = canvas(); header(d, "Testing honestly", "ppo/evaluate.py: 10 full games, 5 lives, real score, seeds 12345–12354")
    res = sorted(glob.glob(str(ROOT / "results" / "goal_f*.json")))
    r = json.load(open(res[-1])) if res else None
    if r:
        for k, s in enumerate(r["scores"]):
            if appear(t, T, k, 10, 0.05, 0.6):
                x = 80 + (k % 5) * 230; y = 150 + (k // 5) * 150
                d.rounded_rectangle([x, y, x + 200, y + 120], 12, fill=PANEL)
                text(d, (x + 100, y + 40), f"game {k + 1}", 16, MUTED, anchor="mm")
                text(d, (x + 100, y + 80), f"{s:.0f}", 34, INK, bold=True, anchor="mm")
        if t > T * 0.65:
            text(d, (80, 480), f"{r['label']}: mean {r['score']:.1f}  (95% interval {r['ci'][0]:.0f}–{r['ci'][1]:.0f})", 26, WARN, bold=True)
            text(d, (80, 530), "goal: a mean of 300 over 10 such games", 22, MUTED)
    text(d, (80, 620), "Training clips rewards and ends an episode at each lost life; the test does neither.", 18, MUTED)
    return im


def l1_s5(t, T):
    im, d = canvas(); header(d, "The website", "publish_loop.sh: every 15 minutes")
    steps = ["export MLflow → JSON", "goal checks, replays,\nsaliency (low priority)", "git commit + push", "GitHub Pages\nbuilds the site"]
    for k, s_ in enumerate(steps):
        on = appear(t, T, k, 4, 0.05, 0.6)
        x = 60 + k * 300
        lines = s_.split("\n")
        box(d, x, 220, 260, 110, lines[0], "\n".join(lines[1:]), ACC if on else PANEL, True, BG if on else MUTED)
        if k < 3:
            arrow(d, (x + 262, 275), (x + 298, 275), MUTED)
    if t > T * 0.65:
        d.line([1180, 330, 1180, 420, 190, 420], fill=WARN, width=3); arrow(d, (190, 420), (190, 332), WARN)
        text(d, (560, 432), "wait 15 minutes, then again", 18, WARN)
    if t > T * 0.8:
        text(d, (60, 520), "the same loop is a watchdog: if the trainer's supervisor is gone", 20, INK)
        text(d, (60, 552), "and the run isn't finished, it starts it again", 20, INK)
    return im


# ------------------------------------------------------------------ lesson 2: from screen to input
def ball_visible(f):
    reg = f[100:188, 8:152]
    return int(((reg[..., 0] > 150) & (reg[..., 1] < 100)).sum()) > 0


def raw_frames():
    """Raw frames with the ball in play: FIRE to launch, then follow the ball with the paddle (RAM 99 = ball x, 72 = paddle x)."""
    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)
    env = gym.make("BreakoutNoFrameskip-v4", repeat_action_probability=0.0)
    env.reset(seed=3); fr = []
    ale = env.unwrapped.ale
    for k in range(700):
        ram = ale.getRAM()
        bx, px = int(ram[99]), int(ram[72])
        a = 1 if k % 40 == 0 else (2 if bx > px + 8 else 3 if bx < px + 4 else 0)
        o, *_ = env.step(a)
        fr.append(o)
    env.close()
    return fr


if LESSON == "lesson2":
    import cv2
    RAW = raw_frames()
    vis = [ball_visible(f) for f in RAW]
    base = next(k for k in range(100, len(RAW) - 80) if all(vis[k:k + 80]))


def gray84(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA)


def l2_s1(t, T):
    im, d = canvas(); header(d, "The raw screen", "210 × 160 pixels, colour, 60 frames a second")
    k = int(t * 60) % 180
    im.paste(Image.fromarray(RAW[k]).resize((320, 420), Image.NEAREST), (120, 130))
    text(d, (520, 220), f"frame {k}", 40, INK, bold=True)
    text(d, (520, 290), "100,800 numbers per frame", 24, MUTED)
    text(d, (520, 330), "× 60 per second", 24, MUTED)
    return im


def l2_s2(t, T):
    im, d = canvas(); header(d, "Skip and merge", "one decision per 4 frames; the last two merged by the brighter pixel")
    fr = [RAW[base + i] for i in range(4)]
    for i, f in enumerate(fr):
        if appear(t, T, i, 4, 0.0, 0.35):
            im.paste(Image.fromarray(f).resize((200, 262), Image.NEAREST), (60 + i * 220, 130))
            text(d, (160 + i * 220, 410), f"frame {base + i}  · same button", 15, MUTED, anchor="mm")
    if t > T * 0.5:
        m = np.maximum(fr[2], fr[3])
        arrow(d, (600, 440), (1010, 300), WARN); arrow(d, (820, 440), (1010, 320), WARN)
        im.paste(Image.fromarray(m).resize((200, 262), Image.NEAREST), (1030, 130))
        text(d, (1130, 410), "max of the last two", 15, WARN, anchor="mm")
    if t > T * 0.7:
        text(d, (60, 480), "rewards of the 4 frames are added up", 20, INK)
        text(d, (60, 520), "the Atari draws some objects only on every other frame: the max keeps the ball visible", 18, MUTED)
    return im


def l2_s3(t, T):
    im, d = canvas(); header(d, "Shrink to gray", "210×160 colour → 84×84 gray (area averaging)")
    m = np.maximum(RAW[base + 2], RAW[base + 3])
    im.paste(Image.fromarray(m).resize((300, 394), Image.NEAREST), (80, 140))
    if t > T * 0.3:
        arrow(d, (400, 330), (520, 330), ACC)
        g = gray84(m)
        im.paste(Image.fromarray(g).resize((336, 336), Image.NEAREST).convert("RGB"), (540, 170))
        text(d, (540, 520), "84 × 84 = 7,056 numbers (14× fewer)", 18, MUTED)
    if t > T * 0.6:
        text(d, (920, 220), "the ball: one or two pixels", 20, WARN, width=320)
        text(d, (920, 300), "the brick colours become shades of gray", 18, MUTED, width=320)
    return im


def l2_s4(t, T):
    im, d = canvas(); header(d, "Stack four", "input = the last 4 processed frames: 4 × 84 × 84")
    k0 = base + 4 * int(t * 4) % 60
    st = [gray84(np.maximum(RAW[k0 + 4 * i + 2], RAW[k0 + 4 * i + 3])) for i in range(4)]
    for i, g in enumerate(st):
        im.paste(Image.fromarray(g).resize((220, 220), Image.NEAREST).convert("RGB"), (60 + i * 240, 140))
        text(d, (170 + i * 240, 375), ["t−3", "t−2", "t−1", "now"][i], 16, MUTED, anchor="mm")
    if t > T * 0.45:
        diff = np.zeros((84, 84, 3), np.uint8)
        for i, g in enumerate(st):
            mv = np.abs(g.astype(int) - st[0].astype(int)) > 25
            diff[mv] = [(80, 80, 255), (80, 200, 255), (255, 200, 80), (255, 90, 60)][i]
        im.paste(Image.fromarray(diff).resize((220, 220), Image.NEAREST), (1000, 420))
        text(d, (60, 450), "what changed between the frames, coloured old → new:", 20, INK)
        text(d, (60, 490), "the moving dot gives direction and speed", 20, WARN)
    return im


def l2_s5(t, T):
    im, d = canvas(); header(d, "Two training shortcuts", "used while learning, never in the test")
    text(d, (80, 140), "1. a lost life ends the training episode", 24, INK, bold=True)
    for k in range(5):
        x = 100 + k * 110
        d.rounded_rectangle([x, 190, x + 90, 240], 8, fill=ACC if k < 3 else PANEL)
        text(d, (x + 45, 215), f"life {k + 1}", 15, BG if k < 3 else MUTED, anchor="mm")
    text(d, (680, 200), "training: each life = one episode", 18, ACC)
    text(d, (680, 228), "test: all 5 lives = one game", 18, MUTED)
    if t > T * 0.45:
        text(d, (80, 330), "2. every reward is clipped to +1", 24, INK, bold=True)
        pts = [("blue", 1, (66, 72, 200)), ("green", 1, (72, 160, 72)), ("yellow", 4, (180, 180, 50)), ("orange", 4, (200, 120, 40)),
               ("red", 7, (200, 72, 72))]
        for k, (n, p, c) in enumerate(pts):
            x = 100 + k * 200
            d.rectangle([x, 380, x + 140, 410], fill=c)
            text(d, (x + 70, 440), f"real {p}", 18, INK, anchor="mm")
            text(d, (x + 70, 472), "train +1", 18, WARN, anchor="mm")
    return im


# ------------------------------------------------------------------ lesson 4: speed and budgets
def bars(d, x0, y0, w, h, items, vmax, unit="", slots=3):
    bw = w / slots
    for k, (lab, v, col) in enumerate(items):
        hh = h * v / vmax
        d.rectangle([x0 + k * bw + 20, y0 + h - hh, x0 + (k + 1) * bw - 20, y0 + h], fill=col)
        text(d, (x0 + (k + 0.5) * bw, y0 + h - hh - 20), f"{v:,.0f}{unit}", 20, INK, bold=True, anchor="mm")
        for i, line in enumerate(lab.split("\n")):
            text(d, (x0 + (k + 0.5) * bw, y0 + h + 22 + 22 * i), line, 16, MUTED, anchor="mm")


def l4_s1(t, T):
    im, d = canvas(); header(d, "Measure first", "agent steps per second, measured on this laptop")
    eb = json.load(open(ROOT / "bench" / "env_bench.json")); tb = json.load(open(ROOT / "bench" / "thread_bench.json"))
    items = [("envpool\n(simulator only)", eb["envpool_n8"], GOOD), ("gymnasium\n(simulator only)", eb["gym_async_n8"], WARN),
             ("full PPO loop", tb["agent_steps_per_sec"]["torch_threads_4"], ACC)]
    n = sum(appear(t, T, k, 3, 0.05, 0.6) for k in range(3))
    bars(d, 150, 150, 900, 400, items[:n], 4200)
    return im


def l4_s2(t, T):
    im, d = canvas(); header(d, "Where the time goes", "one PPO iteration, recent average of this run")
    rows = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))[-50:]
    tot = np.mean([1024 / float(r["sps"]) for r in rows])
    tb = json.load(open(ROOT / "bench" / "thread_bench.json"))
    ro, up = tb["rollout_s"]["4"], tb["update_s"]["4"]
    W = 1000; x0 = 140; y0 = 260
    a = ease(t / (T * 0.5))
    d.rectangle([x0, y0, x0 + W * ro / (ro + up) * a, y0 + 90], fill=GOOD)
    if a >= 1:
        d.rectangle([x0 + W * ro / (ro + up), y0, x0 + W, y0 + 90], fill=ACC)
        text(d, (x0 + W * ro / (ro + up) / 2, y0 + 45), f"play {ro:.1f} s", 20, BG, bold=True, anchor="mm")
        text(d, (x0 + W * (ro + up / 2) / (ro + up), y0 + 45), f"update {up:.1f} s (4 epochs × 4 minibatches)", 20, BG, bold=True, anchor="mm")
    text(d, (140, 420), f"= {ro + up:.1f} s per 1,024 agent steps (benchmark); recent iterations average {tot:.1f} s", 20, INK)
    text(d, (140, 460), "the backward pass through the convolutions is the real cost", 18, MUTED)
    return im


def l4_s3(t, T):
    im, d = canvas(); header(d, "More threads, slower", "PyTorch threads vs agent steps per second")
    tb = json.load(open(ROOT / "bench" / "thread_bench.json"))["agent_steps_per_sec"]
    items = [("4 threads", tb["torch_threads_4"], GOOD), ("6 threads", tb["torch_threads_6"], ACC), ("8 threads", tb["torch_threads_8"], BAD)]
    n = sum(appear(t, T, k, 3, 0.05, 0.5) for k in range(3))
    bars(d, 200, 150, 800, 380, items[:n], 320)
    if t > T * 0.6:
        text(d, (200, 620), "4 physical cores (8 with hyperthreading), shared with envpool's game threads", 19, INK)
    return im


def l4_s4(t, T):
    im, d = canvas(); header(d, "The budget's units", "CleanRL's 10 M = agent steps; 1 step = 4 frames")
    text(d, (100, 160), "the plan", 22, MUTED)
    d.rectangle([260, 150, 260 + 250, 200], fill=BAD); text(d, (530, 160), "10 M frames", 22, INK)
    if t > T * 0.3:
        text(d, (100, 240), "the reference", 22, MUTED)
        d.rectangle([260, 230, 260 + 1000 * ease((t - T * 0.3) / 2), 280], fill=GOOD); text(d, (280, 290), "40 M frames = 10 M agent steps", 20, INK)
    if t > T * 0.55:
        x0, y0, w, h = 100, 400, 1100, 200
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
        text(d, (x0, y0 - 28), "learning rate schedule", 17, MUTED)
        X = lambda f: x0 + f / 40e6 * w; Y = lambda lr: y0 + h - lr / 2.6e-4 * h
        d.line([X(0), Y(2.5e-4), X(10e6), Y(0)], fill=BAD, width=2)
        d.line([X(0), Y(2.5e-4), X(1.5e6), Y(2.5e-4 * (1 - 367 / 2441))], fill=INK, width=3)
        d.line([X(1.5e6), Y(2.5e-4 * (1 - 367 / 9765)), X(40e6), Y(0)], fill=GOOD, width=3)
        text(d, (X(10e6) + 10, Y(0) - 30), "old: 0 at 10 M", 15, BAD); text(d, (X(1.5e6) + 10, Y(2.4e-4) - 2), "stretched at 1.5 M", 15, GOOD)
    return im


def l4_s5(t, T):
    im, d = canvas(); header(d, "What it costs", "")
    rows = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))
    sps = np.mean([float(r["sps"]) for r in rows[-200:]])
    hours = 40e6 / 4 / sps / 3600
    text(d, (100, 200), f"{sps:.0f} agent steps per second (recent average)", 30, INK, bold=True)
    if t > T * 0.25:
        text(d, (100, 270), f"40 M frames ÷ 4 ÷ {sps:.0f} ≈ {hours:.1f} hours", 30, WARN, bold=True)
    if t > T * 0.5:
        text(d, (100, 360), "fits in one night, inside the 20-hour limit", 24, GOOD)
        text(d, (100, 410), "→ no rewrite in C++, Rust or numba was needed", 24, MUTED)
    return im


# ------------------------------------------------------------------ lesson 3: reading PPO's health signals
if LESSON == "lesson3":
    ROWS3 = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))
    F3 = np.array([int(r["frames"]) for r in ROWS3]) / 1e6


def curve(key, smooth=25):
    a = np.array([float(r[key]) if r[key] != "" else np.nan for r in ROWS3])
    if np.isnan(a).any():
        idx = ~np.isnan(a); a = np.interp(np.arange(len(a)), np.where(idx)[0], a[idx])
    pad = np.pad(a, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
    return np.convolve(pad, np.ones(smooth) / smooth, mode="valid")


def chart(d, t, T, key, label, col, lo=None, hi=None, refs=(), x0=110, y0=150, w=1060, h=380):
    a = curve(key); n = max(2, int(len(a) * ease(t / (T * 0.7))))
    lo = lo if lo is not None else float(np.percentile(a, 0.5)) * 0.9
    hi = hi if hi is not None else float(np.percentile(a, 99.5)) * 1.1
    X = lambda f: x0 + f / max(F3[-1], 1) * w; Y = lambda v: y0 + h - (min(max(v, lo), hi) - lo) / (hi - lo) * h
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
    for v in np.linspace(lo, hi, 5):
        d.line([x0, Y(v), x0 + w, Y(v)], fill=(34, 38, 46)); text(d, (x0 - 10, Y(v)), f"{v:.3g}", 14, MUTED, anchor="rm")
    for f in range(0, int(F3[-1]) + 1, 5):
        text(d, (X(f), y0 + h + 16), f"{f}M", 14, MUTED, anchor="mm")
    for v, lab, c in refs:
        d.line([x0, Y(v), x0 + w, Y(v)], fill=c, width=2); text(d, (x0 + w - 6, Y(v) - 20), lab, 15, c, anchor="rm")
    d.line([(X(F3[k]), Y(a[k])) for k in range(n)], fill=col, width=3)
    text(d, (x0, y0 - 28), label + " (smoothed over 25 rounds; x = million frames)", 17, MUTED)


def l3_s1(t, T):
    im, d = canvas(); header(d, "Why look inside", "the score is slow and noisy: training games, mean per round")
    chart(d, t, T, "ep_return_mean", "score of training games", INK, lo=0)
    return im


def l3_s2(t, T):
    im, d = canvas(); header(d, "Entropy", "how random the policy still is; ln 4 ≈ 1.39 = pure chance over 4 buttons")
    chart(d, t, T, "entropy", "policy entropy (nats)", GOOD, lo=0, hi=1.45, refs=[(np.log(4), "pure chance", MUTED), (0.1, "collapsed", BAD)])
    return im


def l3_s3(t, T):
    im, d = canvas(); header(d, "KL divergence", "how much the policy changed in one round of updates")
    chart(d, t, T, "approx_kl", "approx KL (old ‖ new)", ACC, lo=0)
    return im


def l3_s4(t, T):
    im, d = canvas(); header(d, "Clip fraction", "share of samples whose ratio left 0.9 – 1.1")
    chart(d, t, T, "clipfrac", "clip fraction", WARN, lo=0, hi=0.5, refs=[(0.5, "half: always fighting the clip", BAD)])
    return im


def l3_s5(t, T):
    im, d = canvas(); header(d, "The critic", "explained variance: 1 = predicts the returns perfectly, 0 = useless")
    chart(d, t, T, "explained_variance", "explained variance", ACC, lo=0, hi=1.05, refs=[(1.0, "perfect", MUTED)])
    return im


def l3_s6(t, T):
    im, d = canvas(); header(d, "A flat start is not a bug", "first 4 M frames: score vs entropy")
    m = F3 <= 4
    sc = curve("ep_return_mean")[m]; en = curve("entropy")[m]; fx = F3[m]
    for j, (a, lab, col, hi) in enumerate([(sc, "score", INK, max(5, sc.max() * 1.1)), (en, "entropy", GOOD, 1.45)]):
        x0, y0, w, h = 110 + j * 590, 170, 500, 360
        n = max(2, int(len(a) * ease(t / (T * 0.6))))
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(50, 54, 64))
        d.line([(x0 + fx[k] / 4 * w, y0 + h - a[k] / hi * h) for k in range(n)], fill=col, width=3)
        text(d, (x0, y0 - 28), lab, 18, MUTED)
        for f in range(5):
            text(d, (x0 + f / 4 * w, y0 + h + 16), f"{f}M", 14, MUTED, anchor="mm")
    if t > T * 0.6:
        text(d, (110, 600), "flat score for 2 M frames, healthy signals: a slow start, not a bug", 20, INK)
    return im


SCENES = {"lesson1": [l1_s1, l1_s2, l1_s3, l1_s4, l1_s5], "lesson2": [l2_s1, l2_s2, l2_s3, l2_s4, l2_s5],
          "lesson3": [l3_s1, l3_s2, l3_s3, l3_s4, l3_s5, l3_s6],
          "lesson4": [l4_s1, l4_s2, l4_s3, l4_s4, l4_s5]}

if __name__ == "__main__":
    render_film(SCENES[LESSON], f"voice/{LESSON}", ROOT / "docs" / "video" / f"{LESSON}.mp4", ROOT / "video" / "build" / LESSON)
