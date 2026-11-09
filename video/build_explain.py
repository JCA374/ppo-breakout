"""Explainer videos for the 'How it sees' and 'The model' pages. Scenes match video/<name>/script.md.

    python video/build_explain.py howitsees|themodel
Needs docs/viz/* (python ppo/visualize.py) and docs/senses/* (the site's saliency maps of the final model).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
from record import load

NAME = sys.argv[1]
VIZ = ROOT / "docs" / "viz"
INFO = json.load(open(VIZ / "viz.json"))
FINAL = ROOT / "checkpoints" / "main" / "model_039997k.pt"
torch.set_num_threads(4)


def img(name, h=None, w=None):
    im = Image.open(VIZ / name if not str(name).startswith("/") else name).convert("RGB")
    if h and w:
        s = min(h / im.height, w / im.width)
    elif h:
        s = h / im.height
    elif w:
        s = w / im.width
    else:
        return im
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def fade_in(base, im, xy, t, t0, dur=0.6):
    a = ease((t - t0) / dur)
    if a <= 0:
        return
    if a < 1:
        bg = base.crop((xy[0], xy[1], xy[0] + im.width, xy[1] + im.height))
        im = Image.blend(bg, im, a)
    base.paste(im, xy)


def wipe(base, im, xy, frac):
    w = int(im.width * min(1, max(0, frac)))
    if w > 0:
        base.paste(im.crop((0, 0, w, im.height)), xy)


def white_panel(d, x, y, w, h):
    d.rounded_rectangle([x - 8, y - 8, x + w + 8, y + h + 8], 10, fill=(250, 250, 250))


# ------------------------------------------------------------------ How it sees
def h1(t, T):
    im, d = canvas(); header(d, "Only numbers", "84 × 84 gray pixels, 0 = black … 255 = white")
    fade_in(im, img("screen.png", h=420), (40, 130), t, 0.2)
    text(d, (40, 565), "the screen", 18, MUTED)
    fade_in(im, img("in_3.png", h=330), (390, 170), t, T * 0.2)
    text(d, (390, 520), "what the agent gets", 18, MUTED)
    p = img("pixels.png", h=470)
    if t > T * 0.45:
        white_panel(d, 760, 120, p.width, p.height)
    fade_in(im, p, (760, 120), t, T * 0.45)
    return im


def h2(t, T):
    im, d = canvas(); header(d, "Four pictures, one input", "the input is 4 × 84 × 84 numbers: the last four pictures")
    for i in range(4):
        fade_in(im, img(f"in_{i}.png", h=230), (40 + i * 250, 140), t, i * T * 0.1)
        if t > i * T * 0.1:
            text(d, (155 + i * 250, 385), ["t−3", "t−2", "t−1", "now"][i], 18, MUTED, anchor="mm")
    if t > T * 0.5:
        fade_in(im, img("motion.png", h=250), (980, 420), t, T * 0.5)
        text(d, (40, 470), "where the ball was in each picture:", 22, INK)
        for i, (lab, c) in enumerate([("t−2", (70, 200, 255)), ("t−1", (255, 200, 70)), ("now", (255, 80, 50))]):
            d.rectangle([40 + i * 150, 520, 64 + i * 150, 544], fill=c); text(d, (72 + i * 150, 520), lab, 18, INK)
        text(d, (40, 580), "the trail gives direction and speed", 22, WARN, bold=True)
    return im


# the first layer, animated: one real motion filter sliding over the real input
if NAME == "howitsees":
    AG = load(FINAL)
    STK = np.array([np.array(Image.open(VIZ / f"in_{i}.png").convert("L").resize((84, 84), Image.NEAREST)) for i in range(4)])
    W0 = AG.network[0].weight.detach().numpy()
    def _ms(w):
        return float(np.abs(w[3] - w[2]).sum() + np.abs(w[2] - w[1]).sum()) / float(np.abs(w).sum())
    FI = int(np.argmax([_ms(w) for w in W0]))
    with torch.no_grad():
        C1 = AG.network[1](AG.network[0](torch.from_numpy(STK[None].astype(np.float32)) / 255.0))[0, FI].numpy()


def h3(t, T):
    im, d = canvas(); header(d, "The first layer", f"32 filters of 8 × 8 × 4, sliding in steps of 4; here filter {FI}, a motion detector")
    s = 5; x0, y0 = 40, 150
    im.paste(Image.fromarray(STK[3]).resize((84 * s, 84 * s), Image.NEAREST).convert("RGB"), (x0, y0))
    k = int(min(399, (t / (T * 0.75)) * 400)); r, c = divmod(k, 20)
    d.rectangle([x0 + c * 4 * s, y0 + r * 4 * s, x0 + (c * 4 + 8) * s, y0 + (r * 4 + 8) * s], outline=WARN, width=3)
    g = 14; gx, gy = 520, 150
    m = C1.max() + 1e-9
    for j in range(k + 1):
        rr, cc = divmod(j, 20); v = C1[rr, cc] / m
        col = tuple(int(30 + 225 * v * a) for a in (1.0, 0.75, 0.3))
        d.rectangle([gx + cc * g, gy + rr * g, gx + cc * g + g - 1, gy + rr * g + g - 1], fill=col)
    d.rectangle([gx, gy, gx + 20 * g, gy + 20 * g], outline=(60, 64, 76))
    text(d, (gx, gy + 20 * g + 12), "its output: a 20 × 20 map", 17, MUTED)
    text(d, (x0, y0 + 84 * s + 12), "input (newest picture shown)", 17, MUTED)
    if t > T * 0.45:
        fade_in(im, img("filters_top.png", h=560, w=420), (840, 120), t, T * 0.45)
    return im


def h4(t, T):
    im, d = canvas(); header(d, "Layer by layer", "less about where, more about what")
    steps = [("maps_conv1.png", "conv1: 32 maps, 20 × 20"), ("maps_conv2.png", "conv2: 64 maps, 9 × 9 (32 most active shown)"),
             ("maps_conv3.png", "conv3: 64 maps, 7 × 7 (32 most active shown)")]
    k = min(2, int(t / (T / 3)))
    f, cap = steps[k]
    p = img(f, h=520, w=1180)
    white_panel(d, 640 - p.width // 2, 120, p.width, p.height)
    im.paste(p, (640 - p.width // 2, 120))
    text(d, (50, 668), cap, 20, WARN, bold=True)
    return im


def h5(t, T):
    im, d = canvas(); header(d, "A summary of the moment", "512 numbers; the actor and the critic read only these")
    p = img("dense.png", w=1180)
    white_panel(d, 50, 200, p.width, p.height)
    wipe(im, p, (50, 200), t / (T * 0.6))
    if t > T * 0.6:
        text(d, (50, 520), f"{INFO['dense_active']} of 512 active; the rest exactly zero (ReLU cuts negatives to 0)", 22, INK)
    return im


SEN = json.load(open(ROOT / "docs" / "senses" / "senses.json"))


def h6(t, T):
    im, d = canvas(); header(d, "Where it looks", "saliency: blur a patch, measure how much the output changes")
    for j, k in enumerate([1, 2]):
        y = 120 + j * 290
        im.paste(img(ROOT / "docs" / "senses" / f"m{k}_rgb.png", h=270), (60, y))
        fade_in(im, img(ROOT / "docs" / "senses" / f"m{k}_actor.png", h=270), (300, y), t, T * 0.15 + j * 0.3)
        fade_in(im, img(ROOT / "docs" / "senses" / f"m{k}_critic.png", h=270), (600, y), t, T * 0.35 + j * 0.3)
    text(d, (920, 200), "orange: what moves the", 20, (255, 120, 80)); text(d, (920, 228), "button choice", 20, (255, 120, 80))
    if t > T * 0.35:
        text(d, (920, 300), "blue: what moves the", 20, (90, 170, 255)); text(d, (920, 328), "value estimate", 20, (90, 170, 255))
    if t > T * 0.6:
        text(d, (920, 420), "learned from points alone", 22, WARN, bold=True)
    return im


# ------------------------------------------------------------------ The model
LAYERS = [("conv1", 8224, "32 filters 8×8×4"), ("conv2", 32832, "64 filters 4×4×32"), ("conv3", 36928, "64 filters 3×3×64"),
          ("dense", 1606144, "3136 → 512"), ("actor", 2052, "512 → 4"), ("critic", 513, "512 → 1")]


def m1(t, T):
    im, d = canvas(); header(d, "One small network", "1,686,693 weights")
    x = 60
    for k, (n, w, s) in enumerate(LAYERS[:4]):
        if t > k * T * 0.08:
            box_h = 70 + 30 * k
            d.rounded_rectangle([x, 300 - box_h // 2, x + 170, 300 + box_h // 2], 12, fill=ACC if n == "dense" else PANEL)
            text(d, (x + 85, 290), n, 20, BG if n == "dense" else INK, bold=True, anchor="mm")
            text(d, (x + 85, 316), s, 14, BG if n == "dense" else MUTED, anchor="mm")
            if k < 3:
                d.line([x + 170, 300, x + 210, 300], fill=MUTED, width=3)
        x += 210
    if t > T * 0.35:
        d.line([x - 40, 300, x + 10, 240], fill=WARN, width=3); d.line([x - 40, 300, x + 10, 360], fill=ACC, width=3)
        d.rounded_rectangle([x + 10, 210, x + 170, 270], 10, fill=WARN); text(d, (x + 90, 240), "actor", 20, BG, bold=True, anchor="mm")
        d.rounded_rectangle([x + 10, 330, x + 170, 390], 10, fill=ACC); text(d, (x + 90, 360), "critic", 20, BG, bold=True, anchor="mm")
    if t > T * 0.5:
        tot = sum(w for _, w, _ in LAYERS); x0, y0, W = 60, 500, 1160
        cx = x0
        for n, w, _ in LAYERS:
            ww = W * w / tot * ease((t - T * 0.5) / 1.5)
            d.rectangle([cx, y0, cx + ww, y0 + 50], fill=ACC if n == "dense" else (90, 100, 130), outline=BG)
            cx += ww
        text(d, (x0, y0 + 64), f"share of weights: dense {100 * 1606144 / tot:.0f}%, the three conv layers {100 * (8224 + 32832 + 36928) / tot:.1f}%, heads {100 * 2565 / tot:.1f}%", 19, INK)
    return im


def m2(t, T):
    im, d = canvas(); header(d, "Two heads, one moment", "the real outputs of the final model at one moment of a game")
    im.paste(img("screen.png", h=440), (50, 130))
    names = ["NOOP", "FIRE", "RIGHT", "LEFT"]; P = INFO["probs"]
    text(d, (450, 150), "ACTOR: probability of each button", 22, WARN, bold=True)
    for k in range(4):
        w = int(560 * P[k] * ease((t - 0.3) / 1.5))
        d.rectangle([560, 200 + k * 50, 560 + w, 236 + k * 50], fill=WARN)
        text(d, (450, 204 + k * 50), names[k], 20, INK); text(d, (570 + w, 204 + k * 50), f"{100 * P[k]:.0f}%", 18, MUTED)
    if t > T * 0.5:
        text(d, (450, 440), "CRITIC: value (reward still to come, discounted)", 22, ACC, bold=True)
        text(d, (450, 480), f"{INFO['value']:.2f}", 60, ACC, bold=True)
    return im


def m3(t, T):
    im, d = canvas(); header(d, "Over a game", "240 real steps: the actor's probabilities (top) and the critic's value (bottom)")
    p = img("timeline.png", w=1180)
    white_panel(d, 50, 120, p.width, p.height)
    wipe(im, p, (50, 120), t / (T * 0.8))
    return im


def m4(t, T):
    im, d = canvas(); header(d, "The model's own map", "the 512-number summary of every moment, flattened to 2-D (PCA)")
    p = img("embedding.png", w=1180)
    white_panel(d, 50, 115, p.width, p.height)
    fade_in(im, p, (50, 115), t, 0.2, 1.0)
    if t > T * 0.45:
        text(d, (60, 640), "left group: early game, ~73% of the wall standing, value ≈ 4.3", 19, GOOD)
        text(d, (60, 670), "right group: late game, ~7% of the wall left, value ≈ 0.6", 19, (180, 140, 255))
    return im


CORR = [("conv1", 0.778), ("conv2", 0.631), ("conv3", 0.674), ("dense", 0.479), ("actor", 0.426), ("critic", 0.283)]


def m5(t, T):
    im, d = canvas(); header(d, "How training shaped it", "correlation of each layer's weights: after 0.5 M frames vs the final model")
    x0, y0, h = 120, 160, 360
    for k, (n, c) in enumerate(CORR):
        if t > k * T * 0.07:
            x = x0 + k * 175; hh = h * c * ease((t - k * T * 0.07) / 0.8)
            d.rectangle([x, y0 + h - hh, x + 110, y0 + h], fill=ACC if k < 4 else WARN)
            text(d, (x + 55, y0 + h - hh - 22), f"{c:.2f}", 20, INK, bold=True, anchor="mm")
            text(d, (x + 55, y0 + h + 22), n, 18, MUTED, anchor="mm")
    d.line([x0 - 20, y0 + h, x0 + 6 * 175, y0 + h], fill=MUTED)
    if t > T * 0.5:
        text(d, (120, 600), "1.0 = unchanged. The first layer changed least, the heads most.", 21, INK)
        text(d, (120, 636), "The actor's weights grew 10× in size (norm 0.5 → 5.0): a confident policy.", 21, WARN)
    return im


def m6(t, T):
    im, d = canvas(); header(d, "The whole picture", "")
    steps = [("pixels", "4 × 84 × 84"), ("patterns", "3 conv layers"), ("summary", "512 numbers"), ("act", "actor: 4 probabilities"), ("judge", "critic: 1 value")]
    for k, (a, b) in enumerate(steps):
        if t > k * T * 0.12:
            x = 50 + k * 245
            d.rounded_rectangle([x, 260, x + 215, 380], 14, fill=WARN if a == "act" else ACC if a == "judge" else PANEL)
            text(d, (x + 107, 300), a, 26, BG if a in ("act", "judge") else INK, bold=True, anchor="mm")
            text(d, (x + 107, 340), b, 16, BG if a in ("act", "judge") else MUTED, anchor="mm")
            if 0 < k < 4:
                arrow_x = x - 30
                d.line([arrow_x, 320, x, 320], fill=MUTED, width=3)
    if t > T * 0.7:
        text(d, (50, 480), "everything else is learned from points", 30, WARN, bold=True)
    return im


SCENES = {"howitsees": [h1, h2, h3, h4, h5, h6], "themodel": [m1, m2, m3, m4, m5, m6]}

if __name__ == "__main__":
    render_film(SCENES[NAME], f"voice/{NAME}", ROOT / "docs" / "video" / f"{NAME}.mp4", ROOT / "video" / "build" / NAME)
