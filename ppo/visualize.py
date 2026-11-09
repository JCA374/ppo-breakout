"""Visualizations of the trained network for the 'How it sees' and 'The model' pages (and their videos).

    python ppo/visualize.py            # -> docs/viz/*.png + docs/viz/viz.json
Everything comes from real checkpoints of the main run and a real recorded game of the final model.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record import load, record_episode, ACTIONS
from train import ROOT

OUT = ROOT / "docs" / "viz"
FINAL = ROOT / "checkpoints" / "main" / "model_039997k.pt"
STAGES = [("0.5 M", "model_000503k.pt"), ("10 M", "model_010002k.pt"), ("40 M", "model_039997k.pt")]
SEED = 77
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def activations(agent, stack):
    """Outputs of every layer for one 4x84x84 input."""
    x = torch.from_numpy(stack[None].astype(np.float32)) / 255.0
    acts = {}
    with torch.no_grad():
        h = x
        for i, layer in enumerate(agent.network):
            h = layer(h)
            if i in (1, 3, 5, 8):
                acts[{1: "conv1", 3: "conv2", 5: "conv3", 8: "dense"}[i]] = h[0].numpy()
        logits = agent.actor(h)[0]
        acts["probs"] = torch.softmax(logits, 0).numpy(); acts["value"] = float(agent.critic(h)[0, 0])
    return acts


def motion_score(w):
    """How much a conv1 filter compares frames over time: weight difference between frames vs total weight."""
    return float(np.abs(w[3] - w[2]).sum() + np.abs(w[2] - w[1]).sum()) / (float(np.abs(w).sum()) + 1e-9)


def filters_png(ckpt, out, title):
    W = torch.load(ckpt, weights_only=False, map_location="cpu")["model"]["network.0.weight"].numpy()  # 32,4,8,8
    order = np.argsort([-motion_score(w) for w in W])
    fig, axes = plt.subplots(8, 16, figsize=(12, 6.4))
    for k in range(32):
        f = order[k]; m = np.abs(W[f]).max()
        for j in range(4):
            ax = axes[k // 4, (k % 4) * 4 + j]
            ax.imshow(W[f, j], cmap="RdBu_r", vmin=-m, vmax=m); ax.set_xticks([]); ax.set_yticks([])
            if k < 4:
                ax.set_title(["t−3", "t−2", "t−1", "now"][j], fontsize=9)
    fig.suptitle(title + "  (red = positive weight, blue = negative; sorted: strongest motion detectors first)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
    return [round(motion_score(W[f]), 3) for f in order]


def top_filters_png(ckpt, out, k=6):
    W = torch.load(ckpt, weights_only=False, map_location="cpu")["model"]["network.0.weight"].numpy()
    order = np.argsort([-motion_score(w) for w in W])[:k]
    fig, axes = plt.subplots(k, 5, figsize=(6.5, 1.35 * k), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]})
    for r, f in enumerate(order):
        m = np.abs(W[f]).max()
        for j in range(4):
            ax = axes[r, j]; ax.imshow(W[f, j], cmap="RdBu_r", vmin=-m, vmax=m); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["t−3", "t−2", "t−1", "now"][j], fontsize=10)
        ax = axes[r, 4]; d = W[f, 3] - W[f, 2]; md = np.abs(d).max()
        ax.imshow(d, cmap="PuOr_r", vmin=-md, vmax=md); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("now − t−1", fontsize=10)
        axes[r, 0].set_ylabel(f"filter {f}", fontsize=9)
    fig.suptitle("the 6 strongest motion detectors (final model)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


def maps_png(act, out, n, cols, title, size):
    a = act; strength = a.reshape(a.shape[0], -1).max(1)
    top = np.argsort(-strength)[:n]
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size + 0.5))
    for k, ax in enumerate(np.array(axes).ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(top):
            ax.imshow(a[top[k]], cmap="magma", vmin=0, vmax=max(1e-6, a[top[k]].max()))
            ax.set_title(f"#{top[k]}", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
    return int((strength > 0).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    agent = load(FINAL)
    ep = record_episode(agent, SEED, max_steps=4000)
    n = len(ep["action"])
    # a moment with the ball moving fast, low in the court, heading for the paddle (largest motion in rows 55-76)
    mv = [np.abs(ep["obs"][t][3, 55:76].astype(int) - ep["obs"][t][2, 55:76]).sum() for t in range(n)]
    t0 = int(np.argmax(mv[100:n // 2])) + 100
    st = ep["obs"][t0]
    acts = activations(agent, st)
    info = {"seed": SEED, "score": ep["score"], "steps": int(ep["steps"]), "moment": t0}

    # the input
    for i in range(4):
        Image.fromarray(st[i]).resize((252, 252), Image.NEAREST).save(OUT / f"in_{i}.png")
    Image.fromarray(ep["rgb"][t0]).resize((320, 420), Image.NEAREST).save(OUT / "screen.png")
    diff = np.zeros((84, 84, 3), np.uint8)
    for i, c in enumerate([(70, 70, 255), (70, 200, 255), (255, 200, 70), (255, 80, 50)]):
        diff[np.abs(st[i].astype(int) - st[0].astype(int) if i else np.zeros((84, 84))) > 20] = c
    Image.fromarray(diff).resize((252, 252), Image.NEAREST).save(OUT / "motion.png")
    # pixel level, a zoom on the ball
    ys, xs = np.where(np.abs(st[3].astype(int) - st[2].astype(int))[20:72] > 20)   # above the paddle rows
    cy, cx = (int(ys.mean()) + 20, int(xs.mean())) if len(ys) else (60, 42)
    y0, x0 = max(0, cy - 8), max(0, min(84 - 16, cx - 8))
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    crop = st[3, y0:y0 + 16, x0:x0 + 16]
    ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
    for (r, c), v in np.ndenumerate(crop):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=5.5, color="red" if v < 128 else "black")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"16×16 pixels around the ball (rows {y0}–{y0 + 15})", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "pixels.png", dpi=110); plt.close(fig)

    # learned filters over training
    info["motion_scores"] = {}
    for lab, f in STAGES:
        info["motion_scores"][lab] = filters_png(ROOT / "checkpoints" / "main" / f, OUT / f"filters_{lab.split()[0]}.png",
                                                 f"conv1: the 32 learned 8×8×4 filters after {lab} frames")
    top_filters_png(FINAL, OUT / "filters_top.png")
    # what each layer outputs for this moment
    info["active"] = {
        "conv1": maps_png(acts["conv1"], OUT / "maps_conv1.png", 32, 8, "conv1 output: 32 maps of 20×20 (brightest = strongest response)", 1.3),
        "conv2": maps_png(acts["conv2"], OUT / "maps_conv2.png", 32, 8, "conv2 output: the 32 most active of 64 maps, 9×9", 1.1),
        "conv3": maps_png(acts["conv3"], OUT / "maps_conv3.png", 32, 8, "conv3 output: the 32 most active of 64 maps, 7×7", 1.1)}
    d = acts["dense"]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.bar(range(512), d, width=1.0, color="#5b7bdb"); ax.set_xlim(0, 512)
    ax.set_title(f"the 512 numbers that summarise the moment: {int((d > 0).sum())} active, {int((d == 0).sum())} exactly zero (ReLU)", fontsize=10)
    ax.set_xlabel("unit"); fig.tight_layout(); fig.savefig(OUT / "dense.png", dpi=80); plt.close(fig)
    info["probs"] = [round(float(p), 3) for p in acts["probs"]]; info["value"] = round(acts["value"], 3)
    info["action"] = ACTIONS[int(ep["action"][t0])]
    info["dense_active"] = int((d > 0).sum())

    # the outputs over the game
    a, b = max(0, t0 - 120), min(n, t0 + 120)
    P = ep["probs"][a:b]; V = ep["value"][a:b]; R = ep["reward"][a:b]; L = ep["lives"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax1.stackplot(range(a, b), P.T, labels=ACTIONS, colors=["#9aa0b0", "#e8a33a", "#5b7bdb", "#69b56a"], alpha=.9)
    ax1.set_ylim(0, 1); ax1.set_ylabel("actor:\nprobabilities"); ax1.legend(loc="upper left", ncol=4, fontsize=9, frameon=False)
    ax2.plot(range(a, b), V, color="#5b7bdb", lw=1.6, label="critic: value")
    for k in range(a, b):
        if ep["reward"][k] > 0:
            ax2.axvline(k, color="#2f9e44", alpha=.35, lw=1)
        if k and L[k] < L[k - 1]:
            ax2.axvline(k, color="#c92a2a", lw=2)
    ax2.axvline(t0, color="k", ls=":"); ax1.axvline(t0, color="k", ls=":")
    ax2.set_ylabel("critic:\nvalue"); ax2.set_xlabel("agent step (1 step = 4 frames = 1/15 s)")
    ax2.set_title("green = a brick broke, red = a life was lost, dotted = the moment shown above", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "timeline.png", dpi=80); plt.close(fig)

    # the network's internal map: 512-d summaries of every moment of the game, projected to 2-D (PCA)
    feats = []
    with torch.no_grad():
        for k in range(8, n - 1, 2):            # skip the zero-padded start and the final frame
            x = torch.from_numpy(ep["obs"][k][None].astype(np.float32)) / 255.0
            feats.append(agent.network(x)[0].numpy())
    Fm = np.array(feats); Fc = Fm - Fm.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    Z = Fc @ Vt[:2].T
    expl = (S[:2] ** 2 / (S ** 2).sum())
    vals = ep["value"][8:n - 1:2]; acts_k = ep["action"][8:n - 1:2]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = a1.scatter(Z[:, 0], Z[:, 1], c=vals, s=6, cmap="viridis"); fig.colorbar(sc, ax=a1, label="critic value")
    a1.set_title("coloured by the critic's value")
    for i, c in enumerate(["#9aa0b0", "#e8a33a", "#5b7bdb", "#69b56a"]):
        m = acts_k == i; a2.scatter(Z[m, 0], Z[m, 1], s=6, c=c, label=ACTIONS[i])
    a2.legend(markerscale=3, fontsize=9, frameon=False); a2.set_title("coloured by the action it took")
    for ax in (a1, a2):
        ax.set_xlabel(f"direction 1 ({100 * expl[0]:.0f}% of variation)"); ax.set_ylabel(f"direction 2 ({100 * expl[1]:.0f}%)")
    fig.suptitle(f"every 2nd moment of one game ({n} steps, {ep['score']:.0f} points): the 512-number summaries, flattened to 2-D", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "embedding.png", dpi=80); plt.close(fig)
    info["pca_explained"] = [round(float(e), 3) for e in expl]
    info["value_corr_dir1"] = round(float(np.corrcoef(Z[:, 0], vals)[0, 1]), 3)
    info["value_corr_dir2"] = round(float(np.corrcoef(Z[:, 1], vals)[0, 1]), 3)
    json.dump(info, open(OUT / "viz.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
