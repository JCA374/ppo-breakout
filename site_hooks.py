"""Project hooks for publish.py (ml-project-site skill): replays, the Play tab, and the MLflow export."""
import glob, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ppo"))
from draw import frame_panel, encode_mp4, ACTIONS  # noqa: E402

RUN = "main"
CLIP_CAP = 4000   # agent steps per Play clip (a stuck ball can bounce for the full 27k-step limit)


def _frames_of(path):
    m = re.search(r"model_(\d+)k\.pt$", str(path))
    return int(m.group(1)) * 1000 if m else 0


def _fm(frames):
    return f"{frames / 1e6:.1f} M frames"


def checkpoints(run=RUN):
    return sorted(glob.glob(str(ROOT / "checkpoints" / run / "model_*k.pt")), key=_frames_of)


def replay_gif(policy_path, seed, out_gif):
    """First 40 s (at 2x speed) of one game, as a GIF that plays once."""
    import torch
    torch.set_num_threads(1)
    from record import load, record_episode
    ep = record_episode(load(policy_path), seed, max_steps=1200, keep_obs=False)
    fr = _frames_of(policy_path)
    imgs, score = [], 0
    for t, rgb in enumerate(ep["rgb"][:-1]):
        score += ep["reward"][t]
        if t % 2:
            continue
        imgs.append(frame_panel(rgb, scale=1, top=f"{_fm(fr)} trained", bottom=f"score {score:.0f}  2x").quantize(64))
    end = "game over" if not ep["truncated"] else "(clip ends)"
    imgs.append(frame_panel(ep["rgb"][-1], scale=1, top=f"{_fm(fr)} trained", bottom=f"score {ep['score']:.0f}",
                            banner=f"{ep['score']:.0f} pts {end}").quantize(64))
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 20, duration=66, optimize=True)


def models():
    """Early, middle and late checkpoints of the main run, plus the goal-checked best."""
    cks = checkpoints()
    if not cks:
        return []
    picks = {}
    targets = [500_000, 2_000_000, 5_000_000]
    for tg in targets:
        c = min(cks, key=lambda p: abs(_frames_of(p) - tg))
        if abs(_frames_of(c) - tg) <= 600_000:
            picks[c] = None
    picks[cks[-1]] = None
    best = ROOT / "results" / "best.json"
    note_best = None
    if best.exists():
        b = json.load(open(best))
        if (ROOT / b["policy"]).exists():
            picks[str(ROOT / b["policy"])] = None
            note_best = b["policy"]
    out = []
    for c in sorted(picks, key=_frames_of, reverse=True):
        f = _frames_of(c)
        note = "newest checkpoint" if c == cks[-1] else "checkpoint"
        if note_best and c.endswith(note_best):
            note = "goal-checked best"
        out.append({"id": f"f{f // 1000:06d}k", "label": f"After {_fm(f)}", "path": c, "note": note})
    return out[:5]


GAMES = [
    (101, "Game A", "seed 101: no-op start and the agent's own action sampling decide everything that follows"),
    (202, "Game B", "seed 202"),
    (303, "Game C", "seed 303"),
    (404, "Game D", "seed 404"),
    (505, "Game E", "seed 505"),
    (606, "Game F", "seed 606"),
]


def games():
    return [{"id": f"g{s}", "title": t, "desc": d, "seed": s} for s, t, d in GAMES]


def render_play(policy_path, game, out_mp4):
    import torch
    torch.set_num_threads(1)
    from record import load, record_episode
    ep = record_episode(load(policy_path), game["seed"], max_steps=CLIP_CAP, keep_obs=False)
    fr = _frames_of(policy_path)
    imgs, score = [], 0
    n = len(ep["action"])
    for t in range(n):
        score += ep["reward"][t]
        imgs.append(frame_panel(ep["rgb"][t], scale=2, top=f"{_fm(fr)} · {game['title']} · 2x",
                                bottom=f"score {score:.0f}  lives {ep['lives'][t]}  {ACTIONS[ep['action'][t]]}",
                                probs=ep["probs"][t], action=ep["action"][t]))
    cut = ep["truncated"]
    result = f"{ep['score']:.0f} points" + (" (cut)" if cut else "")
    imgs.append(frame_panel(ep["rgb"][-1], scale=2, top=f"{_fm(fr)} trained · {game['title']}",
                            bottom=f"final score {ep['score']:.0f}",
                            banner=("clip cut at 4000 steps" if cut else f"game over: {ep['score']:.0f}")))
    encode_mp4(imgs, out_mp4, fps=30)
    return {"ok": not cut, "result": result, "detail": f"{ep['steps']} agent steps, {ep['score']:.0f} points"}


def milestones(data_dir):
    """First frame count at which the mean of the last >=50 training games reached each milestone."""
    import csv
    m = json.load(open(ROOT / "experiments" / "milestones.json"))
    path = ROOT / "runs" / RUN / "stats.csv"
    games = []   # (frames, sum, count) per iteration
    if path.exists():
        for r in csv.DictReader(open(path)):
            if r["episodes"] and int(r["episodes"]) > 0 and r["ep_return_mean"]:
                games.append((int(r["frames"]), float(r["ep_return_mean"]) * int(r["episodes"]), int(r["episodes"])))
    win = []
    for fr, s_, c in games:
        win.append((s_, c))
        while sum(x[1] for x in win[1:]) >= 50:
            win.pop(0)
        n = sum(x[1] for x in win)
        if n >= 50:
            mean = sum(x[0] for x in win) / n
            for ms in m["milestones"]:
                if "reached" not in ms and mean >= ms["score"]:
                    ms["reached"] = fr
    json.dump(m, open(os.path.join(data_dir, "milestones.json"), "w"))


def senses():
    """Saliency page for the newest checkpoint of the main run (re-rendered when a new checkpoint appears)."""
    cks = checkpoints()
    if not cks:
        return
    out = ROOT / "docs" / "senses"
    meta = out / "senses.json"
    want = cks[-1].split("checkpoints/")[-1]
    if meta.exists() and json.load(open(meta)).get("ckpt") == want:
        return
    import torch
    torch.set_num_threads(1)
    from senses import build
    build(cks[-1], out)


def extra(data_dir):
    from export_mlflow import export
    for name, fn in [("mlflow export", lambda: export(os.path.join(data_dir, "mlflow.json"))),
                     ("milestones", lambda: milestones(data_dir)), ("senses", senses)]:
        try:
            fn()
        except Exception as ex:   # never stop the publisher
            print(name, "failed", ex)
    for f in ["env_bench.json", "thread_bench.json"]:
        src = ROOT / "bench" / f
        if src.exists():
            open(os.path.join(data_dir, f), "w").write(src.read_text())
