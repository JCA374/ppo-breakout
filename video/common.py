"""Shared helpers for the explainer videos: 1280x720 canvas drawing, scene timing from the narration, encoding.

Each film = scenes that match the ## sections of video/<name>/script.md one to one. A scene is a function
f(t, T) -> PIL image (t = seconds into the scene, T = scene length = narration + GAP). Its audio is
voice/<name>/audio/voiceover/chNN.mp3 (from the voice-text skill's export).
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ppo"))
FPS = 25
SW, SH = 1280, 720
BG = (14, 17, 22)
PANEL = (26, 29, 36)
INK = (232, 232, 236)
MUTED = (150, 152, 160)
ACC = (120, 160, 255)
GOOD = (105, 219, 124)
BAD = (255, 120, 120)
WARN = (255, 169, 77)
GAP = 0.6


def font(size, bold=False):
    f = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(f, size)


def canvas():
    im = Image.new("RGB", (SW, SH), BG)
    return im, ImageDraw.Draw(im)


def wrap(d, s, f, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def text(d, xy, s, size=26, fill=INK, bold=False, width=None, spacing=1.3, anchor=None):
    f = font(size, bold)
    x, y = xy
    for l in (wrap(d, s, f, width) if width else [s]):
        d.text((x, y), l, fill=fill, font=f, anchor=anchor)
        y += int(size * spacing)
    return y


def header(d, title, sub=None):
    text(d, (48, 28), title, 34, INK, bold=True)
    if sub:
        text(d, (48, 74), sub, 20, MUTED)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fig_to_img(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return Image.frombuffer("RGBA", (w, h), fig.canvas.buffer_rgba()).convert("RGB")


def scene_lengths(voice):
    """Seconds of narration per scene, from voice/<name>/audio/voiceover/timings.json."""
    p = ROOT / voice / "audio" / "voiceover" / "timings.json"
    t = json.load(open(p))
    items = t if isinstance(t, list) else t.get("scenes", t.get("chapters", t))
    if isinstance(items, dict):
        items = [items[k] for k in sorted(items)]
    out = []
    for it in items:
        out.append(float(it if isinstance(it, (int, float)) else it.get("seconds", it.get("duration", 0))))
    return out


def render_film(scenes, voice, out_mp4, tmp, only=None):
    """scenes: list of f(t, T). Renders each scene with its narration, then concatenates. Never loops."""
    tmp = Path(tmp); tmp.mkdir(parents=True, exist_ok=True)
    lens = scene_lengths(voice)
    assert len(lens) == len(scenes), (len(lens), len(scenes))
    parts = []
    for i, (f, L) in enumerate(zip(scenes, lens)):
        part = tmp / f"scene{i + 1:02d}.mp4"
        parts.append(part)
        if only and (i + 1) not in only and part.exists():
            continue
        T = L + GAP
        audio = ROOT / voice / "audio" / "voiceover" / f"ch{i + 1:02d}.mp3"
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SW}x{SH}",
                              "-r", str(FPS), "-i", "-", "-i", str(audio), "-af", f"apad=pad_dur={GAP}",
                              "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-ac", "1", "-t", f"{T:.3f}", str(part)],
                             stdin=subprocess.PIPE)
        n = int(round(T * FPS))
        for k in range(n):
            p.stdin.write(f(k / FPS, T).convert("RGB").tobytes())
        p.stdin.close(); p.wait()
        print(f"scene {i + 1}: {T:.1f} s", flush=True)
    lst = tmp / "parts.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out_mp4)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out_mp4)],
                               capture_output=True, text=True).stdout)
    print(f"{out_mp4}: {dur:.1f} s")
    return dur
