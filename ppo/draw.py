"""Drawing for clips and replays (no policy imports this). Frames are 210x160 RGB from ALE."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACTIONS = ["NOOP", "FIRE", "RIGHT", "LEFT"]


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frame_panel(rgb, scale=2, top="", bottom="", probs=None, action=None, banner=None):
    """The game frame scaled up, with a caption bar above and below and optionally the action probabilities."""
    im = Image.fromarray(rgb).resize((160 * scale, 210 * scale), Image.NEAREST)
    W, H = im.size
    bar = 26 * scale // 2
    out = Image.new("RGB", (W, H + 2 * bar), (16, 18, 24))
    out.paste(im, (0, bar))
    d = ImageDraw.Draw(out)
    f = font(11 * scale // 2 + 3)
    d.text((6, 4), top, fill=(235, 235, 235), font=f)
    d.text((6, H + bar + 4), bottom, fill=(200, 200, 210), font=f)
    if probs is not None:
        x0 = W - 4 * 34 - 6
        for i, p in enumerate(probs):
            x = x0 + i * 34
            h = int((bar - 6) * float(p))
            col = (255, 200, 60) if i == action else (110, 140, 200)
            d.rectangle([x, H + 2 * bar - 3 - h, x + 26, H + 2 * bar - 3], fill=col)
    if banner:
        fb = font(14 * scale // 2 + 6)
        tw = d.textlength(banner, font=fb)
        d.rectangle([W / 2 - tw / 2 - 10, H / 2, W / 2 + tw / 2 + 10, H / 2 + 40], fill=(0, 0, 0))
        d.text((W / 2 - tw / 2, H / 2 + 6), banner, fill=(255, 220, 90), font=fb)
    return out


def encode_mp4(frames, out, fps=30, hold=45, crf=28):
    """frames: PIL images of equal even size. Plays once (no loop); the last frame is held `hold` frames."""
    w, h = frames[0].size
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for f in frames + [frames[-1]] * hold:
        p.stdin.write(f.convert("RGB").tobytes())
    p.stdin.close(); p.wait()
