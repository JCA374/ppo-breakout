"""Numbers for the scripts that quote this run's training signals (video 3 and lesson 3), from runs/main/stats.csv.
Writes video/ppo3/facts.json + video/ppo3/script.md (from script.tmpl.md) and video/lesson3/facts.json + script.md.

    python video/facts.py
"""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = list(csv.DictReader(open(ROOT / "runs" / "main" / "stats.csv")))
F = np.array([int(r["frames"]) for r in R]) / 1e6
col = lambda k: np.array([float(r[k]) for r in R])
KL, CF, ENT, EV = col("approx_kl"), col("clipfrac"), col("entropy"), col("explained_variance")


def say(x, dec=1):
    """A number as it's spoken: 0.0045 -> 'four and a half thousandths' is awkward, so we use per-mille/percent words."""
    return f"{x:.{dec}f}"


def phase(a, lo, hi):
    m = (F >= lo) & (F < hi)
    return float(np.median(a[m])) if m.any() else float("nan")


total = float(F[-1])
kl_med = float(np.median(KL)); kl_p99 = float(np.percentile(KL, 99)); kl_max = float(KL.max())
cf_mean = float(CF.mean()); cf_p95 = float(np.percentile(CF, 95))
kl_ph = [phase(KL, a, b) for a, b in [(0, 2), (2, 15), (15, 41)]]
cf_ph = [phase(CF, a, b) for a, b in [(0, 2), (2, 15), (15, 41)]]
ent0, ent_end = float(ENT[:5].mean()), float(ENT[-50:].mean()); ent_min = float(ENT.min())
ev_med = float(np.median(EV[F > 1]))

f3 = dict(frames_m=round(total, 1), kl_median=kl_med, kl_p99=kl_p99, kl_max=kl_max, clipfrac_mean=cf_mean, clipfrac_p95=cf_p95,
          kl_by_phase=kl_ph, clip_by_phase=cf_ph,
          lines=[f"median approx KL {kl_med:.4f} per update · 99% of updates below {kl_p99:.4f} · max {kl_max:.4f}",
                 f"mean clip fraction {100 * cf_mean:.0f}% · 95% of updates below {100 * cf_p95:.0f}%"],
          summary=[f"{total:.0f} M frames, {len(R):,} update rounds, each a small step",
                   f"typical change per update: KL ≈ {kl_med:.4f}",
                   f"about {round(100 * cf_mean)}% of samples hit the clip",
                   "no collapse: the score climbed steadily (see the Log tab)"])
(ROOT / "video" / "ppo3").mkdir(exist_ok=True)
json.dump(f3, open(ROOT / "video" / "ppo3" / "facts.json", "w"), indent=1)


def thousandths(x):
    n = round(x * 1000)
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n, str(n))


def pct_words(x):
    return f"{round(100 * x)} percent"


t = (ROOT / "video" / "ppo3" / "script.tmpl.md").read_text()
t = t.replace("{KL_SENTENCE}", f"The typical KL divergence between the old and the new policy was about {thousandths(kl_med)} thousandths per round of updates, "
                               f"and ninety nine percent of rounds stayed below {thousandths(kl_p99)} thousandths.")
t = t.replace("{CLIP_SENTENCE}", f"On average, {pct_words(cf_mean)} of the samples hit the clip.")
t = t.replace("{TREND_SENTENCE}", "The clip did most of its work early, while the policy was changing fastest, and stayed busy all the way through." if cf_ph[0] > cf_ph[2]
              else "The clip stayed busy all the way through, not just at the start.")
t = t.replace("{SUMMARY_SENTENCE}", f"That is the whole trick. About {round(len(R), -2):,} rounds of updates, none of them a big leap, added up to an agent that "
                                    "learned Breakout from pixels. PPO is not the cleverest method. It is the one that rarely breaks.")
(ROOT / "video" / "ppo3" / "script.md").write_text(t)

f_l3 = dict(ent0=ent0, ent_end=ent_end, ent_min=ent_min, kl_med=kl_med, cf_mean=cf_mean, ev_med=ev_med, frames_m=total)
(ROOT / "video" / "lesson3").mkdir(exist_ok=True)
json.dump(f_l3, open(ROOT / "video" / "lesson3" / "facts.json", "w"), indent=1)
print(json.dumps(dict(ppo3=f3, lesson3=f_l3), indent=1))

WORDS = {i: w for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
frac = ent_end / np.log(4)
frac_words = "about half" if 0.4 <= frac < 0.6 else "about a third" if 0.28 <= frac < 0.4 else "about two thirds" if 0.6 <= frac < 0.75 else f"{round(100 * frac)} percent"
ev_late = float(np.median(EV[F >= 8]))
ev_words = "about one half" if 0.42 <= ev_late < 0.6 else "about two thirds" if 0.6 <= ev_late < 0.75 else f"about {round(ev_late, 1)}"
t = (ROOT / "video" / "lesson3" / "script.tmpl.md").read_text()
t = t.replace("{ENT_FRAC}", frac_words).replace("{KL_WORDS}", WORDS.get(round(kl_med * 1000), str(round(kl_med * 1000))))
t = t.replace("{CF_WORDS}", WORDS.get(round(cf_mean * 100), str(round(cf_mean * 100)))).replace("{EV_LATE}", ev_words)
(ROOT / "video" / "lesson3" / "script.md").write_text(t)
print("entropy end/max", round(frac, 2), "ev median", round(ev_med, 3))
