"""SIMULATION-ONLY verification (no model information) that the frozen primary statistic S9 (theta_post_dml) with the
D49 nested-CV ridge behaves correctly when pre-intervention state dimension exceeds the number of items, as in the
real read set (1,280 features vs 300 items). ~50% loss regime. Labels A/B/C as frozen. Not a reselection.
Usage: python sim_highdim_check.py --reps 20 [--worker i --n-workers k | --merge]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s1 import estimands as E  # noqa: E402
from s1.sim_worlds import make_structure, pre_covariates, simulate_items  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "statistics", "v3_statistic_simulation")
WORLDS = ("H3", "null", "rtm", "susceptibility", "H2_intensity", "susceptibility_latent")
D_H = 1280


def decide(t):
    t = np.asarray(t)
    z = np.arctanh(np.clip(t, -0.999, 0.999))
    p = stats.ttest_1samp(z, 0.0, alternative="greater").pvalue
    return bool(p < 0.05 and t.mean() >= 0.10), float(t.mean())


def run(reps, worlds):
    st = make_structure(np.random.default_rng(12345), d_h=D_H)
    st["a_S"] = st["a_S"] * np.sqrt(40 / D_H)          # same total S-signal strength as the d_h = 40 worlds
    st["a_C"] = st["a_C"] * np.sqrt(40 / D_H)
    res = {}
    for w in worlds:
        lab = {"A": 0, "B": 0, "C": 0}
        mp, r2s = [], []
        for r in range(reps):
            rng = np.random.default_rng([2049, r, WORLDS.index(w)])
            tp, tg = [], []
            for s in range(20):
                it = simulate_items(w, 300, rng, d_h=D_H, structure=st, loss_shift=1.5)
                W0 = pre_covariates(it)
                a = (it["M_pre"], it["M_post"], it["C_pre"], it["C_post"])
                G = np.column_stack([it["G"], it["D"], it["F"]])
                tp.append(E.theta_post_dml(*a, W=W0, H_pre=it["H_pre"], seed=1000 * r + s, strata=it["e"])[0])
                tg.append(E.theta_post_dml(*a, W=W0, H_pre=it["H_pre"], seed=1000 * r + s, G=G, strata=it["e"])[0])
                if s == 0:
                    r2s.append(E.crossfit_ridge(np.column_stack([it["H_pre"], W0, it["C_pre"]]), it["C_post"] - it["C_pre"], s)[1])
            sp, m = decide(tp)
            sg, _ = decide(tg)
            lab["A" if (sp and sg) else ("B" if sp else "C")] += 1
            mp.append(m)
        res[w] = {"labels": {k: v / reps for k, v in lab.items()}, "mean_theta_pre": float(np.mean(mp)),
                  "median_r2_pre": float(np.median(r2s))}
        print(w, res[w], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        r = {}
        for f in sorted(os.listdir(OUT)):
            if f.startswith("hd_partial_"):
                r.update(json.load(open(os.path.join(OUT, f))))
        r = {w: r[w] for w in WORLDS if w in r}
        json.dump({"reps": a.reps, "d_h": D_H, "results": r}, open(os.path.join(OUT, "sim_highdim_results.json"), "w"), indent=1)
        lines = ["# High-dimensional verification of S9 with the D49 nested-CV ridge (simulation only)", "",
                 f"{a.reps} replicate studies x 20 seeds x 300 items; pre-state dimension {D_H} (> items); ~50% loss.", "",
                 "| world | mean theta_pre | labels A / B / C | median cross-fitted R2_pre |", "|---|---|---|---|"]
        for w, v in r.items():
            lines.append(f"| {w} | {v['mean_theta_pre']:.3f} | {v['labels']['A']:.2f} / {v['labels']['B']:.2f} / "
                         f"{v['labels']['C']:.2f} | {v['median_r2_pre']:.2f} |")
        open(os.path.join(OUT, "sim_highdim_results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    else:
        ws = WORLDS if a.worker is None else WORLDS[a.worker::a.n_workers]
        json.dump(run(a.reps, ws), open(os.path.join(OUT, f"hd_partial_{a.worker}.json"), "w"), indent=1)
