"""SIMULATION-ONLY check of the binary secondary K1 definition (methods validation; no model information).

K1 = AUROC of the decrease in an ADJUSTED monitor outcome, X_lost vs X_retained, on 1:1 propensity-matched pairs
(propensity from pre-intervention variables only). Outcome definitions compared:
  K1_ancova  resid(rank M_post | RCS(rank M_pre))                       (D37 draft)
  K1_dml     the M-side residual of theta_pre (all pre-intervention information incl. pre-states; D42, adopted)
~50% loss regime (loss_shift 1.5), 50 replicate studies x 20 seeds x 300 items.
Usage: python sim_k1_check.py [--reps 50] [--worker i --n-workers k | --merge]
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
from s1 import metrics as Mx  # noqa: E402
from s1.sim_worlds import make_structure, pre_covariates, simulate_items  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "statistics", "v3_statistic_simulation")
WORLDS = ("H3", "H2_intensity", "susceptibility", "rtm", "null", "susceptibility_latent")


def seed_k1(it, seed):
    W0 = pre_covariates(it)
    dC = it["C_post"] - it["C_pre"]
    P, _, _ = E.crossfit_ridge(np.column_stack([it["H_pre"], W0, it["C_pre"]]), dC, seed, strata=it["e"])
    lost = it["C_post"] < 0
    pairs, smd = E.propensity_match(lost, np.column_stack([it["C_pre"], W0, P]), it["e"], seed + 5)
    if not (len(pairs) >= 50 and smd <= 0.10):
        return np.nan, np.nan
    pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    ra = E.residualize(E.rank01(it["M_post"]), E.rcs(E.rank01(it["M_pre"])))
    rd = E.theta_post_dml(it["M_pre"], it["M_post"], it["C_pre"], it["C_post"], W0, it["H_pre"], seed + 11,
                          strata=it["e"])[1]["resid_M"]
    return Mx.auroc(-ra[pi], -ra[pj]), Mx.auroc(-rd[pi], -rd[pj])


def run(reps, worlds):
    st = make_structure(np.random.default_rng(12345))
    res = {}
    for w in worlds:
        sup = {"K1_ancova": 0, "K1_dml": 0}
        means = {"K1_ancova": [], "K1_dml": []}
        nfeas = []
        for r in range(reps):
            rng = np.random.default_rng([4242, r, WORLDS.index(w)])
            vals = np.array([seed_k1(simulate_items(w, 300, rng, structure=st, loss_shift=1.5), 1000 * r + s) for s in range(20)])
            nfeas.append(int(np.isfinite(vals[:, 0]).sum()))
            for k, col in (("K1_ancova", 0), ("K1_dml", 1)):
                v = vals[:, col][np.isfinite(vals[:, col])]
                if len(v) >= 3:
                    p = stats.ttest_1samp(v, 0.5, alternative="greater").pvalue
                    sup[k] += bool(p < 0.05 and v.mean() >= 0.55)
                    means[k].append(float(v.mean()))
        res[w] = {k: {"support_rate": sup[k] / reps, "mean": float(np.mean(means[k])) if means[k] else float("nan")} for k in sup}
        res[w]["median_feasible_seeds"] = float(np.median(nfeas))
        print(w, res[w], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        r = {}
        for f in sorted(os.listdir(OUT)):
            if f.startswith("k1partial_"):
                r.update(json.load(open(os.path.join(OUT, f))))
        r = {w: r[w] for w in WORLDS if w in r}
        json.dump({"reps": a.reps, "results": r}, open(os.path.join(OUT, "sim_k1_results.json"), "w"), indent=1)
        lines = ["# K1 definition check (simulation only; ~50% loss regime)", "",
                 f"{a.reps} replicate studies x 20 seeds x 300 items. Cell = mean AUROC / support rate (p<.05 and mean >= 0.55).", "",
                 "| world | K1_ancova | K1_dml (adopted) | median feasible seeds (of 20) |", "|---|---|---|---|"]
        for w, v in r.items():
            lines.append(f"| {w} | {v['K1_ancova']['mean']:.3f} / {v['K1_ancova']['support_rate']:.2f} | "
                         f"{v['K1_dml']['mean']:.3f} / {v['K1_dml']['support_rate']:.2f} | {v['median_feasible_seeds']:.0f} |")
        open(os.path.join(OUT, "sim_k1_results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    else:
        ws = WORLDS if a.worker is None else WORLDS[a.worker::a.n_workers]
        r = run(a.reps, ws)
        json.dump(r, open(os.path.join(OUT, f"k1partial_{a.worker if a.worker is not None else 'all'}.json"), "w"), indent=1)
