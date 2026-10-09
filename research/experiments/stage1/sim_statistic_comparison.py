"""SIMULATION-ONLY comparison of candidate primary statistics for the v3 within-target estimand (methods
validation; no model, store or monitor information is used).

For each simulated world (s1/sim_worlds.py) and each loss regime (loss_shift 0: ~20% of targeted items lose the answer;
loss_shift 1.5: ~50%, the regime required by the v3 outcome-diversity gate), R replicate studies of 20 seeds x 300
identically treated items: per seed compute each candidate; seed-level one-sided t-test on atanh(theta) against 0;
a study 'supports' the effect iff p < .05 AND mean theta >= MEI (0.10). Reported: mean theta, support rate.

Candidates:
  S1_delta_lin   change-score partial Spearman, linear rank adjustment (v1/v2 K3 style)
  S3_post_rcs_P  residualized-post, flexible (RCS) rank adjustment, one-sided susceptibility score P (memo v3 sec.4)
  S3_noState     residualized-post, flexible adjustment, pre covariates W only (no pre-state information)
  S9_dml_rank    residualized-post, flexible adjustment, SYMMETRIC cross-fitted pre-state adjustment (ranks)
  S9_dml_logitM  as S9 with the monitor score on the logit scale
  S9_gen         S9 + the generic post-change features (theta_gen)
Binary secondary (within X, propensity matching on pre-intervention variables only):
  K1_prob        AUROC of the raw probability decrease
  K1_ancova      AUROC of the baseline-adjusted monitor residual resid(rank M_post | RCS(rank M_pre))
Labels for S9 + S9_gen (A/B/C) as frozen in the protocol.
Usage: python sim_statistic_comparison.py [--reps 100]
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
from s1.sim_worlds import SCENARIOS, make_structure, pre_covariates, simulate_items  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "statistics", "v3_statistic_simulation")
MEI, MEI_AUC = 0.10, 0.55
N_SEEDS, N_ITEMS = 20, 300
KEYS = ["S1_delta_lin", "S3_post_rcs_P", "S3_noState", "S9_dml_rank", "S9_dml_logitM", "S9_gen", "K1_prob", "K1_ancova"]
SHIFTS = (0.0, 1.5)


def seed_stats(it, seed):
    W0 = pre_covariates(it)
    dC = it["C_post"] - it["C_pre"]
    P, r2_pre, _ = E.crossfit_ridge(np.column_stack([it["H_pre"], W0, it["C_pre"]]), dC, seed, strata=it["e"])
    _, r2_gen, _ = E.crossfit_ridge(it["G"], dC, seed + 1, strata=it["e"])
    W = pre_covariates(it, P)
    Gall = np.column_stack([it["G"], it["D"], it["F"]])
    a = (it["M_pre"], it["M_post"], it["C_pre"], it["C_post"])
    out = {
        "S1_delta_lin": E.theta_delta(*a, W=W, flexible=False),
        "S3_post_rcs_P": E.theta_post(*a, W=W, flexible=True),
        "S3_noState": E.theta_post(*a, W=W0, flexible=True),
        "S9_dml_rank": E.theta_post_dml(*a, W=W0, H_pre=it["H_pre"], seed=seed + 11, strata=it["e"])[0],
        "S9_dml_logitM": E.theta_post_dml(*a, W=W0, H_pre=it["H_pre"], seed=seed + 11, strata=it["e"], scale_M="logit")[0],
        "S9_gen": E.theta_post_dml(*a, W=W0, H_pre=it["H_pre"], seed=seed + 11, G=Gall, strata=it["e"])[0],
        "r2_pre": r2_pre, "r2_gen": r2_gen, "lost_frac": float((it["C_post"] < 0).mean()),
    }
    lost = it["C_post"] < 0
    pairs, smd = E.propensity_match(lost, np.column_stack([it["C_pre"], W0, P]), it["e"], seed + 5)
    ok = len(pairs) >= 50 and smd <= 0.10
    pi, pj = np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int)
    dprob = it["M_post"] - it["M_pre"]
    resid = E.residualize(E.rank01(it["M_post"]), E.rcs(E.rank01(it["M_pre"])))
    out["K1_prob"] = Mx.auroc(-dprob[pi], -dprob[pj]) if ok else np.nan
    out["K1_ancova"] = Mx.auroc(-resid[pi], -resid[pj]) if ok else np.nan
    return out


def decide(vals, auc=False):
    t = np.asarray(vals, float)
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return False, float("nan")
    if auc:
        p = stats.ttest_1samp(t, 0.5, alternative="greater").pvalue
        return bool(p < 0.05 and t.mean() >= MEI_AUC), float(t.mean())
    z = np.arctanh(np.clip(t, -0.999, 0.999))
    p = stats.ttest_1samp(z, 0.0, alternative="greater").pvalue
    return bool(p < 0.05 and t.mean() >= MEI), float(t.mean())


def run(reps, seed0=777, jobs=None):
    """jobs: optional list of (shift, scenario); each job's random streams depend only on (rep, scenario, shift),
    so results are identical whether jobs run in one process or in parallel workers."""
    structure = make_structure(np.random.default_rng(12345))
    res = {}
    for shift, scen in (jobs or [(sh, sc) for sh in SHIFTS for sc in SCENARIOS]):
        if True:
            agg = {k: {"support": 0, "mean": []} for k in KEYS}
            labels = {"A": 0, "B": 0, "C": 0}
            diag = {"r2_pre": [], "r2_gen": [], "lost_frac": []}
            for r in range(reps):
                rng = np.random.default_rng([seed0, r, SCENARIOS.index(scen), int(10 * shift)])
                per = [seed_stats(simulate_items(scen, N_ITEMS, rng, structure=structure, loss_shift=shift), 1000 * r + s)
                       for s in range(N_SEEDS)]
                for k in KEYS:
                    sup, m = decide([d[k] for d in per], auc=k.startswith("K1"))
                    agg[k]["support"] += sup
                    agg[k]["mean"].append(m)
                sp, _ = decide([d["S9_dml_rank"] for d in per])
                sg, _ = decide([d["S9_gen"] for d in per])
                labels["A" if (sp and sg) else ("B" if sp else "C")] += 1
                for k in diag:
                    diag[k] += [d[k] for d in per]
            key = f"shift{shift}|{scen}"
            res[key] = {k: {"support_rate": v["support"] / reps, "mean": float(np.nanmean(v["mean"]))} for k, v in agg.items()}
            res[key]["labels_S9"] = {k: v / reps for k, v in labels.items()}
            res[key].update({f"median_{k}": float(np.median(v)) for k, v in diag.items()})
            print(key, {k: (round(v["mean"], 3), v["support_rate"]) for k, v in res[key].items() if isinstance(v, dict) and "mean" in v},
                  res[key]["labels_S9"], flush=True)
    return res


def write(res, reps):
    os.makedirs(OUT, exist_ok=True)
    json.dump({"reps": reps, "n_seeds": N_SEEDS, "n_items": N_ITEMS, "MEI": MEI, "results": res},
              open(os.path.join(OUT, "sim_results.json"), "w"), indent=1)
    lines = ["# v3 primary-statistic simulation (methods validation only)", "",
             f"{reps} replicate studies x {N_SEEDS} seeds x {N_ITEMS} identically treated items per world and loss regime. "
             "Cell = mean / support rate (theta: one-sided seed-level t on atanh(theta), p<.05 AND mean >= 0.10; "
             "K1: AUROC t vs 0.5, p<.05 AND mean >= 0.55).", "",
             "| regime / world | " + " | ".join(KEYS) + " | labels S9 (A/B/C) | lost frac | R2_pre / R2_gen |",
             "|---|" + "---|" * (len(KEYS) + 3)]
    for key, v in res.items():
        lines.append(f"| {key} | " + " | ".join(f"{v[k]['mean']:.3f} / {v[k]['support_rate']:.2f}" for k in KEYS) +
                     f" | {v['labels_S9']['A']:.2f} / {v['labels_S9']['B']:.2f} / {v['labels_S9']['C']:.2f} | "
                     f"{v['median_lost_frac']:.2f} | {v['median_r2_pre']:.2f} / {v['median_r2_gen']:.2f} |")
    open(os.path.join(OUT, "sim_results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--worker", type=int, default=None, help="parallel worker index (writes a partial file)")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    all_jobs = [(sh, sc) for sh in SHIFTS for sc in SCENARIOS]
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        r = {}
        for f in sorted(os.listdir(OUT)):
            if f.startswith("partial_"):
                r.update(json.load(open(os.path.join(OUT, f))))
        order = [f"shift{sh}|{sc}" for sh, sc in all_jobs]
        write({k: r[k] for k in order if k in r}, a.reps)
    elif a.worker is not None:
        jobs = all_jobs[a.worker::a.n_workers]
        r = run(a.reps, jobs=jobs)
        json.dump(r, open(os.path.join(OUT, f"partial_{a.worker}.json"), "w"), indent=1)
    else:
        write(run(a.reps), a.reps)
