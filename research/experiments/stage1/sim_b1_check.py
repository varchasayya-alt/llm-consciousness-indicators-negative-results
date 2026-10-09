"""SIMULATION-ONLY verification (no model information) that the frozen v3 primary statistic (S9 / theta_post_dml)
and the A/B/C labels behave correctly in a B1-like data-generating process (PI 2026-10-03, item 14). This does NOT
reselect the statistic; if a genuine invalidity appeared, the protocol says stop and report.

B1-like world, per store seed, 300 targeted items ALL receiving identical full-slot deletion:
  C_pre   near ceiling (memory present);  backup b = sqrt(rho)*z_vis + sqrt(1-rho)*z_hid (+ exposure effect);
  z_vis   encoded in the pre-intervention states H_pre (noisy linear code); z_hid revealed only after deletion;
  C_post  = graded function of b (about half lost);
  generic post-change features: a constant deletion signature + item-varying intensity kappa (weakly related to b)
  plus noise; monitors (logit scale, persistent item bias u):
    H3            reads current competence C_post
    vis_state     reads the pre-visible backup AS ENCODED IN THE PRE-STATES (the information that upstream,
                  deletion-invariant read sites carry)                                           -> expect C
    vis_latent    STRESS TEST: reads the latent z_vis exactly (more precisely than the pre-states encode it);
                  by the estimand's definition this is partly post-intervention information (run 1 had only this)
    cue           reads the deletion signature/intensity kappa (H2-like)                        -> expect C or B, never A
    null          no change beyond a global shift                                                -> expect C
rho in {0.3, 0.7}. 50 replicate studies x 20 seeds. Writes results/statistics/v4_b1_simulation.md/json.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s1 import estimands as E  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "statistics")
MONITORS = ("H3", "vis_state", "vis_latent", "cue", "null")


def sim_seed(kind, rho, rng, d_h=40, d_g=16):
    n = 300
    e = (rng.random(n) < 0.5).astype(float)
    rel = rng.integers(0, 4, n)
    C_pre = rng.normal(7.0, 0.6, n)
    z_vis, z_hid = rng.normal(0, 1, n), rng.normal(0, 1, n)
    b = np.sqrt(rho) * z_vis + np.sqrt(1 - rho) * z_hid + 0.3 * (e - 0.5)
    C_post = 3.0 * b + rng.normal(0, 0.3, n)                      # ~half below 0
    a_v = rng.normal(0, 0.5, d_h)
    H = np.outer(z_vis, a_v) + np.outer(C_pre - 7, rng.normal(0, 0.2, d_h)) + rng.normal(0, 1.0, (n, d_h))
    kappa = 0.3 * b + rng.normal(0, 1, n)                         # intensity weakly related to backup
    G = 1.0 + np.outer(kappa, rng.uniform(0.3, 1.0, d_g)) + rng.normal(0, 0.5, (n, d_g))
    u = rng.normal(0, 0.6, n)
    lM_pre = 2.5 + 0.5 * (C_pre - 7) + 0.3 * e + u + rng.normal(0, 0.2, n)
    nz = rng.normal(0, 0.3, n)
    if kind == "H3":
        lM_post = 0.6 * C_post + 0.3 * e + u + nz
    elif kind == "vis_state":
        s_vis = H @ a_v / (a_v @ a_v)
        lM_post = lM_pre - 2.0 + 0.8 * (s_vis - s_vis.mean()) / s_vis.std() + nz
    elif kind == "vis_latent":
        lM_post = lM_pre - 2.0 + 0.8 * z_vis + nz
    elif kind == "cue":
        lM_post = lM_pre - 2.0 - 0.8 * (kappa - kappa.mean()) / kappa.std() + nz
    else:
        lM_post = lM_pre - 2.0 + nz
    sig = lambda x: 1 / (1 + np.exp(-x))
    W0 = np.column_stack([-np.log1p(31 * np.exp(-C_pre)), e, rng.normal(0, 1, n) + e, np.eye(4)[rel][:, 1:]])
    return sig(lM_pre), sig(lM_post), C_pre, C_post, W0, H, G, e


def decide(t):
    t = np.asarray(t)
    z = np.arctanh(np.clip(t, -0.999, 0.999))
    p = stats.ttest_1samp(z, 0.0, alternative="greater").pvalue
    return bool(p < 0.05 and t.mean() >= 0.10), float(t.mean())


def main(reps=50):
    res = {}
    for rho in (0.3, 0.7):
        for kind in MONITORS:
            lab = {"A": 0, "B": 0, "C": 0}
            mp, mg = [], []
            for r in range(reps):
                rng = np.random.default_rng([77, r, MONITORS.index(kind), int(rho * 10)])
                tp, tg = [], []
                for s in range(20):
                    Mp, Mq, Cp, Cq, W0, H, G, e = sim_seed(kind, rho, rng)
                    tp.append(E.theta_post_dml(Mp, Mq, Cp, Cq, W0, H, 1000 * r + s, strata=e)[0])
                    tg.append(E.theta_post_dml(Mp, Mq, Cp, Cq, W0, H, 1000 * r + s, G=G, strata=e)[0])
                sp, m1 = decide(tp)
                sg, m2 = decide(tg)
                lab["A" if (sp and sg) else ("B" if sp else "C")] += 1
                mp.append(m1)
                mg.append(m2)
            key = f"rho{rho}|{kind}"
            res[key] = {"labels": {k: v / reps for k, v in lab.items()}, "mean_theta_pre": float(np.mean(mp)),
                        "mean_theta_gen": float(np.mean(mg))}
            print(key, res[key], flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"reps": reps, "results": res}, open(os.path.join(OUT, "v4_b1_simulation.json"), "w"), indent=1)
    lines = ["# v4 B1-world verification of the frozen primary statistic (simulation only)", "",
             f"{reps} replicate studies x 20 seeds x 300 identically deleted items. rho = share of backup visible before deletion.", "",
             "| world | mean theta_pre | mean theta_gen | labels A / B / C |", "|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v['mean_theta_pre']:.3f} | {v['mean_theta_gen']:.3f} | "
                     f"{v['labels']['A']:.2f} / {v['labels']['B']:.2f} / {v['labels']['C']:.2f} |")
    open(os.path.join(OUT, "v4_b1_simulation.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
