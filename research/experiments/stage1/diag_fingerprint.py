"""Methods diagnostic (calibration only; store states only, NO monitor): which generic features separate
X_lost from displacement-matched Y under a given T-FORGET config? Per-feature univariate AUROC.

Usage: python diag_fingerprint.py --seed 9001 --lr 1e-4 --steps 200 --lam 1.0 [--gamma 10]
Writes results/raw/calibration/diag_fingerprint_<seed>_<tag>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_calibration as rc  # noqa: E402
from s1.config import derive_seed, load_config  # noqa: E402
from s1.diagnostics import generic_features  # noqa: E402
from s1.interventions import forget_qc, forget_with_sham  # noqa: E402
from s1.matching import caliper_match, log_profile  # noqa: E402
from s1.metrics import auroc  # noqa: E402

SITES = [f"L{l}-{p}" for l in range(5) for p in ("s2", "A")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=10.0)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "4")))
    cfg = load_config(os.path.join(HERE, "stage1_config.yaml"))
    c = rc.context(cfg, a.seed)
    fcfg = dict(cfg["interventions"]["T_FORGET"], lr=a.lr, steps=a.steps, lambda_retain=a.lam, gamma=a.gamma)
    t0 = time.perf_counter()
    model, _ = forget_with_sham(c["A"], c["world"], c["sets"]["X"], c["sets"]["Y"], fcfg, c["sig"], derive_seed(a.seed, "forget"))
    t1 = time.perf_counter()
    qc, aux = forget_qc(c["A"], model, c["world"], c["sets"], c["sig"], fcfg["qc"], [1.0], derive_seed(a.seed, "matchF"), c["flu_sd"])
    t2 = time.perf_counter()
    lost, ycor = aux["lostX"], aux["postY_correct"]
    pairs = caliper_match(log_profile(aux["dX"])[lost], log_profile(aux["dY"])[ycor], 1.0, derive_seed(a.seed, "matchF"))
    xi = np.nonzero(lost)[0][[p[0] for p in pairs]]
    yi = np.nonzero(ycor)[0][[p[1] for p in pairs]]
    gX = generic_features(aux["pre_X"]["states"], aux["post_X"]["states"], c["mu"], c["sig"])
    gY = generic_features(aux["pre_Y"]["states"], aux["post_Y"]["states"], c["mu"], c["sig"])
    G2x, G2y = gX["G2"][xi], gY["G2"][yi]
    names = [f"logdisp:{s}" for s in SITES] + [f"cos(delta,pre):{s}" for s in SITES] + [f"relnorm:{s}" for s in SITES]
    per = {n: float(auroc(G2x[:, k], G2y[:, k])) for k, n in enumerate(names)}
    G3x, G3y = gX["G3"][xi], gY["G3"][yi]
    per.update({f"abnormality:{s}": float(auroc(G3x[:, k], G3y[:, k])) for k, s in enumerate(SITES)})
    means = {n: [float(G2x[:, k].mean()), float(G2y[:, k].mean())] for k, n in enumerate(names)}
    # same procedure, different outcome: X_retained (forget objective applied, answer kept) vs Y and vs X_lost
    xr = np.nonzero(~lost)[0]
    G2xr = gX["G2"][xr]
    G2xl = gX["G2"][np.nonzero(lost)[0]]
    G2yc = gY["G2"][np.nonzero(ycor)[0]]
    per_ret = {}
    if len(xr) >= 10:
        for k, n in enumerate(names):
            per_ret[n] = {"Xretained_vs_Y": float(auroc(G2xr[:, k], G2yc[:, k])),
                          "Xlost_vs_Xretained": float(auroc(G2xl[:, k], G2xr[:, k]))}
    from s1.metrics import partial_spearman
    dm = aux["post_X"]["margin"] - aux["pre_X"]["margin"]
    k4 = names.index("cos(delta,pre):L4-A")
    rho = float(partial_spearman(gX["G2"][:, k4], dm, np.zeros((len(dm), 0))))
    # output-distribution change of the same pairs (first-order readout reference)
    ent = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum(-1)
    ref = {"entropy_post_Xlost_mean": float(ent(aux["post_X"]["probs"][xi]).mean()),
           "entropy_post_Y_mean": float(ent(aux["post_Y"]["probs"][yi]).mean()),
           "auroc_entropy_post": float(auroc(ent(aux["post_X"]["probs"][xi]), ent(aux["post_Y"]["probs"][yi])))}
    out = {"seed": a.seed, "config": {"lr": a.lr, "steps": a.steps, "lambda_retain": a.lam, "gamma": a.gamma},
           "n_pairs": len(pairs), "per_feature_auroc": per, "means_Xlost_Y": means, "output_reference": ref,
           "n_X_retained": int(len(xr)), "per_feature_auroc_Xretained": per_ret,
           "spearman_within_X_cosL4A_vs_dmargin": rho,
           "timing_s": {"forget": t1 - t0, "qc": t2 - t1}}
    tag = f"lr{a.lr}_st{a.steps}_lam{a.lam}_g{a.gamma}"
    path = os.path.join(rc.OUT, f"diag_fingerprint_{a.seed}_{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    top = sorted(per.items(), key=lambda kv: -abs(kv[1] - 0.5))[:14]
    for k, v in top:
        print(f"{k:28s} AUROC {v:.3f}  means X_lost/Y {means.get(k, ['', ''])}")
    print(ref, out["timing_s"])
    for n in ("cos(delta,pre):L4-A", "cos(delta,pre):L3-A", "relnorm:L3-A", "logdisp:L4-A", "logdisp:L3-A"):
        if n in per_ret:
            print(f"{n:24s} Xlost-vs-Y {per[n]:.3f} | Xret-vs-Y {per_ret[n]['Xretained_vs_Y']:.3f} | Xlost-vs-Xret {per_ret[n]['Xlost_vs_Xretained']:.3f}")
    print("n_X_retained", len(xr), "spearman within X cos(L4-A) vs dmargin", round(rho, 3))


if __name__ == "__main__":
    main()
