"""Stage-1 power / sensitivity analysis (simulation only -- no experimental data).

Experimental unit = store seed. Seed-level statistics are tested with one-sample /
paired t-tests (one-sided for directional hypotheses) and TOST for equivalence.

Outputs: research/results/statistics/stage1_power_sensitivity.json
"""
import json
import math
import os

import numpy as np

rng = np.random.default_rng(20261001)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "statistics", "stage1_power_sensitivity.json")
NSIM = 40000


def t_crit(n, alpha_one_sided):
    null = rng.standard_normal((200000, n))
    t = null.mean(1) / (null.std(1, ddof=1) / math.sqrt(n))
    return float(np.quantile(t, 1 - alpha_one_sided))


def power_one_sided(n, dz, crit):
    x = rng.standard_normal((NSIM, n)) + dz
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(n))
    return float((t > crit).mean())


def mde(n, alpha, target_power):
    crit = t_crit(n, alpha)
    lo, hi = 0.05, 3.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if power_one_sided(n, mid, crit) >= target_power:
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


def tost_power(n, sd, bound, true_diff, alpha):
    crit = t_crit(n, alpha)
    x = rng.normal(true_diff, sd, (NSIM, n))
    m, se = x.mean(1), x.std(1, ddof=1) / math.sqrt(n)
    t_low, t_high = (m + bound) / se, (bound - m) / se
    return round(float(((t_low > crit) & (t_high > crit)).mean()), 3)


def hanley_mcneil_se(auc, n1, n2):
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))


res = {"note": "simulation only; experimental unit = store seed", "mde_dz": {}, "tost_power": {}, "within_seed_auc_se": {}}
alphas = {"one_sided_0.05": 0.05, "holm_worst_of_6_(0.0083)": 0.05 / 6, "bonf_of_4_(0.0125)": 0.0125}
for n in (10, 15, 20, 30):
    for name, a in alphas.items():
        for pw in (0.8, 0.9):
            res["mde_dz"][f"n={n}|{name}|power={pw}"] = mde(n, a, pw)
for sd in (0.01, 0.02, 0.03, 0.05):
    for a_name, a in (("alpha=0.05", 0.05), ("alpha=0.0083", 0.05 / 6)):
        res["tost_power"][f"n=20|bound=0.05|true_diff=0|sd={sd}|{a_name}"] = tost_power(20, sd, 0.05, 0.0, a)
        res["tost_power"][f"n=20|bound=0.05|true_diff=0.02|sd={sd}|{a_name}"] = tost_power(20, sd, 0.05, 0.02, a)
for auc in (0.55, 0.6, 0.7, 0.8, 0.9):
    for n1, n2 in ((150, 300), (240, 300), (100, 150)):
        res["within_seed_auc_se"][f"auc={auc}|n_pos={n1}|n_neg={n2}"] = round(hanley_mcneil_se(auc, n1, n2), 4)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
for k, v in res.items():
    if isinstance(v, dict):
        print(k)
        for kk, vv in v.items():
            print("  ", kk, vv)
print("wrote", OUT)
