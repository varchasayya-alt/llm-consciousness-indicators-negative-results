"""SM-0 regret-gap check (exact arithmetic only; no training, no data, no models).

Pre-registered in research/memo/sm0_proof_reduction_memo.md, section 6 (committed before this was run).

Primary model M0 (single shift):
  - K contexts. The agent observes n Bernoulli trials in each of contexts 1 and 2, after an unseen shift.
  - The query is in unvisited context 3.
  - Exactly one shift occurs: the self is degraded with probability rho; otherwise the world becomes
    hard in a uniformly random context k (probability (1 - rho)/K each).
  - Success probability is a when nominal and b = a - delta when degraded. Self-degradation and world
    difficulty are indistinguishable within a context.
  - Query: attempt (utility 1 on success, 0 on failure) or opt out (utility c), with c ~ U[0, 1].
    For any predictor p_hat, E_c[utility | x] = p_hat * p* + (1 - p_hat^2) / 2, so regret is
    (p* - p_hat)^2 / 2.

Secondary model M0' (independent causes):
  - The self is degraded with probability rho; each context is independently hard with probability omega.
  - Success probability is q(s, w) = max(a - delta * (s + w), 0.02).

Policies:
  - BAYES: the exact posterior predictive p*.
  - LOCAL: ignores cross-context evidence and uses the prior predictive for context 3.
  - POOL: attribution-blind pooling, which assumes all observed contexts share context 3's degradation
    state and updates on the pooled data.
"""
import json
import math
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm0_regret_gap_results.json")


def pmf(x, n, p):
    return math.comb(n, x) * p ** x * (1.0 - p) ** (n - x)


def fixed_c_regret(pstar, phat, c):
    # Regret at a fixed outside value c: the loss from taking the wrong side of c.
    best = max(pstar, c)
    got = pstar if phat > c else c
    return best - got


def cell_m0(K, rho, a, delta, n):
    b = a - delta
    prior = {"S": rho, "W1": (1 - rho) / K, "W2": (1 - rho) / K, "W3": (1 - rho) / K,
             "Wo": (1 - rho) * (K - 3) / K}
    ps = {"S": (b, b, b), "W1": (b, a, a), "W2": (a, b, a), "W3": (a, a, b), "Wo": (a, a, a)}
    pi0 = rho + (1 - rho) / K                      # prior P(context 3 degraded)
    p_local = a - delta * pi0
    c_fix = p_local                                # secondary: threshold at the prior predictive
    V = Gl = Gp = Fl = Fp = 0.0
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            joint = {h: prior[h] * pmf(x1, n, ps[h][0]) * pmf(x2, n, ps[h][1]) for h in prior}
            px = sum(joint.values())
            if px <= 0.0:
                continue
            pstar = sum(joint[h] * ps[h][2] for h in joint) / px
            ld = pi0 * pmf(x1, n, b) * pmf(x2, n, b)
            ln = (1 - pi0) * pmf(x1, n, a) * pmf(x2, n, a)
            p_pool = a - delta * ld / (ld + ln)
            V += px * (1 + pstar ** 2) / 2
            Gl += px * (pstar - p_local) ** 2 / 2
            Gp += px * (pstar - p_pool) ** 2 / 2
            Fl += px * fixed_c_regret(pstar, p_local, c_fix)
            Fp += px * fixed_c_regret(pstar, p_pool, c_fix)
    return V, Gl, Gp, Fl, Fp


def cell_m0p(rho, a, delta, n, omega):
    def q(level):
        return max(a - delta * level, 0.02)
    Ps = {0: 1 - rho, 1: rho}
    Pw = {0: 1 - omega, 1: omega}
    lev_prior = {}
    for s in (0, 1):
        for w3 in (0, 1):
            lev_prior[s + w3] = lev_prior.get(s + w3, 0.0) + Ps[s] * Pw[w3]
    p_local = sum(lev_prior[l] * q(l) for l in lev_prior)
    c_fix = p_local
    V = Gl = Gp = Fl = Fp = 0.0
    for x1 in range(n + 1):
        for x2 in range(n + 1):
            num = den = 0.0
            for s in (0, 1):
                for w1 in (0, 1):
                    for w2 in (0, 1):
                        lik = Ps[s] * Pw[w1] * Pw[w2] * pmf(x1, n, q(s + w1)) * pmf(x2, n, q(s + w2))
                        pred3 = sum(Pw[w3] * q(s + w3) for w3 in (0, 1))
                        num += lik * pred3
                        den += lik
            if den <= 0.0:
                continue
            pstar = num / den
            pool_post = {l: lev_prior[l] * pmf(x1, n, q(l)) * pmf(x2, n, q(l)) for l in lev_prior}
            z = sum(pool_post.values())
            p_pool = sum(pool_post[l] * q(l) for l in pool_post) / z
            V += den * (1 + pstar ** 2) / 2
            Gl += den * (pstar - p_local) ** 2 / 2
            Gp += den * (pstar - p_pool) ** 2 / 2
            Fl += den * fixed_c_regret(pstar, p_local, c_fix)
            Fp += den * fixed_c_regret(pstar, p_pool, c_fix)
    return V, Gl, Gp, Fl, Fp


def summarize(rows, key):
    vals = sorted(r[key] for r in rows)
    m = len(vals)
    return {"min": vals[0], "q25": vals[m // 4], "median": statistics.median(vals),
            "q75": vals[(3 * m) // 4], "max": vals[-1],
            "frac_ge_0.02": sum(v >= 0.02 for v in vals) / m, "n_cells": m}


def marginal_medians(rows, factors, key):
    out = {}
    for f in factors:
        levels = sorted({r[f] for r in rows})
        out[f] = {str(l): statistics.median(r[key] for r in rows if r[f] == l) for l in levels}
    return out


def row(V, Gl, Gp, Fl, Fp, **params):
    r = dict(params)
    r.update({"V_star": V, "R_transfer": Gl / V, "R_attrib": Gp / V, "R_blind": min(Gl, Gp) / V,
              "Rfix_blind": min(Fl, Fp) / V})
    return r


def main():
    m0 = []
    for K in (3, 6, 12):
        for rho in (0.1, 0.25, 0.5, 0.75):
            for a in (0.9, 0.75):
                for delta in (0.1, 0.2, 0.4):
                    for n in (1, 3, 10, 30):
                        m0.append(row(*cell_m0(K, rho, a, delta, n), K=K, rho=rho, a=a, delta=delta, n=n))
    m0p = []
    for rho in (0.1, 0.25, 0.5, 0.75):
        for a in (0.9, 0.75):
            for delta in (0.1, 0.2, 0.4):
                for n in (1, 3, 10, 30):
                    for omega in (0.1, 0.3):
                        m0p.append(row(*cell_m0p(rho, a, delta, n, omega),
                                       rho=rho, a=a, delta=delta, n=n, omega=omega))
    s = summarize(m0, "R_blind")
    k4_declared_fires = s["max"] < 0.02
    if k4_declared_fires:
        label = "FAIL (K4 fires)"
    elif s["median"] >= 0.02:
        label = "PASS-robust"
    else:
        label = "PASS-weak (regime-dependent)"
    res = {
        "K4": {"declared_rule": "fires iff max over M0 grid of R_blind < 0.02",
               "robust_rule": "PASS-robust iff median over M0 grid of R_blind >= 0.02",
               "declared_fires": k4_declared_fires, "label": label},
        "M0": {"R_blind": s, "R_transfer": summarize(m0, "R_transfer"), "R_attrib": summarize(m0, "R_attrib"),
               "Rfix_blind_secondary": summarize(m0, "Rfix_blind"),
               "marginal_median_R_blind": marginal_medians(m0, ["K", "rho", "a", "delta", "n"], "R_blind"),
               "top5_R_blind": sorted(m0, key=lambda r: -r["R_blind"])[:5]},
        "M0prime_secondary": {"R_blind": summarize(m0p, "R_blind"), "R_transfer": summarize(m0p, "R_transfer"),
                              "R_attrib": summarize(m0p, "R_attrib"),
                              "Rfix_blind_secondary": summarize(m0p, "Rfix_blind"),
                              "marginal_median_R_blind": marginal_medians(m0p, ["rho", "a", "delta", "n", "omega"],
                                                                          "R_blind")},
        "cells": {"M0": m0, "M0prime": m0p},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("K4",)}, indent=1))
    print("M0 R_blind", json.dumps(s))
    print("M0 R_transfer", json.dumps(res["M0"]["R_transfer"]))
    print("M0 R_attrib", json.dumps(res["M0"]["R_attrib"]))
    print("M0 Rfix_blind (secondary)", json.dumps(res["M0"]["Rfix_blind_secondary"]))
    print("M0 marginal medians", json.dumps(res["M0"]["marginal_median_R_blind"]))
    print("M0 top5", json.dumps([{k: r[k] for k in ("K", "rho", "a", "delta", "n", "R_blind")}
                                 for r in res["M0"]["top5_R_blind"]]))
    print("M0' R_blind", json.dumps(res["M0prime_secondary"]["R_blind"]))
    print("M0' R_transfer", json.dumps(res["M0prime_secondary"]["R_transfer"]))
    print("M0' R_attrib", json.dumps(res["M0prime_secondary"]["R_attrib"]))
    print("M0' Rfix_blind (secondary)", json.dumps(res["M0prime_secondary"]["Rfix_blind_secondary"]))
    print("M0' marginal medians", json.dumps(res["M0prime_secondary"]["marginal_median_R_blind"]))


if __name__ == "__main__":
    main()
