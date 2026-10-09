"""C15-R2 design: synthetic power / sensitivity analysis (NO model runs).

Part A: the A-stage W3 decoder (unpaired, n=40 contexts, d=2048, cross-fitted dual ridge, lambda=0.1*mean diag):
        null distribution of R^2 and the transported-signal size needed for R^2 >= 0.25.
Part B: proposed W3a (paired interventional footprint alignment): power vs transport heterogeneity and footprint
        estimation noise -- independent of context variance by construction.
Part C: binomial sensitivity for W0b' and W1-style forced-choice gates.
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(9400)
D = 2048
N = 40
FOLDS = np.arange(N) % 5
LAM = 0.1


def spectrum(alpha, d=D):
    lam = (np.arange(1, d + 1) ** (-alpha)).astype(float)
    return lam / lam.sum()          # total context variance = 1


def ridge_cv_r2(X, y):
    K = X @ X.T
    pred = np.zeros_like(y)
    for f in range(5):
        te = FOLDS == f
        tr = ~te
        Ktt, Ket = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
        rm = Ktt.mean(0)
        km = Ktt.mean()
        Kc = Ktt - rm[None] - rm[:, None] + km
        Kec = Ket - Ket.mean(1, keepdims=True) - rm[None] + km
        reg = LAM * np.diag(Kc).mean() + 1e-12
        my = y[tr].mean()
        a = np.linalg.solve(Kc + reg * np.eye(Kc.shape[0]), y[tr] - my)
        pred[te] = Kec @ a + my
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def part_a(reps=300):
    s = np.tile([-1.0, -0.5, 0.5, 1.0], N // 4)
    out = {}
    for alpha in (0.5, 1.0, 1.5):
        sd = np.sqrt(spectrum(alpha))
        res = {}
        for beta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
            r2 = []
            for _ in range(reps):
                C = rng.standard_normal((N, D)) * sd             # context variation (anisotropic)
                w = rng.standard_normal(D)
                w /= np.linalg.norm(w)
                X = C + beta * s[:, None] * w[None]              # beta = transported signal norm / total context SD
                r2.append(ridge_cv_r2(X, rng.permutation(s) if False else s))
            r2 = np.array(r2)
            res[beta] = {"mean_r2": float(r2.mean()), "p_r2_ge_0.25": float((r2 >= 0.25).mean()),
                         "q05": float(np.quantile(r2, 0.05)), "q95": float(np.quantile(r2, 0.95))}
        out[f"alpha={alpha}"] = res
    return out


def part_b(reps=150, n_ctx=40, n_f=40):
    """Paired interventional design (W3a). Concept k has a true downstream footprint direction f_k (unit; concepts
    share a common component with weight rho). Injection effect in context c (paired difference, so NO context
    variance): D_ck = phi*f_k + sqrt(1-phi^2)*g_k + gamma*e_ck, with g_k a route-specific non-footprint direction
    and e_ck context heterogeneity (isotropic, unit expected norm). Footprint estimate F_hat_k = f_k + kappa*nu_k,
    nu_k = mean of n_f unit-norm noise vectors. Per-unit specificity x_ck = cos(D_ck, F_hat_k) - mean_{k'!=k}
    cos(D_ck, F_hat_k'). Site test: one-sided t-test on the n_conc concept means (df = n_conc-1), alpha=0.05."""
    from scipy import stats
    out = {}
    for n_conc in (8, 16):
        for rho in (0.0, 0.5):
            for phi in (0.0, 0.05, 0.1, 0.2, 0.3):
                for gamma in (1.0, 3.0, 10.0):
                    for kappa in (1.0, 5.0):
                        rej = 0
                        means = []
                        for _ in range(reps):
                            common = rng.standard_normal(D); common /= np.linalg.norm(common)
                            U = rng.standard_normal((n_conc, D)); U /= np.linalg.norm(U, axis=1, keepdims=True)
                            F = math.sqrt(rho) * common[None] + math.sqrt(1 - rho) * U
                            F /= np.linalg.norm(F, axis=1, keepdims=True)
                            Fh = F + kappa * rng.standard_normal((n_conc, D)) / math.sqrt(D * n_f)
                            Fh /= np.linalg.norm(Fh, axis=1, keepdims=True)
                            cm = []
                            for k in range(n_conc):
                                g = rng.standard_normal(D); g -= (g @ F[k]) * F[k]; g /= np.linalg.norm(g)
                                base = phi * F[k] + math.sqrt(1 - phi ** 2) * g
                                Dc = base[None] + gamma * rng.standard_normal((n_ctx, D)) / math.sqrt(D)
                                Dc /= np.linalg.norm(Dc, axis=1, keepdims=True)
                                cos = Dc @ Fh.T                                   # [n_ctx, n_conc]
                                x = cos[:, k] - np.delete(cos, k, axis=1).mean(1)
                                cm.append(x.mean())
                            cm = np.array(cm)
                            t = cm.mean() / (cm.std(ddof=1) / math.sqrt(n_conc))
                            rej += stats.t.sf(t, n_conc - 1) < 0.05
                            means.append(cm.mean())
                        out[f"n_conc={n_conc},rho={rho},phi={phi},gamma={gamma},kappa={kappa}"] = {
                            "power": rej / reps, "mean_specificity": float(np.mean(means))}
    return out


def part_b_sign_test(reps=2000, n_ctx=40):
    """Per-context paired sign statistic: fraction of contexts with cos(D_c, F_hat_k) > max_{k'} cos(D_c, F_hat_k').
    Binomial power vs chance 1/n_conc for n_ctx contexts."""
    from math import comb
    out = {}
    for n_conc in (8, 10):
        p0 = 1.0 / n_conc
        # critical count at alpha=0.05 (one-sided)
        crit = next(k for k in range(n_ctx + 1) if sum(comb(n_ctx, j) * p0 ** j * (1 - p0) ** (n_ctx - j)
                                                      for j in range(k, n_ctx + 1)) <= 0.05)
        pw = {}
        for p1 in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            pw[p1] = sum(comb(n_ctx, j) * p1 ** j * (1 - p1) ** (n_ctx - j) for j in range(crit, n_ctx + 1))
        out[f"n_conc={n_conc}"] = {"chance": p0, "critical_count": crit, "power_by_true_rate": pw}
    return out


def part_c():
    from math import comb

    def p_ge(n, p, thr):
        k0 = math.ceil(thr * n - 1e-9)
        return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k0, n + 1))
    out = {"W0b_threshold_0.40": {}, "W1_threshold_0.30_n100": {}, "W2_threshold_0.25_n20": {}}
    for n in (40, 50, 60):
        out["W0b_threshold_0.40"][n] = {p: round(p_ge(n, p, 0.40), 3) for p in (0.3, 0.4, 0.5, 0.6)}
    out["W1_threshold_0.30_n100"] = {p: round(p_ge(100, p, 0.30), 3) for p in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)}
    out["W2_threshold_0.25_n20"] = {p: round(p_ge(20, p, 0.25), 3) for p in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)}
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("a", "all"):
        res["A_old_W3"] = part_a()
    if which in ("b", "all"):
        res["B_W3a_alignment"] = part_b()
        res["B_W3a_sign_test"] = part_b_sign_test()
    if which in ("c", "all"):
        res["C_binomial"] = part_c()
    print(json.dumps(res, indent=1))
