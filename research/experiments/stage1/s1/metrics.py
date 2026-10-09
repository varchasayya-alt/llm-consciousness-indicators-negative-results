"""Statistics used by the frozen analysis. All functions are deterministic and unit-tested."""
import numpy as np
from scipy import stats


def auroc(pos, neg):
    """Mann-Whitney AUROC = P(pos > neg) + 0.5 P(pos == neg). Returns nan if either group is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = stats.rankdata(allv)          # average ranks -> ties count 0.5
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def dprime_from_auc(auc, clip=(0.001, 0.999)):
    a = np.clip(np.asarray(auc, float), *clip)
    return np.sqrt(2) * stats.norm.ppf(a)


def logit(p, clip=8.0):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p) - np.log1p(-p), -clip, clip)


def partial_spearman(y, x, covariates):
    """Partial Spearman correlation of y and x controlling for covariates.

    All variables are rank-transformed; y-ranks and x-ranks are residualised on covariate ranks
    (with intercept) by least squares; returns the Pearson correlation of the residuals.
    """
    y = stats.rankdata(np.asarray(y, float))
    x = stats.rankdata(np.asarray(x, float))
    C = np.asarray(covariates, float)
    if C.ndim == 1:
        C = C[:, None]
    Cr = np.column_stack([np.ones(len(y))] + [stats.rankdata(C[:, j]) for j in range(C.shape[1])])
    by = np.linalg.lstsq(Cr, y, rcond=None)[0]
    bx = np.linalg.lstsq(Cr, x, rcond=None)[0]
    ry, rx = y - Cr @ by, x - Cr @ bx
    if ry.std() == 0 or rx.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def one_sample_t(values, mu0=0.0, alternative="greater"):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or np.ptp(v) == 0:            # degenerate (constant) sample: deterministic p, never NaN
        m = float(v.mean()) if len(v) else float("nan")
        if alternative == "greater":
            p = 0.0 if (len(v) >= 2 and m > mu0) else 1.0
        elif alternative == "less":
            p = 0.0 if (len(v) >= 2 and m < mu0) else 1.0
        else:
            p = 0.0 if (len(v) >= 2 and m != mu0) else 1.0
        return {"n": int(len(v)), "mean": m, "sd": 0.0, "t": float("nan"), "p": p}
    res = stats.ttest_1samp(v, mu0, alternative=alternative)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "t": float(res.statistic), "p": float(res.pvalue)}


def wilcoxon_signed(values, mu0=0.0, alternative="greater"):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)] - mu0
    if np.all(v == 0):
        return {"p": 1.0}
    res = stats.wilcoxon(v, alternative=alternative)
    return {"p": float(res.pvalue)}


def tost(values, bound, mu0=0.0):
    """Two one-sided tests for equivalence of mean(values) to mu0 within +-bound.
    Returns p = max(p_lower, p_upper); equivalence is declared if p < alpha."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    p_low = stats.ttest_1samp(v, mu0 - bound, alternative="greater").pvalue
    p_up = stats.ttest_1samp(v, mu0 + bound, alternative="less").pvalue
    return {"n": int(len(v)), "mean": float(v.mean()), "p_lower": float(p_low), "p_upper": float(p_up),
            "p": float(max(p_low, p_up)), "bound": bound}


def bootstrap_ci(values, n_boot=10000, seed=0, level=0.95):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def holm(pvalues, alpha=0.05):
    """Holm step-down. Returns (adjusted_p list, reject list) in the input order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist(), (adj < alpha).tolist()


def ece(probs, labels, n_bins=15):
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def brier(probs, labels):
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def auprc(pos, neg):
    scores = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(1, y.sum()))
