"""Small statistics helpers: cross-fitted dual ridge R^2, bootstraps, iso-impact interpolation."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import config as C


def kernel(X):
    X = X.double()
    return X @ X.T


def ridge_cv_r2_kernel(K, y, folds, lam=C.W3_RIDGE_LAMBDA):
    """Cross-fitted R^2 of a dual ridge regression (train-centred features) from a linear kernel K [n, n]."""
    y = y.double()
    pred = torch.zeros_like(y)
    for f in torch.unique(folds):
        te = folds == f
        tr = ~te
        Ktt, Ket = K[tr][:, tr], K[te][:, tr]
        rm = Ktt.mean(0)                     # mean_t K[t, j] over train t
        km = Ktt.mean()
        Kc = Ktt - rm[None, :] - rm[:, None] + km
        Kec = Ket - Ket.mean(1, keepdim=True) - rm[None, :] + km
        reg = lam * float(torch.diagonal(Kc).mean()) + 1e-9
        my = y[tr].mean()
        a = torch.linalg.solve(Kc + reg * torch.eye(Kc.shape[0], dtype=Kc.dtype), y[tr] - my)
        pred[te] = Kec @ a + my
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def ridge_cv_r2(X, y, folds, lam=C.W3_RIDGE_LAMBDA):
    """Cross-fitted R^2 of a dual ridge regression of y [n] on X [n, d]. `folds` [n] ints (group labels)."""
    return ridge_cv_r2_kernel(kernel(X), y, folds, lam)


def fold_ids(n, k=C.W3_FOLDS):
    return torch.arange(n) % k


def boot_ratio_ci(num, den, n_boot, seed, alpha=0.05):
    """Paired percentile CI for mean(num)/mean(den) over units (arrays of equal length)."""
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d = den[i].mean()
        stats.append(num[i].mean() / d if d > 0 else math.inf)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def iso_scale(imp_grid, scales, imp_target):
    """Smallest random-arm scale reaching the J-arm impairment, linear in log(scale) between grid points.

    Returns (s_star, status) with status in {'interpolated', 'at_grid_min_lower_bound', 'unreached_upper_bound'}.
    """
    imp = list(imp_grid)
    if imp[0] >= imp_target:
        return scales[0], "at_grid_min_lower_bound"
    for i in range(1, len(scales)):
        if imp[i] >= imp_target:
            a, b = imp[i - 1], imp[i]
            t = 0.0 if b == a else (imp_target - a) / (b - a)
            ls = math.log(scales[i - 1]) + t * (math.log(scales[i]) - math.log(scales[i - 1]))
            return math.exp(ls), "interpolated"
    return scales[-1], "unreached_upper_bound"


def interp_in_log_scale(values_by_scale, scales, s):
    """Linear interpolation in log(scale) of per-unit values [n_scales, n_units] at scale s."""
    V = np.asarray(values_by_scale, float)
    ls = np.log(np.asarray(scales, float))
    x = math.log(s)
    if x <= ls[0]:
        return V[0]
    if x >= ls[-1]:
        return V[-1]
    j = int(np.searchsorted(ls, x))
    t = (x - ls[j - 1]) / (ls[j] - ls[j - 1])
    return (1 - t) * V[j - 1] + t * V[j]
