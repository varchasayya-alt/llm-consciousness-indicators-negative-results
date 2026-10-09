"""Stage-1 v3 estimands and nuisance models (store-side quantities only; no monitor is trained here).

Primary estimand (decisions D36-D37; selected by the simulation-only comparison in sim_statistic_comparison.py):
  theta_pre  = per-seed partial association between post-intervention monitor score and post-intervention
               competence among identically treated items, adjusting ONLY for pre-intervention information:
               corr( resid(rank M_post | B), resid(rank C_post | B) ),
               B = [RCS(rank C_pre), RCS(rank M_pre), pre covariates W (incl. susceptibility score P)].
  theta_gen  = the same with B additionally containing the pre-declared generic post-change features
               (robustness / H2 diagnostic; potentially post-treatment, never the sole definition of the effect).
  theta_delta= descriptive change-score version: partial Spearman(dM, dC | linear ranks of pre covariates).
RCS = restricted cubic spline with 5 knots at fixed quantiles (.05, .275, .5, .725, .95).
"""
import numpy as np
from scipy import stats

KNOT_Q = (0.05, 0.275, 0.5, 0.725, 0.95)
RIDGE_GRID = 10.0 ** np.arange(-2, 7)          # D49: widened (p > n regime) and selected by inner CV
FOLDS = 5


def rank01(x):
    x = np.asarray(x, float)
    return (stats.rankdata(x) - 0.5) / len(x)


def rcs(x, knot_q=KNOT_Q):
    """Restricted cubic spline basis (linear tails) -> [n, K-1] columns (first column = x)."""
    x = np.asarray(x, float)
    k = np.quantile(x, knot_q)
    k = np.maximum.accumulate(k + np.arange(len(k)) * 1e-9)
    K = len(k)
    scale = (k[-1] - k[0]) ** 2 + 1e-12
    p3 = lambda u: np.maximum(u, 0.0) ** 3
    cols = [x]
    for j in range(K - 2):
        cols.append((p3(x - k[j]) - p3(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + p3(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / scale)
    return np.column_stack(cols)


def residualize(y, C):
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y))] + ([np.asarray(C, float)] if C is not None and np.size(C) else []))
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return y - A @ beta


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _covmat(cols):
    cols = [np.asarray(c, float).reshape(len(c), -1) for c in cols if c is not None and np.size(c)]
    return np.column_stack(cols) if cols else None


def theta_post(M_pre, M_post, C_pre, C_post, W=None, G=None, flexible=True, rank=True):
    """Residualized-post (ANCOVA-style) association. W: pre covariates [n,k]; G: generic post features (theta_gen)."""
    tr = rank01 if rank else (lambda v: np.asarray(v, float))
    base = [rcs(tr(C_pre)), rcs(tr(M_pre))] if flexible else [tr(C_pre), tr(M_pre)]
    extra = [] if W is None else [np.column_stack([tr(c) if len(np.unique(c)) > 2 else c for c in np.asarray(W, float).T])]
    gen = [] if G is None else [np.column_stack([tr(c) for c in np.asarray(G, float).T])]
    B = _covmat(base + extra + gen)
    return _corr(residualize(tr(M_post), B), residualize(tr(C_post), B))


def theta_post_separate(M_pre, M_post, C_pre, C_post, W=None):
    """Variant: residualize C_post on its own baseline (+W) and M_post on its own baseline (+W) separately."""
    Wr = [] if W is None else [np.column_stack([rank01(c) if len(np.unique(c)) > 2 else c for c in np.asarray(W, float).T])]
    rC = residualize(rank01(C_post), _covmat([rcs(rank01(C_pre))] + Wr))
    rM = residualize(rank01(M_post), _covmat([rcs(rank01(M_pre))] + Wr))
    return _corr(rM, rC)


def theta_delta(M_pre, M_post, C_pre, C_post, W=None, flexible=False):
    """Change-score partial Spearman (v1/v2 style). Descriptive in v3."""
    dM, dC = np.asarray(M_post) - np.asarray(M_pre), np.asarray(C_post) - np.asarray(C_pre)
    base = [rcs(rank01(C_pre)), rcs(rank01(M_pre))] if flexible else [rank01(C_pre), rank01(M_pre)]
    extra = [] if W is None else [np.column_stack([rank01(c) if len(np.unique(c)) > 2 else c for c in np.asarray(W, float).T])]
    B = _covmat(base + extra)
    return _corr(residualize(rank01(dM), B), residualize(rank01(dC), B))


# ---------------------------------------------------------------- nuisance models (cross-fitted)
def _folds(n, seed, strata=None):
    rng = np.random.default_rng(seed)
    f = np.empty(n, int)
    groups = [np.arange(n)] if strata is None else [np.nonzero(strata == s)[0] for s in np.unique(strata)]
    for g in groups:
        f[rng.permutation(g)] = np.arange(len(g)) % FOLDS
    return f


def _svd(A):
    """Thin SVD. On the rare LAPACK gesdd non-convergence (D66) the same decomposition is computed with the slower,
    more robust gesvd driver; results are unchanged whenever gesdd converges."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        import scipy.linalg
        return scipy.linalg.svd(A, full_matrices=False, lapack_driver="gesvd")


def _ridge_path(A, yc, Bte, grid):
    """Ridge predictions for every lambda in grid (A standardized train design, yc centred target)."""
    U, s, Vt = _svd(A)
    Uy = U.T @ yc
    return [Bte @ (Vt.T @ ((s / (s ** 2 + lam)) * Uy)) for lam in grid]


def ridge_cv_fit_predict(Xtr, ytr, Xte, grid=RIDGE_GRID, inner_folds=5):
    """Standardize on train; ridge with lambda chosen by INNER 5-fold cross-validation on the training fold over a
    fixed log grid (D49: replaces generalized cross-validation, which selects near-interpolating lambdas when
    features outnumber items and then extrapolates wildly on held-out folds)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    n = len(ytr)
    fold = np.arange(n) % inner_folds          # deterministic (outer folds are already randomised)
    err = np.zeros(len(grid))
    for k in range(inner_folds):
        tr, te = fold != k, fold == k
        mi, si = A[tr].mean(0), A[tr].std(0) + 1e-9
        preds = _ridge_path((A[tr] - mi) / si, ytr[tr] - ytr[tr].mean(), (A[te] - mi) / si, grid)
        err += np.array([((ytr[te] - ytr[tr].mean() - p_) ** 2).sum() for p_ in preds])
    best_l = grid[int(np.argmin(err))]
    pred = _ridge_path(A, ytr - ytr.mean(), B, [best_l])[0] + ytr.mean()
    return pred, float(best_l)


def crossfit_ridge(X, y, seed, strata=None):
    """Out-of-fold ridge predictions and cross-fitted R^2 (1 - SSE/SST)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    f = _folds(len(y), seed, strata)
    pred = np.empty(len(y))
    lams = []
    for k in range(FOLDS):
        te = f == k
        pred[te], lam = ridge_cv_fit_predict(X[~te], y[~te], X[te])
        lams.append(lam)
    r2 = 1.0 - float(((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return pred, r2, lams


def crossfit_logistic_scores(Xpos, Xneg, seed, lam=1.0):
    """Out-of-fold logits of an L2 logistic classifier (s1/diagnostics spec) for pos (targeted) vs neg items.
    Returns (scores_pos, scores_neg)."""
    from .diagnostics import _fit_logreg
    X = np.concatenate([Xpos, Xneg]).astype(np.float64)
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    f = _folds(len(y), seed, strata=y)
    out = np.empty(len(y))
    for k in range(FOLDS):
        te = f == k
        m, s = X[~te].mean(0), X[~te].std(0) + 1e-9
        w, b = _fit_logreg((X[~te] - m) / s, y[~te], lam)
        out[te] = ((X[te] - m) / s) @ w + b
    return out[:len(Xpos)], out[len(Xpos):]


# ---------------------------------------------------------------- binary secondary: propensity matching
def propensity_match(lost, Xpre, exact, seed, caliper_sd=0.2):
    """1:1 nearest-neighbour matching of lost (1) to retained (0) items on the logit of a propensity score
    fitted on PRE-intervention variables only; exact on `exact`; caliper = caliper_sd * SD(logit p);
    seeded random order; without replacement. Returns (pairs [(i_lost, j_ret)], max_abs_smd)."""
    from .diagnostics import _fit_logreg
    lost = np.asarray(lost, bool)
    Xs = (Xpre - Xpre.mean(0)) / (Xpre.std(0) + 1e-9)
    w, b = _fit_logreg(Xs, lost.astype(float), 1.0)
    lp = Xs @ w + b
    cal = caliper_sd * lp.std()
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(lost)[0], np.nonzero(~lost)[0]
    used = np.zeros(len(lost), bool)
    pairs = []
    for i in rng.permutation(pos):
        cand = neg[(~used[neg]) & (exact[neg] == exact[i])]
        if not len(cand):
            continue
        d = np.abs(lp[cand] - lp[i])
        j = int(np.argmin(d))
        if d[j] <= cal:
            used[cand[j]] = True
            pairs.append((int(i), int(cand[j])))
    if not pairs:
        return pairs, float("inf")
    a, c = Xpre[[p[0] for p in pairs]], Xpre[[p[1] for p in pairs]]
    pooled = np.sqrt((a.var(0) + c.var(0)) / 2) + 1e-9
    return pairs, float(np.max(np.abs(a.mean(0) - c.mean(0)) / pooled))


# ---------------------------------------------------------------- symmetric pre-state adjustment (primary candidate)
def theta_post_dml(M_pre, M_post, C_pre, C_post, W, H_pre, seed, G=None, strata=None, scale_M="rank",
                   scale_C="rank", knot_q=KNOT_Q):
    """Residualized-post association with SYMMETRIC adjustment for pre-intervention information:
      1. B = [RCS(rank C_pre), RCS(rank M_pre), W (ranks), (G ranks if theta_gen)]; OLS-residualize rank M_post,
         rank C_post and every column of the pre-intervention state H_pre on B;
      2. cross-fitted ridge (lambda by inner 5-fold CV, D49) predictions of each residualized outcome from the residualized pre-states;
      3. theta = corr of the two final residual vectors.
    Pre-state information (incl. any susceptibility the states encode) is removed from BOTH variables.
    Returns (theta, {'r2_M': cross-fitted R2 of the M step, 'r2_C': of the C step})."""
    tM = rank01 if scale_M == "rank" else (lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4))))
    tC = rank01 if scale_C == "rank" else (lambda v: np.asarray(v, float))
    Wr = None if W is None else np.column_stack([rank01(c) if len(np.unique(c)) > 2 else c for c in np.asarray(W, float).T])
    gen = None if G is None else np.column_stack([rank01(c) for c in np.asarray(G, float).T])
    B = _covmat([rcs(tC(C_pre), knot_q), rcs(tM(M_pre), knot_q), Wr, gen])
    rM, rC = residualize(tM(M_post), B), residualize(tC(C_post), B)
    Hr = np.column_stack([residualize(h, B) for h in np.asarray(H_pre, float).T])
    pM, r2M, _ = crossfit_ridge(Hr, rM, seed, strata)
    pC, r2C, _ = crossfit_ridge(Hr, rC, seed + 7, strata)
    return _corr(rM - pM, rC - pC), {"r2_M": r2M, "r2_C": r2C, "resid_M": rM - pM, "resid_C": rC - pC}


def balance(Xpre, pairs):
    """(max |SMD|, mean |SMD|) over the columns of Xpre for matched pairs (pooled-SD standardized)."""
    if not pairs:
        return float("inf"), float("inf")
    a, c = Xpre[[p[0] for p in pairs]], Xpre[[p[1] for p in pairs]]
    pooled = np.sqrt((a.var(0) + c.var(0)) / 2) + 1e-9
    d = np.abs(a.mean(0) - c.mean(0)) / pooled
    return float(d.max()), float(d.mean())
