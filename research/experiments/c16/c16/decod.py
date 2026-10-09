"""V0 decodability (prereg §4 M1): ridge one-vs-rest digit classifiers, PCA fitted in-fold, 5-fold CV stratified by X."""
from __future__ import annotations

import torch


def _ridge_fit(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def cv_digit_accuracy(feats, xs, folds, lam=1.0, pca_dim=256):
    """feats: [n, d] float; xs: list[int]; folds: list[int] fold id per row. Returns P(tens and ones both correct)."""
    feats = feats.double()
    xs_t = torch.tensor(xs)
    tens, ones = xs_t // 10, xs_t % 10
    correct = torch.zeros(len(xs), dtype=torch.bool)
    for f in sorted(set(folds)):
        te = torch.tensor([k for k, g in enumerate(folds) if g == f])
        tr = torch.tensor([k for k, g in enumerate(folds) if g != f])
        Xtr, Xte = feats[tr], feats[te]
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        k = min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1])
        _, _, Vh = torch.linalg.svd(Xtr, full_matrices=False)
        P = Vh[:k].T
        Ztr, Zte = Xtr @ P, Xte @ P
        Ztr = torch.cat([Ztr, torch.ones(len(Ztr), 1, dtype=Ztr.dtype)], 1)
        Zte = torch.cat([Zte, torch.ones(len(Zte), 1, dtype=Zte.dtype)], 1)
        ok = torch.ones(len(te), dtype=torch.bool)
        for target in (tens, ones):
            Y = torch.nn.functional.one_hot(target[tr], 10).double()
            W = _ridge_fit(Ztr, Y, lam)
            ok &= (Zte @ W).argmax(1) == target[te]
        correct[te] = ok
    return float(correct.double().mean())


def folds_by_x(xs, n_folds):
    seen = {}
    out = []
    for x in xs:
        k = seen.get(x, 0)
        out.append(k % n_folds)
        seen[x] = k + 1
    return out
