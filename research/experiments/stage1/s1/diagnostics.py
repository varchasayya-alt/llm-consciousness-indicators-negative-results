"""Generic-intervention distinguishability diagnostics (methods development / sham validation).

A *separate* diagnostic classifier (never the metacognitive monitor) asks whether forgotten (X_lost)
and sham (Y) items can be told apart from generic intervention statistics that are not competence
read-outs:
  G1 per-site log displacement profile (10)
  G2 G1 + per-site cosine(delta, h_pre) + per-site relative norm change (30)
  G3 per-site change in 'abnormality' = mean squared z-score w.r.t. the intact state distribution (10)
  G4 raw standardised post-intervention read states (1,280) -- an 'edit-detection probe';
     applied to X_retained vs Y (both competent) to detect procedure fingerprints.
Classifier: L2 logistic regression (lambda fixed a priori = 1.0 on standardised features), 5-fold
stratified CV, mean held-out AUROC.
"""
import numpy as np
import torch

from .metrics import auroc

THRESHOLDS = {"ok": 0.75, "serious": 0.90}   # declared before calibration


def generic_features(H_pre, H_post, mu, sigma):
    zpre, zpost = (H_pre - mu) / sigma, (H_post - mu) / sigma
    delta = zpost - zpre
    disp = np.sqrt((delta ** 2).mean(-1))
    g1 = np.log(disp + 1e-3)
    cos = (delta * zpre).sum(-1) / (np.linalg.norm(delta, axis=-1) * np.linalg.norm(zpre, axis=-1) + 1e-9)
    relnorm = (np.linalg.norm(H_post, axis=-1) - np.linalg.norm(H_pre, axis=-1)) / (np.linalg.norm(H_pre, axis=-1) + 1e-9)
    g3 = (zpost ** 2).mean(-1) - (zpre ** 2).mean(-1)
    return {"G1": g1, "G2": np.concatenate([g1, cos, relnorm], 1), "G3": g3,
            "G4": zpost.reshape(len(zpost), -1)}


def _fit_logreg(Xtr, ytr, lam=1.0, iters=200):
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + lam * (w ** 2).sum() / len(yt)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def cv_auroc(Xpos, Xneg, seed=0, folds=5, lam=1.0):
    X = np.concatenate([Xpos, Xneg]).astype(np.float64)
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])
    if len(Xpos) < folds or len(Xneg) < folds:
        return float("nan")
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    scores = np.empty(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-9
        w, b = _fit_logreg((X[tr] - m) / s, y[tr], lam)
        scores[te] = ((X[te] - m) / s) @ w + b
    return auroc(scores[y == 1], scores[y == 0])


def classify(auc):
    if np.isnan(auc):
        return "n/a"
    if auc <= THRESHOLDS["ok"]:
        return "acceptable"
    if auc <= THRESHOLDS["serious"]:
        return "moderate (document; enter as covariate)"
    return "SERIOUS (redesign before freezing)"
