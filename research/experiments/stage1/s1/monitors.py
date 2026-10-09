"""Second-order monitors and their training pools.

Conditions: IN (query tokens), OUT (sorted restricted output distribution + entropy),
INT-{S,T,P,TP} (standardised read set), exploratory OUT-TP and INT+OUT-S, shuffled-label control.
All monitors train for exactly the same number of steps/batch size (equal optimisation budget),
with class-balanced BCE computed on the pool's (mixture) label distribution.
NOTE: calibration scripts must never import this module (enforced by a unit test).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPMonitor(nn.Module):
    def __init__(self, in_dim, hidden=(256, 256), dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class INMonitor(nn.Module):
    """Input-only external observer: own embeddings of (s1, s2, r)."""
    def __init__(self, n_syll, n_rel, emb_dim=64, hidden=(256, 256), dropout=0.1):
        super().__init__()
        self.syl = nn.Embedding(n_syll, emb_dim)
        self.rel = nn.Embedding(n_rel, emb_dim)
        self.head = MLPMonitor(3 * emb_dim, hidden, dropout)

    def embed(self, x):
        return torch.cat([self.syl(x[:, 0]), self.syl(x[:, 1]), self.rel(x[:, 2])], -1)

    def forward(self, x):
        return self.head(self.embed(x))


# ---------------------------------------------------------------- features
def int_features(H, mu, sigma):
    return ((H - mu[None]) / sigma[None]).reshape(len(H), -1).astype(np.float32)


def out_features(probs):
    p = -np.sort(-probs, axis=1)
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1, keepdims=True)
    return np.concatenate([p, ent], 1).astype(np.float32)


def in_features(world, items, names=None):
    nm = world.names[items[:, 0]] if names is None else names
    return np.stack([nm[:, 0], nm[:, 1], items[:, 1]], 1).astype(np.int64)


# ---------------------------------------------------------------- training
def train_monitor(model, pools, mcfg, seed, shuffle_labels=False):
    """pools: list of (features ndarray, labels ndarray, pool_weight). Samples are drawn with replacement;
    each pool gets total probability pool_weight / sum(weights)."""
    g = torch.Generator().manual_seed(seed)
    feats = [torch.as_tensor(f) for f, _, _ in pools]
    labels = [torch.as_tensor(l.astype(np.float32)) for _, l, _ in pools]
    if shuffle_labels:
        labels = [l[torch.randperm(len(l), generator=g)] for l in labels]
    pw = np.array([w for _, _, w in pools], float)
    pw = pw / pw.sum()
    # mixture base rate -> class-balanced weights
    p_pos = float(sum(w * l.float().mean().item() for w, l in zip(pw, labels)))
    p_pos = min(max(p_pos, 1e-3), 1 - 1e-3)
    w_pos, w_neg = 0.5 / p_pos, 0.5 / (1 - p_pos)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    bs = mcfg["batch_size"]
    model.train()
    for step in range(mcfg["steps"]):
        counts = np.bincount(torch.multinomial(torch.as_tensor(pw), bs, replacement=True, generator=g).numpy(),
                             minlength=len(pools))
        xb, yb = [], []
        for k, c in enumerate(counts):
            if c:
                idx = torch.randint(0, len(feats[k]), (int(c),), generator=g)
                xb.append(feats[k][idx])
                yb.append(labels[k][idx])
        x, y = torch.cat(xb), torch.cat(yb)
        w = torch.where(y > 0.5, torch.full_like(y, w_pos), torch.full_like(y, w_neg))
        loss = F.binary_cross_entropy_with_logits(model(x), y, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, batch=8192):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.sigmoid(model(torch.as_tensor(x[i:i + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def make_monitor(kind, world, in_dim=None, mcfg=None):
    hidden = tuple(mcfg["hidden"]) if mcfg else (256, 256)
    drop = mcfg["dropout"] if mcfg else 0.1
    if kind == "IN":
        return INMonitor(world.vocab.n_syll, world.vocab.n_rel, mcfg["in_embedding_dim"], hidden, drop)
    if kind.startswith("OUT") and not kind.startswith("OUT+"):
        return MLPMonitor(in_dim, tuple(mcfg["out_hidden"]), drop)
    return MLPMonitor(in_dim, hidden, drop)
