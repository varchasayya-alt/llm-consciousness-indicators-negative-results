"""Oracle-delta rank estimation (prereg §4 M4), shared by Stage 0, S1b and the planted system.

Delta tensors have shape [n, P, d] (items x prefix positions x width).
fit_basis: per-position mean mu_p and right singular vectors of the pooled centred deltas.
project:   Delta_r = mu_p + U_r U_r^T (Delta - mu_p);  r = 'full' returns Delta unchanged.
rank_star: smallest r in the grid whose gain >= frac * gain(full); None if gain(full) < min_full.
"""
from __future__ import annotations

import math

import torch


def fit_basis(delta, max_rank=128):
    mu = delta.mean(0)                                  # [P, d]
    X = (delta - mu).reshape(-1, delta.shape[-1])       # [n*P, d]
    k = min(max_rank, X.shape[0], X.shape[1])
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return {"mu": mu, "V": Vh[:k].T.contiguous()}      # V: [d, k]


def project(delta, basis, r):
    if r == "full":
        return delta
    mu, V = basis["mu"], basis["V"][:, :int(r)]
    c = delta - mu
    return mu + (c @ V) @ V.T


def nc_delta(delta, mu, gen):
    """Negative control: keep mu_p, replace the item-specific part by a random direction of equal norm."""
    c = delta - mu
    z = torch.randn(c.shape, generator=gen)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return mu + c.norm(dim=-1, keepdim=True) * z


def rank_star(gains: dict, grid, frac, min_full):
    gf = gains.get("full")
    if gf is None or gf < min_full:
        return None
    for r in grid:
        if r == "full":
            return "full"
        if gains.get(r) is not None and gains[r] >= frac * gf:
            return r
    return "full"


def as_number(r, d_model):
    return d_model if r == "full" else int(r)


def geo_mean(a, b, d_model):
    if a is None or b is None:
        return None
    return math.sqrt(as_number(a, d_model) * as_number(b, d_model))


def log2_ratio(a, b, d_model):
    if a is None or b is None:
        return None
    return abs(math.log2(as_number(a, d_model) / as_number(b, d_model)))
