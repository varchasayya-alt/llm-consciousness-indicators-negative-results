"""Batched sparse non-negative gradient pursuit over a dictionary of unit atoms.

Each iteration adds the atom with the largest positive correlation with the residual, then takes one
gradient-pursuit step on the support coefficients (step = |g|^2 / |D_S g|^2) and clips at zero.
Rows whose best remaining correlation is <= 0 stop growing (their new slot stays inactive).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def gp_nonneg(H, D, k, chunk=128, screen=None):
    """H: [n, d] states; D: [V, d] unit atoms. Returns (idx [n, k] long, coef [n, k], recon [n, d]).

    Inactive slots have coef 0 and idx -1. `screen` (int): restrict each row's atom search to its `screen` atoms
    with the largest initial correlation with h (pre-declared approximation, validated against exact GP).
    """
    outs = [_gp_chunk(H[i:i + chunk], D, k, screen) for i in range(0, H.shape[0], chunk)]
    return (torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]), torch.cat([o[2] for o in outs]))


def _gp_chunk(H, D, k, screen=None):
    H = H.float()
    if screen is not None and screen < D.shape[0]:
        cand = (H @ D.T).topk(screen, dim=1).indices          # [n, S]
        Dc = D[cand]                                          # [n, S, d]
        idx_l, coef, recon = _gp_core(H, k, lambda R: torch.einsum("nsd,nd->ns", Dc, R),
                                      lambda t: torch.gather(Dc, 1, t[:, None, None].expand(-1, 1, Dc.shape[2]))[:, 0])
        idx = torch.where(idx_l >= 0, torch.gather(cand, 1, idx_l.clamp_min(0)), idx_l)
        return idx, coef, recon
    return _gp_core(H, k, lambda R: R @ D.T, lambda t: D[t])


def _gp_core(H, k, corr_fn, atom_fn):
    n, d = H.shape
    idx = torch.full((n, k), -1, dtype=torch.long)
    coef = torch.zeros(n, k)
    valid = torch.zeros(n, k, dtype=torch.bool)
    DS = torch.zeros(n, k, d)
    R = H.clone()
    for it in range(k):
        C = corr_fn(R)
        if it:
            sel = idx[:, :it].clamp_min(0)
            C.scatter_(1, sel, float("-inf"))
        best, t = C.max(dim=1)
        ok = best > 0
        idx[:, it] = torch.where(ok, t, torch.full_like(t, -1))
        valid[:, it] = ok
        DS[:, it] = atom_fn(t) * ok[:, None]
        s = it + 1
        G = torch.einsum("nsd,nd->ns", DS[:, :s], R) * valid[:, :s]
        DG = torch.einsum("nsd,ns->nd", DS[:, :s], G)
        step = (G * G).sum(1) / (DG * DG).sum(1).clamp_min(1e-12)
        coef[:, :s] = ((coef[:, :s] + step[:, None] * G).clamp_min(0)) * valid[:, :s]
        R = H - torch.einsum("nsd,ns->nd", DS[:, :s], coef[:, :s])
    return idx, coef, H - R


def top_atoms(idx, coef, m):
    """Indices (into the dictionary) of the m largest-coefficient active atoms per row; -1 padded."""
    order = coef.argsort(dim=1, descending=True)[:, :m]
    sel = torch.gather(idx, 1, order)
    c = torch.gather(coef, 1, order)
    return torch.where(c > 0, sel, torch.full_like(sel, -1))
