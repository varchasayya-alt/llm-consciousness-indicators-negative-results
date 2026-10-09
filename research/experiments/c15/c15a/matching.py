"""Matched non-workspace controls (memo §6, task-independent rows).

For a content vector c and its workspace direction u_J = P_J c / |P_J c|, find a unit u in S_J^perp that is as
close as possible to c's own non-workspace component while matching u_J on
  * lens (output-head) gain      |W_eff J_l u|                     (tolerance x/÷ MATCH_TOL_NORM)
  * final propagation             |J_l u|                           (tolerance x/÷ MATCH_TOL_NORM)
  * generic KL proxy              1/2 E_pos Var_p(W_eff J_l u)/rms^2 (tolerance x/÷ MATCH_TOL_KL)
Norm, layer, position and rank are matched by construction (unit vectors, same site, rank 1).
"""
from __future__ import annotations

import hashlib
import math

import torch

from . import config as C
from .hooks import unit_directions


@torch.no_grad()
def kl_matrix_final(w_eff, final_resid, logits):
    """C_K in the final-block basis: 1/2 mean_pos (W_S^T (diag p - p p^T) W_S) / rms^2 (top-k truncation)."""
    d = w_eff.shape[1]
    Cm = torch.zeros(d, d)
    p_all = torch.softmax(logits.float(), dim=-1)
    for h, p in zip(final_resid.float(), p_all):
        pv, S = p.topk(C.MATCH_KL_TOPK)
        pv = pv / pv.sum()
        W = w_eff[S]
        rms2 = float((h ** 2).mean()) + 1e-6
        Wp = W.T @ pv
        Cm += (W.T @ (W * pv[:, None]) - torch.outer(Wp, Wp)) / rms2
    return 0.5 * Cm / max(len(final_resid), 1)


def _orth_add(basis, v, tol=1e-6):
    for b in basis:
        v = v - (b @ v) * b
    n = v.norm()
    if n > tol:
        basis.append(v / n)
    return basis


class LayerMatcher:
    def __init__(self, J, G_W, C_K, Q, layer_key):
        self.Q = Q
        self.A = {"gain": J.T @ G_W @ J, "prop": J.T @ J, "kl": J.T @ C_K @ J}
        for k in self.A:
            self.A[k] = 0.5 * (self.A[k] + self.A[k].T)
        d = Q.shape[0]
        Pp = torch.eye(d) - Q @ Q.T
        self.Pp = Pp
        self.eig = []
        for k in ("gain", "prop", "kl"):
            M = Pp @ self.A[k] @ Pp
            w, V = torch.linalg.eigh(0.5 * (M + M.T))
            self.eig += [V[:, -i] for i in range(1, C.MATCH_N_EIG + 1)]
        R = unit_directions(f"match-rand|{layer_key}", (C.MATCH_N_RAND, d))
        self.rand = [Pp @ r for r in R]

    def stats(self, u):
        return {k: float(u @ A @ u) for k, A in self.A.items()}

    def perp(self, v):
        return self.Pp @ v

    def match(self, content, uJ, seed_key, kl_scale=1.0, rand_seed=None):
        e_c = self.perp(content)
        if e_c.norm() < 1e-8:
            return None, {"feasible": False, "reason": "content has no perp component"}
        basis = [e_c / e_c.norm()]
        extra = []
        if rand_seed is not None:
            extra = list(unit_directions(f"match-pool|{seed_key}|{rand_seed}", (C.MATCH_N_RAND, self.Q.shape[0])))
        for v in self.eig + self.rand + extra:
            _orth_add(basis, self.perp(v))
        B = torch.stack(basis, 1)                          # [d, m]
        Ab = {k: B.T @ A @ B for k, A in self.A.items()}
        tgt = self.stats(uJ)
        tgt["kl"] = tgt["kl"] * kl_scale
        tol = {"gain": 2 * math.log(C.MATCH_TOL_NORM), "prop": 2 * math.log(C.MATCH_TOL_NORM),
               "kl": math.log(C.MATCH_TOL_KL)}
        best = None
        g = torch.Generator().manual_seed(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))
        m = B.shape[1]
        for rs in range(C.MATCH_RESTARTS):
            z0 = torch.zeros(m)
            z0[0] = 1.0
            z0 = z0 + (0.3 + 0.3 * rs) * torch.randn(m, generator=g)
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.05)
            with torch.enable_grad():
                for step in range(C.MATCH_STEPS):
                    nz = z @ z
                    lr = {k: torch.log((z @ Ab[k] @ z) / nz / tgt[k]) for k in Ab}
                    pen = sum(torch.relu(lr[k].abs() - 0.9 * tol[k]) ** 2 for k in lr)
                    loss = -z[0] / nz.sqrt() + 200.0 * pen
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            zz = z.detach()
            u = B @ zz
            u = self.perp(u)
            u = u / u.norm()
            st = self.stats(u)
            ratios = {k: st[k] / tgt[k] for k in st}
            feas = all(abs(math.log(max(ratios[k], 1e-12))) <= tol[k] for k in ratios)
            cos = float(u @ basis[0])
            if feas and (best is None or cos > best[1]):
                best = (u, cos, ratios)
        if best is None:
            return None, {"feasible": False, "reason": "no restart met tolerances"}
        u, cos, ratios = best
        return u, {"feasible": True, "cos_to_own_perp": cos, "ratios": ratios,
                   "sj_leak": float((self.Q.T @ u).norm())}
