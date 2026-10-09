"""Stage 1a (parametric power / complete-gate FPR) and Stage 1b (synthetic rank recovery). Prereg §5."""
from __future__ import annotations

import math

import numpy as np
import torch

from . import config as C
from . import ranks as RK


# ================================================================== S1b rank recovery
def s1b_rank_recovery(seed, d=896, P=7, n=200, grid=None, frac=0.9, min_full=0.10):
    """Linear-Gaussian native format: delta = mu_p + A z * s_p + eps. Consumers read argmax(L_j A^T delta_mean)."""
    grid = grid or C.S0["ranks"]
    rng = np.random.default_rng(seed)
    out = {}

    def one(r_true, label):
        A = np.linalg.qr(rng.standard_normal((d, r_true)))[0]
        mu = rng.standard_normal((P, d)) * 0.5
        s = 1.0 + 0.5 * rng.random(P)
        Z = rng.standard_normal((n, r_true))
        noise = rng.standard_normal((n, P, d)) * 0.05
        D = mu[None] + (Z @ A.T)[:, None, :] * s[None, :, None] + noise
        D = torch.tensor(D, dtype=torch.float32)
        half = n // 2
        basis = RK.fit_basis(D[:half], max_rank=128)
        Dte, Zte = D[half:], Z[half:]
        Ls = [rng.standard_normal((5, r_true)) for _ in range(4)]          # 4 consumers, 5 classes each
        def acc(Dp):
            m = Dp.mean(1).numpy() - mu.mean(0)                             # patched content estimate
            zhat = m @ A / s.mean()
            return float(np.mean([np.mean((zhat @ L.T).argmax(1) == (Zte @ L.T).argmax(1)) for L in Ls]))
        base = 0.2
        gains = {}
        for r in grid:
            gains[r] = acc(RK.project(Dte, basis, r)) - base
        rs = RK.rank_star(gains, grid, frac, min_full)
        est = RK.as_number(rs, d) if rs is not None else None
        ok = est is not None and (r_true / 2 <= est <= r_true * 2)
        out[label] = {"r_true": r_true, "r_star": rs, "gains": {str(k): v for k, v in gains.items()}, "within_x2": ok}

    for r in C.S1["S1b"]["r_true"]:
        one(r, f"content_r{r}")
    for r in C.S1["S1b"]["rB_true"]:
        one(r, f"bundle_r{r}")
    out["pass"] = all(v["within_x2"] for v in out.values() if isinstance(v, dict))
    return out


# ================================================================== S1a helpers
def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sim_cells(rng, p, n_s, n_x, m, sig_s, sig_x):
    """Bernoulli outcomes with logit-normal seed and cluster effects. Returns cell means [n_s, n_x]."""
    if p <= 0:
        return np.zeros((n_s, n_x))
    lp = _logit(p) + rng.normal(0, sig_s, (n_s, 1)) + rng.normal(0, sig_x, (1, n_x))
    q = 1 / (1 + np.exp(-lp))
    return rng.binomial(m, q) / m


def _boot_weights(rng, B, n_s, n_x):
    ws = rng.multinomial(n_s, [1 / n_s] * n_s, size=B).astype(float)
    wx = rng.multinomial(n_x, [1 / n_x] * n_x, size=B).astype(float)
    return ws, wx


def _wmean(cells, ws, wx):
    num = np.einsum("bs,sx,bx->b", ws, cells, wx)
    den = np.einsum("bs,bx->b", ws, wx) * 1.0
    return num / den


# ================================================================== S1a core classification
ARMS = ("A", "C1", "C2", "C3", "C4", "C5", "C7")
CORE_WORLDS = {
    #            CT_HO by arm                                       CT_U by arm
    "null":      ({a: 0.0 for a in ARMS},                               {a: 0.0 for a in ARMS}),
    "H1_bound":  ({**{a: 0.0 for a in ARMS}, "A": 0.10},                {a: 0.0 for a in ARMS}),
    "reentry":   ({"A": .5, "C1": .5, "C2": .5, "C3": .5, "C4": .05, "C5": .05, "C7": .05},
                  {"A": .4, "C1": .4, "C2": .4, "C3": .4, "C4": .03, "C5": .03, "C7": .03}),
    "broadcast": ({"A": .5, "C1": .25, "C2": .25, "C3": .5, "C4": .05, "C5": .05, "C7": .05},
                  {"A": .4, "C1": .15, "C2": .15, "C3": .4, "C4": .03, "C5": .03, "C7": .03}),
    "full_gw":   ({"A": .5, "C1": .25, "C2": .25, "C3": .3, "C4": .05, "C5": .05, "C7": .05},
                  {"A": .4, "C1": .15, "C2": .15, "C3": .2, "C4": .03, "C5": .03, "C7": .03}),
}
TRUE_CLASS = {"null": "W0", "H1_bound": "W0", "reentry": "W-RE", "broadcast": "W-BC", "full_gw": "W-GW"}
RANK = {"W0": 0, "W-RE": 1, "W-BC": 2, "W-GW": 3}


def classify_core(est, lo, gates):
    """est/lo: dict[(arm, endpoint)] -> point estimate / 2.5% bound of CT (and of differences, key ('d', arm, ep))."""
    h1 = est[("A", "HO")] >= 0.30 and lo[("A", "HO")] >= gates["h1_lb"]
    if not h1:
        return "W0"
    h2 = est[("A", "U")] >= 0.20 and lo[("A", "U")] > gates["h2_lb"]
    h5 = all(est[("d", a, "HO")] >= 0.15 and lo[("d", a, "HO")] > gates["h5_lb"] for a in ("C4", "C5", "C7"))
    if not (h2 and h5):
        return "W0"
    h3 = (all(est[("d", a, "HO")] >= 0.10 and lo[("d", a, "HO")] > gates["h3_lb"] for a in ("C1", "C2"))
          and est[("d", "C1", "U")] >= 0.10 and lo[("d", "C1", "U")] > gates["h3_lb"])
    if not h3:
        return "W-RE"
    h4 = est[("d", "C3", "U")] >= 0.10 and lo[("d", "C3", "U")] > gates["h4_lb"]
    return "W-GW" if h4 else "W-BC"


def s1a_core(seed, sig_s, sig_x, n_s=3, n_x=22, m_ho=4, m_u=3, p_nc=0.05, reps=400, B=2000, gates=None):
    gates = gates or {"h1_lb": 0.10, "h2_lb": 0.0, "h3_lb": 0.0, "h4_lb": 0.0, "h5_lb": 0.0}
    rng = np.random.default_rng(seed)
    res = {}
    for w, (ho, u) in CORE_WORLDS.items():
        counts = {}
        for _ in range(reps):
            ws, wx = _boot_weights(rng, B, n_s, n_x)
            est, lo = {}, {}
            bt = {}
            for ep, ptab, m in (("HO", ho, m_ho * 4), ("U", u, m_u * 3)):
                nc = _sim_cells(rng, p_nc, n_s, n_x, m, sig_s, sig_x)
                for a in ARMS:
                    sw = _sim_cells(rng, min(ptab[a] + p_nc, 0.99), n_s, n_x, m, sig_s, sig_x)
                    ct = sw - nc
                    est[(a, ep)] = float(ct.mean())
                    bt[(a, ep)] = _wmean(ct, ws, wx)
                    lo[(a, ep)] = float(np.percentile(bt[(a, ep)], 2.5))
                for a in ARMS[1:]:
                    d = bt[("A", ep)] - bt[(a, ep)]
                    est[("d", a, ep)] = est[("A", ep)] - est[(a, ep)]
                    lo[("d", a, ep)] = float(np.percentile(d, 2.5))
            c = classify_core(est, lo, gates)
            counts[c] = counts.get(c, 0) + 1
        tot = sum(counts.values())
        dist = {k: v / tot for k, v in counts.items()}
        truth = TRUE_CLASS[w]
        over = sum(v for k, v in dist.items() if RANK[k] > RANK[truth])
        res[w] = {"dist": dist, "true_class": truth, "P_correct": dist.get(truth, 0.0), "P_overclaim": over}
    fpr = max(r["P_overclaim"] for r in res.values())
    power = min(r["P_correct"] for w, r in res.items() if w in ("reentry", "broadcast", "full_gw"))
    return {"worlds": res, "max_overclaim_fpr": fpr, "min_power_true_class": power,
            "n_s": n_s, "n_x": n_x, "sig_s": sig_s, "sig_x": sig_x, "gates": gates}


# ================================================================== S1a H-CD
CAPS = (0.5, 1, 2, 4, 8)
SETS = ("S1", "S2", "S4", "Spm", "Sparam")
IDENT = {"S1": True, "S2": True, "S4": True, "Spm": False, "Sparam": True}


def hcd_truth(world, rB, t_hi=0.4, t_lo=0.1):
    """T(c,S) for each account. rB: dict set -> bundle rank in units of r_X (Sparam = inf)."""
    T = {}
    for c in CAPS:
        for S in SETS:
            if world == "null" or c < 1:
                T[(c, S)] = 0.0
            elif world == "copy":
                T[(c, S)] = t_hi
            elif world == "ib":
                T[(c, S)] = t_hi if IDENT[S] else 0.0
            elif world == "geometry":
                rb = rB[S]
                T[(c, S)] = t_hi if c < rb else t_lo
            else:
                raise ValueError(world)
    return T


def decide_hcd(est, lo, hi, margin):
    if not (est[("T", 2, "S4")] >= margin and lo[("T", 2, "S4")] > 0):
        return "none"
    D, Q, I = ("D",), ("Q",), ("I",)
    if lo[D] > 0 and est[D] >= margin and lo[Q] > 0 and est[Q] >= margin:
        return "geometry"
    if hi[I] < 0 and est[I] <= -margin and not (lo[D] > 0 and est[D] >= margin):
        return "ib"
    if all(lo[k] > -margin and hi[k] < margin for k in (D, Q, I)):
        return "copy"
    return "undetermined"


def s1a_hcd(seed, sig_s, sig_x, rB, n_s, m_cell, n_x=22, p_nc=0.05, reps=300, B=1000, margin=0.10):
    rng = np.random.default_rng(seed)
    out = {}
    for world in ("geometry", "ib", "copy", "null"):
        T = hcd_truth(world, rB)
        counts = {}
        for _ in range(reps):
            ws, wx = _boot_weights(rng, B, n_s, n_x)
            m = max(1, m_cell // n_x)
            bt = {}
            for key in ((2, "S4"), (2, "S1"), (8, "S4"), (2, "Spm")):
                sw = _sim_cells(rng, min(T[key] + p_nc, 0.99), n_s, n_x, m, sig_s, sig_x)
                nc = _sim_cells(rng, p_nc, n_s, n_x, m, sig_s, sig_x)
                ct = sw - nc
                bt[("T",) + key] = (float(ct.mean()), _wmean(ct, ws, wx))
            est, lo, hi = {}, {}, {}
            for k, (e, b) in bt.items():
                est[k], lo[k], hi[k] = e, np.percentile(b, 2.5), np.percentile(b, 97.5)
            for name, a, b in (("D", ("T", 2, "S4"), ("T", 2, "S1")), ("Q", ("T", 2, "S4"), ("T", 8, "S4")),
                               ("I", ("T", 2, "Spm"), ("T", 2, "S4"))):
                diff = bt[a][1] - bt[b][1]
                est[(name,)] = est[a] - est[b]
                lo[(name,)], hi[(name,)] = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
            dcs = decide_hcd(est, lo, hi, margin)
            counts[dcs] = counts.get(dcs, 0) + 1
        tot = sum(counts.values())
        dist = {k: v / tot for k, v in counts.items()}
        truth = "none" if world == "null" else world
        out[world] = {"dist": dist, "P_correct": dist.get(truth, 0.0),
                      "P_wrong_account": sum(v for k, v in dist.items() if k not in (truth, "undetermined", "none"))}
    return {"worlds": out, "n_s": n_s, "m_cell": m_cell, "sig_s": sig_s, "sig_x": sig_x, "rB": rB,
            "min_power": min(out[w]["P_correct"] for w in ("geometry", "ib", "copy")),
            "max_wrong_account": max(out[w]["P_wrong_account"] for w in out)}
