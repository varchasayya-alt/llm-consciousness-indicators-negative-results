"""Workspace arms for the planted system (prereg §5 S1c; memo §3.2 A, amendment A1).

A  : Re-entrant Slot Workspace. Blind write from sentence positions at L_w (cue input = constant e_bar);
     competitive slot attention; gated slot update; LN + fixed noise; shared reader at the read layers.
C1 : identical module and parameter count, but the writer receives the cue-derived representation e(cue)
     (frozen-model mean residual over the cue string) -> can represent untrained consumers' cues.
C3 : Back-Attention-type unrestricted re-entry: read layers attend directly to all sentence positions' L_w states
     at width r_v (no slots, no competition).
Only the planted model is ever trained with these modules (guard in run_c16).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import planted as P
from .subject import EpSpec, SentSpec


class SlotWorkspace(nn.Module):
    def __init__(self, d, d_w, read_layers, K=2, noise=0.1):
        super().__init__()
        self.d, self.d_w, self.K, self.noise = d, d_w, K, noise
        self.read_layers = tuple(read_layers)
        self.ln_h = nn.LayerNorm(d, elementwise_affine=False)
        self.ln_e = nn.LayerNorm(d, elementwise_affine=False)
        self.W_in = nn.Linear(d, d_w, bias=False)
        self.Q = nn.Parameter(torch.randn(K, d_w) / math.sqrt(d_w))
        self.W_qc = nn.Linear(d, d_w, bias=False)        # cue -> slot queries
        self.W_cc = nn.Linear(d, d_w, bias=False)        # cue -> candidates
        self.s0 = nn.Parameter(torch.zeros(K, d_w))
        self.w_z = nn.Linear(2 * d_w, 1)
        self.W_q = nn.Linear(d, d_w, bias=False)         # shared reader
        self.W_o = nn.Linear(d_w, d, bias=False)
        self.g = nn.Parameter(torch.zeros(len(self.read_layers)))   # zero-init gates -> identity at init

    def write(self, H, e):
        """H: [B, T, d] sentence-position residuals at L_w; e: [B, d] cue representation. -> S [B, K, d_w]."""
        ee = self.ln_e(e)
        c = self.W_in(self.ln_h(H)) + self.W_cc(ee)[:, None]                # [B, T, d_w]
        q = self.Q[None] + self.W_qc(ee)[:, None]                            # [B, K, d_w]
        sc = torch.einsum("bkd,btd->bkt", q, c) / math.sqrt(self.d_w)        # [B, K, T]
        comp = sc.softmax(1)                                                 # positions compete for slots
        top = sc.topk(min(2, sc.shape[-1]), -1).indices
        mask = torch.full_like(sc, float("-inf")).scatter(-1, top, 0.0)
        att = (sc + mask).softmax(-1) * comp
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        m = torch.einsum("bkt,btd->bkd", att, c)
        s0 = self.s0[None].expand_as(m)
        z = torch.sigmoid(self.w_z(torch.cat([s0, m], -1)))
        S = F.layer_norm((1 - z) * s0 + z * m, (self.d_w,))
        return S

    def noisy(self, S, gen=None):
        if self.noise <= 0:
            return S
        return S + self.noise * torch.randn(S.shape, generator=gen)

    def read_fn(self, S, start):
        def f(l, x):
            if l not in self.read_layers:
                return x
            gi = self.read_layers.index(l)
            q = self.W_q(F.layer_norm(x[:, start:], (self.d,)))           # [B, U, d_w]
            beta = torch.einsum("bud,bkd->buk", q, S) / math.sqrt(self.d_w)
            r = torch.einsum("buk,bkd->bud", beta.softmax(-1), S)
            add = torch.zeros_like(x)
            add[:, start:] = self.g[gi] * self.W_o(r)
            return x + add
        return f


class ReentryC3(nn.Module):
    """Unrestricted learned re-entry (Back-Attention-type): reads all sentence states at width r_v."""

    def __init__(self, d, r_v, read_layers):
        super().__init__()
        self.d, self.r_v = d, r_v
        self.read_layers = tuple(read_layers)
        self.W_q = nn.Linear(d, r_v, bias=False)
        self.W_k = nn.Linear(d, r_v, bias=False)
        self.W_v = nn.Linear(d, r_v, bias=False)
        self.W_o = nn.Linear(r_v, d, bias=False)
        self.g = nn.Parameter(torch.zeros(len(self.read_layers)))

    def write(self, H, e):
        Hn = F.layer_norm(H, (self.d,))
        return (self.W_k(Hn), self.W_v(Hn))

    def noisy(self, S, gen=None):
        return S

    def read_fn(self, S, start):
        Kt, Vt = S

        def f(l, x):
            if l not in self.read_layers:
                return x
            gi = self.read_layers.index(l)
            q = self.W_q(F.layer_norm(x[:, start:], (self.d,)))
            a = torch.einsum("bur,btr->but", q, Kt) / math.sqrt(self.r_v)
            r = torch.einsum("but,btr->bur", a.softmax(-1), Vt)
            add = torch.zeros_like(x)
            add[:, start:] = self.g[gi] * self.W_o(r)
            return x + add
        return f


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- planted training / evaluation
class PlantedArm:
    def __init__(self, subject, kind, d_w, read_layers, L_w, cue_layer, seed, e_bar=None, r_v=None, noise=0.1):
        if not isinstance(subject, P.PlantedSubject):
            raise TypeError("workspace arms may only be trained on the planted subject in S0/S1")
        torch.manual_seed(seed)
        self.sub, self.kind, self.L_w = subject, kind, L_w
        d = subject.d_model
        self.ws = (ReentryC3(d, r_v or d, read_layers) if kind == "C3"
                   else SlotWorkspace(d, d_w, read_layers, noise=noise))
        self.cue_layer = cue_layer
        self.e_bar = e_bar
        self.cue_cache = {}
        self.gen = torch.Generator().manual_seed(seed + 7)

    def cue(self, consumer, syn):
        key = (consumer, syn)
        if key not in self.cue_cache:
            self.cue_cache[key] = self.sub.cue_rep(consumer, self.cue_layer, syn)
        return self.cue_cache[key]

    def e_for(self, eps, syn):
        if self.kind == "C1":
            return torch.stack([self.cue(ep.consumer, syn) for ep in eps])
        return self.e_bar[None].expand(len(eps), -1)

    def sent_states(self, sents):
        """Pass 1 (no workspace): residual at L_w over the sentence positions."""
        out = []
        for s0 in range(0, len(sents), 256):
            chunk = sents[s0:s0 + 256]
            toks = [P.episode_tokens(EpSpec(s, "copy", None, {}), None)[0] for s in chunk]
            ids, pad = P.batchify([P.to_ids(t) for t in toks])
            with torch.no_grad():
                _, hs = self.sub.m(ids, pad, keep=[self.L_w])
            out.append(hs[self.L_w][:, 1:1 + P.SL])
        return torch.cat(out, 0)

    def slots(self, eps, syn=0, src_sents=None, noisy=True):
        sents = src_sents if src_sents is not None else [ep.sent for ep in eps]
        S = self.ws.write(self.sent_states(sents), self.e_for(eps, syn))
        return self.ws.noisy(S, self.gen) if noisy else S

    def train(self, sampler, steps=1500, batch=64, lr=3e-3, log=print):
        opt = torch.optim.Adam(self.ws.parameters(), lr=lr)
        for step in range(steps):
            eps = sampler(batch)
            S = self.slots(eps)
            toks, a0s, anss = [], [], []
            for ep in eps:
                ans = P.ans_tokens(ep.consumer, ep.cands["own"], ep.table)
                t, pp, _, a0 = P.episode_tokens(ep, ans)
                toks.append(t)
                a0s.append(a0)
                anss.append(ans)
                start = pp[0]
            ids, pad = P.batchify([P.to_ids(t) for t in toks])
            logits, _ = self.sub.m(ids, pad, read_fn=self.ws.read_fn(S, start))
            rows, cols, tgt = [], [], []
            for i, (a0, ans) in enumerate(zip(a0s, anss)):
                for k in range(len(ans)):
                    rows.append(i)
                    cols.append(a0 + k - 1)
                    tgt.append(ids[i, a0 + k])
            loss = F.cross_entropy(logits[rows, cols], torch.stack(tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 500 == 0 or step == steps - 1:
                log(f"    {self.kind} step {step} loss {loss.item():.4f}")

    def run(self, eps, S, syn=0):
        """Evaluate candidates of eps with slots S injected (S indexed like eps)."""
        def factory(idx, start):
            if isinstance(S, tuple):
                Ss = (S[0][idx], S[1][idx])
            else:
                Ss = S[idx]
            return self.ws.read_fn(Ss, start)
        with torch.no_grad():
            return self.sub.evaluate(eps, read_fn_factory=factory, syn=syn)

    def nc_slots(self, S):
        if isinstance(S, tuple):
            K, V = S
            z = torch.randn(V.shape, generator=self.gen)
            z = z / z.norm(dim=-1, keepdim=True) * V.norm(dim=-1, keepdim=True)
            return (K, z)
        z = torch.randn(S.shape, generator=self.gen)
        return z / z.norm(dim=-1, keepdim=True) * S.norm(dim=-1, keepdim=True)


def partner_x(x, consumer, table, pool, rng):
    if consumer == "lookup":
        return rng.choice([v for v, _ in table if v != x])
    c = [v for v in pool if v != x and (v % 2) != (x % 2) and ((v > 50) != (x > 50))]
    if not c:
        c = [v for v in pool if v != x and (v % 2) != (x % 2)] or [v for v in pool if v != x]
    return rng.choice(sorted(c))


def evaluate_arm(arm, eps, pool, seed, syn=0):
    """Returns acc, CT (swap-following minus NC-following on src-competent items), ablation acc, n."""
    rng = random.Random(seed)
    with torch.no_grad():
        S = arm.slots(eps, syn)
        xp = [partner_x(ep.cands["own"], ep.consumer, ep.table, pool, rng) for ep in eps]
        src = []
        for ep, x2 in zip(eps, xp):
            p, args = ep.sent.payload
            args2 = P.random_args(p, x2, rng) or P.random_args("add", x2, rng)
            p2 = p if P.random_args(p, x2, random.Random(0)) is not None else "add"
            src.append(SentSpec("latent", ep.sent.name, (p2, args2)))
        S_sw = arm.slots(eps, syn, src_sents=src)
        S_nc = arm.nc_slots(S)
        if isinstance(S, tuple):
            S_zero = (S[0], torch.zeros_like(S[1]))
        else:
            S_zero = torch.zeros_like(S)
        eps2 = [EpSpec(ep.sent, ep.consumer, ep.table, {"own": ep.cands["own"], "sw": x2}) for ep, x2 in zip(eps, xp)]
        own = arm.run(eps2, S, syn)
        sw = arm.run(eps2, S_sw, syn)
        nc = arm.run(eps2, S_nc, syn)
        ab = arm.run(eps2, S_zero, syn)
        nat = arm.sub.evaluate(eps2, syn=syn)
        txt = arm.sub.evaluate([EpSpec(SentSpec("text", ep.sent.name, (x2,)), ep.consumer, ep.table, {"sw": x2})
                                for ep, x2 in zip(eps, xp)], syn=syn)
    ok = [k for k in range(len(eps)) if txt[k]["sw"]]
    mean = lambda v: float(sum(v) / len(v)) if v else None
    return {"n": len(eps), "n_ok": len(ok), "acc": mean([r["own"] for r in own]),
            "acc_ablate": mean([r["own"] for r in ab]),
            "P_sw_swap": mean([sw[k]["sw"] for k in ok]), "P_sw_nc": mean([nc[k]["sw"] for k in ok]),
            "acc_nc": mean([r["own"] for r in nc]), "acc_nat": mean([r["own"] for r in nat]),
            "P_sw_nat": mean([nat[k]["sw"] for k in ok]),
            "CT": (mean([sw[k]["sw"] for k in ok]) - mean([nc[k]["sw"] for k in ok])) if ok else None}


def cue_invariance(arm, sent, consumers, table):
    """I6: A's slots must be bit-identical across consumer cues (noise off). Each cue is computed in its own
    single-row call, so batch-position floating effects cannot masquerade as cue dependence."""
    outs = []
    with torch.no_grad():
        for c in consumers:
            ep = EpSpec(sent, c, table if c == "lookup" else None, {"own": 0})
            outs.append(arm.slots([ep], noisy=False))
    return float(max((o - outs[0]).abs().max() for o in outs))
