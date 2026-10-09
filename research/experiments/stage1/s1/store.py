"""First-order store: a tiny pre-LN causal transformer with read-out hooks.

Supports (i) collecting the residual stream after embedding and after every block,
(ii) per-row attention knock-out masks (T-ACT), and (iii) per-row residual dropout at chosen
layers (P-family developmental perturbation).
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import world as W

READ_POSITIONS = (2, 4)   # s2, [A]  (frozen on architectural grounds: subject enrichment / answer extraction)


class Block(nn.Module):
    def __init__(self, d, n_heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)
        self.h = n_heads

    def forward(self, x, extra_mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        dh = D // self.h
        q = q.view(B, T, self.h, dh).transpose(1, 2)
        k = k.view(B, T, self.h, dh).transpose(1, 2)
        v = v.view(B, T, self.h, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        mask = causal.unsqueeze(0).unsqueeze(0)
        if extra_mask is not None:                       # [B, T, T] True = blocked
            mask = mask | extra_mask.unsqueeze(1)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class Store(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_width, max_len=8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_width) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, x, knockout=None, dropout=None, collect=False, generator=None, add=None):
        """x: [B,T] tokens.
        knockout: (set_of_layers, mask[B,T,T] bool) -- extra blocked attention edges.
        dropout : tensor rates [B, n_layers] (0 = no dropout at that block's output).
        add     : dict layer -> tensor [B,T,d] added to the residual after that block.
        """
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        resid = [h] if collect else None
        for l, blk in enumerate(self.blocks, start=1):
            em = knockout[1] if (knockout is not None and l in knockout[0]) else None
            h = blk(h, em)
            if add is not None and l in add:
                h = h + add[l]
            if dropout is not None:
                rate = dropout[:, l - 1].view(B, 1, 1)
                if (rate > 0).any():
                    u = torch.rand(h.shape, generator=generator, device=h.device)
                    keep = u >= rate
                    h = torch.where(keep, h / (1 - rate).clamp_min(1e-6), torch.zeros_like(h))
            if collect:
                resid.append(h)
        logits = self.unembed(self.ln_f(h))
        return (logits, resid) if collect else logits


def build_store(scfg, vocab_size):
    return Store(vocab_size, scfg["d_model"], scfg["n_layers"], scfg["n_heads"], scfg["mlp_width"])


def lm_loss(model, seqs):
    """Next-token cross-entropy over non-PAD targets. seqs: LongTensor [B,7]."""
    logits = model(seqs[:, :-1])
    tgt = seqs[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=W.PAD)


# ---------------------------------------------------------------- read-outs
def value_logits(logits_at_A, rels, vocab):
    """Restrict [B, V_total] logits at the [A] position to relation-specific value tokens -> [B, n_val]."""
    idx = torch.as_tensor(vocab.val0 + rels[:, None] * vocab.n_val + np.arange(vocab.n_val)[None, :])
    return torch.gather(logits_at_A, 1, idx)


@torch.no_grad()
def probe(model, world, items, batch=2048, names=None, knockout_fn=None, dropout=None, generator=None,
          answers=None, add_fn=None, want_states=True):
    """One forward pass per batch returning BOTH behaviour and read-set states (identical perturbation
    masks for both). Behaviour: correct, margin (log p(v*) - max other), logp_correct, probs [n,V]
    (restricted to V_r), entropy, pred. States: [n, 2*(L+1), d] (site s = 2*layer + pos_index).
    names  : optional override [n,2] (input corruption / interference entities).
    answers: optional override [n] of true answers.
    """
    model.eval()
    vocab = world.vocab
    out = {k: [] for k in ("correct", "margin", "logp_correct", "probs", "entropy", "pred")}
    states = []
    for i in range(0, len(items), batch):
        it = items[i:i + batch]
        nm = world.names[it[:, 0]] if names is None else names[i:i + batch]
        toks = torch.as_tensor(W.query_tokens(vocab, nm, it[:, 1]))
        ko = knockout_fn(i, len(it)) if knockout_fn is not None else None
        ad = add_fn(i, len(it)) if add_fn is not None else None
        dr = dropout[i:i + batch] if dropout is not None else None
        logits, resid = model(toks, knockout=ko, dropout=dr, collect=True, generator=generator, add=ad)
        if want_states:
            states.append(torch.stack([r[:, p] for r in resid for p in READ_POSITIONS], dim=1).numpy())
        vl = value_logits(logits[:, 4], it[:, 1], vocab)
        lp = torch.log_softmax(vl, -1)
        ans = torch.as_tensor(world.answers[it[:, 0], it[:, 1]] if answers is None else answers[i:i + batch])
        lpc = lp.gather(1, ans[:, None]).squeeze(1)
        other = lp.clone()
        other.scatter_(1, ans[:, None], float("-inf"))
        pred = lp.argmax(-1)
        out["correct"].append((pred == ans).numpy())
        out["margin"].append((lpc - other.max(-1).values).numpy())
        out["logp_correct"].append(lpc.numpy())
        p = lp.exp()
        out["probs"].append(p.numpy())
        out["entropy"].append(-(p * lp).sum(-1).numpy())
        out["pred"].append(pred.numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    if want_states:
        res["states"] = np.concatenate(states)
    return res


def answer_stats(model, world, items, **kw):
    return probe(model, world, items, want_states=False, **kw)


def read_states(model, world, items, **kw):
    return probe(model, world, items, want_states=True, **kw)["states"]


@torch.no_grad()
def name_fluency(model, world, names, batch=4096):
    """Operational familiarity validation: log p(s2 | [M], s1) under the store (behavioural)."""
    model.eval()
    vocab = world.vocab
    out = []
    for i in range(0, len(names), batch):
        nm = names[i:i + batch]
        toks = torch.as_tensor(np.stack([np.full(len(nm), W.M), vocab.syl(nm[:, 0])], axis=1))
        logits = model(toks)[:, 1]
        lp = torch.log_softmax(logits, -1)
        out.append(lp.gather(1, torch.as_tensor(vocab.syl(nm[:, 1]))[:, None]).squeeze(1).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- training
def train_store(world, scfg, seed, epochs, ckpt_fracs=(), eval_every=0, log=None, max_minutes=None):
    """Train a store from scratch. Returns (model, checkpoints{frac: state_dict}, curve)."""
    from .config import set_all_seeds
    set_all_seeds(seed)
    model = build_store(scfg, world.vocab.size)
    seqs, _, _ = training_sequences_cached(world)
    seqs_t = torch.as_tensor(seqs)
    opt_cfg = scfg["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    bs = opt_cfg["batch_size"]
    steps_per_epoch = math.ceil(len(seqs) / bs)
    total = steps_per_epoch * epochs
    floor = opt_cfg["lr_final"] / opt_cfg["lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ckpt_epochs = {max(1, int(round(f * epochs))): f for f in ckpt_fracs}
    ckpts, curve = {}, []
    g = torch.Generator().manual_seed(seed)
    known_items = W.items_where(world, known=True)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(seqs_t), generator=g)
        for i in range(0, len(perm), bs):
            loss = lm_loss(model, seqs_t[perm[i:i + bs]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if ep in ckpt_epochs:
            ckpts[ckpt_epochs[ep]] = copy.deepcopy(model.state_dict())
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            acc = answer_stats(model, world, known_items)["correct"].mean()
            curve.append({"epoch": ep, "trained_fact_acc": float(acc), "minutes": (time.perf_counter() - t0) / 60})
            if log:
                log(f"  epoch {ep}: trained-fact acc {acc:.4f} ({curve[-1]['minutes']:.1f} min)")
        if max_minutes and (time.perf_counter() - t0) / 60 > max_minutes:
            break
    return model, ckpts, curve


_SEQ_CACHE = {}


def training_sequences_cached(world):
    key = id(world)
    if key not in _SEQ_CACHE:
        _SEQ_CACHE[key] = W.training_sequences(world)
    return _SEQ_CACHE[key]


def store_qc(model, world, qcfg):
    """Store-level QC (no monitor quantities). Familiarity validated behaviourally by name fluency."""
    known = W.items_where(world, known=True)
    unknown = W.items_where(world, known=False)
    acc_known = float(answer_stats(model, world, known)["correct"].mean())
    acc_unknown = float(answer_stats(model, world, unknown)["correct"].mean())
    flu = name_fluency(model, world, world.names)
    from .metrics import auroc
    fam_auroc = float(auroc(flu[world.fam_high], flu[~world.fam_high]))
    res = {"trained_fact_acc": acc_known, "unknown_fact_acc": acc_unknown, "fluency_auroc_high_vs_low": fam_auroc,
           "fluency_mean_high": float(flu[world.fam_high].mean()), "fluency_mean_low": float(flu[~world.fam_high].mean())}
    res["pass"] = (acc_known >= qcfg["trained_fact_acc_min"] and acc_unknown <= qcfg["unknown_fact_acc_max"]
                   and fam_auroc >= qcfg["fluency_auroc_min"])
    return res


def site_stats(H):
    """Per-site, per-dimension mean and SD (floor 1e-3) from intact states [n, sites, d]."""
    return H.mean(0), np.maximum(H.std(0), 1e-3)
