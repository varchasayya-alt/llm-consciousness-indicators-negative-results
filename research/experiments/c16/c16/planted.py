"""Stage 1c planted system (prereg §5): a tiny transformer trained from scratch on a token-level copy of the numbers
domain, with the native gap built in (no latent-producer + non-copy-consumer training sequences).

All sentences are padded with a filler token to a fixed length SL, so prefix positions align exactly across latent,
text and bundle runs. Numbers are rendered as 3 digits. Every consumer op has two synonyms (V6 paraphrase control).
The planted model exposes the same subject interface as the HF subject (subject.py) plus workspace hooks.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import materials as M
from .subject import EpSpec, SentSpec

SL = 18                     # fixed sentence length
PREFIX = ["Q", "take", "<N>", "num", "."]
OPS = {"copy": ("what", "value"), "succ": ("plus1", "next"), "plus10": ("plus10", "addten"),
       "parity": ("parity", "evenodd"), "mag": ("mag", "size"), "lookup": ("code", "colour"),
       "verb": ("words", "spell")}
NAMES = [f"n{k}" for k in range(12)]
COLOURS = [f"c{k}" for k in range(6)]


def build_vocab():
    v = ["<pad>", "<bos>", "_"] + [str(k) for k in range(10)] + ["+", "-", "*"] + NAMES
    v += ["is", ".", "Q", "take", "num", "?", "A", ":", "codes", ",", ";", "p1", "p10", "and"]
    v += [t for syn in OPS.values() for t in syn]
    v += ["even", "odd", "larger", "smaller"] + COLOURS
    v += [f"t{k}" for k in range(10, 20)] + [f"T{k}" for k in range(2, 10)] + [f"o{k}" for k in range(1, 10)]
    return {t: i for i, t in enumerate(v)}


VOCAB = build_vocab()


def num3(x):
    return list(f"{x:03d}")


def ans_tokens(consumer, x, table=None):
    if consumer in ("copy", "succ", "plus10"):
        return num3({"copy": x, "succ": x + 1, "plus10": x + 10}[consumer])
    if consumer == "parity":
        return ["even" if x % 2 == 0 else "odd"]
    if consumer == "mag":
        return ["larger" if x > 50 else "smaller"]
    if consumer == "lookup":
        return [dict(table)[x]]
    if consumer == "verb":
        if 10 <= x <= 19:
            return [f"t{x}"]
        t, o = divmod(x, 10)
        return [f"T{t}"] + ([f"o{o}"] if o else [])
    raise ValueError(consumer)


def sentence_tokens(s: SentSpec):
    n = s.name
    if s.kind == "latent":
        p, args = s.payload
        a, b = args
        op = {"add": "+", "sub": "-", "mul": "*"}[p]
        core = ["is"] + num3(a) + [op, str(b), "."]
    elif s.kind == "text":
        core = ["is"] + num3(s.payload[0]) + ["."]
    else:
        bname, x = s.payload
        par = "even" if x % 2 == 0 else "odd"
        mag = "larger" if x > 50 else "smaller"
        if bname == "B1":
            core = ["p1", "is"] + num3(x + 1) + ["."]
        elif bname == "B2":
            core = ["p1", "is"] + num3(x + 1) + [",", "p10", "is"] + num3(x + 10) + ["."]
        elif bname == "B4":
            core = ["p1", "is"] + num3(x + 1) + [",", "p10", "is"] + num3(x + 10) + [";", par, "and", mag, "."]
        elif bname == "Bpm":
            core = ["is", par, "and", mag, "."]
        else:
            raise ValueError(bname)
    pad = SL - 1 - len(core)
    assert pad >= 0, core
    return [n] + ["_"] * pad + core


def question_tokens(consumer, table=None, syn=0):
    q = []
    if consumer == "lookup":
        q += ["codes"]
        for v, c in table:
            q += num3(v) + ["is", c, ","]
    q += [OPS[consumer][syn], "?"]
    return q


def episode_tokens(ep: EpSpec, ans=None, syn=0):
    toks = ["<bos>"] + sentence_tokens(ep.sent)
    prefix_start = len(toks)
    toks += [t if t != "<N>" else ep.sent.name for t in PREFIX]
    toks += question_tokens(ep.consumer, ep.table, syn) + ["A", ":"]
    ans_start = len(toks)
    if ans is not None:
        toks += ans
    sent_pos = list(range(1, 1 + SL))
    return toks, list(range(prefix_start, prefix_start + len(PREFIX))), sent_pos, ans_start


def cue_tokens(consumer, syn=0):
    q = ["codes"] if consumer == "lookup" else []
    return ["<bos>", "Q", "take", "num", "."] + q + [OPS[consumer][syn], "?"]


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal, pad):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, attn_mask=causal, key_padding_mask=pad, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.d, self.n_layers = d, layers

    def forward(self, ids, pad_mask, patch=None, read_fn=None, keep=None):
        """patch: (layer, rows, cols, vecs) added to the residual entering `layer`.
        read_fn(layer, h) -> h: workspace reads applied to the residual entering `layer`.
        keep: list of layers whose entering residual is returned."""
        B, L = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(L))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        hs = {}
        for l, blk in enumerate(self.blocks):
            if patch is not None and patch[0] == l:
                x = x.clone()
                x[patch[1], patch[2]] += patch[3]
            if read_fn is not None:
                x = read_fn(l, x)
            if keep is not None and l in keep:
                hs[l] = x
            x = blk(x, causal, pad_mask)
        if keep is not None and self.n_layers in keep:
            hs[self.n_layers] = x
        logits = self.lnf(x) @ self.emb.weight.T
        return logits, hs


def to_ids(toks):
    return [VOCAB[t] for t in toks]


def batchify(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    pad = torch.ones((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        pad[i, :len(s)] = False
    return ids, pad


# ----------------------------------------------------------------- data for the base model
def planted_split(seed):
    xs = list(range(10, 100))
    xs.remove(50)
    rng = random.Random(seed)
    rng.shuffle(xs)
    return sorted(xs[:45]), sorted(xs[45:])


def random_args(producer, x, rng):
    if producer == "add":
        b = rng.randint(2, min(9, x - 2))
        return (x - b, b)
    if producer == "sub":
        b = rng.randint(2, 9)
        return (x + b, b)
    pairs = M.mul_pairs(x)
    return rng.choice(pairs) if pairs else None


def planted_instances(xs, n_per, seed, producers=("add", "sub", "mul"), tag="pl"):
    """Planted analogue of materials.build_instances (planted names/colours; planted value split)."""
    rng = random.Random(seed)
    out = []
    for x in sorted(xs):
        for p in producers:
            if p == "mul" and not M.mul_pairs(x):
                continue
            for k in range(n_per):
                out.append(M.Instance(f"{tag}-{p}-{x}-{k}", x, p, rng.choice(NAMES), random_args(p, x, rng),
                                      list(random_table(x, xs, rng))))
    return out


def random_table(x, pool, rng):
    d = rng.sample([v for v in pool if v != x], 2)
    vals = [x] + d
    rng.shuffle(vals)
    return tuple(zip(vals, rng.sample(COLOURS, 3)))


def base_sample(rng, xs):
    """One training sequence for the base model (no latent producer + non-copy consumer)."""
    x = rng.choice(xs)
    name = rng.choice(NAMES)
    syn = rng.randint(0, 1)
    kind = rng.random()
    if kind < 0.30:
        p = rng.choice(["add", "sub", "mul"])
        args = random_args(p, x, rng)
        if args is None:
            p, args = "add", random_args("add", x, rng)
        ep = EpSpec(SentSpec("latent", name, (p, args)), "copy", None, {"own": x})
    elif kind < 0.80:
        c = rng.choice(list(OPS))
        tbl = random_table(x, xs, rng) if c == "lookup" else None
        ep = EpSpec(SentSpec("text", name, (x,)), c, tbl, {"own": x})
    else:
        b = rng.choice(["B1", "B2", "B4", "Bpm"])
        cons = {"B1": ["succ"], "B2": ["succ", "plus10"], "B4": ["succ", "plus10", "parity", "mag"],
                "Bpm": ["parity", "mag"]}[b]
        ep = EpSpec(SentSpec("bundle", name, (b, x)), rng.choice(cons), None, {"own": x})
    ans = ans_tokens(ep.consumer, x, ep.table)
    toks, _, _, a0 = episode_tokens(ep, ans, syn)
    return to_ids(toks), a0, len(ans)


def train_base(seed, steps=6000, batch=256, lr=3e-3, log=print, d=128, layers=4, heads=4):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    xs = [x for x in range(10, 100) if x != 50]
    model = TinyGPT(len(VOCAB), d, layers, heads)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        samples = [base_sample(rng, xs) for _ in range(batch)]
        ids, pad = batchify([s[0] for s in samples])
        logits, _ = model(ids, pad)
        loss = 0.0
        tgt_rows, tgt_cols, tgt = [], [], []
        for i, (_, a0, na) in enumerate(samples):
            for k in range(na):
                tgt_rows.append(i)
                tgt_cols.append(a0 + k - 1)
                tgt.append(ids[i, a0 + k])
        loss = F.cross_entropy(logits[tgt_rows, tgt_cols], torch.stack(tgt))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"  base step {step} loss {loss.item():.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------- subject
class PlantedSubject:
    def __init__(self, model, syn=0):
        self.m = model
        self.n_layers = model.n_layers
        self.d_model = model.d
        self.syn = syn
        self.bs = 256

    def _run(self, eps_toks, patch=None, keep=None, read_fn=None):
        ids, pad = batchify([to_ids(t) for t in eps_toks])
        with torch.no_grad():
            return self.m(ids, pad, patch=patch, keep=keep, read_fn=read_fn)

    def prefix_hidden(self, sents, layers, key="prefix", last=None):
        out = []
        for s0 in range(0, len(sents), self.bs):
            chunk = sents[s0:s0 + self.bs]
            toks, pos = [], None
            for s in chunk:
                t, ppos, spos, _ = episode_tokens(EpSpec(s, "copy", None, {}), None, self.syn)
                toks.append(t)
                pos = ppos if key == "prefix" else spos[-last:]
            _, hs = self._run(toks, keep=layers)
            out.append(torch.stack([hs[l][:, pos] for l in layers], 1))
        return torch.cat(out, 0)

    def prod_hidden(self, sents, layers, last=3):
        return self.prefix_hidden(sents, layers, key="sent", last=last)

    def evaluate(self, eps, layer=None, deltas=None, read_fn_factory=None, syn=None):
        syn = self.syn if syn is None else syn
        res = [dict() for _ in eps]
        jobs = []
        for i, ep in enumerate(eps):
            for lab, x in ep.cands.items():
                jobs.append((i, lab, ans_tokens(ep.consumer, x, ep.table)))
        for s0 in range(0, len(jobs), self.bs):
            chunk = jobs[s0:s0 + self.bs]
            toks, a0s, ppos = [], [], None
            for i, lab, ans in chunk:
                t, pp, _, a0 = episode_tokens(eps[i], ans, syn)
                toks.append(t)
                a0s.append(a0)
                ppos = pp
            patch = None
            if deltas is not None:
                rows, cols, vecs = [], [], []
                for b, (i, lab, ans) in enumerate(chunk):
                    if deltas[i] is None:
                        continue
                    for k, p in enumerate(ppos):
                        rows.append(b)
                        cols.append(p)
                        vecs.append(deltas[i][k])
                patch = (layer, torch.tensor(rows), torch.tensor(cols), torch.stack(vecs))
            read_fn = None if read_fn_factory is None else read_fn_factory([c[0] for c in chunk], ppos[0])
            logits, _ = self._run(toks, patch=patch, read_fn=read_fn)
            am = logits.argmax(-1)
            for b, (i, lab, ans) in enumerate(chunk):
                a0 = a0s[b]
                pred = am[b, a0 - 1:a0 - 1 + len(ans)].tolist()
                res[i][lab] = pred == to_ids(ans)
        return res

    def cue_rep(self, consumer, layer, syn=0):
        t = cue_tokens(consumer, syn)
        _, hs = self._run([t], keep=[layer])
        return hs[layer][0, 1:].mean(0)
