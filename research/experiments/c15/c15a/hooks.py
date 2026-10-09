"""Forward passes with residual-stream edits and recording (block-output hooks, jlens convention)."""
from __future__ import annotations

import hashlib

import torch

from .gp import gp_nonneg, top_atoms


@torch.no_grad()
def forward(lm, ids, edits=None, record=(), attn=None, past=None):
    """Run the text decoder. `edits`: {layer: fn(h) -> h'} applied to the block output (in order of depth);
    `record`: layers whose (post-edit) block outputs are returned. Returns (final normed hidden, {layer: h})."""
    edits = edits or {}
    rec = {}
    handles = []

    def make(l):
        def hook(mod, inp, out):
            h = out if torch.is_tensor(out) else out[0]
            if l in edits:
                h = edits[l](h)
            if l in record:
                rec[l] = h.detach()
            if torch.is_tensor(out):
                return h
            return (h,) + tuple(out[1:])
        return hook

    for l in sorted(set(edits) | set(record)):
        handles.append(lm.layers[l].register_forward_hook(make(l)))
    try:
        if past is None:
            out = lm.text(input_ids=ids, attention_mask=attn, use_cache=False)
        else:
            out = lm.text(input_ids=ids, past_key_values=past, use_cache=True)
    finally:
        for hd in handles:
            hd.remove()
    return out.last_hidden_state, rec


@torch.no_grad()
def prefix_cache(lm, ids):
    """KV/recurrent cache after running `ids` (no edits)."""
    return lm.text(input_ids=ids, use_cache=True).past_key_values


@torch.no_grad()
def logits_at(lm, hidden_normed, pos):
    """Logits from the (already final-normed) hidden states at positions pos ([B] long or int)."""
    B = hidden_normed.shape[0]
    if isinstance(pos, int):
        h = hidden_normed[:, pos]
    else:
        h = hidden_normed[torch.arange(B), pos]
    return lm.lm_head(h).float()[:, : lm.n_vocab]


def add_at(pos, vecs):
    """Edit adding vecs[b] at position pos[b] (or a shared int position)."""
    def fn(h):
        h = h.clone()
        B = h.shape[0]
        p = torch.full((B,), pos, dtype=torch.long) if isinstance(pos, int) else pos
        h[torch.arange(B), p] += vecs.to(h.dtype)
        return h
    return fn


class JAblation:
    """Remove, at every position >= 1, the projection onto the span of the top-m GP_k atoms (per position).
    Records the removed norms (used to norm-match the random arm)."""

    def __init__(self, D, k, m, screen=None):
        self.D, self.k, self.m, self.screen = D, k, m, screen
        self.norms = None

    def __call__(self, h):
        B, T, d = h.shape
        H = h[:, 1:].reshape(-1, d).float()
        idx, coef, _ = gp_nonneg(H, self.D, self.k, screen=self.screen)
        top = top_atoms(idx, coef, self.m)                    # [n, m]
        A = self.D[top.clamp_min(0)] * (top >= 0)[..., None]  # [n, m, d]
        Q, _ = torch.linalg.qr(A.transpose(1, 2))             # [n, d, m]
        proj = torch.einsum("ndm,nm->nd", Q, torch.einsum("ndm,nd->nm", Q, H))
        out = h.clone()
        out[:, 1:] = (H - proj).reshape(B, T - 1, d).to(h.dtype)
        n = torch.zeros(B, T)
        n[:, 1:] = proj.norm(dim=1).reshape(B, T - 1)
        self.norms = n
        return out


def unit_directions(seed_key, shape):
    """Deterministic random unit vectors [.., d] from a string key."""
    s = int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16)
    g = torch.Generator().manual_seed(s)
    r = torch.randn(*shape, generator=g)
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class RandomDisplacement:
    """h' = h - s * n * r_hat at positions >= 1, with n the J-arm removed norm at the same (row, position)
    and r_hat a fixed random unit direction per (layer, item, position) -- identical across scales s."""

    def __init__(self, norms, dirs, scale):
        self.norms, self.dirs, self.scale = norms, dirs, scale

    def __call__(self, h):
        out = h.clone()
        disp = self.scale * self.norms[..., None] * self.dirs
        out[:, 1:] = (h[:, 1:].float() - disp[:, 1:]).to(h.dtype)
        return out
