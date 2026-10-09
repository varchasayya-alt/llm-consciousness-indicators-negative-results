"""Task-independent workspace construction: band layers, generic dictionary V_gen, S_J (generic-PCA of
J-reconstructions). Inputs are the published lens, the unembedding and G_select generic paragraphs only."""
from __future__ import annotations

import math
import re
from collections import Counter

import torch

from . import config as C
from .gp import gp_nonneg
from .lens import atoms

_WORD = re.compile(r"^[A-Za-z]{2,}$")


def _round(x):
    return int(math.floor(x + 0.5))


def band_layers(n_layers):
    lo, hi = _round(C.BAND_LO * n_layers), _round(C.BAND_HI * n_layers)
    return list(range(lo, hi + 1, C.BAND_STEP))


def word_initial_ids(tok, n_vocab):
    """Token ids whose surface form is one leading space + >= 2 ASCII letters (byte-level BPE 'Ġ')."""
    toks = tok.convert_ids_to_tokens(list(range(n_vocab)))
    out = []
    for i, t in enumerate(toks):
        if t and t.startswith("Ġ") and _WORD.match(t[1:]):
            out.append(i)
    return out


def vgen_ids(tok, n_vocab, select_token_lists):
    """V_gen = word-initial ids minus the VGEN_DROP_TOP most frequent ones in G_select (ties by id)."""
    base = word_initial_ids(tok, n_vocab)
    bs = set(base)
    cnt = Counter(t for ids in select_token_lists for t in ids if t in bs)
    drop = {t for t, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[: C.VGEN_DROP_TOP]}
    return [t for t in base if t not in drop], sorted(drop)


class Dictionaries:
    """Per-layer unit-atom dictionaries over V_gen, cached (small LRU to bound memory)."""

    def __init__(self, lens, w_eff_vgen, max_cached=4):
        self.lens, self.W, self.max = lens, w_eff_vgen, max_cached
        self._cache = {}

    def __call__(self, layer):
        if layer not in self._cache:
            if len(self._cache) >= self.max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[layer] = atoms(self.lens.J(layer), self.W)
        return self._cache[layer]


@torch.no_grad()
def build_sj(states, D, d_model, screen=None):
    """S_J from generic states [n, d]: PCA of centred GP_25 reconstructions; r = min(r90, floor(d/8))."""
    _, _, recon = gp_nonneg(states, D, C.GP_K, screen=screen)
    Rc = recon - recon.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Rc, full_matrices=False)
    var = S ** 2
    cum = torch.cumsum(var, 0) / var.sum()
    r90 = int((cum < C.SJ_VAR).sum().item()) + 1
    r = min(r90, d_model // C.SJ_CAP_DIV)
    Q = Vh[:r].T.contiguous()                            # [d, r] orthonormal
    Sc = states - states.mean(0, keepdim=True)
    frac_state = float(((Sc @ Q) ** 2).sum() / (Sc ** 2).sum())
    frac_recon = float(cum[r - 1])
    recon_r2 = float(1 - ((states - recon) ** 2).sum() / (Sc ** 2).sum())
    recon_fit_raw = float(1 - ((states - recon) ** 2).sum() / (states ** 2).sum())
    return Q, {"r90": r90, "r": r, "frac_recon_var": frac_recon, "frac_state_var_in_SJ": frac_state,
               "gp_recon_r2_centred": recon_r2, "gp_recon_fit_raw": recon_fit_raw, "n_states": int(states.shape[0])}
