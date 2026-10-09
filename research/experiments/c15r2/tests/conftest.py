import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
sys.path.insert(0, R2)
import c15a2  # noqa: E402,F401  (byte-code off; A-stage import path)


class FakeTok:
    """Word-level byte-BPE-like tokenizer: ' word' -> 'Ġword' (one token), punctuation separate."""

    def __init__(self, n=8192):
        self.n, self.vocab, self.inv = n, {"<pad>": 0}, ["<pad>"]
        self.bos_token = None

    def _id(self, t):
        if t not in self.vocab:
            assert len(self.inv) < self.n, "fake vocab exhausted"
            self.vocab[t] = len(self.inv)
            self.inv.append(t)
        return self.vocab[t]

    def __call__(self, text, add_special_tokens=True):
        toks = re.findall(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]|\n", text)
        return SimpleNamespace(input_ids=[self._id(t.replace(" ", "Ġ")) for t in toks])

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] if i < len(self.inv) else "<unk>" for i in ids]

    def __len__(self):
        return self.n


def tiny_lm(n_layers=6, d=64, vocab=8192, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from c15a.artifacts import wrap_hf
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=d // 4, max_position_embeddings=512,
                      tie_word_embeddings=True)
    hf = Qwen3ForCausalLM(cfg).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    return wrap_hf("tiny", hf, FakeTok(vocab))


def tiny_lens_ck(n_layers=6, d=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    J = {l: (torch.eye(d) + 0.1 * torch.randn(d, d, generator=g)).half() for l in range(n_layers - 1)}
    return {"J": J, "n_prompts": 10, "source_layers": list(range(n_layers - 1)), "d_model": d}


@pytest.fixture(scope="session")
def tiny():
    from c15a.lens import Lens
    return tiny_lm(), Lens(tiny_lens_ck())


# ------------------------------------------------------------------ planted-transport model (PC4)
def _causal_mean(x):
    T = x.shape[1]
    return torch.cumsum(x, dim=1) / torch.arange(1, T + 1, dtype=x.dtype)[None, :, None]


class Identity(torch.nn.Module):
    def forward(self, h):
        return h


class CopyBlock(torch.nn.Module):
    """Planted copy mechanism: every position receives the causal mean of the concept-subspace content of all
    positions up to itself (identity-preserving transport of concept content)."""

    def __init__(self, P, beta):
        super().__init__()
        self.P, self.beta = P, beta

    def forward(self, h):
        return h + self.beta * _causal_mean(h @ self.P.T)


class MixBlock(torch.nn.Module):
    """Generic (identity-scrambling) causal mixing: spreads any content downstream through a random map."""

    def __init__(self, G, beta):
        super().__init__()
        self.G, self.beta = G, beta

    def forward(self, h):
        return h + self.beta * _causal_mean(h @ self.G.T)


class PlantedText(torch.nn.Module):
    def __init__(self, emb, layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding.from_pretrained(emb, freeze=True)
        self.layers = layers

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)
        return SimpleNamespace(last_hidden_state=h)


def planted_lm(w3, d=32, L=8, inj_layer=2, alpha=3.0, beta=4.0, beta_mix=2.0, seed=0):
    """Concept directions c_k = e_k (k < 8). Tokens unique to concept k's paraphrases (and its word) carry
    alpha * c_k. Block inj_layer+1 copies concept content downstream; block inj_layer+2 mixes generically."""
    g = torch.Generator().manual_seed(seed)
    tok = FakeTok(8192)
    concepts = w3["concepts"]
    bags = {}
    for c in concepts:
        ids = set()
        for p in w3["paraphrases"][c] + ["the " + w3["words"][c]]:
            ids |= set(tok(" " + p, add_special_tokens=False).input_ids)
        bags[c] = ids
    ctx_ids = set()
    for r in w3["contexts"]:
        ctx_ids |= set(tok(r["prefix"] + w3["carrier"] + r["cont"], add_special_tokens=True).input_ids)
    emb = torch.randn(8192, d, generator=g) / d ** 0.5
    emb[:, :len(concepts)] = 0.0
    for k, c in enumerate(concepts):
        others = set().union(*[bags[o] for o in concepts if o != c]) | ctx_ids
        for t in bags[c] - others:
            emb[t, k] += alpha
    P = torch.zeros(d, d)
    P[:len(concepts), :len(concepts)] = torch.eye(len(concepts))
    G = torch.randn(d, d, generator=g) / d ** 0.5
    blocks = [Identity() for _ in range(L)]
    blocks[inj_layer + 1] = CopyBlock(P, beta)
    blocks[inj_layer + 2] = MixBlock(G, beta_mix)
    layers = torch.nn.ModuleList(blocks)
    text = PlantedText(emb, layers)
    lm = SimpleNamespace(key="planted", tok=tok, text=text, layers=layers, n_layers=L, d_model=d, n_vocab=8192,
                         lm_head=torch.nn.Linear(d, 8192, bias=False))
    return lm, P
