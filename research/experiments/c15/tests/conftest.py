import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class FakeTok:
    """Word-level byte-BPE-like tokenizer: ' word' -> 'Ġword' (one token), punctuation separate."""

    def __init__(self, n=1024):
        self.n, self.vocab, self.inv = n, {"<pad>": 0}, ["<pad>"]
        self.bos_token = None

    def _id(self, t):
        if t not in self.vocab:
            assert len(self.inv) < self.n, "fake vocab exhausted"
            self.vocab[t] = len(self.inv)
            self.inv.append(t)
        return self.vocab[t]

    def __call__(self, text, add_special_tokens=True):
        toks = re.findall(r" ?[A-Za-z]+| ?[^A-Za-z\s]|\n", text)
        return SimpleNamespace(input_ids=[self._id(t.replace(" ", "Ġ")) for t in toks])

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] if i < len(self.inv) else "<unk>" for i in ids]

    def __len__(self):
        return self.n


def tiny_lm(n_layers=6, d=64, vocab=1024, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from c15a.artifacts import wrap_hf
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=d, intermediate_size=2 * d, num_hidden_layers=n_layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=d // 4, max_position_embeddings=256,
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
    lm = tiny_lm()
    return lm, Lens(tiny_lens_ck())
