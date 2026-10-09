"""FEASIBILITY ONLY -- not a preregistered experiment.

Question: can a tiny causal transformer memorise a synthetic entity-relation-value
fact base on the researcher's laptop CPU in a practical time?

No monitor, no lesion, no confidence analysis is run here, so no confirmatory
hypothesis is examined. Uses a throwaway world seed (12345) that will NOT be
reused in any preregistered experiment.

Output: research/results/raw/feasibility/store_training_feasibility.json
"""
import json
import os
import time

import torch
import torch.nn as nn

SEED = 12345
N_ENT, N_REL, N_VAL, N_SYL = 1000, 4, 50, 64
D_MODEL, N_LAYERS, N_HEADS = 128, 3, 4
BATCH, MAX_MIN, TARGET_ACC = 512, 25, 0.95

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "raw", "feasibility", "store_training_feasibility.json")

g = torch.Generator().manual_seed(SEED)
torch.manual_seed(SEED)

# vocab: [PAD]=0, [Q]=1, [A]=2, syllables, relations, values
Q, A = 1, 2
SYL0 = 3
REL0 = SYL0 + N_SYL
VAL0 = REL0 + N_REL
VOCAB = VAL0 + N_REL * N_VAL

# entities are 2-syllable names (forces composition, like multi-token subjects)
names = torch.randperm(N_SYL * N_SYL, generator=g)[:N_ENT]
ent_tok = torch.stack([SYL0 + names // N_SYL, SYL0 + names % N_SYL], dim=1)
values = torch.randint(0, N_VAL, (N_ENT, N_REL), generator=g)

ents = torch.arange(N_ENT).repeat_interleave(N_REL)
rels = torch.arange(N_REL).repeat(N_ENT)
seqs = torch.stack([torch.full_like(ents, Q), ent_tok[ents, 0], ent_tok[ents, 1],
                    REL0 + rels, torch.full_like(ents, A)], dim=1)          # [N, 5]
targets = VAL0 + rels * N_VAL + values[ents, rels]                           # answer token
N = seqs.shape[0]


class Store(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(8, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, 4 * D_MODEL, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.out = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(L))
        mask = nn.Transformer.generate_square_subsequent_mask(L)
        return self.out(self.enc(h, mask=mask, is_causal=True)[:, -1])


model = Store()
params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
log, t0, epoch, acc = [], time.perf_counter(), 0, 0.0
while True:
    epoch += 1
    perm = torch.randperm(N)
    model.train()
    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        loss = nn.functional.cross_entropy(model(seqs[idx]), targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(seqs).argmax(-1) == targets).float().mean().item()
        el = time.perf_counter() - t0
        log.append({"epoch": epoch, "train_fact_acc": round(acc, 4), "elapsed_min": round(el / 60, 2)})
        print(log[-1], flush=True)
        if acc >= TARGET_ACC or el > MAX_MIN * 60:
            break

res = {"purpose": "feasibility only (training time); no hypothesis tested", "seed": SEED,
       "n_facts": N, "params": params, "d_model": D_MODEL, "n_layers": N_LAYERS,
       "reached_target": acc >= TARGET_ACC, "log": log}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("wrote", OUT)
