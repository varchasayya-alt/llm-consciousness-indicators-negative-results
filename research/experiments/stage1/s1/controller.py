"""ANSWER/LOOKUP controller. Trained on the INTACT store only (CT split).

Utilities: ANSWER = +1 if the store is correct, -lambda if wrong; LOOKUP = 1 - c (external memory is
always correct). With a calibrated estimate p, the Bayes-optimal rule answers iff
p > (1 - c + lambda) / (1 + lambda)   (= 0.85 at c = 0.3, lambda = 1).
"""
import numpy as np
import torch
import torch.nn as nn

from .metrics import logit


def bayes_threshold(c, lam):
    return (1 - c + lam) / (1 + lam)


class Controller(nn.Module):
    def __init__(self, extra_dim=0, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + extra_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, p_hat_logit, c, extra=None):
        x = [p_hat_logit[:, None], c[:, None]]
        if extra is not None:
            x.append(extra)
        return torch.sigmoid(self.net(torch.cat(x, -1)).squeeze(-1))    # P(LOOKUP)


def train_controller(p_hat, correct, ccfg, seed, extra=None):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Controller(extra_dim=0 if extra is None else extra.shape[1], hidden=ccfg["hidden"])
    z = torch.as_tensor(logit(p_hat), dtype=torch.float32)
    y = torch.as_tensor(correct.astype(np.float32))
    ex = None if extra is None else torch.as_tensor(extra, dtype=torch.float32)
    lo, hi = ccfg["cost_train_range"]
    lam = ccfg["lambda_wrong"]
    opt = torch.optim.AdamW(model.parameters(), lr=ccfg["lr"])
    bs = ccfg["batch_size"]
    for _ in range(ccfg["steps"]):
        idx = torch.randint(0, len(y), (bs,), generator=g)
        c = lo + (hi - lo) * torch.rand(bs, generator=g)
        a = model(z[idx], c, None if ex is None else ex[idx])
        util = a * (1 - c) + (1 - a) * (y[idx] - lam * (1 - y[idx]))
        loss = -util.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def lookup_prob(model, p_hat, c, extra=None):
    z = torch.as_tensor(logit(p_hat), dtype=torch.float32)
    cc = torch.full((len(z),), float(c))
    ex = None if extra is None else torch.as_tensor(extra, dtype=torch.float32)
    return model(z, cc, ex).numpy()
