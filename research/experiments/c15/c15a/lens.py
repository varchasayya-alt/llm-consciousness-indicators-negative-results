"""Jacobian lens (Gurnee et al. 2026; reference implementation github.com/anthropics/jacobian-lens).

Conventions (matching jlens): the residual at layer l is the output of decoder block l; J_l maps it into the
output basis of the final block; readout = lm_head(final_norm(J_l @ h)). Atom (J-lens vector) for token t at
layer l: j_{l,t} = J_l^T (gamma * W_U[t]), unit-normalised (gamma = final-norm gain).
"""
from __future__ import annotations

import torch


class Lens:
    def __init__(self, ck):
        self._J = {int(k): v for k, v in ck["J"].items()}   # stored fp16, cast per use
        self.source_layers = sorted(self._J)
        self.d_model = int(ck["d_model"])

    def J(self, layer):
        return self._J[layer].float()

    def transport(self, h, layer):
        return h @ self.J(layer).T

    @torch.no_grad()
    def readout(self, lm, h, layer):
        """Lens logits for residuals h [..., d] at `layer` (jlens `apply` semantics)."""
        return lm.unembed(self.transport(h, layer)).float()


def atoms(J, w_eff_rows):
    """Unit-norm J-lens vectors for the given unembedding rows: [n, d] (rows = (W J) normalised)."""
    A = w_eff_rows @ J
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
