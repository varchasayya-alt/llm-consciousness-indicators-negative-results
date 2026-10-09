"""Abstract episode specs shared by the HF subject (Qwen2.5-0.5B) and the planted subject.

SentSpec:  ("latent", name, (producer, args)) | ("text", name, x) | ("bundle", name, (bundle, x))
EpSpec:    (sent: SentSpec, consumer: str, table: list[(value, colour)] | None, cands: {label: x_value})
           cands maps a label (e.g. 'own', 'swap') to the X whose consumer-answer is checked.
A subject implements:
    n_layers, d_model
    prefix_hidden(sents, layers) -> FloatTensor [n, len(layers), P, d]     (prefix positions)
    prod_hidden(sents, layers, last) -> FloatTensor [n, len(layers), last, d]
    evaluate(eps, layer=None, deltas=None) -> list[{label: bool}]           (greedy correctness, teacher forced)
deltas[i] is a [P, d] tensor added to the residual entering `layer` at episode i's prefix positions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SentSpec:
    kind: str
    name: str
    payload: tuple


@dataclass
class EpSpec:
    sent: SentSpec
    consumer: str
    table: tuple
    cands: dict


def latent(name, producer, args):
    return SentSpec("latent", name, (producer, tuple(args)))


def text(name, x):
    return SentSpec("text", name, (x,))


def bundle(name, b, x):
    return SentSpec("bundle", name, (b, x))
