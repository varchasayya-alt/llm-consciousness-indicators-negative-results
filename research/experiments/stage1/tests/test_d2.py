"""Stage-1 D2 (self-generated competence change; store-only kill test): structural tests required before any D2 run
(PI 2026-10-03, item 17). Seed namespace; input identity; no replay of original knowledge; the matched-development
control (yoked steps, evaluated facts excluded); bookkeeping from world data only; no B1 / monitor code."""
import ast
import copy
import os

import numpy as np
import pytest
import torch

from s1 import world as W
from s1.d2 import (bookkeeping_features, control_continuation, control_sequences, eval_tokens, population,
                   steps_used)
from s1.interventions import interference, interference_data
from s1.store import build_store, probe

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 31)
    torch.manual_seed(0)
    m = build_store(tiny_cfg["store"], w.vocab.size)
    m.eval()
    return w, m


def test_d2_seed_namespace_is_fresh(cfg):
    import run_d2
    sd = cfg["seeds"]
    dev, val = set(sd["d2_development"]), set(sd["d2_validation_proposed"])
    assert dev == {9101, 9102, 9103} == set(run_d2.DEV_SEEDS) and val == {9111, 9112, 9113} == set(run_d2.VAL_SEEDS)
    used = set(range(9001, 9074)) | {9051, 9052, 9053, 12345} | set(range(1001, 1041)) | set(sd["retired"])
    assert not ((dev | val) & used) and not (dev & val)
    for k in ("v4_validation", "v42_development", "confirmatory"):
        assert not ((dev | val) & set(sd[k] if k != "confirmatory" else range(sd[k][0], sd[k][1] + 1)))


def test_input_identity_evaluation_tokens_depend_on_world_only(tiny, tiny_cfg):
    w, m = tiny
    kn = W.items_where(w, known=True)[:40]
    t0 = eval_tokens(w, kn)
    icfg = dict(tiny_cfg["d2"]["interference"], lr=1e-3, max_steps=4, eval_every=2)
    m2, _ = interference(m, w, icfg, seed=1)
    t1 = eval_tokens(w, kn)
    assert np.array_equal(t0, t1) and t0.shape[1] == 5 and (t0[:, 4] == W.A).all()
    assert np.array_equal(t0, W.query_tokens(w.vocab, w.names[kn[:, 0]], kn[:, 1]))   # what probe() evaluates
    assert m2.tok.weight.shape == m.tok.weight.shape                                   # vocabulary unchanged


def test_interference_has_no_replay_of_original_knowledge(tiny, tiny_cfg):
    w, _ = tiny
    seqs, _, _ = interference_data(w, tiny_cfg["d2"]["interference"])
    orig = {tuple(n) for n in w.names}
    new = {tuple(n) for n in w.interf_names}
    assert not (orig & new)                                                            # disjoint name pools
    trained_names = {(int(a), int(b)) for a, b in seqs[:, 1:3]}
    syl = lambda pairs: {(int(w.vocab.syl(np.array([a]))[0]), int(w.vocab.syl(np.array([b]))[0])) for a, b in pairs}
    assert not (trained_names & syl(orig))                                             # no original entity sequence
    assert trained_names <= syl(new)


def test_control_excludes_evaluated_facts_and_is_yoked(tiny, tiny_cfg):
    w, m = tiny
    cs = control_sequences(w)
    ev = W.items_where(w, split=W.EV, known=True)
    ev_seqs = {tuple(r) for r in W.fact_seqs(w.vocab, w.names[ev[:, 0]], ev[:, 1], w.answers[ev[:, 0], ev[:, 1]])}
    assert not ({tuple(r) for r in cs} & ev_seqs)                                       # evaluated facts never replayed
    mt = W.items_where(w, split=W.MT, known=True)
    mt_seq = tuple(W.fact_seqs(w.vocab, w.names[mt[:1, 0]], mt[:1, 1], w.answers[mt[:1, 0], mt[:1, 1]])[0])
    assert mt_seq in {tuple(r) for r in cs}                                             # already-known material
    icfg = dict(tiny_cfg["d2"]["interference"], lr=1e-3)
    calls = []
    import s1.d2 as d2mod
    orig = d2mod._train_on

    def spy(store0, seq_fn, steps, lr, seed, world, *a, **k):
        calls.append((steps, lr))
        return orig(store0, seq_fn, steps, lr, seed, world, *a, **k)
    d2mod._train_on = spy
    try:
        control_continuation(m, w, 7, icfg, seed=2)
    finally:
        d2mod._train_on = orig
    assert calls == [(7, 1e-3)]                                                         # same steps and lr as INTERF
    assert steps_used([{"step": 50}, {"step": 100}], 3000) == 100 and steps_used([], 3000) == 3000


def test_bookkeeping_features_are_world_only_and_deterministic(tiny):
    w, _ = tiny
    it = W.items_where(w, known=True)[:30]
    b1, b2 = bookkeeping_features(w, it), bookkeeping_features(copy.deepcopy(w), it)
    assert b1.shape == (30, 5) and np.array_equal(b1, b2)


def test_population_is_base_correct_EV(tiny):
    w, m = tiny
    pop = population(m, w)
    if len(pop):
        assert (w.split[pop[:, 0], pop[:, 1]] == W.EV).all() and w.known[pop[:, 0], pop[:, 1]].all()
        assert probe(m, w, pop)["correct"].all()


def test_d2_code_has_no_B1_or_monitor_components():
    forbidden = {"s1.memstore", "s1.v4qc", "s1.monitors", "s1.controller", "s1.pipeline", ".memstore", ".v4qc",
                 ".monitors", ".controller", ".pipeline", "memstore", "v4qc", "monitors", "controller", "pipeline"}
    for f in ("run_d2.py", "s1/d2.py"):
        tree = ast.parse(open(os.path.join(STAGE, f), encoding="utf-8").read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not (mods & forbidden), (f, mods & forbidden)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not (names & {"delete_slots", "transplant", "mem_values", "DualStore", "forget_with_sham", "unlearn"})


def test_svd_fallback_gives_identical_ridge_predictions(monkeypatch):
    """D66: on LAPACK gesdd non-convergence, _ridge_path falls back to gesvd; predictions must match the default."""
    from s1 import estimands as ES
    rng = np.random.default_rng(0)
    A, y, B = rng.normal(size=(120, 300)), rng.normal(size=120), rng.normal(size=(30, 300))
    ref = ES._ridge_path(A, y - y.mean(), B, [0.1, 10.0])
    real = np.linalg.svd

    def failing(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge")
    monkeypatch.setattr(np.linalg, "svd", failing)
    alt = ES._ridge_path(A, y - y.mean(), B, [0.1, 10.0])
    monkeypatch.setattr(np.linalg, "svd", real)
    for r, a in zip(ref, alt):
        assert np.allclose(r, a, atol=1e-8)
