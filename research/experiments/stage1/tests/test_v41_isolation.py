"""v4.1 gradient isolation (D52; PI 2026-10-03, item 4): structural invariants that must hold before any v4.1 run.

Groups: P = parametric LM pathway, M = explicit memory mechanism (memstore.param_groups).
  memory-covered, slot AVAILABLE  -> answer/LM loss reaches M only; P gets exactly zero gradient
  memory-covered, ROUTE-DROPPED   -> answer/LM loss reaches P; no answer gradient to any M tensor; retrieval (NULL)
                                     supervision reaches only the addressing subset (keys, q_ln, W_q, log_scale)
  parametric-only facts, mentions -> ordinary LM gradients to P
plus one-optimizer-step tests (P bit-identical after a memory-present step; values / W_o bit-identical after a
route-dropped step).
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from s1 import world as W
from s1.memstore import (ADDRESSING_PARAMS, MEMORY_PARAMS, build_dual_store, init_keys_from_queries, isolation_audit,
                         param_groups, slot_table, train_dual_store, train_step, v41_loss)
from s1.store import lm_loss


@pytest.fixture(scope="module")
def setup(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 11)
    torch.manual_seed(0)
    m = build_dual_store(tiny_cfg["store"], tiny_cfg["memory"], w)
    init_keys_from_queries(m, w)
    seqs, _, idx = W.training_sequences(w)
    slot, _ = slot_table(w)
    rs = torch.as_tensor(np.where(idx >= 0, slot.reshape(-1)[np.maximum(idx, 0)], -1))
    x = torch.as_tensor(seqs)
    is_fact = idx >= 0
    rows = {"covered": np.nonzero(is_fact & (rs.numpy() >= 0))[0][:48],
            "param_only": np.nonzero(is_fact & (rs.numpy() < 0))[0][:48],
            "mention": np.nonzero(~is_fact)[0][:48]}
    return w, m, x, rs, rows


def _batch(setup, kind, dropped):
    _, m, x, rs, rows = setup
    r = torch.as_tensor(rows[kind])
    drop = torch.full((len(r),), bool(dropped)) & (rs[r] >= 0)
    return x[r], rs[r], drop


def _grads(model, xb, rsb, drop, alpha):
    model.zero_grad(set_to_none=True)
    loss, _ = v41_loss(model, xb, rsb, drop, alpha)
    loss.backward()
    P, M = param_groups(model)
    gn = lambda p: 0.0 if p.grad is None else float(p.grad.norm())
    return {n: gn(p) for n, p in P.items()}, {n: gn(p) for n, p in M.items()}


def test_parameter_partition_is_exhaustive_and_disjoint(setup):
    _, m, *_ = setup
    P, M = param_groups(m)
    names = [n for n, _ in m.named_parameters()]
    assert set(P) | set(M) == set(names) and not set(P) & set(M) and len(P) + len(M) == len(names)
    assert set(M) == {"mem_keys", "mem_values", "q_ln.weight", "q_ln.bias", "W_q.weight", "W_o.weight", "log_scale"}
    for must in ("tok.weight", "pos.weight", "ln_f.weight", "ln_f.bias", "unembed.weight"):
        assert must in P
    for b in range(4):                                                      # all four blocks incl. their LayerNorms
        assert {f"blocks.{b}.{k}" for k in ("ln1.weight", "ln2.weight", "qkv.weight", "proj.weight", "fc1.weight",
                                            "fc2.weight")} <= set(P)
    assert all(a in MEMORY_PARAMS for a in ADDRESSING_PARAMS)
    bad = copy.deepcopy(m)
    bad.extra = nn.Parameter(torch.zeros(3))                                # an unclassified parameter must be rejected
    with pytest.raises(ValueError):
        param_groups(bad)


def test_v41_loss_value_equals_v4_loss(setup):
    """Only the gradient routes change: the loss value is the v4 loss."""
    _, m, x, rs, _ = setup
    b = torch.arange(0, 200)
    drop = (rs[b] >= 0) & (torch.rand(200, generator=torch.Generator().manual_seed(1)) < 0.4)
    l41, _ = v41_loss(m, x[b], rs[b], drop, 1.0)
    inp, tgt = x[b][:, :-1], x[b][:, 1:]
    lg, mem = m(inp, row_slot=rs[b].clamp_min(0), row_drop=drop, return_mem=True)
    l4 = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=W.PAD)
    rr, r_ = mem["rows"], rs[b]
    rt = torch.where((r_[rr] >= 0) & ~drop[rr], r_[rr], torch.full_like(r_[rr], m.null_index))
    l4 = l4 + torch.nn.functional.cross_entropy(mem["logits"], rt)
    assert abs(float(l41) - float(l4)) < 1e-5


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_memory_present_covered_batch_gives_zero_gradient_to_every_P_tensor(setup, alpha):
    _, m, *_ = setup
    model = copy.deepcopy(m)
    gP, gM = _grads(model, *_batch(setup, "covered", dropped=False), alpha)
    assert max(gP.values()) == 0.0, {k: v for k, v in gP.items() if v}
    for n in ("mem_values", "W_o.weight", "mem_keys", "W_q.weight", "log_scale"):
        assert gM[n] > 0, n                                                 # the memory route does learn
    slot_rows = _batch(setup, "covered", dropped=False)[1]
    g = model.mem_values.grad
    assert (g[slot_rows].norm(dim=1) > 0).all()                             # each presented fact's own value row


def test_route_dropped_covered_batch_trains_P_and_gives_no_answer_gradient_to_memory(setup):
    _, m, *_ = setup
    model = copy.deepcopy(m)
    gP, gM = _grads(model, *_batch(setup, "covered", dropped=True), alpha=0.0)
    assert min(gP.values()) > 0, {k: v for k, v in gP.items() if not v}    # every P tensor receives answer/LM gradient
    assert max(gM.values()) == 0.0, gM                                       # no answer gradient to any M tensor
    gP, gM = _grads(model, *_batch(setup, "covered", dropped=True), alpha=1.0)
    assert gM["mem_values"] == 0.0 and gM["W_o.weight"] == 0.0              # NULL supervision: addressing only
    assert all(gM[n] > 0 for n in ("mem_keys", "W_q.weight", "log_scale"))
    xb, rsb, drop = _batch(setup, "covered", dropped=True)
    model.zero_grad(set_to_none=True)
    v41_loss(model, xb[:1], rsb[:1], drop[:1], 1.0)[0].backward()          # a single dropped row: its OWN key gets 0
    assert float(model.mem_keys.grad[rsb[0]].abs().max()) == 0.0
    assert float(model.mem_keys.grad[model.null_index].abs().max()) > 0      # ... and NULL is pulled towards the query


def test_parametric_only_and_mention_batches_train_P_normally(setup):
    _, m, *_ = setup
    model = copy.deepcopy(m)
    gP, gM = _grads(model, *_batch(setup, "param_only", dropped=False), alpha=0.0)
    assert min(gP.values()) > 0 and max(gM.values()) == 0.0
    xb, rsb, drop = _batch(setup, "mention", dropped=False)
    model.zero_grad(set_to_none=True)
    v41_loss(model, xb, rsb, drop, 1.0)[0].backward()
    g41 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    ref = copy.deepcopy(m)
    ref.zero_grad(set_to_none=True)
    lm_loss(ref, xb).backward()                                              # ordinary LM gradient
    gref = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
    assert set(gref) <= set(g41)
    for n, g in gref.items():
        assert torch.allclose(g41[n], g, atol=1e-6, rtol=1e-4), n
    P, _ = param_groups(model)
    assert set(gref) <= set(P)                                               # mentions never reach the memory


def _one_step(setup, cfg, kind, dropped):
    _, m, *_ = setup
    model = copy.deepcopy(m)
    oc = cfg["store"]["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    xb, rsb, drop = _batch(setup, kind, dropped)
    train_step(model, opt, xb, rsb, drop, 1.0, iso=True)
    after = dict(model.named_parameters())
    return model, before, after, rsb


def test_optimizer_step_on_memory_present_batch_leaves_every_P_tensor_unchanged(setup, tiny_cfg):
    model, before, after, rsb = _one_step(setup, tiny_cfg, "covered", dropped=False)
    P, M = param_groups(model)
    for n in P:
        assert torch.equal(before[n], after[n]), n                          # bit-identical
    assert not torch.equal(before["mem_values"][rsb], after["mem_values"][rsb])
    assert not torch.equal(before["W_o.weight"], after["W_o.weight"])


def test_optimizer_step_on_route_dropped_batch_changes_P_not_values_or_readout(setup, tiny_cfg):
    model, before, after, _ = _one_step(setup, tiny_cfg, "covered", dropped=True)
    P, _ = param_groups(model)
    for n in P:
        assert not torch.equal(before[n], after[n]), n
    assert torch.equal(before["mem_values"], after["mem_values"])          # bit-identical slot contents and readout
    assert torch.equal(before["W_o.weight"], after["W_o.weight"])


def test_audit_detects_the_v4_shared_gradient_leak(setup):
    """The invariant is not vacuous: under v4 (shared) training, memory-present covered rows DO reach P."""
    _, m, *_ = setup
    xb, rsb, drop = _batch(setup, "covered", dropped=False)
    assert isolation_audit(m, xb, rsb, drop, 1.0) == {"G1_to_P": 0.0, "G23_to_values_Wo": 0.0}
    model = copy.deepcopy(m)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    model.zero_grad(set_to_none=True)
    train_step(model, opt, xb, rsb, drop, 1.0, iso=False)
    P, _ = param_groups(model)
    assert max(float(p.grad.abs().max()) for p in P.values() if p.grad is not None) > 0


def test_training_loop_counts_and_runtime_audit(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 12)
    mc = dict(tiny_cfg["memory"], gradient_isolation=True)
    for p_rd in (0.0, 1.0, 0.3):
        st = {}
        train_dual_store(w, tiny_cfg["store"], mc, seed=5, epochs=2, p_rd=p_rd, stats=st)
        assert st["gradient_isolation"] and (st["presentations"] == 2).all()   # each covered fact once per epoch
        if p_rd == 0.0:
            assert (st["route_drop_count"] == 0).all()
        elif p_rd == 1.0:
            assert (st["route_drop_count"] == 2).all()
        else:
            assert 0.15 < st["route_drop_count"].mean() / 2 < 0.45


def test_dc2_decision_logic():
    """DC-2 is a one-sided randomised dose test: passes for a count that tracks backup, fails for none or reversed."""
    from s1.v4qc import dc2_check
    rng = np.random.default_rng(0)
    cnt = rng.binomial(60, 0.2, 1000)
    margin = 0.3 * (cnt - cnt.mean()) + rng.normal(0, 1, 1000)
    assert dc2_check(cnt, margin)[2]
    assert not dc2_check(cnt, -margin)[2]
    assert not dc2_check(cnt, rng.normal(0, 1, 1000))[2]
