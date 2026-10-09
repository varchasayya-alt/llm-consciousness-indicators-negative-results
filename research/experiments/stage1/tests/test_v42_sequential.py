"""v4.2 sequential development (D58; PI 2026-10-03, item 13): structural tests required before any v4.2 run.

Stage A: omitted covered facts give no answer-learning update; included ones train P normally; dose counts are
         correct and nested across p_rd; the memory group does not change.
Stage B: P bit-identical; optimizer holds M only (no optimizer state for P); no gradient reaches P (so retrieval /
         query learning cannot alter backbone states); NULL value exactly zero and injection cap enforced;
         memory-present answer loss trains M on a toy batch.
Cross-stage: Stage-A P hash unchanged by Stage B; all-slots-masked store == Stage-A network bit-for-bit; Stage-B
         memory improves an intentionally weak toy fact while P is unchanged.
"""
import copy

import numpy as np
import pytest
import torch

from s1 import world as W
from s1.memstore import (all_masked, param_groups, set_injection_cap, slot_table, stageA_epoch_rows, stageA_inclusion,
                         stageB_step, tensor_hash, train_stage_A, train_stage_B)
from s1.store import answer_stats


@pytest.fixture(scope="module")
def v42(tiny_cfg):
    c = copy.deepcopy(tiny_cfg)
    c["memory"] = dict(c["memory"], null_value="fixed_zero", injection_cap_kappa=10.0, development="sequential")
    return c


@pytest.fixture(scope="module")
def world(v42):
    return W.make_world(v42["world"], 21)


@pytest.fixture(scope="module")
def stageA(v42, world):
    return train_stage_A(world, v42["store"], v42["memory"], seed=3, epochs=4, p_rd=0.5, dose_seed=4)


def _rows(world):
    seqs, _, idx = W.training_sequences(world)
    slot, _ = slot_table(world)
    return seqs, np.where(idx >= 0, slot.reshape(-1)[np.maximum(idx, 0)], -1)


# ---------------------------------------------------------------- Stage A
def test_stageA_epoch_composition_omits_covered_facts_by_dose(world):
    seqs, rs = _rows(world)
    U = stageA_inclusion(world, dose_seed=4, epochs=6)
    for ep in range(6):
        r0, inc0 = stageA_epoch_rows(rs, U, ep, 0.0)
        r1, inc1 = stageA_epoch_rows(rs, U, ep, 1.0)
        assert len(inc0) == 0 and (rs[r0] < 0).all()                         # p=0: no covered fact sequence at all
        assert set(rs[inc1]) == set(rs[rs >= 0])                             # p=1: every covered fact
        r3, inc3 = stageA_epoch_rows(rs, U, ep, 0.3)
        r5, inc5 = stageA_epoch_rows(rs, U, ep, 0.5)
        assert set(inc3) <= set(inc5)                                        # common random numbers: nested
        assert set(np.nonzero(rs < 0)[0]) <= set(r3)                         # parametric-only facts + mentions always
        omitted = set(np.nonzero(rs >= 0)[0]) - set(inc3)
        assert not (omitted & set(r3))                                       # omitted covered rows are absent


def test_stageA_counts_match_dose(v42, world):
    for p, lo, hi in ((0.0, 0, 0), (1.0, 3, 3)):
        _, st = train_stage_A(world, v42["store"], v42["memory"], seed=3, epochs=3, p_rd=p, dose_seed=4)
        assert st["presentations"].min() == lo and st["presentations"].max() == hi
    U = stageA_inclusion(world, dose_seed=4, epochs=40)
    assert abs((U < 0.2).sum(1).mean() - 8.0) < 0.6                          # E x p_rd on average


def test_stageA_omitted_covered_answers_cannot_influence_P(v42, world):
    """p_rd = 0: P is bit-identical whatever the covered facts' answers are (they never reach P)."""
    w2 = copy.deepcopy(world)
    cov = world.mem_covered & world.known
    w2.answers = np.where(cov, (world.answers + 7) % world.vocab.n_val, world.answers)
    ma, _ = train_stage_A(world, v42["store"], v42["memory"], seed=3, epochs=2, p_rd=0.0, dose_seed=4)
    mb, _ = train_stage_A(w2, v42["store"], v42["memory"], seed=3, epochs=2, p_rd=0.0, dose_seed=4)
    Pa, _ = param_groups(ma)
    Pb, _ = param_groups(mb)
    assert tensor_hash(Pa) == tensor_hash(Pb)
    mc, _ = train_stage_A(w2, v42["store"], v42["memory"], seed=3, epochs=2, p_rd=1.0, dose_seed=4)
    md, _ = train_stage_A(world, v42["store"], v42["memory"], seed=3, epochs=2, p_rd=1.0, dose_seed=4)
    assert tensor_hash(param_groups(mc)[0]) != tensor_hash(param_groups(md)[0])   # included facts do train P


def test_stageA_included_covered_facts_train_P_with_ordinary_LM_gradient(v42, world):
    from s1.store import lm_loss
    torch.manual_seed(0)
    from s1.memstore import build_dual_store
    m = build_dual_store(v42["store"], v42["memory"], world)
    seqs, rs = _rows(world)
    b = torch.as_tensor(seqs[np.nonzero(rs >= 0)[0][:32]])
    logits = m(b[:, :-1], memory_off=True)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1), ignore_index=W.PAD)
    loss.backward()
    P, M = param_groups(m)
    assert all(p.grad is not None and float(p.grad.norm()) > 0 for p in P.values())
    assert all(p.grad is None for p in M.values())                           # memory never on the Stage-A path
    with torch.no_grad():                                                     # memory_off == the plain transformer
        h = m.tok(b[:, :-1]) + m.pos(torch.arange(b.shape[1] - 1))
        for blk in m.blocks:
            h = blk(h)
        assert torch.equal(m.unembed(m.ln_f(h)), m(b[:, :-1], memory_off=True))
    del lm_loss


def test_stageA_does_not_change_memory_group(stageA, v42, world):
    m, st = stageA
    torch.manual_seed(0)
    from s1.config import set_all_seeds
    from s1.memstore import build_dual_store
    set_all_seeds(3)
    m0 = build_dual_store(v42["store"], v42["memory"], world)
    assert tensor_hash(param_groups(m)[1]) == tensor_hash(param_groups(m0)[1]) == st["M_hash"]
    assert tensor_hash(param_groups(m)[0]) != tensor_hash(param_groups(m0)[0])


# ---------------------------------------------------------------- Stage B
def test_stageB_freezes_P_bit_for_bit_and_changes_M(stageA, v42, world):
    mA, stA = stageA
    mB, stB = train_stage_B(mA, world, v42["store"], v42["memory"], seed=5, epochs=2)
    PA, MA = param_groups(mA)
    PB, MB = param_groups(mB)
    assert tensor_hash(PB) == tensor_hash(PA) == stA["P_hash"] == stB["P_hash"]
    for n in PA:
        assert torch.equal(PA[n], PB[n]), n
    for n in ("mem_values", "W_o.weight", "W_q.weight", "log_scale"):
        assert not torch.equal(MA[n], MB[n]), n


def test_stageB_optimizer_and_gradients_touch_M_only(stageA, v42, world):
    mA, _ = stageA
    m = copy.deepcopy(mA)
    P, M = param_groups(m)
    for p in P.values():
        p.requires_grad_(False)
    set_injection_cap(m, world, 10.0)
    opt = torch.optim.AdamW(list(M.values()), lr=1e-3, weight_decay=0.0)
    kn = W.items_where(world, known=True)
    slot, _ = slot_table(world)
    it = kn[world.mem_covered[kn[:, 0], kn[:, 1]]][:16]
    seqs = torch.as_tensor(W.fact_seqs(world.vocab, world.names[it[:, 0]], it[:, 1], world.answers[it[:, 0], it[:, 1]]))
    rs = torch.as_tensor(slot[it[:, 0], it[:, 1]])
    kind = torch.tensor([0] * 8 + [1] * 8)
    _, leak = stageB_step(m, opt, seqs, rs, kind, 1.0)
    assert leak == 0.0 and all(p.grad is None for p in P.values())          # no gradient reaches P (or its inputs)
    state_ids = {id(p) for p in opt.state}
    assert state_ids <= {id(p) for p in M.values()} and not (state_ids & {id(p) for p in P.values()})


def test_null_value_is_exactly_zero_and_cap_is_enforced(stageA, v42, world):
    mA, _ = stageA
    mB, _ = train_stage_B(mA, world, v42["store"], v42["memory"], seed=5, epochs=2)
    assert float((mB.mem_values * mB.value_mask[:, None])[-1].abs().max()) == 0.0
    kn = W.items_where(world, known=True)
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[kn[:64, 0]], kn[:64, 1]))
    with torch.no_grad():
        _, mem = all_masked(mB)(toks, return_mem=True)
        assert float(mem["u"].abs().max()) == 0.0                           # NULL: no explicit memory contribution
        _, mem = mB(toks, return_mem=True)
        assert float(mem["u"].norm(dim=-1).max()) <= float(mB.inj_cap) * (1 + 1e-5)
    m2 = copy.deepcopy(mB)
    with torch.no_grad():
        m2.W_o.weight.mul_(1e4)                                              # an exploding readout stays capped
        _, mem = m2(toks, return_mem=True)
        assert float(mem["u"].norm(dim=-1).max()) <= float(m2.inj_cap) * (1 + 1e-5)


def test_memory_present_answer_loss_trains_M_on_toy_batch(stageA, v42, world):
    mA, _ = stageA
    m = copy.deepcopy(mA)
    P, M = param_groups(m)
    for p in P.values():
        p.requires_grad_(False)
    set_injection_cap(m, world, 10.0)
    opt = torch.optim.AdamW(list(M.values()), lr=5e-3, weight_decay=0.0)
    kn = W.items_where(world, known=True)
    slot, _ = slot_table(world)
    it = kn[world.mem_covered[kn[:, 0], kn[:, 1]]][:24]
    seqs = torch.as_tensor(W.fact_seqs(world.vocab, world.names[it[:, 0]], it[:, 1], world.answers[it[:, 0], it[:, 1]]))
    rs = torch.as_tensor(slot[it[:, 0], it[:, 1]])
    kind = torch.zeros(len(it), dtype=torch.long)
    losses = [stageB_step(m, opt, seqs, rs, kind, 1.0)[0] for _ in range(60)]
    assert losses[-1] < 0.5 * losses[0]


# ---------------------------------------------------------------- cross-stage
def test_routeA_of_final_store_equals_stageA_network_bit_for_bit(stageA, v42, world):
    mA, _ = stageA
    mB, _ = train_stage_B(mA, world, v42["store"], v42["memory"], seed=5, epochs=2)
    kn = W.items_where(world, known=True)
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[kn[:200, 0]], kn[:200, 1]))
    with torch.no_grad():
        assert torch.equal(all_masked(mB)(toks), mA(toks, memory_off=True))


def test_stageB_rescues_weak_facts_while_P_unchanged(v42, world):
    """p_rd = 0: covered facts were never seen by P (parametric-weak/wrong); Stage B makes them answerable.
    (Tiny world: batch 64 so that Stage B gets ~1,000 optimizer steps; the full configuration has ~4,400.)"""
    scfg = dict(v42["store"], optimizer=dict(v42["store"]["optimizer"], batch_size=64))
    mA, stA = train_stage_A(world, v42["store"], v42["memory"], seed=3, epochs=30, p_rd=0.0, dose_seed=4)
    kn = W.items_where(world, known=True)
    cov = kn[world.mem_covered[kn[:, 0], kn[:, 1]]]
    before = answer_stats(all_masked(mA), world, cov)
    mB, stB = train_stage_B(mA, world, scfg, v42["memory"], seed=5, epochs=100)
    after = answer_stats(mB, world, cov)
    assert after["margin"].mean() > before["margin"].mean() + 1.0
    assert after["correct"].mean() > before["correct"].mean() + 0.3
    assert tensor_hash(param_groups(mB)[0]) == stA["P_hash"] == stB["P_hash"]
