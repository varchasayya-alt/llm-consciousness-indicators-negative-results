"""v4 (Option B1) dual-route store: structural unit tests required before any v4 run (PI, 2026-10-03)."""
import ast
import copy
import os

import numpy as np
import pytest
import torch

from s1 import world as W
from s1.memstore import (DualStore, all_masked, build_dual_store, choose_donors, delete_slots, init_keys_from_queries,
                         m1_stats, m2_probe, memory_probe, slot_table, transplant)
from s1.store import probe

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 11)
    torch.manual_seed(0)
    m = build_dual_store(tiny_cfg["store"], tiny_cfg["memory"], w)
    init_keys_from_queries(m, w)
    m.eval()
    return w, m


def _covered(w):
    kn = W.items_where(w, known=True)
    return kn[w.mem_covered[kn[:, 0], kn[:, 1]]], kn[~w.mem_covered[kn[:, 0], kn[:, 1]]]


def test_parametric_only_facts_have_no_slot_and_covered_facts_unique_slots(tiny):
    w, m = tiny
    slot, S = slot_table(w)
    cov, par = _covered(w)
    assert S == m.n_slots == len(cov)
    assert (slot[par[:, 0], par[:, 1]] == -1).all()
    unk = W.items_where(w, known=False)
    assert (slot[unk[:, 0], unk[:, 1]] == -1).all()
    s = slot[cov[:, 0], cov[:, 1]]
    assert len(np.unique(s)) == len(s) and s.min() == 0 and s.max() == S - 1
    assert 0.5 < len(cov) / len(W.items_where(w, known=True)) < 0.9       # ~70% coverage, random


def test_deletion_changes_only_intended_availability_and_no_weights(tiny):
    w, m = tiny
    cov, _ = _covered(w)
    X = cov[:10]
    d = delete_slots(m, w, X)
    slot, _ = slot_table(w)
    sx = slot[X[:, 0], X[:, 1]]
    assert not d.present[torch.as_tensor(sx)].any()
    keep = np.setdiff1d(np.arange(m.n_slots), sx)
    assert d.present[torch.as_tensor(keep)].all() and m.present.all()      # original untouched
    for (n0, p0), (n1, p1) in zip(m.named_parameters(), d.named_parameters()):
        assert n0 == n1 and torch.equal(p0, p1), n0                         # no parameter (incl. Y/Z slot contents) changes
    with pytest.raises(AssertionError):
        _, par = _covered(w)
        delete_slots(m, w, par[:2])                                          # only memory-covered facts can be deleted


def test_null_slot_always_available_and_used_when_memory_unavailable(tiny):
    w, m = tiny
    cov, par = _covered(w)
    A = all_masked(m)
    mp = memory_probe(A, w, cov[:20])
    assert np.allclose(mp["a_null"], 1.0)                                    # NULL is the only available slot
    assert m.null_index == m.n_slots and m.mem_keys.shape[0] == m.n_slots + 1


def test_retrieval_targets_and_route_dropout_mask_own_slot(tiny):
    w, m = tiny
    cov, _ = _covered(w)
    slot, _ = slot_table(w)
    toks = torch.as_tensor(W.query_tokens(w.vocab, w.names[cov[:8, 0]], cov[:8, 1]))
    rs = torch.as_tensor(slot[cov[:8, 0], cov[:8, 1]])
    drop = torch.tensor([True, False] * 4)
    (_, _), mem = m(toks, collect=True, return_mem=True, row_slot=rs, row_drop=drop)
    own = mem["a"][torch.arange(8), rs]
    assert (own[drop] == 0).all() and (own[~drop] > 0).all()               # dropped rows cannot see their own slot


def test_covered_facts_route_to_own_slot_when_available(tiny):
    w, m = tiny
    cov, _ = _covered(w)
    mp = memory_probe(m, w, cov)
    assert np.mean(mp["a"][:, :-1].argmax(1) == mp["own_slot"]) > 0.95      # self-key init: own slot ranks first


def test_y_transplant_donor_unchanged_recipient_gets_only_donor_value(tiny):
    w, m = tiny
    cov, _ = _covered(w)
    ev = cov[w.split[cov[:, 0], cov[:, 1]] == W.EV]
    Y = ev[:6]
    donors = choose_donors(w, Y, ev[:6], seed=3)
    ok = donors[:, 0] >= 0
    Y, donors = Y[ok], donors[ok]
    assert len(Y) > 0
    assert (w.answers[Y[:, 0], Y[:, 1]] == w.answers[donors[:, 0], donors[:, 1]]).all() and (Y[:, 1] == donors[:, 1]).all()
    assert not np.isin(donors[:, 0], Y[:, 0]).any()
    t = transplant(m, w, Y, donors)
    slot, _ = slot_table(w)
    rs, ds = slot[Y[:, 0], Y[:, 1]], slot[donors[:, 0], donors[:, 1]]
    assert torch.equal(t.mem_values[torch.as_tensor(rs)], m.mem_values[torch.as_tensor(ds)])
    others = np.setdiff1d(np.arange(m.n_slots + 1), rs)
    assert torch.equal(t.mem_values[torch.as_tensor(others)], m.mem_values[torch.as_tensor(others)])   # incl. donors
    assert torch.equal(t.mem_keys, m.mem_keys) and t.present.all()


def test_monitor_features_cannot_access_memory_metadata(tiny):
    """probe() (the only source of monitor features) returns ordinary states and output probabilities only."""
    w, m = tiny
    cov, _ = _covered(w)
    allowed = {"correct", "margin", "logp_correct", "probs", "entropy", "pred", "states"}
    for model in (m, delete_slots(m, w, cov[:5]), all_masked(m)):
        pr = probe(model, w, cov[:12])
        assert set(pr) == allowed and pr["states"].shape[1:] == (10, m.tok.embedding_dim)
    out = m(torch.as_tensor(W.query_tokens(w.vocab, w.names[cov[:3, 0]], cov[:3, 1])))
    assert torch.is_tensor(out)                                               # default forward exposes no memory dict
    prohibited = ("memory_probe", "mem_keys", "mem_values", "present", "a_null", "retrieval_logits", "row_drop",
                  "delete_slots", "transplant", "memstore", "v4qc")
    for f in ("s1/monitors.py", "s1/pipeline.py"):
        tree = ast.parse(open(os.path.join(STAGE, f), encoding="utf-8").read())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not (names & set(prohibited)) and not any("memstore" in x or "v4qc" in x for x in mods), f


def test_m1_and_m2_behave_as_specified(tiny):
    w, m = tiny
    cov, par = _covered(w)
    it = cov[:1]
    st = probe(m, w, it)["states"]
    own_h2 = torch.as_tensor(st[0, 2 * m.inject_after + 1], dtype=torch.float32)
    base = probe(m, w, it)
    s1 = m1_stats(m, w, it, own_h2)                                           # substituting the item's own h2[A] = no-op
    assert np.allclose(s1["margin"], base["margin"], atol=1e-4)
    s2 = m1_stats(m, w, it, torch.zeros_like(own_h2))
    assert not np.allclose(s2["margin"], base["margin"])                      # M1 substitution is effective
    acc_null = m2_probe(all_masked(m), w, cov[:200], seed=1)                  # memory contribution constant (NULL)
    assert 0.0 <= acc_null <= 0.25


def test_calibration_and_v4_code_never_touch_monitors():
    forbidden = {"s1.monitors", "s1.controller", "s1.pipeline", ".monitors", ".controller", ".pipeline",
                 "monitors", "controller", "pipeline"}
    for f in ("s1/memstore.py", "s1/v4qc.py", "s1/estimands.py", "run_calibration_v4.py", "run_calibration_v42.py",
              "run_calibration.py"):
        tree = ast.parse(open(os.path.join(STAGE, f), encoding="utf-8").read())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not (mods & forbidden), f


def test_seed_namespaces_disjoint(cfg):
    import run_calibration_v42 as r42                                        # the current (v4.2, final B1) ladder
    sd = cfg["seeds"]
    dev, val = set(sd["v42_development"]), set(sd["v4_validation"])
    conf = set(range(sd["confirmatory"][0], sd["confirmatory"][1] + 1))
    retired = set(sd["retired"])
    assert dev == {9071, 9072, 9073} == set(r42.DEV_SEEDS) and val == {9051, 9052, 9053} == set(r42.VAL_SEEDS)
    assert not (dev & val) and not ((dev | val) & conf) and not ((dev | val) & retired)
    assert {9021, 9022, 9023, 9031, 9032, 9033, 9041, 9042, 9043, 9061, 9062, 9063} <= retired and not (retired & conf)
    assert r42.RES.endswith("calibration_results_v42.json") and "v42_A_9071" in r42.stage_path("A", 9071, 0.1)


def test_dose_checks_are_as_predeclared(cfg):
    """DC-1 (v4.1, D53): broadly increasing across the fixed grid (pairwise decrease <= .05) and range >= .20."""
    from s1.v4qc import dc1_check
    d = cfg["v4"]["F1"]["dose"]
    assert cfg["v4"]["F1"]["route_dropout_grid"] == [0.05, 0.10, 0.20, 0.35, 0.50]
    ok = dc1_check({0.05: 0.10, 0.10: 0.30, 0.20: 0.55, 0.35: 0.90, 0.50: 0.88}, d["dc1_pairwise_decrease_max"], d["dc1_range_min"])
    assert ok["DC1_pass"]                                                     # small ceiling inversion tolerated
    v4like = dc1_check({0.2: 0.865, 0.35: 0.803, 0.5: 0.849}, d["dc1_pairwise_decrease_max"], d["dc1_range_min"])
    assert not v4like["DC1_pass"] and not v4like["range_ok"] and not v4like["monotone_within_tol"]   # v4 F1 pattern fails
    flat = dc1_check({0.05: 0.80, 0.10: 0.82, 0.20: 0.85, 0.35: 0.88, 0.50: 0.90}, 0.05, 0.20)
    assert not flat["DC1_pass"]                                               # monotone but no meaningful dose
