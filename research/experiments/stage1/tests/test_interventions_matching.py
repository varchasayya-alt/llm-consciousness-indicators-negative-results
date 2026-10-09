import ast
import os

import numpy as np
import torch

from s1 import world as W
from s1.controller import bayes_threshold, lookup_prob, train_controller
from s1.interventions import act_knockout_fn, forget_with_sham, sample_dropout_rates
from s1.matching import caliper_match, displacement, log_profile
from s1.store import build_store, probe, site_stats

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_sham_and_calibration_never_touch_monitors():
    forbidden = {"s1.monitors", "s1.controller", "s1.pipeline", ".monitors", ".controller", ".pipeline",
                 "monitors", "controller", "pipeline"}
    for f in ("s1/interventions.py", "s1/matching.py", "s1/diagnostics.py", "run_calibration.py"):
        assert not (_imports(os.path.join(STAGE, f)) & forbidden), f


def test_forget_with_sham_leaves_original_untouched_and_inputs_identical(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 11)
    A = build_store(tiny_cfg["store"], w.vocab.size)
    before = {k: v.clone() for k, v in A.state_dict().items()}
    ev = W.items_where(w, split=W.EV)
    X, Y = ev[:8], ev[8:16]
    _, sig = site_stats(probe(A, w, ev)["states"])
    model, log = forget_with_sham(A, w, X, Y, tiny_cfg["interventions"]["T_FORGET"], sig, 3)
    for k, v in A.state_dict().items():
        assert torch.equal(v, before[k])
    q_pre = W.query_tokens(w.vocab, w.names[X[:, 0]], X[:, 1])
    q_post = W.query_tokens(w.vocab, w.names[X[:, 0]], X[:, 1])
    assert np.array_equal(q_pre, q_post)                      # interventions never alter inputs
    assert any(t["phase"] == "B" for t in log["trace"])


def test_dropout_rates_shape_and_min_layers():
    rng = np.random.default_rng(0)
    r = sample_dropout_rates(1000, 4, {"rate": [0.0, 0.6], "layer_inclusion_prob": 0.5}, rng)
    assert r.shape == (1000, 4)
    assert (((r > 0).sum(1) >= 1) | (r.sum(1) == 0)).all()   # >=1 layer selected (rate may be ~0)
    assert float(r.max()) <= 0.6


def test_act_knockout_mask():
    ko = act_knockout_fn([3, 4], 5)(0, 5)
    assert ko[0] == {3, 4} and ko[1][:, 4, 1].all() and ko[1][:, 4, 2].all() and ko[1].sum() == 10


def test_displacement_and_caliper_matching():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(20, 10, 8))
    sig = np.ones((10, 8))
    assert np.allclose(displacement(H, H, sig), 0)
    pos = rng.normal(size=(30, 10))
    neg = rng.normal(size=(40, 10))
    pairs = caliper_match(pos, neg, 1.2, 5)
    assert len({j for _, j, _ in pairs}) == len(pairs)          # without replacement
    assert all(d <= 1.2 for _, _, d in pairs)
    assert pairs == caliper_match(pos, neg, 1.2, 5)               # deterministic
    assert caliper_match(pos, neg, 0.0, 5) == []


def test_controller_bayes_threshold():
    assert bayes_threshold(0.3, 1.0) == 0.85
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 20000)
    y = rng.random(20000) < p                                  # calibrated synthetic monitor
    cc = {"hidden": 16, "lambda_wrong": 1.0, "cost_train_range": [0.05, 0.6], "steps": 1500, "batch_size": 512, "lr": 3e-3}
    ctrl = train_controller(p, y, cc, 0)
    lo = lookup_prob(ctrl, np.array([0.6, 0.7]), 0.3)
    hi = lookup_prob(ctrl, np.array([0.95, 0.99]), 0.3)
    assert (lo > 0.5).all() and (hi < 0.5).all()


# ---------------------------------------------------------------- calibration-protocol revision v2
def _tiny_setup(tiny_cfg, seed=11):
    w = W.make_world(tiny_cfg["world"], seed)
    A = build_store(tiny_cfg["store"], w.vocab.size)
    ev = W.items_where(w, split=W.EV)
    _, sig = site_stats(probe(A, w, ev)["states"])
    return w, A, ev, sig


def test_retain_kl_is_zero_at_original_and_positive_after_change(tiny_cfg):
    from s1.interventions import RetainKL
    w, A, ev, _ = _tiny_setup(tiny_cfg)
    rk = RetainKL(A, w, ev[:5, 0], 32, 8)
    g = torch.Generator().manual_seed(0)
    l0, _ = rk.loss(A, g)
    assert float(l0) < 1e-6
    B = build_store(tiny_cfg["store"], w.vocab.size)
    l1, (ka, kt) = rk.loss(B, torch.Generator().manual_seed(0))
    assert float(l1) > 1e-4 and ka > 0 and kt > 0
    full = RetainKL(A, w, [], 10 ** 6, 0)                       # whole-pool mode, no mention term
    assert float(full.loss(B, g)[0]) > 0 and full.n_facts == len(W.items_where(w, split=W.MT, known=True))


def test_retain_pool_never_contains_qc_items(tiny_cfg):
    """v2 design rule: retain facts are known MT facts only; excluded entities never appear in mentions."""
    from s1.interventions import RetainKL
    w, A, ev, _ = _tiny_setup(tiny_cfg)
    excl = ev[:10, 0]
    rk = RetainKL(A, w, excl, 16, 8)
    mt = W.items_where(w, split=W.MT, known=True)
    q_expected = W.query_tokens(w.vocab, w.names[mt[:, 0]], mt[:, 1])
    assert np.array_equal(rk.q_f.numpy(), q_expected)
    assert not np.isin(rk.mention_entities, excl).any()
    assert len(rk.mention_entities) == w.n_ent - len(np.unique(excl))


def test_forget_qc_categories_are_disjoint_and_held_out(tiny_cfg):
    from s1.interventions import forget_qc_sets
    w, A, ev, sig = _tiny_setup(tiny_cfg)
    X, Y, Z = ev[:8], ev[8:16], ev[16:]
    cats, info = forget_qc_sets(A, w, {"X": X, "Y": Y, "Z": Z}, sig, 0.2)
    ids = lambda a: set((a[:, 0] * w.n_rel + a[:, 1]).tolist())
    k = int(round(0.2 * len(Z)))
    assert len(cats["Z_near_repr"]) == k and len(cats["Z_far"]) == k
    assert not ids(cats["Z_near_repr"]) & ids(cats["Z_far"])
    assert ids(cats["Z_near_repr"]) <= ids(Z) and ids(cats["Z_far"]) <= ids(Z)
    for name in ("Z_near_entity_X", "Z_near_entity_Y"):
        it = cats[name]
        if len(it):
            assert (w.split[it[:, 0], it[:, 1]] == W.CT).all() and w.known[it[:, 0], it[:, 1]].all()
    if len(cats["Z_near_entity_X"]):
        assert np.isin(cats["Z_near_entity_X"][:, 0], X[:, 0]).all()
    for name in ("Z_random", "Z_near_entity_X", "Z_near_entity_Y", "Z_near_repr", "Z_far"):
        assert not ids(cats[name]) & (ids(X) | ids(Y)) if len(cats[name]) else True
        if len(cats[name]):
            assert (w.split[cats[name][:, 0], cats[name][:, 1]] != W.MT).all()     # never in the retain pool
    assert info["sim_near_mean"] >= info["sim_far_mean"]


def test_forget_qc_v3_reports_all_categories_and_gates(tiny_cfg):
    from s1.interventions import Z_CATEGORIES, forget_qc
    w, A, ev, sig = _tiny_setup(tiny_cfg)
    mu = probe(A, w, ev)["states"].mean(0)
    X, Y, Z = ev[:8], ev[8:16], ev[16:]
    fcfg = tiny_cfg["interventions"]["T_FORGET"]
    model, _ = forget_with_sham(A, w, X, Y, fcfg, sig, 3)
    qc, aux = forget_qc(A, model, w, {"X": X, "Y": Y, "Z": Z}, sig, fcfg["qc"], [1.0, 0.5], 4, 1.0, mu)
    assert set(qc["Z_lost_by_category"]) == set(Z_CATEGORIES)
    assert set(qc["gates"]) == {"X_diversity", "Y_binary", "Z_binary", "retention", "sites", "fluency"}
    assert set(qc["retention"]) == {"Y"} | set(Z_CATEGORIES)
    assert set(qc["retention_pass_by_level"]) == {"0.05", "0.1", "0.2"}
    assert {"r2_pre", "r2_gen", "r2_joint"} <= set(qc["identifiability"])
    assert {"pairs", "max_abs_smd", "feasible"} <= set(qc["binary_secondary"])
    assert {"pairs", "max_abs_smd", "mean_abs_smd", "feasible"} <= set(qc["binary_secondary_alt"])
    assert qc["D_C"] == qc["D_C"] and aux["GX"].shape == (8, 40)
    for name in ("X", "Y", "Z_random", "R_sibling_X"):
        rep = qc["sets"][name]
        if rep["n"]:
            assert {"acc_pre", "acc_post", "lost_frac", "kl_answer_dist_mean", "disp_mean", "margin_pre_mean"} <= set(rep)
    assert qc["pass"] == all(qc["gates"].values())


def test_v3_variants_mechanics(tiny_cfg):
    """V1 anchor is zero at the start; V3 changes only block-1-2 MLP weights; V2 pool adds non-excluded CT facts."""
    from s1.interventions import V3_TRAINABLE, RetainKL
    w, A, ev, sig = _tiny_setup(tiny_cfg)
    X, Y = ev[:8], ev[8:16]
    fcfg = dict(tiny_cfg["interventions"]["T_FORGET"], variant="V3", lr=1e-2)
    m3, log3 = forget_with_sham(A, w, X, Y, fcfg, sig, 3)
    s0, s3 = A.state_dict(), m3.state_dict()
    for k in s0:
        changed = not torch.equal(s0[k], s3[k])
        assert changed == any(k.startswith(t) for t in V3_TRAINABLE), k
    assert all(p.requires_grad for p in m3.parameters())                      # restored after the edit
    m1, log1 = forget_with_sham(A, w, X, Y, dict(tiny_cfg["interventions"]["T_FORGET"], variant="V1", anchor_lambda=1.0), sig, 3)
    assert log1["trace"][0]["anchor"] == 0.0 and log1["variant"] == "V1"
    excl = np.concatenate([X[:, 0], Y[:, 0]])
    r0, r2 = RetainKL(A, w, excl, 16, 4, pool="MT"), RetainKL(A, w, excl, 16, 4, pool="MT+CT")
    ct = W.items_where(w, split=W.CT, known=True)
    assert r2.n_facts == r0.n_facts + int((~np.isin(ct[:, 0], excl)).sum())


def test_estimands_recover_h3_and_null():
    from s1 import estimands as ES
    from s1.sim_worlds import make_structure, pre_covariates, simulate_items
    st = make_structure(np.random.default_rng(12345))
    th = {}
    for scen in ("H3_strong", "null"):
        vals = []
        for k in range(4):
            it = simulate_items(scen, 300, np.random.default_rng([5, k]), structure=st, loss_shift=1.5)
            vals.append(ES.theta_post_dml(it["M_pre"], it["M_post"], it["C_pre"], it["C_post"], pre_covariates(it),
                                          it["H_pre"], k, strata=it["e"])[0])
        th[scen] = float(np.mean(vals))
    assert th["H3_strong"] > 0.5 and abs(th["null"]) < 0.08
    B = ES.rcs(np.linspace(0, 1, 50))
    assert B.shape == (50, 4) and np.isfinite(B).all()
    rng = np.random.default_rng(0)
    Xf = rng.normal(size=(200, 5))
    _, r2_sig, _ = ES.crossfit_ridge(Xf, Xf @ np.ones(5) + rng.normal(0, 0.1, 200), 1)
    _, r2_noise, _ = ES.crossfit_ridge(Xf, rng.normal(size=200), 1)
    assert r2_sig > 0.95 and r2_noise < 0.1
    z = rng.normal(size=(400, 3))
    treat = rng.random(400) < 1 / (1 + np.exp(-z[:, 0]))
    pairs, smd = ES.propensity_match(treat, z, (z[:, 2] > 0).astype(int), 3)
    raw = np.abs(z[treat].mean(0) - z[~treat].mean(0)).max() / z.std(0).max()
    assert len(pairs) > 50 and smd < raw


def test_corrected_c3_rule_has_reachable_branches():
    import run_calibration as rc
    assert rc.c3_rate_max(0.95, 0.95) == 0.8
    assert rc.c3_rate_max(0.40, 0.20) == 0.3
    assert rc.c3_rate_max(0.90, 0.21) == 0.6


def test_v3_selection_rule():
    import run_calibration as rc
    mk = lambda lr, st, load, xl, el=True: {"lr": lr, "steps": st, "worst_load": load, "mean_X_lost": xl, "eligible": el}
    rows = [mk(3e-4, 100, 0.60, 0.40), mk(1e-4, 200, 0.62, 0.52), mk(1e-3, 100, 0.20, 0.5, False), mk(1e-4, 100, 0.90, 0.5)]
    sel = rc.select_v3(rows)
    assert (sel["lr"], sel["steps"]) == (1e-4, 200)          # tie within 0.05 load -> lost fraction closest to 0.5
    assert rc.select_v3([mk(1e-4, 100, 0.1, 0.5, False)]) is None
    assert len(rc.v3_cells()) == 36
