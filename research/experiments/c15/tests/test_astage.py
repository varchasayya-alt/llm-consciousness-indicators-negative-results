import json
import math
import os

import numpy as np
import pytest
import torch

from c15a import config as C
from c15a import guards
from c15a.artifacts import Z0Error, check_lens_checkpoint
from c15a.gp import gp_nonneg, top_atoms
from c15a.hooks import JAblation, RandomDisplacement, add_at, forward, unit_directions
from c15a.matching import LayerMatcher
from c15a.selection import choose, layer_gate
from c15a.stats import interp_in_log_scale, iso_scale, kernel, ridge_cv_r2, ridge_cv_r2_kernel
from c15a.workspace import band_layers, build_sj, word_initial_ids


def _unit_rows(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n, d, generator=g)
    return D / D.norm(dim=1, keepdim=True)


def test_gp_recovers_planted_sparse_nonneg():
    D = _unit_rows(400, 64, 0)
    g = torch.Generator().manual_seed(1)
    idx = torch.stack([torch.randperm(400, generator=g)[:4] for _ in range(20)])
    coef = torch.rand(20, 4, generator=g) + 1.0
    H = torch.einsum("nk,nkd->nd", coef, D[idx])
    out_idx, out_coef, recon = gp_nonneg(H, D, 25)
    assert (out_coef >= 0).all()
    assert float(((H - recon) ** 2).sum() / (H ** 2).sum()) < 0.02
    rec_sets = [set(i[c > 0.2].tolist()) for i, c in zip(out_idx, out_coef)]
    hits = np.mean([len(set(t.tolist()) & s) / 4 for t, s in zip(idx, rec_sets)])
    assert hits > 0.9


def test_gp_screening_matches_exact_on_planted():
    D = _unit_rows(2000, 64, 7)
    g = torch.Generator().manual_seed(8)
    idx = torch.stack([torch.randperm(2000, generator=g)[:5] for _ in range(30)])
    H = torch.einsum("nk,nkd->nd", torch.rand(30, 5, generator=g) + 1.0, D[idx])
    _, _, r0 = gp_nonneg(H, D, 25)
    _, _, r1 = gp_nonneg(H, D, 25, screen=256)
    tot = float((H ** 2).sum())
    assert float(((H - r0) ** 2).sum()) / tot < 0.01 and float(((H - r1) ** 2).sum()) / tot < 0.01
    assert float(torch.nn.functional.cosine_similarity(r0, r1, dim=1).mean()) > 0.99


def test_top_atoms_orders_by_coefficient():
    idx = torch.tensor([[5, 7, 9, -1]])
    coef = torch.tensor([[0.1, 2.0, 0.5, 0.0]])
    assert top_atoms(idx, coef, 3).tolist() == [[7, 9, 5]]


def test_build_sj_cap_and_orthonormal():
    D = _unit_rows(300, 64, 2)
    S = torch.randn(500, 64)
    Q, info = build_sj(S, D, 64)
    assert Q.shape[1] == info["r"] <= 64 // C.SJ_CAP_DIV
    assert torch.allclose(Q.T @ Q, torch.eye(Q.shape[1]), atol=1e-4)
    assert 0 <= info["frac_state_var_in_SJ"] <= 1


def test_band_layers():
    assert band_layers(28) == [8, 10, 12, 14, 16]
    assert band_layers(24) == [7, 9, 11, 13]
    assert band_layers(36) == [11, 13, 15, 17, 19, 21]
    for L in (24, 28, 36):
        assert max(band_layers(L)) + C.ABL_HALF_WINDOW <= L - 2   # window stays inside lens source layers


def test_word_initial_ids():
    from conftest import FakeTok
    t = FakeTok(64)
    t("Hello world, a big apple 42 x")
    ids = word_initial_ids(t, len(t))
    toks = set(t.convert_ids_to_tokens(ids))
    assert "Ġworld" in toks and "Ġbig" in toks and "Ġa" not in toks and "Hello" not in toks


def _matcher(d=32, r=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    J = torch.eye(d) + 0.2 * torch.randn(d, d, generator=g)
    W = torch.randn(200, d, generator=g)
    G_W = W.T @ W
    A = torch.randn(d, d, generator=g)
    C_K = A @ A.T / d
    return LayerMatcher(J, G_W, C_K, Q, "test"), Q


def test_matcher_perp_unit_and_within_tolerance():
    m, Q = _matcher()
    g = torch.Generator().manual_seed(4)
    c = torch.randn(32, generator=g)
    uJ = Q @ (Q.T @ c)
    uJ = uJ / uJ.norm()
    u, info = m.match(c, uJ, "k")
    assert u is not None and info["feasible"]
    assert abs(float(u.norm()) - 1) < 1e-5
    assert float((Q.T @ u).norm()) < 1e-4
    for k, rt in info["ratios"].items():
        tol = math.log(C.MATCH_TOL_KL) if k == "kl" else 2 * math.log(C.MATCH_TOL_NORM)
        assert abs(math.log(rt)) <= tol + 1e-9


def test_matcher_reports_infeasible():
    m, Q = _matcher()
    c = torch.randn(32)
    uJ = Q @ (Q.T @ c)
    uJ = uJ / uJ.norm()
    m.A["gain"] = m.A["gain"].clone()
    big = 1e6 * torch.outer(uJ, uJ)          # target gain unreachable from S_J^perp
    m.A["gain"] = m.A["gain"] + big
    u, info = m.match(c, uJ, "k2")
    assert u is None and not info["feasible"]


def test_add_at_is_local_and_causal(tiny):
    lm, _ = tiny
    ids = torch.randint(1, 1000, (2, 12))
    v = torch.randn(2, lm.d_model)
    h0, r0 = forward(lm, ids, record=[2, 4])
    h1, r1 = forward(lm, ids, edits={2: add_at(5, v)}, record=[2, 4])
    assert torch.allclose(r1[2][:, 5] - r0[2][:, 5], v, atol=1e-5)
    assert torch.allclose(r1[2][:, :5], r0[2][:, :5]) and torch.allclose(r1[4][:, :5], r0[4][:, :5], atol=1e-6)
    assert not torch.allclose(r1[4][:, 6:], r0[4][:, 6:])


def test_jablation_removes_projection_and_random_matches_norm():
    D = _unit_rows(200, 32, 5)
    h = torch.randn(2, 7, 32)
    ab = JAblation(D, 25, 10)
    out = ab(h)
    assert torch.allclose(out[:, 0], h[:, 0])
    removed = (h - out)[:, 1:]
    assert torch.allclose(removed.norm(dim=-1), ab.norms[:, 1:], atol=1e-4)
    dirs = unit_directions("x", (2, 7, 32))
    rd = RandomDisplacement(ab.norms, dirs, 2.0)
    o2 = rd(h)
    assert torch.allclose((h - o2)[:, 1:].norm(dim=-1), 2.0 * ab.norms[:, 1:], atol=1e-4)
    assert torch.allclose(unit_directions("x", (2, 7, 32)), dirs)


def test_lens_identity_readout_matches_unembed(tiny):
    lm, _ = tiny
    from c15a.lens import Lens
    d = lm.d_model
    lens = Lens({"J": {0: torch.eye(d).half()}, "n_prompts": 1, "source_layers": [0], "d_model": d})
    h = torch.randn(3, d)
    assert torch.allclose(lens.readout(lm, h, 0), lm.unembed(h).float(), atol=1e-5)


def test_check_lens_checkpoint_errors():
    from conftest import tiny_lens_ck
    ck = tiny_lens_ck(6, 64)
    assert check_lens_checkpoint(ck, n_layers=6, d_model=64, hf_model_id="x/y")["n_lens_layers"] == 5
    with pytest.raises(Z0Error):
        check_lens_checkpoint(ck, n_layers=7, d_model=64, hf_model_id="x/y")
    with pytest.raises(Z0Error):
        check_lens_checkpoint(ck, n_layers=6, d_model=32, hf_model_id="x/y")
    ck2 = dict(ck, provenance={"model_id": "other/model", "target_layer": 5})
    with pytest.raises(Z0Error):
        check_lens_checkpoint(ck2, n_layers=6, d_model=64, hf_model_id="x/y")
    ck3 = dict(ck, provenance={"model_id": "x/y", "target_layer": 4})
    with pytest.raises(Z0Error):
        check_lens_checkpoint(ck3, n_layers=6, d_model=64, hf_model_id="x/y")


def test_iso_scale_cases():
    sc = C.RAND_SCALE_GRID
    s, st = iso_scale([0, 2, 4, 8, 12, 20, 30], sc, 6.0)
    assert st == "interpolated" and 2.0 < s < 3.0
    assert iso_scale([0, 1, 1, 1, 1, 1, 1], sc, 6.0)[1] == "unreached_upper_bound"
    assert iso_scale([7, 8, 9, 10, 11, 12, 13], sc, 6.0)[1] == "at_grid_min_lower_bound"
    V = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(interp_in_log_scale(V, [1.0, 4.0], 2.0), [0.2, 0.3])


def _primal_cv_r2(X, y, f, lam=C.W3_RIDGE_LAMBDA):
    X, y = X.double(), y.double()
    pred = torch.zeros_like(y)
    for k in torch.unique(f):
        te, tr = f == k, f != k
        mu, my = X[tr].mean(0), y[tr].mean()
        Xc = X[tr] - mu
        reg = lam * float(torch.diagonal(Xc @ Xc.T).mean())
        w = torch.linalg.solve(Xc.T @ Xc + reg * torch.eye(X.shape[1], dtype=X.dtype), Xc.T @ (y[tr] - my))
        pred[te] = (X[te] - mu) @ w + my
    return 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def test_ridge_kernel_equals_primal_and_detects_injected_signal():
    g = torch.Generator().manual_seed(6)
    n, d = 40, 300
    ctx = torch.randn(n, d, generator=g)                      # context variation
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    s = torch.tensor([(-1.0, -0.5, 0.5, 1.0)[i % 4] for i in range(n)])
    f = torch.arange(n) % 5
    X = ctx + 6.0 * s[:, None] * v[None]                        # rank-1 injected scalar, as in W3
    r2 = ridge_cv_r2(X, s, f)
    assert r2 > 0.5
    assert abs(ridge_cv_r2_kernel(kernel(X), s, f) - r2) < 1e-9
    assert abs(_primal_cv_r2(X, s, f) - r2) < 1e-6
    assert ridge_cv_r2(ctx, s, f) < C.W3_R2_SITE                  # no injection -> not decodable


def test_guards_refuse_unauthorized_phases_and_sealed_confirm(tmp_path, monkeypatch):
    for ph in ("B", "C", "H", "SAT", "F", "RESCUE", "TRAIN"):
        with pytest.raises(guards.GuardError):
            guards.check_phase(ph)
    for ph in C.PHASES_AUTHORIZED:
        guards.check_phase(ph)
    monkeypatch.setattr(guards, "FREEZE_PATH", str(tmp_path / "freeze_astage.json"))
    with pytest.raises(guards.GuardError):
        guards.confirm_gate({"files": {"g_confirm.json": "x"}})


def _fake_res(w1J=0.5, w1P=0.1, w2J=0.4, w2P=0.1, w3=0.4, w4=20.0, si=0.8, ci=1.0, w0=True):
    pl = {"w1": {"hit_J": w1J, "hit_perp": w1P, "ratio": w1J / w1P, "pass": w1J >= .3 and w1J / w1P >= 2},
          "w2": {"rate_J": w2J, "rate_perp": w2P, "ratio": w2J / w2P, "pass": w2J >= .25 and w2J / w2P >= 2},
          "w3": {"diff": w3, "pass": w3 >= .15},
          "w45": {"w4_diff_pp": w4, "pass_w4": w4 >= 10, "SI_iso": si, "pass_w5": si <= 1 and ci <= 1.25}}
    return {"w0": {"pass": w0}, "per_layer": {"8": pl}}


def test_choose_rule():
    out = choose({"qwen3-4b": _fake_res(w1J=0.6), "qwen3-1.7b": _fake_res(), "qwen3.5-2b": _fake_res(si=1.1)})
    g, m, ok, mm = layer_gate(_fake_res()["w0"], _fake_res()["per_layer"]["8"])
    assert ok
    assert out["n_passers"] == 2
    # tie on min margin -> smaller model; replication candidate is the other passer
    assert out["chosen"]["model"] == "qwen3-1.7b" and out["replication_candidate"]["model"] == "qwen3-4b"
    none = choose({"qwen3-4b": _fake_res(w0=False)})
    assert none["chosen"] is None


def test_choose_after_json_round_trip_with_infinite_ratios():
    import run_astage
    res = _fake_res()
    res["per_layer"]["8"]["w1"]["ratio"] = math.inf
    res["per_layer"]["8"]["w2"]["ratio"] = math.inf
    rt = json.loads(json.dumps(run_astage._clean({"qwen3-1.7b": res})))
    assert rt["qwen3-1.7b"]["per_layer"]["8"]["w1"]["ratio"] == "inf"
    out = choose(rt)
    assert out["chosen"]["model"] == "qwen3-1.7b"
    assert out["chosen"]["margins"]["W1"] == min((0.5 - 0.3) / 0.3, (C.MARGIN_RATIO_CAP - 2) / 2)


def test_material_splits_disjoint_and_hashed():
    from c15a.materials import MAT, split_manifest
    from c15a.artifacts import sha256_file
    man = split_manifest()
    sets = {}
    for fn in ("g_select.json", "g_confirm.json", "smoke.json"):
        p = os.path.join(MAT, fn)
        assert sha256_file(p) == man["files"][fn]
        s = json.load(open(p, encoding="utf-8"))
        sets[fn] = {x["text"] for x in s["paragraphs"]} | {x["word"] for x in s["concepts"]} | \
            {x["name"] for x in s["countries"]} | {x["prompt"] for x in s["two_hop"]}
        txt = json.dumps(s).lower()
        for banned in ("satisf", "clause", "valid", "verif", "assignment"):
            assert banned not in txt, banned
    a, b, c = sets.values()
    assert not (a & b) and not (a & c) and not (b & c)
    sel = json.load(open(os.path.join(MAT, "g_select.json"), encoding="utf-8"))
    con = json.load(open(os.path.join(MAT, "g_confirm.json"), encoding="utf-8"))
    assert {t["bridge"] for t in sel["two_hop"]}.isdisjoint({t["bridge"] for t in con["two_hop"]})


def test_astage_end_to_end_on_tiny_model(tiny):
    """Plumbing only: every assay runs and returns its fields on a random tiny model (smoke material)."""
    from c15a.assays import AStage
    from c15a.materials import load_smoke
    lm, lens = tiny
    st = AStage(lm, lens, load_smoke(), layers=[2])
    st.boot_w3, st.boot_w5 = 3, 20
    res = st.run_all()
    pl = res["per_layer"][2]
    for k in ("w1", "w2", "w3", "w45", "dose", "sj"):
        assert k in pl
    assert {"pass_w4", "pass_w5", "SI_iso", "SI_norm", "s_star"} <= set(pl["w45"])
    assert "pass" in res["w0"]
    g, m, ok, mm = layer_gate(res["w0"], pl)
    assert set(g) == {"W0", "W1", "W2", "W3", "W4", "W5"}


def test_runner_has_no_unauthorized_phase():
    import run_astage
    src = open(run_astage.__file__, encoding="utf-8").read()
    for ph in C.PHASES_FORBIDDEN:
        assert f'"{ph}": ' not in src
    pkg = os.path.join(os.path.dirname(run_astage.__file__), "c15a")
    for fn in os.listdir(pkg):
        if fn.endswith(".py"):
            s = open(os.path.join(pkg, fn), encoding="utf-8").read().lower()
            assert "def make_sat" not in s and "def fit_f" not in s and "rescue_route" not in s
