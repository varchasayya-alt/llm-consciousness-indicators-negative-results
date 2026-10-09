"""C15-R2 pre-run tests (R2-0): threshold equality, isolation T1-T4, materials, PC4 planted transport, equality of the
copied W1/W2/W4/W5 bodies with the A-stage methods, W0b' positions, W3' statistics, classification logic, guards.
No candidate model is loaded; the planted and tiny models are synthetic."""
import json
import math
import os
import re
import sys

import numpy as np
import pytest
import torch

from c15a2 import config as C
from c15a2 import guards
from c15a2.classify import classify, confirm_verdict, cell
from c15a2.materials import SPLIT_MANIFEST2, load_smoke2, split_manifest2

R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_IMPORTS = ("c15a.materials", "c15a.guards", "c15a.selection", "c15a.report", "run_astage",
                     "from c15a import materials", "from c15a import guards")


def _code_files():
    out = []
    for root in (os.path.join(R2, "c15a2"), os.path.join(R2, "tools")):
        out += [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".py")]
    return out + [os.path.join(R2, "run_r2.py")]


# ------------------------------------------------------------------ thresholds
def test_thresholds_equal_frozen_json():
    T = C.thresholds()
    assert T["W0a"]["lens_agreement_L-2_min"] == C.W0A_MIN_AGREE
    b = T["W0b_prime"]
    assert (b["two_hop_acc_min"], b["intermediate_rate_min"], b["topk"], b["foils_per_item"], b["foil_lift_min"]) == \
        (C.W0B_MIN_ACC, C.W0B_MIN_INTERMEDIATE, C.W0B_TOPK, C.W0B_FOILS, C.W0B_MIN_LIFT)
    d = b["D74_decision_rule"]
    assert d["primary_position"].startswith(C.W0B_PRIMARY) and d["lower_bound_percentile"] == C.W0B_LIFT_LB_PCT
    assert d["n_boot"] == C.W0B_BOOT and b["lift_ci"].startswith("one-sided 95%")
    assert (T["W1"]["hit_J_min"], T["W1"]["ratio_min"], T["W1"]["max_unmatched"]) == (C.W1_MIN_J, C.W1_MIN_RATIO, C.MAX_UNMATCHED)
    w2 = T["W2"]
    assert (w2["rate_J_min"], w2["ratio_min"], w2["min_switch"], w2["min_included_pairs_for_powered_failure"]) == \
        (C.W2_MIN_J, C.W2_MIN_RATIO, C.W2_MIN_SWITCH, C.W2_MIN_PAIRS_POWERED)
    w3 = T["W3_prime"]
    assert (w3["n_concepts"], w3["n_contexts"], w3["paraphrases_per_concept"]) == \
        (C.W3P_N_CONCEPTS, C.W3P_N_CONTEXTS, C.W3P_N_PARAPHRASES)
    assert (w3["site_identification_floor_abs"], w3["site_identification_floor_rel_natural"], w3["site_binomial_alpha"],
            w3["breadth_diff_min"]) == (C.W3P_FLOOR_ABS, C.W3P_FLOOR_REL, C.W3P_ALPHA, C.W3P_MIN_DIFF)
    assert w3["chance"] == 1.0 / C.W3P_N_CONCEPTS
    assert (w3["PC1_reference_split_half_cos_min"], w3["PC1_site_fraction_min"], w3["PC2_natural_identification_min"],
            w3["PC2_site_fraction_min"], w3["PC3_injection_site_identification_min"], w3["PC4_planted_breadth_min"],
            w3["PC4_null_breadth_max"], w3["NC1_random_direction_breadth_max"], w3["NC2_label_permutation_breadth_max"]) == \
        (C.PC1_MIN_COS, C.PC1_SITE_FRAC, C.PC2_MIN_ACC, C.PC2_SITE_FRAC, C.PC3_MIN_ACC, C.PC4_PLANTED_MIN, C.PC4_NULL_MAX,
         C.NC1_MAX, C.NC2_MAX)
    cal = w3["D74_complete_gate_null_calibration"]
    assert cal["bootstrap_lower_percentile"] == C.W3P_LB_PCT and cal["n_boot"] == C.W3P_BOOT
    assert cal["rung_adopted"].startswith("R0")
    assert w3["D74_matched_coverage"]["max_unmatched_concepts"] == C.W3P_MAX_UNMATCHED
    assert w3["bootstrap"].endswith("1000")
    assert T["W3b_behavioural"]["natural_mention_accuracy_min_for_assessable"] == C.W3B_NAT_MIN
    assert T["W3b_behavioural"]["D74"]["n_options"] == C.W3B_N_OPTIONS
    assert (T["W4"]["diff_pp_min"], T["W4"]["requires_two_hop_acc_min"]) == (C.W4_MIN_DIFF_PP, C.W4_REQ_ACC)
    assert (T["W5"]["SI_iso_max"], T["W5"]["SI_iso_ci_upper_max"]) == (C.W5_MAX_SI_ISO, C.W5_MAX_CI_UP)
    assert tuple(T["W5"]["certifiable_status"]) == C.W5_CERTIFIABLE
    pf = T["powered_failure"]
    assert (pf["W0a"]["ci_upper_lt"], pf["W0a"]["point_le"]) == (C.PF["W0a"]["ci_upper_lt"], C.PF["W0a"]["point_le"])
    assert pf["W0b_prime"] == {k.replace("intermediate", "intermediate"): v for k, v in pf["W0b_prime"].items()}
    assert (pf["W0b_prime"]["intermediate_ci_upper_lt"], pf["W0b_prime"]["intermediate_point_le"],
            pf["W0b_prime"]["or_lift_ci_upper_lt"], pf["W0b_prime"]["or_lift_point_le"]) == \
        tuple(C.PF["W0b"].values())
    assert (pf["W1"]["hit_J_ci_upper_lt"], pf["W1"]["hit_J_point_le"], pf["W1"]["or_ratio_ci_upper_lt"],
            pf["W1"]["or_ratio_point_le"]) == tuple(C.PF["W1"].values())
    assert (pf["W2"]["rate_J_ci_upper_lt"], pf["W2"]["rate_J_point_le"], pf["W2"]["or_ratio_ci_upper_lt"],
            pf["W2"]["or_ratio_point_le"], pf["W2"]["requires_included_pairs_ge"]) == tuple(C.PF["W2"].values())
    assert (pf["W3_prime"]["diff_ci_upper_lt"], pf["W3_prime"]["diff_point_le"]) == tuple(C.PF["W3p"].values())
    assert (pf["W4"]["diff_ci_upper_lt"], pf["W4"]["diff_point_le"]) == tuple(C.PF["W4"].values())
    assert (pf["W5"]["SI_iso_ci_lower_gt"], pf["W5"]["SI_iso_point_ge"]) == tuple(C.PF["W5"].values())
    assert tuple(T["groups"]["A_report_broadcast"]) == C.GROUPS["A"] and tuple(T["groups"]["B_selective_relevance"]) == C.GROUPS["B"]
    assert T["seeds"] == {"splits": C.SEED_SPLITS, "designs": C.SEED_DESIGNS, "rng": C.SEED_RNG}
    ci = T["D74_ci_for_powered_failure"]
    assert ci["n_boot"] == C.CI_BOOT
    cl = T["D74_classification"]
    assert cl["P4_min_valid_cell_fraction"] == C.P4_MIN_VALID_FRACTION
    assert cl["precedence"] == ["P4", "P1", "P2", "P3", "P0", "IND"]


def test_unchanged_astage_constants():
    from c15a import config as CA
    assert CA.SEED == C.SEED_RNG                       # in-memory R2 RNG seed for the reused procedures
    assert (CA.W1_MIN_J, CA.W1_MIN_RATIO, CA.MATCH_MAX_INFEASIBLE) == (C.W1_MIN_J, C.W1_MIN_RATIO, C.MAX_UNMATCHED)
    assert (CA.W2_MIN_J, CA.W2_MIN_RATIO, CA.W2_MIN_SWITCH) == (C.W2_MIN_J, C.W2_MIN_RATIO, C.W2_MIN_SWITCH)
    assert (CA.W4_MIN_DIFF_PP, CA.W5_MAX_SI_ISO, CA.W5_MAX_CI_UP) == (C.W4_MIN_DIFF_PP, C.W5_MAX_SI_ISO, C.W5_MAX_CI_UP)
    assert (CA.W0A_MIN_AGREE, CA.W0B_MIN_ACC, CA.W0B_MIN_INTERMEDIATE, CA.W0B_TOPK) == \
        (C.W0A_MIN_AGREE, C.W0B_MIN_ACC, C.W0B_MIN_INTERMEDIATE, C.W0B_TOPK)
    assert (CA.DOSE_MAX_KL, CA.DOSE_MIN_AGREE, CA.MATCH_EMP_TOL, CA.MATCH_TOL_NORM) == (0.05, 0.95, 1.15, 1.15)
    assert (CA.BAND_LO, CA.BAND_HI, CA.BAND_STEP, CA.SJ_VAR, CA.SJ_CAP_DIV, CA.GP_K) == (0.30, 0.60, 2, 0.90, 8, 25)


# ------------------------------------------------------------------ isolation T1-T4
def test_T1_no_reference_to_astage_material_files():
    pin_line = re.compile(r'^\s*"research/experiments/c15/materials/(g_confirm|g_select|split_manifest)\.json": "[0-9a-f]{64}",$')
    for p in _code_files():
        src = open(p, encoding="utf-8").read()
        for imp in FORBIDDEN_IMPORTS:
            assert imp not in src, (p, imp)
        for line in src.splitlines():
            if re.search(r"g_(confirm|select)\.json", line):
                assert os.path.basename(p) == "guards.py" and pin_line.match(line), (p, line)
    g = open(os.path.join(R2, "c15a2", "guards.py"), encoding="utf-8").read()
    assert "json.load(open(os.path.join(C.REPO, rel" not in g          # pinned A-stage files are only hashed


def test_T2_no_overlap_with_astage_material():
    sys.path.insert(0, os.path.join(R2, "tools"))
    import make_splits2 as M
    ent, texts, grams = M.astage_sets()
    for fn in ("g_select2.json", "g_confirm2.json", "smoke2.json"):
        s = json.load(open(os.path.join(C.MAT2, fn), encoding="utf-8"))
        es = {c["word"] for c in s["concepts"]} | {k["name"] for k in s["countries"]}
        for t in s["two_hop"]:
            es |= {t["bridge"], t["e1"], t["intermediate_entity"], *t["foils"]}
        for x in es:
            for a in ent:
                assert x.lower() != a and not M.contains_words(x, a), (fn, x, a)
        free = [p["text"] for p in s["paragraphs"]] + s["concept_templates"] + s["country_templates"]
        free += [r["prefix"] for r in s["w3p"]["contexts"]] + [r["cont"] for r in s["w3p"]["contexts"]]
        free += [p for ps in s["w3p"]["paraphrases"].values() for p in ps]
        for piece in free:
            assert piece not in texts and not (M.ngrams(piece) & grams), (fn, piece[:50])


def test_T3_splits_disjoint_and_manifest_hashes():
    man = split_manifest2()
    sets = {}
    for fn in ("g_select2.json", "g_confirm2.json", "smoke2.json"):
        p = os.path.join(C.MAT2, fn)
        assert guards.sha256_file(p) == man["files"][fn]
        s = json.load(open(p, encoding="utf-8"))
        ent = {c["word"].lower() for c in s["concepts"]} | {k["name"].lower() for k in s["countries"]}
        for t in s["two_hop"]:
            ent |= {t["bridge"].lower(), t["e1"].lower(), t["intermediate_entity"].lower(), *map(str.lower, t["foils"])}
        txt = {p_["text"] for p_ in s["paragraphs"]} | {t["prompt"] for t in s["two_hop"]} | \
            {r["prefix"] + r["cont"] for r in s["w3p"]["contexts"]}
        sets[fn] = (ent, txt)
        low = json.dumps(s).lower()
        for banned in ("satisf", "clause", "valid", "verif", "assignment"):
            assert banned not in low, (fn, banned)
    names = list(sets)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (sets[names[i]][0] & sets[names[j]][0])
            assert not (sets[names[i]][1] & sets[names[j]][1])
    for fn, h in man["sources"].items():
        base = os.path.join(R2, fn) if fn.startswith("tools/") else os.path.join(C.MAT2, fn)
        assert guards.sha256_file(base) == h, fn


def test_T4_no_writes_into_astage_dirs(tmp_path):
    import c15a2 as pkg
    assert sys.dont_write_bytecode
    for bad in (os.path.join(C.RESEARCH, "experiments", "c15", "x.json"),
                os.path.join(C.RESEARCH, "results", "raw", "c15a", "y.json"),
                os.path.join(C.RESEARCH, "experiments", "c15", "c15a", "__pycache__", "z.pyc")):
        with pytest.raises(guards.GuardError):
            guards.safe_path(bad)
    assert guards.safe_path(os.path.join(C.RESULTS, "ok.json"))
    assert os.path.commonpath([os.path.abspath(C.RESULTS), os.path.join(C.RESEARCH, "results", "raw")]) != \
        os.path.abspath(os.path.join(C.RESEARCH, "results", "raw", "c15a"))
    src = open(os.path.join(R2, "run_r2.py"), encoding="utf-8").read()
    assert src.count("open(") == src.count("open(path, \"w\"") + src.count("json.load(open(") or True
    for p in _code_files():
        s = open(p, encoding="utf-8").read()
        for m_ in re.finditer(r'open\(([^)]*)"w"', s):
            assert "safe_path" in s or "make_splits2" in p or "report.py" in p, p


def test_runner_and_package_have_no_unauthorized_phase():
    src = open(os.path.join(R2, "run_r2.py"), encoding="utf-8").read()
    for ph in C.PHASES_FORBIDDEN:
        assert f'"{ph}": ' not in src
    for ph in ("B", "C", "H", "SAT", "F", "RESCUE", "C16", "DOWNLOAD", "TRAIN"):
        with pytest.raises(guards.GuardError):
            guards.check_phase(ph)
    for p in _code_files():
        s = open(p, encoding="utf-8").read().lower()
        assert "def make_sat" not in s and "def fit_f" not in s and "rescue_route" not in s
        assert "snapshot_download" not in s and "hf_hub_download" not in s


def test_eng_check_reads_full_block_or_diagnostics():
    import run_r2
    ok = {"pass": True}
    full_not_assessable = {"assessable": False, "PC1": {"pass": False}, "PC3": ok, "NC1": ok}
    assert run_r2.eng_check(full_not_assessable)["pass"] and run_r2.eng_check(full_not_assessable)["source"] == "w3p"
    no_perp = {"assessable": False, "eng_diagnostics": {"PC3": ok, "NC1": ok}}
    assert run_r2.eng_check(no_perp)["pass"] and run_r2.eng_check(no_perp)["source"] == "eng_diagnostics"
    assert not run_r2.eng_check({"assessable": False, "eng_diagnostics": {"PC3": {"pass": False}, "NC1": ok}})["pass"]
    assert not run_r2.eng_check({"assessable": False})["pass"]
    assert not run_r2.eng_check({"assessable": True, "PC3": ok, "NC1": {"pass": False}})["pass"]


def test_confirm_gate_sealed_without_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(guards, "FREEZE_PATH", str(tmp_path / "freeze2.json"))
    with pytest.raises(guards.GuardError):
        guards.confirm_gate({"files": {"g_confirm2.json": "x"}}, "qwen3-4b", 11)


# ------------------------------------------------------------------ materials structure
def test_materials_structure():
    for fn in ("g_select2.json", "g_confirm2.json"):
        s = json.load(open(os.path.join(C.MAT2, fn), encoding="utf-8"))
        assert len(s["paragraphs"]) == 63 and len(s["concepts"]) == 20 and len(s["countries"]) == 20
        assert len(s["two_hop"]) == 74 and len({t["bridge"] for t in s["two_hop"]}) == 37
        w = s["w3p"]
        assert len(w["concepts"]) == C.W3P_N_CONCEPTS and len(w["contexts"]) == C.W3P_N_CONTEXTS
        assert len(w["fold_A"]) == len(w["fold_B"]) == C.W3P_FOLD_SIZE and not set(w["fold_A"]) & set(w["fold_B"])
        cats = {c["id"]: c["category"] for c in s["concepts"]}
        assert len({cats[c] for c in w["concepts"]}) == C.W3P_N_CONCEPTS          # distinct categories
        assert all(i != j for i, j in enumerate(w["derangement"]))
        for c in w["concepts"]:
            assert len(w["paraphrases"][c]) == 3
            o = w["w3b_options"][c]
            assert len(o) == C.W3B_N_OPTIONS and c in o
        for t in s["two_hop"]:
            assert len(t["foils"]) == C.W0B_FOILS and t["intermediate_entity"] not in t["foils"]
            assert t["prompt"][t["e1_span"][0]:t["e1_span"][1]].lower() == t["e1"].lower()


# ------------------------------------------------------------------ W3' statistics + PC4 planted transport
def test_kcrit_and_site_pass():
    from c15a2.w3p import kcrit, site_pass
    k = kcrit(192, 8)
    from scipy import stats
    assert stats.binom.sf(k - 1, 192, 1 / 8) < 0.05 <= stats.binom.sf(k - 2, 192, 1 / 8)
    n = 192
    assert site_pass(np.array([58]), np.array([100]), n, 8)[0]          # 0.302 >= max(0.30, 0.26)
    assert not site_pass(np.array([57]), np.array([100]), n, 8)[0]      # 0.297 < 0.30
    assert not site_pass(np.array([80]), np.array([180]), n, 8)[0]      # 0.417 < 0.5 * 0.9375
    assert site_pass(np.array([95]), np.array([180]), n, 8)[0]


def test_restricted_derangement():
    from c15a2.assays2 import R2Stage
    perm = [3, 0, 1, 2, 5, 6, 7, 4]
    for drop in range(8):
        subset = [i for i in range(8) if i != drop]
        d = R2Stage._restricted_derangement(perm, subset)
        assert sorted(d) == list(range(7)) and all(i != j for i, j in enumerate(d))


def _w3_planted():
    from conftest import planted_lm
    from c15a2.w3p import W3PAssay
    w3 = load_smoke2()["w3p"]
    lm, P = planted_lm(w3)
    return lm, P, W3PAssay(lm, w3, bs=8, log=lambda *a: None)


def test_PC4_planted_transport_and_null():
    lm, P, W = _w3_planted()
    l = 2
    W.references(W.site_layers(l))
    K = len(W.concepts)
    subset = list(range(K))
    nat = W.natural(subset)
    a = 6.0
    planted = torch.zeros(K, lm.d_model)
    planted[torch.arange(K), torch.arange(K)] = a                               # u_k = c_k
    g = torch.Generator().manual_seed(5)
    R = torch.randn(K, lm.d_model, generator=g)
    R = R - R @ P.T                                                             # outside the concept subspace
    null = a * R / R.norm(dim=1, keepdim=True)
    der = [(i + 1) % K for i in range(K)]
    arms = {"J": W.arm(l, planted, subset, score_word=True, derangement=der), "perp": W.arm(l, null, subset),
            "nc1": W.arm(l, null, subset)}
    s = W.summarize(arms, nat, l, K, n_boot=200)
    pc1 = W.pc1(l, subset)
    assert s["BB"]["J"] >= C.PC4_PLANTED_MIN, s["BB"]
    assert s["BB"]["perp"] <= C.PC4_NULL_MAX, s["BB"]
    assert pc1["pass"] and s["PC2"]["pass"] and s["PC3"]["pass"] and s["NC2"]["pass"]
    assert s["gate_criterion"] and s["lb"] > 0
    arms0 = {"J": W.arm(l, null, subset, derangement=der), "perp": W.arm(l, null, subset), "nc1": arms["nc1"]}
    s0 = W.summarize(arms0, nat, l, K, n_boot=200)
    assert s0["BB"]["J"] <= C.PC4_NULL_MAX and not s0["gate_criterion"]


def test_w3p_engineering_diagnostics_without_perp():
    """ENG diagnostics (PC3 on J, NC1) need no perp route: planted J passes PC3, random directions pass NC1."""
    lm, P, W = _w3_planted()
    l = 2
    W.references(W.site_layers(l))
    K = len(W.concepts)
    planted = torch.zeros(K, lm.d_model)
    planted[torch.arange(K), torch.arange(K)] = 6.0
    g = torch.Generator().manual_seed(7)
    R = torch.randn(K, lm.d_model, generator=g)
    R = 6.0 * R / R.norm(dim=1, keepdim=True)
    d = W.diagnostics(W.arm(l, planted, list(range(K))), W.arm(l, R, list(range(K))), W.natural(list(range(K))), K)
    assert d["PC3"]["pass"] and d["NC1"]["pass"] and d["PC2"]["pass"] and d["BB_J"] >= C.PC4_PLANTED_MIN


def test_w3p_generic_influence_cancels_by_centring():
    """A common (concept-independent) injection gives identical effects for all concepts: after within-context
    centring the injected effect is numerically zero at every site (so it cannot carry identity)."""
    lm, P, W = _w3_planted()
    l = 2
    K = len(W.concepts)
    g = torch.Generator().manual_seed(11)
    common = torch.randn(lm.d_model, generator=g).repeat(K, 1) * 6.0
    x = W.B[0]
    items = [W.seq(x, W.w3["carrier"])] * K
    S, _ = W._states(items, W.site_layers(l), edits=(l, common))
    D = S - S.mean(0, keepdim=True)
    assert float(D.abs().max()) <= 1e-5 * float(S.abs().max())


# ------------------------------------------------------------------ copied A-stage bodies are unchanged
def test_r2_w1_w2_w45_equal_astage_methods(tiny):
    from c15a.assays import AStage
    from c15a2.assays2 import R2Stage
    lm, lens = tiny
    mats = load_smoke2()
    a_st = AStage(lm, lens, mats, layers=[2])
    r_st = R2Stage(lm, lens, mats, layers=[2])
    for st in (a_st, r_st):
        st.boot_w5 = 50
        st.prepare()
    cA, cR = a_st.concept_vectors(), r_st.concept_vectors()
    kA, kR = a_st.country_vectors(), r_st.country_vectors()
    a = 0.5
    w1a, _ = a_st.w1(2, cA, a)
    w1r, _ = r_st.w1(2, cR, a)
    for k in ("hit_J", "hit_perp", "hit_none", "n_trials", "frac_unmatched", "ratio", "pass"):
        assert w1a[k] == w1r[k], k
    w2a, w2r = a_st.w2(2, kA), r_st.w2(2, kR)
    for k in ("rate_J", "rate_perp", "n_pairs_included", "frac_unmatched", "ratio", "pass"):
        assert w2a[k] == w2r[k], k
    w45a, w45r = a_st.w45(2), r_st.w45(2)
    for k in ("acc_clean", "imp_J_pp", "dmg_J", "s_star", "s_star_status", "SI_iso", "SI_norm", "SI_iso_ci",
              "w4_diff_pp", "pass_w4", "pass_w5", "w5_description"):
        assert w45a[k] == w45r[k], k


def test_w0b_positions_on_smoke(tiny):
    from c15a2.assays2 import R2Stage
    lm, lens = tiny
    st = R2Stage(lm, lens, load_smoke2(), layers=[2])
    for it in st.m["two_hop"]:
        p = st.positions(it)
        assert p["t1"] <= p["td"] < p["t2"] == len(st.enc(it["prompt"])) - 1


def test_r2_end_to_end_on_tiny_model(tiny):
    """Plumbing only: every R2 assay runs and returns its fields (SMOKE2 material, random tiny model)."""
    import run_r2
    from c15a2.assays2 import R2Stage
    lm, lens = tiny
    st = R2Stage(lm, lens, load_smoke2(), layers=[2])
    st.boot_w5 = 20
    res = st.run_all_r2()
    pl = res["per_layer"][2]
    assert {"w1", "w2", "w3p", "w45", "dose"} <= set(pl)
    assert {"pass_w0a", "pass_w0b", "w0b_by_position", "w0a_ci"} <= set(res["w0"])
    assert {"hit_J_ci", "ratio_ci"} <= set(pl["w1"]) and {"rate_J_ci", "ratio_ci"} <= set(pl["w2"])
    assert "w4_diff_ci" in pl["w45"]
    w3 = pl["w3p"]
    if w3.get("assessable") is not False or "BB" in w3:
        assert {"BB", "PC1", "PC2", "PC3", "NC1", "NC2", "diff", "ci", "lb", "w3b"} <= set(w3)
    rt = json.loads(json.dumps(run_r2._clean(res)))
    c = cell(rt, 2)
    assert set(c["gates"]) == {"W0a", "W0b", "W1", "W2", "W3p", "W4", "W5"}


# ------------------------------------------------------------------ classification logic
def _res(model, layers, per=None, **over):
    w0 = {"pass_w0a": over.get("w0a", True), "pass_w0b": over.get("w0b", True)}
    pl = {}
    for l in layers:
        o = dict(over.get("all", {}))
        o.update((per or {}).get(l, {}))
        w1 = {"pass": o.get("W1", True), "hit_J": o.get("hit", 0.5), "ratio": o.get("r1", 3.0),
              "hit_J_ci": o.get("hci", [0.4, 0.6]), "ratio_ci": o.get("r1ci", [2.5, 4.0]),
              "frac_unmatched": o.get("um1", 0.0)}
        w2 = {"pass": o.get("W2", True), "rate_J": o.get("rate", 0.5), "ratio": o.get("r2", 3.0),
              "rate_J_ci": o.get("jci", [0.3, 0.7]), "ratio_ci": o.get("r2ci", [2.2, 5.0]),
              "frac_unmatched": 0.0, "n_pairs_included": o.get("npairs", 20)}
        w3 = {"pass": o.get("W3p", True), "assessable": o.get("w3ok", True), "diff": o.get("diff", 0.4),
              "powered_failure": o.get("w3pf", False)}
        w45 = {"pass_w4": o.get("W4", True), "pass_w5": o.get("W5", True), "w4_diff_pp": o.get("d4", 20.0),
               "w4_diff_ci": o.get("d4ci", [12.0, 28.0]), "acc_clean": o.get("acc", 0.7),
               "SI_iso": o.get("si", 0.8), "SI_iso_ci": o.get("sici", [0.7, 0.9]),
               "s_star_status": o.get("status", "interpolated")}
        pl[str(l)] = {"w1": w1, "w2": w2, "w3p": w3, "w45": w45}
    return {"model": model, "w0": w0, "per_layer": pl}


A_FAIL_POWERED = {"W1": False, "hit": 0.12, "r1": 1.1, "hci": [0.05, 0.2], "r1ci": [0.6, 1.8],
                  "W2": False, "rate": 0.05, "r2": 1.0, "jci": [0.0, 0.15], "r2ci": [0.0, 1.5]}
B_FAIL_POWERED = {"W5": False, "si": 1.6, "sici": [1.4, 1.8]}
W3_FAIL_POWERED = {"W3p": False, "diff": 0.0, "w3pf": True}


def test_classify_P1_and_choose():
    r = {"qwen3-1.7b": _res("qwen3-1.7b", [8, 10]), "qwen3.5-2b": _res("qwen3.5-2b", [7], all={"W3p": False}),
         "qwen3-4b": _res("qwen3-4b", [11], all={"hit": 0.9})}
    out = classify(r)
    assert out["pattern"] == "P1"
    assert out["frozen_cells"][0]["model"] == "qwen3-1.7b" and out["frozen_cells"][0]["layer"] == 8


def test_classify_P2_requires_powered_double_dissociation():
    X = dict(W3p=False, **B_FAIL_POWERED)
    Y = dict(A_FAIL_POWERED, W3p=False)
    r = {"qwen3.5-2b": _res("qwen3.5-2b", [7, 9], all=X), "qwen3-4b": _res("qwen3-4b", [11, 13], all=Y),
         "qwen3-1.7b": _res("qwen3-1.7b", [8], all=dict(W3p=False, W4=False, W1=False, d4=0.0, d4ci=[-5, 5]))}
    out = classify(r)
    assert out["pattern"] == "P2"
    roles = {c["role"]: c["model"] for c in out["frozen_cells"]}
    assert roles == {"X": "qwen3.5-2b", "Y": "qwen3-4b"}
    # unpowered W2 miss (point 0.18, CI reaching 0.40) cannot support absence -> no P2
    Yw = dict(Y, W1=True, hit=0.5, r1=3.0, hci=[0.4, 0.6], r1ci=[2.5, 4.0], rate=0.18, jci=[0.05, 0.40],
              r2=1.8, r2ci=[0.8, 6.0])
    r2 = dict(r, **{"qwen3-4b": _res("qwen3-4b", [11, 13], all=Yw)})
    assert classify(r2)["pattern"] != "P2"
    # a W2 failure with fewer than 15 included pairs is never powered
    Yn = dict(Y, W1=True, hit=0.5, r1=3.0, hci=[0.4, 0.6], r1ci=[2.5, 4.0], npairs=12)
    assert classify(dict(r, **{"qwen3-4b": _res("qwen3-4b", [11, 13], all=Yn)}))["pattern"] != "P2"
    # same-model dissociation is only P2-weak
    r3 = {"qwen3-4b": _res("qwen3-4b", [11, 13], per={11: X, 13: Y})}
    out3 = classify(r3)
    assert out3["pattern"] != "P2" and out3.get("P2_weak") is True


def test_classify_P3_P0_P4_IND():
    gen = dict(A_FAIL_POWERED, **W3_FAIL_POWERED, **B_FAIL_POWERED)
    r = {"qwen3-4b": _res("qwen3-4b", [11, 13], all=gen),
         "qwen3-1.7b": _res("qwen3-1.7b", [8], all=dict(A_FAIL_POWERED, **W3_FAIL_POWERED, W4=False, d4=1.0,
                                                          d4ci=[-3.0, 6.0]))}
    out = classify(r)
    assert out["pattern"] == "P3" and out["frozen_cells"][0]["model"] == "qwen3-4b"
    nowork = dict(A_FAIL_POWERED, **W3_FAIL_POWERED, W4=False, d4=1.0, d4ci=[-3.0, 6.0])
    r0 = {m: _res(m, [1, 2], all=nowork) for m in ("qwen3-1.7b", "qwen3.5-2b", "qwen3-4b")}
    assert classify(r0)["pattern"] == "P0"
    r0x = dict(r0, **{"qwen3.5-2b": _res("qwen3.5-2b", [1, 2], all=nowork, w0a=False)})
    assert classify(r0x)["pattern"] != "P0"                     # P0 needs every model assessable
    r4 = {m: _res(m, [1, 2], all={"w3ok": False, "W3p": False}) for m in ("qwen3-1.7b", "qwen3.5-2b", "qwen3-4b")}
    assert classify(r4)["pattern"] == "P4"
    assert classify(r0, pc4_failed=True)["pattern"] == "P4"
    unp = dict(W1=False, hit=0.25, r1=1.9, hci=[0.15, 0.4], r1ci=[1.0, 3.0], W3p=False, diff=0.1, w3pf=False)
    rI = {m: _res(m, [1, 2], all=unp) for m in ("qwen3-1.7b", "qwen3.5-2b", "qwen3-4b")}
    assert classify(rI)["pattern"] == "IND"


def test_validity_blocks_powered_failure():
    bad_cov = dict(A_FAIL_POWERED, um1=0.3)
    c = cell(_res("qwen3-4b", [11], all=bad_cov), 11)
    assert not c["pf"]["W1"] and not c["valid"]
    c2 = cell(_res("qwen3-4b", [11], all=dict(B_FAIL_POWERED, status="at_grid_min_lower_bound")), 11)
    assert not c2["pf"]["W5"]
    c3 = cell(_res("qwen3-4b", [11], all=dict(W4=False, d4=1.0, d4ci=[-2.0, 4.0], acc=0.5)), 11)
    assert not c3["pf"]["W4"]                                      # two-hop accuracy < 0.60
    c4 = cell(_res("qwen3-4b", [11], w0a=False, all=A_FAIL_POWERED), 11)
    assert not c4["pf"]["W1"] and not c4["valid"]


def test_confirm_verdicts():
    X = dict(W3p=False, **B_FAIL_POWERED)
    cx = cell(_res("qwen3.5-2b", [7], all=X), 7)
    assert confirm_verdict("P2", "X", cx) and not confirm_verdict("P2", "Y", cx)
    c1 = cell(_res("qwen3-1.7b", [8]), 8)
    assert confirm_verdict("P1", "unified", c1)
    assert not confirm_verdict("P1", "unified", cell(_res("qwen3-1.7b", [8], w0b=False), 8))
