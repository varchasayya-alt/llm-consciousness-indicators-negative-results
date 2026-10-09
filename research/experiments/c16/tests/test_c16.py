"""C16 S0/S1 unit, integrity and smoke tests (no Qwen weights are loaded; the tokenizer is)."""
import os
import random
import sys

import pytest
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c16 import config as C          # noqa: E402
from c16 import materials as M       # noqa: E402
from c16 import planted as P         # noqa: E402
from c16 import ranks as RK          # noqa: E402
from c16 import sim                  # noqa: E402
from c16 import stage0               # noqa: E402
from c16 import workspace as W       # noqa: E402
from c16.subject import EpSpec, SentSpec, latent, text, bundle   # noqa: E402


# ------------------------------------------------------------------ config / materials
def test_thresholds_loaded():
    assert C.MODEL_NAME == "Qwen/Qwen2.5-0.5B-Instruct"
    assert C.G["V2b_ct_min"] == 0.30 and C.G["W3_window_ratio_min"] == 4.0
    assert C.U == "verb" and set(C.POOL) == {"succ", "plus10", "parity", "mag", "lookup"}
    assert "S2" not in C.PHASES_AUTHORIZED


def test_split_disjoint_complete_deterministic():
    a = M.split_values()
    b = M.split_values()
    assert a == b
    tr, se, co = a
    assert (len(tr), len(se), len(co)) == (45, 22, 22)
    assert not (set(tr) & set(se)) and not (set(tr) & set(co)) and not (set(se) & set(co))
    assert set(tr) | set(se) | set(co) == set(M.x_values())
    assert 50 not in M.x_values()


def test_sealed_values_refused():
    _, se, _ = M.split_values()
    with pytest.raises(M.SealedError):
        M.build_instances(se[:3], 1, 0)


def test_words_and_answers():
    assert M.words(37) == "thirty-seven" and M.words(40) == "forty" and M.words(13) == "thirteen"
    assert M.answer("succ", 99, None) == "100" and M.answer("mag", 51, None) == "larger"
    tbl = [(37, "red"), (12, "blue"), (80, "green")]
    assert M.answer("lookup", 12, tbl) == "blue"


def test_instances_valid():
    tr, _, _ = M.split_values()
    inst = M.build_instances(tr, 4, C.SEEDS["s0_sampling"])
    for i in inst:
        if i.producer == "add":
            assert sum(i.args) == i.x and i.args[0] >= 2 and 2 <= i.args[1] <= 9
        elif i.producer == "sub":
            assert i.args[0] - i.args[1] == i.x
        else:
            assert i.args[0] * i.args[1] == i.x
        assert i.x in [v for v, _ in i.table] and len(i.table) == 3
        assert all(v in tr for v, _ in i.table)
        assert i.name in C.MAT["names_train"]


def test_cue_strings_have_no_name_or_number():
    for c in C.ALL_CONSUMERS:
        s = M.cue_string(c)
        assert not any(ch.isdigit() for ch in s.replace("50", ""))
        assert all(n not in s for n in C.MAT["names_train"])


# ------------------------------------------------------------------ tokenizer-level (Qwen tokenizer only)
@pytest.fixture(scope="module")
def tok_subject():
    os.environ.setdefault("HF_HOME", C.HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    from c16.lm_hf import HFSubject
    s = object.__new__(HFSubject)
    s.tok = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    s.fmt = "F1"
    return s


def test_names_single_token(tok_subject):
    t = tok_subject.tok
    for n in C.MAT["names_train"] + C.MAT["names_eval_sealed"] + [C.MAT["primer_name"]]:
        assert len(t(n, add_special_tokens=False)["input_ids"]) == 1
        assert len(t(" " + n, add_special_tokens=False)["input_ids"]) == 1


@pytest.mark.parametrize("fmt", ["F1", "F2"])
def test_prefix_identical_across_paired_runs(tok_subject, fmt):
    tok_subject.fmt = fmt
    tr, _, _ = M.split_values()
    inst = M.build_instances(tr, 1, 5)[:40]
    for i in inst:
        for j in C.ALL_CONSUMERS:
            tbl = tuple(i.table) if j == "lookup" else None
            ans = M.answer(j, i.x, tbl)
            encs = [tok_subject.encode(s, j, tbl, ans) for s in
                    (latent(i.name, i.producer, i.args), text(i.name, i.x), bundle(i.name, "B4", i.x),
                     bundle(i.name, "Bpm", i.x))]
            pre = [[e["ids"][k] for k in e["prefix_pos"]] for e in encs]
            tail = [e["ids"][e["prefix_pos"][0]:] for e in encs]
            assert all(p == pre[0] for p in pre) and len(pre[0]) == 7
            assert all(t == tail[0] for t in tail)           # prefix + question + answer identical
            assert encs[0]["ans_pos"] and encs[0]["prod_pos"]
            assert max(encs[0]["prod_pos"]) < encs[0]["prefix_pos"][0]
    tok_subject.fmt = "F1"


def test_answer_boundaries_all_values(tok_subject):
    tr, _, _ = M.split_values()
    s = latent("Ana", "add", (30, 7))
    for x in tr:
        for j in ("copy", "succ", "plus10", "verb", "parity", "mag"):
            e = tok_subject.encode(s, j, None, M.answer(j, x, None))
            assert tok_subject.tok.decode(e["ans_ids"]) == " " + M.answer(j, x, None)


# ------------------------------------------------------------------ ranks / S1b
def test_rank_projection_full_and_zero():
    g = torch.Generator().manual_seed(0)
    D = torch.randn(30, 7, 16, generator=g)
    b = RK.fit_basis(D, max_rank=8)
    assert torch.equal(RK.project(D, b, "full"), D)
    P0 = RK.project(D, b, 8)
    assert P0.shape == D.shape
    assert RK.rank_star({1: 0.0, 2: 0.5, "full": 0.5}, [1, 2, "full"], 0.9, 0.1) == 2
    assert RK.rank_star({1: 0.0, "full": 0.05}, [1, "full"], 0.9, 0.1) is None


def test_s1b_recovers_ranks():
    r = sim.s1b_rank_recovery(123)
    assert r["pass"], r


# ------------------------------------------------------------------ planted model and workspace
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    m = P.TinyGPT(len(P.VOCAB), d=32, layers=4, heads=4)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return P.PlantedSubject(m)


def test_planted_alignment():
    for s in (SentSpec("latent", "n1", ("add", (30, 7))), SentSpec("latent", "n1", ("mul", (6, 7))),
              SentSpec("text", "n1", (37,)), SentSpec("bundle", "n1", ("B4", 37)), SentSpec("bundle", "n1", ("Bpm", 37))):
        toks, ppos, spos, _ = P.episode_tokens(EpSpec(s, "parity", None, {}), None)
        assert len(P.sentence_tokens(s)) == P.SL and ppos == list(range(1 + P.SL, 1 + P.SL + 5))
        assert all(t in P.VOCAB for t in toks)


def test_planted_zero_patch_identity(tiny):
    eps = [EpSpec(latent("n1", "add", (30, 7)), "succ", None, {"own": 37}),
           EpSpec(latent("n2", "sub", (40, 3)), "parity", None, {"own": 37})]
    a = tiny.evaluate(eps)
    b = tiny.evaluate(eps, layer=1, deltas=[torch.zeros(5, 32), torch.zeros(5, 32)])
    assert a == b


def test_workspace_zero_gate_identity_and_cue_invariance(tiny):
    e_bar = torch.zeros(32)
    arm = W.PlantedArm(tiny, "A", 8, (1, 2), 3, 3, 0, e_bar=e_bar)
    eps = [EpSpec(latent("n1", "add", (30, 7)), c, P.random_table(37, list(range(10, 99)), random.Random(0))
                  if c == "lookup" else None, {"own": 37}) for c in ("succ", "parity", "lookup", "verb")]
    with torch.no_grad():
        S = arm.slots(eps)
    assert arm.run(eps, S) == tiny.evaluate(eps)
    assert W.cue_invariance(arm, eps[0].sent, list(P.OPS), eps[2].table) == 0.0
    c1 = W.PlantedArm(tiny, "C1", 8, (1, 2), 3, 3, 0, e_bar=e_bar)
    assert W.n_params(c1.ws) == W.n_params(arm.ws)
    with torch.no_grad():
        for p in c1.ws.W_qc.parameters():
            p.normal_()
        S1 = c1.slots(eps, noisy=False)
    assert float((S1 - S1[0:1]).abs().max()) > 0          # C1 sees the cue


def test_workspace_refuses_hf_subject():
    class Fake:
        d_model = 8
    with pytest.raises(TypeError):
        W.PlantedArm(Fake(), "A", 4, (1,), 2, 2, 0, e_bar=torch.zeros(8))


def test_stage0_pipeline_smoke(tiny):
    xs = list(range(10, 40))
    main = P.planted_instances(xs, 1, 1)
    dec = P.planted_instances(xs, 2, 2)
    cfg = {**C.S0, "decod": {**C.S0["decod"], "layer_min": 2, "pca_dim": 16, "folds": 3},
           "oracle_layers": [0, 1, 2], "oracle_layer_margin_below_Lw": 1, "n_read_layers": 2,
           "items_per_half_per_consumer": 6, "ranks": [1, 4, "full"]}
    halves = (xs[:15], xs[15:])
    # an untrained model fails the gates; force the pipeline through M3/M4 to exercise the code path
    R = stage0.run_stage0(tiny, main, dec, halves, log=lambda *a: None, cfg=cfg, force_continue=True)
    for k in ("M1", "M2", "M3", "M4", "V2", "W"):
        assert k in R
    assert R["core_instrument_valid"] in (True, False)


# ------------------------------------------------------------------ S1a simulator
def test_s1a_core_runs_and_null_is_w0():
    r = sim.s1a_core(1, 0.5, 0.3, reps=20, B=200)
    assert r["worlds"]["null"]["dist"].get("W0", 0) >= 0.9


def test_s1a_hcd_runs():
    r = sim.s1a_hcd(2, 0.5, 0.3, {"S1": 1, "S2": 2, "S4": 4, "Spm": 0.5, "Sparam": float("inf")}, 3, 100,
                    reps=10, B=100)
    assert set(r["worlds"]) == {"geometry", "ib", "copy", "null"}


# ------------------------------------------------------------------ runner guard
def test_runner_refuses_unauthorized_phase():
    import run_c16
    with pytest.raises(run_c16.GuardError):
        run_c16.verify("S2", allow_dirty=True)
