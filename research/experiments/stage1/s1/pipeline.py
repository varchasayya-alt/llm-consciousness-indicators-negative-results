"""Confirmatory per-seed pipeline.

GUARD: run_seed refuses to run unless the protocol is frozen (FROZEN_PROTOCOL.json present and its
config hash matches), except in test_mode (used only by the integration test with UNTRAINED stores,
whose outputs carry no information about the hypotheses).
Outputs per seed: meta.json, items.jsonl, natural.jsonl, prestates.npz (v3: pre-intervention read states of the
primary populations -- FORGET X on A and B, INTERF base-correct EV on A -- for the symmetric pre-state adjustment
of the within-target estimand; float16; SHA-256 recorded in meta.json).
"""
import copy
import json
import os
import time

import numpy as np
import torch

from . import world as W
from .config import config_hash, derive_seed, set_all_seeds
from .controller import lookup_prob, train_controller
from .diagnostics import cv_auroc, generic_features
from .interventions import (act_knockout_fn, act_sham_add_fn, familiarity_boost, fam_qc, forget_qc,
                            forget_with_sham, interference, interference_qc, learn_new, new_qc,
                            sample_dropout_rates)
from .matching import caliper_match, displacement, log_profile
from .monitors import in_features, int_features, make_monitor, out_features, predict, train_monitor
from .store import build_store, name_fluency, probe, site_stats, store_qc, train_store

STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_MARKER = os.path.join(STAGE_DIR, "FROZEN_PROTOCOL.json")
CONFIRMATORY_MONITORS = ["IN", "OUT", "INT-S", "INT-T", "INT-P", "INT-TP"]
EXTRA_MONITORS = ["INT-S-shuffled", "OUT-TP", "INT+OUT-S"]


def assert_frozen(cfg, test_mode):
    if test_mode:
        return
    if not os.path.exists(FROZEN_MARKER):
        raise RuntimeError("Protocol not frozen: refusing to run confirmatory seeds.")
    with open(FROZEN_MARKER) as f:
        marker = json.load(f)
    if marker.get("config_hash") != config_hash(cfg):
        raise RuntimeError("Config hash does not match the frozen protocol.")


def _ladder(run_fn, qc_fn, base_cfg, ladder):
    """Apply the pre-specified fallback ladder: lr multipliers, then step multipliers."""
    attempts = []
    for sm in ladder["step_multipliers"]:
        for lm in ladder["lr_multipliers"]:
            c = dict(base_cfg)
            c["lr"] = base_cfg["lr"] * lm
            for key in ("steps", "max_steps"):
                if key in c:
                    c[key] = int(base_cfg[key] * sm)
            model, extra = run_fn(c)
            qc, aux = qc_fn(model)
            attempts.append({"lr_mult": lm, "step_mult": sm, "qc": qc})
            if qc["pass"]:
                return model, extra, qc, aux, attempts
    return None, None, qc, aux, attempts


class Monitors:
    """Holds trained monitors for one store and builds their features."""

    def __init__(self, world, mu, sigma):
        self.world, self.mu, self.sigma = world, mu, sigma
        self.models = {}

    def features(self, cond, pr, items, names=None, align=None):
        if cond == "IN":
            return in_features(self.world, items, names)
        if cond.startswith("OUT"):
            return out_features(pr["probs"])
        f = int_features(pr["states"], self.mu, self.sigma)
        if align is not None:
            f = (f @ align).astype(np.float32)
        if cond.startswith("INT+OUT"):
            f = np.concatenate([f, out_features(pr["probs"])], 1)
        return f

    def score(self, cond, pr, items, names=None, align=None, model_key=None):
        m = self.models[model_key or cond]
        return predict(m, self.features(cond, pr, items, names, align))


def _pool_T(world, scfg, ckpts, items, vocab_size):
    feats, labels = [], []
    for frac in sorted(ckpts):
        m = build_store(scfg, vocab_size)
        m.load_state_dict(ckpts[frac])
        pr = probe(m, world, items)
        feats.append(pr)
    return feats


def train_monitor_set(world, store, ckpts, scfg, mcfg, items, mu, sigma, seed, which, align=None, log=print):
    """Train the requested monitor conditions on one store. Returns Monitors."""
    mons = Monitors(world, mu, sigma)
    pr_S = probe(store, world, items)
    y_S = pr_S["correct"]
    need_T = any(c in which for c in ("INT-T", "INT-TP", "OUT-TP"))
    need_P = any(c in which for c in ("INT-P", "INT-TP", "OUT-TP"))
    pr_T = _pool_T(world, scfg, ckpts, items, world.vocab.size) if need_T else []
    pr_P = []
    if need_P:
        rng = np.random.default_rng(derive_seed(seed, "Ppool"))
        g = torch.Generator().manual_seed(derive_seed(seed, "Pmask"))
        for _ in range(mcfg["P"]["draws_per_item"]):
            rates = sample_dropout_rates(len(items), store.n_layers, mcfg["P"], rng)
            pr_P.append(probe(store, world, items, dropout=rates, generator=g))

    def intf(prs):
        f = np.concatenate([int_features(p["states"], mu, sigma) for p in prs])
        return (f @ align).astype(np.float32) if align is not None else f

    def outf(prs):
        return np.concatenate([out_features(p["probs"]) for p in prs])

    def lab(prs):
        return np.concatenate([p["correct"] for p in prs])

    specs = {   # built lazily (one pool in memory at a time)
        "IN": ("IN", lambda: [(in_features(world, items), y_S, 1.0)], False),
        "OUT": ("OUT", lambda: [(out_features(pr_S["probs"]), y_S, 1.0)], False),
        "INT-S": ("INT", lambda: [(intf([pr_S]), y_S, 1.0)], False),
        "INT-S-shuffled": ("INT", lambda: [(intf([pr_S]), y_S, 1.0)], True),
        "INT-T": ("INT", lambda: [(intf(pr_T), lab(pr_T), 1.0)], False),
        "INT-P": ("INT", lambda: [(intf(pr_P), lab(pr_P), 1.0)], False),
        "INT-TP": ("INT", lambda: [(intf(pr_T), lab(pr_T), 0.5), (intf(pr_P), lab(pr_P), 0.5)], False),
        "OUT-TP": ("OUT", lambda: [(outf(pr_T), lab(pr_T), 0.5), (outf(pr_P), lab(pr_P), 0.5)], False),
        "INT+OUT-S": ("INT+OUT", lambda: [(np.concatenate([intf([pr_S]), out_features(pr_S["probs"])], 1), y_S, 1.0)], False),
    }
    for cond in which:
        kind, pool_fn, shuffled = specs[cond]
        pools = pool_fn()
        in_dim = None if kind == "IN" else pools[0][0].shape[1]
        model = make_monitor("IN" if kind == "IN" else ("OUT" if kind == "OUT" else "INT"), world, in_dim, mcfg)
        t0 = time.perf_counter()
        mons.models[cond] = train_monitor(model, pools, mcfg, derive_seed(seed, "monitor", cond), shuffle_labels=shuffled)
        log(f"    monitor {cond}: {time.perf_counter() - t0:.0f}s")
    return mons


def ridge_align(F_src, F_tgt, lam):
    """W = argmin ||F_src W - F_tgt||^2 + lam ||W||^2 (fit on INTACT, non-evaluation data only)."""
    A = F_src.T.astype(np.float64) @ F_src + lam * np.eye(F_src.shape[1])
    return np.linalg.solve(A, F_src.T.astype(np.float64) @ F_tgt).astype(np.float32)


def _rec(seed, store_id, interv, set_name, items, world, pre, post, disp, scores, lookups, extra=None):
    rows = []
    for k in range(len(items)):
        e, r = int(items[k, 0]), int(items[k, 1])
        row = {"seed": seed, "store": store_id, "intervention": interv, "set": set_name,
               "item": e * world.n_rel + r, "fam_high": bool(world.fam_high[e]),
               "pre_correct": bool(pre["correct"][k]), "post_correct": bool(post["correct"][k]),
               "pre_margin": float(pre["margin"][k]), "post_margin": float(post["margin"][k]),
               "relation": r, "pre_logp": float(pre["logp_correct"][k]), "post_logp": float(post["logp_correct"][k]),
               "post_maxprob": float(post["probs"][k].max()),
               "disp": None if disp is None else [round(float(x), 5) for x in disp[k]],
               "gen": None if "gen" not in post else [round(float(x), 5) for x in post["gen"][k]],
               "scores": {c: [float(v[0][k]), float(v[1][k])] for c, v in scores.items()},
               "lookup": {c: [float(v[0][k]), float(v[1][k])] for c, v in lookups.items()}}
        if extra:
            for key, arr in extra.items():
                row[key] = arr[k] if not hasattr(arr[k], "item") else arr[k].item()
        rows.append(row)
    return rows


def run_seed(cfg, seed, out_dir, test_mode=False, log=print):
    assert_frozen(cfg, test_mode)
    t_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    wcfg, scfg, mcfg, icfg, ccfg = cfg["world"], cfg["store"], cfg["monitors"], cfg["interventions"], cfg["controller"]
    world = W.make_world(wcfg, derive_seed(seed, "world"))
    epochs = cfg["test_mode_store_epochs"] if test_mode else scfg["epochs"]
    meta = {"seed": seed, "config_hash": config_hash(cfg), "test_mode": test_mode, "timings": {}}
    tm = lambda k, t0: meta["timings"].__setitem__(k, round(time.perf_counter() - t0, 1))

    t0 = time.perf_counter()
    A, ckA, _ = train_store(world, scfg, derive_seed(seed, "storeA"), epochs, scfg["checkpoint_fractions"])
    B, ckB, _ = train_store(world, scfg, derive_seed(seed + scfg["twin_seed_offset"], "storeB"), epochs,
                            scfg["checkpoint_fractions"])
    tm("stores", t0)
    meta["qcA"], meta["qcB"] = store_qc(A, world, scfg["qc"]), store_qc(B, world, scfg["qc"])
    if not test_mode and not (meta["qcA"]["pass"] and meta["qcB"]["pass"]):
        meta.update(excluded=True, reason="store QC")
        _write(out_dir, meta, [], [])
        return meta

    MTi, CTi, EVi = (W.items_where(world, split=s) for s in (W.MT, W.CT, W.EV))
    prA_MT, prB_MT = probe(A, world, MTi), probe(B, world, MTi)
    muA, sigA = site_stats(prA_MT["states"])
    muB, sigB = site_stats(prB_MT["states"])
    flu = name_fluency(A, world, world.names)
    flu_sd = float(flu.std())

    def ev_sets(store, key):
        pr = probe(store, world, EVi, want_states=False)
        trained = world.known[EVi[:, 0], EVi[:, 1]]
        if test_mode:            # untrained test stores: use knownness only (schema check, not science)
            bc, bi = EVi[trained], EVi[~trained]
        else:
            bc = EVi[trained & pr["correct"]]
            bi = EVi[~trained & ~pr["correct"]]
        return W.draw_item_sets(world, bc, bi, icfg["sets"], derive_seed(seed, key)), bc

    setsA, bcA = ev_sets(A, "setsA")
    setsB, bcB = ev_sets(B, "setsB")

    # ---------------- monitors
    t0 = time.perf_counter()
    which_A = CONFIRMATORY_MONITORS + EXTRA_MONITORS
    monA = train_monitor_set(world, A, ckA, scfg, mcfg, MTi, muA, sigA, derive_seed(seed, "monA"), which_A, log=log)
    monB = train_monitor_set(world, B, ckB, scfg, mcfg, MTi, muB, sigB, derive_seed(seed, "monB"), ["INT-S", "INT-TP"], log=log)
    FA = int_features(prA_MT["states"], muA, sigA)
    FB = int_features(prB_MT["states"], muB, sigB)
    Wal = ridge_align(FB, FA, cfg["alignment"]["ridge_lambda"])
    monBal = train_monitor_set(world, B, ckB, scfg, mcfg, MTi, muB, sigB, derive_seed(seed, "monBal"),
                               ["INT-S", "INT-TP"], align=Wal, log=log)
    meta["alignment_r2"] = float(1 - ((FB @ Wal - FA) ** 2).sum() / ((FA - FA.mean(0)) ** 2).sum())
    tm("monitors", t0)

    # ---------------- controllers (intact store, CT split)
    prA_CT = probe(A, world, CTi)
    ctrls = {}
    for cond in CONFIRMATORY_MONITORS:
        p = monA.score(cond, prA_CT, CTi)
        ctrls[cond] = train_controller(p, prA_CT["correct"], ccfg, derive_seed(seed, "ctrl", cond))
    in_emb = lambda items, names=None: monA.models["IN"].embed(torch.as_tensor(in_features(world, items, names))).detach().numpy()
    ctrls["INT-S+in"] = train_controller(monA.score("INT-S", prA_CT, CTi), prA_CT["correct"], ccfg,
                                         derive_seed(seed, "ctrl", "INT-S+in"), extra=in_emb(CTi))
    c_op = ccfg["operating_cost"]

    # ---------------- natural (intact EV)
    prA_EV = probe(A, world, EVi)
    nat_rows = []
    nat_scores = {c: monA.score(c, prA_EV, EVi) for c in which_A}
    trained_ev = world.known[EVi[:, 0], EVi[:, 1]]
    for k in range(len(EVi)):
        e, r = int(EVi[k, 0]), int(EVi[k, 1])
        nat_rows.append({"seed": seed, "item": e * world.n_rel + r, "fam_high": bool(world.fam_high[e]),
                         "trained": bool(trained_ev[k]), "correct": bool(prA_EV["correct"][k]),
                         "scores": {c: float(v[k]) for c, v in nat_scores.items()}})

    def evaluate(store_pre, store_post, items, monitors, conds, names_post=None, ctrl=True, ko_fn=None, add_fn=None):
        pre = probe(store_pre, world, items)
        post = probe(store_post, world, items, names=names_post, knockout_fn=ko_fn, add_fn=add_fn)
        sc, lk = {}, {}
        for cond in conds:
            s_pre = monitors.score(cond, pre, items)
            s_post = monitors.score(cond, post, items, names_post)
            sc[cond] = (s_pre, s_post)
            if ctrl and cond in ctrls:
                lk[cond] = (lookup_prob(ctrls[cond], s_pre, c_op), lookup_prob(ctrls[cond], s_post, c_op))
        if ctrl and "INT-S" in sc:
            lk["INT-S+in"] = (lookup_prob(ctrls["INT-S+in"], sc["INT-S"][0], c_op, in_emb(items)),
                              lookup_prob(ctrls["INT-S+in"], sc["INT-S"][1], c_op, in_emb(items, names_post)))
        disp = displacement(pre["states"], post["states"], monitors.sigma) if names_post is None else None
        if names_post is None:     # generic intervention features (diagnostics G2 cos/relnorm + G3), store states only
            g = generic_features(pre["states"], post["states"], monitors.mu, monitors.sigma)
            post["gen"] = np.concatenate([g["G2"][:, 10:], g["G3"]], 1)
        return pre, post, sc, lk, disp

    rows = []
    prestates = {}
    lad = icfg["fallback_ladder"]
    cal = icfg["caliper"]

    # ---------------- T-FORGET + sham (A)
    t0 = time.perf_counter()
    XA, YA, ZA = setsA["X"], setsA["Y"], setsA["Z"]
    fcfg = icfg["T_FORGET"]
    model_F, _, qcF, _, attF = _ladder(
        lambda c: forget_with_sham(A, world, XA, YA, c, sigA, derive_seed(seed, "forget")),
        lambda m: forget_qc(A, m, world, setsA, sigA, fcfg["qc"], [cal], derive_seed(seed, "matchF"), flu_sd, muA),
        fcfg, lad)
    meta["forget_attempts"] = attF
    if model_F is None and not test_mode:
        meta.update(excluded=True, reason="T-FORGET QC failed after ladder")
        _write(out_dir, meta, [], nat_rows)
        return meta
    if model_F is None:
        model_F = forget_with_sham(A, world, XA, YA, fcfg, sigA, derive_seed(seed, "forget"))[0]
    allF = np.concatenate([XA, YA, ZA])
    setlab = np.array(["X"] * len(XA) + ["Y"] * len(YA) + ["Z"] * len(ZA))
    pre, post, sc, lk, disp = evaluate(A, model_F, allF, monA, which_A)
    pair = _pair_ids(log_profile(disp), (setlab == "X") & ~post["correct"], (setlab == "Y") & post["correct"],
                     cal, derive_seed(seed, "pairF"))
    meta["sham_diagnostics_FORGET"] = sham_diagnostics(pre, post, setlab, pair, muA, sigA, seed)
    # twin level control: A's INT-S on aligned INTACT B states for the same items
    prB_F = probe(B, world, allF)
    twin = predict(monA.models["INT-S"], (int_features(prB_F["states"], muB, sigB) @ Wal).astype(np.float32))
    hX = np.where(setlab == "X", np.cumsum(setlab == "X") - 1, -1)
    prestates["FORGET_A"] = pre["states"][setlab == "X"].reshape(int((setlab == "X").sum()), -1).astype(np.float16)
    rows += _rec(seed, "A", "FORGET", None, allF, world, pre, post, disp, sc, lk,
                 extra={"set": setlab, "pair": pair, "fluency_pre": flu[allF[:, 0]], "twin_level_INT-S": twin,
                        "h_idx": hX})
    tm("forget", t0)

    # ---------------- T-INTERF (A)
    t0 = time.perf_counter()
    ic = icfg["T_INTERF"]
    model_I, _, qcI, _, attI = _ladder(
        lambda c: interference(A, world, c, derive_seed(seed, "interf")),
        lambda m: interference_qc(A, m, world, bcA, sigA, ic["qc"], [cal], derive_seed(seed, "matchI"), muA),
        ic, lad)
    meta["interf_attempts"] = attI
    if model_I is None and not test_mode:
        meta.update(excluded=True, reason="T-INTERF QC failed after ladder")
        _write(out_dir, meta, rows, nat_rows)
        return meta
    if model_I is None:
        model_I = interference(A, world, ic, derive_seed(seed, "interf"))[0]
    pre, post, sc, lk, disp = evaluate(A, model_I, bcA, monA, which_A)
    lost = ~post["correct"]
    pair = _pair_ids(log_profile(disp), lost, ~lost, cal, derive_seed(seed, "pairI"))
    prestates["INTERF_A"] = pre["states"].reshape(len(bcA), -1).astype(np.float16)
    rows += _rec(seed, "A", "INTERF", None, bcA, world, pre, post, disp, sc, lk,
                 extra={"set": np.where(lost, "lost", "retained"), "pair": pair, "fluency_pre": flu[bcA[:, 0]],
                        "h_idx": np.arange(len(bcA))})
    tm("interf", t0)

    # ---------------- T-NEW (A)
    t0 = time.perf_counter()
    nc = icfg["T_NEW"]
    NA, UA = setsA["N"], setsA["U"]
    model_N, _, qcN, _, attN = _ladder(lambda c: learn_new(A, world, NA, UA, c, derive_seed(seed, "new")),
                                       lambda m: (new_qc(A, m, world, NA, UA, bcA, nc["qc"]), None), nc, lad)
    meta["new_attempts"] = attN
    if model_N is not None or test_mode:
        model_N = model_N or learn_new(A, world, NA, UA, nc, derive_seed(seed, "new"))[0]
        itemsN = np.concatenate([NA, UA])
        pre, post, sc, lk, disp = evaluate(A, model_N, itemsN, monA, which_A)
        rows += _rec(seed, "A", "NEW", None, itemsN, world, pre, post, disp, sc, lk,
                     extra={"set": np.array(["N"] * len(NA) + ["U"] * len(UA))})
    tm("new", t0)

    # ---------------- T-FAM (A)
    t0 = time.perf_counter()
    fc = icfg["T_FAM"]
    FsetA, U2A = setsA["F"], setsA["U2"]
    model_M, _, qcM, _, attM = _ladder(lambda c: familiarity_boost(A, world, FsetA, c, derive_seed(seed, "fam"), U2A),
                                       lambda m: (fam_qc(A, m, world, FsetA, U2A, bcA, fc["qc"], flu_sd), None), fc, lad)
    meta["fam_attempts"] = attM
    if model_M is not None or test_mode:
        model_M = model_M or familiarity_boost(A, world, FsetA, fc, derive_seed(seed, "fam"), U2A)[0]
        itemsM = np.concatenate([FsetA, U2A])
        pre, post, sc, lk, disp = evaluate(A, model_M, itemsM, monA, which_A)
        rows += _rec(seed, "A", "FAM", None, itemsM, world, pre, post, disp, sc, lk,
                     extra={"set": np.array(["F"] * len(FsetA) + ["U2"] * len(U2A))})
    tm("fam", t0)

    # ---------------- input corruption (A, store intact)
    CA = setsA["C"]
    rng_c = np.random.default_rng(derive_seed(seed, "corrupt"))
    bad_names = world.unused_names[rng_c.permutation(len(world.unused_names))[:len(CA)]]
    pre, post, sc, lk, _ = evaluate(A, A, CA, monA, which_A, names_post=bad_names)
    rows += _rec(seed, "A", "CORRUPT", None, CA, world, pre, post, None, sc, lk, extra={"set": np.array(["C"] * len(CA))})

    # ---------------- exploratory: T-ACT and REPLACE (A)
    ka = icfg["T_ACT_exploratory"]
    ko = act_knockout_fn(ka["layers"], len(XA))
    prX = probe(A, world, XA)
    prXk = probe(A, world, XA, knockout_fn=ko)
    l0 = ka["layers"][0]
    norm = float(np.linalg.norm(prXk["states"][:, 2 * l0 + 1] - prX["states"][:, 2 * l0 + 1], axis=-1).mean())
    add = act_sham_add_fn(A, world, YA, l0, norm, derive_seed(seed, "actsham"))
    for sname, its, kw in (("X", XA, {"ko_fn": ko}), ("Y", YA, {"add_fn": add})):
        pre, post, sc, lk, disp = evaluate(A, A, its, monA, which_A, **kw)
        rows += _rec(seed, "A", "ACT", None, its, world, pre, post, disp, sc, lk, extra={"set": np.array([sname] * len(its))})
    model_R, _ = forget_with_sham(A, world, XA, YA, fcfg, sigA, derive_seed(seed, "replace"), mode="replace")
    itemsR = np.concatenate([XA, YA])
    pre, post, sc, lk, disp = evaluate(A, model_R, itemsR, monA, which_A)
    rows += _rec(seed, "A", "REPLACE", None, itemsR, world, pre, post, disp, sc, lk,
                 extra={"set": np.array(["X"] * len(XA) + ["Y"] * len(YA))})

    # ---------------- B side: T-FORGET on B; self-specificity
    t0 = time.perf_counter()
    XB, YB, ZB = setsB["X"], setsB["Y"], setsB["Z"]
    model_FB, _, qcFB, _, attFB = _ladder(
        lambda c: forget_with_sham(B, world, XB, YB, c, sigB, derive_seed(seed, "forgetB")),
        lambda m: forget_qc(B, m, world, setsB, sigB, fcfg["qc"], [cal], derive_seed(seed, "matchFB"),
                            float(name_fluency(B, world, world.names).std()), muB),
        fcfg, lad)
    meta["forgetB_attempts"] = attFB
    if model_FB is None and not test_mode:
        meta.update(excluded=True, reason="T-FORGET (store B) QC failed after ladder")
        _write(out_dir, meta, rows, nat_rows)
        return meta
    if model_FB is None:
        model_FB = forget_with_sham(B, world, XB, YB, fcfg, sigB, derive_seed(seed, "forgetB"))[0]
    allB = np.concatenate([XB, YB, ZB])
    labB = np.array(["X"] * len(XB) + ["Y"] * len(YB) + ["Z"] * len(ZB))
    preB, postB = probe(B, world, allB), probe(model_FB, world, allB)
    dispB = displacement(preB["states"], postB["states"], sigB)
    pairB = _pair_ids(log_profile(dispB), (labB == "X") & ~postB["correct"], (labB == "Y") & postB["correct"],
                      cal, derive_seed(seed, "pairFB"))
    scB = {}
    for cond in ("INT-S", "INT-TP"):
        fpre, fpost = int_features(preB["states"], muB, sigB), int_features(postB["states"], muB, sigB)
        scB[f"B->B:{cond}"] = (predict(monB.models[cond], fpre), predict(monB.models[cond], fpost))
        scB[f"Bal->Bal:{cond}"] = (predict(monBal.models[cond], fpre @ Wal), predict(monBal.models[cond], fpost @ Wal))
        scB[f"A->Bal:{cond}"] = (predict(monA.models[cond], (fpre @ Wal).astype(np.float32)),
                                 predict(monA.models[cond], (fpost @ Wal).astype(np.float32)))
        scB[f"A->Braw:{cond}"] = (predict(monA.models[cond], fpre), predict(monA.models[cond], fpost))
    hXB = np.where(labB == "X", np.cumsum(labB == "X") - 1, -1)
    prestates["FORGET_B"] = preB["states"][labB == "X"].reshape(int((labB == "X").sum()), -1).astype(np.float16)
    gB = generic_features(preB["states"], postB["states"], muB, sigB)
    genB = np.concatenate([gB["G2"][:, 10:], gB["G3"]], 1)
    fluB = name_fluency(B, world, world.names)
    postB["gen"] = genB
    rows += _rec(seed, "B", "FORGET", None, allB, world, preB, postB, dispB, scB, {},
                 extra={"set": labB, "pair": pairB, "h_idx": hXB, "fluency_pre": fluB[allB[:, 0]]})
    tm("storeB", t0)

    meta["qc"] = {"forget": qcF, "interf": qcI, "new": qcN, "fam": qcM, "forgetB": qcFB}
    meta["excluded"] = False
    meta["timings"]["total"] = round(time.perf_counter() - t_start, 1)
    _write(out_dir, meta, rows, nat_rows, prestates)
    return meta


def sham_diagnostics(pre, post, setlab, pair, mu, sigma, seed):
    """Generic-statistics distinguishability (G1-G4; s1/diagnostics.py) of X_lost vs matched Y, reported per
    seed (sham_algorithm_spec.md). Store states only -- never a monitor output; never used for exclusion."""
    g = generic_features(pre["states"], post["states"], mu, sigma)
    xm = (setlab == "X") & (pair >= 0)
    ym = (setlab == "Y") & (pair >= 0)
    xo = np.nonzero(xm)[0][np.argsort(pair[xm])]
    yo = np.nonzero(ym)[0][np.argsort(pair[ym])]
    out = {"n_pairs": int(xm.sum())}
    for gname in ("G1", "G2", "G3"):
        out[f"matched_{gname}"] = cv_auroc(g[gname][xo], g[gname][yo], seed=seed) if len(xo) >= 20 else None
    xr = (setlab == "X") & post["correct"]
    yc = (setlab == "Y") & post["correct"]
    out["n_X_retained"] = int(xr.sum())
    out["G4_Xretained_vs_Y"] = cv_auroc(g["G4"][xr], g["G4"][yc], seed=seed) if xr.sum() >= 10 else None
    return out


def _pair_ids(profiles, pos_mask, neg_mask, caliper, seed):
    """Frozen caliper matching -> pair id per row (-1 = unmatched)."""
    pid = -np.ones(len(profiles), dtype=int)
    pos_idx, neg_idx = np.nonzero(pos_mask)[0], np.nonzero(neg_mask)[0]
    for k, (i, j, _) in enumerate(caliper_match(profiles[pos_idx], profiles[neg_idx], caliper, seed)):
        pid[pos_idx[i]] = k
        pid[neg_idx[j]] = k
    return pid


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _write(out_dir, meta, rows, nat_rows, prestates=None):
    if prestates:
        import hashlib
        path = os.path.join(out_dir, "prestates.npz")
        np.savez_compressed(path, **prestates)
        meta["prestates_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=_jsonable)
    with open(os.path.join(out_dir, "items.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_jsonable) + "\n")
    with open(os.path.join(out_dir, "natural.jsonl"), "w") as f:
        for r in nat_rows:
            f.write(json.dumps(r, default=_jsonable) + "\n")
