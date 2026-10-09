"""Stage 1c orchestration (prereg §5): planted base model -> Stage-0 pipeline on it -> core arms -> pilot grid.
Outputs are planted-system pipeline outputs, NOT evidence about LMs or H-CD; they only parameterise S1a variances.
"""
from __future__ import annotations

import math
import os
import random
import time

import torch

from . import config as C
from . import planted as P
from . import stage0 as S0
from . import workspace as W
from .subject import EpSpec, SentSpec

PLANTED_S0_CFG = {**C.S0, "decod": {**C.S0["decod"], "layer_min": 2, "pca_dim": 64},
                  "oracle_layers": [0, 1, 2], "oracle_layer_margin_below_Lw": 1, "n_read_layers": 2}
TRAINED = ("succ", "parity", "mag", "lookup")
HO_PAIRS = (("add", "lookup"), ("sub", "mag"), ("mul", "succ"), ("mul", "parity"))
PRODS = ("add", "sub", "mul")


def episode(rng, x, producer, consumer, pool):
    args = P.random_args(producer, x, rng)
    if args is None:
        return None
    tbl = P.random_table(x, pool, rng) if consumer == "lookup" else None
    return EpSpec(SentSpec("latent", rng.choice(P.NAMES), (producer, args)), consumer, tbl, {"own": x})


def sampler_for(pairs, xs, seed):
    rng = random.Random(seed)

    def f(n):
        out = []
        while len(out) < n:
            p, c = rng.choice(pairs)
            ep = episode(rng, rng.choice(xs), p, c, xs)
            if ep is not None:
                out.append(ep)
        return out
    return f


def eval_set(pairs, xs, seed, per=4):
    rng = random.Random(seed)
    eps = []
    for p, c in pairs:
        for x in xs:
            for _ in range(per):
                ep = episode(rng, x, p, c, xs)
                if ep is not None:
                    eps.append(ep)
    return eps


def _base_competence(sub, xs, n=200, seed=5):
    """Copy (producer) and text-consumer greedy accuracy of the planted base model on its train values."""
    rng = random.Random(seed)
    out = {}
    eps = []
    for _ in range(n):
        ep = episode(rng, rng.choice(xs), rng.choice(PRODS), "copy", xs)
        if ep is not None:
            eps.append(ep)
    out["copy"] = sum(r["own"] for r in sub.evaluate(eps)) / len(eps)
    for c in ("succ", "plus10", "parity", "mag", "lookup", "verb"):
        tx = []
        for _ in range(n // 2):
            x = rng.choice(xs)
            tbl = P.random_table(x, xs, rng) if c == "lookup" else None
            tx.append(EpSpec(SentSpec("text", rng.choice(P.NAMES), (x,)), c, tbl, {"own": x}))
        out[c] = sum(r["own"] for r in sub.evaluate(tx)) / len(tx)
    return out


def run_s1c(log=print, base_steps=4000, arm_steps=1500, pilot_steps=1000, out_dir=None):
    t0 = time.time()
    seeds = C.SEEDS
    R = {}
    log("S1c: training planted base model")
    model = P.train_base(seeds["planted_base"], steps=base_steps, log=log)
    sub = P.PlantedSubject(model)
    xtr0, _ = P.planted_split(seeds["planted_base"])
    comp = _base_competence(sub, xtr0)
    R["base_competence_4000"] = comp
    if min(comp.values()) < 0.80:          # pre-declared single extension (prereg S1c)
        log(f"S1c: base competence {comp} < 0.80 -> extending base training to {2 * base_steps} steps (once)")
        model = P.train_base(seeds["planted_base"], steps=2 * base_steps, log=log)
        sub = P.PlantedSubject(model)
        R["base_competence_extended"] = _base_competence(sub, xtr0)
    if out_dir:
        path = os.path.join(out_dir, "planted_base.pt")
        torch.save(model.state_dict(), path)
        R["planted_base_sha256"] = C.sha256_file(path)
    xtr, xev = P.planted_split(seeds["planted_base"])
    rng_h = random.Random(seeds["halves"])
    hx = xtr[:]
    rng_h.shuffle(hx)
    halves = (sorted(hx[:len(hx) // 2]), sorted(hx[len(hx) // 2:]))
    main = P.planted_instances(xtr, C.MAT["instances_main"], seeds["planted_base"] + 1, tag="plm")
    dec = P.planted_instances(xtr, C.MAT["instances_decod"], seeds["planted_base"] + 2, tag="pld")
    log("S1c: Stage-0 pipeline on the planted model")
    s0 = S0.run_stage0(sub, main, dec, halves, log=log, cfg=PLANTED_S0_CFG, force_continue=True)
    R["stage0_planted"] = s0
    R["S1c_1_pass"] = bool(s0.get("core_instrument_valid"))
    if "M3" not in s0:
        R["stopped"] = "planted Stage-0 stopped before M3"
        return R
    L_w, read_layers = s0["M1"]["L_w"], s0["M3"]["read_layers"]
    rx = s0["M4"]["rX_geomean"] or sub.d_model
    cap = int(min(sub.d_model, max(1, round(2 * rx))))
    R["arm_config"] = {"L_w": L_w, "read_layers": read_layers, "r_X": rx, "capacity_A": cap}
    e_bar = torch.stack([sub.cue_rep(c, L_w, 0) for c in TRAINED]).mean(0)

    tr_pairs = [(p, c) for p in PRODS for c in TRAINED if (p, c) not in HO_PAIRS]
    u_pairs = [(p, "verb") for p in PRODS]
    ev_tr = eval_set(tr_pairs, xev, 31)
    ev_ho = eval_set(list(HO_PAIRS), xev, 32)
    ev_u = eval_set(u_pairs, xev, 33)

    # zero-gate identity at init
    arm0 = W.PlantedArm(sub, "A", cap, read_layers, L_w, L_w, 0, e_bar=e_bar)
    probe = ev_tr[:64]
    base = sub.evaluate([EpSpec(e.sent, e.consumer, e.table, {"own": e.cands["own"]}) for e in probe])
    with torch.no_grad():
        S = arm0.slots(probe)
    init = arm0.run([EpSpec(e.sent, e.consumer, e.table, {"own": e.cands["own"]}) for e in probe], S)
    R["zero_gate_identity"] = all(a["own"] == b["own"] for a, b in zip(base, init))
    nat = lambda eps: sum(r["own"] for r in sub.evaluate(eps)) / len(eps)

    R["native"] = {"TR": nat(ev_tr), "HO": nat(ev_ho), "U": nat(ev_u)}
    core = {}
    for kind in ("A", "C1", "C3"):
        core[kind] = []
        for sd in seeds["planted_core"]:
            log(f"S1c core arm {kind} seed {sd}")
            arm = W.PlantedArm(sub, kind, cap, read_layers, L_w, L_w, sd, e_bar=e_bar, r_v=sub.d_model)
            arm.train(sampler_for(tr_pairs, xtr, sd), steps=arm_steps, log=log)
            rec = {"seed": sd, "n_params": W.n_params(arm.ws),
                   "TR": W.evaluate_arm(arm, ev_tr, xev, sd + 1), "HO": W.evaluate_arm(arm, ev_ho, xev, sd + 2),
                   "U": W.evaluate_arm(arm, ev_u, xev, sd + 3)}
            if kind == "A":
                inst = ev_tr[0]
                rec["cue_invariance_maxabs"] = W.cue_invariance(arm, inst.sent, list(P.OPS),
                                                                P.random_table(inst.cands["own"], xev, random.Random(1)))
            if kind == "C1":
                rec["TR_syn1"] = W.evaluate_arm(arm, ev_tr, xev, sd + 1, syn=1)
            core[kind].append(rec)
            log(f"   TR acc={rec['TR']['acc']:.3f} CT={rec['TR']['CT']} | HO CT={rec['HO']['CT']} | U CT={rec['U']['CT']}")
    R["core"] = core
    mean = lambda v: sum(v) / len(v) if v else None
    A = core["A"]
    reach = mean([r["TR"]["acc"] for r in A]) - R["native"]["TR"]
    ct_tr = mean([r["TR"]["CT"] for r in A if r["TR"]["CT"] is not None])
    g = C.S0["gates"]
    nc_ok = all(r[e]["P_sw_nc"] is not None and abs(r[e]["P_sw_nc"] - r[e]["P_sw_nat"]) <= g["V2c_fprime_shift_max"]
                and abs(r[e]["acc_nc"] - r[e]["acc_nat"]) <= g["V2c_acc_shift_max"]
                for r in A for e in ("TR", "HO", "U"))
    v6 = (mean([r["TR_syn1"]["CT"] for r in core["C1"] if r["TR_syn1"]["CT"] is not None]) /
          max(1e-9, mean([r["TR"]["CT"] for r in core["C1"] if r["TR"]["CT"] is not None])))
    R["S1c_2"] = {"zero_gate_identity": R["zero_gate_identity"],
                  "cue_invariance": all(r["cue_invariance_maxabs"] == 0.0 for r in A),
                  "train_reach": reach, "train_reach_pass": reach >= C.S1["S1c"]["train_reach_min"],
                  "transport_pc_CT_TR": ct_tr, "transport_pc_pass": (ct_tr or 0) >= C.S1["S1c"]["transport_pc_ct_min"],
                  "V6_ratio": v6, "V6_pass": v6 >= C.S1["S1c"]["V6_paraphrase_ratio_min"],
                  "nc_neutral": nc_ok}
    R["S1c_2"]["pass"] = all(R["S1c_2"][k] for k in ("zero_gate_identity", "cue_invariance", "train_reach_pass",
                                                     "transport_pc_pass", "V6_pass", "nc_neutral"))
    log(f"S1c-2: {R['S1c_2']}")

    # ------------------------------------------------------------ pilot grid (A only)
    sets = C.S1["S1c"]["pilot_sets"]
    pilot = {}
    for cx in C.S1["S1c"]["pilot_capacities_xr"]:
        dw = int(min(sub.d_model, max(1, round(cx * rx))))
        for sname, cons in sets.items():
            pairs = [(p, c) for p in PRODS for c in cons]
            key = f"c{cx}_{sname}"
            pilot[key] = []
            for sd in seeds["planted_pilot"]:
                log(f"S1c pilot {key} (d_w={dw}) seed {sd}")
                eb = torch.stack([sub.cue_rep(c, L_w, 0) for c in cons]).mean(0)
                arm = W.PlantedArm(sub, "A", dw, read_layers, L_w, L_w, sd, e_bar=eb)
                arm.train(sampler_for(pairs, xtr, sd), steps=pilot_steps, log=lambda *a: None)
                u = W.evaluate_arm(arm, ev_u, xev, sd + 3)
                tr = W.evaluate_arm(arm, eval_set(pairs, xev, 34, per=2), xev, sd + 4)
                pilot[key].append({"seed": sd, "d_w": dw, "U": u, "TR": tr})
                log(f"   U CT={u['CT']} acc={u['acc']:.3f} | TR CT={tr['CT']}")
    R["pilot"] = pilot

    # seed-level variability (logit scale) for S1a
    def lg(p):
        p = min(max(p, 0.02), 0.98)
        return math.log(p / (1 - p))
    sds = []
    for key, recs in list(pilot.items()) + [("coreA_HO", [{"U": r["HO"]} for r in A]),
                                             ("coreA_U", [{"U": r["U"]} for r in A])]:
        vals = [r["U"]["P_sw_swap"] for r in recs if r["U"]["P_sw_swap"] is not None]
        if len(vals) >= 2 and 0.05 < sum(vals) / len(vals) < 0.95:
            l = [lg(v) for v in vals]
            mu = sum(l) / len(l)
            sds.append(math.sqrt(sum((v - mu) ** 2 for v in l) / (len(l) - 1)))
    R["sigma_seed_logit"] = {"values": sds, "median": sorted(sds)[len(sds) // 2] if sds else None}
    R["elapsed_s"] = time.time() - t0
    return R
