"""Stage 0 measurement pipeline (prereg §4), generic over a subject (HF Qwen or planted).

The subject provides n_layers, d_model, prefix_hidden, prod_hidden, evaluate (see subject.py).
`instances` and `split` are supplied by the caller (numbers domain for both subjects).
Returns a JSON-serialisable dict with all measurements, choices (L_w, l_o, R) and gate verdicts.
"""
from __future__ import annotations

import math
import random
import time
import zlib

import torch

from . import config as C
from . import decod
from . import ranks as RK
from .subject import EpSpec, latent, text, bundle

SIX = ("succ", "plus10", "parity", "mag", "lookup", "verb")


def _acc(xs):
    return float(sum(xs) / len(xs)) if xs else None


def run_stage0(subject, inst_main, inst_decod, halves, log=print, consumers_all=None, cfg=None,
               force_continue=False):
    cfg = cfg or C.S0
    gates = cfg["gates"]
    t_start = time.time()
    hA, hB = set(halves[0]), set(halves[1])
    consumers_all = consumers_all or C.ALL_CONSUMERS
    R = {"subject": type(subject).__name__}
    nL = subject.n_layers

    # ---------------------------------------------------------------- M1 decodability (V0)
    log("M1 decodability")
    layers_all = list(range(nL))
    sents = [latent(i.name, i.producer, i.args) for i in inst_decod]
    H = subject.prod_hidden(sents, layers_all, last=cfg["decod"]["positions_last"])  # [n, L, 3, d]
    dec = {}
    for p in C.PRODUCERS:
        idx = [k for k, i in enumerate(inst_decod) if i.producer == p]
        if len(idx) < 20:
            continue
        xs = [inst_decod[k].x for k in idx]
        folds = decod.folds_by_x(xs, cfg["decod"]["folds"])
        dec[p] = []
        for l in layers_all:
            best = 0.0
            for pos in range(H.shape[2]):
                a = decod.cv_digit_accuracy(H[idx, l, pos], xs, folds, cfg["decod"]["ridge_lambda"],
                                            cfg["decod"]["pca_dim"])
                best = max(best, a)
            dec[p].append(best)
    lmin = cfg["decod"]["layer_min"]
    cand = [l for l in range(lmin, nL)]
    L_w = max(cand, key=lambda l: (dec["add"][l] + dec["sub"][l]) / 2)
    R["M1"] = {"by_producer_layer": dec, "L_w": L_w}
    R["V0"] = {p: dec[p][L_w] for p in dec}
    v0 = all(dec[p][L_w] >= gates["V0_decod_min"] for p in ("add", "sub"))
    log(f"  L_w={L_w} add={dec['add'][L_w]:.3f} sub={dec['sub'][L_w]:.3f} V0={v0}")

    # ---------------------------------------------------------------- M2 competence and gap (V1)
    log("M2 competence")
    nat_eps, txt_eps, keys = [], [], []
    for i in inst_main:
        for j in consumers_all:
            tbl = tuple(i.table) if j == "lookup" else None
            nat_eps.append(EpSpec(latent(i.name, i.producer, i.args), j, tbl, {"own": i.x}))
            txt_eps.append(EpSpec(text(i.name, i.x), j, tbl, {"own": i.x}))
            keys.append((i.iid, j))
    nat = subject.evaluate(nat_eps)
    txt = subject.evaluate(txt_eps)
    native = {k: r["own"] for k, r in zip(keys, nat)}
    textok = {k: r["own"] for k, r in zip(keys, txt)}
    copy_ok = {i.iid: native[(i.iid, "copy")] for i in inst_main}
    elig = {(i.iid, j): copy_ok[i.iid] and textok[(i.iid, j)] for i in inst_main for j in consumers_all}
    m2 = {"copy_by_producer": {}, "text_by_consumer": {}, "cot_by_producer_consumer": {}, "native_eligible": {},
          "kappa": {}, "eligible_rate": {}}
    for p in C.PRODUCERS:
        ii = [i for i in inst_main if i.producer == p]
        if ii:
            m2["copy_by_producer"][p] = _acc([copy_ok[i.iid] for i in ii])
            m2["cot_by_producer_consumer"][p] = {j: _acc([elig[(i.iid, j)] for i in ii]) for j in consumers_all}
    addsub = [i for i in inst_main if i.producer in ("add", "sub")]
    chance = C.CONS["chance"]
    for j in consumers_all:
        m2["text_by_consumer"][j] = _acc([textok[(i.iid, j)] for i in addsub])
        el = [i for i in addsub if elig[(i.iid, j)]]
        m2["eligible_rate"][j] = _acc([elig[(i.iid, j)] for i in addsub])
        a = _acc([native[(i.iid, j)] for i in el])
        m2["native_eligible"][j] = a
        m2["kappa"][j] = None if a is None else (a - chance[j]) / (1 - chance[j])
    R["M2"] = m2
    v1a = all(m2["copy_by_producer"][p] >= gates["V1a_copy_min"] for p in ("add", "sub"))
    v1b_pass = [j for j in C.POOL if m2["text_by_consumer"][j] >= gates["V1b_text_min"]]
    v1b = len(v1b_pass) >= gates["V1b_min_pool_pass"] and m2["text_by_consumer"][C.U] >= gates["V1b_text_min"]
    kap_ok = lambda j: m2["kappa"][j] is not None and m2["kappa"][j] <= gates["V1c_kappa_max"]
    eli_ok = lambda j: m2["eligible_rate"][j] >= gates["V1d_eligible_min"]
    v1_pass = [j for j in v1b_pass if kap_ok(j) and eli_ok(j)]
    v1c = len([j for j in v1b_pass if kap_ok(j)]) >= gates["V1c_min_pool_pass"] and kap_ok(C.U)
    v1d = all(eli_ok(j) for j in [jj for jj in v1b_pass if kap_ok(jj)] + [C.U])
    R["V1"] = {"V1a": v1a, "V1b": v1b, "V1c": v1c, "V1d": v1d, "pool_passing_V1": v1_pass}
    log(f"  V1a={v1a} V1b={v1b} V1c={v1c} V1d={v1d} pool_pass={v1_pass}")
    R["format_fallback_needed"] = not (v0 and v1a and v1b and v1d)
    if R["format_fallback_needed"] and not force_continue:
        R["stopped_at"] = "M2"
        R["elapsed_s"] = time.time() - t_start
        return R

    # ---------------------------------------------------------------- deltas (content) at oracle layers
    log("M3 layer sweep")
    olayers = [l for l in cfg["oracle_layers"] if l <= L_w - cfg["oracle_layer_margin_below_Lw"] and l < nL]
    lat_s = [latent(i.name, i.producer, i.args) for i in addsub]
    txt_s = [text(i.name, i.x) for i in addsub]
    h_lat = subject.prefix_hidden(lat_s, olayers)
    h_txt = subject.prefix_hidden(txt_s, olayers)
    D = h_txt - h_lat                                       # [n, Lo, P, d]
    pos_of = {i.iid: k for k, i in enumerate(addsub)}
    rng = random.Random(C.SEEDS["swap_nc"])
    cap = cfg["items_per_half_per_consumer"]

    def items(j, half):
        el = [i for i in addsub if elig[(i.iid, j)] and i.x in half]
        r = random.Random(zlib.crc32(f"{j}|{sorted(half)}|{C.SEEDS['s0_sampling']}".encode()))
        el = sorted(el, key=lambda i: i.iid)
        r.shuffle(el)
        return el[:cap]

    def gain_eval(j, its, layer, deltas):
        eps = [EpSpec(latent(i.name, i.producer, i.args), j, tuple(i.table) if j == "lookup" else None,
                      {"own": i.x}) for i in its]
        res = subject.evaluate(eps, layer=layer, deltas=deltas)
        acc = _acc([r["own"] for r in res])
        base = _acc([native[(i.iid, j)] for i in its])
        return acc, acc - base

    sweep = {}
    for li, l in enumerate(olayers):
        sweep[l] = {}
        for j in SIX:
            its = items(j, hA)
            if not its:
                continue
            _, g = gain_eval(j, its, l, [D[pos_of[i.iid], li] for i in its])
            sweep[l][j] = g
        log(f"  layer {l}: " + " ".join(f"{j}={sweep[l][j]:+.2f}" for j in sweep[l]))
    meang = {l: sum(sweep[l].values()) / max(1, len(sweep[l])) for l in olayers}
    l_o = max(olayers, key=lambda l: meang[l])
    read_layers = sorted(sorted(olayers, key=lambda l: -meang[l])[:cfg["n_read_layers"]])
    R["M3"] = {"oracle_layers": olayers, "gain": sweep, "mean_gain": meang, "l_o": l_o, "read_layers": read_layers}
    lo_idx = olayers.index(l_o)
    D_lo = D[:, lo_idx]                                     # [n, P, d]
    h_lat_lo = h_lat[:, lo_idx]

    # ---------------------------------------------------------------- M4 ranks (content), cross-fitted
    log("M4 content ranks")
    grid = cfg["ranks"]
    halves_named = {"A": hA, "B": hB}

    def fit_on(half):
        idx = [pos_of[i.iid] for i in addsub if i.x in half]
        return RK.fit_basis(D_lo[idx], max_rank=max(r for r in grid if r != "full"))

    def rank_sweep(Dsrc, consumer_list, label):
        out = {}
        for ev, fit in (("A", "B"), ("B", "A")):
            basis = fit_on_src(Dsrc, halves_named[fit])
            out[ev] = {}
            for j in consumer_list:
                its = items(j, halves_named[ev])
                if not its:
                    continue
                gains, accs = {}, {}
                for r in grid:
                    dl = [RK.project(Dsrc[pos_of[i.iid]], basis, r) for i in its]
                    a, g = gain_eval(j, its, l_o, dl)
                    gains[r], accs[r] = g, a
                out[ev][j] = {"gain": {str(k): v for k, v in gains.items()}, "n": len(its),
                              "r_star": RK.rank_star(gains, grid, cfg["rank_fraction"], cfg["rank_min_full_gain"])}
            log(f"  {label} eval={ev}: " + " ".join(f"{j}:r*={out[ev][j]['r_star']},gf={out[ev][j]['gain']['full']:+.2f}"
                                                  for j in out[ev]))
        return out

    def fit_on_src(Dsrc, half):
        idx = [pos_of[i.iid] for i in addsub if i.x in half]
        return RK.fit_basis(Dsrc[idx], max_rank=max(r for r in grid if r != "full"))

    content = rank_sweep(D_lo, SIX, "content")

    def full_gain(sw, j):
        num = den = 0.0
        for ev in ("A", "B"):
            if j in sw[ev]:
                num += sw[ev][j]["gain"]["full"] * sw[ev][j]["n"]
                den += sw[ev][j]["n"]
        return num / den if den else None

    v2a_gain = {j: full_gain(content, j) for j in SIX}

    # ---------------------------------------------------------------- M4 swap / NC (full delta at l_o)
    log("M4 swap and NC")
    gen = torch.Generator().manual_seed(C.SEEDS["swap_nc"])
    swap = {}
    for j in SIX:
        recs = []
        for half_name, half in halves_named.items():
            other = halves_named["B" if half_name == "A" else "A"]
            mu_other = fit_on(other)["mu"]
            its = items(j, half)
            if not its:
                continue
            xprime = []
            for i in its:
                if j == "lookup":
                    xprime.append(rng.choice([v for v, _ in i.table if v != i.x]))
                else:
                    cands = [v for v in half if v != i.x and (v % 2) != (i.x % 2) and ((v > 50) != (i.x > 50))]
                    xprime.append(rng.choice(sorted(cands)))
            src_s = [text(i.name, xp) for i, xp in zip(its, xprime)]
            h_src = subject.prefix_hidden(src_s, [l_o])[:, 0]
            d_swap = [h_src[k] - h_lat_lo[pos_of[i.iid]] for k, i in enumerate(its)]
            d_nc = [RK.nc_delta(D_lo[pos_of[i.iid]], mu_other, gen) for i in its]
            tb = lambda i: tuple(i.table) if j == "lookup" else None
            src_ok = subject.evaluate([EpSpec(s, j, tb(i), {"sw": xp}) for s, i, xp in zip(src_s, its, xprime)])
            eps = [EpSpec(latent(i.name, i.producer, i.args), j, tb(i), {"sw": xp, "own": i.x})
                   for i, xp in zip(its, xprime)]
            r_sw = subject.evaluate(eps, layer=l_o, deltas=d_swap)
            r_nc = subject.evaluate(eps, layer=l_o, deltas=d_nc)
            r_na = subject.evaluate(eps)
            for k in range(len(its)):
                recs.append({"src_ok": src_ok[k]["sw"], "sw_swap": r_sw[k]["sw"], "sw_nc": r_nc[k]["sw"],
                             "sw_nat": r_na[k]["sw"], "own_nc": r_nc[k]["own"], "own_nat": r_na[k]["own"]})
        ok = [r for r in recs if r["src_ok"]]
        swap[j] = {
            "n": len(recs), "n_src_ok": len(ok),
            "CT": (_acc([r["sw_swap"] for r in ok]) - _acc([r["sw_nc"] for r in ok])) if ok else None,
            "P_swap_sw": _acc([r["sw_swap"] for r in ok]), "P_nc_sw": _acc([r["sw_nc"] for r in ok]),
            "P_nat_sw": _acc([r["sw_nat"] for r in ok]),
            "acc_nc": _acc([r["own_nc"] for r in recs]), "acc_nat": _acc([r["own_nat"] for r in recs]),
        }
        log(f"  {j}: CT={swap[j]['CT']} nc_sw={swap[j]['P_nc_sw']} nat_sw={swap[j]['P_nat_sw']}")

    # ---------------------------------------------------------------- M4 bundles
    log("M4 bundles")
    bund = {}
    for b, S in cfg["bundles"].items():
        h_b = subject.prefix_hidden([bundle(i.name, b, i.x) for i in addsub], [l_o])[:, 0]
        Db = h_b - h_lat_lo
        cl = list(S) + [C.U]
        # source competence of the bundle sentence for its own consumers
        src_comp = {}
        for j in S:
            its = items(j, hA | hB)
            rr = subject.evaluate([EpSpec(bundle(i.name, b, i.x), j, None, {"own": i.x}) for i in its])
            src_comp[j] = _acc([r["own"] for r in rr])
        sw = rank_sweep(Db, cl, f"bundle {b}")
        bund[b] = {"consumers": cl, "sweep": sw, "source_competence": src_comp,
                   "full_gain": {j: full_gain(sw, j) for j in cl}}

    # ---------------------------------------------------------------- gates V2, W
    pool1 = R["V1"]["pool_passing_V1"]
    v2a_pass = [j for j in pool1 if v2a_gain[j] is not None and v2a_gain[j] >= gates["V2a_gain_min"]]
    v2a = len(v2a_pass) >= gates["V2a_min_pool_pass"] and (v2a_gain[C.U] or 0) >= gates["V2a_gain_min"]
    ct_ok = lambda j: swap[j]["CT"] is not None and swap[j]["CT"] >= gates["V2b_ct_min"]
    v2b = len([j for j in pool1 if ct_ok(j)]) >= gates["V2b_min_pool_pass"] and ct_ok(C.U)
    v2c_detail = {}
    for j in SIX:
        s = swap[j]
        okj = (s["P_nc_sw"] is not None and s["P_nat_sw"] is not None and
               abs(s["P_nc_sw"] - s["P_nat_sw"]) <= gates["V2c_fprime_shift_max"] and
               s["acc_nc"] is not None and abs(s["acc_nc"] - s["acc_nat"]) <= gates["V2c_acc_shift_max"])
        v2c_detail[j] = okj
    v2c = all(v2c_detail.values())

    def r_dir(sw, cons, ev):
        vals = [sw[ev][j]["r_star"] for j in cons if j in sw[ev]]
        if not vals or any(v is None for v in vals):
            return None
        return max(vals, key=lambda r: RK.as_number(r, subject.d_model))

    rx_cons = v2a_pass + ([C.U] if (v2a_gain[C.U] or 0) >= gates["V2a_gain_min"] else [])
    rX = {ev: r_dir(content, rx_cons, ev) for ev in ("A", "B")}
    rB = {b: {ev: r_dir(bund[b]["sweep"], list(S), ev) for ev in ("A", "B")} for b, S in cfg["bundles"].items()}
    dm = subject.d_model
    rX_gm = RK.geo_mean(rX["A"], rX["B"], dm)
    rB_gm = {b: RK.geo_mean(rB[b]["A"], rB[b]["B"], dm) for b in rB}
    w1 = (RK.log2_ratio(rX["A"], rX["B"], dm) is not None and RK.log2_ratio(rX["A"], rX["B"], dm) <= gates["W1_log2_ratio_max"]
          and RK.log2_ratio(rB["B4"]["A"], rB["B4"]["B"], dm) is not None
          and RK.log2_ratio(rB["B4"]["A"], rB["B4"]["B"], dm) <= gates["W1_log2_ratio_max"])
    b4_own = [bund["B4"]["full_gain"][j] for j in cfg["bundles"]["B4"] if bund["B4"]["full_gain"][j] is not None]
    b4_own_mean = sum(b4_own) / len(b4_own) if b4_own else None
    u_c, u_b = v2a_gain[C.U], bund["B4"]["full_gain"][C.U]
    w2 = (u_c is not None and u_b is not None and u_c > 0 and u_b <= gates["W2_bundle_U_ratio_max"] * u_c
          and b4_own_mean is not None and b4_own_mean >= gates["W2_bundle_own_gain_min"])
    w3 = rX_gm is not None and rB_gm["B4"] is not None and rB_gm["B4"] >= gates["W3_window_ratio_min"] * rX_gm
    step = lambda r: 0 if r is None else math.log2(r)
    tol = gates["W4_monotone_tolerance_steps"]
    w4 = (None not in (rB_gm["B1"], rB_gm["B2"], rB_gm["B4"]) and
          step(rB_gm["B1"]) <= step(rB_gm["B2"]) + tol and step(rB_gm["B2"]) <= step(rB_gm["B4"]) + tol)
    R["M4"] = {"content": content, "content_full_gain": v2a_gain, "swap": swap, "bundles": bund,
               "rX_by_eval_half": rX, "rX_geomean": rX_gm, "rB_by_eval_half": rB, "rB_geomean": rB_gm,
               "rX_consumers": rx_cons, "B4_own_full_gain_mean": b4_own_mean, "U_gain_content": u_c, "U_gain_B4": u_b}
    R["V2"] = {"V2a": v2a, "V2a_pool_pass": v2a_pass, "V2b": v2b, "V2c": v2c, "V2c_by_consumer": v2c_detail}
    R["W"] = {"W1": w1, "W2": w2, "W3": w3, "W4": w4}
    R["core_instrument_valid"] = bool(v0 and v1a and v1b and v1c and v1d and v2a and v2b and v2c)
    R["phase_diagram_measurable"] = bool(w1 and w2 and w3 and w4)
    R["elapsed_s"] = time.time() - t_start
    log(f"V2={R['V2']} W={R['W']} core_valid={R['core_instrument_valid']}")
    return R
