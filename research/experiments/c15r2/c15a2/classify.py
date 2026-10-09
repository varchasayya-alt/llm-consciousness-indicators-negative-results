"""Mechanical R2 classification (memo sec. 6-7 and 12.3) and confirmation verdicts. Pure functions on result dicts.

Precedence P4 -> P1 -> P2(strong) -> P3 -> P0 -> IND. A failed gate supports absence only as a *powered* failure:
validity holds, the 95% CI lies entirely outside the pass region and the point estimate misses by the margin.
"""
from __future__ import annotations

import math

from . import config as C

GATES = ("W0a", "W0b", "W1", "W2", "W3p", "W4", "W5")
CONTENT = ("W1", "W2", "W3p")


def num(x):
    if isinstance(x, str):
        return float(x)
    return float("nan") if x is None else float(x)


def _rm(r, need):
    return (min(num(r), C.MARGIN_RATIO_CAP) - need) / need


def cell(res, layer):
    """Gate, validity, powered-failure and margin view of one (model, layer) cell."""
    w0 = res["w0"]
    pl = res["per_layer"][str(layer)] if str(layer) in res["per_layer"] else res["per_layer"][layer]
    w1, w2, w3, w45 = pl["w1"], pl["w2"], pl["w3p"], pl["w45"]
    g = {"W0a": bool(w0["pass_w0a"]), "W0b": bool(w0["pass_w0b"]), "W1": bool(w1["pass"]), "W2": bool(w2["pass"]),
         "W3p": bool(w3.get("pass", False)), "W4": bool(w45["pass_w4"]), "W5": bool(w45["pass_w5"])}
    w0a = g["W0a"]
    cov1 = num(w1["frac_unmatched"]) <= C.MAX_UNMATCHED
    cov2 = num(w2["frac_unmatched"]) <= C.MAX_UNMATCHED
    w3_ok = bool(w3.get("assessable", False))
    valid = bool(w0a and w3_ok and cov1 and cov2)
    P = C.PF
    hit, rt, hci, rci = num(w1["hit_J"]), num(w1["ratio"]), w1["hit_J_ci"], w1["ratio_ci"]
    pf = {}
    pf["W1"] = bool(w0a and cov1 and not g["W1"] and (
        (num(hci[1]) < P["W1"]["hit_ci_upper_lt"] and hit <= P["W1"]["hit_point_le"])
        or (num(rci[1]) < P["W1"]["ratio_ci_upper_lt"] and rt <= P["W1"]["ratio_point_le"])))
    rj, r2, jci, r2ci = num(w2["rate_J"]), num(w2["ratio"]), w2["rate_J_ci"], w2["ratio_ci"]
    pf["W2"] = bool(w0a and cov2 and w2["n_pairs_included"] >= P["W2"]["min_pairs"] and not g["W2"] and (
        (num(jci[1]) < P["W2"]["rate_ci_upper_lt"] and rj <= P["W2"]["rate_point_le"])
        or (num(r2ci[1]) < P["W2"]["ratio_ci_upper_lt"] and r2 <= P["W2"]["ratio_point_le"])))
    pf["W3p"] = bool(w0a and w3_ok and w3.get("powered_failure", False))
    d4, d4ci = num(w45["w4_diff_pp"]), w45["w4_diff_ci"]
    pf["W4"] = bool(w0a and num(w45["acc_clean"]) >= C.W4_REQ_ACC and not g["W4"]
                    and num(d4ci[1]) < P["W4"]["diff_ci_upper_lt"] and d4 <= P["W4"]["diff_point_le"])
    si, sci = num(w45["SI_iso"]), w45["SI_iso_ci"]
    cert = w45["s_star_status"] in C.W5_CERTIFIABLE
    pf["W5"] = bool(w0a and cert and not g["W5"] and num(sci[0]) > P["W5"]["si_ci_lower_gt"]
                    and si >= P["W5"]["si_point_ge"])
    m = {"W1": min((hit - C.W1_MIN_J) / C.W1_MIN_J, _rm(rt, C.W1_MIN_RATIO)),
         "W2": min((rj - C.W2_MIN_J) / C.W2_MIN_J, _rm(r2, C.W2_MIN_RATIO)),
         "W3p": (num(w3.get("diff", float("nan"))) - C.W3P_MIN_DIFF) / C.W3P_MIN_DIFF if w3_ok else -math.inf,
         "W4": (d4 - C.W4_MIN_DIFF_PP) / C.W4_MIN_DIFF_PP,
         "W5": (C.W5_MAX_SI_ISO - si) / C.W5_MAX_SI_ISO if math.isfinite(si) else -math.inf}
    m = {k: (v if v == v else -math.inf) for k, v in m.items()}          # NaN -> -inf
    return {"model": res["model"], "layer": int(layer), "gates": g, "valid": valid, "w3_ok": w3_ok, "pf": pf,
            "margins": m,
            "A_pass": g["W1"] and g["W2"], "B_pass": g["W4"] and g["W5"],
            "A_pf": pf["W1"] or pf["W2"], "B_pf": pf["W4"] or pf["W5"],
            "all_pass": all(g.values()), "min_margin": min(m.values()),
            "mA": min(m["W1"], m["W2"]), "mB": min(m["W4"], m["W5"])}


def cells(results_by_model):
    out = []
    for mk, res in results_by_model.items():
        for l in res["per_layer"]:
            out.append(cell(res, l))
    return out


def _size(mk):
    return C.MODEL_SIZE_ORDER.get(mk, 99.0)


def qualifying_models(cs):
    by = {}
    for c in cs:
        by.setdefault(c["model"], []).append(c)
    q = {}
    for mk, lst in by.items():
        frac = sum(c["valid"] for c in lst) / len(lst)
        q[mk] = bool(lst[0]["gates"]["W0a"] and frac >= C.MODEL_MIN_VALID_LAYER_FRACTION)
    return q, by


def classify(results_by_model, pc4_failed=False):
    cs = cells(results_by_model)
    q, by = qualifying_models(cs)
    n_valid = sum(c["valid"] for c in cs)
    frac_valid = n_valid / max(len(cs), 1)
    w3_dead_everywhere = all(not any(c["w3_ok"] for c in lst) for lst in by.values())
    base = {"n_cells": len(cs), "n_valid_cells": n_valid, "frac_valid": frac_valid, "qualifying_models": q,
            "cells": cs}
    # ---------------------------------------------------------------- P4
    if pc4_failed or w3_dead_everywhere or frac_valid < C.P4_MIN_VALID_FRACTION:
        why = ("PC4 failed" if pc4_failed else "W3' controls fail in every model" if w3_dead_everywhere
               else f"only {frac_valid:.2f} of cells assay-valid")
        return dict(base, pattern="P4", reason=why, frozen_cells=[])
    valid = [c for c in cs if c["valid"]]
    # ---------------------------------------------------------------- P1
    p1 = sorted([c for c in valid if c["all_pass"]],
                key=lambda c: (-min(c["margins"][k] for k in ("W1", "W2", "W3p", "W4", "W5")), _size(c["model"]),
                               c["layer"]))
    if p1:
        ch = p1[0]
        rep = next((c for c in p1 if c["model"] != ch["model"]), None)
        return dict(base, pattern="P1", frozen_cells=[{"model": ch["model"], "layer": ch["layer"], "role": "unified",
                                                       "predicted": "passes W0a, W0b', W1, W2, W3', W4, W5"}],
                    replication_candidate=rep and {"model": rep["model"], "layer": rep["layer"]})
    # ---------------------------------------------------------------- P2
    X = [c for c in valid if c["A_pass"] and c["B_pf"]]
    Y = [c for c in valid if c["B_pass"] and c["A_pf"]]
    pairs = [(x, y) for x in X for y in Y if x["model"] != y["model"]]
    weak = any(x["model"] == y["model"] for x in X for y in Y)
    if pairs:
        x, y = sorted(pairs, key=lambda p: (-min(p[0]["mA"], p[1]["mB"]), _size(p[0]["model"]), p[0]["layer"],
                                            p[1]["layer"]))[0]
        return dict(base, pattern="P2", P2_weak_also=weak, frozen_cells=[
            {"model": x["model"], "layer": x["layer"], "role": "X", "predicted": "A passes; B powered failure"},
            {"model": y["model"], "layer": y["layer"], "role": "Y", "predicted": "B passes; A powered failure"}])
    # ---------------------------------------------------------------- P3
    p3 = []
    for mk, lst in by.items():
        v = [c for c in lst if c["valid"]]
        if q[mk] and v and all(c["gates"]["W4"] and c["pf"]["W1"] and c["pf"]["W2"] and c["pf"]["W3p"]
                               and c["pf"]["W5"] for c in v):
            r = sorted(v, key=lambda c: (-c["margins"]["W4"], c["layer"]))[0]
            p3.append(r)
    if p3:
        r = sorted(p3, key=lambda c: (-c["margins"]["W4"], _size(c["model"]), c["layer"]))[0]
        return dict(base, pattern="P3", P2_weak=weak, frozen_cells=[
            {"model": r["model"], "layer": r["layer"], "role": "representative",
             "predicted": "W4 passes; W1, W2, W3', W5 powered failures"}])
    # ---------------------------------------------------------------- P0
    all_q = len(by) == len(C.MODEL_SIZE_ORDER) and all(q.values())
    if all_q and not any(c["A_pass"] or c["gates"]["W3p"] for c in valid) and all(
            all(c["pf"][g] for g in CONTENT if not c["gates"][g]) for c in valid):
        r = sorted(valid, key=lambda c: (max(c["margins"][g] for g in CONTENT), _size(c["model"]), c["layer"]))[0]
        return dict(base, pattern="P0", P2_weak=weak, frozen_cells=[
            {"model": r["model"], "layer": r["layer"], "role": "representative",
             "predicted": "no A or W3' pass; failing content gates powered"}])
    return dict(base, pattern="IND", P2_weak=weak, frozen_cells=[],
                reason="no pre-declared pattern holds (e.g. complementary passes with unpowered failures)")


def confirm_verdict(pattern, role, c):
    """c: cell() view of the frozen cell on G_confirm2."""
    if not c["valid"]:
        return False
    if pattern == "P1":
        return c["all_pass"]
    if pattern == "P2":
        return (c["A_pass"] and c["B_pf"]) if role == "X" else (c["B_pass"] and c["A_pf"])
    if pattern == "P3":
        return c["gates"]["W4"] and all(c["pf"][g] for g in ("W1", "W2", "W3p", "W5"))
    if pattern == "P0":
        return (not c["A_pass"]) and (not c["gates"]["W3p"]) and all(c["pf"][g] for g in CONTENT
                                                                      if not c["gates"][g])
    return False
