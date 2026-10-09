"""Gates, normalised margins, model/layer choice and the FREEZE record (A-stage)."""
from __future__ import annotations

import math

from . import config as C

MODEL_SIZE_ORDER = {"qwen3-1.7b": 1.7, "qwen3.5-2b": 2.0, "qwen3-4b": 4.0}


def _num(x):
    """Floats from result JSON, where non-finite values are serialised as "inf" / "-inf" strings."""
    if isinstance(x, str):
        return float(x)
    return 0.0 if x is None else float(x)


def _ratio_margin(ratio, need):
    r = min(_num(ratio), C.MARGIN_RATIO_CAP)
    return (r - need) / need


def layer_gate(w0, pl):
    w1, w2, w3, w45 = pl["w1"], pl["w2"], pl["w3"], pl["w45"]
    gates = {"W0": bool(w0["pass"]), "W1": bool(w1["pass"]), "W2": bool(w2["pass"]), "W3": bool(w3["pass"]),
             "W4": bool(w45["pass_w4"]), "W5": bool(w45["pass_w5"])}
    si = _num(w45["SI_iso"])
    margins = {
        "W1": min((_num(w1["hit_J"]) - C.W1_MIN_J) / C.W1_MIN_J, _ratio_margin(w1["ratio"], C.W1_MIN_RATIO)),
        "W2": min((_num(w2["rate_J"]) - C.W2_MIN_J) / C.W2_MIN_J, _ratio_margin(w2["ratio"], C.W2_MIN_RATIO)),
        "W3": (_num(w3["diff"]) - C.W3_MIN_DIFF) / C.W3_MIN_DIFF,
        "W4": (_num(w45["w4_diff_pp"]) - C.W4_MIN_DIFF_PP) / C.W4_MIN_DIFF_PP,
        "W5": (C.W5_MAX_SI_ISO - si) / C.W5_MAX_SI_ISO if math.isfinite(si) else -math.inf,
    }
    return gates, margins, all(gates.values()), min(margins.values())


def choose(results_by_model):
    """Highest minimum normalised margin over W1-W5 among (model, layer) passing W0-W5 on G_select;
    ties -> smaller model, then lower layer. The best passer of a different model is recorded as the
    designated replication candidate (PI amendment 9); it is not confirmed now."""
    cands = []
    for mk, res in results_by_model.items():
        for l, pl in res["per_layer"].items():
            g, m, ok, mm = layer_gate(res["w0"], pl)
            cands.append({"model": mk, "layer": int(l), "gates": g, "margins": m, "pass": ok, "min_margin": mm})
    passers = sorted([c for c in cands if c["pass"]],
                     key=lambda c: (-c["min_margin"], MODEL_SIZE_ORDER.get(c["model"], 99), c["layer"]))
    chosen = passers[0] if passers else None
    runner = next((c for c in passers if chosen and c["model"] != chosen["model"]), None)
    return {"candidates": cands, "n_passers": len(passers), "chosen": chosen, "replication_candidate": runner}


def confirm_verdict(conf):
    pl = next(iter(conf["per_layer"].values()))
    g, m, ok, mm = layer_gate(conf["w0"], pl)
    return {"gates": g, "margins": m, "pass": ok, "min_margin": mm,
            "w5_description": pl["w45"]["w5_description"]}
