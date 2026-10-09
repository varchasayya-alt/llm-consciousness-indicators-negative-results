"""C15-R2 runner (PI authorization D74; frozen STOP/GO table: memo/c15r2_preregistration_FROZEN.md sec. 11).

Phases: PINS                       verify A-stage provenance pins (trees, files) -- STOP and report on mismatch
        Z --model K                R2-Z: pinned artifacts / lens re-verified (no downloads); ST-0 exclusion on mismatch
        SMOKE2 --model K           R2-ENG: full R2 pipeline on SMOKE2 material (never evidence); PC3 and NC1 must pass
        SELECT --model K           W0a, W0b', W1, W2, W3', W4, W5 (+ controls, CIs) on G_select2, all band layers
        CLASSIFY                   P4 / P1 / P2 / P3 / P0 / IND by the frozen rules
        FREEZE                     FREEZE2 record + tensors for the pattern's cells (must then be committed)
        CONFIRM --model K --layer L  frozen cell once on G_confirm2
        REPORT                     verdict + tables; A-stage pins re-verified
Any other phase (B, C, H, SAT, F, RESCUE, C16, DOWNLOAD) is refused.

Usage (repo root): .venv/Scripts/python.exe research/experiments/c15r2/run_r2.py PHASE [--model K] [--layer L]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import c15a2  # noqa: E402,F401  (byte-code off; read-only A-stage import path; R2 RNG seed)
import torch  # noqa: E402

from c15a.artifacts import (MANIFEST_PATH, Z0Error, check_lens_checkpoint, lens_paths, load_lens_checkpoint,  # noqa: E402
                            load_model, manifest, z0_files)
from c15a.lens import Lens  # noqa: E402
from c15a.workspace import band_layers, word_initial_ids  # noqa: E402
from c15a2 import config as C  # noqa: E402
from c15a2 import guards  # noqa: E402
from c15a2.classify import cell, classify, confirm_verdict  # noqa: E402
from c15a2.materials import SPLIT_MANIFEST2, load_confirm2, load_select2, load_smoke2, split_manifest2  # noqa: E402

RES = C.RESULTS
MODELS = list(manifest()["models"])


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, torch.Tensor):
        return _clean(o.tolist())
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _clean(o.item())
        except (ValueError, AttributeError):
            pass
    return o


def dump(obj, path):
    path = guards.safe_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(obj), f, indent=1, default=str)
    print("wrote", path, flush=True)


def provenance(phase, key=None):
    import transformers
    return {"phase": phase, "model": key, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_head": guards.git_head(), "config_hash": C.config_hash(),
            "thresholds_sha256": guards.sha256_file(C.THRESHOLDS_PATH),
            "prereg_sha256": guards.sha256_file(C.PREREG_PATH),
            "split_manifest2_sha256": guards.sha256_file(SPLIT_MANIFEST2),
            "artifact_manifest_sha256": guards.sha256_file(MANIFEST_PATH),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "threads": torch.get_num_threads(), "platform": platform.platform()}


def load_all(key):
    lm = load_model(key)
    lp, _ = lens_paths(key)
    ck = load_lens_checkpoint(lp)
    info = check_lens_checkpoint(ck, n_layers=lm.n_layers, d_model=lm.d_model,
                                 hf_model_id=manifest()["models"][key]["hf_model_id"])
    return lm, Lens(ck), info


def _json(path):
    return json.load(open(path, encoding="utf-8"))


def _require(path, what):
    if not os.path.isfile(path):
        raise guards.GuardError(f"{what} missing: {path}")
    return _json(path)


# ------------------------------------------------------------------------------------------------ phases
def phase_pins():
    rec = guards.verify_astage_pins()
    rec["provenance"] = provenance("PINS")
    dump(rec, os.path.join(RES, f"pins_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"))


def phase_z(key):
    guards.verify_astage_pins()
    t0 = time.time()
    rec = {"model_key": key}
    try:
        rec.update(z0_files(key, hash_weights=True))
        lm, lens, linfo = load_all(key)
    except Z0Error as e:
        rec.update({"pass": False, "error": str(e), "decision": "ST-0: exclude this model"})
        dump(rec, os.path.join(RES, f"z2_{key}.json"))
        return
    rec["lens_checkpoint"] = linfo
    rec["tokenizer"] = {"len": len(lm.tok), "n_word_initial": len(word_initial_ids(lm.tok, lm.n_vocab))}
    from c15a.hooks import forward
    sm = load_smoke2()
    agree = n = 0
    for p in sm["paragraphs"]:
        ids = torch.tensor([lm.encode(p["text"])[:64]])
        h, r = forward(lm, ids, record=[lm.n_layers - 2])
        mod = lm.lm_head(h[0, 1:]).argmax(-1)
        lt = lens.readout(lm, r[lm.n_layers - 2][0, 1:], lm.n_layers - 2).argmax(-1)
        agree += int((lt == mod).sum())
        n += mod.numel()
    rec["convention_check"] = {"lens_agreement_L-2_smoke2": agree / n, "threshold": 0.30}
    rec["band_layers"] = band_layers(lm.n_layers)
    rec["pass"] = bool(agree / n >= 0.30)
    rec["seconds"] = time.time() - t0
    rec["provenance"] = provenance("Z", key)
    dump(rec, os.path.join(RES, f"z2_{key}.json"))


def _z_ok(key):
    z = _require(os.path.join(RES, f"z2_{key}.json"), f"R2-Z record for {key}")
    if not z.get("pass"):
        raise guards.GuardError(f"{key} excluded at R2-Z (ST-0)")


def eng_check(w3):
    """R2-ENG criterion (PC3 and NC1 must pass). Uses the full W3' block whenever it was computed (whether or not
    the cell is assessable, e.g. PC1 may fail on the small SMOKE2 reference set) and otherwise the perp-free
    engineering diagnostics computed when matched-perp coverage is insufficient."""
    src_name = "w3p" if "PC3" in w3 else "eng_diagnostics"
    src = w3 if src_name == "w3p" else w3.get("eng_diagnostics", {})
    pc3, nc1 = src.get("PC3"), src.get("NC1")
    return {"source": src_name, "PC3": pc3, "NC1": nc1, "w3p_assessable": bool(w3.get("assessable")),
            "pass": bool(pc3 and nc1 and pc3.get("pass") and nc1.get("pass"))}


def phase_smoke2(key):
    from c15a2.assays2 import R2Stage
    _z_ok(key)
    lm, lens, _ = load_all(key)
    band = band_layers(lm.n_layers)
    l = band[len(band) // 2]
    st = R2Stage(lm, lens, load_smoke2(), layers=[l])
    t0 = time.time()
    res = st.run_all_r2()
    res["eng"] = dict(eng_check(res["per_layer"][l]["w3p"]), layer=l)
    res["seconds"] = time.time() - t0
    res["non_evidential"] = "SMOKE2 material only; never enters any gate, pattern or selection"
    res["provenance"] = provenance("SMOKE2", key)
    dump(res, os.path.join(RES, "engineering", f"smoke2_{key}.json"))


def phase_select(key):
    from c15a2.assays2 import R2Stage
    guards.require_clean_tree()
    guards.verify_astage_pins()
    _z_ok(key)
    for k in MODELS:                 # frozen order: SMOKE2 must have passed on every non-excluded model first
        z = _require(os.path.join(RES, f"z2_{k}.json"), f"R2-Z record for {k}")
        if not z.get("pass"):
            continue
        eng = _require(os.path.join(RES, "engineering", f"smoke2_{k}.json"), f"SMOKE2 record for {k}")
        if not eng["eng"]["pass"]:
            raise guards.GuardError(f"SMOKE2 PC3/NC1 failed on {k}: fix the implementation (no G data)")
    lm, lens, _ = load_all(key)
    st = R2Stage(lm, lens, load_select2())
    t0 = time.time()
    res = st.run_all_r2()
    res["seconds"] = time.time() - t0
    tens = {"Q": st.sj, "C_K": st.C_K, "vgen": torch.tensor(st.vgen), "vgen_dropped": st.vgen_dropped,
            "hbar": st.hbar, "sj_info": st.sj_info}
    tp = guards.safe_path(os.path.join(RES, "select_tensors", f"{key}.pt"))
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    torch.save(tens, tp)
    res["tensors"] = {"path": os.path.relpath(tp, RES), "sha256": guards.sha256_file(tp)}
    res["provenance"] = provenance("SELECT", key)
    dump(res, os.path.join(RES, f"select2_{key}.json"))


def _selected():
    out = {}
    for k in MODELS:
        z = os.path.join(RES, f"z2_{k}.json")
        if os.path.isfile(z) and not _json(z).get("pass"):
            continue                                            # ST-0 exclusion
        out[k] = _require(os.path.join(RES, f"select2_{k}.json"), f"SELECT result for {k}")
    return out


def phase_classify():
    res = _selected()
    out = classify(res)
    out["models_assessed"] = sorted(res)
    out["provenance"] = provenance("CLASSIFY")
    dump(out, os.path.join(RES, "classify2.json"))
    print("PATTERN:", out["pattern"], out.get("reason", ""), [(c["model"], c["layer"]) for c in out["frozen_cells"]])


def phase_freeze():
    cl = _require(os.path.join(RES, "classify2.json"), "classification")
    if cl["pattern"] not in ("P1", "P2", "P3", "P0"):
        raise guards.GuardError(f"pattern {cl['pattern']}: STOP and report (no confirmation)")
    fz = os.path.join(RES, "frozen2")
    os.makedirs(guards.safe_path(fz), exist_ok=True)
    hashes, cells_out = {}, []
    for fc in cl["frozen_cells"]:
        key, l = fc["model"], int(fc["layer"])
        sel = _json(os.path.join(RES, f"select2_{key}.json"))
        tp = os.path.join(RES, sel["tensors"]["path"])
        if guards.sha256_file(tp) != sel["tensors"]["sha256"]:
            raise guards.GuardError("select tensors changed since SELECT")
        T = torch.load(tp, map_location="cpu", weights_only=False)
        files = {f"Q_{key}_L{l}.pt": T["Q"][l], f"C_K_{key}.pt": T["C_K"], f"vgen_{key}.pt": T["vgen"]}
        for name, t in files.items():
            p = guards.safe_path(os.path.join(fz, name))
            torch.save(t, p)
            hashes[name] = guards.sha256_file(p)
        pl = sel["per_layer"][str(l)]
        cells_out.append(dict(fc, a_star=pl["dose"]["a_star"], hbar=pl["hbar"], sj_info=pl["sj"],
                              s_star={"s_star": pl["w45"]["s_star"], "status": pl["w45"]["s_star_status"]},
                              vgen_dropped=T["vgen_dropped"], select_view=cell(sel, l)))
    rec = {"pattern": cl["pattern"], "cells": cells_out, "frozen_tensors": hashes, "config_hash": C.config_hash(),
           "thresholds_sha256": guards.sha256_file(C.THRESHOLDS_PATH),
           "materials": {"g_confirm2_sha256": split_manifest2()["files"]["g_confirm2.json"],
                         "g_select2_sha256": split_manifest2()["files"]["g_select2.json"]},
           "classification_sha256": guards.sha256_file(os.path.join(RES, "classify2.json")),
           "rule": "Each frozen cell is confirmed once on G_confirm2; no re-selection, no second confirmation.",
           "provenance": provenance("FREEZE")}
    dump(rec, guards.FREEZE_PATH)
    print("FREEZE2 written. Commit it before CONFIRM.", flush=True)


def phase_confirm(key, layer):
    from c15a2.assays2 import R2Stage
    guards.require_clean_tree()
    guards.verify_astage_pins()
    mats, fr = load_confirm2(key, layer)
    fc = next(c for c in fr["cells"] if c["model"] == key and int(c["layer"]) == int(layer))
    l = int(layer)
    lm, lens, _ = load_all(key)
    fz = os.path.join(RES, "frozen2")
    frozen = {"Q": {l: torch.load(os.path.join(fz, f"Q_{key}_L{l}.pt"))},
              "C_K": torch.load(os.path.join(fz, f"C_K_{key}.pt")),
              "vgen": torch.load(os.path.join(fz, f"vgen_{key}.pt")).tolist(),
              "vgen_dropped": fc["vgen_dropped"], "hbar": {l: fc["hbar"]}, "sj_info": {l: fc["sj_info"]}}
    st = R2Stage(lm, lens, mats, frozen=frozen, layers=[l])
    t0 = time.time()
    res = st.run_all_r2(frozen_layer_params={"a_star": fc["a_star"], "s_star": fc["s_star"]})
    res["seconds"] = time.time() - t0
    rt = json.loads(json.dumps(_clean(res)))
    view = cell(rt, l)
    res["confirm_view"] = view
    res["verdict"] = {"pattern": fr["pattern"], "role": fc["role"], "reproduced": confirm_verdict(fr["pattern"],
                                                                                                fc["role"], view)}
    res["frozen_record_sha256"] = guards.sha256_file(guards.FREEZE_PATH)
    res["provenance"] = provenance("CONFIRM", key)
    dump(res, guards.confirm_path(key, l))


def phase_report():
    from c15a2.report import write_report
    pins = guards.verify_astage_pins()
    write_report(RES, pins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    a = ap.parse_args()
    guards.check_phase(a.phase)
    if a.phase in ("Z", "SMOKE2", "SELECT", "CONFIRM") and a.model not in MODELS:
        raise SystemExit(f"--model must be one of {MODELS}")
    torch.set_grad_enabled(False)
    {"PINS": phase_pins, "Z": lambda: phase_z(a.model), "SMOKE2": lambda: phase_smoke2(a.model),
     "SELECT": lambda: phase_select(a.model), "CLASSIFY": phase_classify, "FREEZE": phase_freeze,
     "CONFIRM": lambda: phase_confirm(a.model, a.layer), "REPORT": phase_report}[a.phase]()


if __name__ == "__main__":
    main()
