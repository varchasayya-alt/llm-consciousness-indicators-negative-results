"""C15-R A-stage runner (PI authorization 2026-10-04; A-stage only).

Phases: Z0 | BENCH | SMOKE  (per model, engineering; SMOKE/BENCH use SMOKE material only, never W0-W5 evidence)
        SELECT --model K    (W0-W5 on G_select, all band layers; requires clean tree + Z0 pass)
        CHOOSE              (one model/layer by the pre-declared rule)
        FREEZE              (write the frozen record + tensors; must then be committed)
        CONFIRM             (W0-W5 once on G_confirm for the frozen model/layer)
        REPORT              (markdown tables)
Any other phase (B, C, H, SAT, F, RESCUE, ...) is refused by guards.check_phase.

Usage (repo root): .venv/Scripts/python.exe research/experiments/c15/run_astage.py PHASE [--model K]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from c15a import config as C  # noqa: E402
from c15a import guards  # noqa: E402
from c15a.artifacts import (MANIFEST_PATH, Z0Error, check_lens_checkpoint, lens_paths, load_lens_checkpoint,  # noqa: E402
                            load_model, manifest, sha256_file, z0_files)
from c15a.lens import Lens  # noqa: E402
from c15a.materials import SPLIT_MANIFEST, load_confirm, load_select, load_smoke, split_manifest  # noqa: E402
from c15a.workspace import band_layers, word_initial_ids  # noqa: E402

RES = guards.RESULTS
ENG = os.path.join(RES, "engineering")
MODELS = list(manifest()["models"])


def _default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return "inf" if o > 0 else "-inf"
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return o


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=1, default=_default)
    print("wrote", path, flush=True)


def provenance(phase, key=None):
    import transformers
    return {"phase": phase, "model": key, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_head": guards.git_head(), "config_hash": C.config_hash(),
            "artifact_manifest_sha256": sha256_file(MANIFEST_PATH),
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "threads": torch.get_num_threads(), "platform": platform.platform()}


def load_all(key):
    lm = load_model(key)
    lp, _ = lens_paths(key)
    ck = load_lens_checkpoint(lp)
    spec = manifest()["models"][key]
    info = check_lens_checkpoint(ck, n_layers=lm.n_layers, d_model=lm.d_model, hf_model_id=spec["hf_model_id"])
    return lm, Lens(ck), info


def phase_z0(key):
    t0 = time.time()
    rec = z0_files(key, hash_weights=True)
    lm, lens, linfo = load_all(key)
    rec["lens_checkpoint"] = linfo
    rec["tokenizer"] = {"len": len(lm.tok), "bos_token": lm.tok.bos_token, "lm_head_rows": lm.lm_head.weight.shape[0],
                        "n_word_initial": len(word_initial_ids(lm.tok, lm.n_vocab))}
    if lm.lm_head.weight.shape[0] < len(lm.tok):
        raise Z0Error("lm_head has fewer rows than the tokenizer")
    sm = load_smoke()
    from c15a.hooks import forward
    agree = agree_id = n = 0
    for p in sm["paragraphs"]:
        ids = torch.tensor([lm.encode(p["text"])[: C.SEQ_LEN]])
        h, rec_ = forward(lm, ids, record=[lm.n_layers - 2])
        mod = lm.lm_head(h[0, 1:]).argmax(-1)
        hl = rec_[lm.n_layers - 2][0, 1:]
        lt = lens.readout(lm, hl, lm.n_layers - 2).argmax(-1)
        it = lm.unembed(hl).float().argmax(-1)
        agree += int((lt == mod).sum())
        agree_id += int((it == mod).sum())
        n += mod.numel()
    rec["convention_check"] = {"lens_agreement_L-2_smoke": agree / n, "logit_lens_agreement_L-2_smoke": agree_id / n,
                               "threshold": 0.30}
    if agree / n < 0.30:
        raise Z0Error(f"lens convention check failed: agreement {agree / n:.3f} < 0.30")
    rec["band_layers"] = band_layers(lm.n_layers)
    rec["pass"] = True
    rec["seconds"] = time.time() - t0
    rec["provenance"] = provenance("Z0", key)
    dump(rec, os.path.join(RES, f"z0_{key}.json"))


def phase_bench(key):
    from c15a.gp import gp_nonneg
    from c15a.hooks import forward
    from c15a.lens import atoms
    from c15a.workspace import vgen_ids
    lm, lens, _ = load_all(key)
    sm = load_smoke()
    ids1 = torch.tensor([lm.encode(sm["paragraphs"][0]["text"])[: C.SEQ_LEN]])
    ids8 = ids1.repeat(8, 1)
    out = {}
    forward(lm, ids1)
    for name, ids in (("b1_t64", ids1), ("b8_t64", ids8)):
        t0 = time.time()
        reps = 3
        for _ in range(reps):
            forward(lm, ids)
        dt = (time.time() - t0) / reps
        out[name] = {"sec": dt, "tok_per_s": ids.numel() / dt}
    l = band_layers(lm.n_layers)[0]
    vg, _ = vgen_ids(lm.tok, lm.n_vocab, [lm.encode(p["text"]) for p in sm["paragraphs"]])
    t0 = time.time()
    D = atoms(lens.J(l), lm.w_eff()[torch.tensor(vg)])
    out["dict_build_sec"] = time.time() - t0
    _, rec = forward(lm, ids8, record=[l])
    H = rec[l][:, 1:].reshape(-1, lm.d_model)[:256]
    t0 = time.time()
    gp_nonneg(H, D, C.GP_K)
    dt = time.time() - t0
    out["gp_states_per_s"] = H.shape[0] / dt
    out["vgen_size_smoke_freq"] = len(vg)
    # workload projection for SELECT (token-forward equivalents and GP states), per the protocol sizes
    nb = len(band_layers(lm.n_layers))
    tok_rate = out["b8_t64"]["tok_per_s"]
    gp_rate = out["gp_states_per_s"]
    sel = split_manifest()["counts"]["select"]
    text_tok = sel["paragraphs"] * C.SEQ_LEN
    th_tok = sel["two_hop"] * 18
    per_layer_tok = (len(C.DOSE_GRID) * 2 * C.DOSE_N_DIR * C.DOSE_N_SEQ * C.DOSE_LEN
                     + 3 * sel["concepts"] * C.W1_CONTEXTS_PER_CONCEPT * 36
                     + 3 * 3 * sel["countries"] * 14
                     + 2 * C.W3_N_CONCEPTS * C.W3_N_CONTEXTS * 44
                     + (2 + len(C.RAND_SCALE_GRID)) * (text_tok + th_tok))
    gp_states = (sel["paragraphs"] * (C.SEQ_LEN - C.STAT_POS_MIN)
                 + 3 * (sel["paragraphs"] * (C.SEQ_LEN - 1) + th_tok))
    proj = nb * (per_layer_tok / tok_rate + gp_states / gp_rate) + text_tok / tok_rate
    out["projection"] = {"band_layers": nb, "tokens_per_layer": per_layer_tok, "gp_states_per_layer": gp_states,
                         "select_hours_est": proj / 3600.0,
                         "note": "excludes matching optimisation and W3 bootstrap; S0-style estimate"}
    out["provenance"] = provenance("BENCH", key)
    out["non_evidential"] = True
    dump(out, os.path.join(ENG, f"bench_{key}.json"))


def phase_smoke(key):
    from c15a.assays import AStage
    lm, lens, _ = load_all(key)
    sm = load_smoke()
    band = band_layers(lm.n_layers)
    st = AStage(lm, lens, sm, layers=[band[len(band) // 2]])
    st.boot_w3, st.boot_w5 = 5, 50
    t0 = time.time()
    res = st.run_all()
    res["seconds"] = time.time() - t0
    res["non_evidential"] = "SMOKE material only; never enters W0-W5 evidence or selection"
    res["provenance"] = provenance("SMOKE", key)
    dump(res, os.path.join(ENG, f"smoke_{key}.json"))


def phase_select(key):
    from c15a.assays import AStage
    guards.require_clean_tree()
    z0 = os.path.join(RES, f"z0_{key}.json")
    if not (os.path.isfile(z0) and json.load(open(z0))["pass"]):
        raise guards.GuardError(f"Z0 not passed for {key}")
    lm, lens, _ = load_all(key)
    mats = load_select()
    st = AStage(lm, lens, mats)
    t0 = time.time()
    res = st.run_all()
    res["seconds"] = time.time() - t0
    tens = {"Q": st.sj, "C_K": st.C_K, "vgen": torch.tensor(st.vgen), "vgen_dropped": st.vgen_dropped,
            "hbar": st.hbar, "sj_info": st.sj_info}
    tp = os.path.join(RES, "select_tensors", f"{key}.pt")
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    torch.save(tens, tp)
    res["tensors"] = {"path": os.path.relpath(tp, RES), "sha256": sha256_file(tp)}
    res["provenance"] = provenance("SELECT", key)
    dump(res, os.path.join(RES, f"select_{key}.json"))


def phase_choose():
    from c15a.selection import choose
    res = {}
    for k in MODELS:
        p = os.path.join(RES, f"select_{k}.json")
        if not os.path.isfile(p):
            raise guards.GuardError(f"SELECT result missing for {k}; all A-stage candidates must be assessed first")
        res[k] = json.load(open(p, encoding="utf-8"))
    out = choose(res)
    out["provenance"] = provenance("CHOOSE")
    dump(out, os.path.join(RES, "choose_astage.json"))


def phase_freeze():
    ch = json.load(open(os.path.join(RES, "choose_astage.json"), encoding="utf-8"))
    if not ch["chosen"]:
        raise guards.GuardError("no (model, layer) passed W0-W5 on G_select: A-stage STOPS (ST-2); nothing to freeze")
    key, l = ch["chosen"]["model"], ch["chosen"]["layer"]
    sel = json.load(open(os.path.join(RES, f"select_{key}.json"), encoding="utf-8"))
    tp = os.path.join(RES, sel["tensors"]["path"])
    if sha256_file(tp) != sel["tensors"]["sha256"]:
        raise guards.GuardError("select tensors changed since SELECT")
    T = torch.load(tp, map_location="cpu", weights_only=False)
    fz = os.path.join(RES, "frozen")
    os.makedirs(fz, exist_ok=True)
    files = {f"Q_{key}_L{l}.pt": T["Q"][l], f"C_K_{key}.pt": T["C_K"], f"vgen_{key}.pt": T["vgen"]}
    hashes = {}
    for name, t in files.items():
        p = os.path.join(fz, name)
        torch.save(t, p)
        hashes[name] = sha256_file(p)
    pl = sel["per_layer"][str(l)]
    rec = {"model": key, "layer": l, "n_layers_band": sel["band_all"],
           "a_star": pl["dose"]["a_star"], "hbar": pl["hbar"], "sj_info": pl["sj"],
           "s_star": {"s_star": pl["w45"]["s_star"], "status": pl["w45"]["s_star_status"]},
           "vgen_dropped": T["vgen_dropped"], "frozen_tensors": hashes,
           "thresholds": {k: v for k, v in C.config_dict().items() if k.startswith(("W", "MATCH", "DOSE", "ABL",
                                                                                     "RAND", "GP", "SJ", "VGEN"))},
           "config_hash": C.config_hash(),
           "materials": {"g_confirm_sha256": split_manifest()["files"]["g_confirm.json"],
                         "g_select_sha256": split_manifest()["files"]["g_select.json"]},
           "selection": ch["chosen"], "replication_candidate": ch["replication_candidate"],
           "rule": "No other model may be confirmed after G_confirm is seen; on confirmation failure the A-stage STOPS.",
           "provenance": provenance("FREEZE", key)}
    dump(rec, guards.FREEZE_PATH)
    print("FREEZE written. Commit it (git add + commit) before CONFIRM.", flush=True)


def phase_confirm():
    from c15a.assays import AStage
    from c15a.selection import confirm_verdict
    guards.require_clean_tree()
    mats, fr = load_confirm()
    key, l = fr["model"], int(fr["layer"])
    lm, lens, _ = load_all(key)
    fz = os.path.join(RES, "frozen")
    frozen = {"Q": {l: torch.load(os.path.join(fz, f"Q_{key}_L{l}.pt"))},
              "C_K": torch.load(os.path.join(fz, f"C_K_{key}.pt")),
              "vgen": torch.load(os.path.join(fz, f"vgen_{key}.pt")).tolist(),
              "vgen_dropped": fr["vgen_dropped"], "hbar": {l: fr["hbar"]}, "sj_info": {l: fr["sj_info"]}}
    st = AStage(lm, lens, mats, frozen=frozen, layers=[l])
    t0 = time.time()
    res = st.run_all(frozen_layer_params={"a_star": fr["a_star"], "s_star": fr["s_star"]})
    res["seconds"] = time.time() - t0
    res["verdict"] = confirm_verdict(res)
    res["frozen_record_sha256"] = sha256_file(guards.FREEZE_PATH)
    res["provenance"] = provenance("CONFIRM", key)
    dump(res, guards.CONFIRM_PATH)


def phase_report():
    from c15a.report import write_report
    write_report(RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--model")
    a = ap.parse_args()
    guards.check_phase(a.phase)
    if a.phase in ("Z0", "BENCH", "SMOKE", "SELECT") and a.model not in MODELS:
        raise SystemExit(f"--model must be one of {MODELS}")
    torch.set_grad_enabled(False)
    {"Z0": lambda: phase_z0(a.model), "BENCH": lambda: phase_bench(a.model), "SMOKE": lambda: phase_smoke(a.model),
     "SELECT": lambda: phase_select(a.model), "CHOOSE": phase_choose, "FREEZE": phase_freeze,
     "CONFIRM": phase_confirm, "REPORT": phase_report}[a.phase]()


if __name__ == "__main__":
    main()
