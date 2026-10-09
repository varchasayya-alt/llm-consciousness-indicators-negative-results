"""C16 Stage 0 / Stage 1 runner (prereg c16_s0s1_preregistration_FROZEN.md).

Phases: S0 (cached Qwen2.5-0.5B-Instruct, forward only + E4 timing probe), S1B, S1C, S1A, REPORT.
Every phase verifies the frozen hashes (prereg, thresholds JSON, manifest) and a clean tree for the package/memo.
No phase trains anything on the HF model (lm_hf has no training path; E4 asserts no update).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c16 import config as C          # noqa: E402
from c16 import materials as M       # noqa: E402

FROZEN = os.path.join(C.MATERIALS, "frozen_hashes.json")
CLEAN = ("research/experiments/c16", "research/memo/c16_s0s1_preregistration_FROZEN.md",
         "research/memo/c16_s0s1_thresholds_FROZEN.json")


class GuardError(RuntimeError):
    pass


def frozen_hashes():
    return {"prereg": C.sha256_file(C.PREREG_PATH), "thresholds": C.sha256_file(C.THRESHOLDS_PATH),
            "manifest": C.sha256_file(C.MANIFEST_PATH)}


def verify(phase, allow_dirty=False):
    if phase not in C.PHASES_AUTHORIZED:
        raise GuardError(f"phase {phase} not authorized (S0/S1 only; no Stage-2 training)")
    rec = json.load(open(FROZEN, encoding="utf-8"))
    now = frozen_hashes()
    for k, v in rec.items():
        if now.get(k) != v:
            raise GuardError(f"frozen hash mismatch for {k}")
    man = json.load(open(C.MANIFEST_PATH, encoding="utf-8"))
    if man != M.manifest(write=False):
        raise GuardError("manifest does not match the frozen split rule")
    if not allow_dirty:
        out = subprocess.run(["git", "status", "--porcelain", "--", *CLEAN], cwd=C.REPO, capture_output=True,
                             text=True).stdout.strip()
        if out:
            raise GuardError(f"dirty tree for frozen paths:\n{out}")


def log_to(path):
    f = open(path, "a", encoding="utf-8")

    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return log


def dump(name, obj):
    os.makedirs(C.RESULTS, exist_ok=True)
    path = os.path.join(C.RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    return path


# ---------------------------------------------------------------------------------------------- phases
def phase_s0(log):
    import torch
    from c16 import stage0
    from c16.lm_hf import HFSubject
    from c16.subject import EpSpec, latent
    tr, _, _ = M.split_values()
    hA, hB = M.halves(tr)
    main = M.build_instances(tr, C.MAT["instances_main"], C.SEEDS["s0_sampling"], tag="main")
    dec = M.build_instances(tr, C.MAT["instances_decod"], C.SEEDS["s0_sampling"] + 1, tag="dec")
    log(f"S0: {len(main)} main instances, {len(dec)} decod instances; X_train={len(tr)}")
    results = {"hashes": frozen_hashes(), "config_hash": C.config_hash()}
    for fmt in ("F1", "F2"):
        sub = HFSubject(fmt=fmt)
        log(f"S0 format {fmt}: model {C.MODEL_NAME} layers={sub.n_layers} d={sub.d_model} "
            f"threads={torch.get_num_threads()}")
        # M0 engineering
        probe = [latent(i.name, i.producer, i.args) for i in main[:8]]
        eng = {"E2_zero_patch_maxabs": sub.zero_patch_identity(probe),
               "E2_hook_vs_hidden_maxabs": sub.hook_matches_hidden_states(probe[0], layer=5)}
        log(f"  E2 {eng}")
        if eng["E2_zero_patch_maxabs"] != 0.0 or eng["E2_hook_vs_hidden_maxabs"] != 0.0:
            raise GuardError(f"engineering identity check failed: {eng}")
        tok0, t0 = sub.n_tokens, time.time()
        r = stage0.run_stage0(sub, main, dec, (hA, hB), log=log)
        eng["E3_tokens"] = sub.n_tokens - tok0
        eng["E3_wall_s"] = time.time() - t0
        eng["E3_tok_per_s_forward"] = (sub.n_tokens - tok0) / max(1e-9, sub.t_forward)
        # E4 timing probe (no optimizer step; asserts no update)
        eps = [EpSpec(latent(i.name, i.producer, i.args), "lookup", tuple(i.table), {"own": i.x}) for i in main[:8]]
        eng["E4"] = sub.backward_timing(eps)
        log(f"  E3/E4 {eng}")
        r["engineering"] = eng
        r["format"] = fmt
        results[fmt] = r
        dump(f"s0_{fmt}.json", r)
        if not r.get("format_fallback_needed"):
            results["final_format"] = fmt
            break
        log(f"S0: format {fmt} failed V0/V1a/b/d -> " + ("trying F2 (pre-declared fallback)" if fmt == "F1" else "STOP"))
        del sub
    else:
        results["final_format"] = None
    fin = results.get(results["final_format"]) if results["final_format"] else results["F2"]
    results["summary"] = {
        "final_format": results["final_format"],
        "core_instrument_valid": bool(fin.get("core_instrument_valid")) if results["final_format"] else False,
        "phase_diagram_measurable": bool(fin.get("phase_diagram_measurable")) if results["final_format"] else False,
        "V1": fin.get("V1"), "V2": fin.get("V2"), "W": fin.get("W"), "L_w": fin.get("M1", {}).get("L_w"),
        "l_o": fin.get("M3", {}).get("l_o"), "read_layers": fin.get("M3", {}).get("read_layers"),
        "rX_geomean": fin.get("M4", {}).get("rX_geomean"), "rB_geomean": fin.get("M4", {}).get("rB_geomean"),
    }
    dump("s0_final.json", results)
    log(f"S0 summary: {results['summary']}")


def phase_s1b(log):
    from c16 import sim
    r = sim.s1b_rank_recovery(C.SEEDS["s1b"])
    dump("s1b.json", r)
    log(f"S1b pass={r['pass']} " + " ".join(f"{k}:{v['r_star']}" for k, v in r.items() if isinstance(v, dict)))


def phase_s1c(log):
    from c16 import s1c
    r = s1c.run_s1c(log=log, out_dir=C.RESULTS)
    dump("s1c.json", r)
    log(f"S1c done: S1c-1={r.get('S1c_1_pass')} S1c-2={r.get('S1c_2', {}).get('pass')} "
        f"sigma_seed={r.get('sigma_seed_logit', {}).get('median')}")


def phase_s1a(log):
    from c16 import sim
    s0 = json.load(open(os.path.join(C.RESULTS, "s0_final.json"), encoding="utf-8"))
    s1c = json.load(open(os.path.join(C.RESULTS, "s1c.json"), encoding="utf-8"))
    cfg = C.S1["S1a"]
    sig_s = s1c.get("sigma_seed_logit", {}).get("median")
    sig_s_list = [sig_s] if sig_s is not None else [0.5, 1.0]
    fmt = s0["summary"]["final_format"]
    fin = s0.get(fmt) if fmt else None
    p_nc = 0.05
    if fin and "M4" in fin:
        vals = [v["P_nc_sw"] for v in fin["M4"]["swap"].values() if v["P_nc_sw"] is not None]
        p_nc = sum(vals) / len(vals) if vals else 0.05
    rB = {"S1": 1.0, "S2": 2.0, "S4": 4.0, "Spm": 0.5, "Sparam": float("inf")}
    rb_source = "default"
    if fin and fin.get("phase_diagram_measurable"):
        rx = fin["M4"]["rX_geomean"]
        g = fin["M4"]["rB_geomean"]
        rB = {"S1": g["B1"] / rx, "S2": g["B2"] / rx, "S4": g["B4"] / rx, "Spm": g["Bpm"] / rx, "Sparam": float("inf")}
        rb_source = "stage0"
    out = {"inputs": {"sigma_seed": sig_s_list, "p_nc": p_nc, "rB_over_rX": rB, "rB_source": rb_source}}
    # ---- core classification: complete-gate FPR with the pre-declared margin rule
    gates = {"h1_lb": 0.10, "h2_lb": 0.0, "h3_lb": 0.0, "h4_lb": 0.0, "h5_lb": 0.0}
    core = []
    for step in range(7):
        rows = []
        for ss in sig_s_list:
            for sx in cfg["sigma_X"]:
                r = sim.s1a_core(C.SEEDS["s1a"] + step, ss, sx, n_s=3, p_nc=p_nc, reps=cfg["sim_reps"],
                                 B=cfg["boot_reps"], gates=gates)
                rows.append(r)
                log(f"S1a core step {step} sig_s={ss} sig_x={sx}: overclaim={r['max_overclaim_fpr']:.3f} "
                    f"power={r['min_power_true_class']:.3f}")
        core.append({"gates": dict(gates), "rows": rows})
        if max(r["max_overclaim_fpr"] for r in rows) <= cfg["fpr_max"]:
            break
        for k in ("h1_lb", "h3_lb", "h4_lb", "h5_lb"):
            gates[k] = round(gates[k] + cfg["margin_step"], 4)
    out["core"] = core
    out["core_final_gates"] = core[-1]["gates"]
    out["core_final_fpr"] = max(r["max_overclaim_fpr"] for r in core[-1]["rows"])
    out["core_final_power"] = min(r["min_power_true_class"] for r in core[-1]["rows"])
    # ---- H-CD
    e4 = fin["engineering"]["E4"]["sec_per_example"] if fin and "engineering" in fin else None
    run_h = None if e4 is None else (20000 * e4 + 2000 * e4 / 3) / 3600.0
    hcd = []
    for n_s in cfg["n_seeds_grid"]:
        for m in cfg["items_per_cell_grid"]:
            rows = [sim.s1a_hcd(C.SEEDS["s1a"] + 100, ss, sx, rB, n_s, m, p_nc=p_nc, margin=cfg["hcd_margin"])
                    for ss in sig_s_list for sx in cfg["sigma_X"]]
            pw = min(r["min_power"] for r in rows)
            wr = max(r["max_wrong_account"] for r in rows)
            cost_min = None if run_h is None else 4 * n_s * run_h
            cost_full = None if run_h is None else 25 * n_s * run_h
            hcd.append({"n_s": n_s, "m_cell": m, "min_power": pw, "max_wrong_account": wr,
                        "cpu_h_4cells": cost_min, "cpu_h_full_grid": cost_full, "rows": rows})
            log(f"S1a H-CD n_s={n_s} m={m}: power={pw:.3f} wrong={wr:.3f} cpu_h(4 cells)={cost_min}")
    ok = [h for h in hcd if h["min_power"] >= cfg["power_min"] and h["max_wrong_account"] <= cfg["fpr_max"]]
    best = min(ok, key=lambda h: (h["cpu_h_4cells"] or 0, h["n_s"], h["m_cell"])) if ok else None
    out["hcd"] = hcd
    out["hcd_minimal"] = None if best is None else {k: best[k] for k in ("n_s", "m_cell", "min_power",
                                                                       "max_wrong_account", "cpu_h_4cells",
                                                                       "cpu_h_full_grid")}
    out["stage2_run_cpu_h_estimate"] = run_h
    measurable = bool(fin and fin.get("phase_diagram_measurable"))
    within = best is not None and best["cpu_h_4cells"] is not None and best["cpu_h_4cells"] <= cfg["hcd_budget_cpu_h"]
    out["hcd_central_recommended"] = bool(measurable and within)
    dump("s1a.json", out)
    log(f"S1a done: core gates={out['core_final_gates']} fpr={out['core_final_fpr']:.3f} "
        f"power={out['core_final_power']:.3f}; H-CD minimal={out['hcd_minimal']}; central={out['hcd_central_recommended']}")


PHASES = {"S0": phase_s0, "S1B": phase_s1b, "S1C": phase_s1c, "S1A": phase_s1a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES) + ["REPORT"])
    a = ap.parse_args()
    verify(a.phase)
    os.makedirs(os.path.join(C.RESULTS, "logs"), exist_ok=True)
    log = log_to(os.path.join(C.RESULTS, "logs", f"{a.phase}.log"))
    log(f"phase {a.phase} start; config_hash={C.config_hash()}")
    if a.phase == "REPORT":
        from c16 import report
        report.write(log)
    else:
        PHASES[a.phase](log)
    log(f"phase {a.phase} done")


if __name__ == "__main__":
    main()
