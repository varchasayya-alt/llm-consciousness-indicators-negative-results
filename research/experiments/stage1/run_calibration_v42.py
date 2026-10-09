"""Stage-1 v4.2 (Option B1, sequential development; FINAL B1 revision) METHODS / FEASIBILITY ladder. Stores and
store-only diagnostics ONLY. (v4 ladder: commit 2bcef69, calibration_results_v4.json; v4.1: run_calibration_v4.py at
79b63d2, calibration_results_v41.json -- historical evidence only.)

NEVER trains or evaluates a scientific monitor or controller (unit-tested: this file may not import s1.monitors,
s1.controller or s1.pipeline). Rules: logs/calibration_plan.md "Revision v4.2" (committed before any v4.2 run);
D57-D60. Seeds: development 9071-9073; validation 9051-9053 ONCE after the complete recipe is FROZEN.

Store = Stage A (parametric route alone, memory off; covered fact sequences included per epoch with prob. p_rd;
E_A = 60 fixed, no epoch extension) -> Stage B (all P frozen; memory group only; NULL value fixed at zero;
injection cap kappa x median |hA|; E_B = 200 fixed).
V4.2-F0A  9071 x p_rd {0.05, 0.10, 0.20, 0.35, 0.50}, Stage A only: f0a_gates per p (Route-A acc on covered trained
          EV in [.30,.70], IQR >= 1, parametric-only >= .90, unknown <= .10, fluency >= .80); select A acc closest
          to .50 among eligible; DC-1 across the grid; DC-2 at the selected p. Failure -> B1 TERMINATED.
V4.2-F0B  9071, selected Stage-A checkpoint -> Stage B; f0b gates (store QC, F1 redundancy, covered >= .98, rescue,
          no-damage, INT-1, SC-1, SC-2, route-A equivalence). Failure -> B1 TERMINATED.
V4.2-F1C  9072, 9073 at the frozen p_rd: Stage A (f0a gates + DC-2) + Stage B (f0b gates). Failure -> STOP B1.
F2-F4, P2, FREEZE, VAL: unchanged from Revisions v4 / v4.1 (VAL applies the F0A per-p gates, DC-2 and F0B gates).
TRAINA / TRAINB: utilities that train and cache one stage (no evaluation), for parallel processes.
Exit code 3 = a pre-declared stop condition fired.
Usage: python run_calibration_v42.py --phase F0A | F0B | F1C | F234 | P2 | FREEZE | VAL | TRAINA | TRAINB
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s1 import world as W  # noqa: E402
from s1.config import derive_seed, load_config  # noqa: E402
from s1.interventions import generic40, identifiability, item_pre_covariates  # noqa: E402
from s1.memstore import (build_dual_store, choose_donors, delete_slots, load_state_compat,  # noqa: E402
                         parametric_interference, train_stage_A, train_stage_B, transplant)
from s1.store import name_fluency, probe, site_stats  # noqa: E402
from s1.v4qc import (audit, base_correct_ev, dc1_check, delete_qc, draw_v4_sets, f0a_gates,  # noqa: E402
                     stageA_report, stageB_report)

DEV_SEEDS = [9071, 9072, 9073]          # v4.2 development (9041-9043 v4, 9061 v4.1: history; 9062-9063 retired unused)
VAL_SEEDS = [9051, 9052, 9053]
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "raw", "calibration")
STORE_DIR = os.path.join(HERE, "_calib")
RES = os.path.join(OUT, "calibration_results_v42.json")
STOP = 3


def log(msg):
    print(msg, flush=True)


def load_res():
    return json.load(open(RES)) if os.path.exists(RES) else {}


def save_res(res):
    os.makedirs(OUT, exist_ok=True)
    tmp = RES + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, RES)


def world_of(cfg, seed):
    return W.make_world(cfg["world"], derive_seed(seed, "world"))


def stage_path(stage, seed, p_rd):
    return os.path.join(STORE_DIR, f"v42_{stage}_{seed}_prd{float(p_rd)}.pt")


def _save(path, model, meta, arrays=None):
    os.makedirs(STORE_DIR, exist_ok=True)
    if arrays:
        np.savez(path[:-3] + "_arrays.npz", **arrays)
    json.dump(meta, open(path[:-3] + "_meta.json", "w"), indent=1)
    torch.save(model.state_dict(), path)                      # written last: its existence marks a complete stage


def _load(cfg, world, path):
    m = build_dual_store(cfg["store"], cfg["memory"], world)
    load_state_compat(m, torch.load(path))
    m.eval()
    meta = json.load(open(path[:-3] + "_meta.json"))
    arr = path[:-3] + "_arrays.npz"
    if os.path.exists(arr):
        meta["arrays"] = dict(np.load(arr))
    return m, meta


def get_stage_A(cfg, seed, p_rd, eval_every=0):
    """Train (or load) the Stage-A network for (seed, p_rd). E_A is fixed by config (no extension rule)."""
    world = world_of(cfg, seed)
    E = cfg["v4"]["stage_A_epochs"]
    path = stage_path("A", seed, p_rd)
    if os.path.exists(path):
        m, meta = _load(cfg, world, path)
        assert meta["epochs"] == E, (path, meta["epochs"], E)
        return world, m, meta
    m, st = train_stage_A(world, cfg["store"], cfg["memory"], derive_seed(seed, "storeA"), E, p_rd,
                          derive_seed(seed, "stageA_dose"), eval_every=eval_every, log=log if eval_every else None)
    meta = {"seed": seed, "p_rd": float(p_rd), "epochs": E, "threads": torch.get_num_threads(), "minutes": st["minutes"],
            "curve": st["curve"], "epoch_sizes": st["epoch_sizes"], "P_hash": st["P_hash"], "M_hash": st["M_hash"]}
    _save(path, m, meta, {"presentations": st["presentations"]})
    meta["arrays"] = {"presentations": st["presentations"]}
    return world, m, meta


def get_store(cfg, seed, p_rd, E=None, eval_every=0):
    """Train (or load) the complete v4.2 store (Stage A -> Stage B). Returns (world, store, meta). E is ignored
    (kept for call compatibility with the unchanged F2-F4 / P2 code): both stage lengths are fixed by config."""
    world, mA, metaA = get_stage_A(cfg, seed, p_rd)
    EB = cfg["v4"]["stage_B_epochs"]
    path = stage_path("B", seed, p_rd)
    if os.path.exists(path):
        m, meta = _load(cfg, world, path)
        assert meta["epochs_B"] == EB and meta["P_hash"] == metaA["P_hash"], (path, meta)
        return world, m, meta
    m, st = train_stage_B(mA, world, cfg["store"], cfg["memory"], derive_seed(seed, "stageB"), EB,
                          eval_every=eval_every, log=log if eval_every else None)
    assert st["P_hash"] == metaA["P_hash"]
    meta = {"seed": seed, "p_rd": float(p_rd), "epochs_B": EB, "threads": torch.get_num_threads(),
            "minutes": st["minutes"], "curve": st["curve"], "P_hash": st["P_hash"], "M_hash_after": st["M_hash_after"],
            "hA_reference_median": st["hA_reference_median"], "injection_cap": st["injection_cap"]}
    _save(path, m, meta)
    return world, m, meta


def _clean(meta):
    return None if meta is None else {k: v for k, v in meta.items() if k != "arrays"}


# ---------------------------------------------------------------- F0A (Stage A dose feasibility)
def stageA_eval(cfg, seed, p):
    world, mA, meta = get_stage_A(cfg, seed, p)
    v4 = cfg["v4"]
    rep = stageA_report(mA, world, meta["arrays"]["presentations"], p, v4["stage_A_epochs"],
                        dc2_alpha=v4["F1"]["dose"]["dc2_alpha"])
    ok, g = f0a_gates(rep, v4)
    log(f"F0A {seed} p_rd={p}: eligible={ok} {g} | A acc {rep['A_acc']:.3f} IQR {rep['A_margin_iqr']:.2f} margin mean "
        f"{rep['A_margin_mean']:.2f} median {rep['A_margin_median']:.2f} | presentations {rep['presentations_mean_pop']:.2f} "
        f"(exp {rep['expected_presentations']:.1f}) rho {rep['spearman_count_vs_A_margin']:.3f} "
        f"(p {rep['spearman_p_one_sided']:.2g}) | exposure hi/lo {rep['A_acc_exposure_high']:.3f}/"
        f"{rep['A_acc_exposure_low']:.3f} | param-only {rep['param_only_trained_acc']:.3f} unknown "
        f"{rep['unknown_acc']:.3f} fluency {rep['fluency_auroc_high_vs_low']:.3f}")
    return {"report": rep, "gates": g, "eligible": ok, "train": _clean(meta)}


def phase_F0A(cfg):
    res = load_res()
    v4 = cfg["v4"]
    rows = {}
    for p in v4["F1"]["route_dropout_grid"]:
        rows[str(float(p))] = stageA_eval(cfg, DEV_SEEDS[0], p)
        res.setdefault("F0A", {})["grid_%d" % DEV_SEEDS[0]] = rows
        save_res(res)
    d = v4["F1"]["dose"]
    dc1 = dc1_check({float(p): rows[p]["report"]["A_acc"] for p in rows}, d["dc1_pairwise_decrease_max"],
                    d["dc1_range_min"])
    res["F0A"]["DC1"] = dc1
    ok = [p for p in rows if rows[p]["eligible"]]
    if not ok:
        res["F0A"].update(selected_p_rd=None, pass_=False, stop="F0A: no p_rd in the fixed grid is eligible (K-2)")
        save_res(res)
        log(f"F0A: no eligible p_rd -> B1 TERMINATED (DC-1 {dc1})")
        return STOP
    sel = min(ok, key=lambda p: abs(rows[p]["report"]["A_acc"] - 0.5))
    dc2 = rows[sel]["report"]["DC2_pass"]
    res["F0A"].update(selected_p_rd=float(sel), DC2_selected=dc2)
    if not (dc1["DC1_pass"] and dc2):
        res["F0A"].update(pass_=False, stop="F0A: dose response not coherent (DC-1/DC-2; K-2d)")
        save_res(res)
        log(f"F0A: p_rd={sel} eligible but dose coherence fails (DC-1 {dc1}; DC-2 {dc2}) -> B1 TERMINATED")
        return STOP
    res["F0A"]["pass_"] = True
    save_res(res)
    log(f"F0A pass: selected p_rd={sel} (DC-1 {dc1}; DC-2 pass)")
    return 0


# ---------------------------------------------------------------- F0B (Stage B integration feasibility)
def stageB_eval(cfg, seed, p, eval_every=0):
    world, mA, _ = get_stage_A(cfg, seed, p)
    _, mB, meta = get_store(cfg, seed, p, eval_every=eval_every)
    rep = stageB_report(mB, mA, world, derive_seed(seed, "ablation", p), cfg["v4"])
    ab, sc = rep["ablation"], rep["scale"]
    log(f"F0B {seed} p_rd={p}: pass={rep['pass']} gates={rep['gates']} | trained {rep['store_qc']['trained_fact_acc']:.4f} "
        f"covered {rep['covered_integrated_acc']:.4f} own-slot {rep['store_qc']['own_slot_attention_mean']:.3f} | A acc "
        f"{ab['A_acc_covered_bc']:.3f} IQR {ab['A_margin_iqr_covered_bc']:.2f} M1 {ab['M1_acc_covered_bc']:.3f} M2 "
        f"{ab['M2_acc_covered']:.3f} | tiers { {k: (v['n'], v['integrated_acc']) for k, v in rep['tiers'].items()} } | "
        f"param-only {rep['param_only_acc_stageA']:.4f}->{rep['param_only_acc_integrated']:.4f} | INT-1 rho "
        f"{rep['INT1_spearman_stageA_vs_integrated_margin']:.3f} | scale |u|/|hA| cov {sc['u_over_hA_median_covered']:.2f} "
        f"null {sc['u_over_hA_median_param_only']:.4f} at-cap {sc['frac_covered_at_cap']:.2f} |W_o| {sc['W_o_fro']:.1f} "
        f"| routeA==stageA {rep['routeA_equivalent_to_stageA']}")
    return {"report": rep, "pass": rep["pass"], "train": _clean(meta)}


def phase_F0B(cfg):
    res = load_res()
    assert res.get("F0A", {}).get("pass_"), "F0B only after F0A passes"
    p = res["F0A"]["selected_p_rd"]
    r = stageB_eval(cfg, DEV_SEEDS[0], p, eval_every=20)
    res["F0B"] = r
    save_res(res)
    if not r["pass"]:
        log("F0B: memory integration on the frozen parametric network fails -> B1 TERMINATED")
        return STOP
    return 0


# ---------------------------------------------------------------- F1C (replication on 9072, 9073)
def phase_F1C(cfg):
    res = load_res()
    assert res.get("F0B", {}).get("pass"), "F1C only after F0A and F0B pass"
    p = res["F0A"]["selected_p_rd"]
    conf = {}
    for s in DEV_SEEDS[1:]:
        a = stageA_eval(cfg, s, p)
        b = stageB_eval(cfg, s, p)
        conf[str(s)] = {"F0A": a, "F0B": b,
                        "confirmed": bool(a["eligible"] and a["report"]["DC2_pass"] and b["pass"])}
        res.setdefault("F1C", {})["seeds"] = conf
        save_res(res)
    res["F1C"]["pass"] = bool(all(v["confirmed"] for v in conf.values()))
    res["F1"] = {"selected_p_rd": p, "pass": res["F1C"]["pass"]}           # key read by the unchanged later phases
    save_res(res)
    log(f"F1C p_rd={p}: pass={res['F1C']['pass']} ({ {s: v['confirmed'] for s, v in conf.items()} })")
    return 0 if res["F1C"]["pass"] else STOP


# ---------------------------------------------------------------- F2-F4 (unchanged gates)
def intervened(cfg, seed, p_rd, E=None):
    """Intact store, its context, and the intervened copy (T-DELETE(X) + Y same-answer value transplant)."""
    world, A, _ = get_store(cfg, seed, p_rd)
    MT = W.items_where(world, split=W.MT)
    mu, sig = site_stats(probe(A, world, MT)["states"])
    sets = draw_v4_sets(world, A, cfg["v4"]["sets"], derive_seed(seed, "setsV4"))
    from s1.v4qc import v4_categories
    cats, _ = v4_categories(A, world, sets, cfg["v4"]["qc"]["near_key_fraction"])
    excl = np.concatenate([sets["X"], sets["Y"]] + [c for c in cats.values() if len(c)])
    donors = choose_donors(world, sets["Y"], excl, derive_seed(seed, "donors"))
    has = donors[:, 0] >= 0
    sets["Y"], donors = sets["Y"][has], donors[has]
    post = transplant(delete_slots(A, world, sets["X"]), world, sets["Y"], donors)
    flu_sd = float(name_fluency(A, world, world.names).std())
    return dict(world=world, A=A, post=post, mu=mu, sig=sig, sets=sets, donors=donors, flu_sd=flu_sd,
                n_Y_without_donor=int((~has).sum()))


def ident_ok(list_of_idf, cfg):
    t = cfg["identifiability"]
    med = {k: float(np.median([d[k] for d in list_of_idf])) for k in ("r2_gen", "r2_pre", "r2_joint")}
    return bool(med["r2_gen"] < t["r2_gen_max"] and med["r2_pre"] < t["r2_pre_max"] and med["r2_joint"] < t["r2_joint_max"]), med


def evaluate_F234(cfg, seeds, p_rd, tag):
    per = {}
    for s in seeds:
        c = intervened(cfg, s, p_rd)
        q, aux = delete_qc(c["A"], c["post"], c["world"], c["sets"], c["sig"], c["mu"], cfg["v4"]["qc"],
                           derive_seed(s, "v4qc"), c["flu_sd"])
        au = audit(c["A"], c["post"], c["world"], c["sets"], aux, derive_seed(s, "audit"))
        q["audit"] = au
        q["n_Y_without_donor"] = c["n_Y_without_donor"]
        per[str(s)] = q
        log(f"[{tag}] seed {s}: F2 {q['gates_F2']} | Y {q['gates_Y']} | F3 {q['gates_F3']} | lost {q['X_lost_frac']:.2f} "
            f"IQR {q['C_post_X_iqr']:.2f} tiers {q['tiers']} expo {q['exposure']} | load {q['retention_load_max']:.2f} "
            f"disp {max(q['displacement_ratio'].values()):.3f} leak {max(q['leakage'].values()):.4f} | ident {q['identifiability']} "
            f"| audit C_post {au['C_post']}")
    out = {"seeds": per}
    out["F2_pass"] = all(v["pass_F2"] for v in per.values())
    out["Y_pass"] = all(v["pass_Y"] for v in per.values())
    out["F3_pass"] = all(v["pass_F3"] for v in per.values())
    idok, med = ident_ok([v["identifiability"] for v in per.values()], cfg)
    out["identifiability_median"] = med
    out["kill_bookkeeping"] = any(v["audit"]["C_post"]["book"] >= cfg["v4"]["qc"]["bookkeeping_r2_kill"] for v in per.values())
    out["F4_pass"] = bool(idok and not out["kill_bookkeeping"])
    out["F6_alt_all"] = all(v["binary_secondary_alt"]["feasible"] for v in per.values())
    return out


def ladder_verdict(out):
    for key, name in (("F2_pass", "V4-F2 locality (K-3)"), ("Y_pass", "Y negative control (K-7: stop/report)"),
                      ("F3_pass", "V4-F3 outcome diversity (K-4)"), ("F4_pass", "V4-F4 identifiability / trivial cue (K-5/K-6)")):
        if not out[key]:
            return name
    return None


def phase_F234(cfg):
    res = load_res()
    assert res.get("F1", {}).get("pass"), "F2-F4 only after F0A, F0B and F1C pass"
    p = res["F1"]["selected_p_rd"]
    out = evaluate_F234(cfg, DEV_SEEDS, p, "dev")
    out["stopped_at"] = ladder_verdict(out)
    res["F234"] = out
    save_res(res)
    log(f"F2-F4: F2 {out['F2_pass']} Y {out['Y_pass']} F3 {out['F3_pass']} F4 {out['F4_pass']} (ident {out['identifiability_median']}, "
        f"book kill {out['kill_bookkeeping']}) -> stopped_at={out['stopped_at']}")
    return 0 if out["stopped_at"] is None else STOP


# ---------------------------------------------------------------- P2 (parametric-route interference; unchanged)
def p2_eval(cfg, seed, p_rd, icfg):
    world, A, _ = get_store(cfg, seed, p_rd)
    MT = W.items_where(world, split=W.MT)
    mu, sig = site_stats(probe(A, world, MT)["states"])
    _, bc_cov, bc_par = base_correct_ev(A, world)
    t0 = time.perf_counter()
    m, trace = parametric_interference(A, world, icfg, derive_seed(seed, "interfP2"))
    p0, p1 = probe(A, world, bc_par), probe(m, world, bc_par)
    W0, Cp = item_pre_covariates(A, world, bc_par, p0)
    G = generic40(p0["states"], p1["states"], mu, sig)
    idf, _ = identifiability(p1["margin"] - p0["margin"], p0["states"], W0, Cp, G, derive_seed(seed, "p2id"), strata=W0[:, 1])
    from s1.store import answer_stats
    cov_after = answer_stats(m, world, bc_cov)
    q = np.quantile(p1["margin"], [0.25, 0.75])
    return {"lost_frac_param_only": float((~p1["correct"]).mean()), "C_post_iqr": float(q[1] - q[0]),
            "covered_lost_frac": float((~cov_after["correct"]).mean()), "identifiability": idf,
            "steps_used": trace[-1]["step"] if trace else None, "final_new_acc": trace[-1]["stop_metric"] if trace else None,
            "n_population": int(len(bc_par)), "seconds": time.perf_counter() - t0}


def phase_P2(cfg):
    res = load_res()
    assert res.get("F234", {}).get("stopped_at", "missing") is None, "P2 only after F0-F4 pass"
    p = res["F1"]["selected_p_rd"]
    P = cfg["v4"]["P2"]
    rows = []
    for lr in P["lr_grid"]:
        icfg = dict(P, lr=lr)
        row = {"lr": lr, "seeds": {}}
        for s in DEV_SEEDS:
            row["seeds"][str(s)] = p2_eval(cfg, s, p, icfg)
            v = row["seeds"][str(s)]
            log(f"P2 lr={lr} seed {s}: lost {v['lost_frac_param_only']:.3f} covered lost {v['covered_lost_frac']:.3f} "
                f"steps {v['steps_used']} ident {v['identifiability']}")
        S = list(row["seeds"].values())
        lo, hi = P["lost_select_range"]
        row["mean_lost"] = float(np.mean([v["lost_frac_param_only"] for v in S]))
        row["in_range"] = all(lo <= v["lost_frac_param_only"] <= hi for v in S)
        row["identifiable"], row["identifiability_median"] = ident_ok([v["identifiability"] for v in S], cfg)
        row["eligible"] = bool(row["in_range"] and row["identifiable"])
        rows.append(row)
        res["P2"] = {"grid": rows}
        save_res(res)
    ok = [r for r in rows if r["eligible"]]
    sel = min(ok, key=lambda r: abs(r["mean_lost"] - P["target_lost"])) if ok else None
    res["P2"]["selected"] = None if sel is None else {"lr": sel["lr"], "max_steps": int(math.ceil(1.5 * max(
        v["steps_used"] for v in sel["seeds"].values()) / 100.0)) * 100}
    save_res(res)
    log(f"P2 selected: {res['P2']['selected']}")
    return 0 if sel is not None else STOP


# ---------------------------------------------------------------- FREEZE + VAL
def phase_FREEZE(cfg):
    """Record the complete development recipe (only if F0A, F0B, F1C, F2-F4 and P2 all passed)."""
    import subprocess
    res = load_res()
    ok = bool(res.get("F0A", {}).get("pass_") and res.get("F0B", {}).get("pass") and res.get("F1", {}).get("pass")
              and res.get("F234", {}).get("stopped_at", "x") is None and res.get("P2", {}).get("selected"))
    if not ok:
        log("FREEZE refused: the development recipe is incomplete")
        return STOP
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    res["FROZEN"] = {"stage_A_epochs": cfg["v4"]["stage_A_epochs"], "stage_B_epochs": cfg["v4"]["stage_B_epochs"],
                     "p_rd": res["F1"]["selected_p_rd"], "P2": res["P2"]["selected"], "code_commit": head,
                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_res(res)
    log(f"FROZEN recipe: {res['FROZEN']}")
    return 0


def phase_VAL(cfg):
    """Frozen v4.2 recipe ONCE on 9051-9053: Stage A (F0A gates + DC-2), Stage B (F0B gates), F2-F4, P2."""
    res = load_res()
    assert res.get("FROZEN") and "VAL" not in res, "VAL runs once, and only on the frozen recipe"
    assert res["FROZEN"]["stage_A_epochs"] == cfg["v4"]["stage_A_epochs"] and res["FROZEN"]["stage_B_epochs"] == cfg["v4"]["stage_B_epochs"]
    v4 = cfg["v4"]
    p = res["FROZEN"]["p_rd"]
    out = {"recipe": res["FROZEN"], "F01": {}}
    for s in VAL_SEEDS:
        a = stageA_eval(cfg, s, p)
        b = stageB_eval(cfg, s, p)
        out["F01"][str(s)] = {"F0A": a, "F0B": b, "eligible": bool(a["eligible"] and a["report"]["DC2_pass"] and b["pass"])}
        log(f"VAL {s}: F0A/F0B eligible={out['F01'][str(s)]['eligible']}")
    f234 = evaluate_F234(cfg, VAL_SEEDS, p, "val")
    out["F234"] = f234
    out["stopped_at"] = ladder_verdict(f234)
    icfg = dict(v4["P2"], **res["FROZEN"]["P2"])
    out["P2"] = {str(s): p2_eval(cfg, s, p, icfg) for s in VAL_SEEDS}
    lo, hi = v4["P2"]["lost_select_range"]
    p2_ok = all(lo <= v["lost_frac_param_only"] <= hi for v in out["P2"].values())
    p2_id, p2_med = ident_ok([v["identifiability"] for v in out["P2"].values()], cfg)
    out["P2_pass"] = bool(p2_ok and p2_id)
    out["P2_identifiability_median"] = p2_med
    out["PASS"] = bool(all(v["eligible"] for v in out["F01"].values()) and out["stopped_at"] is None and out["P2_pass"])
    res["VAL"] = out
    save_res(res)
    log(f"VAL PASS={out['PASS']} (F0/F1 {[v['eligible'] for v in out['F01'].values()]}, F2-F4 stopped_at={out['stopped_at']}, P2 {out['P2_pass']})")
    return 0 if out["PASS"] else STOP


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "stage1_config.yaml"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prd", type=float)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
    if a.phase in ("TRAINA", "TRAINB"):
        assert a.seed in DEV_SEEDS + VAL_SEEDS, "TRAIN: development or validation seeds only"
        if a.seed in VAL_SEEDS:
            assert load_res().get("FROZEN"), "validation stores only after the recipe is frozen"
        t0 = time.perf_counter()
        if a.phase == "TRAINA":
            get_stage_A(cfg, a.seed, a.prd, eval_every=10)
        else:
            get_store(cfg, a.seed, a.prd, eval_every=20)
        log(f"{a.phase} {a.seed} p_rd={a.prd}: {(time.perf_counter() - t0) / 60:.1f} min, threads {torch.get_num_threads()}")
        sys.exit(0)
    phases = {"F0A": phase_F0A, "F0B": phase_F0B, "F1C": phase_F1C, "F234": phase_F234, "P2": phase_P2,
              "FREEZE": phase_FREEZE, "VAL": phase_VAL}
    sys.exit(phases[a.phase](cfg) or 0)
