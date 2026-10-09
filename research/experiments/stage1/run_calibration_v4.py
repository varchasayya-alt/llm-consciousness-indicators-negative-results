"""Stage-1 v4.1 (Option B1 + gradient-isolated learning, D52) METHODS / FEASIBILITY ladder. Stores + store-only
diagnostics ONLY. (The v4 ladder that stopped at V4-F1 K-2 is reproducible from commit 2bcef69; its results are in
calibration_results_v4.json and are historical evidence only.)

NEVER trains or evaluates a scientific monitor or controller (unit-tested: this file may not import s1.monitors,
s1.controller or s1.pipeline). Rules: logs/calibration_plan.md "Revision v4.1" (committed before any v4.1 run);
D46-D48, D52-D54. Seeds: development 9061-9063 (all selection); validation 9051-9053 used ONCE after the complete
recipe (F0-F4 + P2) is frozen. 9041-9043 are not used.

V4.1-F0   9061, p_rd = 0.35, E = 60: dual-store QC (trained >= .98, unknown <= .10, fluency AUROC >= .80, own-slot
          attention >= .90). If trained acc < .98 at 60: C1 rule on 9061 (200 epochs, E = min(200, ceil10(1.5 E99))).
          Gradient-isolation invariants are asserted inside training (first batch of every epoch).
V4.1-F1G  9061 x p_rd {0.05, 0.10, 0.20, 0.35, 0.50} (fixed): ablations A / M1 / M2 / integrated + dose report.
          F1-eligible: store QC; A acc on covered base-correct EV in [.30,.70] with IQR(A margin) >= 1 nat; memory
          sufficient (M1 >= .90 OR M2 >= .90; one-sided pass = reported discrepancy); parametric-only trained facts
          integrated acc >= .90. Select A acc closest to 0.5 among eligible. Then dose coherence (pre-declared):
          DC-1 across the grid (pairwise decrease <= .05 and acc(.50) - acc(.05) >= .20) and DC-2 at the selected
          p_rd (Spearman rho(route-drop count, A margin) > 0, one-sided p < .01). None eligible -> K-2 STOP;
          eligible but DC-1 or DC-2 fails -> K-2d STOP (p_rd does not control backup).
V4.1-F1C  9062, 9063 at the frozen p_rd: same F1 gates + DC-2. Any failure -> STOP.
V4.1-F2..F4, P2, VAL: unchanged from Revision v4 (VAL adds DC-2 to the F1 gates).
TRAIN     utility: train/cache one store (--seed, --prd, --epochs); no evaluation. Used to train grid stores in
          parallel processes; it changes nothing about which stores are trained or how they are evaluated.
Exit code 3 = a pre-declared stop/kill condition fired.
Usage: python run_calibration_v4.py --phase F0 | F1G | F1C | F234 | P2 | VAL | TRAIN
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
from s1.memstore import (build_dual_store, choose_donors, delete_slots, dual_store_qc, parametric_interference,  # noqa: E402
                         train_dual_store, transplant)
from s1.store import name_fluency, probe, site_stats  # noqa: E402
from s1.v4qc import (V4_CATEGORIES, ablation_report, audit, base_correct_ev, dc1_check, delete_qc,  # noqa: E402
                      dose_report, draw_v4_sets, f1_eligible)

DEV_SEEDS = [9061, 9062, 9063]          # v4.1 development (9041-9043: v4 history, not used)
VAL_SEEDS = [9051, 9052, 9053]
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "raw", "calibration")
STORE_DIR = os.path.join(HERE, "_calib")
RES = os.path.join(OUT, "calibration_results_v41.json")
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


def store_path(seed, p_rd):
    return os.path.join(STORE_DIR, f"v41_store_{seed}_prd{p_rd}.pt")


def get_store(cfg, seed, p_rd, epochs, res=None, eval_every=0, tag=None):
    """Train (or load) the dual store for (seed, p_rd). Training uses derive_seed(seed, 'storeA').
    A sidecar (<store>_meta.json + <store>_counts.npz) records epochs, isolation flag, threads, time, the training
    curve and the per-slot route-drop / presentation counts (store-internal; dose diagnostic only). A cached store is
    used only if its recorded epochs and isolation flag match the request."""
    world = world_of(cfg, seed)
    path = store_path(seed, p_rd if tag is None else tag)
    meta_p, cnt_p = path[:-3] + "_meta.json", path[:-3] + "_counts.npz"
    iso = bool(cfg["memory"].get("gradient_isolation", False))
    if os.path.exists(path):
        meta = json.load(open(meta_p))
        assert meta["epochs"] == epochs and meta["gradient_isolation"] == iso, (path, meta, epochs, iso)
        m = build_dual_store(cfg["store"], cfg["memory"], world)
        m.load_state_dict(torch.load(path))
        m.eval()
        c = np.load(cnt_p)
        meta["counts"] = {"route_drop_count": c["route_drop_count"], "presentations": c["presentations"]}
        return world, m, meta
    t0 = time.perf_counter()
    st = {}
    m, _, curve = train_dual_store(world, cfg["store"], cfg["memory"], derive_seed(seed, "storeA"), epochs, p_rd,
                                   eval_every=eval_every, log=log if eval_every else None, stats=st)
    os.makedirs(STORE_DIR, exist_ok=True)
    meta = {"seed": seed, "p_rd": p_rd, "epochs": epochs, "gradient_isolation": iso, "threads": torch.get_num_threads(),
            "minutes": (time.perf_counter() - t0) / 60, "curve": curve}
    np.savez(cnt_p, route_drop_count=st["route_drop_count"], presentations=st["presentations"])
    json.dump(meta, open(meta_p, "w"), indent=1)
    torch.save(m.state_dict(), path)                    # written last: its existence marks a complete store
    meta["counts"] = {"route_drop_count": st["route_drop_count"], "presentations": st["presentations"]}
    return world, m, meta


def _info(meta):
    """JSON-safe training info (counts summarised; the per-slot arrays stay in the sidecar)."""
    if meta is None:
        return None
    out = {k: v for k, v in meta.items() if k != "counts"}
    if "counts" in meta:
        out["route_drop_count_mean"] = float(np.mean(meta["counts"]["route_drop_count"]))
        out["presentations_mean"] = float(np.mean(meta["counts"]["presentations"]))
    return out


def phase_TRAIN(cfg, seed, p_rd, epochs):
    """Utility: train and cache one store; nothing is evaluated."""
    _, _, meta = get_store(cfg, seed, p_rd, epochs)
    log(f"TRAIN {seed} p_rd={p_rd} E={epochs}: {meta['minutes']:.1f} min, threads {meta['threads']}")
    return 0


# ---------------------------------------------------------------- F0
def phase_F0(cfg):
    res = load_res()
    v4 = cfg["v4"]
    world, m, info = get_store(cfg, DEV_SEEDS[0], 0.35, v4["E_store"], eval_every=10)
    qc = dual_store_qc(m, world, v4["store_qc"])
    out = {"seed": DEV_SEEDS[0], "p_rd": 0.35, "epochs": v4["E_store"], "train": _info(info), "qc": qc}
    E = v4["E_store"]
    if qc["trained_fact_acc"] < v4["store_qc"]["trained_fact_acc_min"]:      # C1 rule (pre-declared)
        log("F0: trained-fact gate missed at 60 epochs -> C1 rule (200 epochs)")
        world, m200, info200 = get_store(cfg, DEV_SEEDS[0], 0.35, 200, eval_every=10, tag="0.35_C1rule_E200")
        e99 = next((c["epoch"] for c in info200["curve"] if c["trained_fact_acc"] >= 0.99), None)
        E = 200 if e99 is None else min(200, int(math.ceil(1.5 * e99 / 10.0)) * 10)
        out["C1_rule"] = {"E99": e99, "E_store": E}
        old, new = store_path(DEV_SEEDS[0], 0.35), store_path(DEV_SEEDS[0], "0.35_E60_superseded")
        for suf in ("_meta.json", "_counts.npz"):
            os.replace(old[:-3] + suf, new[:-3] + suf)
        os.replace(old, new)
        world, m, info = get_store(cfg, DEV_SEEDS[0], 0.35, E, eval_every=10)
        qc = dual_store_qc(m, world, v4["store_qc"])
        out.update(qc_after_C1=qc, train_after_C1=_info(info))
    out["E_store"] = E
    out["pass"] = bool(qc["pass"])
    res["F0"] = out
    save_res(res)
    log(f"F0: {qc} -> pass={qc['pass']} (E_store={E})")
    return 0 if qc["pass"] else STOP


# ---------------------------------------------------------------- F1 (v4.1: fixed 5-point grid + dose coherence)
def f1_store_eval(cfg, seed, p, E):
    """F1 gates + dose report for one (seed, p_rd) store."""
    v4 = cfg["v4"]
    world, m, info = get_store(cfg, seed, p, E)
    qc = dual_store_qc(m, world, v4["store_qc"])
    rep = ablation_report(m, world, derive_seed(seed, "ablation", p))
    ok, gates = f1_eligible(rep, qc, v4["F1"])
    dose = dose_report(m, world, info["counts"]["route_drop_count"], p, E, v4["sets"], derive_seed(seed, "setsV4"),
                       dc2_alpha=v4["F1"]["dose"]["dc2_alpha"])
    log(f"F1 {seed} p_rd={p}: eligible={ok} gates={gates} | A acc {rep['A_acc_covered_bc']:.3f} IQR "
        f"{rep['A_margin_iqr_covered_bc']:.2f} M1 {rep['M1_acc_covered_bc']:.3f} M2 {rep['M2_acc_covered']:.3f} "
        f"integ {rep['integrated_acc_covered_bc']:.3f} param-only {rep['integrated_acc_param_only_trained']:.3f} | dose: "
        f"count {dose['count_mean_bc_cov']:.1f} (exp {dose['expected_count']:.1f}) A margin mean {dose['A_margin_mean']:.2f} "
        f"X lost {dose['X_lost_after_T_DELETE']} rho {dose['spearman_count_vs_A_margin']:.3f} "
        f"(p {dose['spearman_p_one_sided']:.2g}) | QC {qc}")
    return {"qc": qc, "ablation": rep, "gates": gates, "eligible": ok, "dose": dose, "train": _info(info)}


def phase_F1G(cfg):
    """9061 x fixed grid; selection; dose coherence (DC-1, DC-2 at the selected p_rd)."""
    res = load_res()
    v4 = cfg["v4"]
    E = res["F0"]["E_store"]
    rows = {}
    for p in v4["F1"]["route_dropout_grid"]:
        rows[str(p)] = f1_store_eval(cfg, DEV_SEEDS[0], p, E)
        res.setdefault("F1", {})["grid_%d" % DEV_SEEDS[0]] = rows
        save_res(res)
    d = v4["F1"]["dose"]
    dc1 = dc1_check({float(p): rows[p]["ablation"]["A_acc_covered_bc"] for p in rows}, d["dc1_pairwise_decrease_max"],
                    d["dc1_range_min"])
    res["F1"]["DC1"] = dc1
    ok = [p for p in rows if rows[p]["eligible"]]
    if not ok:
        res["F1"].update(selected_p_rd=None, stop="K-2: no p_rd in the fixed grid is F1-eligible")
        save_res(res)
        log(f"F1: no p_rd eligible on {DEV_SEEDS[0]} -> K-2 STOP (DC-1 {dc1})")
        return STOP
    sel = min(ok, key=lambda p: abs(rows[p]["ablation"]["A_acc_covered_bc"] - 0.5))
    dc2 = rows[sel]["dose"]["DC2_pass"]
    res["F1"].update(selected_p_rd=float(sel), DC2_selected_9061=dc2)
    if not (dc1["DC1_pass"] and dc2):
        res["F1"]["stop"] = "K-2d: an F1-eligible p_rd exists but the dose-response is not coherent"
        save_res(res)
        log(f"F1: p_rd={sel} eligible but dose coherence fails (DC-1 {dc1}; DC-2 {dc2}) -> K-2d STOP")
        return STOP
    save_res(res)
    log(f"F1 grid: selected p_rd={sel} (DC-1 {dc1}; DC-2 pass) -> confirm on {DEV_SEEDS[1:]}")
    return 0


def phase_F1C(cfg):
    """Confirmation of the single frozen p_rd on 9062, 9063: F1 gates + DC-2."""
    res = load_res()
    E, sel = res["F0"]["E_store"], res["F1"]["selected_p_rd"]
    conf = {}
    for s in DEV_SEEDS[1:]:
        conf[str(s)] = f1_store_eval(cfg, s, sel, E)
        conf[str(s)]["confirmed"] = bool(conf[str(s)]["eligible"] and conf[str(s)]["dose"]["DC2_pass"])
    res["F1"]["confirm"] = conf
    res["F1"]["pass"] = bool(all(v["confirmed"] for v in conf.values()))
    save_res(res)
    log(f"F1 selected p_rd={sel}; confirmation pass={res['F1']['pass']} "
        f"({ {s: (v['eligible'], v['dose']['DC2_pass']) for s, v in conf.items()} })")
    return 0 if res["F1"]["pass"] else STOP


# ---------------------------------------------------------------- F2-F4
def intervened(cfg, seed, p_rd, E):
    """Intact store, its context, and the intervened copy (T-DELETE(X) + Y same-answer value transplant)."""
    world, A, _ = get_store(cfg, seed, p_rd, E)
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


def evaluate_F234(cfg, seeds, p_rd, E, tag):
    per = {}
    for s in seeds:
        c = intervened(cfg, s, p_rd, E)
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
    assert res["F0"]["pass"] and res["F1"].get("pass"), "F2-F4 only after F0 and F1 (incl. confirmation) pass"
    E, p = res["F0"]["E_store"], res["F1"]["selected_p_rd"]
    out = evaluate_F234(cfg, DEV_SEEDS, p, E, "dev")
    out["stopped_at"] = ladder_verdict(out)
    res["F234"] = out
    save_res(res)
    log(f"F2-F4: F2 {out['F2_pass']} Y {out['Y_pass']} F3 {out['F3_pass']} F4 {out['F4_pass']} (ident {out['identifiability_median']}, "
        f"book kill {out['kill_bookkeeping']}) -> stopped_at={out['stopped_at']}")
    return 0 if out["stopped_at"] is None else STOP


# ---------------------------------------------------------------- P2 (parametric-route interference)
def p2_eval(cfg, seed, p_rd, E, icfg):
    world, A, _ = get_store(cfg, seed, p_rd, E)
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
    E, p = res["F0"]["E_store"], res["F1"]["selected_p_rd"]
    P = cfg["v4"]["P2"]
    rows = []
    for lr in P["lr_grid"]:
        icfg = dict(P, lr=lr)
        row = {"lr": lr, "seeds": {}}
        for s in DEV_SEEDS:
            row["seeds"][str(s)] = p2_eval(cfg, s, p, E, icfg)
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
    """Record the complete development recipe (only if F0, F1 + confirmation, F2-F4 and P2 all passed). The chain
    commits the results file after this phase and before VAL."""
    import subprocess
    res = load_res()
    ok = bool(res["F0"]["pass"] and res["F1"].get("pass") and res.get("F234", {}).get("stopped_at", "x") is None
              and res.get("P2", {}).get("selected"))
    if not ok:
        log("FREEZE refused: the development recipe is incomplete")
        return STOP
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    res["FROZEN"] = {"E_store": res["F0"]["E_store"], "p_rd": res["F1"]["selected_p_rd"], "P2": res["P2"]["selected"],
                     "code_commit": head, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_res(res)
    log(f"FROZEN recipe: {res['FROZEN']}")
    return 0


def phase_VAL(cfg):
    """Frozen v4.1 recipe ONCE on 9051-9053: stores, F0 QC, F1 ablation gates + DC-2, F2-F4 gates, P2 at the frozen
    config."""
    res = load_res()
    assert res.get("FROZEN") and "VAL" not in res, "VAL runs once, and only on the frozen recipe"
    v4 = cfg["v4"]
    E, p = res["FROZEN"]["E_store"], res["FROZEN"]["p_rd"]
    out = {"recipe": {"E_store": E, "p_rd": p, "P2": res["FROZEN"]["P2"]}, "F01": {}}
    for s in VAL_SEEDS:
        r = f1_store_eval(cfg, s, p, E)
        r["eligible"] = bool(r["eligible"] and r["dose"]["DC2_pass"])
        out["F01"][str(s)] = r
        log(f"VAL {s}: F0/F1 eligible (incl. DC-2)={r['eligible']} gates={r['gates']}")
    f234 = evaluate_F234(cfg, VAL_SEEDS, p, E, "val")
    out["F234"] = f234
    out["stopped_at"] = ladder_verdict(f234)
    icfg = dict(v4["P2"], **res["FROZEN"]["P2"])
    out["P2"] = {str(s): p2_eval(cfg, s, p, E, icfg) for s in VAL_SEEDS}
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
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
    if a.phase == "TRAIN":
        assert a.seed in DEV_SEEDS + VAL_SEEDS or a.seed == 12345, "TRAIN: development, validation or burned seed only"
        if a.seed in VAL_SEEDS:
            assert load_res().get("FROZEN"), "validation stores only after the recipe is frozen"
        sys.exit(phase_TRAIN(cfg, a.seed, a.prd, a.epochs))
    phases = {"F0": phase_F0, "F1G": phase_F1G, "F1C": phase_F1C, "F234": phase_F234, "P2": phase_P2,
              "FREEZE": phase_FREEZE, "VAL": phase_VAL}
    sys.exit(phases[a.phase](cfg) or 0)
