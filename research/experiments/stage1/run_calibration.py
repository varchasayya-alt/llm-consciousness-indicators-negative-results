"""Stage-1 CALIBRATION (methods development only). Stores + interventions only.

THIS SCRIPT NEVER TRAINS OR EVALUATES A METACOGNITIVE MONITOR OR CONTROLLER (enforced by a unit test:
it must not import s1.monitors, s1.controller or s1.pipeline). The generic diagnostic classifier (s1.diagnostics)
and the store-only nuisance models (s1.estimands) are used only for intervention QC and identifiability diagnostics.

History (results kept in calibration_results.json; code in git history):
  v1 (b13fc54): C1 E_store = 60; C2 stores 9001-9005; C3 (flawed rule) rate max 0.6; C4 FAILED (collateral; D26).
  v2 (f1e8efb): C3r rate max 0.6; C4v2 selectivity solved; C4val FAILED (Y-sham procedure fingerprint; D34).
  9001-9005 and 9011-9013 are historical only and are NOT used to select any v3 recipe.

v3 rules (decisions D36-D41; logs/calibration_plan.md "Revision v3"; committed BEFORE any v3 run):
 C2dev  train fresh development stores 9031-9033 (E_store = 60); store QC must pass.
 C4dev  T-FORGET mechanism family, identical procedure for all X items:
          V0 lr {1e-4,3e-4,1e-3} x steps {100,200};  V1 = V0 x anchor_lambda {0.01,1,100};
          V2 lr x steps (MT+CT retain pool);         V3 lr x steps (only block-1-2 MLPs trainable).
        Fixed: gamma 10, lambda_retain 100, retain batches 1024/256.
        Eligible = on every dev seed all per-seed gates pass (X diversity [.35,.65] & IQR(C_post) >= 1; Y lost <= .03;
        Z lost <= .05 per category; continuous retention for Y and all Z categories at 0.10; Y displacement match
        >= 8/10 sites; |dfluency| <= .1 SD) AND binary-secondary feasibility (>= 50 matched pairs, max SMD <= .10)
        on every dev seed AND median identifiability over dev seeds: R2_gen < .90, R2_pre < .90, R2_joint < .95.
        Select: smallest worst-seed retention load (max metric/threshold); ties within 0.05 load -> mean X lost
        fraction closest to 0.5 -> smaller lr*steps. If none eligible: STOP (move toward option B; report).
 C2val  train fresh validation stores 9021-9023.
 C4val  frozen T-FORGET recipe ONCE on 9021-9023 (fallback ladder allowed); same per-seed gates + F6 per seed +
        identifiability medians over the 3 validation seeds. Failure: STOP and report (no retuning on 9021-9023).
 C7     T-INTERF on dev seeds: lr {3e-4,1e-3,3e-3}, max 3000 steps, stop at new-fact acc >= .95; eligible: lost
        fraction in [.10,.50] on every dev seed AND P2 identifiability medians within thresholds; choose mean lost
        closest to .25; max_steps = ceil100(1.5 * max steps used).
 C8     T-NEW on dev seeds: lr {5e-4,1e-3,2e-3} x steps {50,100,200} x lambda_retain {1,10}; v3 QC incl. continuous
        collateral anchor; smallest lr*steps, tie -> smaller lambda.
 C9     T-FAM on dev seeds: lr {5e-4,1e-3,2e-3} at 16 (then 32, 64) presentations; v3 QC; smallest passing.
 C10    Y negative-control fingerprint (G1-G4, reported) and INTERF diagnostics on dev seeds; caliper for the
        DESCRIPTIVE matched contrasts = smallest c in {.25,.5,.75,1} with >= 100 FORGET (X_lost,Y) and >= 60
        INTERF pairs on every dev seed, else 1.0 (descriptive only; not a gate).
 VAL    complete frozen recipe ONCE on 9021-9023 (FORGET, INTERF, NEW, FAM; ladder allowed): every per-seed QC +
        F6 + identifiability medians for P1 and P2. Failure: STOP and report.
Usage: python run_calibration.py --phase C2dev | C4dev | C2val | C4val | C7 | C8 | C9 | C10 | VAL
Exit code 3 = a pre-declared acceptance criterion failed (the chain stops; report, never repair silently).
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
from s1.diagnostics import cv_auroc, generic_features  # noqa: E402
from s1.interventions import (fam_qc, familiarity_boost, forget_qc, forget_with_sham, interference,  # noqa: E402
                              interference_qc, learn_new, new_qc)
from s1.matching import caliper_match, log_profile  # noqa: E402
from s1.store import build_store, name_fluency, probe, site_stats, store_qc, train_store  # noqa: E402

DEV_SEEDS = [9031, 9032, 9033]
VAL_SEEDS = [9021, 9022, 9023]
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "raw", "calibration")
STORE_DIR = os.path.join(HERE, "_calib")
RES = os.path.join(OUT, "calibration_results.json")
CALIPERS = [0.25, 0.5, 0.75, 1.0]
CRITERION_FAILED = 3
TIE_LOAD = 0.05


def log(msg):
    print(msg, flush=True)


def load_res():
    return json.load(open(RES)) if os.path.exists(RES) else {}


def save_res(res):
    os.makedirs(OUT, exist_ok=True)
    tmp = RES + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, RES)


def v3res(res):
    return res.setdefault("v3", {})


def get_world(cfg, seed):
    return W.make_world(cfg["world"], derive_seed(seed, "world"))


def load_store(cfg, world, seed):
    m = build_store(cfg["store"], world.vocab.size)
    m.load_state_dict(torch.load(os.path.join(STORE_DIR, f"store_{seed}.pt")))
    m.eval()
    return m


def context(cfg, seed):
    """Store-level context for interventions: world, store, mu/sigma, item sets, fluency SD."""
    world = get_world(cfg, seed)
    A = load_store(cfg, world, seed)
    MTi, EVi = W.items_where(world, split=W.MT), W.items_where(world, split=W.EV)
    mu, sig = site_stats(probe(A, world, MTi)["states"])
    pr = probe(A, world, EVi, want_states=False)
    trained = world.known[EVi[:, 0], EVi[:, 1]]
    bc, bi = EVi[trained & pr["correct"]], EVi[~trained & ~pr["correct"]]
    sets = W.draw_item_sets(world, bc, bi, cfg["interventions"]["sets"], derive_seed(seed, "setsA"))
    flu_sd = float(name_fluency(A, world, world.names).std())
    return dict(world=world, A=A, mu=mu, sig=sig, sets=sets, bc=bc, flu_sd=flu_sd, MTi=MTi)


def ladder_configs(base, ladder):
    """Pre-registered fallback ladder (sham_algorithm_spec.md point 5): lr x {1,.5,2} at steps x 1, then x 2."""
    for sm in ladder["step_multipliers"]:
        for lm in ladder["lr_multipliers"]:
            c = dict(base, lr=base["lr"] * lm)
            for key in ("steps", "max_steps"):
                if key in c:
                    c[key] = int(base[key] * sm)
            yield {"lr_mult": lm, "step_mult": sm}, c


def c3_rate_max(rho_01, rho_06):
    """Corrected C3 rule (v2, D29) on the retained-competence fraction (kept for the record; unit-tested)."""
    if rho_06 > 0.9:
        return 0.8
    if rho_01 < 0.5:
        return 0.3
    return 0.6


# ---------------------------------------------------------------- stores
def _train_and_qc(cfg, seeds, e_store):
    os.makedirs(STORE_DIR, exist_ok=True)
    out = {}
    for seed in seeds:
        world = get_world(cfg, seed)
        t0 = time.perf_counter()
        A, _, _ = train_store(world, cfg["store"], derive_seed(seed, "storeA"), e_store, ())
        mins = (time.perf_counter() - t0) / 60
        torch.save(A.state_dict(), os.path.join(STORE_DIR, f"store_{seed}.pt"))
        qc = store_qc(A, world, cfg["store"]["qc"])
        out[str(seed)] = {"qc": qc, "minutes": mins}
        log(f"seed {seed}: {mins:.1f} min, QC {qc}")
    return out


def phase_train(cfg, seeds, key):
    res = load_res()
    e_store = res["C1"]["E_store"]
    out = _train_and_qc(cfg, seeds, e_store)
    ok = all(v["qc"]["pass"] for v in out.values())
    v3res(res)[key] = {"E_store": e_store, "seeds": out, "all_pass": ok}
    save_res(res)
    log(f"{key} store QC all pass: {ok}")
    return 0 if ok else CRITERION_FAILED


# ---------------------------------------------------------------- T-FORGET v3
def forget_eval(cfg, c, fcfg, seed):
    t0 = time.perf_counter()
    model, flog = forget_with_sham(c["A"], c["world"], c["sets"]["X"], c["sets"]["Y"], fcfg, c["sig"],
                                   derive_seed(seed, "forget"))
    qc, aux = forget_qc(c["A"], model, c["world"], c["sets"], c["sig"], fcfg["qc"], CALIPERS,
                        derive_seed(seed, "matchF"), c["flu_sd"], c["mu"])
    qc["seconds"] = time.perf_counter() - t0
    qc["trace_last"] = flog["trace"][-1]
    qc["n_trainable"] = flog["n_trainable"]
    qc["retain_pool"] = flog["retain_pool"]
    return model, qc, aux


def ident_ok(qcs, cfg):
    t = cfg["identifiability"]
    med = {k: float(np.median([q["identifiability"][k] for q in qcs])) for k in ("r2_gen", "r2_pre", "r2_joint")}
    ok = med["r2_gen"] < t["r2_gen_max"] and med["r2_pre"] < t["r2_pre_max"] and med["r2_joint"] < t["r2_joint_max"]
    return bool(ok), med


def summarise_cell(row, cfg):
    S = list(row["seeds"].values())
    row["all_pass"] = all(v["pass"] for v in S)
    row["all_F6"] = all(v["binary_secondary"]["feasible"] for v in S)
    row["identifiable"], row["identifiability_median"] = ident_ok(S, cfg)
    row["worst_load"] = float(max(v["retention_load_max"] for v in S))
    row["mean_X_lost"] = float(np.mean([v["X_lost_frac"] for v in S]))
    row["eligible"] = bool(row["all_pass"] and row["all_F6"] and row["identifiable"])
    row["all_F6_alt"] = all(v["binary_secondary_alt"]["feasible"] for v in S)                 # recorded only (D43)
    row["eligible_alt"] = bool(row["all_pass"] and row["all_F6_alt"] and row["identifiable"])
    return row


def select_v3(rows):
    elig = [r for r in rows if r["eligible"]]
    if not elig:
        return None
    best = min(r["worst_load"] for r in elig)
    tied = [r for r in elig if r["worst_load"] <= best + TIE_LOAD]
    return min(tied, key=lambda r: (abs(r["mean_X_lost"] - 0.5), r["lr"] * r["steps"]))


def v3_cells():
    lrs, steps = (1e-4, 3e-4, 1e-3), (100, 200)
    cells = [("V0", lr, st, 0.0) for lr in lrs for st in steps]
    cells += [("V1", lr, st, a) for a in (0.01, 1.0, 100.0) for lr in lrs for st in steps]
    cells += [("V2", lr, st, 0.0) for lr in lrs for st in steps]
    cells += [("V3", lr, st, 0.0) for lr in lrs for st in steps]
    return cells


def phase_C4dev(cfg):
    res = load_res()
    R = v3res(res)
    base = dict(cfg["interventions"]["T_FORGET"])
    ctxs = {s: context(cfg, s) for s in DEV_SEEDS}
    rows = R.setdefault("C4dev", {}).setdefault("grid", [])
    done = {(r["variant"], r["lr"], r["steps"], r["anchor_lambda"]) for r in rows}
    for var, lr, st, anc in v3_cells():
        if (var, lr, st, anc) in done:
            continue
        fcfg = dict(base, variant=var, lr=lr, steps=st, anchor_lambda=anc)
        row = {"variant": var, "lr": lr, "steps": st, "anchor_lambda": anc, "seeds": {}}
        for s, c in ctxs.items():
            _, qc, _ = forget_eval(cfg, c, fcfg, s)
            row["seeds"][str(s)] = qc
        summarise_cell(row, cfg)
        rows.append(row)
        log(f"[{var}] lr={lr} st={st} anc={anc}: eligible={row['eligible']} worst_load={row['worst_load']:.2f} "
            f"F6={row['all_F6']} F6alt={row['all_F6_alt']} ident={row['identifiability_median']} | " +
            " | ".join(f"{s}: X {v['X_lost_frac']:.2f} iqr {v['C_post_X_iqr']:.1f} Y {v['Y_lost_frac']:.3f} "
                       f"Zmax {v['Z_lost_max_over_categories']:.3f} load {v['retention_load_max']:.2f} "
                       f"sites {v['sites_within_tol']} flu {v['fluency_change_sd']:.2f} pairs {v['binary_secondary']['pairs']}"
                       f"/{v['binary_secondary']['max_abs_smd']:.2f} failed {v['failed_gates']} ({v['seconds']:.0f}s)"
                       for s, v in row["seeds"].items()))
        save_res(res)
    sel = select_v3(rows)
    alt = select_v3([dict(r, eligible=r["eligible_alt"]) for r in rows])     # F6' alternative, recorded only (D43)
    R["C4dev"]["selected_under_F6alt_recorded_only"] = None if alt is None else {k: alt[k] for k in ("variant", "lr", "steps", "anchor_lambda")}
    R["C4dev"]["selected"] = None if sel is None else {k: sel[k] for k in ("variant", "lr", "steps", "anchor_lambda")}
    R["C4dev"]["selected_summary"] = None if sel is None else {k: sel[k] for k in ("worst_load", "mean_X_lost", "identifiability_median")}
    save_res(res)
    log(f"C4dev selected: {R['C4dev']['selected']}")
    return 0 if sel is not None else CRITERION_FAILED


def forget_with_ladder(cfg, c, base, s):
    attempts = []
    for rung, fcfg in ladder_configs(base, cfg["interventions"]["fallback_ladder"]):
        model, qc, aux = forget_eval(cfg, c, fcfg, s)
        attempts.append({"rung": rung, "pass": bool(qc["pass"]), "qc": qc})
        if qc["pass"]:
            break
    return attempts, model, aux


def phase_C4val(cfg):
    res = load_res()
    R = v3res(res)
    sel = R["C4dev"]["selected"]
    base = dict(cfg["interventions"]["T_FORGET"], **sel)
    out = {}
    for s in VAL_SEEDS:
        c = context(cfg, s)
        att, _, _ = forget_with_ladder(cfg, c, base, s)
        q = att[-1]["qc"]
        out[str(s)] = {"attempts": att, "pass": att[-1]["pass"], "base_config_passed": att[0]["pass"],
                       "F6": q["binary_secondary"]["feasible"]}
        log(f"C4val seed {s}: pass={att[-1]['pass']} rungs={len(att)} failed={q['failed_gates']} X {q['X_lost_frac']:.2f} "
            f"load {q['retention_load_max']:.2f} F6 {q['binary_secondary']} ident {q['identifiability']}")
    ok_seeds = all(v["pass"] and v["F6"] for v in out.values())
    idok, med = ident_ok([v["attempts"][-1]["qc"] for v in out.values()], cfg)
    passed = bool(ok_seeds and idok)
    R["C4val"] = {"recipe": sel, "seeds": out, "all_seeds_pass": ok_seeds, "identifiability_median": med,
                  "identifiable": idok, "PASS": passed}
    save_res(res)
    log(f"C4val PASS={passed} (seeds_ok={ok_seeds}, identifiable={idok} {med})")
    return 0 if passed else CRITERION_FAILED


# ---------------------------------------------------------------- C7-C9 (dev seeds)
def phase_C7(cfg):
    res = load_res()
    R = v3res(res)
    base = dict(cfg["interventions"]["T_INTERF"], max_steps=3000)
    rows = []
    ctxs = {s: context(cfg, s) for s in DEV_SEEDS}
    for lr in (3e-4, 1e-3, 3e-3):
        row = {"lr": lr, "seeds": {}}
        for s, c in ctxs.items():
            t0 = time.perf_counter()
            m, trace = interference(c["A"], c["world"], dict(base, lr=lr), derive_seed(s, "interf"))
            qc, _ = interference_qc(c["A"], m, c["world"], c["bc"], c["sig"], cfg["interventions"]["T_INTERF"]["qc"],
                                    CALIPERS, derive_seed(s, "matchI"), c["mu"])
            qc.update(steps_used=trace[-1]["step"] if trace else None,
                      final_new_acc=trace[-1]["stop_metric"] if trace else None, seconds=time.perf_counter() - t0)
            row["seeds"][str(s)] = qc
            log(f"C7 lr={lr} seed {s}: lost {qc['lost_frac']:.3f} steps {qc['steps_used']} ident {qc['identifiability']} "
                f"pairs {qc['matched_pairs']} ({qc['seconds']:.0f}s)")
        S = list(row["seeds"].values())
        row["mean_lost"] = float(np.mean([v["lost_frac"] for v in S]))
        row["all_in_range"] = all(0.10 <= v["lost_frac"] <= 0.50 for v in S)
        row["identifiable"], row["identifiability_median"] = ident_ok(S, cfg)
        row["eligible"] = bool(row["all_in_range"] and row["identifiable"])
        rows.append(row)
        R["C7"] = {"grid": rows, "selected": None}
        save_res(res)
    ok = [r for r in rows if r["eligible"]]
    sel = min(ok, key=lambda r: abs(r["mean_lost"] - 0.25)) if ok else None
    sel_cfg = None
    if sel:
        used = max(v["steps_used"] for v in sel["seeds"].values())
        sel_cfg = {"lr": sel["lr"], "max_steps": int(math.ceil(1.5 * used / 100.0)) * 100}
    R["C7"] = {"grid": rows, "selected": sel_cfg}
    save_res(res)
    log(f"T-INTERF selected: {sel_cfg}")
    return 0 if sel_cfg else CRITERION_FAILED


def phase_C8(cfg):
    res = load_res()
    R = v3res(res)
    base = dict(cfg["interventions"]["T_NEW"])
    rows = []
    ctxs = {s: context(cfg, s) for s in DEV_SEEDS}
    for lr in (5e-4, 1e-3, 2e-3):
        for st in (50, 100, 200):
            for lam in (1.0, 10.0):
                row = {"lr": lr, "steps": st, "lambda_retain": lam, "seeds": {}}
                for s, c in ctxs.items():
                    m, _ = learn_new(c["A"], c["world"], c["sets"]["N"], c["sets"]["U"],
                                     dict(base, lr=lr, steps=st, lambda_retain=lam), derive_seed(s, "new"))
                    row["seeds"][str(s)] = new_qc(c["A"], m, c["world"], c["sets"]["N"], c["sets"]["U"], c["bc"], base["qc"])
                row["all_pass"] = all(v["pass"] for v in row["seeds"].values())
                rows.append(row)
                log(f"C8 lr={lr} st={st} lam={lam}: pass={row['all_pass']} " + str(
                    {s: (round(v['N_learned_frac'], 2), round(v['U_correct_frac'], 3), round(v['collateral_bc']['lost'], 3),
                         round(v['collateral_bc']['abs_margin_over_DC'], 3), round(v['collateral_near_entity']['abs_margin_over_DC'], 3))
                     for s, v in row['seeds'].items()}))
                R["C8"] = {"grid": rows, "selected": None}
                save_res(res)
    ok = [r for r in rows if r["all_pass"]]
    sel = min(ok, key=lambda r: (r["lr"] * r["steps"], r["lambda_retain"])) if ok else None
    R["C8"] = {"grid": rows, "selected": None if sel is None else {k: sel[k] for k in ("lr", "steps", "lambda_retain")}}
    save_res(res)
    log(f"T-NEW selected: {R['C8']['selected']}")
    return 0 if sel else CRITERION_FAILED


def phase_C9(cfg):
    res = load_res()
    R = v3res(res)
    base = dict(cfg["interventions"]["T_FAM"], lambda_retain=1.0)
    rows, sel = [], None
    ctxs = {s: context(cfg, s) for s in DEV_SEEDS}
    for steps in (16, 32, 64):
        for lr in (5e-4, 1e-3, 2e-3):
            row = {"lr": lr, "steps": steps, "lambda_retain": 1.0, "seeds": {}}
            for s, c in ctxs.items():
                m, _ = familiarity_boost(c["A"], c["world"], c["sets"]["F"], dict(base, lr=lr, steps=steps),
                                         derive_seed(s, "fam"), c["sets"]["U2"])
                row["seeds"][str(s)] = fam_qc(c["A"], m, c["world"], c["sets"]["F"], c["sets"]["U2"], c["bc"], base["qc"], c["flu_sd"])
            row["all_pass"] = all(v["pass"] for v in row["seeds"].values())
            rows.append(row)
            log(f"C9 steps={steps} lr={lr}: pass={row['all_pass']} " + str(
                {s: (round(v['fluency_rise_rel_sd'], 2), round(v['F_correct_frac'], 3), round(v['collateral_bc']['abs_margin_over_DC'], 3))
                 for s, v in row['seeds'].items()}))
        ok = [r for r in rows if r["all_pass"] and r["steps"] == steps]
        if ok:
            sel = min(ok, key=lambda r: r["lr"])
            break
    R["C9"] = {"grid": rows, "selected": None if sel is None else {"lr": sel["lr"], "steps": sel["steps"], "lambda_retain": 1.0}}
    save_res(res)
    log(f"T-FAM selected: {R['C9']['selected']}")
    return 0 if sel else CRITERION_FAILED


# ---------------------------------------------------------------- C10 and VAL
def y_fingerprint(c, aux, seed):
    """Y negative control vs targeted X: generic-statistics distinguishability (reported; not a v3 gate)."""
    lost, ycor = aux["lostX"], aux["postY_correct"]
    out = {"matched": {}}
    pX, pY = log_profile(aux["dX"]), log_profile(aux["dY"])
    gX, gY = aux["GX"], aux["GY"]
    for cal in CALIPERS:
        pairs = caliper_match(pX[lost], pY[ycor], cal, derive_seed(seed, "matchF"))
        d = {"pairs": len(pairs)}
        if len(pairs) >= 20:
            xi = np.nonzero(lost)[0][[p[0] for p in pairs]]
            yi = np.nonzero(ycor)[0][[p[1] for p in pairs]]
            d["G1-G3_all40"] = cv_auroc(gX[xi], gY[yi], seed=seed)
        out["matched"][str(cal)] = d
    out["X_all_vs_Y_all40"] = cv_auroc(gX, gY[ycor], seed=seed)
    return out


def interf_with_ladder(cfg, c, base, s):
    attempts = []
    qcfg = cfg["interventions"]["T_INTERF"]["qc"]
    for rung, icfg in ladder_configs(base, cfg["interventions"]["fallback_ladder"]):
        m, trace = interference(c["A"], c["world"], icfg, derive_seed(s, "interf"))
        qc, aux = interference_qc(c["A"], m, c["world"], c["bc"], c["sig"], qcfg, CALIPERS, derive_seed(s, "matchI"), c["mu"])
        qc["steps_used"] = trace[-1]["step"] if trace else None
        attempts.append({"rung": rung, "pass": bool(qc["pass"]), "qc": qc})
        if qc["pass"]:
            break
    return attempts


def phase_C10(cfg):
    res = load_res()
    R = v3res(res)
    fbase = dict(cfg["interventions"]["T_FORGET"], **R["C4dev"]["selected"])
    ibase = dict(cfg["interventions"]["T_INTERF"], **R["C7"]["selected"])
    out = {}
    for s in DEV_SEEDS:
        c = context(cfg, s)
        fa, _, aux = forget_with_ladder(cfg, c, fbase, s)
        ia = interf_with_ladder(cfg, c, ibase, s)
        out[str(s)] = {"forget_attempts": fa, "interf_attempts": ia, "Y_fingerprint": y_fingerprint(c, aux, s)}
        log(f"C10 seed {s}: FORGET pass={fa[-1]['pass']} pairs={fa[-1]['qc']['matched_pairs_Xlost_Y']} | INTERF "
            f"pass={ia[-1]['pass']} lost={ia[-1]['qc']['lost_frac']:.3f} pairs={ia[-1]['qc']['matched_pairs']} | "
            f"Y fingerprint {out[str(s)]['Y_fingerprint']}")
        R["C10"] = {"seeds": out}
        save_res(res)
    cal_sel = None
    for cal in CALIPERS:
        if all(out[str(s)]["forget_attempts"][-1]["qc"]["matched_pairs_Xlost_Y"][str(cal)] >= 100 and
               out[str(s)]["interf_attempts"][-1]["qc"]["matched_pairs"][str(cal)] >= 60 for s in DEV_SEEDS):
            cal_sel = cal
            break
    R["C10"] = {"seeds": out, "caliper_selected": cal_sel if cal_sel is not None else 1.0,
                "caliper_rule_met": cal_sel is not None}
    save_res(res)
    log(f"C10 caliper (descriptive contrasts): {R['C10']['caliper_selected']} (rule met: {cal_sel is not None})")
    return 0


def phase_VAL(cfg):
    """Complete frozen recipe, ONCE, on fresh validation seeds 9021-9023."""
    res = load_res()
    R = v3res(res)
    icfg = cfg["interventions"]
    fbase = dict(icfg["T_FORGET"], **R["C4dev"]["selected"])
    ibase = dict(icfg["T_INTERF"], **R["C7"]["selected"])
    nbase = dict(icfg["T_NEW"], **R["C8"]["selected"])
    mbase = dict(icfg["T_FAM"], **R["C9"]["selected"])
    out, all_ok = {}, True
    for s in VAL_SEEDS:
        c = context(cfg, s)
        fa, _, _ = forget_with_ladder(cfg, c, fbase, s)
        ia = interf_with_ladder(cfg, c, ibase, s)
        na, ma = [], []
        for rung, nc in ladder_configs(nbase, icfg["fallback_ladder"]):
            m, _ = learn_new(c["A"], c["world"], c["sets"]["N"], c["sets"]["U"], nc, derive_seed(s, "new"))
            q = new_qc(c["A"], m, c["world"], c["sets"]["N"], c["sets"]["U"], c["bc"], nc["qc"])
            na.append({"rung": rung, "pass": q["pass"], "qc": q})
            if q["pass"]:
                break
        for rung, mc in ladder_configs(mbase, icfg["fallback_ladder"]):
            m, _ = familiarity_boost(c["A"], c["world"], c["sets"]["F"], mc, derive_seed(s, "fam"), c["sets"]["U2"])
            q = fam_qc(c["A"], m, c["world"], c["sets"]["F"], c["sets"]["U2"], c["bc"], mc["qc"], c["flu_sd"])
            ma.append({"rung": rung, "pass": q["pass"], "qc": q})
            if q["pass"]:
                break
        ok = (fa[-1]["pass"] and fa[-1]["qc"]["binary_secondary"]["feasible"] and ia[-1]["pass"]
              and na[-1]["pass"] and ma[-1]["pass"])
        all_ok &= bool(ok)
        out[str(s)] = {"forget": fa, "interf": ia, "new": na, "fam": ma, "pass": bool(ok)}
        log(f"VAL seed {s}: pass={ok} | FORGET {fa[-1]['pass']} ({len(fa)}) | INTERF {ia[-1]['pass']} ({len(ia)}) | "
            f"NEW {na[-1]['pass']} ({len(na)}) | FAM {ma[-1]['pass']} ({len(ma)})")
        R["VAL"] = {"seeds": out}
        save_res(res)
    idF, medF = ident_ok([out[str(s)]["forget"][-1]["qc"] for s in VAL_SEEDS], cfg)
    idI, medI = ident_ok([out[str(s)]["interf"][-1]["qc"] for s in VAL_SEEDS], cfg)
    passed = bool(all_ok and idF and idI)
    R["VAL"] = {"seeds": out, "identifiability_P1": medF, "identifiability_P2": medI, "PASS": passed}
    save_res(res)
    log(f"VAL PASS={passed} (per-seed {all_ok}; P1 identifiable {idF} {medF}; P2 identifiable {idI} {medI})")
    return 0 if passed else CRITERION_FAILED


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "stage1_config.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
    phases = {"C2dev": lambda: phase_train(cfg, DEV_SEEDS, "C2dev"), "C4dev": lambda: phase_C4dev(cfg),
              "C2val": lambda: phase_train(cfg, VAL_SEEDS, "C2val"), "C4val": lambda: phase_C4val(cfg),
              "C7": lambda: phase_C7(cfg), "C8": lambda: phase_C8(cfg), "C9": lambda: phase_C9(cfg),
              "C10": lambda: phase_C10(cfg), "VAL": lambda: phase_VAL(cfg)}
    sys.exit(phases[a.phase]() or 0)
