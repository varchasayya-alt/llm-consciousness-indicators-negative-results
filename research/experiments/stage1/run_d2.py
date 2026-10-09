"""Stage-1 D2 STORE-ONLY kill test: self-generated competence change through continued learning.

NEVER trains or evaluates a monitor or controller and uses NO B1 component (unit-tested AST guard).
Rules: d2_protocol.md, d2_S0_satisfiability_note.md (committed before any D2 run); decisions D62-D65.
Seeds: D2 development 9101-9103 only. Proposed D2 validation 9111-9113 is refused by this runner.

TRAIN  --seed : base store (plain v1-v3 store, 60 epochs, derive_seed(seed, "storeA"); checkpoints saved).
S1            : base-store QC (G0) on every development seed. Any failure -> STOP.
S2     --seed : for each lr in the frozen grid: INTERF (stop at new-fact acc >= .95 every 50 steps, max 3000) and the
                yoked matched-development control (CTRL, same steps / optimizer / lr / batch). Stores only.
S3     --seed : per (lr, seed) store-only report (d2.d2_report): G1, G2, G4, G5 + report-only diagnostics.
SELECT        : G3 medians per lr; eligibility; selection rule (mean lost closest to .25); frozen max_steps.
                No eligible lr -> STOP (D2 kill report; D3 proposal).
Exit code 3 = a pre-declared stop condition fired.
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
from s1.d2 import control_continuation, d2_report, population, steps_used  # noqa: E402
from s1.interventions import interference  # noqa: E402
from s1.store import build_store, probe, site_stats, store_qc, train_store  # noqa: E402

DEV_SEEDS = [9101, 9102, 9103]
VAL_SEEDS = [9111, 9112, 9113]          # proposed D2 validation: untouched, refused here
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "raw", "calibration")
STORE_DIR = os.path.join(HERE, "_calib")
RES = os.path.join(OUT, "d2_results.json")
STOP = 3


def log(msg):
    print(msg, flush=True)


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p + ".tmp", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(p + ".tmp", p)


def world_of(cfg, seed):
    return W.make_world(cfg["world"], derive_seed(seed, "world"))


def path(kind, seed, lr=None):
    return os.path.join(STORE_DIR, f"d2_{kind}_{seed}" + ("" if lr is None else f"_lr{float(lr):g}") + ".pt")


def seed_res(seed):
    return os.path.join(OUT, f"d2_S3_{seed}.json")


def get_base(cfg, seed, train_if_missing=True):
    world = world_of(cfg, seed)
    p = path("base", seed)
    m = build_store(cfg["store"], world.vocab.size)
    if os.path.exists(p):
        m.load_state_dict(torch.load(p))
        m.eval()
        return world, m
    assert train_if_missing, f"base store missing: {p}"
    E = cfg["d2"]["base_store_epochs"]
    t0 = time.perf_counter()
    m, ckpts, curve = train_store(world, cfg["store"], derive_seed(seed, "storeA"), E, cfg["store"]["checkpoint_fractions"],
                                  eval_every=10, log=log)
    m.eval()
    os.makedirs(STORE_DIR, exist_ok=True)
    torch.save(ckpts, path("ckpts", seed))
    json.dump({"seed": seed, "epochs": E, "curve": curve, "minutes": (time.perf_counter() - t0) / 60,
               "threads": torch.get_num_threads()}, open(path("base", seed)[:-3] + "_meta.json", "w"), indent=1)
    torch.save(m.state_dict(), p)                                # written last
    return world, m


def phase_TRAIN(cfg, seed):
    get_base(cfg, seed)
    return 0


def phase_S1(cfg):
    res = load_json(RES)
    out = {}
    for s in DEV_SEEDS:
        world, m = get_base(cfg, s, train_if_missing=False)
        qc = store_qc(m, world, cfg["store"]["qc"])
        meta = json.load(open(path("base", s)[:-3] + "_meta.json"))
        out[str(s)] = {"qc": qc, "train": meta, "n_population": int(len(population(m, world)))}
        log(f"S1 {s}: QC {qc} | population n {out[str(s)]['n_population']} | {meta['minutes']:.1f} min")
    res["S1"] = {"seeds": out, "pass": bool(all(v["qc"]["pass"] for v in out.values()))}
    save_json(RES, res)
    if not res["S1"]["pass"]:
        log("S1: a base store fails QC (G0) -> STOP")
        return STOP
    return 0


def phase_S2(cfg, seed):
    world, A = get_base(cfg, seed, train_if_missing=False)
    d2 = cfg["d2"]
    for lr in d2["interference"]["lr_grid"]:
        icfg = dict(d2["interference"], lr=lr)
        pi, pc = path("interf", seed, lr), path("ctrl", seed, lr)
        if os.path.exists(pc):
            continue
        t0 = time.perf_counter()
        mI, trace = interference(A, world, icfg, derive_seed(seed, "interf"))
        n = steps_used(trace, icfg["max_steps"])
        mC = control_continuation(A, world, n, icfg, derive_seed(seed, "ctrl"))
        json.dump({"seed": seed, "lr": lr, "steps_used": n, "trace": trace, "minutes": (time.perf_counter() - t0) / 60},
                  open(pi[:-3] + "_meta.json", "w"), indent=1)
        torch.save(mI.state_dict(), pi)
        torch.save(mC.state_dict(), pc)                          # written last
        log(f"S2 {seed} lr={lr}: steps {n}, final new-fact acc {trace[-1]['stop_metric'] if trace else None} "
            f"({(time.perf_counter() - t0) / 60:.1f} min)")
    return 0


def phase_S3(cfg, seed):
    world, A = get_base(cfg, seed, train_if_missing=False)
    items = population(A, world)
    MT = W.items_where(world, split=W.MT)
    mu, sig = site_stats(probe(A, world, MT)["states"])
    d2 = cfg["d2"]
    out = load_json(seed_res(seed))                              # resume: reports already computed are kept (D66)
    for lr in d2["interference"]["lr_grid"]:
        if str(float(lr)) in out:
            continue
        meta = json.load(open(path("interf", seed, lr)[:-3] + "_meta.json"))
        mI, mC = build_store(cfg["store"], world.vocab.size), build_store(cfg["store"], world.vocab.size)
        mI.load_state_dict(torch.load(path("interf", seed, lr)))
        mC.load_state_dict(torch.load(path("ctrl", seed, lr)))
        mI.eval()
        mC.eval()
        t0 = time.perf_counter()
        rep = d2_report(A, mI, mC, world, items, mu, sig, derive_seed(seed, "d2qc", lr), meta["trace"], meta["steps_used"],
                        d2["gates"], cfg["v4"]["qc"]["binary_secondary"])
        out[str(float(lr))] = rep
        log(f"S3 {seed} lr={lr}: gates {rep['gates']} | lost {rep['lost_frac']:.3f} IQR(dC) {rep['dC_iqr']:.2f} "
            f"IQR(C_post) {rep['C_post_iqr']:.2f} dC median {rep['dC_quantiles'][3]:.2f} | ident {rep['identifiability']} "
            f"| audit C_post {rep['audit']['C_post']} | ctrl lost {rep['control']['lost_frac']:.3f} r2(dC|ctrl) "
            f"{rep['control']['r2_dC_interf_given_dC_ctrl']:.3f} | steps {rep['steps_used']} ({(time.perf_counter() - t0) / 60:.1f} min)")
        save_json(seed_res(seed), out)
    return 0


def phase_SELECT(cfg):
    res = load_json(RES)
    assert res.get("S1", {}).get("pass"), "SELECT only after S1 passes"
    d2, t = cfg["d2"], cfg["identifiability"]
    per = {str(s): load_json(seed_res(s)) for s in DEV_SEEDS}
    rows = {}
    for lr in d2["interference"]["lr_grid"]:
        k = str(float(lr))
        R = [per[str(s)][k] for s in DEV_SEEDS]
        med = {m: float(np.median([r["identifiability"][m] for r in R])) for m in ("r2_pre", "r2_gen", "r2_joint")}
        g3 = bool(med["r2_pre"] < t["r2_pre_max"] and med["r2_gen"] < t["r2_gen_max"] and med["r2_joint"] < t["r2_joint_max"])
        seedg = {g: all(r["gates"][g] for r in R) for g in R[0]["gates"]}
        rows[k] = {"identifiability_median": med, "G3_identifiable": g3, "per_seed_gates_all": seedg,
                   "mean_lost": float(np.mean([r["lost_frac"] for r in R])),
                   "max_steps_used": int(max(r["steps_used"] for r in R)),
                   "eligible": bool(g3 and all(seedg.values()))}
        log(f"SELECT lr={lr}: eligible={rows[k]['eligible']} G3 {g3} {med} | per-seed {seedg} | mean lost {rows[k]['mean_lost']:.3f}")
    ok = [k for k in rows if rows[k]["eligible"]]
    sel = min(ok, key=lambda k: abs(rows[k]["mean_lost"] - d2["selection"]["target_lost"])) if ok else None
    res["S3"] = {"grid": rows, "selected": None if sel is None else {
        "lr": float(sel), "max_steps": int(math.ceil(1.5 * rows[sel]["max_steps_used"] / 100.0)) * 100}}
    save_json(RES, res)
    if sel is None:
        log("SELECT: no interference setting satisfies the binding D2 gates -> STOP (D2 kill report; D3 proposal)")
        return STOP
    log(f"SELECT: {res['S3']['selected']} -> D2 store-only kill test PASSED; STOP before any monitor")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config", default=os.path.join(HERE, "stage1_config.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
    if a.phase in ("TRAIN", "S2", "S3"):
        assert a.seed in DEV_SEEDS, "D2 store-only phase: development seeds 9101-9103 only"
        sys.exit({"TRAIN": phase_TRAIN, "S2": phase_S2, "S3": phase_S3}[a.phase](cfg, a.seed) or 0)
    sys.exit({"S1": phase_S1, "SELECT": phase_SELECT}[a.phase](cfg) or 0)
