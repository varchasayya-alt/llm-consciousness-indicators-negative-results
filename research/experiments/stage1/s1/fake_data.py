"""FABRICATED results for the analysis dry run (v3).

Every number here is drawn from hand-specified random distributions (s1/sim_worlds.py for the primary
within-target populations). Nothing is derived from any Stage-1 model, store, monitor, calibration run or
confirmatory run. The worlds exist only to check that the frozen analysis (tests, Holm families, A/B/C
interpretation labels, identifiability checks, exclusions, tables, figures) behaves correctly:
  H3_strong, H3, H3_plus_generic   -> P1/P2 supported, label A
  H2_generic, H2_intensity         -> theta_pre supported but label B (never A)
  susceptibility, rtm, null        -> label C
  susceptibility_latent            -> stress test (documented limitation; not asserted)
"""
import json
import os

import numpy as np

from .sim_worlds import H3_LIKE, make_structure, simulate_items

CONDS = ["IN", "OUT", "INT-S", "INT-T", "INT-P", "INT-TP", "INT-S-shuffled", "OUT-TP", "INT+OUT-S"]
SCENARIO = {"name": "H3"}
STRUCT = make_structure(np.random.default_rng(12345))
LOSS_SHIFT = 1.5        # about half of the targeted items lose competence (v3 outcome-diversity range)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _population(n, base, scen):
    """Primary population with per-monitor scores (all monitors on the SAME simulated items)."""
    world = "rtm" if scen == "rtm" else scen
    mk = lambda s: simulate_items(s, n, np.random.default_rng(base), structure=STRUCT, loss_shift=LOSS_SHIFT)
    pop = mk(world)
    sc = {}
    for c in ("INT-S", "INT-T", "INT-P", "INT+OUT-S"):
        r = mk(world)
        sc[c] = (r["M_pre"], r["M_post"])
    tp = mk("H3_strong" if scen in H3_LIKE else world)
    sc["INT-TP"] = (tp["M_pre"], tp["M_post"])
    rng = np.random.default_rng(base + 1)
    out = lambda C, nz: _sig(2.0 + 0.8 * (C - 5.9) + nz)
    nz = rng.normal(0, 0.3, n)
    sc["OUT"] = (out(pop["C_pre"], nz), out(pop["C_post"], nz + rng.normal(0, 0.2, n)))
    sc["OUT-TP"] = sc["OUT"]
    pre = sc["INT-S"][0]
    sc["INT-S-shuffled"] = (pre, np.clip(pre + rng.normal(0, 0.02, n), 1e-3, 1 - 1e-3))
    sc["IN"] = (np.clip(rng.beta(9, 1.2, n), 1e-3, 1 - 1e-3),) * 2
    disp = np.exp(0.3 * pop["G"][:, np.arange(10) % 16] - 1.0)
    gen = pop["G"][:, (10 + np.arange(30)) % 16] + rng.normal(0, 0.05, (n, 30))
    return pop, sc, disp, gen


def _lookups(sc):
    lk = {c: (_sig((0.85 - sc[c][0]) * 20), _sig((0.85 - sc[c][1]) * 20)) for c in ("IN", "OUT", "INT-S", "INT-T", "INT-P", "INT-TP")}
    lk["INT-S+in"] = (_sig((0.85 - sc["INT-S"][0]) * 10), _sig((0.85 - sc["INT-S"][1]) * 10))
    return lk


def _rows(seed, store, interv, sets, fam, pre_c, post_c, pre_m, post_m, scores, lookups, rng, extra=None,
          disp=None, gen=None, relation=None, pre_logp=None, post_logp=None, fluency=None):
    n = len(sets)
    relation = rng.integers(0, 4, n) if relation is None else relation
    pre_logp = -np.log1p(31 * np.exp(-np.asarray(pre_m))) if pre_logp is None else pre_logp
    post_logp = -np.log1p(31 * np.exp(-np.asarray(post_m))) if post_logp is None else post_logp
    fluency = rng.normal(-2, 0.5, n) if fluency is None else fluency
    rows = []
    for k in range(n):
        r = {"seed": seed, "store": store, "intervention": interv, "set": str(sets[k]), "item": int(k),
             "fam_high": bool(fam[k]), "pre_correct": bool(pre_c[k]), "post_correct": bool(post_c[k]),
             "pre_margin": float(pre_m[k]), "post_margin": float(post_m[k]), "relation": int(relation[k]),
             "pre_logp": float(pre_logp[k]), "post_logp": float(post_logp[k]), "fluency_pre": float(fluency[k]),
             "post_maxprob": float(rng.uniform(0.2, 0.95)),
             "disp": None if disp is None else [float(x) for x in disp[k]],
             "gen": None if gen is None else [float(x) for x in gen[k]],
             "scores": {c: [float(a[k]), float(b[k])] for c, (a, b) in scores.items()},
             "lookup": {c: [float(a[k]), float(b[k])] for c, (a, b) in lookups.items()}}
        if extra:
            for key, arr in extra.items():
                r[key] = arr[k].item() if hasattr(arr[k], "item") else arr[k]
        rows.append(r)
    return rows


def _static_set(rng, n, conds, change_sd=0.02, margin_drop=0.2):
    pre_m = rng.normal(5.9, 0.9, n)
    post_m = pre_m - np.abs(rng.normal(margin_drop, 0.1, n))
    sc = {}
    for c in conds:
        pre = np.clip(rng.beta(9, 1.2, n), 1e-3, 1 - 1e-3)
        sc[c] = (pre, np.clip(pre + rng.normal(0, change_sd, n), 1e-3, 1 - 1e-3))
    return pre_m, post_m, sc


def make_fake_seed(seed, out_dir, excluded=False):
    rng = np.random.default_rng(10_000_000 + seed)     # fabricated; unrelated to any model seed
    scen = SCENARIO["name"]
    os.makedirs(out_dir, exist_ok=True)
    meta = {"seed": seed, "fake": True, "excluded": excluded, "scenario": scen}
    if excluded:
        meta["reason"] = "store QC (fabricated)"
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
        open(os.path.join(out_dir, "items.jsonl"), "w").close()
        open(os.path.join(out_dir, "natural.jsonl"), "w").close()
        return
    rows, pre_states = [], {}
    # ---- FORGET A: X from the simulated world; Y, Z static references
    pop, sc, disp, gen = _population(300, 20_000_000 + seed * 10, scen)
    lostX = pop["C_post"] < 0
    rows += _rows(seed, "A", "FORGET", np.array(["X"] * 300), pop["e"].astype(bool), np.ones(300, bool), ~lostX,
                  pop["C_pre"], pop["C_post"], sc, _lookups(sc), rng,
                  extra={"pair": np.where(lostX, np.arange(300), -1), "h_idx": np.arange(300),
                         "twin_level_INT-S": np.clip(rng.beta(9, 1.2, 300), 1e-3, 1 - 1e-3)},
                  disp=disp, gen=gen, relation=pop["relation"], pre_logp=pop["logp_pre"], post_logp=pop["logp_post"],
                  fluency=pop["fluency"])
    pre_states["FORGET_A"] = pop["H_pre"].astype(np.float16)
    for name, n in (("Y", 300), ("Z", 800)):
        pm, qm, scs = _static_set(rng, n, CONDS)
        dY = np.exp(0.3 * rng.normal(0.3, 1, (n, 10)) - 1.0)
        gY = rng.normal(0.2 if name == "Y" else 0.0, 1, (n, 30))
        pair = np.where(np.arange(n) < int(lostX.sum()), np.arange(n), -1) if name == "Y" else -np.ones(n, int)
        rows += _rows(seed, "A", "FORGET", np.array([name] * n), rng.random(n) < 0.5, np.ones(n, bool), np.ones(n, bool),
                      pm, qm, scs, _lookups(scs), rng, extra={"pair": pair, "h_idx": -np.ones(n, int),
                                                             "twin_level_INT-S": np.clip(rng.beta(9, 1.2, n), 1e-3, 1 - 1e-3)},
                      disp=dY, gen=gY)
    # ---- INTERF A: all base-correct items equally exposed (simulated world)
    nI = 1000
    popI, scI, dI, gI = _population(nI, 30_000_000 + seed * 10, scen)
    lostI = popI["C_post"] < 0
    pairI = -np.ones(nI, int)
    li, ri = np.nonzero(lostI)[0], np.nonzero(~lostI)[0]
    m = min(len(li), len(ri), 150)
    pairI[li[:m]] = np.arange(m)
    pairI[ri[:m]] = np.arange(m)
    rows += _rows(seed, "A", "INTERF", np.where(lostI, "lost", "retained"), popI["e"].astype(bool), np.ones(nI, bool), ~lostI,
                  popI["C_pre"], popI["C_post"], scI, _lookups(scI), rng, extra={"pair": pairI, "h_idx": np.arange(nI)},
                  disp=dI, gen=gI, relation=popI["relation"], pre_logp=popI["logp_pre"], post_logp=popI["logp_post"],
                  fluency=popI["fluency"])
    pre_states["INTERF_A"] = popI["H_pre"].astype(np.float16)
    # ---- NEW, FAM, CORRUPT, ACT, REPLACE (A): simple fabricated references
    sets = np.array(["N"] * 240 + ["U"] * 240)
    learned = (sets == "N") & (rng.random(480) < 0.85)
    scN = {}
    for c in CONDS:
        pre = np.clip(rng.beta(2, 5, 480), 1e-3, 1 - 1e-3)
        up = 0.25 if (c != "IN" and scen in H3_LIKE) else 0.0
        scN[c] = (pre, np.clip(pre + np.where(learned, rng.normal(up, 0.1, 480), rng.normal(0, 0.03, 480)), 1e-3, 1 - 1e-3))
    rows += _rows(seed, "A", "NEW", sets, np.arange(480) % 2 == 0, np.zeros(480, bool), learned,
                  rng.normal(-6, 1, 480), np.where(learned, rng.normal(3, 1, 480), rng.normal(-6, 1, 480)), scN, _lookups(scN), rng)
    sets = np.array(["F"] * 300 + ["U2"] * 300)
    scM = {}
    for c in CONDS:
        pre = np.clip(rng.beta(2, 5, 600), 1e-3, 1 - 1e-3)
        scM[c] = (pre, np.clip(pre + rng.normal(0, 0.03, 600), 1e-3, 1 - 1e-3))
    rows += _rows(seed, "A", "FAM", sets, np.zeros(600, bool), np.zeros(600, bool), np.zeros(600, bool),
                  rng.normal(-6, 1, 600), rng.normal(-6, 1, 600), scM, _lookups(scM), rng)
    sets = np.array(["C"] * 240)
    scC = {c: (np.clip(rng.beta(9, 1.2, 240), 1e-3, 1 - 1e-3), np.clip(rng.beta(2, 5, 240), 1e-3, 1 - 1e-3)) for c in CONDS}
    rows += _rows(seed, "A", "CORRUPT", sets, rng.random(240) < 0.6, np.ones(240, bool), rng.random(240) < 0.03,
                  rng.normal(5.9, 0.9, 240), rng.normal(-6, 1, 240), scC, _lookups(scC), rng)
    for ep in ("ACT", "REPLACE"):
        sets = np.array(["X"] * 300 + ["Y"] * 300)
        lost = (sets == "X") & (rng.random(600) < 0.7)
        pm, qm, scE = _static_set(rng, 600, CONDS)
        rows += _rows(seed, "A", ep, sets, rng.random(600) < 0.6, np.ones(600, bool), ~lost, pm,
                      np.where(lost, -0.5, qm), scE, _lookups(scE), rng)
    # ---- B FORGET: targeted set X_B with self-specificity score keys
    popB, scB0, dB, gB = _population(300, 40_000_000 + seed * 10, scen)
    lostB = popB["C_post"] < 0
    scB = {}
    for c in ("INT-S", "INT-TP"):
        for key in ("B->B", "Bal->Bal"):
            scB[f"{key}:{c}"] = scB0[c]
        scB[f"A->Bal:{c}"] = (scB0[c][0], np.clip(scB0[c][1] + rng.normal(0, 0.01, 300), 1e-3, 1 - 1e-3))
        pre = scB0[c][0]
        scB[f"A->Braw:{c}"] = (pre, np.clip(pre + rng.normal(0, 0.02, 300), 1e-3, 1 - 1e-3))
    rows += _rows(seed, "B", "FORGET", np.array(["X"] * 300), popB["e"].astype(bool), np.ones(300, bool), ~lostB,
                  popB["C_pre"], popB["C_post"], scB, {}, rng, extra={"pair": -np.ones(300, int), "h_idx": np.arange(300)},
                  disp=dB, gen=gB, relation=popB["relation"], pre_logp=popB["logp_pre"], post_logp=popB["logp_post"],
                  fluency=popB["fluency"])
    pre_states["FORGET_B"] = popB["H_pre"].astype(np.float16)
    with open(os.path.join(out_dir, "items.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(os.path.join(out_dir, "prestates.npz"), **pre_states)
    n = 3000
    trained = rng.random(n) < 0.55
    correct = trained | (rng.random(n) < 0.03)
    with open(os.path.join(out_dir, "natural.jsonl"), "w") as f:
        for k in range(n):
            s = {c: float(np.clip(rng.beta(9, 1.2) if correct[k] else rng.beta(2, 5), 1e-3, 1 - 1e-3)) for c in CONDS}
            f.write(json.dumps({"seed": seed, "item": k, "fam_high": bool(rng.random() < 0.5), "trained": bool(trained[k]),
                                "correct": bool(correct[k]), "scores": s}) + "\n")
    meta["qc"] = {"forget": {"site_ratios_X_over_Y": list(rng.uniform(0.9, 1.1, 10))}}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))


def make_fake_dataset(root, n_valid=22, excluded_seeds=(1003, 1011), scenario="H3"):
    SCENARIO["name"] = scenario
    seed, made = 1001, 0
    while made < n_valid:
        exc = seed in excluded_seeds
        make_fake_seed(seed, os.path.join(root, f"seed_{seed}"), excluded=exc)
        if not exc:
            made += 1
        seed += 1
