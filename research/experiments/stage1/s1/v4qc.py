"""Stage-1 v4 (B1) store-only QC, ablations and trivial-decoder audit. NO monitor is trained or evaluated here.

Pre-declared (calibration_plan.md Revision v4; D46-D48):
  F1 redundancy: ablation A (all slots masked), M1 (mean pre-injection h2[A] + memory term), M2 (cross-fitted linear
     readout of the answer from the retrieved memory contribution u = W_o r), integrated store.
  F2 locality: continuous collateral (ratios to X <= 0.10 for margin, log p, KL; |dmargin| <= 0.10 D_C; binary),
     read-site displacement of each Z category <= 0.10 x X's; retrieval leakage mu_X(z) reported.
  F3 diversity: X lost in [.35,.65]; IQR(C_post) >= 1 nat; tiers (retained-strong / weakened / lost) >= 15% each;
     exposure alone: cross-fitted R2(C_post | exposure) <= .50 and max(AUROC, 1-AUROC)(exposure -> lost) <= .80;
     binary-secondary feasibility (approved F6 binding, F6' recorded).
  F4 identifiability: F3-F5 (median over seeds) and the trivial-decoder audit (R2_pre, R2_book, R2_generic,
     R2_output, R2_full, dR2_full|pre, dR2_full|output); kill if within-X R2(C_post | bookkeeping) >= .90.
v4.1 (calibration_plan.md Revision v4.1; D52-D53): dose_report (route-dropped presentation counts vs backup) and the
  pre-declared dose-coherence checks DC-1 (across the p_rd grid) and DC-2 (item level, per seed).
v4.2 (Revision v4.2; D58-D59): stageA_report / f0a_gates (parametric dose feasibility on the Stage-A network) and
  stageB_report / f0b_gates (memory integration on the frozen Stage-A network: store QC, redundancy, rescue,
  no-damage, integration INT-1, scale SC-1/SC-2, route-A equivalence).
"""
import numpy as np

from . import estimands as ES
from . import world as W
from .interventions import (_set_report, generic40, identifiability, item_pre_covariates, natural_gap,
                            near_entity_facts, retention_continuous)
from .matching import displacement
from .memstore import (all_masked, delete_slots, dual_store_qc, h2A_mean, m1_stats, m2_probe, memory_probe,
                       slot_table)
from .metrics import auroc
from .store import answer_stats, name_fluency, probe

V4_CATEGORIES = ("Z_random", "Z_param", "Z_near_entity_X", "Z_near_entity_Y", "Z_near_key", "Z_far")


def base_correct_ev(model, world):
    ev = W.items_where(world, split=W.EV)
    pr = answer_stats(model, world, ev)
    tr = world.known[ev[:, 0], ev[:, 1]]
    bc = ev[tr & pr["correct"]]
    cov = world.mem_covered[bc[:, 0], bc[:, 1]]
    return bc, bc[cov], bc[~cov]


def draw_v4_sets(world, model, sizes, seed):
    """X, Y from memory-covered base-correct EV facts (exposure-stratified); Z = all other base-correct EV facts."""
    rng = np.random.default_rng(seed)
    bc, bc_cov, bc_par = base_correct_ev(model, world)
    fam = world.fam_high
    X = W._stratified_take(rng, bc_cov, fam, sizes["X"] // 2, sizes["X"] - sizes["X"] // 2)
    ids = lambda a: set((a[:, 0] * world.n_rel + a[:, 1]).tolist())
    rest = bc_cov[[i not in ids(X) for i in bc_cov[:, 0] * world.n_rel + bc_cov[:, 1]]]
    Y = W._stratified_take(rng, rest, fam, sizes["Y"] // 2, sizes["Y"] - sizes["Y"] // 2)
    used = ids(X) | ids(Y)
    Z = bc[[i not in used for i in bc[:, 0] * world.n_rel + bc[:, 1]]]
    return {"X": X, "Y": Y, "Z": Z, "bc": bc, "bc_cov": bc_cov, "bc_par": bc_par}


def v4_categories(store0, world, sets, near_frac=0.2):
    X, Y, Z = sets["X"], sets["Y"], sets["Z"]
    slot, _ = slot_table(world)
    cov = world.mem_covered[Z[:, 0], Z[:, 1]]
    cats = {"Z_random": Z, "Z_param": Z[~cov],
            "Z_near_entity_X": near_entity_facts(store0, world, X[:, 0]),
            "Z_near_entity_Y": near_entity_facts(store0, world, Y[:, 0])}
    Zc = Z[cov]
    K = store0.mem_keys.detach()
    Kn = (K / K.norm(dim=-1, keepdim=True)).numpy()
    kx = Kn[slot[X[:, 0], X[:, 1]]]
    kz = Kn[slot[Zc[:, 0], Zc[:, 1]]]
    sim = (kz @ kx.T).max(1)
    k = max(1, int(round(near_frac * len(Zc))))
    order = np.argsort(-sim, kind="stable")
    cats["Z_near_key"], cats["Z_far"] = Zc[order[:k]], Zc[order[-k:]]
    return cats, {"key_sim_near_mean": float(sim[order[:k]].mean()), "key_sim_far_mean": float(sim[order[-k:]].mean())}


def leakage(store0, world, items, X):
    """mu_X(z): pre-deletion retrieval mass that z places on the deleted (X) slots."""
    if not len(items):
        return np.zeros(0)
    slot, _ = slot_table(world)
    a = memory_probe(store0, world, items)["a"]
    return a[:, slot[X[:, 0], X[:, 1]]].sum(1)


def tiers(C_pre, C_post):
    lost = C_post < 0
    strong = C_post >= 0.5 * C_pre
    weak = ~lost & ~strong
    return {"retained_strong": float(strong.mean()), "weakened": float(weak.mean()), "lost": float(lost.mean())}


def delete_qc(store0, post, world, sets, sigma, mu, qcfg, seed, fluency_sd):
    """Per-seed F2 (locality) + F3 (diversity) + Y QC for one intervened store (T-DELETE(X) and Y transplant)."""
    X, Y = sets["X"], sets["Y"]
    cats, ksim = v4_categories(store0, world, sets, qcfg["near_key_fraction"])
    res, pr = {"sets": {}, "key_similarity": ksim}, {}
    for name, items in {"X": X, "Y": Y, **cats}.items():
        if not len(items):
            res["sets"][name] = {"n": 0}
            continue
        p0, p1 = probe(store0, world, items), probe(post, world, items)
        d = displacement(p0["states"], p1["states"], sigma)
        rep = _set_report(p0, p1, d)
        rep["margin_pre_mean"] = float(p0["margin"].mean())
        res["sets"][name] = rep
        pr[name] = (p0, p1, d)
    D_C = natural_gap(store0, world)
    res["D_C"] = D_C
    lv = qcfg["retention_sensitivity"]
    res["retention"] = {c: retention_continuous(res["sets"][c], res["sets"]["X"], D_C, lv) for c in ("Y",) + V4_CATEGORIES}
    res["retention_load_max"] = float(max(v["load"] for v in res["retention"].values()))
    res["retention_pass_by_level"] = {str(l): bool(all(v["pass"][str(l)] for v in res["retention"].values())) for l in lv}
    dX = res["sets"]["X"]["disp_mean"]
    res["displacement_ratio"] = {c: (res["sets"][c]["disp_mean"] / dX if res["sets"][c].get("n") else 0.0) for c in V4_CATEGORIES}
    res["leakage"] = {c: (float(np.mean(leakage(store0, world, cats[c], X))) if len(cats[c]) else 0.0) for c in V4_CATEGORIES}
    res["leakage_Y"] = float(np.mean(leakage(store0, world, Y, X)))
    zl = {c: res["sets"][c].get("lost_frac", 0.0) for c in V4_CATEGORIES}
    res["Z_lost_by_category"] = zl
    # ---- F3 diversity over X
    p0X, p1X, _ = pr["X"]
    C_pre, C_post = p0X["margin"], p1X["margin"]
    q = np.quantile(C_post, [0.25, 0.75])
    res["X_lost_frac"] = float((C_post < 0).mean())
    res["C_post_X_iqr"] = float(q[1] - q[0])
    res["tiers"] = tiers(C_pre, C_post)
    expo = world.fam_high[X[:, 0]].astype(float)
    _, r2_expo, _ = ES.crossfit_ridge(expo[:, None], C_post, seed + 11)
    a_expo = auroc(expo[C_post < 0], expo[C_post >= 0]) if (C_post < 0).any() and (C_post >= 0).any() else 0.5
    res["exposure"] = {"r2_C_post": float(r2_expo), "auroc_lost": float(max(a_expo, 1 - a_expo)),
                       "lost_frac_high": float((C_post[expo == 1] < 0).mean()), "lost_frac_low": float((C_post[expo == 0] < 0).mean())}
    res["C_post_quantiles"] = [float(x) for x in np.quantile(C_post, [0.05, 0.25, 0.5, 0.75, 0.95])]
    res["C_pre_quantiles"] = [float(x) for x in np.quantile(C_pre, [0.05, 0.25, 0.5, 0.75, 0.95])]
    # ---- binary-secondary feasibility (approved F6 binding; F6' recorded -- D43)
    W0, Cp = item_pre_covariates(store0, world, X, p0X)
    GX = generic40(p0X["states"], p1X["states"], mu, sigma)
    idf, P = identifiability(C_post - C_pre, p0X["states"], W0, Cp, GX, seed + 101, strata=W0[:, 1])
    res["identifiability"] = idf
    lostX = C_post < 0
    Xpre = np.column_stack([Cp, W0, P])
    pairs, smd = ES.propensity_match(lostX, Xpre, W0[:, 1], seed + 303, qcfg["binary_secondary"]["caliper_sd"])
    res["binary_secondary"] = {"pairs": len(pairs), "max_abs_smd": smd,
                               "feasible": bool(len(pairs) >= qcfg["binary_secondary"]["min_pairs"] and smd <= qcfg["binary_secondary"]["smd_max"])}
    pairs2, _ = ES.propensity_match(lostX, Xpre, (W0[:, 1] * world.n_rel + X[:, 1]).astype(int), seed + 303,
                                    qcfg["binary_secondary"]["caliper_sd"])
    mx2, mn2 = ES.balance(Xpre, pairs2)
    res["binary_secondary_alt"] = {"pairs": len(pairs2), "max_abs_smd": mx2, "mean_abs_smd": mn2,
                                   "feasible": bool(len(pairs2) >= qcfg["binary_secondary"]["min_pairs"] and mx2 <= 0.25 and mn2 <= 0.10)}
    nm = world.names[np.concatenate([X[:, 0], Y[:, 0]])]
    res["fluency_change_sd"] = float((name_fluency(post, world, nm) - name_fluency(store0, world, nm)).mean() / fluency_sd)
    # ---- gates
    lo, hi = qcfg["X_lost_range"]
    t = res["tiers"]
    res["gates_F2"] = {
        "Z_binary": bool(max(zl.values()) <= qcfg["Z_lost_max"]),
        "retention": res["retention_pass_by_level"]["0.1"],
        "displacement": bool(max(res["displacement_ratio"].values()) <= qcfg["displacement_ratio_max"]),
        "fluency": bool(abs(res["fluency_change_sd"]) <= qcfg["familiarity_change_max_sd"])}
    res["gates_Y"] = {"Y_binary": bool(res["sets"]["Y"]["lost_frac"] <= qcfg["Y_lost_max"]),
                      "Y_retention": bool(res["retention"]["Y"]["pass"]["0.1"])}
    res["gates_F3"] = {
        "lost_range": bool(lo <= res["X_lost_frac"] <= hi),
        "iqr": bool(res["C_post_X_iqr"] >= qcfg["C_post_iqr_min"]),
        "tiers": bool(min(t.values()) >= qcfg["tier_min"]),
        "exposure": bool(res["exposure"]["r2_C_post"] <= qcfg["exposure_r2_max"] and res["exposure"]["auroc_lost"] <= qcfg["exposure_auroc_max"]),
        "F6": res["binary_secondary"]["feasible"]}
    for k in ("gates_F2", "gates_Y", "gates_F3"):
        res[k.replace("gates", "pass")] = bool(all(res[k].values()))
    return res, {"pre_X": p0X, "post_X": p1X, "GX": GX, "W0": W0, "C_pre": Cp, "P": P}


def audit(store0, post, world, sets, aux, seed):
    """Trivial-decoder audit (store-only): how predictable is post-deletion competence, and from what?
    Cross-fitted ridge R^2 of C_post (and of dC) within X from each information source, plus incremental terms."""
    X = sets["X"]
    p0, p1 = aux["pre_X"], aux["post_X"]
    C_pre, C_post = p0["margin"], p1["margin"]
    dC = C_post - C_pre
    strata = aux["W0"][:, 1]
    mp1 = memory_probe(post, world, X)
    mp0 = memory_probe(store0, world, X)
    book = np.column_stack([mp1["a_null"], mp1["a_max"], mp1["entropy"], np.linalg.norm(mp1["u"], axis=1),
                            np.linalg.norm(mp1["u"] - mp0["u"], axis=1)])
    pre = np.column_stack([p0["states"].reshape(len(X), -1), aux["W0"], C_pre])
    sp = -np.sort(-p1["probs"], 1)
    out = np.column_stack([sp, p1["entropy"]])
    full = p1["states"].reshape(len(X), -1)
    gen = aux["GX"]
    srcs = {"pre": pre, "book": book, "generic": gen, "output": out, "full": full,
            "full+pre": np.column_stack([full, pre]), "full+output": np.column_stack([full, out])}
    res = {}
    for tname, y in (("C_post", C_post), ("dC", dC)):
        r = {k: float(ES.crossfit_ridge(v, y, seed + i, strata)[1]) for i, (k, v) in enumerate(srcs.items())}
        r["dR2_full_given_pre"] = r["full+pre"] - r["pre"]
        r["dR2_full_given_output"] = r["full+output"] - r["output"]
        res[tname] = r
    # deletion status + bookkeeping across X and untouched Z (deletion status varies there)
    Z = sets["Z"]
    mz1 = memory_probe(post, world, Z)
    pz1 = answer_stats(post, world, Z)
    bookZ = np.column_stack([mz1["a_null"], mz1["a_max"], mz1["entropy"], np.linalg.norm(mz1["u"], axis=1),
                             np.linalg.norm(mz1["u"] - memory_probe(store0, world, Z)["u"], axis=1)])
    status = np.concatenate([np.ones(len(X)), np.zeros(len(Z))])
    yy = np.concatenate([C_post, pz1["margin"]])
    res["across_XZ"] = {"r2_status_only": float(ES.crossfit_ridge(status[:, None], yy, seed + 50)[1]),
                        "r2_status_plus_book": float(ES.crossfit_ridge(np.column_stack([status, np.vstack([book, bookZ])]), yy, seed + 51)[1])}
    res["kill_bookkeeping"] = bool(res["C_post"]["book"] >= 0.90)
    return res


def ablation_report(model, world, seed, m2_n=3000):
    """F1 redundancy: A (parametric-only), M1, M2, integrated -- on memory-covered base-correct EV facts; plus
    parametric-only facts and unknown facts."""
    bc, bc_cov, bc_par = base_correct_ev(model, world)
    A = all_masked(model)
    a_cov = answer_stats(A, world, bc_cov)
    q = np.quantile(a_cov["margin"], [0.25, 0.75])
    kn = W.items_where(world, known=True)
    par = kn[~world.mem_covered[kn[:, 0], kn[:, 1]]]
    cov_all = kn[world.mem_covered[kn[:, 0], kn[:, 1]]]
    mean_vec = h2A_mean(model, world, bc_cov)
    m1 = m1_stats(model, world, bc_cov, mean_vec)
    rng = np.random.default_rng(seed)
    m2_items = cov_all[rng.permutation(len(cov_all))[:m2_n]]
    m2 = m2_probe(model, world, m2_items, seed + 1)
    m2_par = m2_probe(model, world, par[rng.permutation(len(par))[:min(m2_n, len(par))]], seed + 2)
    integ = answer_stats(model, world, bc_cov)
    return {"n_bc_cov": int(len(bc_cov)), "n_bc_par": int(len(bc_par)),
            "A_acc_covered_bc": float(a_cov["correct"].mean()), "A_margin_iqr_covered_bc": float(q[1] - q[0]),
            "A_margin_quantiles": [float(x) for x in np.quantile(a_cov["margin"], [0.05, 0.25, 0.5, 0.75, 0.95])],
            "A_acc_param_only_trained": float(answer_stats(A, world, par)["correct"].mean()),
            "integrated_acc_param_only_trained": float(answer_stats(model, world, par)["correct"].mean()),
            "integrated_acc_covered_bc": float(integ["correct"].mean()),
            "M1_acc_covered_bc": float(m1["correct"].mean()), "M2_acc_covered": m2, "M2_acc_param_only_control": m2_par,
            "A_exposure_acc_high": float(a_cov["correct"][world.fam_high[bc_cov[:, 0]]].mean()),
            "A_exposure_acc_low": float(a_cov["correct"][~world.fam_high[bc_cov[:, 0]]].mean())}


def f1_eligible(rep, qc, qcfg):
    lo, hi = qcfg["A_acc_range"]
    mem_ok = rep["M1_acc_covered_bc"] >= qcfg["memory_sufficiency_min"] or rep["M2_acc_covered"] >= qcfg["memory_sufficiency_min"]
    g = {"store_qc": bool(qc["pass"]), "A_range": bool(lo <= rep["A_acc_covered_bc"] <= hi),
         "A_iqr": bool(rep["A_margin_iqr_covered_bc"] >= qcfg["A_margin_iqr_min"]), "memory_sufficient": bool(mem_ok),
         "param_only_answerable": bool(rep["integrated_acc_param_only_trained"] >= qcfg["param_only_acc_min"])}
    g["M1_M2_discrepancy"] = bool((rep["M1_acc_covered_bc"] >= qcfg["memory_sufficiency_min"]) != (rep["M2_acc_covered"] >= qcfg["memory_sufficiency_min"]))
    return all(v for k, v in g.items() if k != "M1_M2_discrepancy"), g


# ---------------------------------------------------------------- v4.1 dose diagnostic (D53; store-only)
def dose_report(model, world, drop_count, p_rd, epochs, sizes, seed, dc2_alpha=0.01):
    """Dose diagnostic for one store. drop_count [S]: route-dropped presentations per slot during training (the only
    presentations through which a memory-covered fact can teach the parametric group under v4.1). Population: the
    memory-covered base-correct EV facts (as for F1). Route dropout is randomised per presentation, so the per-fact
    count is randomly assigned and its association with backup is a randomised dose-response.
    DC-2 (item level): Spearman rho(count, route-A margin) > 0 with one-sided p < dc2_alpha."""
    bc, bc_cov, _ = base_correct_ev(model, world)
    slot, _ = slot_table(world)
    cnt = np.asarray(drop_count)[slot[bc_cov[:, 0], bc_cov[:, 1]]]
    a = answer_stats(all_masked(model), world, bc_cov)
    rho, p, dc2 = dc2_check(cnt, a["margin"], dc2_alpha)
    by_count = {}
    for c in np.unique(cnt):
        k = cnt == c
        by_count[str(int(c))] = {"n": int(k.sum()), "A_acc": float(a["correct"][k].mean()),
                                 "A_margin_mean": float(a["margin"][k].mean())}
    try:                                                   # the F2/F3 X draw (needs enough base-correct covered facts)
        sets = draw_v4_sets(world, model, sizes, seed)
        lost = float(1.0 - answer_stats(delete_slots(model, world, sets["X"]), world, sets["X"])["correct"].mean())
        n_x = int(len(sets["X"]))
    except ValueError as e:
        lost, n_x = None, f"X draw impossible: {e}"
    allc = np.asarray(drop_count)
    return {"p_rd": float(p_rd), "expected_count": float(p_rd * epochs),
            "count_mean_all_covered": float(allc.mean()), "count_sd_all_covered": float(allc.std()),
            "count_mean_bc_cov": float(cnt.mean()), "count_quantiles_bc_cov": [float(x) for x in np.quantile(cnt, [0.05, 0.25, 0.5, 0.75, 0.95])],
            "A_acc": float(a["correct"].mean()), "A_margin_mean": float(a["margin"].mean()),
            "A_margin_median": float(np.median(a["margin"])), "X_lost_after_T_DELETE": lost, "n_X": n_x,
            "spearman_count_vs_A_margin": float(rho), "spearman_p_one_sided": float(p),
            "DC2_pass": dc2, "by_count": by_count, "n": int(len(bc_cov))}


def dc2_check(count, margin, alpha=0.01):
    """DC-2 (item-level randomised dose-response, pre-declared): Spearman rho(route-drop count, route-A margin) > 0
    with one-sided p < alpha. Returns (rho, p, pass)."""
    from scipy import stats
    rho, p = stats.spearmanr(count, margin, alternative="greater")
    return float(rho), float(p), bool(rho > 0 and p < alpha)


def dc1_check(grid_acc, tol=0.05, min_range=0.20):
    """DC-1 (aggregate dose-response across the fixed p_rd grid, pre-declared): route-A accuracy is broadly increasing
    -- for every pair p_i < p_j, acc(p_j) >= acc(p_i) - tol -- AND acc(max p) - acc(min p) >= min_range.
    grid_acc: {p_rd: route-A accuracy}."""
    ps = sorted(grid_acc)
    worst = max([grid_acc[ps[i]] - grid_acc[ps[j]] for i in range(len(ps)) for j in range(i + 1, len(ps))] + [-np.inf])
    rng_ = grid_acc[ps[-1]] - grid_acc[ps[0]]
    from scipy import stats
    rho = float(stats.spearmanr(ps, [grid_acc[p] for p in ps])[0]) if len(ps) > 2 else float("nan")
    return {"max_pairwise_decrease": float(worst), "range": float(rng_), "spearman_p_rd_vs_A_acc": rho,
            "monotone_within_tol": bool(worst <= tol), "range_ok": bool(rng_ >= min_range),
            "DC1_pass": bool(worst <= tol and rng_ >= min_range)}


# ---------------------------------------------------------------- v4.2 (D58-D59; store-only)
def _trained(world, covered, split=None):
    it = W.items_where(world, known=True, split=split)
    return it[world.mem_covered[it[:, 0], it[:, 1]] == covered]


def stageA_report(modelA, world, presentations, p_rd, epochs, dc2_alpha=0.01):
    """V4.2-F0A report on the Stage-A network (memory never used; all slots masked == memory_off exactly).
    Route-A population: memory-covered TRAINED EV facts (base-correctness is not defined before Stage B)."""
    A = all_masked(modelA)
    cov_ev = _trained(world, True, W.EV)
    par = _trained(world, False)
    slot, _ = slot_table(world)
    cnt = np.asarray(presentations)[slot[cov_ev[:, 0], cov_ev[:, 1]]]
    a = answer_stats(A, world, cov_ev)
    q = np.quantile(a["margin"], [0.05, 0.25, 0.5, 0.75, 0.95])
    rho, pval, dc2 = dc2_check(cnt, a["margin"], dc2_alpha)
    hi = world.fam_high[cov_ev[:, 0]]
    by_count = {}
    for c in np.unique(cnt):
        k = cnt == c
        by_count[str(int(c))] = {"n": int(k.sum()), "A_acc": float(a["correct"][k].mean()),
                                 "A_margin_mean": float(a["margin"][k].mean())}
    flu = name_fluency(modelA, world, world.names)
    allc = np.asarray(presentations)
    return {"p_rd": float(p_rd), "n": int(len(cov_ev)),
            "A_acc": float(a["correct"].mean()), "A_margin_mean": float(a["margin"].mean()),
            "A_margin_median": float(q[2]), "A_margin_iqr": float(q[3] - q[1]), "A_margin_quantiles": [float(x) for x in q],
            "A_acc_exposure_high": float(a["correct"][hi].mean()), "A_acc_exposure_low": float(a["correct"][~hi].mean()),
            "expected_presentations": float(p_rd * epochs), "presentations_mean_all_covered": float(allc.mean()),
            "presentations_sd_all_covered": float(allc.std()), "presentations_mean_pop": float(cnt.mean()),
            "presentations_quantiles_pop": [float(x) for x in np.quantile(cnt, [0.05, 0.25, 0.5, 0.75, 0.95])],
            "spearman_count_vs_A_margin": rho, "spearman_p_one_sided": pval, "DC2_pass": dc2, "by_count": by_count,
            "param_only_trained_acc": float(answer_stats(A, world, par)["correct"].mean()),
            "unknown_acc": float(answer_stats(A, world, W.items_where(world, known=False))["correct"].mean()),
            "fluency_auroc_high_vs_low": float(auroc(flu[world.fam_high], flu[~world.fam_high]))}


def f0a_gates(rep, cfg_v4):
    """Per-p_rd F0A eligibility (pre-declared): Route-A acc in [.30,.70] and IQR >= 1 nat (existing F1 rule);
    parametric-only facts sufficiently learned (existing param_only_acc_min .90); Stage-A store sanity (unknown <= .10,
    fluency AUROC >= .80; existing store QC values)."""
    f1, qc = cfg_v4["F1"], cfg_v4["store_qc"]
    lo, hi = f1["A_acc_range"]
    g = {"A_range": bool(lo <= rep["A_acc"] <= hi), "A_iqr": bool(rep["A_margin_iqr"] >= f1["A_margin_iqr_min"]),
         "param_only_learned": bool(rep["param_only_trained_acc"] >= f1["param_only_acc_min"]),
         "unknown": bool(rep["unknown_acc"] <= qc["unknown_fact_acc_max"]),
         "fluency": bool(rep["fluency_auroc_high_vs_low"] >= qc["fluency_auroc_min"])}
    return all(g.values()), g


def stageB_report(modelB, modelA, world, seed, cfg_v4, dc2_alpha=0.01):
    """V4.2-F0B report: the integrated store after Stage B versus its own frozen Stage-A network."""
    from scipy import stats
    b = cfg_v4["F0B"]
    kn = W.items_where(world, known=True)
    cov, par = _trained(world, True), _trained(world, False)
    qc = dual_store_qc(modelB, world, cfg_v4["store_qc"])
    rep = ablation_report(modelB, world, seed)
    okf1, gf1 = f1_eligible(rep, qc, cfg_v4["F1"])
    a0 = answer_stats(all_masked(modelA), world, cov)            # Stage-A (parametric) status of each covered fact
    a1 = answer_stats(modelB, world, cov)                        # integrated
    m0, m1 = a0["margin"], a1["margin"]
    tiers = {"stageA_correct_strong (m>=1)": m0 >= 1, "stageA_weak (|m|<1)": np.abs(m0) < 1,
             "stageA_wrong_strong (m<=-1)": m0 <= -1, "stageA_correct (m>0)": m0 > 0, "stageA_wrong (m<=0)": m0 <= 0}
    tier_rep = {k: {"n": int(v.sum()), "integrated_acc": float(a1["correct"][v].mean()) if v.any() else None,
                    "integrated_margin_mean": float(m1[v].mean()) if v.any() else None,
                    "stageA_margin_mean": float(m0[v].mean()) if v.any() else None} for k, v in tiers.items()}
    rho, pint = stats.spearmanr(m0, m1, alternative="greater")
    par0 = float(answer_stats(all_masked(modelA), world, par)["correct"].mean())
    par1 = float(answer_stats(modelB, world, par)["correct"].mean())
    mpc, mpp = memory_probe(modelB, world, cov), memory_probe(modelB, world, par)
    uc, up = np.linalg.norm(mpc["u"], axis=1), np.linalg.norm(mpp["u"], axis=1)
    cap = float(modelB.inj_cap)
    null_inj = float(np.linalg.norm(modelB.W_o((modelB.mem_values * modelB.value_mask[:, None])[-1]).detach().numpy()))
    toks = W.query_tokens(world.vocab, world.names[kn[:600, 0]], kn[:600, 1])
    import torch
    with torch.no_grad():
        eq = bool(torch.equal(all_masked(modelB)(torch.as_tensor(toks)), modelA(torch.as_tensor(toks), memory_off=True)))
    scale = {"v_null_effective_norm": float((modelB.mem_values * modelB.value_mask[:, None])[-1].norm()),
             "W_o_fro": float(modelB.W_o.weight.norm()), "W_o_v_null_norm": null_inj,
             "slot_value_norm_mean": float(modelB.mem_values[:-1].norm(dim=1).mean()),
             "injection_cap": cap, "hA_norm_median_covered": float(np.median(mpc["hA_norm"])),
             "u_norm_median_covered": float(np.median(uc)), "u_norm_median_param_only": float(np.median(up)),
             "u_over_hA_median_covered": float(np.median(uc / mpc["hA_norm"])),
             "u_over_hA_median_param_only": float(np.median(up / mpp["hA_norm"])),
             "frac_covered_at_cap": float(np.mean(uc >= 0.999 * cap))}
    out = {"store_qc": qc, "ablation": rep, "f1_gates": gf1, "f1_eligible": okf1, "tiers": tier_rep,
           "covered_integrated_acc": float(a1["correct"].mean()), "param_only_acc_stageA": par0,
           "param_only_acc_integrated": par1, "INT1_spearman_stageA_vs_integrated_margin": float(rho),
           "INT1_p_one_sided": float(pint), "scale": scale, "routeA_equivalent_to_stageA": eq}
    g = {"store_qc": bool(qc["pass"]), "f1_redundancy": bool(okf1),
         "covered_integrated": bool(out["covered_integrated_acc"] >= b["covered_integrated_acc_min"]),
         "rescue": bool(tier_rep["stageA_wrong (m<=0)"]["n"] == 0
                        or tier_rep["stageA_wrong (m<=0)"]["integrated_acc"] >= b["rescue_acc_min"]),
         "no_damage_covered": bool(tier_rep["stageA_correct (m>0)"]["n"] == 0
                                   or tier_rep["stageA_correct (m>0)"]["integrated_acc"] >= b["no_damage_acc_min"]),
         "no_damage_param_only": bool(par1 >= par0 - b["param_only_drop_max"]),
         "INT1_integration": bool(rho > 0 and pint < b["int1_alpha"]),
         "SC2_null_route_injection": bool(scale["u_over_hA_median_param_only"] <= b["sc2_null_u_over_hA_max"]),
         "SC1_covered_injection_scale": bool(scale["u_over_hA_median_covered"] <= b["sc1_u_over_hA_max"]),
         "routeA_equivalence": eq}
    out["gates"] = g
    out["pass"] = all(g.values())
    return out
