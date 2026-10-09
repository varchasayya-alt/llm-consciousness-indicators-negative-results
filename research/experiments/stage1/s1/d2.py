"""Stage-1 D2: self-generated competence change through continued learning (STORE-ONLY kill test).

The intervention is development itself: the plain v1-v3 parametric store keeps learning new facts about new entities
with no replay (interventions.interference). Every base-correct EV fact experiences the same process.
NO explicit memory, deletion, unlearning, targeted edit, transplant or any B1 component; NO monitor (unit-tested).
Rules: d2_protocol.md, d2_S0_satisfiability_note.md, decisions D62-D65.
"""
import numpy as np
import torch
from scipy import stats

from . import estimands as ES
from . import world as W
from .interventions import _train_on, generic40, identifiability, item_pre_covariates
from .matching import displacement
from .metrics import auroc
from .store import answer_stats, name_fluency, probe


def population(store0, world):
    """Base-correct EV facts of the base store (all equally exposed to the continued learning)."""
    ev = W.items_where(world, split=W.EV)
    pr = answer_stats(store0, world, ev)
    tr = world.known[ev[:, 0], ev[:, 1]]
    return ev[tr & pr["correct"]]


def eval_tokens(world, items):
    """The evaluation input of each item ([Q s1 s2 r A]); a pure function of the world (input identity, G5)."""
    return W.query_tokens(world.vocab, world.names[items[:, 0]], items[:, 1])


def control_sequences(world):
    """CTRL material: already-known material EXCLUDING the evaluated (EV-split) facts -- MT/CT known-fact sequences
    plus all original mention sequences (approved multiplicities)."""
    seqs, is_fact, idx = W.training_sequences(world)
    split = world.split.reshape(-1)[np.maximum(idx, 0)]
    keep = ~is_fact | (split != W.EV)
    return seqs[keep]


def control_continuation(store0, world, steps, icfg, seed):
    """Matched-development control (not a sham): same optimizer (Adam), lr and batch size as INTERF and exactly
    `steps` steps (yoked to INTERF on the same seed and lr), on CTRL material. Returns the continued store."""
    seqs = torch.as_tensor(control_sequences(world))
    bs = int(icfg["batch_size"])

    def seq_fn(step, g):
        return seqs[torch.randint(0, len(seqs), (bs,), generator=g)]

    model, _ = _train_on(store0, seq_fn, int(steps), icfg["lr"], seed, world)
    return model


def steps_used(trace, max_steps):
    return int(trace[-1]["step"]) if trace else int(max_steps)


def bookkeeping_features(world, items):
    """Intervention bookkeeping (world/data only; never model-derived): overlap of each item with the interference
    data -- # interference facts with the item's relation and answer value, # with its relation, # interference
    entities sharing its first / second syllable, exposure class."""
    ir = world.interf_rels
    ents = np.repeat(np.arange(len(world.interf_names)), ir.shape[1])
    rr = ir.reshape(-1)
    iv = world.interf_answers[ents, rr]
    v = world.answers[items[:, 0], items[:, 1]]
    n_rv = np.array([np.sum((rr == r) & (iv == a)) for r, a in zip(items[:, 1], v)], dtype=float)
    n_r = np.array([np.sum(rr == r) for r in items[:, 1]], dtype=float)
    s1, s2 = world.names[items[:, 0], 0], world.names[items[:, 0], 1]
    o1 = np.array([np.sum(world.interf_names[:, 0] == x) for x in s1], dtype=float)
    o2 = np.array([np.sum(world.interf_names[:, 1] == x) for x in s2], dtype=float)
    return np.column_stack([n_rv, n_r, o1, o2, world.fam_high[items[:, 0]].astype(float)])


def _q(x, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    return [float(v) for v in np.quantile(x, qs)]


def _iqr(x):
    q = np.quantile(x, [0.25, 0.75])
    return float(q[1] - q[0])


def d2_report(store0, post, ctrl, world, items, mu, sig, seed, trace, n_steps, gcfg, qc_bin):
    """Store-only D2 report for one (lr, seed): gates G1, G2, G4, G5 (G3 is evaluated on medians across seeds) and
    every report-only diagnostic of d2_protocol.md section 7."""
    tok0 = eval_tokens(world, items)
    p0, p1, pc = probe(store0, world, items), probe(post, world, items), probe(ctrl, world, items)
    tok1 = eval_tokens(world, items)
    C_pre, C_post, C_ctrl = p0["margin"], p1["margin"], pc["margin"]
    dC, dCc = C_post - C_pre, C_ctrl - C_pre
    W0, Cp = item_pre_covariates(store0, world, items, p0)
    strata = W0[:, 1]
    G = generic40(p0["states"], p1["states"], mu, sig)
    idf, P = identifiability(dC, p0["states"], W0, Cp, G, seed + 101, strata=strata)
    lost = C_post < 0
    # ---- trivial-decoder audit (C_post and dC from each information source)
    book = bookkeeping_features(world, items)
    pre = np.column_stack([p0["states"].reshape(len(items), -1), W0, Cp])
    sp = -np.sort(-p1["probs"], 1)
    out = np.column_stack([sp, p1["entropy"]])
    full = p1["states"].reshape(len(items), -1)
    srcs = {"pre": pre, "book": book, "generic": G, "output": out, "full": full,
            "full+pre": np.column_stack([full, pre]), "full+output": np.column_stack([full, out])}
    aud = {}
    for tname, y in (("C_post", C_post), ("dC", dC)):
        r = {k: float(ES.crossfit_ridge(v, y, seed + 200 + i, strata)[1]) for i, (k, v) in enumerate(srcs.items())}
        r["dR2_full_given_pre"] = r["full+pre"] - r["pre"]
        r["dR2_full_given_output"] = r["full+output"] - r["output"]
        aud[tname] = r
    # ---- displacement (generic change magnitude), INTERF and CTRL
    d1 = displacement(p0["states"], p1["states"], sig)
    dc = displacement(p0["states"], pc["states"], sig)
    # ---- exposure / relation strata
    expo = world.fam_high[items[:, 0]]
    _, r2_expo, _ = ES.crossfit_ridge(expo[:, None].astype(float), C_post, seed + 11)
    a_expo = auroc(expo[lost].astype(float), expo[~lost].astype(float)) if lost.any() and (~lost).any() else 0.5
    rel = {str(r): {"n": int((items[:, 1] == r).sum()), "lost": float(lost[items[:, 1] == r].mean()),
                    "dC_mean": float(dC[items[:, 1] == r].mean())} for r in range(world.n_rel)}
    # ---- familiarity diagnostics
    names = world.names[items[:, 0]]
    f0, f1, fc = name_fluency(store0, world, names), name_fluency(post, world, names), name_fluency(ctrl, world, names)
    rho_f, p_f = stats.spearmanr(dC, f1 - f0)
    # ---- control continuation (generic optimisation drift)
    _, r2_ctrl, _ = ES.crossfit_ridge(dCc[:, None], dC, seed + 31, strata)
    rho_c, p_c = stats.spearmanr(dC, dCc)
    # ---- binary-secondary feasibility (report-only in D2)
    Xpre = np.column_stack([Cp, W0, P])
    pairs, smd = ES.propensity_match(lost, Xpre, strata, seed + 303, qc_bin["caliper_sd"])
    pairs2, _ = ES.propensity_match(lost, Xpre, (strata * world.n_rel + items[:, 1]).astype(int), seed + 303,
                                    qc_bin["caliper_sd"])
    mx2, mn2 = ES.balance(Xpre, pairs2) if len(pairs2) else (float("nan"), float("nan"))
    kn = W.items_where(world, known=True)
    rep = {
        "n": int(len(items)), "steps_used": int(n_steps), "trace": trace,
        "acc_population_pre": float(p0["correct"].mean()), "acc_population_post": float(p1["correct"].mean()),
        "acc_all_known_pre": float(answer_stats(store0, world, kn)["correct"].mean()),
        "acc_all_known_post": float(answer_stats(post, world, kn)["correct"].mean()),
        "lost_frac": float(lost.mean()), "dC_quantiles": _q(dC), "dC_mean": float(dC.mean()), "dC_sd": float(dC.std()),
        "dC_iqr": _iqr(dC), "C_pre_quantiles": _q(C_pre), "C_post_quantiles": _q(C_post), "C_post_iqr": _iqr(C_post),
        "displacement": {"site_mean": [float(x) for x in d1.mean(0)], "item_mean_quantiles": _q(d1.mean(1))},
        "generic40_mean": [float(x) for x in G.mean(0)],
        "exposure": {"r2_C_post": float(r2_expo), "auroc_lost": float(max(a_expo, 1 - a_expo)),
                     "lost_high": float(lost[expo].mean()), "lost_low": float(lost[~expo].mean()),
                     "dC_mean_high": float(dC[expo].mean()), "dC_mean_low": float(dC[~expo].mean())},
        "relation": rel,
        "familiarity": {"dfluency_interf_mean": float((f1 - f0).mean()), "dfluency_ctrl_mean": float((fc - f0).mean()),
                        "spearman_dC_vs_dfluency": float(rho_f), "p": float(p_f)},
        "control": {"lost_frac": float((C_ctrl < 0).mean()), "dC_quantiles": _q(dCc), "dC_mean": float(dCc.mean()),
                    "displacement_item_mean_quantiles": _q(dc.mean(1)), "interference_specific_lost":
                    float(lost.mean() - (C_ctrl < 0).mean()), "r2_dC_interf_given_dC_ctrl": float(r2_ctrl),
                    "spearman_dC_interf_vs_ctrl": float(rho_c), "p": float(p_c)},
        "identifiability": idf, "audit": aud,
        "D50": {"pre_state_dim": int(p0["states"].shape[1] * p0["states"].shape[2]), "covariates": int(W0.shape[1] + 1),
                "n": int(len(items)), "p_over_n": float((p0["states"].shape[1] * p0["states"].shape[2] + W0.shape[1] + 1) / len(items)),
                "residual_sd_dC_given_pre": float(np.std(dC - P)), "dC_sd": float(dC.std())},
        "binary_secondary": {"pairs": int(len(pairs)), "max_abs_smd": float(smd),
                             "feasible": bool(len(pairs) >= qc_bin["min_pairs"] and smd <= qc_bin["smd_max"])},
        "binary_secondary_alt": {"pairs": int(len(pairs2)), "max_abs_smd": float(mx2), "mean_abs_smd": float(mn2),
                                 "feasible": bool(len(pairs2) >= qc_bin["min_pairs"] and mx2 <= 0.25 and mn2 <= 0.10)},
    }
    lo, hi = gcfg["lost_range"]
    rep["gates"] = {"G1_lost_range": bool(lo <= rep["lost_frac"] <= hi), "G2_dC_iqr": bool(rep["dC_iqr"] >= gcfg["dC_iqr_min"]),
                    "G4_no_bookkeeping": bool(aud["C_post"]["book"] < gcfg["bookkeeping_r2_kill"]),
                    "G5_input_identity": bool(np.array_equal(tok0, tok1))}
    return rep
