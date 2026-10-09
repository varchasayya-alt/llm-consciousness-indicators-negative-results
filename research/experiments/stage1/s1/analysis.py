"""Frozen confirmatory analysis (Stage-1 v3, within-target identification; decisions D36-D41).
Reads only raw per-seed outputs; no analytical decisions after unblinding.

Primary estimand (per seed, then across store seeds; items are never replications):
  theta_pre = s1.estimands.theta_post_dml: residualized-post association of the monitor's post-intervention score
              with post-intervention competence (answer margin) among IDENTICALLY TREATED items, adjusting only for
              pre-intervention information: RCS of rank C_pre and rank M_pre, pre covariates W0, and a cross-fitted
              symmetric adjustment for the pre-intervention read states. Selected by the simulation-only comparison
              (results/statistics/v3_statistic_simulation; D37).
  P1: T-FORGET targeted set X (all items receive the identical procedure).
  P2: T-INTERF, all base-correct EV items (all equally exposed).
Robustness (reported, fixed): theta_gen (+ 40 generic post-change features; family G, interpretation labels),
theta_noState (no pre-state step), theta_P (one-sided susceptibility score P_i), theta_delta (change-score, descriptive).
Interpretation labels (frozen): A theta_pre & theta_gen supported; B theta_pre supported, theta_gen not; C theta_pre not
supported; 'non-identifiable' if the study-level identifiability check fails (median R2_gen >= .90, R2_pre >= .90 or
R2_joint >= .95 over seeds).
Inferential status of the labels (D54, pre-data, PI 2026-10-03; the statistic and labels themselves are unchanged):
only A counts as (confirmatory) support for the competence-monitoring hypothesis; B is NOT affirmative support
(ambiguous / change-structure-sensitive tracking not separable from residual susceptibility or generic state-change
information; simulation D50: diffuse unrecovered susceptibility gives false B in ~25% of studies, never false A);
C = no evidence.
Usage:  python run_analysis.py --raw <raw_dir> --out <out_dir> [--n-seeds 20]
"""
import json
import os

import numpy as np
import pandas as pd

from . import estimands as ES
from . import metrics as Mx
from .config import derive_seed

MEI_AUROC = 0.55          # minimum effect of interest (AUROC endpoints)
MEI_THETA = 0.10          # minimum effect of interest (theta and z(theta) differences)
EQUIV_REL = 0.10          # K2/F2 equivalence bound: 10% of the natural known-unknown separation (D3)
ALPHA = 0.05
IDENT = {"r2_gen_max": 0.90, "r2_pre_max": 0.90, "r2_joint_max": 0.95}
DISP_COLS = [f"d{s}" for s in range(10)]
GEN_COLS = [f"g{s}" for s in range(30)]   # per-site cos(delta, pre), relative norm change, abnormality change
INT_CONDS = ["INT-S", "INT-T", "INT-P", "INT-TP"]
BIN_MIN_PAIRS, BIN_SMD_MAX, BIN_CALIPER = 50, 0.10, 0.2


# ---------------------------------------------------------------- loading
def load_seed(path):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    items, nat = [], []
    if os.path.exists(os.path.join(path, "items.jsonl")):
        with open(os.path.join(path, "items.jsonl")) as f:
            items = [json.loads(l) for l in f if l.strip()]
    if os.path.exists(os.path.join(path, "natural.jsonl")):
        with open(os.path.join(path, "natural.jsonl")) as f:
            nat = [json.loads(l) for l in f if l.strip()]
    pz = os.path.join(path, "prestates.npz")
    meta["_prestates"] = dict(np.load(pz)) if os.path.exists(pz) else {}
    return meta, items, nat


def flatten(items):
    rows = []
    for r in items:
        base = {k: v for k, v in r.items() if k not in ("scores", "lookup", "disp", "gen")}
        for c, (a, b) in r["scores"].items():
            base[f"pre::{c}"], base[f"post::{c}"] = a, b
        for c, (a, b) in r.get("lookup", {}).items():
            base[f"lpre::{c}"], base[f"lpost::{c}"] = a, b
        if r.get("disp") is not None:
            for s, v in enumerate(r["disp"]):
                base[f"d{s}"] = v
        if r.get("gen") is not None:
            for s, v in enumerate(r["gen"]):
                base[f"g{s}"] = v
        rows.append(base)
    return pd.DataFrame(rows)


def select_seeds(raw_root, n_required):
    seeds, excluded = [], []
    for name in sorted(os.listdir(raw_root)):
        p = os.path.join(raw_root, name)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "meta.json")):
            continue
        meta, items, nat = load_seed(p)
        if meta.get("excluded"):
            excluded.append({"seed": meta["seed"], "reason": meta.get("reason")})
            continue
        seeds.append((meta, items, nat))
    seeds.sort(key=lambda t: t[0]["seed"])
    return seeds[:n_required], excluded, len(seeds)


# ---------------------------------------------------------------- per-seed building blocks
def _ds(df, cond):
    return df[f"post::{cond}"] - df[f"pre::{cond}"]


def ti_matched(df, cond, pos_set, neg_set, restrict=None):
    """AUROC of the score DECREASE, displacement-matched pairs only (pair >= 0). DESCRIPTIVE in v3."""
    d = df[df["pair"] >= 0]
    if restrict is not None:
        d = d[restrict(d)]
    pos = d[(d["set"] == pos_set)]
    neg = d[(d["set"] == neg_set)]
    if pos_set == "X":
        pos = pos[~pos["post_correct"]]
        neg = neg[neg["post_correct"]]
    return Mx.auroc(-_ds(pos, cond), -_ds(neg, cond))


def d_nat(nat, cond):
    known = nat[nat["trained"] & nat["correct"]][f"s::{cond}"]
    unknown = nat[~nat["trained"] & ~nat["correct"]][f"s::{cond}"]
    return float(known.mean() - unknown.mean())


def pre_W0(d):
    """Pre-intervention covariates (no monitor output): log p(v*) pre, exposure class, name fluency, relation."""
    rel = np.eye(4)[d["relation"].astype(int).values][:, 1:]
    return np.column_stack([d["pre_logp"].values, d["fam_high"].astype(float).values, d["fluency_pre"].values, rel])


def generic40(d):
    """Pre-declared generic post-change features: log displacement (10) + cos/relnorm (20) + abnormality (10)."""
    return np.column_stack([np.log(d[DISP_COLS].values + 1e-3), d[GEN_COLS].values])


class Population:
    """Identically treated items of one seed with their pre-intervention states."""

    def __init__(self, d, H, seed_key):
        self.d = d.reset_index(drop=True)
        self.H = np.asarray(H, float)[self.d["h_idx"].astype(int).values]
        self.W0 = pre_W0(self.d)
        self.C_pre, self.C_post = self.d["pre_margin"].values, self.d["post_margin"].values
        self.strata = self.d["fam_high"].astype(int).values
        self.key = seed_key
        self._P = None

    def sub(self, mask, key):
        p = Population.__new__(Population)
        p.d, p.H, p.W0 = self.d[mask].reset_index(drop=True), self.H[mask], self.W0[mask]
        p.C_pre, p.C_post, p.strata, p.key, p._P = self.C_pre[mask], self.C_post[mask], self.strata[mask], key, None
        return p

    def scores(self, cond, lookup=False):
        if lookup:      # tracking means LOOKUP rises as competence falls -> use 1 - P(lookup)
            return 1 - self.d[f"lpre::{cond}"].values, 1 - self.d[f"lpost::{cond}"].values
        return self.d[f"pre::{cond}"].values, self.d[f"post::{cond}"].values

    def theta(self, cond, kind="pre", lookup=False):
        Mpre, Mpost = self.scores(cond, lookup)
        s = derive_seed(self.key, "folds")        # same nuisance folds for every monitor/condition of a population
        if kind == "pre":
            return ES.theta_post_dml(Mpre, Mpost, self.C_pre, self.C_post, self.W0, self.H, s, strata=self.strata)[0]
        if kind == "gen":
            return ES.theta_post_dml(Mpre, Mpost, self.C_pre, self.C_post, self.W0, self.H, s, G=generic40(self.d),
                                     strata=self.strata)[0]
        if kind == "noState":
            return ES.theta_post(Mpre, Mpost, self.C_pre, self.C_post, W=self.W0, flexible=True)
        if kind == "P":
            return ES.theta_post(Mpre, Mpost, self.C_pre, self.C_post, W=np.column_stack([self.W0, self.P()]), flexible=True)
        if kind == "delta":
            return ES.theta_delta(Mpre, Mpost, self.C_pre, self.C_post, W=self.W0, flexible=False)
        raise ValueError(kind)

    def m_residual(self, cond):
        """Monitor outcome adjusted for ALL pre-intervention information (the M-side residual of theta_pre)."""
        Mpre, Mpost = self.scores(cond)
        return ES.theta_post_dml(Mpre, Mpost, self.C_pre, self.C_post, self.W0, self.H, derive_seed(self.key, "folds"),
                                 strata=self.strata)[1]["resid_M"]

    def P(self):
        """Cross-fitted susceptibility score: ridge prediction of dC from pre-states + W0 + C_pre."""
        if self._P is None:
            Xp = np.column_stack([self.H, self.W0, self.C_pre])
            self._P, self._r2_pre, _ = ES.crossfit_ridge(Xp, self.C_post - self.C_pre, derive_seed(self.key, "P"), self.strata)
        return self._P

    def identifiability(self):
        dC = self.C_post - self.C_pre
        self.P()
        G = generic40(self.d)
        _, r2_gen, _ = ES.crossfit_ridge(G, dC, derive_seed(self.key, "r2gen"), self.strata)
        _, r2_joint, _ = ES.crossfit_ridge(np.column_stack([self.H, self.W0, self.C_pre, G]), dC,
                                           derive_seed(self.key, "r2joint"), self.strata)
        return {"r2_pre": float(self._r2_pre), "r2_gen": float(r2_gen), "r2_joint": float(r2_joint)}


def _z(t):
    return float(np.arctanh(np.clip(t, -0.999, 0.999))) if np.isfinite(t) else np.nan


def seed_endpoints(meta, items, nat_rows):
    df = flatten(items)
    nat = pd.DataFrame([{**{k: v for k, v in r.items() if k != "scores"},
                         **{f"s::{c}": s for c, s in r["scores"].items()}} for r in nat_rows])
    H = meta.get("_prestates", {})
    seed = meta["seed"]
    A = df[df["store"] == "A"]
    F = A[A["intervention"] == "FORGET"]
    I = A[A["intervention"] == "INTERF"]
    out = {"seed": seed}
    conds = [c[len("pre::"):] for c in F.columns if c.startswith("pre::") and F[c].notna().all()]
    popX = Population(F[F["set"] == "X"], H["FORGET_A"], (seed, "P1"))
    popI = Population(I, H["INTERF_A"], (seed, "P2"))
    # ---- primary and robustness thetas
    for c in conds:
        if c == "IN":
            out[f"P1|{c}"] = out[f"P2|{c}"] = np.nan      # input-only monitor: score unchanged by construction
            continue
        out[f"P1|{c}"] = popX.theta(c, "pre")
        out[f"P2|{c}"] = popI.theta(c, "pre")
    for kind in ("gen", "noState", "P", "delta"):
        out[f"P1_{kind}|INT-S"] = popX.theta("INT-S", kind)
        out[f"P2_{kind}|INT-S"] = popI.theta("INT-S", kind)
    for kind in ("gen",):
        out[f"P1_{kind}|INT-TP"] = popX.theta("INT-TP", kind)
        out[f"P1_{kind}|OUT"] = popX.theta("OUT", kind)
    for pname, pop in (("P1", popX), ("P2", popI)):
        for k, v in pop.identifiability().items():
            out[f"{pname}_{k}"] = v
    # fingerprint strength vs competence change within X (pre-declared diagnostic)
    Yd = F[F["set"] == "Y"]
    fX, _ = ES.crossfit_logistic_scores(generic40(popX.d), generic40(Yd), derive_seed(seed, "fp"))
    out["P1_fingerprint_vs_dC_spearman"] = float(pd.Series(fX).corr(pd.Series(popX.C_post - popX.C_pre), method="spearman"))
    # ---- K1: matched binary within X (pre-intervention propensity only; never monitor outputs)
    lost = ~popX.d["post_correct"].values.astype(bool)
    Xpre = np.column_stack([popX.C_pre, popX.W0, popX.P()])
    pairs, smd = ES.propensity_match(lost, Xpre, popX.strata, derive_seed(seed, "K1"), BIN_CALIPER)
    out["K1_pairs"], out["K1_max_smd"] = len(pairs), smd
    ok = len(pairs) >= BIN_MIN_PAIRS and smd <= BIN_SMD_MAX
    pi, pj = np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int)
    for c in INT_CONDS + ["OUT"]:
        rM = popX.m_residual(c)            # monitor outcome adjusted for all pre-intervention information (D42)
        out[f"K1|{c}"] = Mx.auroc(-rM[pi], -rM[pj]) if ok else np.nan
    out["K1"] = out["K1|INT-S"]
    # ---- K2: Y negative control (sham response relative to unrelated Z), D_nat units
    dn = d_nat(nat, "INT-S")
    out["D_nat|INT-S"] = dn
    Yc = F[(F["set"] == "Y") & F["post_correct"]]
    Zc = F[(F["set"] == "Z") & F["post_correct"]]
    out["K2"] = float((_ds(Yc, "INT-S").mean() - _ds(Zc, "INT-S").mean()) / dn)
    # ---- K3 developmental, K4 composed behaviour
    out["K3"] = _z(out["P1|INT-TP"]) - _z(out["P1|INT-S"])
    out["K4"] = popX.theta("INT-S", "pre", lookup=True)
    out["K4_C+in"] = popX.theta("INT-S+in", "pre", lookup=True)
    # ---- F family
    hi = popX.d["fam_high"].values.astype(bool)
    out["F1"] = popX.sub(hi, (seed, "F1")).theta("INT-S", "pre")
    FM = A[A["intervention"] == "FAM"]
    out["F2"] = float((_ds(FM[FM["set"] == "F"], "INT-S").mean() - _ds(FM[FM["set"] == "U2"], "INT-S").mean()) / dn) if len(FM) else np.nan
    NW = A[(A["intervention"] == "NEW") & ~A["fam_high"]]
    if len(NW):
        pos = NW[(NW["set"] == "N") & NW["post_correct"]]
        neg = NW[(NW["set"] == "U") & ~NW["post_correct"]]
        out["F3"] = Mx.auroc(_ds(pos, "INT-S"), _ds(neg, "INT-S"))
    else:
        out["F3"] = np.nan
    # ---- S family (system-specific coupling) on store B's targeted set
    Bf = df[(df["store"] == "B") & (df["intervention"] == "FORGET")]
    popB = Population(Bf[Bf["set"] == "X"], H["FORGET_B"], (seed, "S"))
    for c in ("INT-S", "INT-TP"):
        for key in ("B->B", "Bal->Bal", "A->Bal", "A->Braw"):
            out[f"thetaB|{key}:{c}"] = popB.theta(f"{key}:{c}", "pre")
        out[f"S|{c}"] = _z(out[f"thetaB|Bal->Bal:{c}"]) - _z(out[f"thetaB|A->Bal:{c}"])
    # ---- planned secondary / descriptive
    for ep, pop in (("P1", popX), ("P2", popI)):
        g = lambda c: _z(out[f"{ep}|{c}"])
        out[f"interaction_z|{ep}"] = g("INT-TP") - g("INT-T") - g("INT-P") + g("INT-S")
        r = lambda c: out[f"{ep}|{c}"]
        out[f"interaction_raw|{ep}"] = r("INT-TP") - r("INT-T") - r("INT-P") + r("INT-S")
    dp = lambda c: float(Mx.dprime_from_auc(out[f"K1|{c}"])) if np.isfinite(out[f"K1|{c}"]) else np.nan
    out["interaction_dprime|K1"] = dp("INT-TP") - dp("INT-T") - dp("INT-P") + dp("INT-S")
    out["SEC_INTvsOUT_z|P1"] = _z(out["P1|INT-S"]) - _z(out["P1|OUT"])
    out["SEC_INTvsOUT_z|P2"] = _z(out["P2|INT-S"]) - _z(out["P2|OUT"])
    Xl = F[(F["set"] == "X") & ~F["post_correct"]]
    out["DESC_old_P1_XlostVsY|INT-S"] = ti_matched(F, "INT-S", "X", "Y")         # fingerprint-confounded (D34)
    out["DESC_old_K1_XlostVsZ|INT-S"] = Mx.auroc(-_ds(Xl, "INT-S"), -_ds(Zc, "INT-S"))
    out["DESC_TI_INTERF_matched|INT-S"] = ti_matched(I, "INT-S", "lost", "retained")
    mm = F[F["pair"] >= 0]
    pos, neg = mm[(mm["set"] == "X") & ~mm["post_correct"]], mm[(mm["set"] == "Y") & mm["post_correct"]]
    out["twin_level_AUROC"] = Mx.auroc(-pos["twin_level_INT-S"], -neg["twin_level_INT-S"]) if "twin_level_INT-S" in mm else np.nan
    for c in conds:
        out[f"natAUROC|{c}"] = Mx.auroc(nat[nat["correct"]][f"s::{c}"], nat[~nat["correct"]][f"s::{c}"]) if f"s::{c}" in nat else np.nan
        C = A[A["intervention"] == "CORRUPT"]
        out[f"corrupt_AUROC|{c}"] = Mx.auroc(-C[f"post::{c}"], -C[f"pre::{c}"]) if len(C) else np.nan
    for ep in ("ACT", "REPLACE"):
        E_ = A[A["intervention"] == ep]
        for c in ("INT-S", "OUT", "INT-TP"):
            if len(E_):
                xs = E_[E_["set"] == "X"]
                ys = E_[E_["set"] == "Y"]
                xs = xs[~xs["post_correct"]] if ep == "ACT" else xs
                out[f"EXPL_{ep}|{c}"] = Mx.auroc(-_ds(xs, c), -_ds(ys[ys["post_correct"]], c))
    for nm, sub in (("Xlost", Xl), ("Xret", F[(F["set"] == "X") & F["post_correct"]]), ("Y", Yc), ("Z", Zc)):
        out[f"lookup_rate_pre|{nm}"] = float((sub["lpre::INT-S"] > 0.5).mean())
        out[f"lookup_rate_post|{nm}"] = float((sub["lpost::INT-S"] > 0.5).mean())
    fq = meta.get("qc", {}).get("forget", {})
    out["disp_ratio_median"] = float(np.median(fq["site_ratios_X_over_Y"])) if fq.get("site_ratios_X_over_Y") else np.nan
    return out


# ---------------------------------------------------------------- tests
def _theta_decision(t, adj_p):
    if adj_p < ALPHA and t["mean"] >= t["mei"]:
        return "supported"
    if t["ci"][1] < t["mei"]:
        return "falsified (strong form)"
    return "inconclusive"


def run_tests(S):
    """S: per-seed endpoints. Returns tests with Holm within families P, G, K, F, S."""
    T = []

    def theta_test(name, family, col, mei=MEI_THETA, on_z=True):
        v = S[col].astype(float).values
        v = v[np.isfinite(v)]
        z = np.arctanh(np.clip(v, -0.999, 0.999)) if on_z else v
        t = Mx.one_sample_t(z, 0.0, "greater")
        t.update(name=name, family=family, column=col, mean=float(v.mean()), ci=Mx.bootstrap_ci(v), mei=mei,
                 wilcoxon_p=Mx.wilcoxon_signed(z, 0.0, "greater")["p"], kind="theta")
        T.append(t)

    def auc_test(name, family, col):
        v = S[col].astype(float).values
        v = v[np.isfinite(v)]
        t = Mx.one_sample_t(v, 0.5, "greater")
        t.update(name=name, family=family, column=col, ci=Mx.bootstrap_ci(v), mei=MEI_AUROC,
                 wilcoxon_p=Mx.wilcoxon_signed(v, 0.5, "greater")["p"], kind="theta")
        T.append(t)

    def twosided(name, family, col, mu0, z=False):
        v = S[col].astype(float).values
        v = v[np.isfinite(v)]
        vv = np.arctanh(np.clip(v, -0.999, 0.999)) if z else v
        t = Mx.one_sample_t(vv, mu0, "two-sided")
        t.update(name=name, family=family, column=col, mu0=mu0, mean=float(v.mean()), ci=Mx.bootstrap_ci(v), kind="two-sided")
        T.append(t)

    theta_test("P1 within-target FORGET theta_pre (INT-S)", "P", "P1|INT-S")
    theta_test("P2 within-exposed INTERF theta_pre (INT-S)", "P", "P2|INT-S")
    theta_test("G1 P1 generic-adjusted theta_gen", "G", "P1_gen|INT-S")
    theta_test("G2 P2 generic-adjusted theta_gen", "G", "P2_gen|INT-S")
    auc_test("K1 matched lost vs retained within X (INT-S)", "K", "K1")
    tt = Mx.tost(S["K2"].values, EQUIV_REL)
    tt.update(name="K2 Y negative control (TOST, rel. bound 0.10)", family="K", column="K2", kind="tost",
              ci=Mx.bootstrap_ci(S["K2"].values))
    T.append(tt)
    theta_test("K3 developmental TP>S (z(theta) difference)", "K", "K3", on_z=False)
    theta_test("K4 composed behaviour (INT-S->C, theta on 1-P(LOOKUP))", "K", "K4")
    twosided("F1 familiar stratum theta_pre", "F", "F1", 0.0, z=True)
    twosided("F2 familiarity-boost false rise (rel.)", "F", "F2", 0.0)
    twosided("F3 newly learned (low fam.)", "F", "F3", 0.5)
    twosided("S1 system-specific coupling INT-S (z diff)", "S", "S|INT-S", 0.0)
    twosided("S2 system-specific coupling INT-TP (z diff)", "S", "S|INT-TP", 0.0)
    for fam in ("P", "G", "K", "F", "S"):
        idx = [i for i, t in enumerate(T) if t["family"] == fam]
        adj, rej = Mx.holm([T[i]["p"] for i in idx], ALPHA)
        for i, a, r in zip(idx, adj, rej):
            T[i]["p_holm"], T[i]["reject"] = a, r
    for t in T:
        if t["kind"] == "theta":
            t["decision"] = _theta_decision(t, t["p_holm"])
        elif t["kind"] == "tost":
            t["decision"] = "equivalent" if t["p_holm"] < ALPHA else ("non-equivalent" if abs(t["mean"]) > EQUIV_REL else "inconclusive")
        else:
            t["decision"] = "different from null" if t["p_holm"] < ALPHA else "not different"
    return T


def study_checks(S, n_valid, n_required):
    ch = {
        "a_enough_valid_seeds": bool(n_valid >= n_required),
        "b_natural_auroc_INT-S_median": float(np.nanmedian(S["natAUROC|INT-S"])),
        "b_pass": bool(np.nanmedian(S["natAUROC|INT-S"]) >= 0.65),
        "c_twin_level_auroc_median": float(np.nanmedian(S["twin_level_AUROC"])),
        "c_flag": bool(np.nanmedian(S["twin_level_AUROC"]) > 0.60),
        "d_disp_ratio_median": float(np.nanmedian(S["disp_ratio_median"])),
        "d_flag": bool(not (0.67 <= np.nanmedian(S["disp_ratio_median"]) <= 1.5)),
    }
    for p in ("P1", "P2"):
        med = {k: float(np.nanmedian(S[f"{p}_{k}"])) for k in ("r2_gen", "r2_pre", "r2_joint")}
        ch[f"e_identifiability_{p}"] = med
        ch[f"e_identifiable_{p}"] = bool(med["r2_gen"] < IDENT["r2_gen_max"] and med["r2_pre"] < IDENT["r2_pre_max"]
                                         and med["r2_joint"] < IDENT["r2_joint_max"])
    return ch


LABEL_STATUS = {   # D54 (pre-data): inferential status of each label
    "A": "confirmatory support for competence monitoring (subject to all other preregistered gates)",
    "B": "NOT affirmative support: ambiguous / change-structure-sensitive tracking, not separable from residual "
         "susceptibility or generic state-change information",
    "C": "no evidence for the monitoring hypothesis",
    "non-identifiable (no claim)": "no claim (design cannot separate the hypotheses)"}


def interpretation_labels(T, checks):
    """Frozen A/B/C labels (D36): A theta_pre & theta_gen supported; B theta_pre supported only; C otherwise."""
    dec = {t["column"]: t["decision"] for t in T}
    lab = {}
    for p, g in (("P1", "P1_gen|INT-S"), ("P2", "P2_gen|INT-S")):
        if not checks[f"e_identifiable_{p}"]:
            lab[p] = "non-identifiable (no claim)"
        elif dec[f"{p}|INT-S"] == "supported":
            lab[p] = "A" if dec[g] == "supported" else "B"
        else:
            lab[p] = "C"
    return lab


# ---------------------------------------------------------------- outputs
def make_figures(S, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    conds = ["OUT", "INT-S", "INT-T", "INT-P", "INT-TP"]
    for ep, name in (("P1", "FORGET"), ("P2", "INTERF")):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for k, c in enumerate(conds):
            v = S[f"{ep}|{c}"].astype(float).values
            ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(v)), v, s=12, alpha=0.6)
            ax.errorbar(k + 0.3, np.nanmean(v), yerr=1.96 * np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v))),
                        fmt="o", color="black", capsize=3)
        ax.axhline(0.0, ls="--", c="grey", lw=1)
        ax.axhline(MEI_THETA, ls=":", c="grey", lw=1)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_ylabel("theta_pre (within-target)")
        ax.set_title(f"{ep}: competence tracking among identically treated items ({name})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_primary_{name}.png"), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    cc = ["IN", "OUT", "INT-S", "INT-T", "INT-P", "INT-TP"]
    ax.bar(range(len(cc)), [np.nanmean(S[f"natAUROC|{c}"]) for c in cc])
    ax.set_xticks(range(len(cc)), cc)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("natural AUROC (intact EV)")
    ax.set_title("Observational-equivalence panel")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_natural.png"), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, ep in zip(axes, ("P1", "P2")):
        grid = np.array([[np.nanmean(S[f"{ep}|INT-S"]), np.nanmean(S[f"{ep}|INT-P"])],
                         [np.nanmean(S[f"{ep}|INT-T"]), np.nanmean(S[f"{ep}|INT-TP"])]])
        ax.imshow(grid, cmap="viridis")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
        ax.set_xticks([0, 1], ["P absent", "P present"])
        ax.set_yticks([0, 1], ["T absent", "T present"])
        ax.set_title(f"theta_pre {ep}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_developmental.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    keys = ["Xlost", "Xret", "Y", "Z"]
    pre = [np.nanmean(S[f"lookup_rate_pre|{k}"]) for k in keys]
    post = [np.nanmean(S[f"lookup_rate_post|{k}"]) for k in keys]
    ax.bar(np.arange(4) - 0.2, pre, 0.4, label="pre")
    ax.bar(np.arange(4) + 0.2, post, 0.4, label="post")
    ax.set_xticks(range(4), ["X lost", "X retained", "Y sham", "Z unrelated"])
    ax.set_ylabel("LOOKUP rate (c = 0.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_behaviour.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    keys = ["B->B", "Bal->Bal", "A->Bal", "A->Braw"]
    for k, c in enumerate(("INT-S", "INT-TP")):
        ax.bar(np.arange(4) + (k - 0.5) * 0.4, [np.nanmean(S[f"thetaB|{kk}:{c}"]) for kk in keys], 0.4, label=c)
    ax.set_xticks(range(4), keys)
    ax.axhline(0.0, ls="--", c="grey", lw=1)
    ax.set_ylabel("theta_pre on store B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_selfspecificity.png"), dpi=150)
    plt.close(fig)


def write_tables(T, S, checks, labels, excluded, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    S.to_csv(os.path.join(out_dir, "seed_level_endpoints.csv"), index=False)
    with open(os.path.join(out_dir, "confirmatory_tests.json"), "w") as f:
        json.dump({"tests": T, "study_checks": checks, "labels": labels,
                   "label_status": {k: LABEL_STATUS[v] for k, v in labels.items()}, "excluded_seeds": excluded},
                  f, indent=1, default=float)
    lines = ["| Test | Family | n | Mean | 95% CI | p | p (Holm) | Decision |", "|---|---|---|---|---|---|---|---|"]
    for t in T:
        lines.append(f"| {t['name']} | {t['family']} | {t['n']} | {t['mean']:.3f} | [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}] "
                     f"| {t['p']:.2e} | {t['p_holm']:.2e} | {t['decision']} |")
    lines += ["", "Interpretation labels (frozen; A: theta_pre and theta_gen supported; B: theta_pre only; C: neither): "
              + json.dumps(labels), "", "Inferential status (D54; only A is affirmative support): "
              + json.dumps({k: LABEL_STATUS[v] for k, v in labels.items()}),
              "", "Study-level checks: " + json.dumps(checks), "", f"Excluded seeds: {excluded}"]
    sec = [c for c in S.columns if c.startswith(("P1_", "P2_", "SEC_", "interaction_", "EXPL_", "DESC_", "K1_", "K4_"))]
    lines += ["", "## Planned secondary, robustness and descriptive estimates (no confirmatory claims; seed-bootstrap 95% CI)", "",
              "| Quantity | n | Mean | 95% CI |", "|---|---|---|---|"]
    for c in sec:
        v = S[c].astype(float).dropna().values
        if len(v) >= 2:
            lo, hi = Mx.bootstrap_ci(v)
            lines.append(f"| {c.replace('|', ' / ')} | {len(v)} | {v.mean():.3f} | [{lo:.3f}, {hi:.3f}] |")
    with open(os.path.join(out_dir, "confirmatory_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(raw_root, out_dir, n_required=20):
    seeds, excluded, n_valid = select_seeds(raw_root, n_required)
    S = pd.DataFrame([seed_endpoints(m, it, nt) for m, it, nt in seeds])
    T = run_tests(S)
    checks = study_checks(S, n_valid, n_required)
    labels = interpretation_labels(T, checks)
    write_tables(T, S, checks, labels, excluded, os.path.join(out_dir, "statistics"))
    make_figures(S, os.path.join(out_dir, "figures"))
    return T, S, checks, excluded, labels
