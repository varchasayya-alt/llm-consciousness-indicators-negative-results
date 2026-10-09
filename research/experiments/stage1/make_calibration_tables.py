"""Render calibration_results.json into markdown tables (no hand transcription of numbers).

Store/intervention quantities only; calibration never produces monitor quantities.
Usage: python make_calibration_tables.py  ->  results/processed/calibration/calibration_tables.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "raw", "calibration", "calibration_results.json")
OUT = os.path.join(ROOT, "results", "processed", "calibration", "calibration_tables.md")
CATS = ("Z_random", "Z_near_entity_X", "Z_near_entity_Y", "Z_near_repr", "Z_far")


def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def c1c2(r, md):
    if "C1" in r:
        md.append("## C1 store training length\n")
        md.append(table(["seed", "E99", "minutes (200 ep)"],
                        [[s, v["E99"], f(v["minutes"], 1)] for s, v in r["C1"].items() if s != "E_store"]))
        md.append(f"\nE_store = **{r['C1']['E_store']}**\n")
    for key, title in (("C2", "C2 calibration stores (9001–9005)"), ("C2val", "C2val fresh validation stores (9011–9013)")):
        if key in r:
            md.append(f"\n## {title}\n")
            md.append(table(["seed", "trained acc", "unknown acc", "fluency AUROC", "pass", "min", "EV trained"],
                            [[s, f(v["qc"]["trained_fact_acc"]), f(v["qc"]["unknown_fact_acc"]),
                              f(v["qc"]["fluency_auroc_high_vs_low"]), v["qc"]["pass"], f(v["minutes"], 1), v["n_EV_trained"]]
                             for s, v in r[key]["seeds"].items()]))


def c3(r, md):
    if "C3" in r:
        md.append("\n## C3 (v1 rule; statistic over ALL MT items, flawed: D25/D29)\n")
        rates = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"]
        md.append(table(["seed"] + rates + ["pool U(0,.6)"],
                        [[s] + [f(v[k]) for k in rates] + [f(v["pool_accuracy_U(0,0.6)"])] for s, v in r["C3"]["seeds"].items()]))
        md.append(f"\nv1 rule output: rate max = {r['C3']['rate_max']}\n")
    if "C3r" in r:
        x = r["C3r"]
        md.append("\n## C3r (corrected rule; retained-competence fraction on trained & intact-correct MT items)\n")
        rates = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"]
        md.append(table(["seed", "n"] + rates + ["pool U(0,.6)"],
                        [[s, v["n_base_correct_MT"]] + [f(v[k]) for k in rates] + [f(v["pool_rho_U(0,0.6)"])]
                         for s, v in x["seeds"].items()]))
        md.append(f"\nmean ρ(0.1) = {f(x['mean_rho_rate0.1'])}, mean ρ(0.6) = {f(x['mean_rho_rate0.6'])} → rate max "
                  f"**{x['rate_max']}** (v1: {x['v1_rate_max']}; changed: {x['changed_vs_v1']})\n")


def c4v1(r, md):
    if "C4" in r:
        md.append("\n## C4 v1 (FAILED: collateral; D26)\n")
        md.append(table(["lr", "steps", "γ", "seed", "X lost", "Y lost", "Z lost", "sites", "Δfluency SD"],
                        [[row["lr"], row["steps"], row["gamma"], s, f(v["X_lost_frac"]), f(v["Y_lost_frac"]),
                          f(v["Z_lost_frac"]), v["sites_within_tol"], f(v["fluency_change_sd"], 2)]
                         for row in r["C4"]["grid"] for s, v in row["seeds"].items()]))


def _rng(vals, d=2):
    return f"{min(vals):.{d}f}–{max(vals):.{d}f}"


def c4v2(r, md):
    if "C4v2" not in r:
        return
    g = r["C4v2"]["grid"]
    md.append("\n## C4 v2 development grid (seeds 9001–9003)\n")
    md.append("Ranges are over the three dev seeds. Z columns: max over seeds of the lost fraction. "
              "C = mean over seeds of the worst category. fp-ok = calipers passing the fingerprint gate.\n\n")
    rows = []
    for row in g:
        S = list(row["seeds"].values())
        zc = [f(max(v["Z_lost_by_category"][c] for v in S)) for c in CATS]
        failed = sorted({x for v in S for x in v["failed_gates"]})
        rows.append([row["stage"], row["lr"], row["steps"], row["lambda_retain"], row["gamma"],
                     _rng([v["X_lost_frac"] for v in S]), f(max(v["Y_lost_frac"] for v in S))] + zc +
                    [min(v["sites_within_tol"] for v in S), f(max(abs(v["fluency_change_sd"]) for v in S), 2),
                     min(v["matched_pairs"]["1.0"] for v in S), f(row["C_collateral"]), f(row["mean_max_abs_log_ratio"]),
                     ",".join(str(c) for c in row["fingerprint_ok_calipers"]) or "—",
                     ",".join(failed) or "—", "**yes**" if row["eligible"] else "no"])
    md.append(table(["stage", "lr", "steps", "λ_R", "γ", "X lost", "Y lost max", "Z_rand", "Z_nearX", "Z_nearY",
                     "Z_nearRepr", "Z_far", "sites min", "max abs Δflu", "pairs@1.0 min", "C", "max abs log r", "fp-ok",
                     "failed gates", "eligible"], rows))
    sel = r["C4v2"].get("selected")
    md.append(f"\n**Selected:** {sel}  {r['C4v2'].get('selected_summary')}\n")
    if sel:
        row = next(x for x in g if all(x[k] == sel[k] for k in sel))
        detail(row["seeds"], md, "Selected cell: per-seed detail")


def detail(seeds, md, title):
    md.append(f"\n### {title}\n")
    rows = []
    for s, v in seeds.items():
        for name in ("X", "Y") + CATS + ("R_sibling_X",):
            rep = v["sets"].get(name, {"n": 0})
            if not rep.get("n"):
                rows.append([s, name, 0] + ["—"] * 7)
                continue
            rows.append([s, name, rep["n"], f(rep["acc_pre"]), f(rep["acc_post"]), f(rep["lost_frac"]),
                         f(rep["margin_change_mean"], 2), f(rep["logp_correct_change_mean"], 3),
                         f(rep["kl_answer_dist_mean"], 4), f(rep["disp_mean"], 3)])
    md.append(table(["seed", "set", "n", "acc pre", "acc post", "lost", "Δmargin", "Δlog p(v*)", "KL(p0‖p1)",
                     "mean displacement"], rows))
    rows = []
    for s, v in seeds.items():
        fp = v.get("fingerprint") or {}
        m = fp.get("matched", {})
        rows.append([s, f(v["fluency_change_sd"], 3), v["sites_within_tol"], f(v["max_abs_log_ratio"]),
                     " ".join(f"{c}:{m[c]['pairs']}" for c in m),
                     " ".join(f"{c}:{f(m[c].get('max_G123'))}" for c in m),
                     f(fp.get("all_G1")), f(fp.get("all_G2")), f(fp.get("all_G3")),
                     f(fp.get("G4_Xretained_vs_Y")), fp.get("n_X_retained"), f(fp.get("G4_Xlost_vs_Y_reference")),
                     f(v["representational_similarity"]["sim_near_mean"]), f(v["representational_similarity"]["sim_far_mean"])])
    md.append("\n" + table(["seed", "Δfluency SD", "sites", "max abs log r", "pairs by caliper", "max G1–G3 (matched) by caliper",
                            "G1 all", "G2 all", "G3 all", "G4 Xret vs Y", "n Xret", "G4 Xlost vs Y (ref)",
                            "sim near", "sim far"], rows))


def c4val(r, md):
    if "C4val" not in r:
        return
    x = r["C4val"]
    md.append(f"\n## C4val: frozen recipe once on fresh seeds 9011–9013. PASS = **{x['PASS']}**\n")
    md.append(f"Recipe {x['recipe']}; fingerprint-ok calipers {x['fingerprint_ok_calipers']}\n\n")
    md.append(table(["seed", "rungs used", "base config passed", "pass"],
                    [[s, len(v["attempts"]), v["base_config_passed"], v["pass"]] for s, v in x["seeds"].items()]))
    detail({s: v["attempts"][-1]["qc"] for s, v in x["seeds"].items()}, md, "C4val per-seed detail (final rung)")


def c7c8c9(r, md):
    if "C7" in r:
        md.append("\n## C7 T-INTERF (dev seeds)\n")
        md.append(table(["lr", "seed", "lost", "steps used", "new-fact acc", "pairs .25/.5/.75/1"],
                        [[row["lr"], s, f(v["lost_frac"]), v["steps_used"], f(v["final_new_acc"]),
                          "/".join(str(v["matched_pairs"][c]) for c in ("0.25", "0.5", "0.75", "1.0"))]
                         for row in r["C7"]["grid"] for s, v in row["seeds"].items()]))
        md.append(f"\n**Selected:** {r['C7']['selected']}\n")
    if "C8" in r:
        md.append("\n## C8 T-NEW (v2; dev seeds)\n")
        md.append(table(["lr", "steps", "λ_R", "N learned (min)", "U correct (max)", "collateral bc (max)",
                         "collateral near-entity (max)", "all pass"],
                        [[row["lr"], row["steps"], row["lambda_retain"],
                          f(min(v["N_learned_frac"] for v in row["seeds"].values())),
                          f(max(v["U_correct_frac"] for v in row["seeds"].values())),
                          f(max(v["collateral_bc_lost"] for v in row["seeds"].values())),
                          f(max(v["collateral_near_entity_lost"] for v in row["seeds"].values())), row["all_pass"]]
                         for row in r["C8"]["grid"]]))
        md.append(f"\n**Selected:** {r['C8']['selected']}\n")
    if "C9" in r:
        md.append("\n## C9 T-FAM (v2; dev seeds)\n")
        md.append(table(["steps", "lr", "rise vs U2 SD (min)", "F correct (max)", "collateral bc (max)", "U2 Δfluency SD (range)", "all pass"],
                        [[row["steps"], row["lr"], f(min(v["fluency_rise_rel_sd"] for v in row["seeds"].values()), 2),
                          f(max(v["F_correct_frac"] for v in row["seeds"].values())),
                          f(max(v["collateral_bc_lost"] for v in row["seeds"].values())),
                          _rng([v["U2_fluency_change_sd"] for v in row["seeds"].values()], 3), row["all_pass"]]
                         for row in r["C9"]["grid"]]))
        md.append(f"\n**Selected:** {r['C9']['selected']}\n")


def c10(r, md):
    if "C10" not in r or "caliper_selected" not in r["C10"]:
        return
    x = r["C10"]
    md.append(f"\n## C10 sham validation + caliper (9001–9005). Caliper selected = **{x['caliper_selected']}**; "
              f"seeds passing QC: {x['seeds_passing_qc']}\n")
    rows = []
    for s, v in x["seeds"].items():
        fq, iq = v["forget_attempts"][-1]["qc"], v["interf_attempts"][-1]["qc"]
        rows.append([s, v["forget_pass"], len(v["forget_attempts"]), f(fq["X_lost_frac"]), f(fq["Z_lost_max_over_categories"]),
                     "/".join(str(fq["matched_pairs"][c]) for c in ("0.25", "0.5", "0.75", "1.0")),
                     v["interf_pass"], len(v["interf_attempts"]), f(iq["lost_frac"]),
                     "/".join(str(iq["matched_pairs"][c]) for c in ("0.25", "0.5", "0.75", "1.0"))])
    md.append(table(["seed", "FORGET pass", "rungs", "X lost", "Z worst", "F pairs .25/.5/.75/1", "INTERF pass", "rungs",
                     "I lost", "I pairs .25/.5/.75/1"], rows))
    detail({s: v["forget_attempts"][-1]["qc"] for s, v in x["seeds"].items()}, md, "C10 FORGET per-seed detail")
    md.append("\n### C10 INTERF generic diagnostics (lost vs retained)\n")
    md.append(table(["seed", "G1 all", "G2 all", "G3 all", "matched@sel G1/G2/G3", "lost post max-prob"],
                    [[s, f(v["interf_attempts"][-1]["qc"]["interf_diag"].get("all_G1")),
                      f(v["interf_attempts"][-1]["qc"]["interf_diag"].get("all_G2")),
                      f(v["interf_attempts"][-1]["qc"]["interf_diag"].get("all_G3")),
                      "/".join(f(v["interf_attempts"][-1]["qc"]["interf_diag"].get(f"matched_{x['caliper_selected']}", {}).get(gn))
                               for gn in ("G1", "G2", "G3")),
                      f(v["interf_attempts"][-1]["qc"]["interf_diag"].get("Ilost_post_maxprob_mean"))]
                     for s, v in x["seeds"].items()]))


def val(r, md):
    if "VAL" not in r or "PASS" not in r["VAL"]:
        return
    x = r["VAL"]
    md.append(f"\n## VAL: complete recipe once on 9011–9013 (caliper {x['caliper']}). PASS = **{x['PASS']}**\n")
    ck = str(x["caliper"])
    rows = []
    for s, v in x["seeds"].items():
        fq = v["forget"][-1]["qc"]
        iq = v["interf"][-1]["qc"]
        nq = v["new"][-1]["qc"]
        mq = v["fam"][-1]["qc"]
        fp = (fq.get("fingerprint") or {}).get("matched", {}).get(ck, {})
        rows.append([s, v["pass"], f"{v['forget'][-1]['pass']} ({len(v['forget'])})", f(fq["X_lost_frac"]),
                     f(fq["Z_lost_max_over_categories"]), fq["matched_pairs"][ck], f(fp.get("max_G123")),
                     f"{v['interf'][-1]['pass']} ({len(v['interf'])})", f(iq["lost_frac"]), iq["matched_pairs"][ck],
                     f"{v['new'][-1]['pass']} ({len(v['new'])})", f(nq["N_learned_frac"]), f(nq["collateral_bc_lost"]),
                     f"{v['fam'][-1]['pass']} ({len(v['fam'])})", f(mq["fluency_rise_rel_sd"], 2)])
    md.append(table(["seed", "PASS", "FORGET (rungs)", "X lost", "Z worst", "F pairs", "max G1–G3", "INTERF (rungs)",
                     "I lost", "I pairs", "NEW (rungs)", "N learned", "NEW collateral", "FAM (rungs)", "FAM rise SD"], rows))
    detail({s: v["forget"][-1]["qc"] for s, v in x["seeds"].items()}, md, "VAL FORGET per-seed detail")


def v3_detail(seeds, md, title):
    md.append(f"\n### {title}\n")
    rows = []
    for s, v in seeds.items():
        for name in ("X", "Y") + CATS + ("R_sibling_X",):
            rep = v["sets"].get(name, {"n": 0})
            if not rep.get("n"):
                continue
            rt = v["retention"].get(name, {})
            rows.append([s, name, rep["n"], f(rep["margin_pre_mean"], 2), f(rep["lost_frac"]), f(rep["margin_change_mean"], 2),
                         f(rep["logp_correct_change_mean"], 3), f(rep["kl_answer_dist_mean"], 4), f(rep["disp_mean"], 3),
                         f(rt.get("ratio_margin")), f(rt.get("ratio_logp")), f(rt.get("ratio_kl")), f(rt.get("abs_margin_over_DC")),
                         f(rt.get("load"), 2)])
    md.append(table(["seed", "set", "n", "margin pre", "lost", "dmargin", "dlog p(v*)", "KL", "disp", "ratio margin",
                     "ratio log p", "ratio KL", "abs/D_C", "load (<=1 passes)"], rows))
    rows = []
    for s, v in seeds.items():
        idf, bs = v["identifiability"], v["binary_secondary"]
        rows.append([s, f(v["X_lost_frac"]), f(v["C_post_X_iqr"], 2), f(v["D_C"], 2), f(v["retention_load_max"], 2),
                     " ".join(f"{k}:{'Y' if ok else 'N'}" for k, ok in v["retention_pass_by_level"].items()),
                     v["sites_within_tol"], f(v["fluency_change_sd"], 3), f(idf["r2_gen"], 2), f(idf["r2_pre"], 2),
                     f(idf["r2_joint"], 2), f(v["fingerprint_vs_dC_spearman"], 2), f"{bs['pairs']}/{f(bs['max_abs_smd'], 2)}",
                     ",".join(v["failed_gates"]) or "—"])
    md.append("\n" + table(["seed", "X lost", "IQR C_post", "D_C", "retention load", "retention pass @.05/.10/.20",
                            "Y sites", "dfluency SD", "R2_gen", "R2_pre", "R2_joint", "rho(F_i, dC)", "K1 pairs/SMD",
                            "failed gates"], rows))


def v3_tables(r, md):
    R = r.get("v3")
    if not R:
        return
    md.append("\n# Revision v3 (within-target design; dev seeds 9031-9033, validation 9021-9023)\n")
    for key, title in (("C2dev", "C2dev development stores"), ("C2val", "C2val validation stores")):
        if key in R:
            md.append(f"\n## {title}\n")
            md.append(table(["seed", "trained acc", "unknown acc", "fluency AUROC", "pass", "min"],
                            [[s, f(v["qc"]["trained_fact_acc"]), f(v["qc"]["unknown_fact_acc"]),
                              f(v["qc"]["fluency_auroc_high_vs_low"]), v["qc"]["pass"], f(v["minutes"], 1)]
                             for s, v in R[key]["seeds"].items()]))
    if "C4dev" in R:
        md.append("\n## C4dev T-FORGET mechanism family (V0-V3)\n")
        rows = []
        for row in R["C4dev"]["grid"]:
            S = list(row["seeds"].values())
            rows.append([row["variant"], row["lr"], row["steps"], row["anchor_lambda"], _rng([v["X_lost_frac"] for v in S]),
                         f(min(v["C_post_X_iqr"] for v in S), 2), f(max(v["Y_lost_frac"] for v in S)),
                         f(max(v["Z_lost_max_over_categories"] for v in S)), f(row["worst_load"], 2),
                         min(v["sites_within_tol"] for v in S), f(max(abs(v["fluency_change_sd"]) for v in S), 3),
                         "/".join(f(row["identifiability_median"][k], 2) for k in ("r2_gen", "r2_pre", "r2_joint")),
                         min(v["binary_secondary"]["pairs"] for v in S),
                         ",".join(sorted({g for v in S for g in v["failed_gates"]})) or "—", "**yes**" if row["eligible"] else "no"])
        md.append(table(["variant", "lr", "steps", "anchor", "X lost", "IQR min", "Y lost max", "Z lost max", "worst load",
                         "sites min", "max abs dflu", "median R2 gen/pre/joint", "K1 pairs min", "failed gates", "eligible"], rows))
        md.append(f"\n**Selected:** {R['C4dev'].get('selected')} {R['C4dev'].get('selected_summary')}\n")
        sel = R["C4dev"].get("selected")
        if sel:
            row = next(x for x in R["C4dev"]["grid"] if all(x[k] == sel[k] for k in sel))
            v3_detail(row["seeds"], md, "Selected recipe on dev seeds")
    if "C4val" in R:
        x = R["C4val"]
        md.append(f"\n## C4val: frozen recipe once on 9021-9023. PASS = **{x['PASS']}**\n")
        md.append(f"identifiability medians {x['identifiability_median']}\n\n")
        md.append(table(["seed", "rungs", "base passed", "per-seed pass", "F6"],
                        [[s, len(v["attempts"]), v["base_config_passed"], v["pass"], v["F6"]] for s, v in x["seeds"].items()]))
        v3_detail({s: v["attempts"][-1]["qc"] for s, v in x["seeds"].items()}, md, "C4val per-seed detail")
    if "C7" in R:
        md.append("\n## C7 T-INTERF (dev seeds)\n")
        md.append(table(["lr", "seed", "lost", "steps used", "new-fact acc", "R2 gen/pre/joint", "IQR C_post", "pairs .25/.5/.75/1"],
                        [[row["lr"], s, f(v["lost_frac"]), v["steps_used"], f(v["final_new_acc"]),
                          "/".join(f(v["identifiability"][k], 2) for k in ("r2_gen", "r2_pre", "r2_joint")), f(v["C_post_iqr"], 2),
                          "/".join(str(v["matched_pairs"][c]) for c in ("0.25", "0.5", "0.75", "1.0"))]
                         for row in R["C7"]["grid"] for s, v in row["seeds"].items()]))
        md.append(f"\n**Selected:** {R['C7'].get('selected')}\n")
    if "C8" in R:
        md.append("\n## C8 T-NEW (dev seeds)\n")
        md.append(table(["lr", "steps", "lambda", "N learned min", "U correct max", "coll bc lost max", "coll bc abs/D_C max",
                         "coll near abs/D_C max", "pass"],
                        [[row["lr"], row["steps"], row["lambda_retain"], f(min(v["N_learned_frac"] for v in row["seeds"].values())),
                          f(max(v["U_correct_frac"] for v in row["seeds"].values())),
                          f(max(v["collateral_bc"]["lost"] for v in row["seeds"].values())),
                          f(max(v["collateral_bc"]["abs_margin_over_DC"] for v in row["seeds"].values())),
                          f(max(v["collateral_near_entity"]["abs_margin_over_DC"] for v in row["seeds"].values())), row["all_pass"]]
                         for row in R["C8"]["grid"]]))
        md.append(f"\n**Selected:** {R['C8'].get('selected')}\n")
    if "C9" in R:
        md.append("\n## C9 T-FAM (dev seeds)\n")
        md.append(table(["steps", "lr", "rise SD min", "F correct max", "coll abs/D_C max", "pass"],
                        [[row["steps"], row["lr"], f(min(v["fluency_rise_rel_sd"] for v in row["seeds"].values()), 2),
                          f(max(v["F_correct_frac"] for v in row["seeds"].values())),
                          f(max(v["collateral_bc"]["abs_margin_over_DC"] for v in row["seeds"].values())), row["all_pass"]]
                         for row in R["C9"]["grid"]]))
        md.append(f"\n**Selected:** {R['C9'].get('selected')}\n")
    if "C10" in R and "caliper_selected" in R["C10"]:
        x = R["C10"]
        md.append(f"\n## C10 Y negative control and descriptive caliper (dev seeds): caliper {x['caliper_selected']} "
                  f"(rule met: {x['caliper_rule_met']})\n")
        md.append(table(["seed", "FORGET pass", "X-Y pairs .25/.5/.75/1", "Y fingerprint AUROC (40 generic; matched @.5)",
                         "X all vs Y all", "INTERF pass", "INTERF lost"],
                        [[s, v["forget_attempts"][-1]["pass"],
                          "/".join(str(v["forget_attempts"][-1]["qc"]["matched_pairs_Xlost_Y"][c]) for c in ("0.25", "0.5", "0.75", "1.0")),
                          f(v["Y_fingerprint"]["matched"]["0.5"].get("G1-G3_all40")), f(v["Y_fingerprint"]["X_all_vs_Y_all40"]),
                          v["interf_attempts"][-1]["pass"], f(v["interf_attempts"][-1]["qc"]["lost_frac"])]
                         for s, v in x["seeds"].items()]))
    if "VAL" in R and "PASS" in R["VAL"]:
        x = R["VAL"]
        md.append(f"\n## VAL: complete recipe once on 9021-9023. PASS = **{x['PASS']}**\n")
        md.append(f"P1 identifiability {x['identifiability_P1']}; P2 identifiability {x['identifiability_P2']}\n\n")
        md.append(table(["seed", "PASS", "FORGET (rungs)", "F6", "INTERF (rungs)", "INTERF lost", "NEW (rungs)", "FAM (rungs)"],
                        [[s, v["pass"], f"{v['forget'][-1]['pass']} ({len(v['forget'])})",
                          v["forget"][-1]["qc"]["binary_secondary"]["feasible"], f"{v['interf'][-1]['pass']} ({len(v['interf'])})",
                          f(v["interf"][-1]["qc"]["lost_frac"]), f"{v['new'][-1]['pass']} ({len(v['new'])})",
                          f"{v['fam'][-1]['pass']} ({len(v['fam'])})"] for s, v in x["seeds"].items()]))
        v3_detail({s: v["forget"][-1]["qc"] for s, v in x["seeds"].items()}, md, "VAL T-FORGET per-seed detail")


def main():
    r = json.load(open(RES))
    md = ["# Calibration tables (auto-generated from `results/raw/calibration/calibration_results.json`)\n\n"
          "No monitor or controller quantity exists in calibration (enforced by unit test).\n"]
    for fn in (c1c2, c3, c4v1, c4v2, c4val, c7c8c9, c10, val, v3_tables):
        fn(r, md)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(md))
    print(OUT)


if __name__ == "__main__":
    main()
