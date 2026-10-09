"""R2 result tables and the final verdict (auto-generated markdown + JSON)."""
from __future__ import annotations

import json
import os

from .classify import cell


def _f(x, nd=2):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _ci(c, nd=2):
    return f"[{_f(c[0], nd)},{_f(c[1], nd)}]" if isinstance(c, list) and len(c) == 2 else "-"


def model_table(res):
    w0 = res["w0"]
    bp = w0["w0b_by_position"]
    lines = [f"### {res['model']}",
             f"W0a lens agreement@L-2 {_f(w0['w0a_lens_agreement_L-2'], 3)} {_ci(w0['w0a_ci'], 3)} -> "
             f"{'PASS' if w0['pass_w0a'] else 'FAIL'}; two-hop acc {_f(w0['two_hop_acc'])} (n={w0['two_hop_n']})",
             "W0b' (primary t_d): " + "; ".join(
                 f"{p}{'*' if p == 'td' else ''} rate {_f(v['rate'])} foil {_f(v['foil_rate'])} lift {_f(v['lift'])} "
                 f"(LB {_f(v['lift_lb_one_sided_95'])})" for p, v in bp.items())
             + f" -> {'PASS' if w0['pass_w0b'] else 'FAIL'}", ""]
    hdr = ["layer", "a*", "W1 hit J/perp [CI J]", "W2 rate J/perp (n)", "W3' BB J/perp diff [CI] ctrl",
           "W3b J/perp/nat", "W4 diff pp [CI]", "W5 SI_iso [CI]", "SI_norm", "gates 0a0b1 2 3'4 5", "valid", "PF"]
    rows = []
    for l, pl in res["per_layer"].items():
        c = cell(res, l)
        w1, w2, w3, w45 = pl["w1"], pl["w2"], pl["w3p"], pl["w45"]
        bb = w3.get("BB", {})
        ctrl = "ok" if w3.get("assessable") else "NA"
        w3b = w3.get("w3b", {})
        rows.append([str(l), _f(pl["dose"]["a_star"]),
                     f"{_f(w1['hit_J'])}/{_f(w1['hit_perp'])} {_ci(w1['hit_J_ci'])}",
                     f"{_f(w2['rate_J'])}/{_f(w2['rate_perp'])} ({w2['n_pairs_included']})",
                     f"{_f(bb.get('J', '-'))}/{_f(bb.get('perp', '-'))} {_f(w3.get('diff', '-'))} {_ci(w3.get('ci'))} {ctrl}",
                     f"{_f(w3b.get('hit_J', '-'))}/{_f(w3b.get('hit_perp', '-'))}/{_f(w3b.get('hit_natural_word', '-'))}",
                     f"{_f(w45['w4_diff_pp'], 1)} {_ci(w45['w4_diff_ci'], 1)}",
                     f"{_f(w45['SI_iso'])} {_ci(w45['SI_iso_ci'])}", _f(w45["SI_norm"]),
                     "".join("Y" if c["gates"][g] else "n" for g in ("W0a", "W0b", "W1", "W2", "W3p", "W4", "W5")),
                     "Y" if c["valid"] else "n",
                     ",".join(g for g, v in c["pf"].items() if v) or "-"])
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(res_dir, pins):
    parts = ["# C15-R2 result tables (auto-generated)", ""]
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("select2_") and fn.endswith(".json"):
            parts += [model_table(json.load(open(os.path.join(res_dir, fn), encoding="utf-8"))), ""]
    verdict = {"pins_end": pins}
    cp = os.path.join(res_dir, "classify2.json")
    if os.path.isfile(cp):
        cl = json.load(open(cp, encoding="utf-8"))
        verdict["pattern_select"] = cl["pattern"]
        parts += ["## Classification on G_select2", f"Pattern: **{cl['pattern']}** {cl.get('reason', '')}; "
                  f"frozen cells: {[(c['model'], c['layer'], c['role']) for c in cl['frozen_cells']]}; "
                  f"assay-valid cells {cl['n_valid_cells']}/{cl['n_cells']}", ""]
        confs = []
        for fc in cl["frozen_cells"]:
            p = os.path.join(res_dir, f"confirm2_{fc['model']}_L{fc['layer']}.json")
            if os.path.isfile(p):
                c = json.load(open(p, encoding="utf-8"))
                confs.append(c["verdict"]["reproduced"])
                parts += [f"## Confirmation (G_confirm2, once): {fc['model']} layer {fc['layer']} ({fc['role']})",
                          model_table(c), "", f"Reproduced: {c['verdict']['reproduced']}", ""]
        if cl["frozen_cells"]:
            if len(confs) == len(cl["frozen_cells"]):
                verdict["final"] = f"{cl['pattern']} confirmed" if all(confs) else "not confirmed"
            else:
                verdict["final"] = "confirmation incomplete"
        else:
            verdict["final"] = cl["pattern"]
        parts += [f"## Final verdict: **{verdict['final']}**", ""]
    with open(os.path.join(res_dir, "r2_tables.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))
    with open(os.path.join(res_dir, "verdict2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(verdict, f, indent=1)
    print("wrote", os.path.join(res_dir, "r2_tables.md"))
