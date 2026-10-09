"""Markdown result tables for the A-stage (selection per model/layer, choice, confirmation)."""
from __future__ import annotations

import json
import os

from .selection import layer_gate


def _f(x, nd=3):
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _tbl(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def model_table(res):
    w0 = res["w0"]
    lines = [f"### {res['model']} (G_{'select' if 'verdict' not in res else 'confirm'})",
             f"W0: lens agreement@L-2 {_f(w0['w0a_lens_agreement_L-2'])}, two-hop acc {_f(w0['two_hop_acc'])} "
             f"(n={w0['two_hop_n']}), intermediate rate {_f(w0['intermediate_rate_among_correct'])} -> "
             f"{'PASS' if w0['pass'] else 'FAIL'}", ""]
    rows = []
    for l, pl in res["per_layer"].items():
        g, m, ok, mm = layer_gate(w0, pl)
        w1, w2, w3, w45 = pl["w1"], pl["w2"], pl["w3"], pl["w45"]
        rows.append([str(l), str(pl["sj"]["r"]), _f(pl["dose"]["a_star"], 2),
                     f"{_f(w1['hit_J'], 2)}/{_f(w1['hit_perp'], 2)}/{_f(w1['hit_none'], 2)}",
                     f"{_f(w2['rate_J'], 2)}/{_f(w2['rate_perp'], 2)} (n={w2['n_pairs_included']})",
                     f"{_f(w3['BB_J'], 2)}/{_f(w3['BB_perp'], 2)} [{_f(w3['ci'][0], 2)},{_f(w3['ci'][1], 2)}]",
                     f"{_f(w45['imp_J_pp'], 1)}/{_f(w45['imp_rand1_pp'], 1)}",
                     f"{_f(w45['SI_iso'], 2)} [..{_f(w45['SI_iso_ci'][1], 2)}] s*={_f(w45['s_star'], 2)}",
                     _f(w45["SI_norm"], 2),
                     "".join("Y" if g[k] else "n" for k in ("W0", "W1", "W2", "W3", "W4", "W5")),
                     _f(mm, 2)])
    lines.append(_tbl(rows, ["layer", "r", "a*", "W1 hit J/perp/none", "W2 rate J/perp", "W3 BB J/perp [CI]",
                             "W4 imp J/rand1 (pp)", "W5 SI_iso [CI up]", "SI_norm", "gates W0-5", "min margin"]))
    return "\n".join(lines)


def write_report(res_dir):
    parts = ["# C15-R A-stage result tables (auto-generated)", ""]
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("select_") and fn.endswith(".json"):
            parts += [model_table(json.load(open(os.path.join(res_dir, fn), encoding="utf-8"))), ""]
    ch = os.path.join(res_dir, "choose_astage.json")
    if os.path.isfile(ch):
        c = json.load(open(ch, encoding="utf-8"))
        parts += ["## Choice", f"passers: {c['n_passers']}; chosen: {c['chosen'] and (c['chosen']['model'], c['chosen']['layer'])}; "
                  f"replication candidate: {c['replication_candidate'] and (c['replication_candidate']['model'], c['replication_candidate']['layer'])}", ""]
    cf = os.path.join(res_dir, "confirm_astage.json")
    if os.path.isfile(cf):
        c = json.load(open(cf, encoding="utf-8"))
        parts += ["## Confirmation (G_confirm, run once)", model_table(c), "",
                  f"Verdict: {'PASS' if c['verdict']['pass'] else 'FAIL'}; W5: {c['verdict']['w5_description']}", ""]
    p = os.path.join(res_dir, "astage_tables.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("wrote", p)
