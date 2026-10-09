"""Build the frozen A-stage material splits G_select / G_confirm / SMOKE (seed 9300).

Tokenizer-only filters (no model run): paragraphs must give >= 64 tokens and concept nouns must be single
tokens (with a leading space) under all three candidate tokenizers. All trial designs (W1 contexts and
distractors, W2 pairs, W3 concepts/contexts/scalars, dose-rule sequences) are drawn here, so that they are
fixed and hashed before any model is run.

Usage (from repo root): .venv/Scripts/python.exe research/experiments/c15/tools/make_splits.py
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C15 = os.path.dirname(HERE)
sys.path.insert(0, C15)
sys.path.insert(0, os.path.join(C15, "materials"))

from c15a import config as C  # noqa: E402
from c15a.artifacts import manifest, model_dir  # noqa: E402
import source_items as SI  # noqa: E402
import source_paragraphs as SP  # noqa: E402

MAT = os.path.join(C15, "materials")


def tokenizers():
    from transformers import AutoTokenizer
    return {k: AutoTokenizer.from_pretrained(model_dir(k)) for k in manifest()["models"]}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def designs(rng, paras, concepts, countries, n_w1_ctx, n_w3_conc, n_w3_ctx, n_dose_seq):
    pid = [p["id"] for p in paras]
    cid = [c["id"] for c in concepts]
    w1 = []
    for c in cid:
        ctx = list(rng.choice(pid, size=n_w1_ctx, replace=False))
        others = [x for x in cid if x != c]
        dis = list(rng.choice(others, size=min(C.W1_N_OPTIONS - 1, len(others)), replace=False))
        opts = [c] + dis
        rng.shuffle(opts)
        w1.append({"concept": c, "contexts": ctx, "options": opts})
    names = [x["id"] for x in countries]
    while True:
        perm = list(rng.permutation(names))
        if all(a != b for a, b in zip(names, perm)):
            break
    w2 = [{"A": a, "B": b} for a, b in zip(names, perm)]
    w3_conc = list(rng.permutation(cid))[:n_w3_conc]
    w3_ctx = list(rng.choice(pid, size=n_w3_ctx, replace=False))
    reps = int(np.ceil(n_w3_ctx / len(C.W3_SCALARS)))
    scal = list(rng.permutation(np.tile(C.W3_SCALARS, reps)[:n_w3_ctx]))
    dose = {"paragraphs": list(rng.choice(pid, size=n_dose_seq, replace=False)), "concepts": w3_conc}
    return {"w1_trials": w1, "w2_pairs": w2,
            "w3": {"concepts": w3_conc, "contexts": w3_ctx, "scalars": [float(s) for s in scal]},
            "dose": dose}


def main():
    toks = tokenizers()
    rng = np.random.default_rng(C.SEED)
    dropped = {"paragraphs": [], "nouns": []}

    def long_enough(text):
        return all(len(t(text, add_special_tokens=True).input_ids) >= C.SEQ_LEN for t in toks.values())

    def single(word):
        return all(len(t(" " + word, add_special_tokens=False).input_ids) == 1 for t in toks.values())

    paras = []
    for i, p in enumerate(SP.PARAGRAPHS):
        (paras.append({"id": f"p{i:03d}", "text": p}) if long_enough(p) else dropped["paragraphs"].append(i))
    smoke_paras = [{"id": f"sp{i}", "text": p} for i, p in enumerate(SP.SMOKE_PARAGRAPHS) if long_enough(p)]
    ok_nouns = []
    for w in SI.CONCEPT_NOUNS:
        (ok_nouns.append(w) if single(w) else dropped["nouns"].append(w))
    if len(ok_nouns) < 50:
        raise SystemExit(f"only {len(ok_nouns)} single-token nouns; need 50")
    main_nouns, smoke_nouns = ok_nouns[:40], ok_nouns[40:50]
    concepts = [{"id": f"c_{w}", "word": w} for w in main_nouns]
    countries = [{"id": f"k_{n}", "name": n, "capital": c, "language": l, "currency": u}
                 for n, c, l, u in SI.COUNTRIES]
    bridges = sorted({b for b, *_ in SI.TWO_HOP})
    two_hop = [{"id": f"t{i:03d}", "bridge": b, "prompt": p, "answer": a, "intermediates": im}
               for i, (b, p, a, im) in enumerate(SI.TWO_HOP)]

    def halves(items):
        idx = rng.permutation(len(items))
        h = len(items) // 2 + (len(items) % 2)
        return [items[i] for i in sorted(idx[:h])], [items[i] for i in sorted(idx[h:])]

    p_sel, p_con = halves(paras)
    c_sel, c_con = halves(concepts)
    k_sel, k_con = halves(countries)
    b_sel, b_con = halves(bridges)
    t_sel = [t for t in two_hop if t["bridge"] in set(b_sel)]
    t_con = [t for t in two_hop if t["bridge"] in set(b_con)]

    common = {"report_suffix": SI.REPORT_SUFFIX, "country_queries": SI.COUNTRY_QUERIES}
    out = {}
    for name, ps, cs, ks, ts, ctpl, ktpl in (
            ("select", p_sel, c_sel, k_sel, t_sel, SI.CONCEPT_TEMPLATES_SELECT, SI.COUNTRY_TEMPLATES_SELECT),
            ("confirm", p_con, c_con, k_con, t_con, SI.CONCEPT_TEMPLATES_CONFIRM, SI.COUNTRY_TEMPLATES_CONFIRM)):
        d = designs(rng, ps, cs, ks, C.W1_CONTEXTS_PER_CONCEPT, C.W3_N_CONCEPTS, C.W3_N_CONTEXTS, C.DOSE_N_SEQ)
        out[name] = {"split": name, "paragraphs": ps, "concepts": cs, "concept_templates": ctpl,
                     "countries": ks, "country_templates": ktpl, "two_hop": ts, **common, **d}
    s_conc = [{"id": f"sc_{w}", "word": w} for w in smoke_nouns]
    s_ctry = [{"id": f"sk_{n}", "name": n, "capital": c, "language": l, "currency": u}
              for n, c, l, u in SI.SMOKE_COUNTRIES]
    s_two = [{"id": f"st{i}", "bridge": b, "prompt": p, "answer": a, "intermediates": im}
             for i, (b, p, a, im) in enumerate(SI.SMOKE_TWO_HOP)]
    sd = designs(rng, smoke_paras, s_conc, s_ctry, 2, 1, 6, 2)
    sd["w1_trials"] = sd["w1_trials"][:2]
    out["smoke"] = {"split": "smoke", "non_evidential": True, "paragraphs": smoke_paras, "concepts": s_conc,
                    "concept_templates": SI.CONCEPT_TEMPLATES_SMOKE, "countries": s_ctry,
                    "country_templates": SI.COUNTRY_TEMPLATES_SMOKE, "two_hop": s_two, **common, **sd}

    # disjointness checks
    def texts(s):
        return {p["text"] for p in s["paragraphs"]} | {c["word"] for c in s["concepts"]} | \
            {k["name"] for k in s["countries"]} | {t["prompt"] for t in s["two_hop"]}
    a, b, c = texts(out["select"]), texts(out["confirm"]), texts(out["smoke"])
    assert not (a & b) and not (a & c) and not (b & c), "split overlap"

    files = {}
    for name in ("select", "confirm", "smoke"):
        fn = {"select": "g_select.json", "confirm": "g_confirm.json", "smoke": "smoke.json"}[name]
        p = os.path.join(MAT, fn)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out[name], f, indent=1, ensure_ascii=False)
        files[fn] = sha(p)
    src = {fn: sha(os.path.join(MAT, fn)) for fn in ("source_paragraphs.py", "source_items.py")}
    man = {"seed": C.SEED, "files": files, "sources": src, "dropped": dropped,
           "counts": {k: {"paragraphs": len(v["paragraphs"]), "concepts": len(v["concepts"]),
                          "countries": len(v["countries"]), "two_hop": len(v["two_hop"])} for k, v in out.items()},
           "tokenizer_sha256": {k: sha(os.path.join(model_dir(k), "tokenizer.json")) for k in toks}}
    with open(os.path.join(MAT, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man["counts"]), "dropped:", dropped)


if __name__ == "__main__":
    main()
