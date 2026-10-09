"""Build the frozen R2 material splits G_select2 / G_confirm2 / SMOKE2 (seed 9500 splits, 9501 designs).

Tokenizer-only filters and checks (no model run). Every trial design is drawn here, so that it is fixed and hashed
before any model is run. The build fails (no output) if any check fails:
  * paragraphs >= 64 tokens; concept nouns and ' thing' single tokens (all three tokenizers);
  * paraphrases contain neither the concept word (any case, plural) nor any token id of the word's tokenizations;
  * slot contexts: prefix >= 20 tokens; continuation >= 24 tokens and token-identical after every slot version;
    no concept word and no 'thing' inside the context;
  * two-hop items: e1 / descriptive-mention spans unique in the prompt, prefix-stable tokenization, t1 <= td < t2;
    intermediate absent from its prompt; answer first token differs from the intermediate's;
  * foils: same type, not mentioned in the prompt, first token distinct from the intermediate's and the answer's;
  * T2 (no entity / text overlap with A-stage material) and T3 (pairwise split disjointness);
  * vocabulary audit (no SAT / validity vocabulary).

The A-stage overlap check reads the A-stage *source* files (source_items.py, source_paragraphs.py) as text; no
model sees them and the sealed A-stage confirm split file is never opened here.

Usage (repo root): .venv/Scripts/python.exe research/experiments/c15r2/tools/make_splits2.py
"""
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
sys.path.insert(0, R2)
sys.path.insert(0, os.path.join(R2, "materials2"))

import c15a2  # noqa: E402,F401  (sets up the read-only A-stage import path)
from c15a2 import config as C  # noqa: E402
from c15a2.guards import sha256_file  # noqa: E402
from c15a.artifacts import manifest, model_dir  # noqa: E402
import source_contexts2 as SC  # noqa: E402
import source_items2 as SI  # noqa: E402
import source_paragraphs2 as SP  # noqa: E402

MAT2 = os.path.join(R2, "materials2")
ASTAGE_MAT = os.path.join(os.path.dirname(R2), "c15", "materials")
SEQ_LEN = 64
BANNED = ("satisf", "clause", "valid", "verif", "assignment")
STOP = {"the", "of", "and", "a", "an", "in", "on", "de", "la", "at", "to"}


class BuildError(RuntimeError):
    pass


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tokenizers():
    from transformers import AutoTokenizer
    return {k: AutoTokenizer.from_pretrained(model_dir(k)) for k in manifest()["models"]}


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower()) if w not in STOP]


def contains_words(a, b):
    """True if the word sequence of a is a contiguous subsequence of b's (or vice versa)."""
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    if len(wa) > len(wb):
        wa, wb = wb, wa
    n = len(wa)
    return any(wb[i:i + n] == wa for i in range(len(wb) - n + 1))


def ngrams(text, n=8):
    w = re.findall(r"[a-z]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


# --------------------------------------------------------------------------------------------- A-stage sets
def astage_sets():
    ai = _load_module(os.path.join(ASTAGE_MAT, "source_items.py"), "astage_items")
    ap = _load_module(os.path.join(ASTAGE_MAT, "source_paragraphs.py"), "astage_paragraphs")
    ent = set(w.lower() for w in ai.CONCEPT_NOUNS)
    for row in ai.COUNTRIES + ai.SMOKE_COUNTRIES:
        ent.add(row[0].lower())
        ent.add(row[1].lower())
    prompts = []
    for b, p, a, im in ai.TWO_HOP + ai.SMOKE_TWO_HOP:
        ent.add(b.lower())
        ent.add(a.lower())
        ent.update(x.lower() for x in im)
        prompts.append(p)
        for w in re.findall(r"\b[A-Z][A-Za-z'\-]+", p)[1:]:
            if w not in ("Fact", "The"):
                ent.add(w.lower())
    texts = list(ap.PARAGRAPHS) + list(ap.SMOKE_PARAGRAPHS)
    texts += ai.CONCEPT_TEMPLATES_SELECT + ai.CONCEPT_TEMPLATES_CONFIRM + ai.CONCEPT_TEMPLATES_SMOKE
    texts += ai.COUNTRY_TEMPLATES_SELECT + ai.COUNTRY_TEMPLATES_CONFIRM + ai.COUNTRY_TEMPLATES_SMOKE
    grams = set()
    for t in texts:
        grams |= ngrams(t)
    return ent, set(texts) | set(prompts), grams


# --------------------------------------------------------------------------------------------- checks
def check_paragraph_lengths(toks, paras):
    bad = [i for i, p in enumerate(paras)
           if any(len(t(p, add_special_tokens=True).input_ids) < SEQ_LEN for t in toks.values())]
    if bad:
        raise BuildError(f"paragraphs shorter than {SEQ_LEN} tokens: {bad}")


def single_token(toks, word):
    return all(len(t(" " + word, add_special_tokens=False).input_ids) == 1 for t in toks.values())


def forbidden_ids(t, word):
    """Token ids that would carry the concept word's surface form: the single token ' w' and every variant
    (no space / capitalised / plural, with or without space) that is itself a single token."""
    ids = set(t(" " + word, add_special_tokens=False).input_ids)
    for v in (word, word.capitalize(), " " + word.capitalize(), word + "s", " " + word + "s", " " + word + "es"):
        enc = t(v, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def check_paraphrases(toks, word, paras):
    for p in paras:
        if not p.startswith("the "):
            raise BuildError(f"paraphrase must start with 'the ': {p!r}")
        if re.search(r"\b" + re.escape(word), p, re.I) or re.search(r"\bthing\b", p, re.I):
            raise BuildError(f"paraphrase of {word!r} contains the word or 'thing': {p!r}")
        for k, t in toks.items():
            shared = set(t(" " + p, add_special_tokens=False).input_ids) & forbidden_ids(t, word)
            if shared:
                raise BuildError(f"paraphrase of {word!r} contains word tokens {shared} ({k}): {p!r}")


def check_context(toks, ctx, versions, banned_words):
    prefix, cont = ctx
    low = (prefix + " " + cont).lower()
    for w in list(banned_words) + ["thing"]:
        if re.search(r"\b" + re.escape(w) + r"(s|es)?\b", low):
            raise BuildError(f"context contains {w!r}: {prefix[:40]!r}")
    for k, t in toks.items():
        if len(t(prefix, add_special_tokens=True).input_ids) < 20:
            raise BuildError(f"prefix < 20 tokens ({k}): {prefix[:40]!r}")
        ref = None
        for v in versions:
            head = t(prefix + v, add_special_tokens=True).input_ids
            full = t(prefix + v + cont, add_special_tokens=True).input_ids
            if full[:len(head)] != head:
                raise BuildError(f"slot tokenization not prefix-stable ({k}): {v!r}")
            tail = full[len(head):]
            withs = t(prefix + v + cont + SI.W3B_SUFFIX, add_special_tokens=True).input_ids
            last = len(head) + max(C.W3P_OFFSETS)          # one past the last W3' site index
            if withs[:last] != full[:last]:
                raise BuildError(f"W3b suffix changes W3' site tokens ({k}): {cont[-30:]!r}")
            if ref is None:
                ref = tail
            elif tail != ref:
                raise BuildError(f"continuation tokens differ across versions ({k}): {v!r}")
        if len(ref) < 24:
            raise BuildError(f"continuation < 24 tokens ({k}): {cont[:40]!r}")


def token_pos(t, prompt, end):
    head = t(prompt[:end], add_special_tokens=True).input_ids
    full = t(prompt, add_special_tokens=True).input_ids
    if full[:len(head)] != head:
        raise BuildError(f"span tokenization not prefix-stable: {prompt[:end]!r}")
    return len(head) - 1, len(full) - 1


def first_tok(t, s):
    return t(" " + s, add_special_tokens=False).input_ids[0]


def span(prompt, sub):
    for s in (sub, sub[:1].upper() + sub[1:]):
        i = prompt.find(s)
        if i >= 0:
            if prompt.find(s, i + 1) >= 0:
                raise BuildError(f"span {s!r} not unique in {prompt!r}")
            return [i, i + len(s)]
    raise BuildError(f"span {sub!r} not found in {prompt!r}")


def build_two_hop(toks, bridges, foil_pool, rng, sid):
    items = []
    for bid, typ, aliases, e1, its, inter in bridges:
        for j, (prompt, answer, desc) in enumerate(its):
            if re.search(r"\b" + re.escape(inter) + r"\b", prompt, re.I):
                raise BuildError(f"intermediate {inter!r} appears in {prompt!r}")
            e1s, ds = span(prompt, e1), span(prompt, desc)
            for k, t in toks.items():
                t1, t2 = token_pos(t, prompt, e1s[1])
                td, _ = token_pos(t, prompt, ds[1])
                if not (t1 <= td < t2):
                    raise BuildError(f"positions t1={t1} td={td} t2={t2} invalid ({k}): {prompt!r}")
                if first_tok(t, answer) in {first_tok(t, a) for a in aliases}:
                    raise BuildError(f"answer first token equals intermediate's ({k}): {prompt!r}")
            cands = [f for f in foil_pool[typ] if f != inter and not re.search(r"\b" + re.escape(f) + r"\b", prompt, re.I)]
            ok = []
            for f in cands:
                if all(first_tok(t, f) not in {first_tok(t, a) for a in aliases} | {first_tok(t, answer)}
                       for t in toks.values()):
                    ok.append(f)
            if len(ok) < C.W0B_FOILS:
                raise BuildError(f"only {len(ok)} admissible foils for {bid}")
            foils = [ok[i] for i in sorted(rng.choice(len(ok), size=C.W0B_FOILS, replace=False))]
            items.append({"id": f"{sid}_{bid}_{j}", "bridge": bid, "type": typ, "prompt": prompt, "answer": answer,
                          "intermediates": list(aliases), "intermediate_entity": inter, "e1": e1, "e1_span": e1s,
                          "desc": desc, "desc_span": ds, "foils": foils})
    return items


def designs(rng, paras, concepts, countries, n_w1_ctx, n_dose_seq, dose_concepts):
    """W1 trials, W2 pairs and dose design: identical procedure to the A-stage make_splits.designs()."""
    pid = [p["id"] for p in paras]
    cid = [c["id"] for c in concepts]
    w1 = []
    for c in cid:
        ctx = list(rng.choice(pid, size=n_w1_ctx, replace=False))
        others = [x for x in cid if x != c]
        dis = list(rng.choice(others, size=min(9, len(others)), replace=False))
        opts = [c] + dis
        rng.shuffle(opts)
        w1.append({"concept": c, "contexts": ctx, "options": opts})
    names = [x["id"] for x in countries]
    while True:
        perm = list(rng.permutation(names))
        if all(a != b for a, b in zip(names, perm)):
            break
    w2 = [{"A": a, "B": b} for a, b in zip(names, perm)]
    dose = {"paragraphs": list(rng.choice(pid, size=n_dose_seq, replace=False)), "concepts": list(dose_concepts)}
    return {"w1_trials": w1, "w2_pairs": w2, "dose": dose}


def derangement(rng, n):
    while True:
        p = list(rng.permutation(n))
        if all(i != j for i, j in enumerate(p)):
            return [int(x) for x in p]


def w3p_design(rng, toks, concept_rows, contexts, split_concepts, n_fold, word_of):
    """concept_rows: the W3' concepts; contexts: list of (prefix, cont)."""
    cids = [c["id"] for c in concept_rows]
    for c in concept_rows:
        check_paraphrases(toks, c["word"], SI.PARAPHRASES[c["word"]])
    versions = [" the thing"] + [" the " + c["word"] for c in concept_rows] + \
        [" " + p for c in concept_rows for p in SI.PARAPHRASES[c["word"]]]
    banned = [w for v in SI.CONCEPTS.values() for w in v] + list(SI.SMOKE_CONCEPTS)
    for ctx in contexts:
        check_context(toks, ctx, versions, banned)
    order = rng.permutation(len(contexts))
    ctx_rows = [{"id": f"x{i:03d}", "prefix": contexts[i][0], "cont": contexts[i][1]} for i in range(len(contexts))]
    fold_a = [ctx_rows[i]["id"] for i in sorted(order[:n_fold])]
    fold_b = [ctx_rows[i]["id"] for i in sorted(order[n_fold:2 * n_fold])]
    opts = {}
    all_ids = [c["id"] for c in split_concepts]
    for c in cids:
        others = [x for x in all_ids if x != c]
        dis = list(rng.choice(others, size=min(C.W3B_N_OPTIONS - 1, len(others)), replace=False))
        o = [c] + dis
        rng.shuffle(o)
        opts[c] = o
    return {"concepts": cids, "words": {c: word_of[c] for c in cids},
            "paraphrases": {c: SI.PARAPHRASES[word_of[c]] for c in cids},
            "contexts": ctx_rows, "fold_A": fold_a, "fold_B": fold_b,
            "derangement": derangement(rng, len(cids)), "w3b_options": opts,
            "carrier": " the thing", "w3b_suffix": SI.W3B_SUFFIX}


# --------------------------------------------------------------------------------------------- main
def main():
    toks = tokenizers()
    rs = np.random.default_rng(C.SEED_SPLITS)
    rd = np.random.default_rng(C.SEED_DESIGNS)
    a_ent, a_texts, a_grams = astage_sets()

    # paragraphs
    check_paragraph_lengths(toks, SP.PARAGRAPHS + SP.SMOKE_PARAGRAPHS)
    paras = [{"id": f"q{i:03d}", "text": p} for i, p in enumerate(SP.PARAGRAPHS)]
    idx = rs.permutation(len(paras))
    p_sel = [paras[i] for i in sorted(idx[:63])]
    p_con = [paras[i] for i in sorted(idx[63:126])]
    p_smk = [{"id": f"sq{i}", "text": p} for i, p in enumerate(SP.SMOKE_PARAGRAPHS)]

    # concepts: each category split 2/2
    for w in [w for v in SI.CONCEPTS.values() for w in v] + SI.SMOKE_CONCEPTS + ["thing"]:
        if not single_token(toks, w):
            raise BuildError(f"{w!r} is not a single token in all tokenizers")
    c_sel, c_con = [], []
    for cat, ws in SI.CONCEPTS.items():
        perm = rs.permutation(len(ws))
        for j, i in enumerate(perm):
            row = {"id": f"c2_{ws[i]}", "word": ws[i], "category": cat}
            (c_sel if j < 2 else c_con).append(row)
    c_smk = [{"id": f"sc2_{w}", "word": w, "category": "smoke"} for w in SI.SMOKE_CONCEPTS]

    # countries: stratified by number of bridges, split by country (intermediates never cross splits)
    by_country = defaultdict(list)
    for b in SI.COUNTRY_BRIDGES:
        by_country[b[5]].append(b)
    crow = {r[0]: r for r in SI.COUNTRIES}
    strata = defaultdict(list)
    for name in sorted(by_country):
        strata[len(by_country[name])].append(name)
    k_sel, k_con = [], []
    flip = int(rs.integers(0, 2))
    for n in sorted(strata):
        names = [strata[n][i] for i in rs.permutation(len(strata[n]))]
        for j, name in enumerate(names):
            ((k_sel if (j + flip) % 2 == 0 else k_con)).append(name)
        flip = (flip + len(names)) % 2
    w2_only = [r[0] for r in SI.COUNTRIES if r[0] not in by_country]
    perm = rs.permutation(len(w2_only))
    w2o_sel = [w2_only[i] for i in sorted(perm[: len(w2_only) // 2 + len(w2_only) % 2])]
    w2o_con = [w2_only[i] for i in sorted(perm[len(w2_only) // 2 + len(w2_only) % 2:])]

    def w2_list(inter_names, extra):
        pool = [n for n in inter_names if crow[n][4]] + extra
        if len(pool) < 20:
            raise BuildError(f"only {len(pool)} W2-capable countries in a split")
        pick = rs.choice(len(pool), size=20, replace=False)
        return [pool[i] for i in sorted(pick)]

    def country_rows(names, prefix):
        return [{"id": f"{prefix}_{n.replace(' ', '_')}", "name": n, "capital": crow[n][1], "language": crow[n][2],
                 "currency": crow[n][3]} for n in names]

    w2_sel, w2_con = w2_list(k_sel, w2o_sel), w2_list(k_con, w2o_con)

    # bridges per split: 25 country + 6 animal + 6 character
    def pick_bridges(inter_names):
        cb = [b for n in inter_names for b in by_country[n]]
        if len(cb) < 25:
            raise BuildError(f"only {len(cb)} country bridges in a split")
        sel = rs.choice(len(cb), size=25, replace=False)
        return [cb[i] for i in sorted(sel)]
    an, ch = SI.ANIMAL_BRIDGES, SI.CHARACTER_BRIDGES
    pa, pc = rs.permutation(len(an)), rs.permutation(len(ch))
    b_sel = pick_bridges(k_sel) + [an[i] for i in sorted(pa[:6])] + [ch[i] for i in sorted(pc[:6])]
    b_con = pick_bridges(k_con) + [an[i] for i in sorted(pa[6:])] + [ch[i] for i in sorted(pc[6:])]
    unused = sorted({b[0] for b in SI.COUNTRY_BRIDGES} - {b[0] for b in b_sel + b_con})

    def foil_pool(bridges, inter_names, w2_names):
        pool = {"country": sorted(set(inter_names) | set(w2_names)),
                "animal": [b[5] for b in bridges if b[1] == "animal"],
                "character": [b[5] for b in bridges if b[1] == "character"]}
        return pool
    th_sel = build_two_hop(toks, b_sel, foil_pool(b_sel, k_sel, w2_sel), rd, "s2")
    th_con = build_two_hop(toks, b_con, foil_pool(b_con, k_con, w2_con), rd, "c2")
    th_smk = build_two_hop(toks, SI.SMOKE_BRIDGES, {"animal": SI.SMOKE_FOILS["animal"],
                                                     "character": SI.SMOKE_FOILS["character"]}, rd, "sm2")

    # W3' design: 8 distinct categories, one concept each
    def w3p_concepts(rows):
        cats = sorted({r["category"] for r in rows})
        chosen = [cats[i] for i in sorted(rd.choice(len(cats), size=C.W3P_N_CONCEPTS, replace=False))]
        out = []
        for cat in chosen:
            opts = [r for r in rows if r["category"] == cat]
            out.append(opts[int(rd.integers(0, len(opts)))])
        return out
    cidx = rs.permutation(len(SC.CONTEXTS))
    x_sel = [SC.CONTEXTS[i] for i in sorted(cidx[:48])]
    x_con = [SC.CONTEXTS[i] for i in sorted(cidx[48:96])]
    w3_sel_c, w3_con_c = w3p_concepts(c_sel), w3p_concepts(c_con)
    wsel = {c["id"]: c["word"] for c in c_sel}
    wcon = {c["id"]: c["word"] for c in c_con}
    wsmk = {c["id"]: c["word"] for c in c_smk}
    w3_sel = w3p_design(rd, toks, w3_sel_c, x_sel, c_sel, C.W3P_FOLD_SIZE, wsel)
    w3_con = w3p_design(rd, toks, w3_con_c, x_con, c_con, C.W3P_FOLD_SIZE, wcon)
    w3_smk = w3p_design(rd, toks, c_smk, SC.SMOKE_CONTEXTS, c_smk, len(SC.SMOKE_CONTEXTS) // 2, wsmk)
    for d, pfx in ((w3_sel, "xs"), (w3_con, "xc"), (w3_smk, "xm")):
        ren = {r["id"]: f"{pfx}{r['id'][1:]}" for r in d["contexts"]}
        for r in d["contexts"]:
            r["id"] = ren[r["id"]]
        d["fold_A"] = [ren[i] for i in d["fold_A"]]
        d["fold_B"] = [ren[i] for i in d["fold_B"]]

    common = {"report_suffix": SI.REPORT_SUFFIX, "country_queries": SI.COUNTRY_QUERIES}
    out = {}
    for name, ps, cs, ks, ts, ctpl, ktpl, w3 in (
            ("select2", p_sel, c_sel, country_rows(w2_sel, "k2s"), th_sel, SI.CONCEPT_TEMPLATES_SELECT,
             SI.COUNTRY_TEMPLATES_SELECT, w3_sel),
            ("confirm2", p_con, c_con, country_rows(w2_con, "k2c"), th_con, SI.CONCEPT_TEMPLATES_CONFIRM,
             SI.COUNTRY_TEMPLATES_CONFIRM, w3_con)):
        d = designs(rd, ps, cs, ks, 5, 8, w3["concepts"][:4])
        out[name] = {"split": name, "paragraphs": ps, "concepts": cs, "concept_templates": ctpl, "countries": ks,
                     "country_templates": ktpl, "two_hop": ts, "w3p": w3, **common, **d}
    smk_k = [{"id": f"k2m_{r[0]}", "name": r[0], "capital": r[1], "language": r[2], "currency": r[3]}
             for r in SI.SMOKE_COUNTRIES]
    sd = designs(rd, p_smk, c_smk, smk_k, 2, 2, w3_smk["concepts"][:4])
    sd["w1_trials"] = sd["w1_trials"][:2]
    out["smoke2"] = {"split": "smoke2", "non_evidential": True, "paragraphs": p_smk, "concepts": c_smk,
                     "concept_templates": SI.CONCEPT_TEMPLATES_SMOKE, "countries": smk_k,
                     "country_templates": SI.COUNTRY_TEMPLATES_SMOKE, "two_hop": th_smk, "w3p": w3_smk,
                     **common, **sd}

    # ---------------------------------------------------------------- T3: pairwise disjointness
    def entities(s):
        e = {c["word"] for c in s["concepts"]} | {k["name"] for k in s["countries"]}
        for t in s["two_hop"]:
            e |= {t["bridge"], t["e1"], t["intermediate_entity"], *t["foils"]}
        return e

    def texts(s):
        t = {p["text"] for p in s["paragraphs"]} | {x["prompt"] for x in s["two_hop"]}
        t |= {r["prefix"] + "|" + r["cont"] for r in s["w3p"]["contexts"]}
        t |= {p for ps in s["w3p"]["paraphrases"].values() for p in ps}
        return t
    names = list(out)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = out[names[i]], out[names[j]]
            if texts(a) & texts(b):
                raise BuildError(f"T3 text overlap {names[i]} / {names[j]}")
            for x in entities(a):
                for y in entities(b):
                    if x.lower() == y.lower() or contains_words(x, y):
                        raise BuildError(f"T3 entity overlap {names[i]}:{x!r} / {names[j]}:{y!r}")

    # ---------------------------------------------------------------- T2: no overlap with A-stage material
    for s in out.values():
        for x in entities(s):
            for a in a_ent:
                if x.lower() == a or contains_words(x, a):
                    raise BuildError(f"T2 entity overlap with A-stage: {x!r} ~ {a!r}")
        free_text = {p["text"] for p in s["paragraphs"]} | set(s["concept_templates"]) | set(s["country_templates"])
        free_text |= {r["prefix"] for r in s["w3p"]["contexts"]} | {r["cont"] for r in s["w3p"]["contexts"]}
        free_text |= {p for ps in s["w3p"]["paraphrases"].values() for p in ps}
        for piece in free_text:      # two-hop prompts share the unchanged query frames by design; entity-checked above
            if piece in a_texts or ngrams(piece) & a_grams:
                raise BuildError(f"T2 text overlap with A-stage: {piece[:60]!r}")

    # ---------------------------------------------------------------- vocabulary audit + write
    files = {}
    for name in names:
        txt = json.dumps(out[name], ensure_ascii=False).lower()
        for bad in BANNED:
            if bad in txt:
                raise BuildError(f"banned vocabulary {bad!r} in {name}")
        fn = {"select2": "g_select2.json", "confirm2": "g_confirm2.json", "smoke2": "smoke2.json"}[name]
        p = os.path.join(MAT2, fn)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out[name], f, indent=1, ensure_ascii=False)
        files[fn] = sha256_file(p)
    src = {fn: sha256_file(os.path.join(MAT2, fn))
           for fn in ("source_items2.py", "source_paragraphs2.py", "source_contexts2.py")}
    src["tools/make_splits2.py"] = sha256_file(os.path.abspath(__file__))
    man = {"seeds": {"splits": C.SEED_SPLITS, "designs": C.SEED_DESIGNS, "rng": C.SEED_RNG},
           "files": files, "sources": src,
           "unused_country_bridges": unused,
           "counts": {k: {"paragraphs": len(v["paragraphs"]), "concepts": len(v["concepts"]),
                          "countries": len(v["countries"]), "two_hop": len(v["two_hop"]),
                          "w3p_concepts": len(v["w3p"]["concepts"]), "w3p_contexts": len(v["w3p"]["contexts"])}
                      for k, v in out.items()},
           "tokenizer_sha256": {k: sha256_file(os.path.join(model_dir(k), "tokenizer.json")) for k in toks},
           "checks": ["lengths", "single_token", "paraphrase_tokens", "context_tokenization", "two_hop_spans",
                      "foils", "T2_astage_disjoint", "T3_split_disjoint", "vocabulary"]}
    with open(os.path.join(MAT2, "split_manifest2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man["counts"]), "unused bridges:", unused)


if __name__ == "__main__":
    main()
