"""W0-W5 task-independent workspace assays (A-stage). See experiments/c15/astage_protocol.md.

One `AStage` object runs every assay for one model on one material split. On G_select it builds S_J, the
dictionary, the dose and the iso-impact scale; on G_confirm it receives them frozen and never re-fits.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch

from . import config as C
import copy

from .hooks import JAblation, RandomDisplacement, add_at, forward, logits_at, prefix_cache, unit_directions
from .matching import LayerMatcher, kl_matrix_final
from .stats import boot_ratio_ci, fold_ids, interp_in_log_scale, iso_scale, kernel, ridge_cv_r2_kernel
from .workspace import Dictionaries, band_layers, build_sj, vgen_ids


def _batches(n, bs):
    for i in range(0, n, bs):
        yield list(range(i, min(n, i + bs)))


def _by_length(seqs):
    groups = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[len(s)].append(i)
    return groups


def _unit(v):
    return v / v.norm().clamp_min(1e-12)


class AStage:
    def __init__(self, lm, lens, mats, *, frozen=None, layers=None, log=print, bs=8):
        self.lm, self.lens, self.m, self.log, self.bs = lm, lens, mats, log, bs
        self.L, self.d = lm.n_layers, lm.d_model
        self.frozen = frozen
        self.band_all = band_layers(self.L)
        self.layers = layers if layers is not None else list(self.band_all)
        self.timings = {}
        self.enc = lambda s: lm.tok(s, add_special_tokens=True).input_ids
        self.tok1 = lambda w: lm.tok(" " + w, add_special_tokens=False).input_ids[0]
        self.paras = {p["id"]: self.enc(p["text"])[: C.SEQ_LEN] for p in mats["paragraphs"]}
        self.para_order = [p["id"] for p in mats["paragraphs"]]
        self.concepts = {c["id"]: c["word"] for c in mats["concepts"]}
        self.countries = {k["id"]: k for k in mats["countries"]}
        self.w_eff = lm.w_eff()
        self.G_W = self.w_eff.T @ self.w_eff
        self.cache = {}
        self.boot_w3, self.boot_w5 = C.W3_BOOT, C.W5_BOOT

    # ------------------------------------------------------------------ basics
    def _t(self, name, t0):
        self.timings[name] = self.timings.get(name, 0.0) + time.time() - t0

    def run(self, seqs, edits_fn=None, record=(), pos_logits=None):
        """Run equal-length sequences (list of id lists) in batches. edits_fn(batch_idx) -> edits dict.
        Returns (list of final normed hidden [T, d] per seq, {layer: list of [T, d]})."""
        hid, rec = [None] * len(seqs), defaultdict(lambda: [None] * len(seqs))
        for b in _batches(len(seqs), self.bs):
            ids = torch.tensor([seqs[i] for i in b])
            edits = edits_fn(b) if edits_fn else None
            h, r = forward(self.lm, ids, edits=edits, record=record)
            for j, i in enumerate(b):
                hid[i] = h[j]
                for l in r:
                    rec[l][i] = r[l][j]
        return hid, rec

    def logits(self, h_rows):
        return self.lm.lm_head(h_rows).float()[..., : self.lm.n_vocab]

    # ------------------------------------------------------------------ workspace prep
    def prepare(self):
        t0 = time.time()
        sel_ids = [self.paras[p] for p in self.para_order]
        if self.frozen is None:
            self.vgen, self.vgen_dropped = vgen_ids(self.lm.tok, self.lm.n_vocab, sel_ids)
        else:
            self.vgen, self.vgen_dropped = self.frozen["vgen"], self.frozen.get("vgen_dropped", [])
        self.dicts = Dictionaries(self.lens, self.w_eff[torch.tensor(self.vgen)])
        # generic states (positions >= STAT_POS_MIN) at the needed layers + final block for the KL matrix
        need = sorted(set(self.layers) | {self.L - 1, self.L - 2})
        hid, rec = self.run(sel_ids, record=need)
        self.clean_text_hidden = hid
        self.clean_text_rec = rec
        sl = slice(C.STAT_POS_MIN, C.SEQ_LEN)
        self.sj, self.sj_info, self.hbar = {}, {}, {}
        for l in self.layers:
            S = torch.cat([r[sl] for r in rec[l]]).float()
            if self.frozen is None:
                Q, info = build_sj(S, self.dicts(l), self.d, screen=C.gp_screen(self.lm.key))
                info["gp_screen"] = C.gp_screen(self.lm.key)
                self.sj[l], self.sj_info[l] = Q, info
                self.hbar[l] = float(S.norm(dim=1).median())
            else:
                self.sj[l], self.sj_info[l] = self.frozen["Q"][l], self.frozen["sj_info"][l]
                self.hbar[l] = self.frozen["hbar"][l]
        if self.frozen is None:
            fin = torch.cat([r[sl] for r in rec[self.L - 1]])
            g = torch.Generator().manual_seed(C.SEED)
            pick = torch.randperm(fin.shape[0], generator=g)[: C.MATCH_KL_NPOS]
            lg = self.logits(torch.cat([h[sl] for h in hid])[pick])
            self.C_K = kl_matrix_final(self.w_eff, fin[pick], lg)
        else:
            self.C_K = self.frozen["C_K"]
        self.matchers = {l: LayerMatcher(self.lens.J(l), self.G_W, self.C_K, self.sj[l], f"{self.lm.key}|{l}")
                         for l in self.layers}
        self._t("prepare", t0)

    def directions(self, l, content, key, norm=None):
        """(u_J, P_J c, u_perp, info) for a content vector at layer l (u_perp None if unmatched); cached.

        norm=None: proxy matching only (used by the dose rule). norm=float or "natural" (= |P_J c|): the perp
        control is additionally calibrated so that its EMPIRICAL generic KL (single-position injection at that
        norm, G-split calibration paragraphs) is within MATCH_EMP_TOL of u_J's; otherwise it is unmatched."""
        ck = ("dir", l, key, norm)
        if ck in self.cache:
            return self.cache[ck]
        Q = self.sj[l]
        pj = Q @ (Q.T @ content)
        uJ = _unit(pj)
        up, info = self.matchers[l].match(content, uJ, f"{self.lm.key}|{l}|{key}")
        info["pj_frac"] = float(pj.norm() / content.norm().clamp_min(1e-12))
        if norm is not None and up is not None:
            nv = float(pj.norm()) if norm == "natural" else float(norm)
            eJ = self.emp_kl(l, uJ, nv)
            pool = []
            for sc in C.MATCH_EMP_SCALES:
                for rs in range(C.MATCH_EMP_SEEDS):
                    u2, i2 = self.matchers[l].match(content, uJ, f"{self.lm.key}|{l}|{key}|{sc}|{rs}", kl_scale=sc,
                                                    rand_seed=rs)
                    if u2 is not None and all(float(u2 @ q[1]) < 0.999 for q in pool):
                        pool.append((i2["cos_to_own_perp"], u2, i2))
            pool.sort(key=lambda t: -t[0])
            chosen, tried = None, []
            for cos, u2, i2 in pool:
                eP = self.emp_kl(l, u2, nv)
                ratio = eP / max(eJ, 1e-12)
                tried.append((round(cos, 4), round(ratio, 4)))
                if max(eJ, eP) < C.MATCH_EMP_FLOOR or abs(math.log(max(ratio, 1e-12))) <= math.log(C.MATCH_EMP_TOL):
                    chosen = (u2, dict(i2, emp_kl_perp=eP))
                    break
            info = dict(chosen[1] if chosen else info, pj_frac=info["pj_frac"], emp_kl_J=eJ, norm_used=nv,
                        emp_pool_tried=tried, emp_kl_matched=chosen is not None, pool_size=len(pool))
            up = chosen[0] if chosen else None
            if chosen is None:
                info["feasible"] = False
                info["reason"] = "no pool candidate matched the empirical generic KL within tolerance"
        self.cache[ck] = (uJ, pj, up, info)
        return self.cache[ck]

    def _cal(self):
        """Calibration paragraphs: cached prefix (positions < DOSE_POS) and clean log-probs on the subsequent window."""
        if "cal" not in self.cache:
            ids = torch.tensor([self.paras[p][: C.DOSE_POS + C.DOSE_WIN + 1]
                                for p in self.m["dose"]["paragraphs"][: C.MATCH_EMP_NSEQ]])
            past = prefix_cache(self.lm, ids[:, : C.DOSE_POS])
            h, _ = forward(self.lm, ids[:, C.DOSE_POS:], past=copy.deepcopy(past))
            lp0 = torch.log_softmax(self.logits(h[:, 1:]), -1)
            self.cache["cal"] = (ids, past, lp0)
        return self.cache["cal"]

    def emp_kl(self, l, u, norm):
        """Mean next-token KL(clean || injected) on the subsequent positions DOSE_POS+1..DOSE_POS+DOSE_WIN after a
        single-position injection of norm*u at DOSE_POS (G-split calibration paragraphs; prefix cache reused)."""
        ids, past, lp0 = self._cal()
        V = (norm * u)[None].repeat(ids.shape[0], 1)
        h, _ = forward(self.lm, ids[:, C.DOSE_POS:], edits={l: add_at(0, V)}, past=copy.deepcopy(past))
        lp = torch.log_softmax(self.logits(h[:, 1:]), -1)
        return float((lp0.exp() * (lp0 - lp)).sum(-1).mean())

    # ------------------------------------------------------------------ content vectors
    def _template_pos(self, prefix, full):
        pi, fi = self.enc(prefix), self.enc(full)
        if fi[: len(pi)] != pi:
            return None, fi
        return len(pi) - 1, fi

    def concept_vectors(self):
        t0 = time.time()
        seqs, meta = [], []
        for cid, w in self.concepts.items():
            for ti, tpl in enumerate(self.m["concept_templates"]):
                pre = tpl.split("{w}")[0] + " " + w
                pos, ids = self._template_pos(pre, tpl.replace("{w}", " " + w))
                if pos is not None:
                    seqs.append(ids)
                    meta.append((cid, pos))
        vecs = self._vectors(seqs, meta)
        self._t("concept_vectors", t0)
        return vecs

    def country_vectors(self):
        t0 = time.time()
        seqs, meta = [], []
        for kid, k in self.countries.items():
            for tpl in self.m["country_templates"]:
                ids = self.enc(tpl.replace("{X}", k["name"]))
                seqs.append(ids)
                meta.append((kid, len(ids) - 1))
        vecs = self._vectors(seqs, meta)
        self._t("country_vectors", t0)
        return vecs

    def _vectors(self, seqs, meta):
        acc = {l: defaultdict(list) for l in self.layers}
        for T, idx in _by_length(seqs).items():
            _, rec = self.run([seqs[i] for i in idx], record=self.layers)
            for j, i in enumerate(idx):
                key, pos = meta[i]
                for l in self.layers:
                    acc[l][key].append(rec[l][j][pos].float())
        out = {}
        for l in self.layers:
            means = {k: torch.stack(v).mean(0) for k, v in acc[l].items()}
            grand = torch.stack(list(means.values())).mean(0)
            out[l] = {k: v - grand for k, v in means.items()}
        return out

    # ------------------------------------------------------------------ W0
    def w0(self):
        t0 = time.time()
        sl = slice(C.STAT_POS_MIN, C.SEQ_LEN)
        agree, n = 0, 0
        for b in _batches(len(self.para_order), self.bs):
            h_l = torch.stack([self.clean_text_rec[self.L - 2][i][sl] for i in b]).float()
            lens_top = self.lens.readout(self.lm, h_l, self.L - 2)[..., : self.lm.n_vocab].argmax(-1)
            mod_top = self.logits(torch.stack([self.clean_text_hidden[i][sl] for i in b])).argmax(-1)
            agree += int((lens_top == mod_top).sum())
            n += lens_top.numel()
        w0a = agree / max(n, 1)
        th = self.two_hop_clean()
        correct = [i for i, ok in enumerate(th["correct"]) if ok]
        hits = 0
        for i in correct:
            item = self.m["two_hop"][i]
            alias = {self.tok1(a) for a in item["intermediates"]}
            hit = False
            for l in self.band_all:
                lg = self.lens.readout(self.lm, th["last_states"][l][i][None], l)[0, : self.lm.n_vocab]
                if alias & set(lg.topk(C.W0B_TOPK).indices.tolist()):
                    hit = True
                    break
            hits += hit
        acc = th["acc"]
        inter = hits / max(len(correct), 1)
        res = {"w0a_lens_agreement_L-2": w0a, "two_hop_acc": acc, "two_hop_n": len(th["correct"]),
               "intermediate_rate_among_correct": inter, "n_correct": len(correct),
               "pass_w0a": w0a >= C.W0A_MIN_AGREE, "pass_intermediate": inter >= C.W0B_MIN_INTERMEDIATE,
               "w4_assessable": acc >= C.W0B_MIN_ACC}
        res["pass"] = bool(res["pass_w0a"] and res["pass_intermediate"])
        self._t("w0", t0)
        return res

    def two_hop_clean(self):
        if "two_hop" in self.cache:
            return self.cache["two_hop"]
        items = self.m["two_hop"]
        seqs = [self.enc(it["prompt"]) for it in items]
        ans = [self.tok1(it["answer"]) for it in items]
        correct = [False] * len(items)
        last = {l: [None] * len(items) for l in self.band_all}
        for T, idx in _by_length(seqs).items():
            hid, rec = self.run([seqs[i] for i in idx], record=self.band_all)
            for j, i in enumerate(idx):
                correct[i] = int(self.logits(hid[j][T - 1]).argmax()) == ans[i]
                for l in self.band_all:
                    last[l][i] = rec[l][j][T - 1].float()
        out = {"seqs": seqs, "ans": ans, "correct": correct, "acc": float(np.mean(correct)), "last_states": last}
        self.cache["two_hop"] = out
        return out

    # ------------------------------------------------------------------ dose rule (G_select only)
    def dose_rule(self, l, cvec):
        """a* = largest grid dose at which a single-position J-direction injection keeps mean next-token KL <=
        DOSE_MAX_KL and top-1 agreement >= DOSE_MIN_AGREE over the subsequent positions DOSE_POS+1..DOSE_POS+DOSE_WIN
        (G-split dose paragraphs; J arm only, memo D69). Also reports the propagation norm at a later layer."""
        t0 = time.time()
        ids = [self.paras[p][: C.DOSE_LEN] for p in self.m["dose"]["paragraphs"]]
        conc = self.m["dose"]["concepts"][: C.DOSE_N_DIR]
        dirs = [_unit(self.sj[l] @ (self.sj[l].T @ cvec[l][c])) for c in conc]
        P, W = C.DOSE_POS, C.DOSE_WIN
        prop_l = min(l + math.ceil(0.1 * self.L), self.L - 1)
        hid0, rec0 = self.run(ids, record=[prop_l])
        lp0 = [torch.log_softmax(self.logits(h[P + 1: P + W + 1]), -1) for h in hid0]
        robust = []
        for x in lp0:
            t2 = x.topk(2, dim=-1).values
            robust.append((t2[:, 0] - t2[:, 1]) >= C.DOSE_ROBUST_MARGIN)
        table = []
        for a in C.DOSE_GRID:
            ag, kl, pr = [], [], []
            for u in dirs:
                vec = (a * self.hbar[l] * u)[None].repeat(len(ids), 1)
                hid, rec = self.run(ids, edits_fn=lambda b, v=vec: {l: add_at(P, v[b])}, record=[prop_l])
                for s_ in range(len(ids)):
                    lp = torch.log_softmax(self.logits(hid[s_][P + 1: P + W + 1]), -1)
                    rb = robust[s_]
                    ag.append(float((lp.argmax(-1) == lp0[s_].argmax(-1))[rb].float().mean()) if rb.any() else 1.0)
                    kl.append(float((lp0[s_].exp() * (lp0[s_] - lp)).sum(-1).mean()))
                    pr.append(float((rec[prop_l][s_][P] - rec0[prop_l][s_][P]).norm()))
            table.append({"a": a, "agree": float(np.mean(ag)), "kl": float(np.mean(kl)), "prop_norm": float(np.mean(pr))})
        ok = []
        for r in table:                     # monotone admissibility: every grid dose up to a* must satisfy the rule
            if r["agree"] >= C.DOSE_MIN_AGREE and r["kl"] <= C.DOSE_MAX_KL:
                ok.append(r["a"])
            else:
                break
        a_star = max(ok) if ok else C.DOSE_GRID[0]
        self._t("dose", t0)
        return a_star, {"table": table, "a_star": a_star, "no_grid_dose_meets_rule": not ok, "prop_layer": prop_l}

    # ------------------------------------------------------------------ W1 reportability
    def w1(self, l, cvec, a):
        t0 = time.time()
        suffix = self.enc(self.m["report_suffix"])
        P = C.W1_CONTEXT_LEN - 1
        trials, dirs, unmatched = [], {}, []
        for tr in self.m["w1_trials"]:
            c = tr["concept"]
            if c not in dirs:
                uJ, _, up, info = self.directions(l, cvec[l][c], c, norm=a * self.hbar[l])
                dirs[c] = (uJ, up, info)
                if up is None:
                    unmatched.append(c)
            for ctx in tr["contexts"]:
                trials.append((c, self.paras[ctx][: C.W1_CONTEXT_LEN] + suffix,
                               [self.tok1(self.concepts[o]) for o in tr["options"]],
                               tr["options"].index(c)))
        trials = [t for t in trials if dirs[t[0]][1] is not None]
        seqs = [t[1] for t in trials]
        res = {"none": None, "J": None, "perp": None}
        for arm in (("none", "J", "perp") if trials else ()):
            if arm == "none":
                last = [h[-1] for h in self.run(seqs)[0]]
            else:
                k = 0 if arm == "J" else 1
                V = torch.stack([a * self.hbar[l] * dirs[t[0]][k] for t in trials])
                hid, _ = self.run(seqs, edits_fn=lambda b, V=V: {l: add_at(P, V[b])})
                last = [h[-1] for h in hid]
            hits = []
            for t, h in zip(trials, last):
                lg = self.logits(h)[torch.tensor(t[2])]
                hits.append(int(lg.argmax()) == t[3])
            res[arm] = float(np.mean(hits)) if hits else None
        n_conc = len(dirs)
        frac_unmatched = len(unmatched) / max(n_conc, 1)
        J, Pp = res["J"] or 0.0, res["perp"] or 0.0
        ratio = J / Pp if Pp > 0 else math.inf
        out = {"hit_J": J, "hit_perp": Pp, "hit_none": res["none"], "n_trials": len(trials),
               "frac_unmatched": frac_unmatched, "ratio": ratio,
               "matching": {c: {k: v for k, v in d[2].items()} for c, d in dirs.items()}}
        out["pass"] = bool(frac_unmatched <= C.MATCH_MAX_INFEASIBLE and J >= C.W1_MIN_J and ratio >= C.W1_MIN_RATIO)
        self._t("w1", t0)
        return out, {c: (d[0], d[1]) for c, d in dirs.items()}

    # ------------------------------------------------------------------ W2 cross-function broadcast
    def w2(self, l, kvec):
        t0 = time.time()
        q = self.m["country_queries"]
        rows = []
        unmatched = 0
        for pr in self.m["w2_pairs"]:
            A, B = self.countries[pr["A"]], self.countries[pr["B"]]
            content = kvec[l][pr["B"]] - kvec[l][pr["A"]]
            uJ, pj, up, info = self.directions(l, content, pr["A"] + ">" + pr["B"], norm="natural")
            if up is None:
                unmatched += 1
                continue
            for f, tpl in q.items():
                pre = tpl.split("{X}")[0] + A["name"]
                pos, ids = self._template_pos(pre, tpl.replace("{X}", A["name"]))
                if pos is None:
                    continue
                ta, tb = self.tok1(A[f]), self.tok1(B[f])
                rows.append({"pair": pr["A"] + ">" + pr["B"], "f": f, "ids": ids, "pos": pos, "ta": ta, "tb": tb,
                             "dJ": pj, "dP": pj.norm() * up})
        res = {}
        for arm in ("none", "J", "perp"):
            out = [None] * len(rows)
            for T, idx in _by_length([r["ids"] for r in rows]).items():
                seqs = [rows[i]["ids"] for i in idx]
                if arm == "none":
                    ef = None
                else:
                    V = torch.stack([rows[i]["dJ" if arm == "J" else "dP"] for i in idx])
                    pos = torch.tensor([rows[i]["pos"] for i in idx])
                    ef = lambda b, V=V, pos=pos: {l: add_at(pos[b], V[b])}
                hid, _ = self.run(seqs, edits_fn=ef)
                for j, i in enumerate(idx):
                    lp = torch.log_softmax(self.logits(hid[j][T - 1]), -1)
                    out[i] = float(lp[rows[i]["tb"]] - lp[rows[i]["ta"]])
            res[arm] = out
        per_pair = defaultdict(lambda: {"elig": 0, "J": 0, "perp": 0})
        for i, r in enumerate(rows):
            if r["ta"] == r["tb"] or res["none"][i] >= 0:
                continue
            pp = per_pair[r["pair"]]
            pp["elig"] += 1
            pp["J"] += res["J"][i] > 0
            pp["perp"] += res["perp"][i] > 0
        incl = [p for p in per_pair.values() if p["elig"] >= 2]
        sJ = float(np.mean([p["J"] >= C.W2_MIN_SWITCH for p in incl])) if incl else 0.0
        sP = float(np.mean([p["perp"] >= C.W2_MIN_SWITCH for p in incl])) if incl else 0.0
        frac_unmatched = unmatched / max(len(self.m["w2_pairs"]), 1)
        ratio = sJ / sP if sP > 0 else math.inf
        out = {"rate_J": sJ, "rate_perp": sP, "n_pairs_included": len(incl), "frac_unmatched": frac_unmatched,
               "ratio": ratio}
        out["pass"] = bool(frac_unmatched <= C.MATCH_MAX_INFEASIBLE and len(incl) > 0 and sJ >= C.W2_MIN_J
                           and ratio >= C.W2_MIN_RATIO)
        self._t("w2", t0)
        return out

    # ------------------------------------------------------------------ W3 cross-position transport
    def w3(self, l, dirs_w1, cvec, a):
        t0 = time.time()
        d3 = self.m["w3"]
        P = C.W3_CONTEXT_LEN - 1
        ctx = [self.paras[c][: C.W3_CONTEXT_LEN + C.W3_CONT_LEN] for c in d3["contexts"]]
        s = torch.tensor(d3["scalars"])
        layers = list(range(l + C.W3_LAYER_STEP, self.L - 1, C.W3_LAYER_STEP))
        positions = list(range(P + 1, P + 1 + C.W3_CONT_LEN, C.W3_POS_STEP))
        folds = fold_ids(len(ctx))
        r2 = {"J": [], "perp": []}
        Xs = {"J": [], "perp": []}
        for c in d3["concepts"]:
            if c in dirs_w1:
                uJ, up = dirs_w1[c]
            else:
                uJ, _, up, _ = self.directions(l, cvec[l][c], c, norm=a * self.hbar[l])
            if up is None:
                continue
            for arm, u in (("J", uJ), ("perp", up)):
                V = (s[:, None] * a * self.hbar[l]) * u[None]
                _, rec = self.run(ctx, edits_fn=lambda b, V=V: {l: add_at(P, V[b])}, record=layers)
                K = {(L_, p): kernel(torch.stack([rec[L_][i][p] for i in range(len(ctx))]).float())
                     for L_ in layers for p in positions}
                Xs[arm].append(K)
                r2[arm].append({k: ridge_cv_r2_kernel(v, s, folds) for k, v in K.items()})

        def bb(r2list):
            return float(np.mean([np.mean([v >= C.W3_R2_SITE for v in r.values()]) for r in r2list])) if r2list else 0.0
        BJ, BP = bb(r2["J"]), bb(r2["perp"])
        rng = np.random.default_rng(C.SEED + l)
        diffs = []
        n = len(ctx)
        for _ in range(self.boot_w3):
            i = torch.tensor(rng.integers(0, n, n))
            fb = folds[i]
            vals = {}
            for arm in ("J", "perp"):
                vals[arm] = float(np.mean([np.mean([ridge_cv_r2_kernel(K[k][i][:, i], s[i], fb) >= C.W3_R2_SITE
                                                    for k in K]) for K in Xs[arm]])) if Xs[arm] else 0.0
            diffs.append(vals["J"] - vals["perp"])
        lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (0.0, 0.0)
        out = {"BB_J": BJ, "BB_perp": BP, "diff": BJ - BP, "ci": [float(lo), float(hi)],
               "n_concepts": len(r2["J"]), "n_sites": len(layers) * len(positions),
               "mean_r2_J": float(np.mean([np.mean(list(r.values())) for r in r2["J"]])) if r2["J"] else None,
               "mean_r2_perp": float(np.mean([np.mean(list(r.values())) for r in r2["perp"]])) if r2["perp"] else None}
        out["pass"] = bool(out["diff"] >= C.W3_MIN_DIFF and lo > 0)
        self._t("w3", t0)
        return out

    # ------------------------------------------------------------------ W4 / W5 ablation
    def _abl_sets(self):
        th = self.two_hop_clean()
        text = [self.paras[p] for p in self.para_order]
        return th, text

    def _clean_text_stats(self, text):
        if "text_clean" in self.cache:
            return self.cache["text_clean"]
        sl = slice(C.STAT_POS_MIN, C.SEQ_LEN)
        top = [self.logits(h[sl]).argmax(-1) for h in self.clean_text_hidden]
        self.cache["text_clean"] = top
        return top

    def _ablate_run(self, l, mode, scale=None, norms=None):
        """mode 'J' records removed norms; mode 'R' applies the random displacement with given norms."""
        th, text = self._abl_sets()
        win = [x for x in range(l - C.ABL_HALF_WINDOW, l + C.ABL_HALF_WINDOW + 1)]
        sets = {"two_hop": th["seqs"], "text": text}
        out = {"two_hop_correct": [None] * len(th["seqs"]), "text_change": [None] * len(text)}
        rec_norms = {"two_hop": {}, "text": {}}
        clean_top = self._clean_text_stats(text)
        sl = slice(C.STAT_POS_MIN, C.SEQ_LEN)
        for name, seqs in sets.items():
            for T, idx in _by_length(seqs).items():
                for b in _batches(len(idx), self.bs):
                    rows = [idx[j] for j in b]
                    ids = torch.tensor([seqs[i] for i in rows])
                    if mode == "J":
                        objs = {w: JAblation(self.dicts(w), C.GP_K, C.ABL_TOP, screen=C.gp_screen(self.lm.key)) for w in win}
                        edits = dict(objs)
                    else:
                        edits = {}
                        for w in win:
                            nrm = torch.stack([norms[name][(w, i)] for i in rows])
                            dirs = torch.stack([unit_directions(f"{self.lm.key}|rand|{w}|{name}|{i}", (T, self.d))
                                                for i in rows])
                            edits[w] = RandomDisplacement(nrm, dirs, scale)
                    h, _ = forward(self.lm, ids, edits=edits)
                    if mode == "J":
                        for w in win:
                            for j, i in enumerate(rows):
                                rec_norms[name][(w, i)] = objs[w].norms[j]
                    for j, i in enumerate(rows):
                        if name == "two_hop":
                            out["two_hop_correct"][i] = int(self.logits(h[j][T - 1]).argmax()) == th["ans"][i]
                        else:
                            top = self.logits(h[j][sl]).argmax(-1)
                            out["text_change"][i] = float((top != clean_top[i]).float().mean())
        return out, rec_norms

    def w45(self, l, s_star_frozen=None):
        t0 = time.time()
        th, _ = self._abl_sets()
        acc0 = th["acc"]
        J, norms = self._ablate_run(l, "J")
        impJ = 100.0 * (acc0 - float(np.mean(J["two_hop_correct"])))
        dmgJ = np.array(J["text_change"])
        res = {"acc_clean": acc0, "imp_J_pp": impJ, "dmg_J": float(dmgJ.mean())}
        if s_star_frozen is None:
            imps, dmgs = [], []
            for s in C.RAND_SCALE_GRID:
                R, _ = self._ablate_run(l, "R", scale=s, norms=norms)
                imps.append(100.0 * (acc0 - float(np.mean(R["two_hop_correct"]))))
                dmgs.append(R["text_change"])
            s_star, status = iso_scale(imps, C.RAND_SCALE_GRID, impJ)
            dmg_iso = interp_in_log_scale(dmgs, C.RAND_SCALE_GRID, s_star)
            res.update({"imp_rand_grid_pp": imps, "dmg_rand_grid": [float(np.mean(x)) for x in dmgs],
                        "imp_rand1_pp": imps[0], "dmg_rand1": float(np.mean(dmgs[0]))})
            dmg1 = np.array(dmgs[0])
        else:
            s_star, status = s_star_frozen["s_star"], s_star_frozen["status"]
            R1, _ = self._ablate_run(l, "R", scale=1.0, norms=norms)
            Rs, _ = self._ablate_run(l, "R", scale=s_star, norms=norms)
            dmg1 = np.array(R1["text_change"])
            dmg_iso = np.array(Rs["text_change"])
            res.update({"imp_rand1_pp": 100.0 * (acc0 - float(np.mean(R1["two_hop_correct"]))),
                        "dmg_rand1": float(dmg1.mean()),
                        "imp_rand_sstar_pp_confirm": 100.0 * (acc0 - float(np.mean(Rs["two_hop_correct"])))})
        si_norm = float(dmgJ.mean() / dmg1.mean()) if dmg1.mean() > 0 else math.inf
        si_iso = float(dmgJ.mean() / np.mean(dmg_iso)) if np.mean(dmg_iso) > 0 else math.inf
        ci = boot_ratio_ci(dmgJ, dmg_iso, self.boot_w5, C.SEED + l)
        w4_diff = impJ - res["imp_rand1_pp"]
        res.update({"s_star": s_star, "s_star_status": status, "SI_norm": si_norm, "SI_iso": si_iso,
                    "SI_iso_ci": list(ci), "w4_diff_pp": w4_diff})
        res["pass_w4"] = bool(acc0 >= C.W0B_MIN_ACC and w4_diff >= C.W4_MIN_DIFF_PP)
        certifiable = status in ("interpolated", "unreached_upper_bound")
        res["pass_w5"] = bool(certifiable and si_iso <= C.W5_MAX_SI_ISO and ci[1] <= C.W5_MAX_CI_UP)
        res["w5_description"] = (
            "selective at matched functional impact but anti-selective at matched intervention norm"
            if res["pass_w5"] and si_norm > 1.0 else
            ("selective at matched functional impact and at matched norm" if res["pass_w5"] else "not selective"))
        self._t("w45", t0)
        return res

    # ------------------------------------------------------------------ orchestration
    def run_all(self, frozen_layer_params=None):
        """Select: all band layers with fitted dose/s*. Confirm: frozen params for the single frozen layer."""
        self.prepare()
        out = {"model": self.lm.key, "band_all": self.band_all, "layers": self.layers,
               "vgen_size": len(self.vgen), "w0": self.w0(), "per_layer": {}}
        cvec = self.concept_vectors()
        kvec = self.country_vectors()
        for l in self.layers:
            self.log(f"[{self.lm.key}] layer {l}")
            pl = {"sj": self.sj_info[l], "hbar": self.hbar[l]}
            if frozen_layer_params is None:
                a, dose = self.dose_rule(l, cvec)
                pl["dose"] = dose
            else:
                a = frozen_layer_params["a_star"]
                pl["dose"] = {"a_star": a, "frozen": True}
            pl["w1"], dirs = self.w1(l, cvec, a)
            pl["w2"] = self.w2(l, kvec)
            pl["w3"] = self.w3(l, dirs, cvec, a)
            pl["w45"] = self.w45(l, None if frozen_layer_params is None else frozen_layer_params["s_star"])
            out["per_layer"][l] = pl
            self.log(f"[{self.lm.key}] layer {l} done: W1 {pl['w1']['pass']} W2 {pl['w2']['pass']} "
                     f"W3 {pl['w3']['pass']} W4 {pl['w45']['pass_w4']} W5 {pl['w45']['pass_w5']}")
        out["timings_s"] = self.timings
        return out
