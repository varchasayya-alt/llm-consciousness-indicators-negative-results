"""R2 assays for one model on one material split.

R2Stage subclasses the A-stage AStage (experiments/c15/c15a/assays.py, read-only): S_J, V_gen, dictionaries, the
matched-perp construction, the dose rule and concept/country vectors are inherited unchanged. W1, W2 and W4/W5 are
the A-stage method bodies copied verbatim with per-unit outcomes retained (for the D74 CIs); tests/test_r2.py
asserts that their gate outputs equal the A-stage methods' outputs. New: W0a CI, W0b' (primary t_d), W3', W3b.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch

from c15a import config as CA
from c15a.assays import AStage, _by_length
from c15a.hooks import add_at, unit_directions
from c15a.stats import boot_ratio_ci, interp_in_log_scale, iso_scale

from . import config as C
from .w3p import W3PAssay


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def boot_mean_ci(values, n_boot, seed, pct=C.CI_PCT):
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return [_pct(m, pct[0]), _pct(m, pct[1])]


def ratio(a, b):
    return a / b if b > 0 else math.inf


class R2Stage(AStage):
    def __init__(self, lm, lens, mats, **kw):
        super().__init__(lm, lens, mats, **kw)
        self.w3 = W3PAssay(lm, mats["w3p"], bs=self.bs, log=self.log)

    # ------------------------------------------------------------------ W0a + W0b'
    def positions(self, item):
        p = item["prompt"]
        full = self.enc(p)
        out = {}
        for name, end in (("t1", item["e1_span"][1]), ("td", item["desc_span"][1])):
            head = self.enc(p[:end])
            if full[:len(head)] != head:
                raise RuntimeError(f"span tokenization not prefix-stable: {item['id']}")
            out[name] = len(head) - 1
        out["t2"] = len(full) - 1
        return out

    def w0_r2(self):
        t0 = time.time()
        sl = slice(CA.STAT_POS_MIN, CA.SEQ_LEN)
        per_para = []
        for b in (list(range(i, min(len(self.para_order), i + self.bs))) for i in range(0, len(self.para_order), self.bs)):
            h_l = torch.stack([self.clean_text_rec[self.L - 2][i][sl] for i in b]).float()
            lens_top = self.lens.readout(self.lm, h_l, self.L - 2)[..., : self.lm.n_vocab].argmax(-1)
            mod_top = self.logits(torch.stack([self.clean_text_hidden[i][sl] for i in b])).argmax(-1)
            per_para += (lens_top == mod_top).float().mean(1).tolist()
        n_pos = CA.SEQ_LEN - CA.STAT_POS_MIN
        w0a = float(np.mean(per_para))           # equal positions per paragraph -> equals the pooled A-stage rate
        w0a_ci = boot_mean_ci(per_para, C.CI_BOOT, C.SEED_RNG)
        th = self.two_hop_clean()                # unchanged A-stage two-hop pass (accuracy, final states)
        items = self.m["two_hop"]
        correct = [i for i, ok in enumerate(th["correct"]) if ok]
        pos = [self.positions(it) for it in items]
        # band-layer states at t1, td, t2 for the correct items
        seqs = [th["seqs"][i] for i in correct]
        states = {p: {l: [None] * len(correct) for l in self.band_all} for p in C.W0B_POSITIONS}
        for T, idx in _by_length(seqs).items():
            _, rec = self.run([seqs[j] for j in idx], record=self.band_all)
            for jj, j in enumerate(idx):
                for p in C.W0B_POSITIONS:
                    for l in self.band_all:
                        states[p][l][j] = rec[l][jj][pos[correct[j]][p]].float()
        res = {"w0a_lens_agreement_L-2": w0a, "w0a_ci": w0a_ci, "n_paragraphs": len(per_para),
               "positions_per_paragraph": n_pos, "two_hop_acc": th["acc"], "two_hop_n": len(items),
               "n_correct": len(correct), "pass_w0a": bool(w0a >= C.W0A_MIN_AGREE)}
        res["w0a_powered_failure"] = bool(w0a_ci[1] < C.PF["W0a"]["ci_upper_lt"] and w0a <= C.PF["W0a"]["point_le"])
        by_pos = {}
        for p in C.W0B_POSITIONS:
            hit, foil = [], []
            for l in self.band_all:
                if not correct:
                    break
                H = torch.stack(states[p][l])
                top = self.lens.readout(self.lm, H, l)[:, : self.lm.n_vocab].topk(C.W0B_TOPK).indices
                for j in range(len(correct)):
                    s = set(top[j].tolist())
                    it = items[correct[j]]
                    a = {self.tok1(x) for x in it["intermediates"]} & s
                    f = [self.tok1(x) in s for x in it["foils"]]
                    if l == self.band_all[0]:
                        hit.append(bool(a))
                        foil.append(f)
                    else:
                        hit[j] = hit[j] or bool(a)
                        foil[j] = [u or v for u, v in zip(foil[j], f)]
            if not hit:                          # no correctly answered item: rates undefined -> gate fails
                by_pos[p] = {"rate": 0.0, "foil_rate": 0.0, "lift": 0.0, "lift_lb_one_sided_95": float("nan"),
                             "rate_ci": [float("nan")] * 2, "lift_ci": [float("nan")] * 2, "n_items": 0}
                continue
            hit = np.asarray(hit, float)
            foil = np.asarray(foil, float).reshape(len(hit), -1)
            lift_i = hit - foil.mean(1)
            rate, lift = float(hit.mean()), float(lift_i.mean())
            rng = np.random.default_rng(C.SEED_RNG)
            boots = lift_i[rng.integers(0, len(lift_i), (C.W0B_BOOT, len(lift_i)))].mean(1)
            by_pos[p] = {"rate": rate, "foil_rate": float(foil.mean()), "lift": lift,
                         "lift_lb_one_sided_95": _pct(boots, C.W0B_LIFT_LB_PCT),
                         "rate_ci": boot_mean_ci(hit, C.CI_BOOT, C.SEED_RNG + 1),
                         "lift_ci": [_pct(boots, C.CI_PCT[0]), _pct(boots, C.CI_PCT[1])], "n_items": len(hit)}
        res["w0b_by_position"] = by_pos
        prim = by_pos[C.W0B_PRIMARY]
        res["w0b_primary_position"] = C.W0B_PRIMARY
        res["pass_w0b"] = bool(th["acc"] >= C.W0B_MIN_ACC and prim["rate"] >= C.W0B_MIN_INTERMEDIATE
                               and prim["lift"] >= C.W0B_MIN_LIFT and prim["lift_lb_one_sided_95"] > 0)
        res["w0b_powered_failure"] = bool(th["acc"] >= C.W0B_MIN_ACC and (
            (prim["rate_ci"][1] < C.PF["W0b"]["intermediate_ci_upper_lt"] and prim["rate"] <= C.PF["W0b"]["intermediate_point_le"])
            or (prim["lift_ci"][1] < C.PF["W0b"]["or_lift_ci_upper_lt"] and prim["lift"] <= C.PF["W0b"]["or_lift_point_le"])))
        res["w4_assessable"] = bool(th["acc"] >= C.W4_REQ_ACC)
        res["pass"] = bool(res["pass_w0a"] and res["pass_w0b"])
        self._t("w0_r2", t0)
        return res

    # ------------------------------------------------------------------ W1 (A-stage body + per-trial outcomes)
    def w1(self, l, cvec, a):
        t0 = time.time()
        suffix = self.enc(self.m["report_suffix"])
        P = CA.W1_CONTEXT_LEN - 1
        trials, dirs, unmatched = [], {}, []
        for tr in self.m["w1_trials"]:
            c = tr["concept"]
            if c not in dirs:
                uJ, _, up, info = self.directions(l, cvec[l][c], c, norm=a * self.hbar[l])
                dirs[c] = (uJ, up, info)
                if up is None:
                    unmatched.append(c)
            for ctx in tr["contexts"]:
                trials.append((c, self.paras[ctx][: CA.W1_CONTEXT_LEN] + suffix,
                               [self.tok1(self.concepts[o]) for o in tr["options"]],
                               tr["options"].index(c)))
        trials = [t for t in trials if dirs[t[0]][1] is not None]
        seqs = [t[1] for t in trials]
        res = {"none": None, "J": None, "perp": None}
        per = {}
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
            per[arm] = hits
        n_conc = len(dirs)
        frac_unmatched = len(unmatched) / max(n_conc, 1)
        J, Pp = res["J"] or 0.0, res["perp"] or 0.0
        rt = J / Pp if Pp > 0 else math.inf
        out = {"hit_J": J, "hit_perp": Pp, "hit_none": res["none"], "n_trials": len(trials),
               "frac_unmatched": frac_unmatched, "ratio": rt,
               "matching": {c: {k: v for k, v in d[2].items()} for c, d in dirs.items()}}
        out["pass"] = bool(frac_unmatched <= CA.MATCH_MAX_INFEASIBLE and J >= CA.W1_MIN_J and rt >= CA.W1_MIN_RATIO)
        # ---- R2 additions: cluster bootstrap over concepts (D74), baseline-corrected rate (reported)
        if trials:
            conc = [t[0] for t in trials]
            cl = sorted(set(conc))
            byc = {c: [i for i, x in enumerate(conc) if x == c] for c in cl}
            hJ, hP = np.asarray(per["J"], float), np.asarray(per["perp"], float)
            rng = np.random.default_rng(C.SEED_RNG + l)
            bj, br = [], []
            for _ in range(C.CI_BOOT):
                pick = [byc[cl[i]] for i in rng.integers(0, len(cl), len(cl))]
                idx = [i for g in pick for i in g]
                j_, p_ = hJ[idx].mean(), hP[idx].mean()
                bj.append(j_)
                br.append(ratio(j_, p_))
            out["hit_J_ci"] = [_pct(bj, C.CI_PCT[0]), _pct(bj, C.CI_PCT[1])]
            out["ratio_ci"] = [_pct(br, C.CI_PCT[0]), _pct(br, C.CI_PCT[1])]
            out["hit_J_minus_none"] = J - (res["none"] or 0.0)
        else:
            out["hit_J_ci"] = out["ratio_ci"] = [float("nan"), float("nan")]
        self._t("w1", t0)
        return out, {c: (d[0], d[1]) for c, d in dirs.items()}

    # ------------------------------------------------------------------ W2 (A-stage body + per-pair outcomes)
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
                    ef = lambda b, V=V, pos=pos: {l: add_at(pos[b], V[b])}  # noqa: E731
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
        sJ = float(np.mean([p["J"] >= CA.W2_MIN_SWITCH for p in incl])) if incl else 0.0
        sP = float(np.mean([p["perp"] >= CA.W2_MIN_SWITCH for p in incl])) if incl else 0.0
        frac_unmatched = unmatched / max(len(self.m["w2_pairs"]), 1)
        rt = sJ / sP if sP > 0 else math.inf
        o = {"rate_J": sJ, "rate_perp": sP, "n_pairs_included": len(incl), "frac_unmatched": frac_unmatched,
             "ratio": rt}
        o["pass"] = bool(frac_unmatched <= CA.MATCH_MAX_INFEASIBLE and len(incl) > 0 and sJ >= CA.W2_MIN_J
                         and rt >= CA.W2_MIN_RATIO)
        # ---- R2 additions: bootstrap over included pairs (D74)
        if incl:
            sj = np.asarray([p["J"] >= CA.W2_MIN_SWITCH for p in incl], float)
            spp = np.asarray([p["perp"] >= CA.W2_MIN_SWITCH for p in incl], float)
            rng = np.random.default_rng(C.SEED_RNG + l)
            idx = rng.integers(0, len(incl), (C.CI_BOOT, len(incl)))
            bj, bp = sj[idx].mean(1), spp[idx].mean(1)
            br = [ratio(x, y) for x, y in zip(bj, bp)]
            o["rate_J_ci"] = [_pct(bj, C.CI_PCT[0]), _pct(bj, C.CI_PCT[1])]
            o["ratio_ci"] = [_pct(br, C.CI_PCT[0]), _pct(br, C.CI_PCT[1])]
        else:
            o["rate_J_ci"] = o["ratio_ci"] = [float("nan"), float("nan")]
        self._t("w2", t0)
        return o

    # ------------------------------------------------------------------ W4 / W5 (A-stage body + per-item outcomes)
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
            R1c = None
            for s in CA.RAND_SCALE_GRID:
                R, _ = self._ablate_run(l, "R", scale=s, norms=norms)
                if R1c is None:
                    R1c = R["two_hop_correct"]
                imps.append(100.0 * (acc0 - float(np.mean(R["two_hop_correct"]))))
                dmgs.append(R["text_change"])
            s_star, status = iso_scale(imps, CA.RAND_SCALE_GRID, impJ)
            dmg_iso = interp_in_log_scale(dmgs, CA.RAND_SCALE_GRID, s_star)
            res.update({"imp_rand_grid_pp": imps, "dmg_rand_grid": [float(np.mean(x)) for x in dmgs],
                        "imp_rand1_pp": imps[0], "dmg_rand1": float(np.mean(dmgs[0]))})
            dmg1 = np.array(dmgs[0])
        else:
            s_star, status = s_star_frozen["s_star"], s_star_frozen["status"]
            R1, _ = self._ablate_run(l, "R", scale=1.0, norms=norms)
            Rs, _ = self._ablate_run(l, "R", scale=s_star, norms=norms)
            R1c = R1["two_hop_correct"]
            dmg1 = np.array(R1["text_change"])
            dmg_iso = np.array(Rs["text_change"])
            res.update({"imp_rand1_pp": 100.0 * (acc0 - float(np.mean(R1["two_hop_correct"]))),
                        "dmg_rand1": float(dmg1.mean()),
                        "imp_rand_sstar_pp_confirm": 100.0 * (acc0 - float(np.mean(Rs["two_hop_correct"])))})
        si_norm = float(dmgJ.mean() / dmg1.mean()) if dmg1.mean() > 0 else math.inf
        si_iso = float(dmgJ.mean() / np.mean(dmg_iso)) if np.mean(dmg_iso) > 0 else math.inf
        ci = boot_ratio_ci(dmgJ, dmg_iso, self.boot_w5, CA.SEED + l)
        w4_diff = impJ - res["imp_rand1_pp"]
        res.update({"s_star": s_star, "s_star_status": status, "SI_norm": si_norm, "SI_iso": si_iso,
                    "SI_iso_ci": list(ci), "w4_diff_pp": w4_diff})
        res["pass_w4"] = bool(acc0 >= CA.W0B_MIN_ACC and w4_diff >= CA.W4_MIN_DIFF_PP)
        certifiable = status in ("interpolated", "unreached_upper_bound")
        res["pass_w5"] = bool(certifiable and si_iso <= CA.W5_MAX_SI_ISO and ci[1] <= CA.W5_MAX_CI_UP)
        res["w5_description"] = (
            "selective at matched functional impact but anti-selective at matched intervention norm"
            if res["pass_w5"] and si_norm > 1.0 else
            ("selective at matched functional impact and at matched norm" if res["pass_w5"] else "not selective"))
        # ---- R2 additions: paired item bootstrap for the W4 difference (D74)
        d_i = 100.0 * (np.asarray(R1c, float) - np.asarray(J["two_hop_correct"], float))
        res["w4_diff_ci"] = boot_mean_ci(d_i, C.CI_BOOT, C.SEED_RNG + l)
        res["s_star_certifiable"] = bool(certifiable)
        self._t("w45", t0)
        return res

    # ------------------------------------------------------------------ W3' + W3b
    def w3_refs(self, layers=None):
        L = layers if layers is not None else sorted({x for l in self.layers for x in self.w3.site_layers(l)})
        self.w3.references(L)
        self.w3.natural(list(range(len(self.w3.concepts))))

    def w3p(self, l, dirs_w1, cvec, a):
        t0 = time.time()
        W = self.w3
        norm = a * self.hbar[l]
        K0 = len(W.concepts)
        uJ, up = [], []
        for c in W.concepts:
            if c in dirs_w1:
                u1, u2 = dirs_w1[c]
            else:
                u1, _, u2, _ = self.directions(l, cvec[l][c], c, norm=norm)
            uJ.append(u1)
            up.append(u2)
        missing = [W.concepts[i] for i in range(K0) if up[i] is None]
        out = {"unmatched_concepts": missing, "norm": norm}
        if len(missing) > C.W3P_MAX_UNMATCHED:
            out.update({"assessable": False, "reason": f"{len(missing)} W3' concepts lack a matched perp",
                        "pass": False, "diff": float("nan"), "powered_failure": False})
            out["eng_diagnostics"] = self._w3p_diagnostics(l, uJ, norm)       # reported only (no perp needed)
            self._t("w3p", t0)
            return out
        subset = [i for i in range(K0) if up[i] is not None]
        K = len(subset)
        perm = self.m["w3p"]["derangement"]
        der = self._restricted_derangement(perm, subset)
        VJ = torch.stack([norm * uJ[i] for i in subset])
        VP = torch.stack([norm * up[i] for i in subset])
        R = unit_directions(f"{self.lm.key}|w3p-rand|{l}", (K0, self.d))[torch.tensor(subset)]
        VR = norm * R
        emb = self.lm.text.embed_tokens.weight
        VL = torch.stack([norm * (lambda e: e / e.norm())(emb[self.tok1(W.words[W.concepts[i]])].float())
                          for i in subset])
        opts = {}
        for c, o in self.m["w3p"]["w3b_options"].items():
            opts[c] = ([self.tok1(self.concepts[x]) for x in o], o.index(c))
        arms = {"J": W.arm(l, VJ, subset, w3b_options=opts, score_word=True, derangement=der),
                "perp": W.arm(l, VP, subset, w3b_options=opts),
                "nc1": W.arm(l, VR, subset),
                "nc3": W.arm(l, VL, subset, score_word=True)}
        sl = W.site_layers(l)
        li = [W._ref["layers"].index(x) for x in sl]
        nat = W.natural(subset)[:, :, li]
        s = W.summarize(arms, nat, l, K)
        s["PC1"] = W.pc1(l, subset)
        s["controls_pass"] = bool(s["controls_pass"] and s["PC1"]["pass"])
        s["assessable"] = s["controls_pass"]
        s["pass"] = bool(s["assessable"] and s["gate_criterion"])
        s["powered_failure"] = bool(s["assessable"] and s["ci"][1] < C.PF["W3p"]["diff_ci_upper_lt"]
                                    and s["diff"] <= C.PF["W3p"]["diff_point_le"])
        s["subset"] = [W.concepts[i] for i in subset]
        s["site_layers"] = sl
        # W3b (reported only)
        if not hasattr(self, "_w3b_base") or self._w3b_base[0] != tuple(subset):
            self._w3b_base = (tuple(subset), W.w3b_baselines(opts, subset))
        none, natw = self._w3b_base[1]
        w3b = {"hit_none": float(none.mean()), "hit_natural_word": float(natw.mean()),
               "hit_J": float(arms["J"]["w3b_hits"].mean()), "hit_perp": float(arms["perp"]["w3b_hits"].mean())}
        w3b["assessable"] = bool(w3b["hit_natural_word"] >= C.W3B_NAT_MIN)
        for k in ("hit_J", "hit_perp"):
            key = "J" if k == "hit_J" else "perp"
            w3b[k + "_ci"] = boot_mean_ci(arms[key]["w3b_hits"].reshape(-1), C.CI_BOOT, C.SEED_RNG + l)
        s["w3b"] = w3b
        out.update(s)
        self._t("w3p", t0)
        return out

    def _w3p_diagnostics(self, l, uJ, norm):
        """PC3 (J route), NC1, PC1, PC2 on all W3' concepts; used when W3' is not assessable (reported only)."""
        W = self.w3
        subset = list(range(len(W.concepts)))
        K = len(subset)
        VJ = torch.stack([norm * uJ[i] for i in subset])
        VR = norm * unit_directions(f"{self.lm.key}|w3p-rand|{l}", (K, self.d))
        aj, ar = W.arm(l, VJ, subset), W.arm(l, VR, subset)
        li = [W._ref["layers"].index(x) for x in W.site_layers(l)]
        d = W.diagnostics(aj, ar, W.natural(subset)[:, :, li], K)
        d["PC1"] = W.pc1(l, subset)
        return d

    @staticmethod
    def _restricted_derangement(perm, subset):
        """Fixed derangement (design seed) restricted to the concept subset: map each kept concept to the next kept
        concept along the permutation cycle (still a derangement of the subset)."""
        keep = set(subset)
        pos = {k: j for j, k in enumerate(subset)}
        out = []
        for k in subset:
            nxt = perm[k]
            while nxt not in keep:
                nxt = perm[nxt]
            if nxt == k:
                nxt = subset[(pos[k] + 1) % len(subset)]
            out.append(pos[nxt])
        return out

    # ------------------------------------------------------------------ orchestration
    def run_all_r2(self, frozen_layer_params=None):
        self.prepare()
        out = {"model": self.lm.key, "band_all": self.band_all, "layers": self.layers,
               "vgen_size": len(self.vgen), "w0": self.w0_r2(), "per_layer": {}}
        cvec = self.concept_vectors()
        kvec = self.country_vectors()
        self.w3_refs()
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
            pl["w3p"] = self.w3p(l, dirs, cvec, a)
            pl["w45"] = self.w45(l, None if frozen_layer_params is None else frozen_layer_params["s_star"])
            out["per_layer"][l] = pl
            self.log(f"[{self.lm.key}] layer {l} done: W1 {pl['w1']['pass']} W2 {pl['w2']['pass']} "
                     f"W3' {pl['w3p'].get('pass')} W4 {pl['w45']['pass_w4']} W5 {pl['w45']['pass_w5']}")
        out["timings_s"] = dict(self.timings, **{"w3p_" + k: v for k, v in self.w3.timings.items()})
        return out
