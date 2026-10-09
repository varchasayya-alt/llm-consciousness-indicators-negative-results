"""W3': identity-specific transport (memo/c15r2_preregistration_FROZEN.md sec. 3 and 12.3), and W3b.

Model-agnostic core: it needs only lm.tok, lm.layers / lm.text (through c15a.hooks.forward), lm.n_layers and
lm.d_model, so the identical code runs on the real candidates and on the planted-transport model of the PC4 test.

* References F_k(site): centred (across the concept set S) mean over fold A x {p1, p2} of the block output at
  site = (layer, offset after the slot NP's last token), paraphrase NPs only. Word references (NC3) likewise.
* Natural benchmark: p3 in fold B, centred across S within each context; identification against F.
* Injection arms: at layer l, add vec_k to the ' thing' token of the fold-B carrier; centre the 8 injected effects
  across S within each context (generic influence cancels); identification argmax_k' cos(D_ck, F_k') == k.
* Site criterion: one-sided binomial p < 0.05 vs 1/|S| on |S| x 24 trials AND acc >= max(0.30, 0.5 acc_nat).
* Gate: BB_J - BB_perp >= 0.15 and the 2.5th percentile of 1000 contexts x concepts bootstrap replicates > 0.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

from c15a.hooks import add_at, forward

from . import config as C


def _batches(n, bs):
    for i in range(0, n, bs):
        yield list(range(i, min(n, i + bs)))


def kcrit(n, k):
    """Smallest count with one-sided binomial p < alpha against chance 1/k on n trials."""
    return next(c for c in range(n + 1) if stats.binom.sf(c - 1, n, 1.0 / k) < C.W3P_ALPHA)


def site_pass(cnt, cnt_nat, n, k):
    acc, accn = cnt / n, cnt_nat / n
    return (cnt >= kcrit(n, k)) & (acc >= np.maximum(C.W3P_FLOOR_ABS, C.W3P_FLOOR_REL * accn) - 1e-12)


class W3PAssay:
    def __init__(self, lm, w3, *, bs=8, log=print):
        self.lm, self.w3, self.bs, self.log = lm, w3, bs, log
        self.enc = lambda s: lm.tok(s, add_special_tokens=True).input_ids
        self.ctx = {r["id"]: r for r in w3["contexts"]}
        self.A, self.B = list(w3["fold_A"]), list(w3["fold_B"])
        self.concepts = list(w3["concepts"])
        self.words = w3["words"]
        self.para = w3["paraphrases"]
        self.offsets = (0,) + tuple(C.W3P_OFFSETS)
        self.timings = defaultdict(float)
        self._ref = None
        self._nat = {}

    # ------------------------------------------------------------------ sequences
    def seq(self, cid, np_text, suffix=""):
        r = self.ctx[cid]
        head = self.enc(r["prefix"] + np_text)
        full = self.enc(r["prefix"] + np_text + r["cont"] + suffix)
        last = len(head) + max(C.W3P_OFFSETS)
        if full[:len(head)] != head:
            raise RuntimeError(f"slot tokenization not prefix-stable: {cid} {np_text!r}")
        if suffix and full[:last] != self.enc(r["prefix"] + np_text + r["cont"])[:last]:
            raise RuntimeError(f"suffix changes W3' site tokens: {cid}")
        if len(full) < last:
            raise RuntimeError(f"continuation too short: {cid}")
        return full, len(head) - 1

    def site_layers(self, l):
        return list(range(l + C.W3P_LAYER_STEP, self.lm.n_layers - 1, C.W3P_LAYER_STEP))

    def _states(self, items, layers, edits=None, logits_at_end=False):
        """items: list of (ids, slot_end). Returns states [n, len(layers), len(offsets), d] (float32) and,
        optionally, final-position logits. edits: per-item vectors [n, d] added at the slot at layer edits_layer."""
        n = len(items)
        out = torch.zeros(n, len(layers), len(self.offsets), self.lm.d_model)
        last_logits = [None] * n
        groups = defaultdict(list)
        for i, (ids, _) in enumerate(items):
            groups[len(ids)].append(i)
        for T, idx in groups.items():
            for b in _batches(len(idx), self.bs):
                rows = [idx[j] for j in b]
                ids = torch.tensor([items[i][0] for i in rows])
                pos = torch.tensor([items[i][1] for i in rows])
                ed = None
                if edits is not None:
                    el, V = edits
                    ed = {el: add_at(pos, V[rows])}
                h, rec = forward(self.lm, ids, edits=ed, record=layers)
                for li, L in enumerate(layers):
                    for oi, o in enumerate(self.offsets):
                        out[rows, li, oi] = rec[L][torch.arange(len(rows)), pos + o].float()
                if logits_at_end:
                    lg = self.lm.lm_head(h[:, -1]).float()[:, : self.lm.n_vocab]
                    for j, i in enumerate(rows):
                        last_logits[i] = lg[j]
        return out, last_logits

    # ------------------------------------------------------------------ references (once per model)
    def references(self, layers):
        """Raw (uncentred) mean states per concept for p1, p2 and the word version over fold A, at `layers`."""
        t0 = time.time()
        K = len(self.concepts)
        M = {v: torch.zeros(K, len(layers), len(self.offsets), self.lm.d_model) for v in ("p1", "p2", "word")}
        for ki, c in enumerate(self.concepts):
            for v, np_text in (("p1", " " + self.para[c][0]), ("p2", " " + self.para[c][1]),
                               ("word", " the " + self.words[c])):
                S, _ = self._states([self.seq(x, np_text) for x in self.A], layers)
                M[v][ki] = S.mean(0)
        self._ref = {"layers": list(layers), "M": M}
        self.timings["references"] += time.time() - t0
        return self._ref

    def _centred_refs(self, subset):
        M = self._ref["M"]
        idx = torch.tensor(subset)
        out = {}
        for v in ("p1", "p2", "word"):
            X = M[v][idx]
            out[v] = X - X.mean(0, keepdim=True)
        P = M["p1"][idx] + M["p2"][idx]
        out["para"] = 0.5 * (P - P.mean(0, keepdim=True))
        return out

    @staticmethod
    def _identify(D, F):
        """D: [K, nL, nO, d] centred effects; F: [K, nL, nO, d] references. Returns argmax index [K, nL, nO]."""
        Dn = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Fn = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = torch.einsum("klod,jlod->kjlo", Dn, Fn)
        return cos.argmax(1)

    def natural(self, subset):
        """Correctness [C_B, K, nL_ref, nO] of the held-out paraphrase p3 (fold B) against the paraphrase refs."""
        key = tuple(subset)
        if key in self._nat:
            return self._nat[key]
        t0 = time.time()
        refs = self._centred_refs(subset)
        K = len(subset)
        out = np.zeros((len(self.B), K, len(self._ref["layers"]), len(self.offsets)), bool)
        for ci, x in enumerate(self.B):
            items = [self.seq(x, " " + self.para[self.concepts[k]][2]) for k in subset]
            S, _ = self._states(items, self._ref["layers"])
            D = S - S.mean(0, keepdim=True)
            am = self._identify(D, refs["para"])
            out[ci] = (am == torch.arange(K)[:, None, None]).numpy()
        self._nat[key] = out
        self.timings["natural"] += time.time() - t0
        return out

    # ------------------------------------------------------------------ injection arms
    def arm(self, l, vecs, subset, *, w3b_options=None, score_word=False, derangement=None):
        """vecs: [K, d] injection vectors (rows follow `subset`). Returns per-arm arrays over fold B:
        correct [C, K, nL, nO] against paraphrase refs (and word refs / deranged labels if requested), and W3b
        final-position hits (if w3b_options given: {concept id: option token ids, answer index})."""
        t0 = time.time()
        refs = self._centred_refs(subset)
        ref_layers = self._ref["layers"]
        sl = self.site_layers(l)
        li = [ref_layers.index(L) for L in sl]
        K = len(subset)
        res = {"para": np.zeros((len(self.B), K, len(sl), len(self.offsets)), bool)}
        if score_word:
            res["word"] = np.zeros_like(res["para"])
        if derangement is not None:
            res["deranged"] = np.zeros_like(res["para"])
        hits = np.zeros((len(self.B), K), bool) if w3b_options else None
        suffix = self.w3["w3b_suffix"] if w3b_options else ""
        for ci, x in enumerate(self.B):
            items = [self.seq(x, self.w3["carrier"], suffix)] * K
            S, lg = self._states(items, sl, edits=(l, vecs), logits_at_end=bool(w3b_options))
            D = S - S.mean(0, keepdim=True)
            am = self._identify(D, refs["para"][:, li])
            res["para"][ci] = (am == torch.arange(K)[:, None, None]).numpy()
            if score_word:
                aw = self._identify(D, refs["word"][:, li])
                res["word"][ci] = (aw == torch.arange(K)[:, None, None]).numpy()
            if derangement is not None:
                res["deranged"][ci] = (am == torch.tensor(derangement)[:, None, None]).numpy()
            if w3b_options:
                for j, k in enumerate(subset):
                    opt_ids, ans = w3b_options[self.concepts[k]]
                    hits[ci, j] = int(lg[j][torch.tensor(opt_ids)].argmax()) == ans
        res["w3b_hits"] = hits
        res["site_layers"] = sl
        self.timings["arms"] += time.time() - t0
        return res

    def w3b_baselines(self, w3b_options, subset):
        """W3b 'none' arm (carrier, no injection) and natural-word control (fold B), final-position hits."""
        none = np.zeros((len(self.B), len(subset)), bool)
        nat = np.zeros_like(none)
        for ci, x in enumerate(self.B):
            _, lg = self._states([self.seq(x, self.w3["carrier"], self.w3["w3b_suffix"])], [0], logits_at_end=True)
            items = [self.seq(x, " the " + self.words[self.concepts[k]], self.w3["w3b_suffix"]) for k in subset]
            _, lgn = self._states(items, [0], logits_at_end=True)
            for j, k in enumerate(subset):
                opt_ids, ans = w3b_options[self.concepts[k]]
                none[ci, j] = int(lg[0][torch.tensor(opt_ids)].argmax()) == ans
                nat[ci, j] = int(lgn[j][torch.tensor(opt_ids)].argmax()) == ans
        return none, nat

    # ------------------------------------------------------------------ statistics
    def pc1(self, l, subset):
        refs = self._centred_refs(subset)
        li = [self._ref["layers"].index(L) for L in self.site_layers(l)]
        a, b = refs["p1"][:, li, 1:], refs["p2"][:, li, 1:]          # downstream offsets only
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)     # [K, nL, nO-1]
        med = cos.median(0).values
        frac = float((med >= C.PC1_MIN_COS).float().mean())
        return {"median_cos_by_site_mean": float(med.mean()), "frac_sites": frac, "pass": frac >= C.PC1_SITE_FRAC}

    @staticmethod
    def diagnostics(arm_j, arm_nc1, nat, K):
        """Engineering diagnostics that need no perp route: PC3 on the J route (offset 0, layer l+2), NC1 (random
        directions, site criterion vs the natural benchmark), PC2, and mean identification accuracies."""
        Cn = nat.shape[0]
        n = Cn * K
        flat = lambda X: X[:, :, :, 1:].reshape(Cn, K, -1)          # noqa: E731
        cntn = flat(nat).sum((0, 1))
        cj = flat(arm_j["para"]).sum((0, 1))
        cr = flat(arm_nc1["para"]).sum((0, 1))
        c0 = int(arm_j["para"][:, :, 0, 0].sum())
        pc2_site = (cntn >= kcrit(n, K)) & (cntn / n >= C.PC2_MIN_ACC)
        bb_r = float(site_pass(cr, cntn, n, K).mean())
        return {"K": K, "n_trials": n, "kcrit": kcrit(n, K),
                "PC3": {"by_route": {"J": {"acc": c0 / n, "pass": bool(c0 >= kcrit(n, K) and c0 / n >= C.PC3_MIN_ACC)}},
                        "pass": bool(c0 >= kcrit(n, K) and c0 / n >= C.PC3_MIN_ACC)},
                "NC1": {"BB": bb_r, "pass": bool(bb_r <= C.NC1_MAX)},
                "PC2": {"frac_sites": float(pc2_site.mean()), "pass": bool(pc2_site.mean() >= C.PC2_SITE_FRAC)},
                "BB_J": float(site_pass(cj, cntn, n, K).mean()),
                "mean_acc_J": float(cj.mean() / n), "mean_acc_nc1": float(cr.mean() / n),
                "mean_acc_nat": float(cntn.mean() / n)}

    @staticmethod
    def summarize(arms, nat, l, K, n_boot=C.W3P_BOOT, seed=None):
        """arms: {"J": res, "perp": res, "nc1": res, "nc3": res}; nat: [C, K, nL_sites, nO] natural correctness at
        the arm's site layers. Downstream sites = offsets 1.. (index 1:)."""
        Cn = nat.shape[0]
        n = Cn * K
        flat = lambda X: X[:, :, :, 1:].reshape(Cn, K, -1)          # noqa: E731
        cnt = {a: flat(r["para"]).sum((0, 1)) for a, r in arms.items()}
        cntn = flat(nat).sum((0, 1))
        sp = {a: site_pass(cnt[a], cntn, n, K) for a in cnt}
        BB = {a: float(sp[a].mean()) for a in sp}
        out = {"K": K, "n_trials": n, "kcrit": kcrit(n, K), "n_sites": int(cntn.shape[0]),
               "BB": BB, "mean_acc": {a: float(cnt[a].mean() / n) for a in cnt},
               "mean_acc_nat": float(cntn.mean() / n)}
        diff = BB["J"] - BB["perp"]
        # PC2
        pc2_site = (cntn >= kcrit(n, K)) & (cntn / n >= C.PC2_MIN_ACC)
        out["PC2"] = {"frac_sites": float(pc2_site.mean()), "pass": bool(pc2_site.mean() >= C.PC2_SITE_FRAC)}
        # PC3: offset 0 at layer l+2 (first site layer)
        pc3 = {}
        for a in ("J", "perp"):
            c0 = int(arms[a]["para"][:, :, 0, 0].sum())
            pc3[a] = {"acc": c0 / n, "pass": bool(c0 >= kcrit(n, K) and c0 / n >= C.PC3_MIN_ACC)}
        out["PC3"] = {"by_route": pc3, "pass": bool(pc3["J"]["pass"] or pc3["perp"]["pass"])}
        # NC1, NC2
        out["NC1"] = {"BB": BB.get("nc1", 0.0), "pass": bool(BB.get("nc1", 0.0) <= C.NC1_MAX)}
        cd = flat(arms["J"]["deranged"]).sum((0, 1))
        bb_nc2 = float(site_pass(cd, cntn, n, K).mean())
        out["NC2"] = {"BB": bb_nc2, "pass": bool(bb_nc2 <= C.NC2_MAX)}
        # NC3 / LEI (reported only)
        if "nc3" in arms and "word" in arms["J"]:
            accJw = flat(arms["J"]["word"]).sum((0, 1)) / n
            accJp = cnt["J"] / n
            out["NC3"] = {"LEI": float((accJw - accJp).mean()),
                          "acc_lexical_vs_paraphrase_refs": float(cnt["nc3"].mean() / n),
                          "acc_lexical_vs_word_refs": float(flat(arms["nc3"]["word"]).sum((0, 1)).mean() / n)
                          if "word" in arms["nc3"] else None}
        # bootstrap over contexts x concepts (references fixed; acc_nat re-estimated)
        rng = np.random.default_rng(C.SEED_RNG + l if seed is None else seed)
        wc = rng.multinomial(Cn, np.full(Cn, 1.0 / Cn), size=n_boot).astype(np.float64)
        wk = rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot).astype(np.float64)
        bc = {a: np.rint(np.einsum("bc,bk,cks->bs", wc, wk, flat(arms[a]["para"]).astype(np.float64), optimize=True))
              for a in ("J", "perp")}
        bn = np.rint(np.einsum("bc,bk,cks->bs", wc, wk, flat(nat).astype(np.float64), optimize=True))
        d = site_pass(bc["J"], bn, n, K).mean(1) - site_pass(bc["perp"], bn, n, K).mean(1)
        lo, hi = np.percentile(d, list(C.CI_PCT))
        lb = float(np.percentile(d, C.W3P_LB_PCT))
        out.update({"diff": diff, "ci": [float(lo), float(hi)], "lb": lb})
        out["controls_pass"] = bool(out["PC2"]["pass"] and out["PC3"]["pass"] and out["NC1"]["pass"]
                                    and out["NC2"]["pass"])
        out["gate_criterion"] = bool(diff >= C.W3P_MIN_DIFF and lb > 0)
        return out
