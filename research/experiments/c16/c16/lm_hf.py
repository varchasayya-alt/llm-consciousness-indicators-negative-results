"""HF subject: cached Qwen2.5-0.5B-Instruct, fp32, CPU, forward only (prereg §4).

Correctness is greedy correctness by teacher forcing. Patching adds a delta to the residual entering decoder layer l
(hidden_states[l]) at the episode's prefix positions via a forward pre-hook. No parameter is ever trained here; the E4
timing probe in stage0 asserts that.
"""
from __future__ import annotations

import os
import time

import torch

from . import config as C
from . import materials as M
from .subject import EpSpec, SentSpec

os.environ.setdefault("HF_HOME", C.HF_HOME)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def render_sentence(s: SentSpec) -> str:
    if s.kind == "latent":
        p, args = s.payload
        return M.producer_sentence(s.name, p, args)
    if s.kind == "text":
        return M.text_sentence(s.name, s.payload[0])
    if s.kind == "bundle":
        b, x = s.payload
        return M.bundle_sentence(s.name, b, x)
    raise ValueError(s.kind)


class HFSubject:
    PREFIX_LEN = 7

    def __init__(self, fmt="F1", batch_size=32, threads=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.fmt = fmt
        self.bs = batch_size
        self.tok = AutoTokenizer.from_pretrained(C.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, dtype=torch.float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.n_tokens = 0
        self.t_forward = 0.0

    # ------------------------------------------------------------ encoding
    def encode(self, sent: SentSpec, consumer="copy", table=None, ans=None):
        """Return dict(ids, prefix_pos, prod_pos, ans_pos, ans_ids). ans: answer string or None."""
        sentence = render_sentence(sent)
        p, span = M.prompt(sentence, sent.name, consumer, table, fmt=self.fmt)
        full = p + ("" if ans is None else " " + ans)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        prefix_pos = [k for k, (a, b) in enumerate(offs) if a >= span[0] and b <= span[1]]
        assert len(prefix_pos) == self.PREFIX_LEN, (full, prefix_pos)
        assert self.tok.decode([ids[k] for k in prefix_pos]) == M.PREFIX_FMT.format(name=sent.name)
        sent_start = span[0] - len(sentence) - 1
        prod_pos = [k for k, (a, b) in enumerate(offs) if sent_start <= a < span[0]]
        ans_pos, ans_ids = [], []
        if ans is not None:
            starts = [k for k, (a, b) in enumerate(offs) if a >= len(p)]
            assert starts and offs[starts[0]][0] == len(p), "answer must start at a token boundary"
            assert all(offs[k][1] <= len(p) for k in range(starts[0])), "prompt/answer token straddle"
            ans_pos = starts
            ans_ids = [ids[k] for k in starts]
        return {"ids": ids, "prefix_pos": prefix_pos, "prod_pos": prod_pos, "ans_pos": ans_pos, "ans_ids": ans_ids}

    # ------------------------------------------------------------ forward
    def _forward(self, encs, layer=None, deltas=None, hidden_layers=None):
        n = len(encs)
        L = max(len(e["ids"]) for e in encs)
        ids = torch.full((n, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, L), dtype=torch.long)
        for i, e in enumerate(encs):
            ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
            mask[i, :len(e["ids"])] = 1
        handle = None
        if deltas is not None:
            rows, cols, vecs = [], [], []
            for i, (e, d) in enumerate(zip(encs, deltas)):
                if d is None:
                    continue
                for k, pos in enumerate(e["prefix_pos"]):
                    rows.append(i)
                    cols.append(pos)
                    vecs.append(d[k])
            if rows:
                r = torch.tensor(rows)
                c = torch.tensor(cols)
                v = torch.stack(vecs).to(torch.float32)

                def hook(module, args, kwargs):
                    if args:
                        h = args[0].clone()
                        h[r, c] += v
                        return (h,) + tuple(args[1:]), kwargs
                    h = kwargs["hidden_states"].clone()
                    h[r, c] += v
                    kwargs["hidden_states"] = h
                    return args, kwargs
                handle = self.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
        t0 = time.time()
        try:
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=hidden_layers is not None,
                                 use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        self.t_forward += time.time() - t0
        self.n_tokens += int(mask.sum())
        return out

    def _batches(self, items):
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]["ids"]))
        for s in range(0, len(order), self.bs):
            yield order[s:s + self.bs]

    # ------------------------------------------------------------ API
    def prefix_hidden(self, sents, layers, key="prefix_pos", last=None):
        encs = [self.encode(s) for s in sents]
        P = self.PREFIX_LEN if key == "prefix_pos" else last
        res = torch.zeros((len(encs), len(layers), P, self.d_model))
        for idx in self._batches([(e,) for e in encs]):
            out = self._forward([encs[i] for i in idx], hidden_layers=layers)
            hs = out.hidden_states
            for b, i in enumerate(idx):
                pos = encs[i][key] if key == "prefix_pos" else encs[i][key][-last:]
                for li, l in enumerate(layers):
                    res[i, li] = hs[l][b, pos]
        return res

    def prod_hidden(self, sents, layers, last=3):
        return self.prefix_hidden(sents, layers, key="prod_pos", last=last)

    def evaluate(self, eps, layer=None, deltas=None):
        """Greedy correctness for every candidate of every episode."""
        jobs = []          # (enc, ep_index, labels-checked-from-this-forward)
        for i, ep in enumerate(eps):
            answers = {lab: M.answer(ep.consumer, x, ep.table) for lab, x in ep.cands.items()}
            toks = {lab: self.tok(" " + a, add_special_tokens=False)["input_ids"] for lab, a in answers.items()}
            single = all(len(t) == 1 for t in toks.values())
            if single:
                lab0 = next(iter(answers))
                jobs.append((self.encode(ep.sent, ep.consumer, ep.table, answers[lab0]), i,
                             {lab: toks[lab][0] for lab in answers}))
            else:
                for lab, a in answers.items():
                    jobs.append((self.encode(ep.sent, ep.consumer, ep.table, a), i, {lab: None}))
        res = [dict() for _ in eps]
        for idx in self._batches(jobs):
            encs = [jobs[j][0] for j in idx]
            dl = None if deltas is None else [deltas[jobs[j][1]] for j in idx]
            out = self._forward(encs, layer=layer, deltas=dl)
            logits = out.logits
            for b, j in enumerate(idx):
                enc, i, labs = jobs[j]
                pred_pos = [p - 1 for p in enc["ans_pos"]]
                am = logits[b, pred_pos].argmax(-1).tolist()
                for lab, single_tok in labs.items():
                    if single_tok is not None:
                        res[i][lab] = am[0] == single_tok
                    else:
                        res[i][lab] = am == enc["ans_ids"]
        return res

    # ------------------------------------------------------------ engineering
    def zero_patch_identity(self, sents):
        encs = [self.encode(s, "parity", None, "odd") for s in sents]
        a = self._forward(encs).logits
        z = [torch.zeros(self.PREFIX_LEN, self.d_model) for _ in encs]
        b = self._forward(encs, layer=3, deltas=z).logits
        return float((a - b).abs().max())

    def hook_matches_hidden_states(self, sent, layer=5):
        """The tensor seen by the pre-hook of `layer` equals output.hidden_states[layer]."""
        enc = self.encode(sent)
        seen = {}

        def hook(module, args, kwargs):
            seen["h"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
            return None
        h = self.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            out = self._forward([enc], hidden_layers=[layer])
        finally:
            h.remove()
        return float((seen["h"] - out.hidden_states[layer]).abs().max())

    def backward_timing(self, eps, layers=(4, 8, 12), rank=64, reps=3):
        """E4: forward+backward through the frozen model with a zero-init dummy adapter; NO optimizer step."""
        A = [torch.zeros(self.d_model, rank, requires_grad=True) for _ in layers]
        B = [(torch.randn(rank, self.d_model) * 0.01).requires_grad_(True) for _ in layers]
        before = [a.detach().clone() for a in A] + [b.detach().clone() for b in B]
        handles = []
        for li, l in enumerate(layers):
            def hook(module, args, kwargs, li=li):
                h = args[0] if args else kwargs["hidden_states"]
                h = h + (h @ A[li]) @ B[li]
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = h
                return args, kwargs
            handles.append(self.layers[l].register_forward_pre_hook(hook, with_kwargs=True))
        times = []
        try:
            for _ in range(reps):
                encs = []
                for ep in eps:
                    lab = next(iter(ep.cands))
                    encs.append(self.encode(ep.sent, ep.consumer, ep.table,
                                            M.answer(ep.consumer, ep.cands[lab], ep.table)))
                n = len(encs)
                L = max(len(e["ids"]) for e in encs)
                ids = torch.full((n, L), self.pad_id, dtype=torch.long)
                mask = torch.zeros((n, L), dtype=torch.long)
                for i, e in enumerate(encs):
                    ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
                    mask[i, :len(e["ids"])] = 1
                t0 = time.time()
                out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                loss = 0.0
                for i, e in enumerate(encs):
                    pp = [p - 1 for p in e["ans_pos"]]
                    loss = loss + torch.nn.functional.cross_entropy(out.logits[i, pp], torch.tensor(e["ans_ids"]))
                loss.backward()
                times.append(time.time() - t0)
        finally:
            for h in handles:
                h.remove()
        after = [a.detach() for a in A] + [b.detach() for b in B]
        assert all(torch.equal(x, y) for x, y in zip(before, after)), "timing probe must not update parameters"
        assert all(p.grad is None for p in self.model.parameters()), "LM weights must not receive gradients"
        return {"batch": len(eps), "sec_per_batch": sorted(times)[len(times) // 2],
                "sec_per_example": sorted(times)[len(times) // 2] / len(eps)}
