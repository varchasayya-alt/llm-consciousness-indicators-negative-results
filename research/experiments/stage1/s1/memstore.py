"""Stage-1 v4 (Option B1): dual-route store with an explicit, locally editable fact memory.

Architecture (v4_design_memo.md sec.1): the v1-v3 pre-LN transformer (4 blocks) plus a key-value fact memory
read at the [A] position after block 2:
    q = normalize(W_q [LN h2[s1]; LN h2[s2]; LN h2[r]]);  a = softmax(scale * cos(q, k_j) + log m_j) over covered-fact
    slots and a learned NULL (query source and cosine retrieval: implementation deviation D47, for trainability);
    r = sum_j a_j v_j;  h2[A] <- h2[A] + W_o r;  blocks 3-4 integrate; unembedding as before.
Memory-covered facts (world.mem_covered, 70% of trained facts, random) own one slot; parametric-only and unknown
facts own none. Retrieval is supervised (own slot if available, else NULL). During store training each
presentation of a covered fact has its own slot masked with probability p_rd (route dropout) -- the same operation
as T-DELETE, so 'memory unavailable' is in-distribution for the store.

Read set (unchanged): residual at s2 and [A] after layers 0..4. The [A] state after block 2 is recorded BEFORE
memory injection; s2 sites and [A] layers 0-2 are therefore parametric-only (causal mask), [A] layers 3-4 integrated.

MONITOR-INPUT PROHIBITION (D46): retrieval weights, a_null, slot masks/occupancy, addresses, the memory table,
route-dropout state and intervention identity are store-internal. `probe()` never returns them; they are exposed
only by `memory_probe()`, which is for store-only QC and the trivial-decoder audit. Monitor code must not import
this module's memory_probe (unit-tested).

v4.1 GRADIENT ISOLATION (D52; PI 2026-10-03). Architecture and inference are unchanged; only the learning routes:
    PARAMETRIC group P = tok, pos, blocks.0-3 (incl. their LayerNorms), ln_f, unembed      (PARAMETRIC_PARAMS)
    MEMORY group M     = mem_keys, mem_values (incl. NULL row S), q_ln, W_q, W_o, log_scale (MEMORY_PARAMS)
    (exhaustive and disjoint: param_groups() raises otherwise; unit-tested)
  G1  memory-covered fact, own slot AVAILABLE: the whole sequence loss reaches M only (forward with every P tensor
      detached), so P receives exactly zero gradient from it.
  G2  memory-covered fact, own slot ROUTE-DROPPED, and G3 parametric-only facts and mentions: the LM loss reaches P
      only; the injected memory term W_o r is detached (no answer gradient to any M parameter).
  Retrieval CE (every fact row): the query input (block-2 residual at s1, s2, r) is detached, so it trains only the
      addressing subset ADDRESSING_PARAMS = mem_keys, q_ln, W_q, log_scale (never P, mem_values or W_o).
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import world as W
from .store import Block, answer_stats, lm_loss, name_fluency, probe, value_logits

A_POS = 4

# v4.1 parameter partition (D52). Name prefixes of model.named_parameters(); every parameter must match exactly one.
MEMORY_PARAMS = ("mem_keys", "mem_values", "q_ln.", "W_q.", "W_o.", "log_scale")
PARAMETRIC_PARAMS = ("tok.", "pos.", "blocks.", "ln_f.", "unembed.")
ADDRESSING_PARAMS = ("mem_keys", "q_ln.", "W_q.", "log_scale")      # the M subset that retrieval CE may train


def param_groups(model):
    """(P, M): dicts name -> parameter. Raises if any parameter is in neither or both groups."""
    P, M = {}, {}
    for n, p in model.named_parameters():
        in_m, in_p = n.startswith(MEMORY_PARAMS), n.startswith(PARAMETRIC_PARAMS)
        if in_m == in_p:
            raise ValueError(f"parameter {n!r} must belong to exactly one of P / M (in P: {in_p}, in M: {in_m})")
        (M if in_m else P)[n] = p
    return P, M


def slot_table(world):
    """slot index per (entity, relation) for memory-covered facts (row-major order), -1 otherwise; S slots."""
    cov = world.mem_covered
    slot = -np.ones(cov.shape, dtype=np.int64)
    e, r = np.nonzero(cov)
    slot[e, r] = np.arange(len(e))
    return slot, int(len(e))


class DualStore(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, mlp_width, n_slots, d_key=64, inject_after=2, max_len=8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_width) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers
        # explicit fact memory (slot S = NULL)
        self.n_slots, self.d_key, self.inject_after = n_slots, d_key, inject_after
        self.mem_keys = nn.Parameter(torch.randn(n_slots + 1, d_key) / math.sqrt(d_key))
        self.mem_values = nn.Parameter(torch.randn(n_slots + 1, d_model) * 0.02)
        self.q_ln = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(3 * d_model, d_key, bias=False)   # query from block-2 residual at s1, s2, r (D47)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))    # cosine-retrieval temperature (learned)
        self.register_buffer("present", torch.ones(n_slots, dtype=torch.bool))   # availability (T-DELETE edits this)
        # v4.2 (D58): value_mask[S] = 0 makes the NULL value exactly zero ("no explicit memory contribution");
        # inj_cap bounds the norm of the injected term (inf = no cap: v4 / v4.1 behaviour, bit-identical).
        self.register_buffer("value_mask", torch.ones(n_slots + 1))
        self.register_buffer("inj_cap", torch.tensor(float("inf")))

    @property
    def null_index(self):
        return self.n_slots

    def query(self, h_ctx):
        """h_ctx [n, 3, d]: block-2 residual at the query-token positions (s1, s2, r)."""
        return F.normalize(self.W_q(self.q_ln(h_ctx).reshape(len(h_ctx), -1)), dim=-1)

    def retrieval_logits(self, h_ctx, row_slot=None, row_drop=None):
        q = self.query(h_ctx)
        lg = (q @ F.normalize(self.mem_keys, dim=-1).t()) * self.log_scale.exp()      # cosine retrieval
        mask = torch.cat([self.present, torch.ones(1, dtype=torch.bool, device=h_ctx.device)])
        lg = lg.masked_fill(~mask.unsqueeze(0), float("-inf"))
        if row_drop is not None and row_drop.any():                      # route dropout: mask the row's OWN slot
            own = torch.zeros_like(lg, dtype=torch.bool)
            idx = torch.nonzero(row_drop).squeeze(1)
            own[idx, row_slot[idx]] = True
            lg = lg.masked_fill(own, float("-inf"))
        return lg

    def forward(self, x, knockout=None, dropout=None, collect=False, generator=None, add=None,
                row_slot=None, row_drop=None, return_mem=False, m1_mean=None, detach_query=False, detach_memory=False,
                memory_off=False):
        """Same interface as Store.forward, plus (training / store-only diagnostics):
        row_slot [B] own slot (-1 none), row_drop [B] bool (route dropout), return_mem -> also return a dict with
        retrieval logits/weights for [A] rows, m1_mean [d] -> M1 ablation (pre-injection h2[A] replaced by m1_mean).
        v4.1 training only (D52; values are unchanged, only gradients): detach_query -> the query input is detached
        from the backbone; detach_memory -> the injected term W_o r is detached (no answer gradient to M).
        v4.2 Stage A (D58): memory_off -> no retrieval and no injection (the plain parametric transformer)."""
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        resid = [h] if collect else None
        mem = None
        isA = (x[:, A_POS] == W.A) if T > A_POS else torch.zeros(B, dtype=torch.bool)
        for l, blk in enumerate(self.blocks, start=1):
            em = knockout[1] if (knockout is not None and l in knockout[0]) else None
            h = blk(h, em)
            if add is not None and l in add:
                h = h + add[l]
            if dropout is not None:
                rate = dropout[:, l - 1].view(B, 1, 1)
                if (rate > 0).any():
                    u = torch.rand(h.shape, generator=generator, device=h.device)
                    keep = u >= rate
                    h = torch.where(keep, h / (1 - rate).clamp_min(1e-6), torch.zeros_like(h))
            if collect:
                resid.append(h)                                          # [A] after block 2 recorded PRE-injection
            if l == self.inject_after and isA.any() and not memory_off:
                ia = torch.nonzero(isA).squeeze(1)
                hA = h[ia, A_POS]
                hq = h[ia, 1:4]
                lg = self.retrieval_logits(hq.detach() if detach_query else hq, None if row_slot is None else row_slot[ia],
                                           None if row_drop is None else row_drop[ia])
                a = torch.softmax(lg, -1)
                r = a @ (self.mem_values * self.value_mask[:, None])
                u = self.W_o(r)
                if torch.isfinite(self.inj_cap):                         # v4.2 scale control (D58)
                    u = u * (self.inj_cap / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
                base = hA if m1_mean is None else m1_mean.expand_as(hA)
                h = h.clone()
                h[ia, A_POS] = base + (u.detach() if detach_memory else u)
                if return_mem:
                    mem = {"rows": ia, "logits": lg, "a": a, "r": r, "u": u, "hA": hA}
        logits = self.unembed(self.ln_f(h))
        out = (logits, resid) if collect else logits
        return (out, mem) if return_mem else out


def build_dual_store(scfg, mcfg, world):
    _, S = slot_table(world)
    m = DualStore(world.vocab.size, scfg["d_model"], scfg["n_layers"], scfg["n_heads"], scfg["mlp_width"], S,
                  d_key=mcfg["d_key"], inject_after=mcfg["inject_after_block"])
    if mcfg.get("null_value", "learned") == "fixed_zero":                 # v4.2 (D58)
        m.value_mask[S] = 0.0
    return m


def load_state_compat(model, sd):
    """Load a store state dict; stores saved before v4.2 lack the value_mask / inj_cap buffers (defaults kept)."""
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"value_mask", "inj_cap"}, (missing, unexpected)
    return model


@torch.no_grad()
def init_keys_from_queries(model, world, batch=4096):
    """Implementation detail (D47): initialise each fact slot's key with the (untrained) network's own retrieval
    query for that fact, so cosine retrieval is trainable from the start. Keys remain free parameters afterwards."""
    slot, S = slot_table(world)
    e, r = np.nonzero(slot >= 0)
    order = slot[e, r]
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[e], r))
    qs = []
    for i in range(0, len(toks), batch):
        x = toks[i:i + batch]
        h = model.tok(x) + model.pos(torch.arange(x.shape[1]))
        for l, blk in enumerate(model.blocks, start=1):
            h = blk(h)
            if l == model.inject_after:
                break
        qs.append(model.query(h[:, 1:4]))
    q = torch.cat(qs)
    model.mem_keys.data[torch.as_tensor(order)] = q


# ---------------------------------------------------------------- v4.1 gradient-isolated loss (D52)
def _p_detached_call(model, args, kwargs):
    """Forward with every parametric-group tensor replaced by a detached copy (same values; no gradient to P)."""
    P, _ = param_groups(model)
    return torch.func.functional_call(model, {n: p.detach() for n, p in P.items()}, args, kwargs)


def v41_loss(model, x, rs, drop, alpha):
    """v4.1 training loss for one batch. Its VALUE equals the v4 loss (mean next-token CE over non-PAD targets +
    alpha * mean retrieval CE over fact rows); only the gradient routes differ (module docstring, D52).
    x [B,7] sequences; rs [B] own slot (-1: parametric-only fact or mention); drop [B] bool route dropout.
    Returns (loss, parts) with parts[group] = (lm_part, retrieval_part) as floats."""
    inp, tgt = x[:, :-1], x[:, 1:]
    n_tok = (tgt != W.PAD).sum().clamp_min(1)
    n_A = (inp[:, A_POS] == W.A).sum().clamp_min(1)
    g1 = (rs >= 0) & ~drop
    total, parts = None, {}
    for name, rows in (("G1_memory_only", g1), ("G23_parametric", ~g1)):
        idx = torch.nonzero(rows).squeeze(1)
        if len(idx) == 0:
            continue
        kw = dict(row_slot=rs[idx].clamp_min(0), row_drop=drop[idx], return_mem=True, detach_query=True,
                  detach_memory=(name != "G1_memory_only"))
        logits, mem = _p_detached_call(model, (inp[idx],), kw) if name == "G1_memory_only" else model(inp[idx], **kw)
        lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt[idx].reshape(-1), ignore_index=W.PAD,
                             reduction="sum") / n_tok
        ret = torch.zeros((), dtype=lm.dtype)
        if mem is not None and alpha:
            rsr, dr = rs[idx][mem["rows"]], drop[idx][mem["rows"]]
            rt = torch.where((rsr >= 0) & ~dr, rsr, torch.full_like(rsr, model.null_index))
            ret = F.cross_entropy(mem["logits"], rt, reduction="sum") / n_A
        loss = lm + alpha * ret
        total = loss if total is None else total + loss
        parts[name] = (float(lm), float(ret))
    return total, parts


def isolation_audit(model, x, rs, drop, alpha):
    """Structural invariants of v4.1 on a real batch (D52): the largest |gradient| that (i) G1 rows (covered, slot
    available) put on any P tensor, and (ii) G2/G3 rows (route-dropped, parametric-only, mentions) put on mem_values
    or W_o. Both must be exactly 0. Uses autograd.grad (does not touch .grad or the optimizer)."""
    P, M = param_groups(model)
    g1 = (rs >= 0) & ~drop
    out = {}
    for name, rows, targets in (("G1_to_P", g1, list(P.values())),
                                ("G23_to_values_Wo", ~g1, [M["mem_values"], M["W_o.weight"]])):
        idx = torch.nonzero(rows).squeeze(1)
        out[name] = 0.0
        if len(idx):
            loss, _ = v41_loss(model, x[idx], rs[idx], drop[idx], alpha)
            grads = torch.autograd.grad(loss, targets, allow_unused=True, retain_graph=False)
            out[name] = max(0.0 if g is None else float(g.abs().max()) for g in grads)
    return out


# ---------------------------------------------------------------- training
def train_step(model, opt, x, rs, drop, alpha, iso):
    """One optimizer step. iso=True: v4.1 routed loss (D52). iso=False: the v4 shared-gradient loss (for the record).
    Gradients are reset to None, so tensors that receive no gradient in this batch are skipped by the optimizer."""
    if iso:
        loss, _ = v41_loss(model, x, rs, drop, alpha)
    else:
        inp, tgt = x[:, :-1], x[:, 1:]
        logits, mem = model(inp, row_slot=rs.clamp_min(0), row_drop=drop, return_mem=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=W.PAD)
        if mem is not None and alpha:
            rows = mem["rows"]
            rt = torch.where((rs[rows] >= 0) & ~drop[rows], rs[rows], torch.full_like(rs[rows], model.null_index))
            loss = loss + alpha * F.cross_entropy(mem["logits"], rt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


def train_dual_store(world, scfg, mcfg, seed, epochs, p_rd, ckpt_fracs=(), eval_every=0, log=None, stats=None):
    """LM loss on all training sequences + retrieval CE on fact rows (target: own slot if available, else NULL).
    Route dropout masks a covered fact's own slot with probability p_rd per presentation.
    mcfg['gradient_isolation'] (v4.1, D52): route gradients by group (v41_loss); the isolation invariants are
    asserted on the first batch of every epoch. stats (optional dict) receives per-slot counts of presentations and
    route-dropped presentations (store-internal route-dropout state: store-only diagnostics, never monitor input)."""
    from .config import set_all_seeds
    set_all_seeds(seed)
    model = build_dual_store(scfg, mcfg, world)
    init_keys_from_queries(model, world)
    seqs, is_fact, item_idx = W.training_sequences(world)
    slot, S = slot_table(world)
    iso = bool(mcfg.get("gradient_isolation", False))
    drop_count, pres_count = np.zeros(S, dtype=np.int64), np.zeros(S, dtype=np.int64)
    flat_slot = slot.reshape(-1)
    row_slot_all = torch.as_tensor(np.where(item_idx >= 0, flat_slot[np.maximum(item_idx, 0)], -1))
    seqs_t = torch.as_tensor(seqs)
    oc = scfg["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    bs = oc["batch_size"]
    steps_per_epoch = math.ceil(len(seqs) / bs)
    total = steps_per_epoch * epochs
    floor = oc["lr_final"] / oc["lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    ckpt_epochs = {max(1, int(round(f * epochs))): f for f in ckpt_fracs}
    ckpts, curve = {}, []
    g = torch.Generator().manual_seed(seed)
    alpha = mcfg["retrieval_loss_weight"]
    known_items = W.items_where(world, known=True)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(seqs_t), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = seqs_t[b]
            rs = row_slot_all[b]
            drop = (rs >= 0) & (torch.rand(len(b), generator=g) < p_rd)
            if iso and i == 0:                                           # structural invariants (D52): exact zeros
                aud = isolation_audit(model, x, rs, drop, alpha)
                if aud["G1_to_P"] != 0.0 or aud["G23_to_values_Wo"] != 0.0:
                    raise RuntimeError(f"v4.1 gradient isolation violated at epoch {ep}: {aud}")
            train_step(model, opt, x, rs, drop, alpha, iso)
            sched.step()
            cov_b = (rs >= 0).numpy()
            np.add.at(pres_count, rs.numpy()[cov_b], 1)
            np.add.at(drop_count, rs.numpy()[drop.numpy()], 1)
        if ep in ckpt_epochs:
            ckpts[ckpt_epochs[ep]] = copy.deepcopy(model.state_dict())
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            acc = answer_stats(model, world, known_items)["correct"].mean()
            curve.append({"epoch": ep, "trained_fact_acc": float(acc), "minutes": (time.perf_counter() - t0) / 60})
            if log:
                log(f"  epoch {ep}: trained-fact acc {acc:.4f} ({curve[-1]['minutes']:.1f} min)")
    model.eval()
    if stats is not None:
        stats.update(route_drop_count=drop_count, presentations=pres_count, gradient_isolation=iso)
    return model, ckpts, curve


# ---------------------------------------------------------------- v4.2 sequential development (D58)
def tensor_hash(tensors):
    """SHA-256 over the raw bytes of a name -> tensor dict (sorted by name)."""
    import hashlib
    h = hashlib.sha256()
    for n in sorted(tensors):
        h.update(n.encode())
        h.update(tensors[n].detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _cosine_lr(opt, base, floor, frac):
    for gr in opt.param_groups:
        gr["lr"] = base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def stageA_inclusion(world, dose_seed, epochs):
    """Common random numbers for the Stage-A dose: U[slot, epoch] ~ U(0,1); a memory-covered fact's LM training
    sequence is included in epoch e iff U[slot, e] < p_rd. Inclusion sets are therefore nested across p_rd."""
    _, S = slot_table(world)
    return np.random.default_rng(dose_seed).random((S, epochs))


def stageA_epoch_rows(rs_all, U, ep, p_rd):
    """Row indices of one Stage-A epoch (unshuffled): all rows without a slot (parametric-only facts, mentions) plus
    the memory-covered fact rows whose U[slot, ep] < p_rd. Returns (rows, included covered rows)."""
    cov_rows, other_rows = np.nonzero(rs_all >= 0)[0], np.nonzero(rs_all < 0)[0]
    inc = cov_rows[U[rs_all[cov_rows], ep] < p_rd]
    return np.concatenate([other_rows, inc]), inc


def train_stage_A(world, scfg, mcfg, seed, epochs, p_rd, dose_seed, eval_every=0, log=None):
    """v4.2 Stage A (D58): the parametric route alone. The explicit memory does not participate (memory_off: no
    retrieval, no injection) and receives no update. Each epoch contains every parametric-only fact sequence and
    every mention sequence (approved multiplicities), plus each memory-covered fact's ordinary LM sequence only if
    U[slot, epoch] < p_rd; otherwise that fact contributes nothing in that epoch. Optimizer: AdamW over P only.
    Returns (model, stats): stats has per-slot presentation counts, epoch sizes, P and M hashes."""
    from .config import set_all_seeds
    set_all_seeds(seed)
    model = build_dual_store(scfg, mcfg, world)
    P, M = param_groups(model)
    m_hash0 = tensor_hash(M)
    seqs, is_fact, item_idx = W.training_sequences(world)
    slot, S = slot_table(world)
    rs_all = np.where(item_idx >= 0, slot.reshape(-1)[np.maximum(item_idx, 0)], -1)
    U = stageA_inclusion(world, dose_seed, epochs)
    seqs_t = torch.as_tensor(seqs)
    oc = scfg["optimizer"]
    opt = torch.optim.AdamW(list(P.values()), lr=oc["lr"], weight_decay=oc["weight_decay"])
    bs, floor = oc["batch_size"], oc["lr_final"] / oc["lr"]
    g = torch.Generator().manual_seed(seed)
    counts = np.zeros(S, dtype=np.int64)
    sizes, curve = [], []
    known_items = W.items_where(world, known=True)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        rows, inc = stageA_epoch_rows(rs_all, U, ep - 1, p_rd)
        counts[rs_all[inc]] += 1
        rows = rows[torch.randperm(len(rows), generator=g).numpy()]
        n_steps = math.ceil(len(rows) / bs)
        sizes.append(int(len(rows)))
        for k in range(n_steps):
            _cosine_lr(opt, oc["lr"], floor, (ep - 1 + k / n_steps) / epochs)
            x = seqs_t[torch.as_tensor(rows[k * bs:(k + 1) * bs])]
            logits = model(x[:, :-1], memory_off=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=W.PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            model.eval()
            acc = answer_stats(all_masked(model), world, known_items)["correct"].mean()
            curve.append({"epoch": ep, "trained_fact_acc_routeA": float(acc), "minutes": (time.perf_counter() - t0) / 60})
            if log:
                log(f"  stage A epoch {ep}: trained-fact acc (parametric) {acc:.4f} ({curve[-1]['minutes']:.1f} min)")
    model.eval()
    if tensor_hash(M) != m_hash0:
        raise RuntimeError("Stage A changed a memory-group tensor")
    stats = {"presentations": counts, "epoch_sizes": sizes, "curve": curve, "P_hash": tensor_hash(P), "M_hash": m_hash0,
             "minutes": (time.perf_counter() - t0) / 60}
    return model, stats


@torch.no_grad()
def set_injection_cap(model, world, kappa):
    """Scale control (D58): cap = kappa x median pre-injection [A] residual norm over trained facts, computed once
    on the frozen Stage-A network (P never changes afterwards, so the reference is fixed)."""
    kn = W.items_where(world, known=True)
    st = probe(model, world, kn)["states"]
    s = float(np.median(np.linalg.norm(st[:, 2 * model.inject_after + 1], axis=1)))
    model.inj_cap.fill_(kappa * s)
    return s


def stageB_rows(world):
    """Stage-B rows (one epoch): (kind 0) every memory-covered trained fact with its slot AVAILABLE: answer loss +
    retrieval CE to its own slot; (kind 1) every memory-covered trained fact with its own slot masked: retrieval CE
    to NULL only; (kind 2) every parametric-only trained fact: retrieval CE to NULL only."""
    kn = W.items_where(world, known=True)
    cov = kn[world.mem_covered[kn[:, 0], kn[:, 1]]]
    par = kn[~world.mem_covered[kn[:, 0], kn[:, 1]]]
    items = np.concatenate([cov, cov, par])
    kind = np.concatenate([np.zeros(len(cov), int), np.ones(len(cov), int), np.full(len(par), 2)])
    return items, kind


def stageB_step(model, opt, seqs, rs, kind, alpha):
    """One Stage-B optimizer step (P frozen). seqs [b,7] fact sequences; rs [b] own slot (-1 parametric-only);
    kind [b] (stageB_rows). Answer loss: CE of the answer token at [A] for kind-0 rows only. Retrieval CE on all
    rows. Returns (loss, largest |grad| on any P tensor -- must be 0: P has no gradient at all)."""
    x = seqs[:, :5]
    drop = kind == 1
    logits, mem = model(x, row_slot=rs.clamp_min(0), row_drop=drop, return_mem=True)
    ans = kind == 0
    loss = torch.zeros(())
    if ans.any():
        loss = F.cross_entropy(logits[ans, A_POS], seqs[ans, 5])
    tgt = torch.where(kind == 0, rs, torch.full_like(rs, model.null_index))
    loss = loss + alpha * F.cross_entropy(mem["logits"], tgt[mem["rows"]])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    P, _ = param_groups(model)
    leak = max((0.0 if p.grad is None else float(p.grad.abs().max())) for p in P.values())
    opt.step()
    return float(loss), leak


def train_stage_B(modelA, world, scfg, mcfg, seed, epochs, eval_every=0, log=None):
    """v4.2 Stage B (D58): every P tensor frozen (requires_grad False; the optimizer holds M only); the memory learns
    to write into the fixed Stage-A computation. Self-key initialisation (D47) on the frozen network, then the
    injection cap is set (scale control). Asserted: optimizer parameter set == M; no gradient ever reaches P; P hash
    unchanged. Returns (model, stats)."""
    model = copy.deepcopy(modelA)
    P, M = param_groups(model)
    p_hash0 = tensor_hash(P)
    for p in P.values():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    init_keys_from_queries(model, world)
    ref = set_injection_cap(model, world, mcfg["injection_cap_kappa"])
    oc = scfg["optimizer"]
    opt = torch.optim.AdamW(list(M.values()), lr=oc["lr"], weight_decay=oc["weight_decay"])
    if {id(p) for gr in opt.param_groups for p in gr["params"]} != {id(p) for p in M.values()}:
        raise RuntimeError("Stage-B optimizer must hold exactly the memory group")
    items, kind = stageB_rows(world)
    slot, _ = slot_table(world)
    seqs = torch.as_tensor(W.fact_seqs(world.vocab, world.names[items[:, 0]], items[:, 1],
                                       world.answers[items[:, 0], items[:, 1]]))
    rs = torch.as_tensor(slot[items[:, 0], items[:, 1]])
    kind_t = torch.as_tensor(kind)
    bs, floor = oc["batch_size"], oc["lr_final"] / oc["lr"]
    alpha = mcfg["retrieval_loss_weight"]
    g = torch.Generator().manual_seed(seed)
    curve, t0 = [], time.perf_counter()
    cov_items = items[kind == 0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(items), generator=g)
        n_steps = math.ceil(len(perm) / bs)
        for k in range(n_steps):
            _cosine_lr(opt, oc["lr"], floor, (ep - 1 + k / n_steps) / epochs)
            b = perm[k * bs:(k + 1) * bs]
            _, leak = stageB_step(model, opt, seqs[b], rs[b], kind_t[b], alpha)
            if leak != 0.0:
                raise RuntimeError(f"Stage B: gradient reached a frozen P tensor (epoch {ep})")
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            model.eval()
            mp = memory_probe(model, world, cov_items[:3000])
            acc = answer_stats(model, world, cov_items)["correct"].mean()
            curve.append({"epoch": ep, "covered_integrated_acc": float(acc), "W_o_fro": float(model.W_o.weight.norm()),
                          "u_over_hA_median": float(np.median(np.linalg.norm(mp["u"], axis=1) / mp["hA_norm"])),
                          "minutes": (time.perf_counter() - t0) / 60})
            if log:
                log(f"  stage B epoch {ep}: covered integrated acc {acc:.4f} |W_o| {curve[-1]['W_o_fro']:.1f} "
                    f"median |u|/|hA| {curve[-1]['u_over_hA_median']:.2f} ({curve[-1]['minutes']:.1f} min)")
    model.eval()
    for p in P.values():
        p.requires_grad_(True)
    if tensor_hash(P) != p_hash0:
        raise RuntimeError("Stage B changed a parametric-group tensor")
    stats = {"P_hash": p_hash0, "M_hash_after": tensor_hash(M), "hA_reference_median": ref,
             "injection_cap": float(model.inj_cap), "curve": curve, "minutes": (time.perf_counter() - t0) / 60}
    return model, stats


# ---------------------------------------------------------------- store-only memory diagnostics (NOT for monitors)
@torch.no_grad()
def memory_probe(model, world, items, batch=2048):
    """Store-internal retrieval quantities for QC / trivial-decoder audit ONLY (prohibited monitor inputs):
    a (retrieval distribution incl. NULL), a_null, own-slot weight, max weight, entropy, u = W_o r."""
    model.eval()
    slot, _ = slot_table(world)
    out = {k: [] for k in ("a", "u", "hA_norm")}
    for i in range(0, len(items), batch):
        it = items[i:i + batch]
        toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[it[:, 0]], it[:, 1]))
        (_, _), mem = model(toks, collect=True, return_mem=True)
        out["a"].append(mem["a"].numpy())
        out["u"].append(mem["u"].numpy())                                 # the injected term (after any cap)
        out["hA_norm"].append(mem["hA"].norm(dim=-1).numpy())             # pre-injection [A] residual norm
    a = np.concatenate(out["a"])
    own = slot[items[:, 0], items[:, 1]]
    a_own = np.where(own >= 0, a[np.arange(len(a)), np.maximum(own, 0)], np.nan)
    ent = -(a * np.log(np.maximum(a, 1e-12))).sum(1)
    return {"a": a, "a_null": a[:, -1], "a_own": a_own, "a_max": a.max(1), "entropy": ent, "u": np.concatenate(out["u"]),
            "own_slot": own, "hA_norm": np.concatenate(out["hA_norm"])}


def delete_slots(model, world, items):
    """T-DELETE: copy of the store with the memory slots of `items` unavailable (identical full-slot deletion).
    No parameter changes; only the availability buffer `present` differs."""
    m = copy.deepcopy(model)
    slot, _ = slot_table(world)
    s = slot[items[:, 0], items[:, 1]]
    assert (s >= 0).all(), "T-DELETE targets must be memory-covered facts"
    m.present[torch.as_tensor(s)] = False
    m.eval()
    return m


def choose_donors(world, recipients, exclude_items, seed):
    """Y transplant donors: for each recipient (memory-covered fact), a random memory-covered MT-split fact with the
    same relation and the same answer value, not in exclude_items and not of any recipient/excluded entity.
    Returns donors [n,2] (row = -1 if none)."""
    rng = np.random.default_rng(seed)
    cov = world.mem_covered
    mt = W.items_where(world, split=W.MT)
    mt = mt[cov[mt[:, 0], mt[:, 1]]]
    excl_ids = set((exclude_items[:, 0] * world.n_rel + exclude_items[:, 1]).tolist())
    excl_ents = set(exclude_items[:, 0].tolist()) | set(recipients[:, 0].tolist())
    keep = np.array([(e * world.n_rel + r) not in excl_ids and e not in excl_ents for e, r in mt], dtype=bool)
    pool = mt[keep]
    key = pool[:, 1] * world.vocab.n_val + world.answers[pool[:, 0], pool[:, 1]]
    used = np.zeros(len(pool), dtype=bool)
    donors = -np.ones_like(recipients)
    for k in rng.permutation(len(recipients)):
        e, r = recipients[k]
        cand = np.nonzero((key == r * world.vocab.n_val + world.answers[e, r]) & ~used)[0]
        if len(cand):
            j = cand[rng.integers(len(cand))]
            used[j] = True
            donors[k] = pool[j]
    return donors


def transplant(model, world, recipients, donors):
    """Y negative control: copy of the store in which each recipient slot's VALUE is replaced by its donor's value.
    Donor slots, keys and all other parameters are unchanged."""
    m = copy.deepcopy(model)
    slot, _ = slot_table(world)
    rs = slot[recipients[:, 0], recipients[:, 1]]
    ds = slot[donors[:, 0], donors[:, 1]]
    assert (rs >= 0).all() and (ds >= 0).all()
    with torch.no_grad():
        m.mem_values[torch.as_tensor(rs)] = model.mem_values[torch.as_tensor(ds)].clone()
    m.eval()
    return m


# ---------------------------------------------------------------- route ablations (store-only)
def all_masked(model):
    """Ablation A (parametric-only): every fact slot unavailable (NULL only)."""
    m = copy.deepcopy(model)
    m.present[:] = False
    m.eval()
    return m


@torch.no_grad()
def h2A_mean(model, world, items):
    """Mean pre-injection [A] state after block 2 over `items` (for M1)."""
    st = probe(model, world, items)["states"]
    return torch.as_tensor(st[:, 2 * model.inject_after + 1].mean(0), dtype=torch.float32)


@torch.no_grad()
def m1_stats(model, world, items, mean_vec, batch=2048):
    """M1 memory-only ablation (memo sec.14): pre-injection h2[A] replaced by its mean over covered base-correct
    facts; retrieval uses the actual query; memory term kept; blocks 3-4 as normal. Returns correct, margin."""
    model.eval()
    corr, marg = [], []
    for i in range(0, len(items), batch):
        it = items[i:i + batch]
        toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[it[:, 0]], it[:, 1]))
        logits = model(toks, m1_mean=mean_vec)
        lp = torch.log_softmax(value_logits(logits[:, A_POS], it[:, 1], world.vocab), -1)
        ans = torch.as_tensor(world.answers[it[:, 0], it[:, 1]])
        lpc = lp.gather(1, ans[:, None]).squeeze(1)
        other = lp.clone()
        other.scatter_(1, ans[:, None], float("-inf"))
        corr.append((lp.argmax(-1) == ans).numpy())
        marg.append((lpc - other.max(-1).values).numpy())
    return {"correct": np.concatenate(corr), "margin": np.concatenate(marg)}


def m2_probe(model, world, items, seed, lam=1.0, folds=5):
    """M2 memory-sufficiency (pre-declared, D46): can the retrieved memory contribution u = W_o r ALONE supply the
    answer? Cross-fitted L2 multinomial logistic readout from u to the answer value, one readout per relation
    (32 classes), 5 folds; evaluation-only (never part of the store). Returns held-out accuracy."""
    mp = memory_probe(model, world, items)
    U = mp["u"].astype(np.float64)
    y = world.answers[items[:, 0], items[:, 1]]
    rng = np.random.default_rng(seed)
    pred = np.empty(len(items), dtype=int)
    for r in np.unique(items[:, 1]):
        idx = np.nonzero(items[:, 1] == r)[0]
        f = rng.permutation(len(idx)) % folds
        for k in range(folds):
            tr, te = idx[f != k], idx[f == k]
            mu, sd = U[tr].mean(0), U[tr].std(0) + 1e-9
            Xt = torch.as_tensor((U[tr] - mu) / sd, dtype=torch.float32)
            yt = torch.as_tensor(y[tr])
            Wt = torch.zeros(Xt.shape[1], world.vocab.n_val, requires_grad=True)
            bt = torch.zeros(world.vocab.n_val, requires_grad=True)
            opt = torch.optim.LBFGS([Wt, bt], max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(Xt @ Wt + bt, yt) + lam * (Wt ** 2).sum() / len(yt)
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                pred[te] = (torch.as_tensor((U[te] - mu) / sd, dtype=torch.float32) @ Wt + bt).argmax(-1).numpy()
    return float((pred == y).mean())


# ---------------------------------------------------------------- store QC (v4)
def dual_store_qc(model, world, qcfg):
    known = W.items_where(world, known=True)
    unknown = W.items_where(world, known=False)
    acc_known = float(answer_stats(model, world, known)["correct"].mean())
    acc_unknown = float(answer_stats(model, world, unknown)["correct"].mean())
    flu = name_fluency(model, world, world.names)
    from .metrics import auroc
    fam_auroc = float(auroc(flu[world.fam_high], flu[~world.fam_high]))
    cov = known[world.mem_covered[known[:, 0], known[:, 1]]]
    par = known[~world.mem_covered[known[:, 0], known[:, 1]]]
    mc, mp_ = memory_probe(model, world, cov), memory_probe(model, world, par)
    res = {"trained_fact_acc": acc_known, "unknown_fact_acc": acc_unknown, "fluency_auroc_high_vs_low": fam_auroc,
           "acc_covered": float(answer_stats(model, world, cov)["correct"].mean()),
           "acc_param_only": float(answer_stats(model, world, par)["correct"].mean()),
           "own_slot_attention_mean": float(np.nanmean(mc["a_own"])),
           "null_attention_param_only_mean": float(mp_["a_null"].mean()),
           "n_slots": int(model.n_slots)}
    res["pass"] = bool(acc_known >= qcfg["trained_fact_acc_min"] and acc_unknown <= qcfg["unknown_fact_acc_max"]
                       and fam_auroc >= qcfg["fluency_auroc_min"] and res["own_slot_attention_mean"] >= qcfg["own_slot_attention_min"])
    return res


# ---------------------------------------------------------------- P2: interference in the PARAMETRIC route only
def parametric_interference(store0, world, icfg, seed):
    """P2 (designed in the v4 memo sec.15; calibrated only after F0-F4 pass): fine-tune ONLY the transformer
    (parametric) weights on facts + mentions of the reserved interference entities, with no replay. The memory
    table and its read/write interface (keys, values, NULL, W_q, W_o, query LN, temperature) are frozen.
    Stops when new-fact accuracy >= stop_new_fact_acc (checked every eval_every steps) or at max_steps."""
    torch.manual_seed(seed)
    model = copy.deepcopy(store0)
    model.train()
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(not any(n_.startswith(t) for t in MEMORY_PARAMS))
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    names = world.interf_names
    rels = world.interf_rels
    ents = np.repeat(np.arange(len(names)), rels.shape[1])
    rr = rels.reshape(-1)
    facts = W.fact_seqs(world.vocab, names[ents], rr, world.interf_answers[ents, rr])
    ments = W.mention_seqs(world.vocab, np.repeat(names, icfg["mentions_per_entity"], axis=0))
    seqs = torch.as_tensor(np.concatenate([facts, ments]))
    q_items = np.stack([np.zeros(len(ents), int), rr], axis=1)
    opt = torch.optim.Adam(params, lr=icfg["lr"])
    g = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(int(icfg["max_steps"])):
        b = seqs[torch.randint(0, len(seqs), (int(icfg["batch_size"]),), generator=g)]
        loss = lm_loss(model, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % int(icfg["eval_every"]) == 0:
            model.eval()
            acc = float(answer_stats(model, world, q_items, names=names[ents], answers=world.interf_answers[ents, rr])["correct"].mean())
            trace.append({"step": step + 1, "stop_metric": acc})
            model.train()
            if acc >= icfg["stop_new_fact_acc"]:
                break
    for p_ in model.parameters():
        p_.requires_grad_(True)
    model.eval()
    return model, trace
