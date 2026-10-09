"""Store interventions. No function in this module reads monitor outputs (sham non-circularity).

T-FORGET + sham (frozen algorithm, 'forget-then-relearn with displacement matching'; v2 retain objective)
------------------------------------------------------------------------------------------------------
Both target sets receive item-specific optimisation at every step (matched optimisation exposure
and parameter-update opportunity):
  phase A (steps 1..K/2):   X and Y both receive the forgetting objective.
  phase B (steps K/2+1..K): X keeps the forgetting objective; Y receives CE toward its correct
                            answer (competence restored) + gamma * L_match, where L_match matches
                            Y's per-site mean read-state displacement to X's (stop-gradient on X).
Throughout: lambda_retain * L_R, where L_R penalises deviation of the edited store from a FROZEN copy of
the original intact store (RetainKL below). Forgetting objective: 'uniform' = CE(uniform over V_r, p(.|x));
'replace' = CE toward a random new value (REPLACE, exploratory).

Calibration-protocol revision v2 (logs/decisions.md D27-D28): the v1 retain term (LM loss on random training
sequences, ~2/3 of them name-only mentions) left other facts nearly unconstrained (C4 v1: 50-63% collateral
loss). v2 replaces it by RetainKL, and every quantity used in an intervention's QC is computed on items or
entities that never enter the retain pool.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from . import estimands as ES
from . import world as W
from .diagnostics import generic_features
from .matching import caliper_match, displacement, log_profile, set_level_ratios
from .store import READ_POSITIONS, answer_stats, lm_loss, name_fluency, probe, read_states, value_logits


def _query_pass(model, world, items, names=None):
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[items[:, 0]] if names is None else names, items[:, 1]))
    logits, resid = model(toks, collect=True)
    vl = value_logits(logits[:, 4], items[:, 1], world.vocab)
    states = torch.stack([r[:, p] for r in resid for p in READ_POSITIONS], dim=1)
    return torch.log_softmax(vl, -1), states


def _torch_disp(states, H0, sigma_t):
    return torch.sqrt((((states - H0) / sigma_t) ** 2).mean(-1) + 1e-8)      # [n, sites]


class RetainKL:
    """Retention term against a FROZEN copy of the original intact store (reference log-probs precomputed).

    L_R = KL_ans + KL_tok
      KL_ans: mean over a random batch of retain facts of KL(p0(.|q) || p_theta(.|q)), full-vocabulary
              next-token distribution at the [A] position (competence preservation);
      KL_tok: mean over the next-token positions (s1, s2, [E]) of a random batch of mention sequences of
              KL(p0 || p_theta) (name-level behaviour, i.e. operational familiarity).
    Retain pool (frozen design rule): known MT-split facts of all entities, and mention sequences of all
    entities except `exclude_entities`. EV and CT facts never enter the pool, so retained-fact QC (EV Z
    categories, CT near-entity facts) is always computed on held-out items.
    """

    def __init__(self, store0, world, exclude_entities, fact_batch, mention_batch, pool="MT"):
        """pool 'MT' (v2/V0): known MT facts; 'MT+CT' (v3 V2): plus known CT facts of entities NOT in
        exclude_entities (the near-entity QC facts -- CT facts of excluded entities -- stay held out)."""
        facts = W.items_where(world, split=W.MT, known=True)
        if pool == "MT+CT":
            ct = W.items_where(world, split=W.CT, known=True)
            ct = ct[~np.isin(ct[:, 0], np.asarray(exclude_entities, dtype=int))]
            facts = np.concatenate([facts, ct])
        elif pool != "MT":
            raise ValueError(pool)
        self.q_f = torch.as_tensor(W.query_tokens(world.vocab, world.names[facts[:, 0]], facts[:, 1]))
        ents = np.setdiff1d(np.arange(world.n_ent), np.asarray(exclude_entities, dtype=int))
        self.m_in = torch.as_tensor(W.mention_seqs(world.vocab, world.names[ents])[:, :3])   # [M s1 s2] -> s1 s2 E
        self.mention_entities = ents
        store0.eval()
        with torch.no_grad():
            self.lp0_f = torch.log_softmax(store0(self.q_f)[:, 4], -1)
            self.lp0_m = torch.log_softmax(store0(self.m_in), -1) if len(ents) else None
        self.fb, self.mb = int(fact_batch), int(mention_batch)
        self.n_facts, self.n_mention_entities = len(facts), int(len(ents))

    def loss(self, model, g):
        """fact_batch >= pool size means the whole MT pool at every step (no sampling)."""
        if self.fb >= len(self.q_f):
            i = torch.arange(len(self.q_f))
        else:
            i = torch.randint(0, len(self.q_f), (self.fb,), generator=g)
        lp = torch.log_softmax(model(self.q_f[i])[:, 4], -1)
        kl_a = F.kl_div(lp, self.lp0_f[i], log_target=True, reduction="none").sum(-1).mean()
        kl_t = torch.zeros(())
        if self.mb and self.lp0_m is not None:
            j = torch.randint(0, len(self.m_in), (self.mb,), generator=g)
            lpm = torch.log_softmax(model(self.m_in[j]), -1)
            kl_t = F.kl_div(lpm, self.lp0_m[j], log_target=True, reduction="none").sum(-1).mean()
        return kl_a + kl_t, (float(kl_a.detach()), float(kl_t.detach()))


VARIANTS = ("V0", "V1", "V2", "V3")
V3_TRAINABLE = ("blocks.0.fc1.", "blocks.0.fc2.", "blocks.1.fc1.", "blocks.1.fc2.")   # MLPs of blocks 1-2


def forget_with_sham(store0, world, X, Y, fcfg, sigma, seed, mode="uniform"):
    """Returns (model, log). fcfg: steps, lr, gamma, lambda_retain, retain_fact_batch, retain_mention_batch,
    variant (v3 mechanism family; default V0), anchor_lambda (V1).
      V0  v2 mechanism: RetainKL on known MT facts + mentions (X, Y entities excluded)
      V1  V0 + parameter-space anchor  anchor_lambda * sum_p ||theta_p - theta0_p||^2
      V2  V0 with the retain pool enlarged to known MT + CT facts of entities outside X and Y
      V3  V0 with only the MLP weights of blocks 1-2 trainable (localized edit; motivation from
          locate-and-edit work, not proof of the right location in this store)
    Mention retain excludes X and Y entities, so the |fluency change| criterion stays a held-out check.
    The identical procedure is applied to every X item (within-target design, D36)."""
    variant = fcfg.get("variant", "V0")
    assert variant in VARIANTS, variant
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = copy.deepcopy(store0)
    retain = RetainKL(store0, world, np.concatenate([X[:, 0], Y[:, 0]]), fcfg["retain_fact_batch"],
                      fcfg["retain_mention_batch"], pool="MT+CT" if variant == "V2" else "MT")
    if variant == "V3":
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(any(n_.startswith(t) for t in V3_TRAINABLE))
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    theta0 = [p_.detach().clone() for p_ in params] if variant == "V1" else None
    lam_p = float(fcfg.get("anchor_lambda", 0.0)) if variant == "V1" else 0.0
    model.train()
    with torch.no_grad():
        _, H0x = _query_pass(model, world, X)
        _, H0y = _query_pass(model, world, Y)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32)
    V = world.vocab.n_val
    if mode == "replace":
        new_x = (world.answers[X[:, 0], X[:, 1]] + rng.integers(1, V, len(X))) % V
        new_y = (world.answers[Y[:, 0], Y[:, 1]] + rng.integers(1, V, len(Y))) % V
        new_x, new_y = torch.as_tensor(new_x), torch.as_tensor(new_y)
    true_y = torch.as_tensor(world.answers[Y[:, 0], Y[:, 1]])
    opt = torch.optim.Adam(params, lr=fcfg["lr"])
    K = int(fcfg["steps"])
    KA = K // 2
    gamma, lam = fcfg["gamma"], fcfg["lambda_retain"]
    log = []
    g = torch.Generator().manual_seed(seed)
    for step in range(K):
        lpx, Sx = _query_pass(model, world, X)
        lpy, Sy = _query_pass(model, world, Y)
        if mode == "uniform":
            lx = -lpx.mean()
            ly_forget = -lpy.mean()
        else:
            lx = F.nll_loss(lpx, new_x)
            ly_forget = F.nll_loss(lpy, new_y)
        if step < KA:
            ly, lm = ly_forget, torch.zeros(())
        else:
            ly = F.nll_loss(lpy, true_y)
            dx = _torch_disp(Sx, H0x, sigma_t).mean(0)
            dy = _torch_disp(Sy, H0y, sigma_t).mean(0)
            lm = ((dy - dx.detach()) ** 2).sum() / (dx.detach() ** 2).sum().clamp_min(1e-6)
        lr_, (ka, kt) = retain.loss(model, g)
        loss = lx + ly + gamma * lm + lam * lr_
        anc = torch.zeros(())
        if lam_p:
            anc = sum(((p_ - t0) ** 2).sum() for p_, t0 in zip(params, theta0))
            loss = loss + lam_p * anc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, K // 10) == 0 or step == K - 1:
            log.append({"step": step, "phase": "A" if step < KA else "B", "L_x": float(lx.detach()),
                        "L_y": float(ly.detach()), "L_match": float(lm.detach()), "KL_ans": ka, "KL_tok": kt,
                        "anchor": float(anc.detach())})
    for p_ in model.parameters():
        p_.requires_grad_(True)
    model.eval()
    return model, {"trace": log, "mode": mode, "variant": variant,
                   "retain_pool": {"facts": retain.n_facts, "mention_entities": retain.n_mention_entities},
                   "n_trainable": int(sum(p_.numel() for p_ in params))}


def _train_on(store0, seq_fn, steps, lr, seed, world, retain=None, lam=0.0, stop_fn=None, eval_every=0):
    torch.manual_seed(seed)
    model = copy.deepcopy(store0)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(int(steps)):
        loss = lm_loss(model, seq_fn(step, g))
        if retain is not None and lam:
            loss = loss + lam * retain.loss(model, g)[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if stop_fn is not None and eval_every and (step + 1) % eval_every == 0:
            model.eval()
            done, val = stop_fn(model)
            trace.append({"step": step + 1, "stop_metric": val})
            model.train()
            if done:
                break
    model.eval()
    return model, trace


def interference_data(world, icfg):
    """Training sequences of T-INTERF (facts + mentions of the reserved interference entities only; no replay of the
    original knowledge) and the new-fact query bookkeeping (entity index per fact, relation per fact)."""
    names = world.interf_names
    rels = world.interf_rels
    ents = np.repeat(np.arange(len(names)), rels.shape[1])
    rr = rels.reshape(-1)
    facts = W.fact_seqs(world.vocab, names[ents], rr, world.interf_answers[ents, rr])
    ments = W.mention_seqs(world.vocab, np.repeat(names, icfg["mentions_per_entity"], axis=0))
    return np.concatenate([facts, ments]), ents, rr


def interference(store0, world, icfg, seed):
    """T-INTERF: learn facts about new entities with NO replay. Lost/retained EV items emerge naturally."""
    names = world.interf_names
    seqs, ents, rr = interference_data(world, icfg)
    seqs = torch.as_tensor(seqs)
    bs = int(icfg["batch_size"])
    q_items = np.stack([np.zeros(len(ents), int), rr], axis=1)

    def seq_fn(step, g):
        return seqs[torch.randint(0, len(seqs), (bs,), generator=g)]

    def stop_fn(model):
        acc = answer_stats(model, world, q_items, names=names[ents], answers=world.interf_answers[ents, rr])["correct"].mean()
        return acc >= icfg["stop_new_fact_acc"], float(acc)

    return _train_on(store0, seq_fn, icfg["max_steps"], icfg["lr"], seed, world, None, 0.0, stop_fn, icfg["eval_every"])


def learn_new(store0, world, N, U, ncfg, seed):
    """T-NEW: learn N's facts; U's entities receive matched mention presentations (exposure matching).
    Retain: RetainKL with mentions of N and U entities excluded (their exposure is the manipulation)."""
    facts = torch.as_tensor(W.fact_seqs(world.vocab, world.names[N[:, 0]], N[:, 1], world.answers[N[:, 0], N[:, 1]]))
    ments = torch.as_tensor(W.mention_seqs(world.vocab, world.names[U[:, 0]]))
    both = torch.cat([facts, ments])
    retain = RetainKL(store0, world, np.concatenate([N[:, 0], U[:, 0]]), ncfg["retain_fact_batch"],
                      ncfg["retain_mention_batch"])
    return _train_on(store0, lambda s, g: both, ncfg["steps"], ncfg["lr"], seed, world, retain, ncfg["lambda_retain"])


def familiarity_boost(store0, world, Fset, fcfg, seed, control=None):
    """T-FAM: extra mention presentations for F's entities only (no fact training).
    Retain: RetainKL with mentions of F and control (U2) entities excluded."""
    ments = torch.as_tensor(W.mention_seqs(world.vocab, world.names[Fset[:, 0]]))
    excl = Fset[:, 0] if control is None else np.concatenate([Fset[:, 0], control[:, 0]])
    retain = RetainKL(store0, world, excl, fcfg["retain_fact_batch"], fcfg["retain_mention_batch"])
    return _train_on(store0, lambda s, g: ments, fcfg["steps"], fcfg["lr"], seed, world, retain, fcfg["lambda_retain"])


# ---------------------------------------------------------------- T-ACT (exploratory, inference-time)
def act_knockout_fn(knock_layers, n_total):
    """Block attention from [A] (pos 4) to s1, s2 (pos 1, 2) at the given layers, for all rows."""
    def fn(start, n):
        m = torch.zeros(n, 5, 5, dtype=torch.bool)
        m[:, 4, 1] = True
        m[:, 4, 2] = True
        return (set(knock_layers), m)
    return fn


def act_sham_add_fn(model, world, Y, layer, norm, seed):
    """Matched-norm residual perturbation at (layer, pos 4), orthogonal to the gradient of the
    correct-answer margin (so it is designed to preserve competence)."""
    g = torch.Generator().manual_seed(seed)
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[Y[:, 0]], Y[:, 1]))
    delta = torch.zeros(len(Y), 5, model.tok.embedding_dim, requires_grad=True)
    logits = model(toks, add={layer: delta})
    lp = torch.log_softmax(value_logits(logits[:, 4], Y[:, 1], world.vocab), -1)
    ans = torch.as_tensor(world.answers[Y[:, 0], Y[:, 1]])
    lp.gather(1, ans[:, None]).sum().backward()
    grad = delta.grad[:, 4].detach()
    r = torch.randn(grad.shape, generator=g)
    gn = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = r - (r * gn).sum(-1, keepdim=True) * gn
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12) * norm
    full = torch.zeros(len(Y), 5, grad.shape[-1])
    full[:, 4] = r

    def fn(start, n):
        return {layer: full[start:start + n]}
    return fn


# ---------------------------------------------------------------- P-family (developmental perturbation)
def sample_dropout_rates(n, n_layers, pcfg, rng):
    """Each row: each block output included w.p. layer_inclusion_prob (>= min_layers); rate ~ U(lo, hi)."""
    lo, hi = pcfg["rate"]
    inc = rng.random((n, n_layers)) < pcfg["layer_inclusion_prob"]
    empty = ~inc.any(1)
    inc[empty, rng.integers(0, n_layers, empty.sum())] = True
    rates = rng.uniform(lo, hi, (n, 1)) * inc
    return torch.as_tensor(rates, dtype=torch.float32)


# ---------------------------------------------------------------- QC (store-level only)
Z_CATEGORIES = ("Z_random", "Z_near_entity_X", "Z_near_entity_Y", "Z_near_repr", "Z_far")


def near_entity_facts(store0, world, entities):
    """Known, intact-correct CT-split facts of the given entities (held out of every retain pool)."""
    ct = W.items_where(world, split=W.CT, known=True)
    ct = ct[np.isin(ct[:, 0], np.asarray(entities, dtype=int))]
    return ct[answer_stats(store0, world, ct)["correct"]] if len(ct) else ct


def forget_qc_sets(store0, world, sets, sigma, near_repr_fraction):
    """Pre-declared retained-fact QC categories for T-FORGET (no category item is targeted or retained-trained):
      Z_random        : all base-correct EV items not in X or Y (the v1 'Z');
      Z_near_entity_X : known, intact-correct CT facts of X's entities (same entity, other relation);
      Z_near_entity_Y : the same for Y's entities (sham side);
      Z_near_repr     : the near_repr_fraction of Z_random most similar to X in intact read-state space
                        (max cosine over X items of the centred, sigma-standardised 10-site read state);
      Z_far           : the near_repr_fraction of Z_random least similar to X by the same measure.
    Diagnostic only (inside the retain pool): R_sibling_X = known MT facts of X's entities."""
    X, Y, Z = sets["X"], sets["Y"], sets["Z"]
    HX, HY, HZ = (read_states(store0, world, v) for v in (X, Y, Z))
    ref = np.concatenate([HX, HY, HZ]).mean(0)

    def unit(H):
        f = ((H - ref) / sigma).reshape(len(H), -1)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    sim = (unit(HZ) @ unit(HX).T).max(1)
    k = max(1, int(round(near_repr_fraction * len(Z))))
    order = np.argsort(-sim, kind="stable")
    mt = W.items_where(world, split=W.MT, known=True)
    out = {"Z_random": Z, "Z_near_entity_X": near_entity_facts(store0, world, X[:, 0]),
           "Z_near_entity_Y": near_entity_facts(store0, world, Y[:, 0]),
           "Z_near_repr": Z[order[:k]], "Z_far": Z[order[-k:]],
           "R_sibling_X": mt[np.isin(mt[:, 0], X[:, 0])]}
    return out, {"sim_near_mean": float(sim[order[:k]].mean()), "sim_far_mean": float(sim[order[-k:]].mean())}


def _kl_rows(p_pre, p_post):
    return (p_pre * (np.log(np.maximum(p_pre, 1e-12)) - np.log(np.maximum(p_post, 1e-12)))).sum(-1)


def _set_report(pr0, pr1, d):
    """Before/after competence, output-distribution change and read-site displacement for one item set."""
    if len(pr0["correct"]) == 0:
        return {"n": 0}
    pre_ok = pr0["correct"]
    return {"n": int(len(pre_ok)), "acc_pre": float(pre_ok.mean()), "acc_post": float(pr1["correct"].mean()),
            "lost_frac": float((pre_ok & ~pr1["correct"]).sum() / max(1, pre_ok.sum())),
            "margin_change_mean": float((pr1["margin"] - pr0["margin"]).mean()),
            "logp_correct_change_mean": float((pr1["logp_correct"] - pr0["logp_correct"]).mean()),
            "kl_answer_dist_mean": float(_kl_rows(pr0["probs"], pr1["probs"]).mean()),
            "disp_mean": float(d.mean()), "disp_per_site": [round(float(x), 5) for x in d.mean(0)]}


def natural_gap(store0, world):
    """D_C: mean margin of base-correct known EV minus mean margin of unknown EV in the intact store (nats)."""
    ev = W.items_where(world, split=W.EV)
    pr = answer_stats(store0, world, ev)
    tr = world.known[ev[:, 0], ev[:, 1]]
    return float(pr["margin"][tr & pr["correct"]].mean() - pr["margin"][~tr & ~pr["correct"]].mean())


def item_pre_covariates(store0, world, items, pr0=None):
    """W0 (pre-intervention, store-only): log p(v*) pre, exposure class, name fluency, relation dummies.
    Returns (W0 [n,6], C_pre [n])."""
    pr0 = pr0 if pr0 is not None else answer_stats(store0, world, items)
    flu = name_fluency(store0, world, world.names[items[:, 0]])
    rel = np.eye(world.n_rel)[items[:, 1]][:, 1:]
    W0 = np.column_stack([pr0["logp_correct"], world.fam_high[items[:, 0]].astype(float), flu, rel])
    return W0, pr0["margin"]


def generic40(H_pre, H_post, mu, sigma):
    """Pre-declared generic post-change feature set (40): per-site log displacement, cos(delta, h_pre),
    relative norm change, abnormality change (s1/diagnostics G2 + G3)."""
    g = generic_features(H_pre, H_post, mu, sigma)
    return np.concatenate([g["G2"], g["G3"]], 1)


def identifiability(dC, H_pre, W0, C_pre, G, seed, strata=None):
    """Within-population store-only diagnostics (failure criteria F3-F5): cross-fitted R^2 of the competence
    change from generic post-change features (r2_gen), from pre-intervention information incl. pre-states (r2_pre),
    and from both (r2_joint). Also returns the cross-fitted pre-information prediction P (susceptibility score)."""
    Hf = H_pre.reshape(len(H_pre), -1)
    Xpre = np.column_stack([Hf, W0, C_pre])
    P, r2_pre, _ = ES.crossfit_ridge(Xpre, dC, seed, strata)
    _, r2_gen, _ = ES.crossfit_ridge(G, dC, seed + 1, strata)
    _, r2_joint, _ = ES.crossfit_ridge(np.column_stack([Xpre, G]), dC, seed + 2, strata)
    return {"r2_pre": r2_pre, "r2_gen": r2_gen, "r2_joint": r2_joint}, P


def retention_continuous(rep, X_rep, D_C, levels):
    """Continuous retention (v3 Sec.7) for one category: ratios to X's mean change and absolute margin anchor.
    load = max(metric / threshold) at the primary level 0.10 (pass iff load <= 1); pass reported at each level."""
    if not rep.get("n"):
        return {"n": 0, "load": 0.0, "pass": {str(l): True for l in levels}}
    rm = abs(rep["margin_change_mean"]) / max(abs(X_rep["margin_change_mean"]), 1e-9)
    rl = abs(rep["logp_correct_change_mean"]) / max(abs(X_rep["logp_correct_change_mean"]), 1e-9)
    rk = rep["kl_answer_dist_mean"] / max(X_rep["kl_answer_dist_mean"], 1e-12)
    ra = abs(rep["margin_change_mean"]) / D_C
    out = {"ratio_margin": rm, "ratio_logp": rl, "ratio_kl": rk, "abs_margin_over_DC": ra,
           "rel_margin_change": rep["margin_change_mean"] / max(rep.get("margin_pre_mean", np.nan), 1e-9)}
    out["load"] = float(max(rm, rl, rk, ra) / 0.10)
    out["pass"] = {str(l): bool(max(rm, rl, rk, ra) <= l) for l in levels}
    return out


def forget_qc(store0, model, world, sets, sigma, qcfg, calipers, seed, fluency_sd, mu):
    """T-FORGET QC (v3, D38). Per-seed store-level gates:
      X_diversity  X lost fraction in X_lost_range and IQR(C_post over X) >= C_post_iqr_min
      Y_binary     Y lost <= Y_lost_max;   Z_binary  lost <= Z_lost_max in each Z category
      retention    continuous retention (ratios to X <= 0.10 for margin, log p, KL; |dmargin| <= 0.10 D_C) for Y and
                   every Z category
      sites        >= sites_min site ratios D_X/D_Y in [ratio_lo, ratio_hi] (Y displacement match; negative control)
      fluency      |fluency change of X, Y entities| <= familiarity_change_max_sd
    'pass' = all of the above (used by the per-seed fallback ladder). Also reported (gated at calibration level):
    F6 binary-secondary feasibility, F3-F5 identifiability, Y fingerprint diagnostics, sensitivity levels."""
    X, Y = sets["X"], sets["Y"]
    cats, sim_info = forget_qc_sets(store0, world, sets, sigma, qcfg["near_repr_fraction"])
    allsets = {"X": X, "Y": Y, **cats}
    res, pr = {"sets": {}}, {}
    for name, items in allsets.items():
        if len(items) == 0:
            res["sets"][name] = {"n": 0}
            continue
        p0, p1 = probe(store0, world, items), probe(model, world, items)
        d = displacement(p0["states"], p1["states"], sigma)
        res["sets"][name] = _set_report(p0, p1, d)
        res["sets"][name]["margin_pre_mean"] = float(p0["margin"].mean())
        pr[name] = (p0, p1, d)
    res["representational_similarity"] = sim_info
    p0X, p1X, dX = pr["X"]
    p0Y, p1Y, dY = pr["Y"]
    lostX = ~p1X["correct"]
    ycor = p1Y["correct"]
    res["X_lost_frac"] = res["sets"]["X"]["lost_frac"]
    res["Y_lost_frac"] = res["sets"]["Y"]["lost_frac"]
    res["Z_lost_frac"] = res["sets"]["Z_random"]["lost_frac"]
    zl = {c: res["sets"][c].get("lost_frac", 0.0) for c in Z_CATEGORIES}
    res["Z_lost_by_category"] = zl
    res["Z_lost_max_over_categories"] = float(max(zl.values()))
    q = np.quantile(p1X["margin"], [0.25, 0.75])
    res["C_post_X_iqr"] = float(q[1] - q[0])
    D_C = natural_gap(store0, world)
    res["D_C"] = D_C
    levels = qcfg["retention_sensitivity"]
    res["retention"] = {c: retention_continuous(res["sets"][c], res["sets"]["X"], D_C, levels) for c in ("Y",) + Z_CATEGORIES}
    res["retention_load_max"] = float(max(v["load"] for v in res["retention"].values()))
    res["retention_pass_by_level"] = {str(l): bool(all(v["pass"][str(l)] for v in res["retention"].values())) for l in levels}
    ratios = set_level_ratios(dX, dY)
    res["site_ratios_X_over_Y"] = ratios.tolist()
    res["sites_within_tol"] = int(((ratios >= qcfg["ratio_lo"]) & (ratios <= qcfg["ratio_hi"])).sum())
    res["max_abs_log_ratio"] = float(np.max(np.abs(np.log(np.maximum(ratios, 1e-12)))))
    pX, pY = log_profile(dX), log_profile(dY)
    res["matched_pairs_Xlost_Y"] = {str(c): len(caliper_match(pX[lostX], pY[ycor], c, seed)) for c in calipers}
    nm = world.names[np.concatenate([X[:, 0], Y[:, 0]])]
    res["fluency_change_sd"] = float((name_fluency(model, world, nm) - name_fluency(store0, world, nm)).mean() / fluency_sd)
    # within-X identifiability (F3-F5), fingerprint-vs-outcome, binary-secondary feasibility (F6)
    dC = p1X["margin"] - p0X["margin"]
    W0, C_pre = item_pre_covariates(store0, world, X, p0X)
    GX, GY = generic40(p0X["states"], p1X["states"], mu, sigma), generic40(p0Y["states"], p1Y["states"], mu, sigma)
    idf, P = identifiability(dC, p0X["states"], W0, C_pre, GX, seed + 101, strata=W0[:, 1])
    res["identifiability"] = idf
    fX, _ = ES.crossfit_logistic_scores(GX, GY, seed + 202)
    res["fingerprint_vs_dC_spearman"] = float(stats.spearmanr(fX, dC)[0])
    bs = qcfg["binary_secondary"]
    Xpre = np.column_stack([C_pre, W0, P])
    pairs, smd = ES.propensity_match(lostX, Xpre, W0[:, 1], seed + 303, bs["caliper_sd"])
    res["binary_secondary"] = {"pairs": len(pairs), "max_abs_smd": smd,
                               "feasible": bool(len(pairs) >= bs["min_pairs"] and smd <= bs["smd_max"])}
    # F6' (pre-declared ALTERNATIVE, recorded only; the approved F6 above stays binding -- D43): exact matching on
    # exposure x relation; balance = max |SMD| <= 0.25 and mean |SMD| <= 0.10 (sampling-noise-calibrated)
    pairs2, _ = ES.propensity_match(lostX, Xpre, (W0[:, 1] * world.n_rel + X[:, 1]).astype(int), seed + 303, bs["caliper_sd"])
    mx2, mn2 = ES.balance(Xpre, pairs2)
    res["binary_secondary_alt"] = {"pairs": len(pairs2), "max_abs_smd": mx2, "mean_abs_smd": mn2,
                                   "feasible": bool(len(pairs2) >= bs["min_pairs"] and mx2 <= 0.25 and mn2 <= 0.10)}
    lo, hi = qcfg["X_lost_range"]
    gates = {"X_diversity": bool(lo <= res["X_lost_frac"] <= hi and res["C_post_X_iqr"] >= qcfg["C_post_iqr_min"]),
             "Y_binary": res["Y_lost_frac"] <= qcfg["Y_lost_max"],
             "Z_binary": res["Z_lost_max_over_categories"] <= qcfg["Z_lost_max"],
             "retention": res["retention_pass_by_level"]["0.1"],
             "sites": res["sites_within_tol"] >= qcfg["sites_min"],
             "fluency": abs(res["fluency_change_sd"]) <= qcfg["familiarity_change_max_sd"]}
    res["gates"] = {k: bool(v) for k, v in gates.items()}
    res["failed_gates"] = [k for k, v in gates.items() if not v]
    res["pass"] = bool(all(gates.values()))
    return res, {"dX": dX, "dY": dY, "lostX": lostX, "postY_correct": ycor, "pre_X": p0X, "post_X": p1X,
                 "pre_Y": p0Y, "post_Y": p1Y, "GX": GX, "GY": GY}


def interference_qc(store0, model, world, base_correct_ev, sigma, qcfg, calipers, seed, mu=None):
    """T-INTERF QC (v3): lost fraction in lost_range (per-seed gate). Reported: matched lost/retained pairs
    (descriptive secondary) and, if mu is given, P2 identifiability diagnostics (F3-F5 on this population)."""
    p0, p1 = probe(store0, world, base_correct_ev), probe(model, world, base_correct_ev)
    lost = ~p1["correct"]
    res = {"lost_frac": float(lost.mean())}
    d = displacement(p0["states"], p1["states"], sigma)
    p = log_profile(d)
    res["matched_pairs"] = {str(c): len(caliper_match(p[lost], p[~lost], c, seed)) for c in calipers}
    res["site_ratios_lost_over_retained"] = set_level_ratios(d[lost], d[~lost]).tolist() if lost.any() else None
    res["post_max_prob_lost_mean"] = float(p1["probs"][lost].max(-1).mean()) if lost.any() else None
    res["margin_change_mean"] = float((p1["margin"] - p0["margin"]).mean())
    q = np.quantile(p1["margin"], [0.25, 0.75])
    res["C_post_iqr"] = float(q[1] - q[0])
    if mu is not None:
        W0, C_pre = item_pre_covariates(store0, world, base_correct_ev, p0)
        G = generic40(p0["states"], p1["states"], mu, sigma)
        res["identifiability"], _ = identifiability(p1["margin"] - p0["margin"], p0["states"], W0, C_pre, G,
                                                    seed + 101, strata=W0[:, 1])
    res["pass"] = bool(qcfg["lost_range"][0] <= res["lost_frac"] <= qcfg["lost_range"][1])
    return res, {"d": d, "lost": lost, "pre": p0, "post": p1}


def _collateral_continuous(store0, model, world, items, D_C):
    a0, a1 = answer_stats(store0, world, items), answer_stats(model, world, items)
    dm = float((a1["margin"] - a0["margin"]).mean())
    return {"lost": float((a0["correct"] & ~a1["correct"]).sum() / max(1, a0["correct"].sum())),
            "margin_change_mean": dm, "abs_margin_over_DC": abs(dm) / D_C,
            "kl_mean": float(_kl_rows(a0["probs"], a1["probs"]).mean())}


def new_qc(store0, model, world, N, U, base_correct_ev, qcfg):
    """T-NEW QC (v3): N learned; U not learned; held-out collateral on base-correct EV and near-entity CT facts of
    N's entities, binary (<= collateral_max) AND continuous (|mean dmargin| <= collateral_abs_frac_DC * D_C)."""
    a_n = answer_stats(model, world, N)["correct"].mean()
    a_u = answer_stats(model, world, U)["correct"].mean()
    D_C = natural_gap(store0, world)
    cb = _collateral_continuous(store0, model, world, base_correct_ev, D_C)
    near = near_entity_facts(store0, world, N[:, 0])
    cn = _collateral_continuous(store0, model, world, near, D_C) if len(near) else {"lost": 0.0, "abs_margin_over_DC": 0.0}
    f = qcfg["collateral_abs_frac_DC"]
    return {"N_learned_frac": float(a_n), "U_correct_frac": float(a_u), "collateral_bc": cb, "collateral_near_entity": cn,
            "collateral_bc_lost": cb["lost"], "collateral_near_entity_lost": cn["lost"], "n_near_entity": int(len(near)),
            "D_C": D_C,
            "pass": bool(a_n >= qcfg["N_learned_min"] and a_u <= qcfg["U_correct_max"]
                         and cb["lost"] <= qcfg["collateral_max"] and cn["lost"] <= qcfg["collateral_max"]
                         and cb["abs_margin_over_DC"] <= f and cn["abs_margin_over_DC"] <= f)}


def fam_qc(store0, model, world, Fset, U2, base_correct_ev, qcfg, fluency_sd):
    """T-FAM QC (v3): familiarity rise vs U2; no fact competence created; held-out collateral on base-correct EV,
    binary and continuous (|mean dmargin| <= collateral_abs_frac_DC * D_C)."""
    rise_f = name_fluency(model, world, world.names[Fset[:, 0]]) - name_fluency(store0, world, world.names[Fset[:, 0]])
    rise_u = name_fluency(model, world, world.names[U2[:, 0]]) - name_fluency(store0, world, world.names[U2[:, 0]])
    rel = float((rise_f.mean() - rise_u.mean()) / fluency_sd)
    acc_f = float(answer_stats(model, world, Fset)["correct"].mean())
    D_C = natural_gap(store0, world)
    cb = _collateral_continuous(store0, model, world, base_correct_ev, D_C)
    return {"fluency_rise_rel_sd": rel, "F_correct_frac": acc_f, "collateral_bc": cb, "collateral_bc_lost": cb["lost"],
            "U2_fluency_change_sd": float(rise_u.mean() / fluency_sd), "D_C": D_C,
            "pass": bool(rel >= qcfg["familiarity_rise_min_sd"] and acc_f <= qcfg["F_correct_max"]
                         and cb["lost"] <= qcfg["collateral_max"] and cb["abs_margin_over_DC"] <= qcfg["collateral_abs_frac_DC"])}
