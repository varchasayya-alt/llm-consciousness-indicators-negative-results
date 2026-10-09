"""Store-only mechanism diagnostic for a v4.1 store (no monitor; nothing is selected from it).
Usage: python diag_v41_store.py <store.pt> <seed> [out.json]
Reports, for memory-covered and parametric-only trained facts: accuracy by route (integrated / all slots masked),
retrieval (own-slot, NULL, leakage), magnitude of the injected memory term u = W_o r relative to the pre-injection
state hA, M2 (is answer information in u?), and how strongly the downstream (blocks 3-4 + readout) responds to the
memory channel: directional derivative of the answer margin along u, versus along hA, at [A] after block 2."""
import json
import os
import sys

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from s1 import world as W  # noqa: E402
from s1.config import derive_seed, load_config  # noqa: E402
from s1.memstore import A_POS, all_masked, build_dual_store, m2_probe, memory_probe  # noqa: E402
from s1.store import answer_stats, value_logits  # noqa: E402




def directional(model, world, items, n=600):
    """d margin / d eps for h[A] <- hA + u + eps * dir, dir in {u/|u|, hA/|hA|, random unit}; mean |derivative|."""
    it = items[:n]
    toks = torch.as_tensor(W.query_tokens(world.vocab, world.names[it[:, 0]], it[:, 1]))
    ans = torch.as_tensor(world.answers[it[:, 0], it[:, 1]])
    out = {}
    with torch.no_grad():
        h = model.tok(toks) + model.pos(torch.arange(toks.shape[1]))
        for l, blk in enumerate(model.blocks, start=1):
            h = blk(h)
            if l == model.inject_after:
                break
        lg = model.retrieval_logits(h[:, 1:4])
        u = model.W_o(torch.softmax(lg, -1) @ model.mem_values)
        hA = h[:, A_POS].clone()
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(hA.shape, generator=g)
    dirs = {"along_u": u / u.norm(dim=1, keepdim=True), "along_hA": hA / hA.norm(dim=1, keepdim=True),
            "random": rnd / rnd.norm(dim=1, keepdim=True)}
    for name, d in dirs.items():
        eps = torch.zeros(len(it), 1, requires_grad=True)
        hh = h.clone()
        hh[:, A_POS] = hA + u + eps * d
        x = hh
        for l, blk in enumerate(model.blocks, start=1):
            if l <= model.inject_after:
                continue
            x = blk(x)
        logits = model.unembed(model.ln_f(x))[:, A_POS]
        lp = torch.log_softmax(value_logits(logits, it[:, 1], world.vocab), -1)
        lpc = lp.gather(1, ans[:, None]).squeeze(1)
        other = lp.clone()
        other.scatter_(1, ans[:, None], float("-inf"))
        marg = lpc - other.max(-1).values
        (gr,) = torch.autograd.grad(marg.sum(), eps)
        out[name] = float(gr.abs().mean())
    out["u_norm_mean"] = float(u.norm(dim=1).mean())
    out["hA_norm_mean"] = float(hA.norm(dim=1).mean())
    return out


def main(path, seed, out_path=None):
    torch.set_num_threads(int(os.environ.get("S1_THREADS", "2")))
    cfg = load_config(os.path.join(STAGE, "stage1_config.yaml"))
    world = W.make_world(cfg["world"], derive_seed(seed, "world"))
    m = build_dual_store(cfg["store"], cfg["memory"], world)
    m.load_state_dict(torch.load(path))
    m.eval()
    kn = W.items_where(world, known=True)
    rng = np.random.default_rng(0)
    cov = kn[world.mem_covered[kn[:, 0], kn[:, 1]]]
    par = kn[~world.mem_covered[kn[:, 0], kn[:, 1]]]
    cov, par = cov[rng.permutation(len(cov))[:3000]], par[rng.permutation(len(par))[:3000]]
    A = all_masked(m)
    res = {}
    for name, it in (("covered", cov), ("param_only", par)):
        mp = memory_probe(m, world, it)
        r = {"acc_integrated": float(answer_stats(m, world, it)["correct"].mean()),
             "acc_route_A": float(answer_stats(A, world, it)["correct"].mean()),
             "a_null_mean": float(mp["a_null"].mean()), "a_max_mean": float(mp["a_max"].mean())}
        if name == "covered":
            r["a_own_mean"] = float(np.nanmean(mp["a_own"]))
            r["leak_mass_mean"] = float(1 - np.nanmean(mp["a_own"]) - mp["a_null"].mean())
        else:
            r["leak_mass_mean"] = float(1 - mp["a_null"].mean())
        r["M2_answer_from_u"] = m2_probe(m, world, it[:2000], seed=1)
        r["downstream_sensitivity"] = directional(m, world, it)
        res[name] = r
    res["W_o_fro"] = float(m.W_o.weight.norm())
    res["mem_values_norm_mean"] = float(m.mem_values[:-1].norm(dim=1).mean())
    res["null_value_norm"] = float(m.mem_values[-1].norm())
    res["log_scale"] = float(m.log_scale.exp())
    print(json.dumps(res, indent=1))
    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
