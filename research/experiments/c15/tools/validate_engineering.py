"""Engineering validation on SMOKE material only (non-evidential): GP screening fidelity vs exact GP, and
cached-prefix empirical KL vs an uncached forward. Writes results/raw/c15a/engineering/validate_<model>.json."""
import copy, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
torch.set_grad_enabled(False)
from run_astage import load_all, dump, ENG, provenance
from c15a import config as C
from c15a.assays import AStage
from c15a.gp import gp_nonneg
from c15a.hooks import add_at, forward
from c15a.materials import load_smoke

key = sys.argv[1]
lm, lens, _ = load_all(key)
sm = load_smoke()
band = __import__("c15a.workspace", fromlist=["band_layers"]).band_layers(lm.n_layers)
st = AStage(lm, lens, sm, layers=[band[len(band) // 2]])
st.prepare()
l = st.layers[0]
D = st.dicts(l)
sl = slice(C.STAT_POS_MIN, C.SEQ_LEN)
H = torch.cat([r[sl] for r in st.clean_text_rec[l]]).float()[:256]
t0 = time.time(); i0, c0, r0 = gp_nonneg(H, D, C.GP_K); t_exact = time.time() - t0
t0 = time.time(); i1, c1, r1 = gp_nonneg(H, D, C.GP_K, screen=C.GP_SCREEN); t_scr = time.time() - t0
res0 = ((H - r0) ** 2).sum(1); res1 = ((H - r1) ** 2).sum(1)
overlap = [len(set(a[a >= 0].tolist()) & set(b[b >= 0].tolist())) / max(1, len(set(a[a >= 0].tolist()))) for a, b in zip(i0, i1)]
top10 = []
from c15a.gp import top_atoms
for a, b in zip(top_atoms(i0, c0, 10), top_atoms(i1, c1, 10)):
    top10.append(len(set(a.tolist()) & set(b.tolist())) / 10)
out = {"gp": {"n": int(H.shape[0]), "sec_exact": t_exact, "sec_screened": t_scr,
              "rel_residual_ratio_mean": float((res1 / res0).mean()), "rel_residual_ratio_p95": float((res1 / res0).quantile(0.95)),
              "atom_set_overlap_mean": float(sum(overlap) / len(overlap)), "top10_overlap_mean": float(sum(top10) / len(top10)),
              "recon_cos_mean": float(torch.nn.functional.cosine_similarity(r0, r1, dim=1).mean())}}
# ablation-level equivalence (the quantity W4/W5 use): removed component, exact vs screened
from c15a.hooks import JAblation
hh = torch.stack(st.clean_text_rec[l][:4]).float()
def _removed(screen):
    ab = JAblation(D, C.GP_K, C.ABL_TOP, screen=screen); o = ab(hh); return (hh - o)[:, 1:].reshape(-1, hh.shape[-1])
pe, ps = _removed(None), _removed(C.GP_SCREEN)
out["ablation"] = {"removed_cos_mean": float(torch.nn.functional.cosine_similarity(pe, ps, dim=1).mean()),
                   "removed_norm_ratio_mean": float((ps.norm(dim=1) / pe.norm(dim=1)).mean())}
# cached vs uncached empirical KL
u = torch.randn(lm.d_model); u = u / u.norm(); nv = st.hbar[l]
ids, past, lp0 = st._cal()
kl_c = st.emp_kl(l, u, nv)
full = ids
h0, _ = forward(lm, full)
V = (nv * u)[None].repeat(full.shape[0], 1)
h1, _ = forward(lm, full, edits={l: add_at(C.DOSE_POS, V)})
a = torch.log_softmax(st.logits(h0[:, C.DOSE_POS + 1:]), -1); b = torch.log_softmax(st.logits(h1[:, C.DOSE_POS + 1:]), -1)
kl_u = float((a.exp() * (a - b)).sum(-1).mean())
out["emp_kl_cache_check"] = {"cached": kl_c, "uncached": kl_u, "abs_diff": abs(kl_c - kl_u)}
out["pass"] = (out["gp"]["rel_residual_ratio_mean"] <= 1.02 and out["ablation"]["removed_cos_mean"] >= 0.99
               and abs(out["ablation"]["removed_norm_ratio_mean"] - 1) <= 0.02
               and abs(kl_c - kl_u) <= 1e-3 * max(1.0, kl_u) + 1e-5)
out["non_evidential"] = True
out["provenance"] = provenance("SMOKE", key)
dump(out, os.path.join(ENG, f"validate_{key}.json"))
print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))
