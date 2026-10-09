"""C15-R2 final W3' simulation (synthetic, NO model runs).

W3' (frozen design): identity-specific transport.
  * Natural reference F_k: downstream footprint of PARAPHRASE mentions of concept k (never the word itself),
    centred across concepts within each context; estimated on context fold A from paraphrases p1, p2.
  * Benchmark: held-out paraphrase p3 on fold B -> natural identification accuracy acc_nat.
  * Injection effects D_ck: injected route-R content for concept k at a neutral slot, centred across the injected
    concepts within the same context (generic influence common to all injections cancels exactly).
  * Identification: argmax_k' cos(D_ck, F_hat_k') == k  (8-way; chance 0.125).
  * Site criterion: acc_R >= max(0.30, 0.5 * acc_nat) AND one-sided binomial p < 0.05 vs 1/8.

Model of the downstream signal (unit-scale, d = 2048):
  conceptual footprint direction f_k (concept-specific); lexical-surface direction l_k (word-specific), with
  cos(l_k, f_k) = sqrt(rho_lc); generic route influence g (common to all concepts -> removed by centring);
  context heterogeneity gamma * e_ck.
  Injection:   D_ck = phi_c f_k + phi_l l_k + g + gamma e_ck
  Paraphrase reference / benchmark: f_k + gamma_nat e  (+ estimation noise from n_ref samples)
  Word-mention reference (diagnostic only): l_k + f_k-part ... used to show the lexical-carryover hazard.
"""
import json
import math
import sys

import numpy as np
from scipy import stats

rng = np.random.default_rng(9502)
D = 2048
K = 8
N_CTX = 24          # injection fold
N_REF = 48          # paraphrase-reference samples per concept (2 paraphrases x 24 contexts)
N_BENCH = 24        # held-out paraphrase benchmark samples per concept


def unit(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def centre(X):          # X: [K, n, D] -> subtract mean over concepts (within context index)
    return X - X.mean(0, keepdims=True)


def make_world(rho_lc):
    f = unit(rng.standard_normal((K, D)))
    lo = unit(rng.standard_normal((K, D)))
    lo = unit(lo - (lo * f).sum(1, keepdims=True) * f)
    l = unit(math.sqrt(rho_lc) * f + math.sqrt(1 - rho_lc) * lo)
    g = unit(rng.standard_normal(D))
    return f, l, g


def samples(base, n, gamma):
    """base: [K, D] signal per concept -> [K, n, D] samples with context heterogeneity gamma (unit-norm noise)."""
    noise = rng.standard_normal((K, n, D)) / math.sqrt(D)
    return base[:, None, :] + gamma * noise


def ident_acc(Dk, Fh):
    """Dk: [K, n, D] centred effects; Fh: [K, D] references -> identification accuracy."""
    Fn = unit(Fh)
    cos = np.einsum("knd,jd->knj", unit(Dk), Fn)
    return float((cos.argmax(-1) == np.arange(K)[:, None]).mean())


def site_pass(acc, acc_nat, n):
    k = round(acc * n)
    p = stats.binom.sf(k - 1, n, 1.0 / K)
    return (acc >= max(0.30, 0.5 * acc_nat)) and (p < 0.05)


def one_site(phi_c, phi_l, gamma, gamma_nat, rho_lc, ref="paraphrase"):
    f, l, g = make_world(rho_lc)
    # references
    if ref == "paraphrase":
        R = samples(f, N_REF, gamma_nat)
    else:                                   # word-mention reference: lexical surface + concept
        R = samples(unit(l + f), N_REF, gamma_nat)
    Fh = centre(R).mean(1)
    Bn = centre(samples(f, N_BENCH, gamma_nat))
    acc_nat = ident_acc(Bn, Fh)
    base = phi_c * f + phi_l * l + g[None, :]
    Dk = centre(samples(base, N_CTX, gamma))
    acc = ident_acc(Dk, Fh)
    return acc, acc_nat, site_pass(acc, acc_nat, K * N_CTX)


def grid(reps=200):
    out = {}
    for gamma_nat in (1.0, 3.0):
        for gamma in (1.0, 3.0, 10.0):
            for phi_c in (0.0, 0.05, 0.1, 0.2, 0.3):
                r = [one_site(phi_c, 0.0, gamma, gamma_nat, 0.3) for _ in range(reps)]
                out[f"gnat={gamma_nat},gamma={gamma},phi_c={phi_c}"] = {
                    "p_site_pass": float(np.mean([x[2] for x in r])),
                    "mean_acc": float(np.mean([x[0] for x in r])), "mean_acc_nat": float(np.mean([x[1] for x in r]))}
    return out


def lexical_hazard(reps=200):
    """Pure lexical carryover (phi_c = 0, phi_l = 0.5): does it pass with a word-mention reference vs the
    paraphrase reference? rho_lc = cosine^2 between the lexical-surface and conceptual footprints."""
    out = {}
    for rho_lc in (0.0, 0.1, 0.3, 0.5):
        for ref in ("word", "paraphrase"):
            r = [one_site(0.0, 0.5, 3.0, 1.0, rho_lc, ref=ref) for _ in range(reps)]
            out[f"rho_lc={rho_lc},ref={ref}"] = {"p_site_pass": float(np.mean([x[2] for x in r])),
                                                 "mean_acc": float(np.mean([x[0] for x in r]))}
    return out


def generic_only(reps=400):
    """Generic downstream influence only (phi_c = phi_l = 0, strong common g): false-positive rate."""
    r = [one_site(0.0, 0.0, 3.0, 1.0, 0.3) for _ in range(reps)]
    return {"p_site_pass": float(np.mean([x[2] for x in r])), "mean_acc": float(np.mean([x[0] for x in r]))}


def breadth_gate(reps=2000, n_sites=80):
    """Breadth gate BB_J - BB_perp >= 0.15 with a bootstrap-style lower bound > 0, assuming site passes are
    Bernoulli(p_J), Bernoulli(p_perp) with within-model site correlation via a shared beta-distributed rate
    (design effect ~ 3). Reports P(gate passes)."""
    out = {}
    for pJ, pP in ((0.05, 0.05), (0.2, 0.05), (0.3, 0.1), (0.4, 0.1), (0.5, 0.2), (0.6, 0.2), (0.8, 0.5)):
        ok = 0
        for _ in range(reps):
            aJ = rng.beta(pJ * 10 + 1e-3, (1 - pJ) * 10 + 1e-3)
            aP = rng.beta(pP * 10 + 1e-3, (1 - pP) * 10 + 1e-3)
            bj = rng.random(n_sites) < aJ
            bp = rng.random(n_sites) < aP
            diff = bj.mean() - bp.mean()
            se = math.sqrt((aJ * (1 - aJ) + aP * (1 - aP)) / n_sites * 3.0)
            ok += diff >= 0.15 and diff - 1.645 * se > 0
        out[f"pJ={pJ},pPerp={pP}"] = ok / reps
    return out


if __name__ == "__main__":
    res = {"site_power": grid(), "lexical_hazard": lexical_hazard(), "generic_only": generic_only(),
           "breadth_gate": breadth_gate()}
    print(json.dumps(res, indent=1))
