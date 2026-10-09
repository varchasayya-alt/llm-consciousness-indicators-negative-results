"""Simulated item-level worlds for METHODS VALIDATION ONLY (no model information; never hypothesis data).

One generator for identically treated items of one store seed (T-FORGET X items or T-INTERF base-correct items).
Latent structure:
  S        pre-existing susceptibility (partly visible in pre-intervention states H_pre)
  z_I      item-varying effective intervention intensity (measured by generic post-change features G, D, F)
  eps      idiosyncratic competence outcome (visible only in the post-intervention state)
  C_pre    baseline answer margin; competence change depends on z_I, S, C_pre (nonlinearly in 'rtm'), eps
  monitor  logit M = item bias u + mapping (scenario-specific)
Scenarios (monitor post-intervention):
  H3              reads current competence (same mapping pre and post)
  H3_strong       stronger, less noisy H3
  H3_plus_generic H3 + response to measured generic change
  H2_generic      responds only to measured generic change statistics (anomaly detector)
  H2_intensity    responds only to true intervention intensity (measured by G/D/F with noise)
  susceptibility  responds only to a STATE-VISIBLE fragility feature s_vis (a fixed projection of H_pre that
                  correlates with S); a monitor can only respond to what the states encode
  susceptibility_latent  STRESS TEST: responds to latent S directly (not fully encoded in the states)
  rtm             null w.r.t. competence, strong nonlinear baseline dependence in both M and C
  null            no response beyond a global shift
"""
import numpy as np

SCENARIOS = ("H3", "H3_strong", "H3_plus_generic", "H2_generic", "H2_intensity", "susceptibility", "rtm", "null",
             "susceptibility_latent")
H3_LIKE = ("H3", "H3_strong", "H3_plus_generic")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_items(scenario, n, rng, d_h=40, d_g=16, structure=None, loss_shift=0.0):
    """Return dict of per-item arrays for one store seed. `structure` (fixed loadings) may be shared across seeds.
    loss_shift (default 0, used by the statistic simulation) shifts the loss index; the fake-data dry run uses 1.5 so
    that about half of the targeted items lose competence, as required by the v3 outcome-diversity gate."""
    if structure is None:
        structure = make_structure(np.random.default_rng(12345), d_h, d_g)
    a_S, a_C, aI, ac = structure["a_S"], structure["a_C"], structure["aI"], structure["ac"]
    e = (rng.random(n) < 0.5).astype(float)
    fluency = 2.0 * e + rng.normal(0, 0.6, n)
    relation = rng.integers(0, 4, n)
    C_pre = np.clip(rng.normal(5.9 + 0.2 * e, 0.9, n), 1.0, 9.0)
    S = rng.normal(0, 1, n)
    z_I = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    c = (C_pre - 5.9) / 0.9
    if scenario == "rtm":
        eta = 1.0 * z_I * 0.6 + 0.6 * S - 1.4 * c - 0.8 * c ** 2 + 0.7 * eps + 0.6
    else:
        eta = 0.8 * z_I + 0.8 * S - 0.5 * c + 1.0 * eps
    eta = eta + loss_shift
    rho = _sig(-1.5 * eta)                                   # fraction of competence remaining
    C_post = rho * C_pre + (1 - rho) * (-0.6) + rng.normal(0, 0.15, n)
    logp = lambda C: -np.log1p(31.0 * np.exp(-C))
    H_pre = np.outer(S, a_S) + np.outer(c, a_C) + rng.normal(0, 1.0, (n, d_h))
    G = np.outer(z_I, aI) + np.outer(1 - rho, ac) + rng.normal(0, 0.5, (n, d_g))
    D = G[:, :4].mean(1)
    F = G @ structure["v_fp"] + rng.normal(0, 0.3, n)
    g_score = (G @ structure["v_anom"])
    g_score = (g_score - g_score.mean()) / (g_score.std() + 1e-9)
    u = rng.normal(0, 0.6, n)
    mu = 2.0
    lM_pre = mu + 0.6 * c * 0.9 + 0.4 * e + u + rng.normal(0, 0.2, n)
    nz = rng.normal(0, 0.3, n)
    h3 = lambda beta, noise: mu + beta * (C_post - 5.9) + 0.4 * e + u + rng.normal(0, noise, n)
    if scenario == "H3":
        lM_post = h3(0.6, 0.3)
    elif scenario == "H3_strong":
        lM_post = h3(0.9, 0.15)
    elif scenario == "H3_plus_generic":
        lM_post = h3(0.6, 0.3) - 0.8 * g_score
    elif scenario == "H2_generic":
        lM_post = lM_pre - 1.0 - 0.8 * g_score + nz
    elif scenario == "H2_intensity":
        lM_post = lM_pre - 1.0 - 0.8 * z_I + nz
    elif scenario == "susceptibility":
        s_vis = H_pre @ a_S / (a_S @ a_S)
        s_vis = (s_vis - s_vis.mean()) / (s_vis.std() + 1e-9)
        lM_post = lM_pre - 0.8 - 0.7 * s_vis + nz
    elif scenario == "susceptibility_latent":
        lM_post = lM_pre - 0.8 - 0.7 * S + nz
    elif scenario == "rtm":
        dev = lM_pre - mu
        lM_post = mu - 1.0 + 0.4 * dev + 0.35 * dev ** 2 + nz
    elif scenario == "null":
        lM_post = lM_pre - 0.5 + nz
    else:
        raise ValueError(scenario)
    return {"e": e, "fluency": fluency, "relation": relation, "C_pre": C_pre, "C_post": C_post,
            "logp_pre": logp(C_pre), "logp_post": logp(C_post), "M_pre": _sig(lM_pre), "M_post": _sig(lM_post),
            "H_pre": H_pre, "G": G, "D": D, "F": F, "S": S, "z_I": z_I}


def make_structure(rng, d_h=40, d_g=16):
    """Fixed loadings: S and C_pre visible in pre-states; G mostly intensity, weakly competence-laden."""
    return {"a_S": rng.normal(0, 0.5, d_h), "a_C": rng.normal(0, 0.3, d_h),
            "aI": rng.uniform(0.3, 1.0, d_g), "ac": rng.uniform(0.0, 0.4, d_g),
            "v_fp": rng.normal(0, 1, d_g) / np.sqrt(d_g), "v_anom": rng.uniform(0.2, 1.0, d_g)}


def pre_covariates(it, P=None):
    """W: pre-intervention covariates (no monitor output): logp_pre, exposure, fluency, relation dummies, [P]."""
    rel = np.eye(4)[it["relation"]][:, 1:]
    cols = [it["logp_pre"], it["e"], it["fluency"], rel]
    if P is not None:
        cols.append(P)
    return np.column_stack(cols)
