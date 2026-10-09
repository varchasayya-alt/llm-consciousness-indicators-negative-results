| Test | Family | n | Mean | 95% CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| P1 within-target FORGET theta_pre (INT-S) | P | 20 | 0.092 | [0.063, 0.120] | 3.03e-06 | 3.03e-06 | inconclusive |
| P2 within-exposed INTERF theta_pre (INT-S) | P | 20 | 0.085 | [0.068, 0.102] | 6.84e-09 | 1.37e-08 | inconclusive |
| G1 P1 generic-adjusted theta_gen | G | 20 | 0.102 | [0.075, 0.126] | 1.44e-07 | 1.44e-07 | supported |
| G2 P2 generic-adjusted theta_gen | G | 20 | 0.100 | [0.084, 0.116] | 1.48e-10 | 2.96e-10 | supported |
| K1 matched lost vs retained within X (INT-S) | K | 6 | 0.556 | [0.529, 0.586] | 9.10e-03 | 1.82e-02 | supported |
| K2 Y negative control (TOST, rel. bound 0.10) | K | 20 | -0.000 | [-0.001, 0.001] | 3.63e-32 | 1.45e-31 | equivalent |
| K3 developmental TP>S (z(theta) difference) | K | 20 | 0.000 | [0.000, 0.000] | 1.00e+00 | 1.00e+00 | falsified (strong form) |
| K4 composed behaviour (INT-S->C, theta on 1-P(LOOKUP)) | K | 20 | 0.092 | [0.063, 0.120] | 3.03e-06 | 9.10e-06 | inconclusive |
| F1 familiar stratum theta_pre | F | 20 | 0.107 | [0.065, 0.147] | 7.82e-05 | 2.35e-04 | different from null |
| F2 familiarity-boost false rise (rel.) | F | 20 | 0.000 | [-0.002, 0.003] | 6.98e-01 | 8.39e-01 | not different |
| F3 newly learned (low fam.) | F | 20 | 0.491 | [0.470, 0.513] | 4.20e-01 | 8.39e-01 | not different |
| S1 system-specific coupling INT-S (z diff) | S | 20 | 0.004 | [-0.002, 0.009] | 2.54e-01 | 5.09e-01 | not different |
| S2 system-specific coupling INT-TP (z diff) | S | 20 | 0.002 | [-0.003, 0.009] | 4.49e-01 | 5.09e-01 | not different |

Interpretation labels (frozen; A: theta_pre and theta_gen supported; B: theta_pre only; C: neither): {"P1": "C", "P2": "C"}

Study-level checks: {"a_enough_valid_seeds": true, "b_natural_auroc_INT-S_median": 0.9970975040932906, "b_pass": true, "c_twin_level_auroc_median": 0.5102172767014752, "c_flag": false, "d_disp_ratio_median": 1.0054814468565072, "d_flag": false, "e_identifiability_P1": {"r2_gen": 0.23992183648860266, "r2_pre": 0.16564815212425288, "r2_joint": 0.3899631287383207}, "e_identifiable_P1": true, "e_identifiability_P2": {"r2_gen": 0.2446850285001948, "r2_pre": 0.18496968474277753, "r2_joint": 0.4270381417069395}, "e_identifiable_P2": true}

Excluded seeds: [{'seed': 1003, 'reason': 'store QC (fabricated)'}, {'seed': 1011, 'reason': 'store QC (fabricated)'}]

## Planned secondary, robustness and descriptive estimates (no confirmatory claims; seed-bootstrap 95% CI)

| Quantity | n | Mean | 95% CI |
|---|---|---|---|
| P1_gen / INT-S | 20 | 0.102 | [0.075, 0.126] |
| P2_gen / INT-S | 20 | 0.100 | [0.084, 0.116] |
| P1_noState / INT-S | 20 | 0.450 | [0.432, 0.468] |
| P2_noState / INT-S | 20 | 0.434 | [0.422, 0.446] |
| P1_P / INT-S | 20 | 0.187 | [0.161, 0.216] |
| P2_P / INT-S | 20 | 0.148 | [0.131, 0.164] |
| P1_delta / INT-S | 20 | 0.404 | [0.385, 0.424] |
| P2_delta / INT-S | 20 | 0.387 | [0.375, 0.398] |
| P1_gen / INT-TP | 20 | 0.102 | [0.075, 0.126] |
| P1_gen / OUT | 20 | 0.856 | [0.844, 0.868] |
| P1_r2_pre | 20 | 0.160 | [0.144, 0.176] |
| P1_r2_gen | 20 | 0.231 | [0.214, 0.247] |
| P1_r2_joint | 20 | 0.388 | [0.368, 0.408] |
| P2_r2_pre | 20 | 0.186 | [0.178, 0.194] |
| P2_r2_gen | 20 | 0.248 | [0.237, 0.259] |
| P2_r2_joint | 20 | 0.425 | [0.418, 0.432] |
| P1_fingerprint_vs_dC_spearman | 20 | 0.096 | [0.058, 0.137] |
| K1_pairs | 20 | 89.450 | [86.550, 92.200] |
| K1_max_smd | 20 | 0.116 | [0.102, 0.131] |
| K4_C+in | 20 | 0.092 | [0.063, 0.120] |
| interaction_z / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_z / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_dprime / K1 | 6 | 0.000 | [0.000, 0.000] |
| SEC_INTvsOUT_z / P1 | 20 | -1.418 | [-1.461, -1.374] |
| SEC_INTvsOUT_z / P2 | 20 | -1.443 | [-1.464, -1.420] |
| DESC_old_P1_XlostVsY / INT-S | 20 | 0.924 | [0.918, 0.931] |
| DESC_old_K1_XlostVsZ / INT-S | 20 | 0.926 | [0.920, 0.932] |
| DESC_TI_INTERF_matched / INT-S | 20 | 0.718 | [0.706, 0.729] |
| EXPL_ACT / INT-S | 20 | 0.503 | [0.491, 0.515] |
| EXPL_ACT / OUT | 20 | 0.494 | [0.485, 0.503] |
| EXPL_ACT / INT-TP | 20 | 0.497 | [0.486, 0.509] |
| EXPL_REPLACE / INT-S | 20 | 0.496 | [0.488, 0.504] |
| EXPL_REPLACE / OUT | 20 | 0.507 | [0.498, 0.516] |
| EXPL_REPLACE / INT-TP | 20 | 0.494 | [0.483, 0.505] |
