| Test | Family | n | Mean | 95% CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| P1 within-target FORGET theta_pre (INT-S) | P | 20 | 0.497 | [0.480, 0.514] | 3.56e-21 | 3.56e-21 | supported |
| P2 within-exposed INTERF theta_pre (INT-S) | P | 20 | 0.509 | [0.501, 0.518] | 6.48e-27 | 1.30e-26 | supported |
| G1 P1 generic-adjusted theta_gen | G | 20 | 0.010 | [-0.011, 0.031] | 1.96e-01 | 1.96e-01 | falsified (strong form) |
| G2 P2 generic-adjusted theta_gen | G | 20 | 0.013 | [0.000, 0.026] | 3.52e-02 | 7.03e-02 | falsified (strong form) |
| K1 matched lost vs retained within X (INT-S) | K | 6 | 0.758 | [0.726, 0.788] | 1.15e-05 | 2.30e-05 | supported |
| K2 Y negative control (TOST, rel. bound 0.10) | K | 20 | -0.000 | [-0.001, 0.001] | 3.63e-32 | 1.45e-31 | equivalent |
| K3 developmental TP>S (z(theta) difference) | K | 20 | 0.000 | [0.000, 0.000] | 1.00e+00 | 1.00e+00 | falsified (strong form) |
| K4 composed behaviour (INT-S->C, theta on 1-P(LOOKUP)) | K | 20 | 0.497 | [0.480, 0.514] | 3.56e-21 | 1.07e-20 | supported |
| F1 familiar stratum theta_pre | F | 20 | 0.494 | [0.464, 0.526] | 7.56e-16 | 2.27e-15 | different from null |
| F2 familiarity-boost false rise (rel.) | F | 20 | 0.000 | [-0.002, 0.003] | 6.98e-01 | 8.39e-01 | not different |
| F3 newly learned (low fam.) | F | 20 | 0.491 | [0.470, 0.513] | 4.20e-01 | 8.39e-01 | not different |
| S1 system-specific coupling INT-S (z diff) | S | 20 | 0.004 | [0.001, 0.007] | 3.52e-02 | 7.04e-02 | not different |
| S2 system-specific coupling INT-TP (z diff) | S | 20 | 0.002 | [-0.001, 0.005] | 1.45e-01 | 1.45e-01 | not different |

Interpretation labels (frozen; A: theta_pre and theta_gen supported; B: theta_pre only; C: neither): {"P1": "B", "P2": "B"}

Study-level checks: {"a_enough_valid_seeds": true, "b_natural_auroc_INT-S_median": 0.9970975040932906, "b_pass": true, "c_twin_level_auroc_median": 0.5102172767014752, "c_flag": false, "d_disp_ratio_median": 1.0054814468565072, "d_flag": false, "e_identifiability_P1": {"r2_gen": 0.23992183648860266, "r2_pre": 0.16564815212425288, "r2_joint": 0.3899631287383207}, "e_identifiable_P1": true, "e_identifiability_P2": {"r2_gen": 0.2446850285001948, "r2_pre": 0.18496968474277753, "r2_joint": 0.4270381417069395}, "e_identifiable_P2": true}

Excluded seeds: [{'seed': 1003, 'reason': 'store QC (fabricated)'}, {'seed': 1011, 'reason': 'store QC (fabricated)'}]

## Planned secondary, robustness and descriptive estimates (no confirmatory claims; seed-bootstrap 95% CI)

| Quantity | n | Mean | 95% CI |
|---|---|---|---|
| P1_gen / INT-S | 20 | 0.010 | [-0.011, 0.031] |
| P2_gen / INT-S | 20 | 0.013 | [0.000, 0.026] |
| P1_noState / INT-S | 20 | 0.453 | [0.433, 0.473] |
| P2_noState / INT-S | 20 | 0.454 | [0.444, 0.464] |
| P1_P / INT-S | 20 | 0.503 | [0.487, 0.519] |
| P2_P / INT-S | 20 | 0.507 | [0.499, 0.516] |
| P1_delta / INT-S | 20 | 0.415 | [0.394, 0.434] |
| P2_delta / INT-S | 20 | 0.409 | [0.400, 0.417] |
| P1_gen / INT-TP | 20 | 0.010 | [-0.011, 0.031] |
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
| K4_C+in | 20 | 0.497 | [0.480, 0.514] |
| interaction_z / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_z / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_dprime / K1 | 6 | 0.000 | [0.000, 0.000] |
| SEC_INTvsOUT_z / P1 | 20 | -0.964 | [-1.011, -0.919] |
| SEC_INTvsOUT_z / P2 | 20 | -0.967 | [-0.986, -0.947] |
| DESC_old_P1_XlostVsY / INT-S | 20 | 0.948 | [0.941, 0.954] |
| DESC_old_K1_XlostVsZ / INT-S | 20 | 0.949 | [0.942, 0.955] |
| DESC_TI_INTERF_matched / INT-S | 20 | 0.739 | [0.726, 0.752] |
| EXPL_ACT / INT-S | 20 | 0.503 | [0.491, 0.515] |
| EXPL_ACT / OUT | 20 | 0.494 | [0.485, 0.503] |
| EXPL_ACT / INT-TP | 20 | 0.497 | [0.486, 0.509] |
| EXPL_REPLACE / INT-S | 20 | 0.496 | [0.488, 0.504] |
| EXPL_REPLACE / OUT | 20 | 0.507 | [0.498, 0.516] |
| EXPL_REPLACE / INT-TP | 20 | 0.494 | [0.483, 0.505] |
