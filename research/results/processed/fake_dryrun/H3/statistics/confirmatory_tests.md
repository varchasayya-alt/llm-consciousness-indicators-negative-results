| Test | Family | n | Mean | 95% CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| P1 within-target FORGET theta_pre (INT-S) | P | 20 | 0.798 | [0.785, 0.811] | 4.23e-23 | 4.23e-23 | supported |
| P2 within-exposed INTERF theta_pre (INT-S) | P | 20 | 0.800 | [0.793, 0.807] | 1.30e-28 | 2.61e-28 | supported |
| G1 P1 generic-adjusted theta_gen | G | 20 | 0.715 | [0.695, 0.735] | 1.96e-20 | 1.96e-20 | supported |
| G2 P2 generic-adjusted theta_gen | G | 20 | 0.723 | [0.714, 0.731] | 2.34e-27 | 4.69e-27 | supported |
| K1 matched lost vs retained within X (INT-S) | K | 6 | 0.889 | [0.868, 0.906] | 1.54e-07 | 1.54e-07 | supported |
| K2 Y negative control (TOST, rel. bound 0.10) | K | 20 | -0.000 | [-0.001, 0.001] | 3.63e-32 | 1.45e-31 | equivalent |
| K3 developmental TP>S (z(theta) difference) | K | 20 | 0.275 | [0.260, 0.290] | 5.92e-19 | 1.18e-18 | supported |
| K4 composed behaviour (INT-S->C, theta on 1-P(LOOKUP)) | K | 20 | 0.798 | [0.785, 0.811] | 4.23e-23 | 1.27e-22 | supported |
| F1 familiar stratum theta_pre | F | 20 | 0.785 | [0.768, 0.802] | 6.83e-21 | 1.37e-20 | different from null |
| F2 familiarity-boost false rise (rel.) | F | 20 | 0.000 | [-0.002, 0.003] | 6.98e-01 | 6.98e-01 | not different |
| F3 newly learned (low fam.) | F | 20 | 0.990 | [0.987, 0.993] | 6.05e-36 | 1.82e-35 | different from null |
| S1 system-specific coupling INT-S (z diff) | S | 20 | 0.011 | [0.007, 0.015] | 1.02e-04 | 1.02e-04 | different from null |
| S2 system-specific coupling INT-TP (z diff) | S | 20 | 0.146 | [0.131, 0.161] | 5.99e-14 | 1.20e-13 | different from null |

Interpretation labels (frozen; A: theta_pre and theta_gen supported; B: theta_pre only; C: neither): {"P1": "A", "P2": "A"}

Study-level checks: {"a_enough_valid_seeds": true, "b_natural_auroc_INT-S_median": 0.9970975040932906, "b_pass": true, "c_twin_level_auroc_median": 0.5102172767014752, "c_flag": false, "d_disp_ratio_median": 1.0054814468565072, "d_flag": false, "e_identifiability_P1": {"r2_gen": 0.23992183648860266, "r2_pre": 0.16564815212425288, "r2_joint": 0.3899631287383207}, "e_identifiable_P1": true, "e_identifiability_P2": {"r2_gen": 0.2446850285001948, "r2_pre": 0.18496968474277753, "r2_joint": 0.4270381417069395}, "e_identifiable_P2": true}

Excluded seeds: [{'seed': 1003, 'reason': 'store QC (fabricated)'}, {'seed': 1011, 'reason': 'store QC (fabricated)'}]

## Planned secondary, robustness and descriptive estimates (no confirmatory claims; seed-bootstrap 95% CI)

| Quantity | n | Mean | 95% CI |
|---|---|---|---|
| P1_gen / INT-S | 20 | 0.715 | [0.695, 0.735] |
| P2_gen / INT-S | 20 | 0.723 | [0.714, 0.731] |
| P1_noState / INT-S | 20 | 0.837 | [0.829, 0.846] |
| P2_noState / INT-S | 20 | 0.836 | [0.830, 0.841] |
| P1_P / INT-S | 20 | 0.808 | [0.796, 0.820] |
| P2_P / INT-S | 20 | 0.805 | [0.799, 0.812] |
| P1_delta / INT-S | 20 | 0.870 | [0.862, 0.877] |
| P2_delta / INT-S | 20 | 0.871 | [0.868, 0.874] |
| P1_gen / INT-TP | 20 | 0.816 | [0.801, 0.830] |
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
| K4_C+in | 20 | 0.798 | [0.785, 0.811] |
| interaction_z / P1 | 20 | 0.275 | [0.260, 0.290] |
| interaction_raw / P1 | 20 | 0.080 | [0.074, 0.086] |
| interaction_z / P2 | 20 | 0.310 | [0.304, 0.317] |
| interaction_raw / P2 | 20 | 0.087 | [0.084, 0.090] |
| interaction_dprime / K1 | 6 | 0.510 | [0.474, 0.560] |
| SEC_INTvsOUT_z / P1 | 20 | -0.413 | [-0.444, -0.383] |
| SEC_INTvsOUT_z / P2 | 20 | -0.428 | [-0.447, -0.411] |
| DESC_old_P1_XlostVsY / INT-S | 20 | 1.000 | [1.000, 1.000] |
| DESC_old_K1_XlostVsZ / INT-S | 20 | 1.000 | [1.000, 1.000] |
| DESC_TI_INTERF_matched / INT-S | 20 | 0.854 | [0.843, 0.863] |
| EXPL_ACT / INT-S | 20 | 0.503 | [0.491, 0.515] |
| EXPL_ACT / OUT | 20 | 0.494 | [0.485, 0.503] |
| EXPL_ACT / INT-TP | 20 | 0.497 | [0.486, 0.509] |
| EXPL_REPLACE / INT-S | 20 | 0.496 | [0.488, 0.504] |
| EXPL_REPLACE / OUT | 20 | 0.507 | [0.498, 0.516] |
| EXPL_REPLACE / INT-TP | 20 | 0.494 | [0.483, 0.505] |
