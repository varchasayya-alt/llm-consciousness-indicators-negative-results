| Test | Family | n | Mean | 95% CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| P1 within-target FORGET theta_pre (INT-S) | P | 20 | 0.020 | [-0.004, 0.044] | 6.33e-02 | 1.27e-01 | falsified (strong form) |
| P2 within-exposed INTERF theta_pre (INT-S) | P | 20 | 0.001 | [-0.014, 0.017] | 4.57e-01 | 4.57e-01 | falsified (strong form) |
| G1 P1 generic-adjusted theta_gen | G | 20 | 0.017 | [-0.007, 0.044] | 1.02e-01 | 2.04e-01 | falsified (strong form) |
| G2 P2 generic-adjusted theta_gen | G | 20 | -0.002 | [-0.016, 0.014] | 5.74e-01 | 5.74e-01 | falsified (strong form) |
| K1 matched lost vs retained within X (INT-S) | K | 2 | 0.518 | [0.480, 0.557] | 3.59e-01 | 7.18e-01 | inconclusive |
| K2 Y negative control (TOST, rel. bound 0.10) | K | 20 | -0.000 | [-0.001, 0.001] | 3.63e-32 | 1.45e-31 | equivalent |
| K3 developmental TP>S (z(theta) difference) | K | 20 | 0.000 | [0.000, 0.000] | 1.00e+00 | 1.00e+00 | falsified (strong form) |
| K4 composed behaviour (INT-S->C, theta on 1-P(LOOKUP)) | K | 20 | 0.020 | [-0.004, 0.044] | 6.33e-02 | 1.90e-01 | falsified (strong form) |
| F1 familiar stratum theta_pre | F | 20 | 0.042 | [0.005, 0.080] | 4.70e-02 | 1.41e-01 | not different |
| F2 familiarity-boost false rise (rel.) | F | 20 | 0.000 | [-0.002, 0.003] | 6.98e-01 | 8.39e-01 | not different |
| F3 newly learned (low fam.) | F | 20 | 0.491 | [0.470, 0.513] | 4.20e-01 | 8.39e-01 | not different |
| S1 system-specific coupling INT-S (z diff) | S | 20 | 0.003 | [-0.001, 0.007] | 2.40e-01 | 4.79e-01 | not different |
| S2 system-specific coupling INT-TP (z diff) | S | 20 | 0.000 | [-0.005, 0.006] | 9.63e-01 | 9.63e-01 | not different |

Interpretation labels (frozen; A: theta_pre and theta_gen supported; B: theta_pre only; C: neither): {"P1": "C", "P2": "C"}

Study-level checks: {"a_enough_valid_seeds": true, "b_natural_auroc_INT-S_median": 0.9970975040932906, "b_pass": true, "c_twin_level_auroc_median": 0.5121771759900413, "c_flag": false, "d_disp_ratio_median": 1.0054814468565072, "d_flag": false, "e_identifiability_P1": {"r2_gen": 0.09164990142632362, "r2_pre": 0.5022268856800162, "r2_joint": 0.5517506427082987}, "e_identifiable_P1": true, "e_identifiability_P2": {"r2_gen": 0.1356161932216718, "r2_pre": 0.5972185785655459, "r2_joint": 0.6488809753283166}, "e_identifiable_P2": true}

Excluded seeds: [{'seed': 1003, 'reason': 'store QC (fabricated)'}, {'seed': 1011, 'reason': 'store QC (fabricated)'}]

## Planned secondary, robustness and descriptive estimates (no confirmatory claims; seed-bootstrap 95% CI)

| Quantity | n | Mean | 95% CI |
|---|---|---|---|
| P1_gen / INT-S | 20 | 0.017 | [-0.007, 0.044] |
| P2_gen / INT-S | 20 | -0.002 | [-0.016, 0.014] |
| P1_noState / INT-S | 20 | 0.024 | [-0.000, 0.048] |
| P2_noState / INT-S | 20 | 0.007 | [-0.007, 0.022] |
| P1_P / INT-S | 20 | 0.020 | [-0.004, 0.045] |
| P2_P / INT-S | 20 | 0.001 | [-0.013, 0.015] |
| P1_delta / INT-S | 20 | 0.180 | [0.161, 0.199] |
| P2_delta / INT-S | 20 | 0.179 | [0.168, 0.190] |
| P1_gen / INT-TP | 20 | 0.017 | [-0.007, 0.044] |
| P1_gen / OUT | 20 | 0.764 | [0.749, 0.777] |
| P1_r2_pre | 20 | 0.490 | [0.448, 0.524] |
| P1_r2_gen | 20 | 0.100 | [0.082, 0.119] |
| P1_r2_joint | 20 | 0.546 | [0.515, 0.571] |
| P2_r2_pre | 20 | 0.592 | [0.581, 0.602] |
| P2_r2_gen | 20 | 0.139 | [0.129, 0.149] |
| P2_r2_joint | 20 | 0.655 | [0.646, 0.663] |
| P1_fingerprint_vs_dC_spearman | 20 | 0.037 | [0.009, 0.066] |
| K1_pairs | 20 | 57.050 | [54.400, 59.700] |
| K1_max_smd | 20 | 0.148 | [0.135, 0.163] |
| K4_C+in | 20 | 0.020 | [-0.004, 0.044] |
| interaction_z / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P1 | 20 | 0.000 | [0.000, 0.000] |
| interaction_z / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_raw / P2 | 20 | 0.000 | [0.000, 0.000] |
| interaction_dprime / K1 | 2 | 0.000 | [0.000, 0.000] |
| SEC_INTvsOUT_z / P1 | 20 | -1.119 | [-1.155, -1.080] |
| SEC_INTvsOUT_z / P2 | 20 | -1.153 | [-1.173, -1.133] |
| DESC_old_P1_XlostVsY / INT-S | 20 | 0.888 | [0.877, 0.899] |
| DESC_old_K1_XlostVsZ / INT-S | 20 | 0.889 | [0.878, 0.900] |
| DESC_TI_INTERF_matched / INT-S | 20 | 0.524 | [0.510, 0.539] |
| EXPL_ACT / INT-S | 20 | 0.503 | [0.491, 0.515] |
| EXPL_ACT / OUT | 20 | 0.494 | [0.485, 0.503] |
| EXPL_ACT / INT-TP | 20 | 0.497 | [0.486, 0.509] |
| EXPL_REPLACE / INT-S | 20 | 0.496 | [0.488, 0.504] |
| EXPL_REPLACE / OUT | 20 | 0.507 | [0.498, 0.516] |
| EXPL_REPLACE / INT-TP | 20 | 0.494 | [0.483, 0.505] |
