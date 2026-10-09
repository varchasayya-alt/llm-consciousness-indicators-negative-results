# Stage-1 PROPOSED PROTOCOL v3: within-target identification (for PI approval; NOT frozen, NOT registered)

| Field | Value |
|---|---|
| Version | v3.0-proposed (2026-10-02) |
| Supersedes | `protocol_FINAL_PROPOSED.md` (v1.0-proposed), whose primary X_lost-vs-Y contrast is confounded by a procedure fingerprint (D34) |
| Design basis | `v3_design_memo.md` plus the PI's v3 decisions (2026-10-02); decisions D36–D41 |
| Config | `stage1_config.yaml` v0.4 → `stage1_config_FINAL.yaml` after calibration (hash recorded at freeze) |
| Code | `s1/` at the commit recorded at freeze. Primary statistic: `s1/estimands.theta_post_dml`. Analysis: `s1/analysis.py` |
| Companion documents | `rival_hypotheses.md`, `claim_boundaries.md`, `sham_algorithm_spec.md` (Y as negative control), `calibration_report.md`, `results/statistics/v3_statistic_simulation/` |
| Status | **No confirmatory seed has been run. No scientific monitor has ever been trained.** |

**Retired seeds:**

- 12345 (feasibility);
- 9001–9005 (v1/v2 calibration; historical only);
- 9011–9013 (v2 validation, consumed);
- 1–99 (unit tests);
- 10,000,000+ (fabricated data).

## 0. Why this revision is legitimate

- No scientific monitor or controller has been trained or evaluated on any store. Calibration code cannot import monitor, controller or pipeline modules (unit-tested).
- No confirmatory seed has been run. The protocol is not frozen and nothing is registered.
- The trigger is a calibration-discovered confound found with store-only diagnostics (D34), not an observed monitor effect.
- The design and the statistic were fixed before any v3 store run. The statistic was chosen by a simulation-only comparison (§4).

## 1. Research question

> Among items subjected to the **same** intervention procedure, does an independently trained monitor track the degree to which the observed system's competence **actually** changes? And does a controller that uses the monitor adapt information-seeking accordingly, under internal interventions never encountered in training?

The question is asked against three explicit rivals: H1 difficulty, H2 anomaly / procedure detection, and H3 competence monitoring (`rival_hypotheses.md`). Every claim is Level 1 (`claim_boundaries.md`).

## 2. Hypothesis families (confirmatory)

The unit is the **store seed** (n = 20). Items are never replications.

### Family P: co-primary (Holm, FWER .05)

- **P1.** θ_pre(INT-S) > 0 among the 300 identically targeted T-FORGET items X. MEI 0.10.
- **P2.** θ_pre(INT-S) > 0 among all base-correct EV items under T-INTERF (all equally exposed). MEI 0.10.

### Family G: generic-adjusted robustness (Holm)

- **G1, G2.** θ_gen(INT-S) > 0 for the P1 and P2 populations.
- These determine the interpretation labels (§5). They do not redefine the effect.

### Family K: key secondary (Holm)

- **K1.** Matched binary: AUROC of the decrease in the monitor outcome **adjusted for all pre-intervention information** (the M-side residual of θ_pre), X_lost vs X_retained. See D42.
  - Because K1 adjusts only for pre-intervention information, it shares θ_pre's sensitivity to intervention intensity. It is read under the same A/B/C label.
  - Matching: 1:1 on a propensity score built from pre-intervention variables only (C_pre, log p_pre, exposure, fluency, relation, susceptibility score), never monitor outputs.
  - Exact on exposure; caliper 0.2 SD of the logit propensity.
  - Requirements: ≥ 50 pairs and every SMD ≤ 0.10, else the seed is missing for K1.
  - Test: AUROC > 0.5, MEI 0.55.
- **K2.** Y negative control: TOST that mean ΔM(Y) − mean ΔM(Z) lies within ±0.10 × D_nat.
- **K3.** Developmental: z(θ_pre INT-TP) − z(θ_pre INT-S) > 0, MEI 0.10.
- **K4.** Composed behaviour: θ_pre with the monitor replaced by 1 − P(LOOKUP) of the INT-S → controller system, within X ("compositional zero-shot transfer"), MEI 0.10.

### Family F: familiarity (two-sided, Holm)

- **F1.** θ_pre within the high-exposure stratum of X.
- **F2.** T-FAM false rise (D_nat units). Unchanged from v1.
- **F3.** T-NEW newly learned vs exposure-matched still-unknown, low-exposure stratum. Unchanged from v1.

### Family S: system-specific coupling (two-sided, Holm)

- **S1, S2.** z(θ_pre, B̃→B̃) − z(θ_pre, A→B̃) on store B's targeted set, for INT-S and INT-TP.
- Ridge alignment is fit only on intact MT-split states.

## 3. System, world, monitors, controller

These are unchanged from v1.0-proposed §4–§5:

- store d = 128, L = 4, E_store = 60;
- read set: s2 and [A] × layers 0–4, frozen on architectural grounds;
- monitors IN / OUT / INT-{S, T, P, TP}, 8,000 steps × batch 256;
- P-family dropout rate ~ U(0, 0.6) (C3r);
- controller trained on the intact CT split; the C+in variant;
- twin store B;
- operational familiarity via name fluency.

**The read set is not revised.** The late-[A] procedure fingerprint is handled by the design and by measured adjustment, not by representational selection.

## 4. Primary statistic (fixed by simulation before any v3 store run; D37)

For a population of identically treated items with competence C (answer margin) and monitor score M:

1. **Baseline set** B = [RCS(rank C_pre), RCS(rank M_pre), pre covariates W0 (ranks)].
   - RCS = restricted cubic spline with 5 knots at quantiles .05/.275/.5/.725/.95.
   - W0 = log p_pre(v\*), exposure class, pre-intervention name fluency, relation indicators.
2. **OLS step.** Residualise rank M_post, rank C_post and every column of the pre-intervention read state H_pre (10 sites × 128) on B.
3. **Pre-state step.** Cross-fitted ridge (5 folds, stratified by exposure; λ by generalised cross-validation over 10^{-2…4}) predicts each residualised outcome from the residualised pre-states. The predictions are subtracted, so pre-state information (including any susceptibility the states encode) is removed **from both** variables.
   - The folds depend only on the population, so every monitor and condition shares them.
4. θ_s = Pearson correlation of the two final residual vectors.
5. **Seed-level test.** One-sided one-sample t-test on atanh θ_s. Wilcoxon signed-rank test and seed-bootstrap 95% CI reported.
   - **Supported:** Holm p < .05 and mean θ ≥ 0.10.
   - **Falsified (strong form):** upper CI < 0.10.
   - **Otherwise:** inconclusive.

**Why this statistic** (`results/statistics/v3_statistic_simulation/sim_results.md`). The comparison used 100 replicate studies × 20 seeds × 300 items for each world and each loss regime (~20% and ~50% of targeted items lose the answer).

| Candidate | Result |
|---|---|
| **Selected S9** | 0.00 support rate in the null, strong regression-to-the-mean and state-visible susceptibility worlds in both regimes. Pure-procedure and intensity-only worlds get label B in 100% of studies and **never** A. H3, strong H3 and H3 + generic fingerprint worlds get label A in 100% of studies (mean θ 0.80–0.97) |
| v1/v2 change-score partial Spearman | False support under regression-to-the-mean: 17% (~20% loss) and 63% (~50% loss). Under susceptibility: 87% / 54% |
| One-sided P_i score | False support under susceptibility: 99–100% |
| No pre-state step | False support under susceptibility: 100% |

**Documented limitation.** If a monitor responded to fragility that is *not* encoded in the pre-intervention states (the latent-susceptibility stress test), S9 gives a false label A in 30–35% of studies, with θ ≈ 0.09, just below the MEI.

**Reported, not decisive:**

| Variant | Definition |
|---|---|
| θ_noState | No pre-state step |
| θ_P | One-sided P_i covariate (memo v3 §4) |
| θ_delta | Change-score; descriptive |
| θ_gen | Pre-state step plus the 40 generic post-change features: per-site log displacement, cos(Δ, h_pre), relative norm change, abnormality change |

## 5. Interpretation labels (frozen)

| Label | Condition | Reading |
|---|---|---|
| **A** | θ_pre **and** θ_gen supported | Tracking that cannot be reduced to the measured generic intervention intensity / change structure |
| **B** | θ_pre supported, θ_gen not | Tracks a representation change correlated with competence; competence monitoring is not separated from generic internal-change detection |
| **C** | θ_pre not supported | No evidence for the proposed monitoring effect |
| **Non-identifiable** | Study-level identifiability check fails (§8) | No claim |

**Material differences** between θ_noState, θ_P and θ_pre are reported and interpreted as susceptibility contributions.

## 6. Interventions (calibrated values from `calibration_report.md`)

| Intervention | Specification | Per-seed QC (store-level only) |
|---|---|---|
| **T-FORGET (v3)** | Identical procedure for all 300 X items: uniform-target forgetting with the forget-then-relearn Y negative control and displacement matching. RetainKL to the frozen original (λ_R 100). Mechanism variant ⟦CAL⟧ ∈ {V0, V1 anchor, V2 enlarged pool, V3 block-1–2 MLPs}; lr ⟦CAL⟧; steps ⟦CAL⟧; anchor ⟦CAL⟧ | (1) X lost ∈ [0.35, 0.65] **and** IQR(C_post over X) ≥ 1 nat. (2) Y lost ≤ 3%; lost ≤ 5% in each of 5 Z categories. (3) **Continuous retention** for Y and each Z category: \|Δmargin\|, \|Δlog p\| and KL each ≤ 0.10 × X's; \|Δmargin\| ≤ 0.10 × D_C; sensitivity at 0.05/0.20. (4) ≥ 8/10 Y displacement ratios in [0.8, 1.25]. (5) \|Δfluency\| ≤ 0.1 SD |
| **T-INTERF** | 1,000 new entities × 2 facts + 8 mentions; no replay; lr ⟦CAL⟧; ≤ ⟦CAL⟧ steps | Lost fraction ∈ [0.05, 0.60] |
| **T-NEW** | N facts + exposure-matched U mentions + RetainKL; lr, steps, λ_R ⟦CAL⟧ | N learned ≥ 70%; U correct ≤ 5%; collateral (base-correct EV and near-entity CT) binary ≤ 5% **and** \|Δmargin\| ≤ 0.10 × D_C |
| **T-FAM** | ⟦CAL⟧ extra mention presentations of F's entities + RetainKL | Rise ≥ 0.5 SD vs U2; F correct ≤ 5%; collateral binary ≤ 5% and \|Δmargin\| ≤ 0.10 × D_C |
| Input corruption, T-ACT, REPLACE | Unchanged | — |

- **Fallback ladder:** lr × {1, 0.5, 2}, then steps × {1, 2}. At most 6 attempts.
- **Y is a negative control,** not the primary identification control. The matched-pair minimum is removed. Y's G1–G4 fingerprint is reported per seed.

## 7. Seeds and power

- **Confirmatory:** the first 20 of 1001–1040 that pass all store- and intervention-level QC.
- **v3 methods development:** 9031–9033. **Fresh validation:** 9021–9023, used **once** per component.
- **Power:** the within-seed SE of θ_s is about 0.06 at n = 300. With 20 seeds, 80% power is reached at mean θ ≈ 0.04–0.05 (dz 0.58), so MEI 0.10 is comfortably detectable. The simulation's H3 worlds were supported in 100% of replicate studies.

## 8. Exclusions and failure criteria (frozen)

**Methods phase** (store-only; development seeds, then once on validation seeds):

| Code | Failure condition |
|---|---|
| F1 | No mechanism variant meets the continuous retention criteria |
| F2 | X lost ∉ [0.35, 0.65] or IQR(C_post) < 1 nat |
| F3 | Median cross-fitted R²(ΔC \| 40 generic features) ≥ 0.90 |
| F4 | Median R²(ΔC \| pre-intervention information) ≥ 0.90 |
| F5 | Median R²(ΔC \| both) ≥ 0.95 |
| F6 | < 50 balanced matched pairs on any seed |
| F7 | Y fails its QC |
| F8 | Fresh validation fails |

F1, F2, F3–F5 and F8 also apply to the P2 population (T-INTERF), as relevant. **No threshold is relaxed after results.** Failure of F1 + F2, F3–F5, or F8 means a move toward Option B (an explicitly local memory) after a report to the PI.

**Confirmatory phase:**

- **Seed-level:**
  - Store A or B fails QC → replace.
  - T-FORGET (A or B) or T-INTERF fails per-seed QC after the ladder → replace.
  - T-NEW / T-FAM fails → that intervention is excluded for the seed.
  - K1-infeasible seeds are missing for K1 only.
- **Study-level stop:**
  - < 20 valid seeds in 1001–1040;
  - > 5 T-FORGET machinery failures among the first 25 seeds;
  - median natural AUROC of INT-S < 0.65;
  - **new:** P1 (or P2) non-identifiable (median R²_gen ≥ 0.90, R²_pre ≥ 0.90 or R²_joint ≥ 0.95). The affected family gets no claim.
- **Flags:** median twin level AUROC > 0.60; median Y displacement ratio outside [0.67, 1.5].

## 9. Claims

See `claim_boundaries.md`. **Revised by D54 (pre-data): only label A is affirmative support; label B is not** (it is reported as ambiguous, change-structure-sensitive tracking). The strongest permitted claim, if P1 has label A:

> "A separately developed monitor tracks intervention-induced changes in the current competence of the neural system it observes, beyond what can be inferred from unchanged inputs and measured pre-intervention susceptibility."

If labels are A for P1 and P2, add:

> "The tracking cannot be readily reduced to the measured generic signatures of the intervention and generalises across qualitatively different competence changes."

**Never claimed:** phenomenal consciousness, self-awareness, subjective experience, human-like metacognition, a conscious LLM.

## 10. Order of operations

1. PI approves the calibration package.
2. Write `stage1_config_FINAL.yaml`; record its hash and the commit in `FROZEN_PROTOCOL.json`.
3. Commit.
4. Create the OSF Registration.
5. Only then run confirmatory seeds.

The analysis is a single command and is run once.
