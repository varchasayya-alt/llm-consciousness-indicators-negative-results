# Stage-1 FINAL PROPOSED PROTOCOL (for PI approval; NOT frozen, NOT registered)

| Field | Value |
|---|---|
| Version | v1.0-proposed (2026-10-01) |
| Supersedes | `preregistration_stage1.md` (draft v0.1) |
| Config | `stage1_config_FINAL.yaml` (hash recorded at freeze) |
| Code | `s1/` package at the commit recorded at freeze |
| Companion documents (part of this protocol) | `sham_algorithm_spec.md`, `rival_hypotheses.md`, `claim_boundaries.md`, `calibration_report.md` |
| Status | **No confirmatory seed has been run.** Calibration seeds 9001–9005 are retired |

## 1. Research question

Can an independently trained metacognitive monitor learn to detect **item-specific changes in the competence** of another neural system from that system's internal state, and can a controller that uses it adapt information-seeking under **internal interventions never encountered in training**?

The question is asked against three explicit rivals:

- H1 input-difficulty estimation;
- H2 intervention/anomaly detection;
- H3 competence monitoring.

The predictions for each are in `rival_hypotheses.md`. Every claim is Level 1 (`claim_boundaries.md`).

## 2. Hypothesis families (confirmatory)

All endpoints are seed-level (unit = store seed, n = 20). "TI" = AUROC of the monitor-score *decrease* (pre → post, same item, same input). Ties count 0.5.

### Family P: co-primary (Holm, FWER .05)

- **P1.** TI_FORGET(INT-S) > 0.5.
  - Population: matched pairs (X_lost, Y) under held-out targeted forgetting with the displacement-matched sham.
  - Minimum effect of interest (MEI): 0.55.
- **P2.** TI_INTERF(INT-S) > 0.5.
  - Population: matched pairs (lost, retained) under held-out later-learning interference.
  - MEI: 0.55.
  - Elevated to co-primary because uniform forgetting makes output-based uncertainty trivial, whereas interference produces naturally arising errors.

### Family K: key secondary (Holm)

- **K1** unrelated-knowledge specificity: AUROC(X_lost vs post-correct Z) > 0.5.
- **K2** sham equivalence: TOST that mean Δs(Y) − mean Δs(Z) lies within ±0.10 × D_nat (§8).
- **K3** continuous competence tracking under FORGET: partial Spearman ρ > 0 (MEI 0.10; §7).
- **K4** the same under INTERF.
- **K5** developmental: d′(TI_FORGET, INT-TP) − d′(TI_FORGET, INT-S) > 0 (MEI 0.10).
- **K6** composed behaviour: AUROC of the *increase* in P(LOOKUP) (X_lost vs Y, matched pairs), INT-S → controller, c = 0.3. This is "compositional zero-shot transfer".

### Family F: familiarity (two-sided, Holm)

- **F1** familiar-but-lost: P1 restricted to high-exposure matched items.
- **F2** familiarity-only boost: (mean Δs(F) − mean Δs(U2)) / D_nat. The TOST with ±0.10 is reported alongside.
- **F3** unfamiliar but newly learned: AUROC(increase; learned N vs exposure-matched still-unknown U), low-exposure stratum.

### Family S: system-specific coupling (two-sided, Holm)

- **S1:** d′(B̃→B̃) − d′(A→B̃), INT-S.
- **S2:** the same for INT-TP.

B̃ = store B's read states mapped into A's coordinates by ridge regression fit **only** on intact MT-split states (λ = 1.0). A→B (raw) and B→B are reported as references.

## 3. Rival explanations

See `rival_hypotheses.md` (pre-registered, part of the protocol). Joint-pattern readings are pre-registered there.

## 4. System

**Store (first-order).**

- Pre-LN causal transformer: d = 128, 4 layers, 4 heads, MLP 512, max length 8, about 0.9M parameters.
- Training: next-token CE on all sequences.
- AdamW lr 2e-3, cosine schedule to 2e-4, batch 512, weight decay 0.
- **E_store = ⟦CAL⟧ epochs.**
- Checkpoints at {5, 10, 15, 20, 30, 40, 60, 80, 100}% of training.

**Read set R** (frozen on architectural grounds before calibration; D6):

- residual stream at positions s2 (subject enrichment) and [A] (answer extraction), layers 0–4;
- 10 sites × 128 dimensions;
- standardised with the intact store's MT-split per-dimension mean and SD.

**Monitors:**

| Monitor | Input | Network |
|---|---|---|
| IN | Own 64-d embeddings of (s1, s2, r) | MLP 256 → 256 → 1 |
| OUT | Sorted restricted output distribution (32) + entropy | MLP 64 → 64 → 1 |
| INT-{S, T, P, TP} | R (1,280-d) | MLP 256 → 256 → 1 |

- Common to all: GELU, dropout 0.1, sigmoid output, class-balanced BCE, AdamW lr 1e-3, weight decay 1e-4.
- **Exactly 8,000 steps × batch 256** for every monitor (identical optimisation budget).
- Exploratory monitors: OUT-TP, INT+OUT-S, and a shuffled-label INT-S control.

**Controller.**

- MLP (2 → 16 → 1) on (logit p̂, cost c).
- Utility: ANSWER = +1 if correct, −1 if wrong; LOOKUP = 1 − c.
- Trained on the **intact store's CT split only**, c ~ U(0.05, 0.6), 3,000 steps.
- Test operating point: c = 0.3. The Bayes threshold there is p̂ = 0.85.
- Variant **C+in** additionally reads the IN monitor's frozen input embedding (secondary).

**Twin store B.** Same world and data, different initialisation and data order (seed + 5000). Its own monitors (INT-S, INT-TP; plain and aligned) and its own T-FORGET run.

## 5. World and operational familiarity

**World.**

- 3,000 entities with 2-syllable names (80-syllable inventory).
- 1,000 names reserved for interference; 2,400 unused names for input corruption.
- 4 relations × 32 values, assigned uniformly at random, so facts cannot be inferred.
- Per entity: 2 relations → MT (monitor training), 1 → CT (controller training), 1 → EV (evaluation).

**Familiarity is an exposure manipulation, never a hidden-state label.**

- High-exposure entities: 8 mention presentations per epoch. Low-exposure entities: 1.
- Coupled knownness: P(fact trained | high) = 0.8; P(fact trained | low) = 0.3.
- Each trained fact is presented once per epoch regardless of exposure.

**Validation of the manipulation is behavioural:**

- **Name fluency** F(e) = log p_store(s2 | [M], s1).
- Store QC requires the AUROC(high vs low) of fluency to be ≥ 0.80.
- T-FAM must raise boosted entities' fluency by ≥ 0.5 SD relative to controls.
- T-FORGET must leave the fluency of X and Y entities within ±0.1 SD.

**Language rule.** Hidden-state directions that decode exposure are called "exposure-decodable directions", never "familiarity representations".

## 6. Interventions (calibrated values from `calibration_report.md`)

| Intervention | Specification | QC (store-level only) |
|---|---|---|
| **T-FORGET + sham** | `sham_algorithm_spec.md`. lr ⟦CAL⟧, steps ⟦CAL⟧, γ ⟦CAL⟧, β 1.0, retain batch 512 | X lost ≥ 50%; Y lost ≤ 3%; Z lost ≤ 5%; ≥ 8/10 site ratios in [0.8, 1.25]; \|Δfluency\| ≤ 0.1 SD; ≥ 100 matched pairs at caliper ⟦CAL⟧ |
| **T-INTERF** | 1,000 new entities × 2 facts + 8 mentions; no replay; lr ⟦CAL⟧; stop at new-fact accuracy ≥ 0.95 or ⟦CAL⟧ steps | Lost fraction in [0.05, 0.60]; ≥ 60 matched pairs |
| **T-NEW** | N facts + exposure-matched U mentions + retain; lr ⟦CAL⟧, steps ⟦CAL⟧ | N learned ≥ 70%; U correct ≤ 5% |
| **T-FAM** | ⟦CAL⟧ extra mention presentations of F's entities + retain; lr ⟦CAL⟧ | Fluency rise ≥ 0.5 SD vs U2; F correct ≤ 5% |
| Input corruption | 240 base-correct items; entity name → unused name | — |
| Exploratory | T-ACT (attention knock-out, layers 3–4, with matched-norm gradient-orthogonal sham); REPLACE (confident replacement with sham) | — |

- **Fallback ladder** for every intervention: lr × {1, 0.5, 2}, then steps × {1, 2}. At most 6 attempts.
- **Training-time vs test-time separation:**

| | Changes seen in training |
|---|---|
| T history | Acquisition during the store's own training |
| P history | Transient random residual dropout, rate ~ U(0, ⟦CAL⟧) |
| S history | None |

  - **Held-out for all histories:** T-FORGET and T-INTERF. These are the confirmatory endpoints.
  - **Not held-out for T / TP:** T-NEW, which is reported as "seen-type" and is never claimed as held-out for those histories.

## 7. Continuous competence-change analysis (K3, K4)

**Per seed, within one intervened store,** over all base-correct EV items exposed to the intervention (FORGET: X ∪ Y ∪ Z; INTERF: all base-correct EV):

- **Δmonitor** = logit(s_post) − logit(s_pre), clipped at ±8.
- **Δcompetence** = margin_post − margin_pre. Margin = log p(v\*) − max_{v≠v\*} log p(v).
- **Covariates:**
  - 10-site log-displacement profile (activation displacement);
  - exposure class and pre-intervention name fluency (familiarity);
  - logit(s_pre) and margin_pre (item difficulty / pre-level).
- **Statistic:** partial Spearman ρ, i.e. the Pearson correlation of the residuals of rank(Δmonitor) and rank(Δcompetence) after least-squares regression on the ranked covariates plus an intercept.

Design choices:

- **Global model change** is absorbed because the analysis is within one store (intercept).
- **Intervention class** is handled by separate tests (K3, K4).
- **No item is selected on whether it became incorrect.**
- **Inference:** one-sided one-sample t-test of the 20 seed-level ρ values against 0 (MEI 0.10). **Items are never treated as independent replications.**
- **Robustness (planned secondary):** the same model with set-membership indicators (X / Y / Z) added. This is a stricter within-set version.

## 8. Statistical procedures (frozen)

**Seed-level tests:**

- one-sided one-sample t-tests (H0 at chance or 0) for P1, P2, K1, K3, K4, K6;
- paired one-sided t-test on d′ differences for K5;
- TOST for K2 (both one-sided tests at the Holm-adjusted level);
- two-sided t-tests for F and S;
- Wilcoxon signed-rank and seed-bootstrap 95% CIs (10,000 resamples) reported for all.

**Transforms and families:** d′ = √2·Φ⁻¹(AUROC), with AUROC clipped to [.001, .999]. Holm is applied within P, K, F and S separately.

**Decision rules (implemented in `s1/analysis.py`):**

| Endpoint type | Supported | Falsified (strong form) | Otherwise |
|---|---|---|---|
| AUROC | Holm-adjusted p < .05 **and** mean ≥ 0.55 | Bootstrap upper 95% bound < 0.55 | Inconclusive |
| ρ and d′ | Holm-adjusted p < .05 **and** mean ≥ MEI | Upper bound < MEI | Inconclusive |
| TOST (K2) | Equivalent if Holm-adjusted p < .05 | Non-equivalent if \|mean\| > bound | Inconclusive |

**Equivalence bound (D3), justified.**

- The K2 and F2 bound is **±0.10 × D_nat**, where D_nat = (mean INT-S score on known, trained-and-correct, intact EV items) − (mean on unknown, untrained-and-incorrect, items) for that seed.
- *Task geometry:* D_nat is the full natural signal the monitor carries about competence. A sham response below 10% of it cannot reverse any ordinal conclusion about competence and is an order of magnitude smaller than the expected lesion response.
- *Why not a fixed ±0.05:*
  - A fixed bound is scale-dependent.
  - For a poorly separating monitor (e.g., D_nat = 0.1), a fixed ±0.05 bound would declare "equivalence" for a sham response equal to 50% of its competence signal.
  - For a well-separating monitor (D_nat ≈ 0.8), ±0.10 × D_nat ≈ ±0.08, close to the provisionally accepted ±0.05.
- *Controller consequence:* at c = 0.3 the controller's threshold is p̂ = 0.85. A sham shift of 0.1 × D_nat moves items by under one-tenth of the known–unknown gap, so lookup flips on Y are bounded accordingly. This is reported descriptively as the sham lookup false-alarm rate.
- *Precision:* calibration gives no monitor data by design. In simulation with n = 20 seeds, TOST power is ≥ 0.9 when the seed SD of the normalised difference is ≤ 0.05 (`stage1_power_sensitivity.json`). No empirical monitor variance was used.
- The bound was fixed **before** any monitor output existed.

**Developmental interaction (planned secondary; "synergistic developmental interaction").**

- Contrast: I = d′(TP) − d′(T) − d′(P) + d′(S) on TI_FORGET and TI_INTERF, with seed-paired 95% bootstrap CIs.
- The interaction is also reported on the raw AUROC and logit-AUROC scales.
- "Superadditive/synergistic" is claimed **only if**:
  - the CI excludes 0 on all three scales, **and**
  - mean TP > max(mean T, mean P).

## 9. Seeds, power, independence

- **20 confirmatory store seeds:** the first 20 seeds from 1001–1040 passing all store-level and intervention-level QC.
- **Retired seeds:**
  - 12345 (feasibility);
  - 9001–9005 (calibration);
  - unit-test seeds 1–99;
  - fabricated-data seeds (10,000,000+).
- **Sensitivity (simulation; unit = seed):**
  - minimum detectable dz = 0.58 (one-sided α = .05);
  - minimum detectable dz = 0.79 (α = .0083, a conservative bound for families with up to 6 tests).
- **Item counts** improve within-seed precision (AUROC SE ≈ 0.02–0.03) but **never count as replications**.

## 10. Exclusions and failure (frozen)

**Seed-level** (computed by code from store-level quantities before any monitor evaluation):

- store A or B QC fails → replace;
- T-FORGET (A or B) or T-INTERF fails QC after the ladder → replace. The co-primaries require these;
- T-NEW or T-FAM fails → that intervention is excluded for that seed (F tests use the available seeds).

**Item-level:** unmatched items are excluded only from matched-pair endpoints; lucky guesses are never base-correct.

**Study-level stop (no confirmatory claims):**

- fewer than 20 valid seeds in 1001–1040;
- more than 5 sham-machinery failures among the first 25 seeds;
- median natural AUROC of INT-S < 0.65.

**Flags (reported; they change interpretation):**

- median twin level AUROC > 0.60 → level-based metrics are H1-confounded;
- median displacement ratio outside [0.67, 1.5] → P1 and K2 non-confirmatory.

## 11. Reproducibility, registration, order of operations

**Execution command:** `python run_confirmatory.py --config stage1_config_FINAL.yaml --out ../../results/raw/stage1`.

- It refuses to run unless `FROZEN_PROTOCOL.json` (config hash + commit) exists.
- Analysis: `python run_analysis.py --raw ../../results/raw/stage1 --out ../../results/processed/stage1`. Single command; no decisions after unblinding.

**Order of operations:**

1. PI approves this package.
2. Write `stage1_config_FINAL.yaml` hash and commit into `FROZEN_PROTOCOL.json`.
3. Commit.
4. Create an **OSF Registration** (current Registrations/Preregistration workflow, not a mutable project). It contains this protocol, the companion documents, the config, the commit hash and the SHA-256s.
5. Only then run confirmatory seeds.

## 12. Changes from draft v0.1

See `calibration_report.md` §"Changes from the draft protocol".
