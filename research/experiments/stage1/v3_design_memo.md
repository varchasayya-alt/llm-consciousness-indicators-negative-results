# Stage-1 v3 DESIGN MEMO: within-target identification

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Status | **For PI approval. No v3 run has been performed.** The only computations behind this memo are existing results plus a seconds-long, store-only probe of intact dev stores (`diag_v3memo_DC_and_v2_ratios.json`) |
| Supersedes | The X_lost-vs-Y primary identification of `protocol_FINAL_PROPOSED.md` (v1.0-proposed) |
| Evidence base | `calibration_stop_report_C4.md`; `decisions.md` D26–D34 |

## 0. Why this redesign is legitimate (documented)

- **No scientific monitor or controller has ever been trained or evaluated on any store.** Calibration code cannot import monitor, controller or pipeline modules (unit-tested), and no monitor output exists anywhere in the repository.
- **No confirmatory seed (1001–1040) has been run.** `FROZEN_PROTOCOL.json` does not exist. Nothing has been registered publicly.
- **The trigger is a calibration-discovered confound found with store-only diagnostics** (C4val fingerprint failure; the procedure-membership signature; D34). It is not an observed monitor effect.
- **The redesign is PI-directed (2026-10-02).** Option C is rejected. The 0.90 fingerprint gate is **neither relaxed nor reinterpreted**. It stays on record as the reason the v2 Y sham cannot be the primary identification control. Y's change of role (§6) follows from the redesign; it does not reopen the gate.
- **The read set is unchanged.** No layers are removed. The late-[A] fingerprint is handled by the design (§1) and by measured adjustment (§4), not by representational selection.

## 1. Why within-target identification separates competence monitoring from procedure detection

**The calibration evidence** (L4-[A] direction feature; selected v2 recipe):

| Comparison | Separation (AUROC) |
|---|---|
| X_lost vs Y | 0.87–0.88 |
| X_retained vs Y | 0.81–0.87 |
| X_lost vs X_retained | 0.58–0.61 |

Within X, the feature's Spearman correlation with Δmargin is only 0.05–0.16. The generic signature is therefore close to a **targeting indicator** I_i that is essentially constant within X.

**Why X_lost vs Y fails:** every positive item is targeted and every negative is not, so outcome and procedure are perfectly confounded.

**Why within X works:** within X, I_i = 1 for every item. Any monitor component driven by "was targeted" is a constant within the store. It cannot create an item-by-item association between ΔM and ΔC. Only item-varying components can. The remaining procedure-side confound is item-varying *effective intensity*, which is measured (displacement, fingerprint score) and adjusted for (§4).

**The dose is fixed for all X items.** A randomised per-item dose was considered and rejected for the primary analysis: dose would by design drive both ΔC and the fingerprint.

**Predictions of the rival hypotheses for the primary estimand θ:**

| Rival | Prediction |
|---|---|
| H1 (input difficulty) | ≈ 0. Inputs are unchanged, and static input properties are absorbed by the pre-level covariates |
| H2 (intervention/anomaly detection) | ≈ 0 once effective intensity is adjusted for |
| H3 (competence monitoring) | > 0 |

**This also repairs v1/v2's continuous analysis (old K3).** It pooled X ∪ Y ∪ Z, so ΔC differences partly encoded procedure membership.

## 2. Causal ambiguity that remains

| # | Ambiguity | Handling |
|---|---|---|
| a | **Intensity confound.** The nominal procedure is the same, but the effective perturbation varies by item. Items that move more may both lose more competence and look more "changed". Generic statistics at the answer-computing sites are partly competence-bearing. Adjusting for all of them risks *over-adjustment* (conservative); adjusting for few risks residual intensity confounding | Pre-registered primary uses a low-dimensional intensity adjustment (D_i, F_i). Full generic adjustment is a pre-specified robustness analysis. A fixed interpretation table covers both (§4) |
| b | **Unmeasured susceptibility** that drives both forgetting and the monitor's response | Reduced by pre-intervention covariates, including a pre-state susceptibility predictor (§5). Not eliminated |
| c | **Output readout vs privileged internal access (E2).** Within-X tracking can be implemented by reading output-proximal states. Under uniform-target forgetting, output entropy separates lost from sham items at AUROC 1.0, so OUT will also track ΔC | P1 separates H3 from H2, **not** internal from first-order access. That stays with P2 (INTERF, naturally arising errors) and the INT-vs-OUT comparisons |
| d | **Shared computation.** ΔC is produced by the same forward pass whose intermediate states the monitor reads. A monitor that partly recomputes the readout tracks ΔC "for free" | Within the Level-1 meaning of competence monitoring. Claims say "tracks the store's state-dependent competence" and nothing stronger |
| e | **One intervention family.** Uniform forgetting yields low-confidence errors | Generalisation is P2's role |
| f | **Binary secondary:** selection on which items resist forgetting | Reduced by matching on measured pre-intervention variables only. Stated as residual |

## 3. Exact primary estimand

**Population.** Per store seed s: the targeted set X_s. These are 300 base-correct EV items (150 high / 150 low exposure), randomly assigned, all receiving the identical T-FORGET v3 procedure at a fixed dose.

**Definitions.**

- Competence: C_i = log p(v\*_i | q_i) − max_{v≠v\*} log p(v | q_i), over the relation's 32 values (nats). ΔC_i = C_i^post − C_i^pre.
- Monitor change: ΔM_i = clip(logit s_post − logit s_pre, ±8), for monitor m = INT-S.
- Per-seed estimand: θ_s = partial Spearman ρ(ΔM_i, ΔC_i | W_i, V_i) over i ∈ X_s.
  - W_i = pre-intervention covariates.
  - V_i = procedure-intensity covariates.
- Population estimand: **θ = E_s[θ_s]**.
  - H0: θ ≤ 0. H1: θ > 0.
  - Minimum effect of interest: **0.10**.

**Interpretation.** Because C_pre is in W, the estimand is (for linear partialling) the association between post-intervention monitor score and post-intervention competence beyond everything known before the intervention. In other words: does the monitor read the store's *current* state?

**P2 (T-INTERF), analogous.** The population is all base-correct EV items. All are equally exposed and there is no targeting, so V_i = D_i only.

## 4. Statistical analysis (fixed before any monitor exists)

**Covariates.**

- **W_i (pre-intervention):**
  - C_pre;
  - log p_pre(v\*);
  - logit M_pre (the outcome's own baseline; used only in regression, **never in matching or selection**);
  - exposure class;
  - pre-intervention name fluency;
  - relation (3 indicators);
  - **P_i, the susceptibility score:** a cross-fitted ridge prediction of ΔC_i from the item's standardised pre-intervention read states (1,280-d) plus the other W scalars.
    - 5-fold within X_s, stratified by exposure.
    - λ chosen by generalised cross-validation inside each training fold over the fixed grid 10^{-2…4}.
    - Store-only; never sees a monitor.
- **V_i (post-intervention, generic, store-only):**
  - **D_i** = mean log read-site displacement over the 10 sites.
  - **F_i** = cross-fitted fingerprint score: the out-of-fold logit of the `s1/diagnostics` L2 logistic classifier (λ = 1) on the 40 generic features (per-site log displacement, cos(Δ, h_pre), relative norm change, abnormality change), trained to separate targeted X from displacement-matched Y. 5-fold.

**Statistic.**

1. Rank-transform ΔM, ΔC and the continuous covariates (indicators untransformed).
2. Residualise the ΔM ranks and ΔC ranks on [1, covariates] by least squares.
3. θ_s = Pearson correlation of the two residual vectors (existing `metrics.partial_spearman`).

**Inference (unit = store seed, n = 20; items are never replications).**

- z_s = atanh θ_s.
- One-sided one-sample t-test of z_s.
- Wilcoxon signed-rank test and seed-bootstrap 95% CI reported alongside.
- Holm correction within family P (P1, P2).
- **Supported:** Holm p < .05 **and** mean θ ≥ 0.10.
- **Falsified (strong form):** upper bootstrap bound < 0.10.
- **Otherwise:** inconclusive.

**Power.** The within-seed SE of θ_s is about 1/√(300 − 14) ≈ 0.06. At n = 20 seeds with seed SD ≥ 0.06, 80% power is reached at mean θ ≈ 0.04–0.05, so MEI 0.10 is comfortably detectable. This will be confirmed with the existing power script.

**Pre-specified robustness analyses** (reported; not used for the decision):

| Code | Variation |
|---|---|
| R-a | V = the full 40 generic features |
| R-b | No V |
| R-c | ΔC = Δlog p(v\*) |
| R-d | Mixed model: rank ΔM ~ rank ΔC + covariates + (1 + rank ΔC \| seed) |

**Pre-registered interpretation:**

| Pattern | Reading |
|---|---|
| Primary > 0 **and** R-a > 0 | Tracking beyond procedure intensity and beyond all generic change statistics. Strongest H3 support |
| Primary > 0, R-a ≈ 0 | Not separable from generic change structure at the read sites. Claim limited to "change-structure-sensitive competence tracking" |
| R-b > 0, primary ≈ 0 | Explained by procedure intensity: **H2** |

## 5. Controlling baseline forgetting susceptibility

1. The within-item Δ removes the main effects of static item properties.
2. W covariates: C_pre, log p_pre, exposure, fluency, relation, M_pre.
3. **P_i** absorbs fragility that is predictable from the same states the monitor reads, taken *before* the intervention. The monitor gets no credit for recognising pre-existing fragility.
4. Random, exposure-stratified assignment to X, with the store seed as the replication unit.
5. Diagnostic R²_pre: the cross-fitted R² of ΔC from pre-intervention information (gate F4, §8).
6. **Binary secondary (K1):** X_lost vs X_retained, matched 1:1 on a propensity score built from **pre-intervention variables only** (C_pre, log p_pre, exposure, fluency, relation, P_i; never monitor outputs and never M_pre).
   - Nearest neighbour, without replacement, seeded random order.
   - Caliper 0.2 SD of the logit propensity; exact match on exposure class.
   - Requirements: every standardised mean difference ≤ 0.10, and ≥ 50 pairs.
   - Endpoint: seed-level AUROC(−ΔM) vs 0.5; MEI 0.55.
   - Interpreted as **supporting evidence only**.

## 6. Role of the old Y sham

- **Kept unchanged** (forget-then-relearn, displacement-magnitude matched) as a **negative control** for: "does the monitor respond merely because an internal state moved by a matched amount, without competence loss?"
- **K2:** TOST that mean ΔM(Y) − mean ΔM(Z) lies within ±0.10 × D_nat.
- **Recorded limitation:** Y differs from X in the direction of change at [A] L3–4 (matched G2 ≈ 0.85–0.90). Y therefore does not control direction-specific procedure detection; the within-X design does. Y's G1–G4 are reported per seed. Y no longer gates the study.
- **Old P1** (X_lost vs matched Y) becomes a descriptive secondary, labelled "fingerprint-confounded". **Old K1** (X_lost vs Z) is confounded in the same way and is also demoted to descriptive.
- **To remain a valid negative control, Y must still pass:**
  - binary lost ≤ 3%;
  - ≥ 8/10 site displacement ratios within [0.8, 1.25];
  - the continuous retention criteria of §7.

## 7. New continuous retention criterion (Z and Y)

**Metrics.** Computed per category c ∈ {Y, Z_random, Z_near_entity_X, Z_near_entity_Y, Z_near_repr, Z_far} and per seed:

- mean Δmargin;
- mean Δlog p(v\*);
- mean KL(p₀ ‖ p₁) on the restricted answer distribution.

Each is reported absolute, relative to X, and relative to the category's own pre-intervention margin.

**Gate (every category, every seed):**

1. **Selectivity:** |mean Δmargin_c| / |mean Δmargin_X| ≤ 0.10. The same ratio bound applies to Δlog p, and mean KL_c / mean KL_X ≤ 0.10.
2. **Absolute anchor:** |mean Δmargin_c| ≤ 0.10 × D_C.
   - D_C = mean margin of base-correct known EV − mean margin of unknown EV in the intact store (seed-specific).
   - Dev stores: D_C = 11.9–12.5 nats, so the bound is about 1.2 nats.
3. **Binary:** lost ≤ 5% (unchanged).

**Justification.**

- (1) puts "substantially smaller" into practice as an order-of-magnitude separation between targeted and collateral change on every scale.
- (2) mirrors the protocol's scale-free ±0.10 × D_nat equivalence logic. Collateral drift below 10% of the natural known–unknown gap cannot reverse ordinal competence relations, and it keeps Y and Z usable as near-unchanged references for K2.
- Sensitivity is reported at 0.05, 0.10 and 0.20. The same anchor (2) applies to T-NEW and T-FAM collateral.

**Transparency check.** This was computed from existing v2 data *after* the threshold was chosen on the grounds above.

- v2's selected recipe **fails**: Z/X margin ratio 0.26–0.38; absolute Δmargin 1.4–2.2 nats (> 1.2). Its Δlog p and KL ratios pass (0.02–0.05).
- Across the **entire** v2 grid, the best worst-category margin ratio is 0.28. It barely depends on dose (0.28–0.39 from 40% to 80% X loss).
- Retain-pool facts stay untouched (Δmargin ≈ 0).
- This "patched only where supervised" pattern is consistent with the distortion of non-target representations reported for knowledge editing in synthetic transformers [nishi2025shattering].
- **Implication:** the v2 mechanism will likely not meet this criterion. v3 development must test pre-declared mechanism variants (§11), and a failure there feeds §10.

## 8. Conditions under which v3 still fails

All of these are pre-declared and store-only. They are evaluated on the dev seeds, then once on fresh validation seeds 9021–9023.

| Code | Failure condition |
|---|---|
| F1 | No variant meets §7 |
| F2 | **Outcome diversity:** X lost fraction outside [0.35, 0.65], or IQR of C_post < 1 nat. Both binary groups are needed and outcome variance should be large |
| F3 | **Generic determinacy:** cross-fitted R²(ΔC \| 40 generic features) ≥ 0.90 (median over seeds). The fingerprint practically determines the competence change, so no dissociation is possible |
| F4 | **Susceptibility determinacy:** cross-fitted R²(ΔC \| pre-intervention information including P_i) ≥ 0.90. The outcome is fixed before the intervention |
| F5 | **Joint:** R²(ΔC \| pre + generic) ≥ 0.95 |
| F6 | Binary secondary infeasible: fewer than 50 balanced pairs |
| F7 | Y fails its QC (§6), so the negative control is unusable |
| F8 | The selected recipe fails any of F1–F7 on the fresh validation seeds (fallback ladder allowed) |

- The 0.90/0.95 thresholds are a priori conventions, consistent with the existing 0.90 "serious" diagnostic convention.
- Within-X diagnostics also reported: Spearman(F_i, ΔC_i), R²_gen, R²_pre and R²_pre+gen.
- **At the confirmatory stage**, there is one new study-level stop in addition to the existing ones: if the median within-seed R²_gen ≥ 0.90, P1 is declared non-identifiable and no claim is made.

## 9. Does Option A add identification?

**Little.** The within-X design holds the procedure fixed by construction, which is what A tries to approximate for a separately built sham.

A forces Y's late-[A] state to move *against* its pre-state while keeping the answer. At those sites the pre-state is dominated by answer evidence, so there are two likely outcomes:

- A successful direction match lowers Y's margin. Y becomes an "X_retained-like", partially forgotten item, which the within-X design already contains naturally.
- Or the match is achieved through non-answer components engineered at exactly the states the monitor reads. That is the overmatching problem you flagged.

**Recommendation:** keep A as a low-priority exploratory branch, run only after v3 feasibility.

- Development seeds only.
- A pre-declared raw-Δ probe (G4Δ) as its diagnostic.
- §7 applied to Y: if direction matching costs Y more than §7 allows, A defeats itself.

## 10. Conditions that force a move to Option B

Move to B if, after the pre-declared v3 dev grid (all mechanism variants), any of the following holds:

- no variant meets §7 together with F2 (retention plus outcome diversity) on the dev seeds;
- F3, F4 or F5 holds;
- the selected v3 recipe fails fresh validation. Only **one** revision cycle is allowed, to avoid serial adaptation.

**What B offers:**

- item-local deletion, giving zero collateral by construction and exact ΔC;
- graded dose (scaling a slot's value);
- a natural sham (a norm-matched perturbation of unused slots).

**What B costs:** a more synthetic memory, narrower generalisation claims, and fresh store calibration.

## 11. Proposed next steps (only after approval of this memo)

**A. Write and commit before any run** (your steps 1–10):

- revised protocol: estimand, model, QC, Y role, fingerprint diagnostic, failure criteria;
- config;
- code: P_i, F_i, R² diagnostics, continuous QC and matching;
- unit tests, including simulated data with a known θ and known R² structure;
- an updated fake-data dry run.

**B. Pre-declared mechanism variants for the v3 dev grid.** The procedure is identical across X in every variant.

| Variant | Description |
|---|---|
| V0 | v2 mechanism (baseline) |
| V1 | V0 + parameter-space anchor λ_P‖θ − θ₀‖². This generalises the "do not change" pressure to unsupervised facts |
| V2 | Retain pool enlarged to all known MT+CT facts except those of X/Y entities. Near-entity QC stays held out |
| V3 | Localised edit: only the layer-1–2 MLP weights trainable. Architectural rationale from locate-and-edit work [meng2022rome]; not monitor-based |

**C. Grid, selection and seeds.**

- Grid: lr × steps per variant.
- Selection: among cells passing §7, F2–F7 and Y QC on all dev seeds:
  1. smallest worst-case §7 ratio;
  2. then lost fraction closest to 0.5;
  3. then smaller lr × steps.
- Seeds: dev 9001–9003 (diagnostic 9004–9005); fresh validation **9021–9023**, used once; 9011–9013 retired.

**D. Compute and restrictions.**

- Estimate about 3–4 CPU-hours plus about 20 minutes for the validation stores.
- **No monitor training** until the store and intervention pass the v3 criteria.

**E. Proposed confirmatory family structure** (for your decision):

| Family | Content |
|---|---|
| P | **P1** θ_FORGET (within X); **P2** θ_INTERF (within exposed) |
| K | **K1** matched X_lost vs X_retained; **K2** Y negative control (TOST); **K3** developmental Δz(θ), INT-TP − INT-S; **K4** composed behaviour, θ for ΔP(LOOKUP) within X ("compositional zero-shot transfer") |
| F | **F1** θ_FORGET in the high-exposure stratum; **F2/F3** unchanged |
| S | θ-based B̃→B̃ vs A→B̃ |

Descriptive only: old X_lost-vs-Y, X_lost-vs-Z, R-a to R-d, per-seed fingerprints, INTERF matched binary.

**Decisions needed from you:**

1. Approve the within-X primary estimand and statistic (§3–4).
2. Approve the §7 thresholds.
3. Approve mechanism variants V0–V3.
4. Make **P2** continuous (my proposal, for coherence) or keep the matched binary.
5. Reuse dev seeds 9001–9003, or start fresh dev seeds (e.g. 9031–9033)?
