# Stage-1 calibration plan (written and committed BEFORE any calibration seed is run)

**Date:** 2026-10-01.

**Seeds:** 9001–9005. These are methods-development seeds only, and they are retired permanently after calibration.

## Allowed

Calibration may be used to set or check:

- store training length;
- P-family rate range (only to avoid degeneracy);
- intervention strengths and hyperparameters (T-FORGET + sham, T-INTERF, T-NEW, T-FAM);
- the matching caliper;
- sham feasibility;
- generic-distinguishability diagnostics;
- runtime;
- numerical stability;
- environment behaviour (QC quantities).

## Forbidden

- Training or evaluating any metacognitive monitor or controller. This is enforced by a unit test: `run_calibration.py` and `s1/interventions.py` may not import `s1.monitors`, `s1.controller` or `s1.pipeline`.
- Choosing read sites, monitor architectures, metrics or intervention classes based on anything monitor-related.
- Estimating any confirmatory effect.

## Read set (frozen now, on architectural grounds — D6)

- **Positions:** s2 (subject enrichment) and [A] (answer extraction).
- **Layers:** 0–4.
- **Not subject to calibration.**

## Pre-declared decision rules

These are verbatim from the `run_calibration.py` docstring.

| Phase | Rule |
|---|---|
| C1 | Train 9001 and 9002 for 200 cosine epochs. E99 = first evaluated epoch with trained-fact accuracy ≥ 0.99. **E_store = min(200, ceil10(1.5 × max E99))** |
| C2 | Train all five calibration stores with E_store. Store QC must pass on all five |
| C3 | Keep the dropout rate ~ U(0, 0.6), unless mean MT accuracy at rate 0.6 exceeds 0.9 (then raise the max to 0.8) or mean accuracy at rate 0.1 is below 0.5 (then lower the max to 0.3) |
| C4 | Grid: lr {5e-4, 1e-3, 2e-3} × steps {100, 200, 400} × γ {1, 10, 100}, on 9001–9003. **Eligible:** configs passing all T-FORGET QC criteria *except* the matched-pair count, on all three seeds. **Select:** the eligible config minimising mean_seed max_site \|log(D̄_X/D̄_Y)\|. **Tie-break:** smaller lr × steps |
| C5 | Caliper = smallest value in {0.25, 0.5, 0.75, 1.0} giving ≥ 100 FORGET pairs and ≥ 60 INTERF pairs on all five seeds |
| C6 | The selected T-FORGET config must pass full QC on held-out calibration seeds 9004–9005 |
| C7 | T-INTERF: lr {3e-4, 1e-3, 3e-3}, max 3,000 steps, stop at new-fact accuracy ≥ 0.95. Choose the lr whose mean lost fraction is closest to 0.25, subject to lost fraction in [0.10, 0.50] on all of 9001–9003. max_steps = ceil100(1.5 × max steps used) |
| C8 | T-NEW: lr {5e-4, 1e-3, 2e-3} × steps {50, 100, 200}. Choose the smallest lr × steps passing QC plus collateral loss on base-correct EV ≤ 5%, on 9001–9003 |
| C9 | T-FAM: lr {5e-4, 1e-3, 2e-3} at 16 presentations; if none passes, try 32, then 64. Choose the smallest passing |
| C10 | Sham validation with the separate diagnostic classifier (G1–G4, `s1/diagnostics.py`), on all five seeds |

**C10 thresholds** (CV AUROC, forgotten vs sham):

| CV AUROC | Classification | Action |
|---|---|---|
| ≤ 0.75 | Acceptable | None |
| 0.75–0.90 | Moderate | Document; generic features enter the continuous analysis as covariates |
| > 0.90 on G1–G3 | **SERIOUS** | Redesign the sham before freezing |

- G4 (raw-state probe) on **X_retained vs Y** is the procedure-fingerprint test.
- G4 on X_lost vs Y is reported only as a competence-informed reference.

## If a rule cannot be satisfied

The failure is documented as a methods-design failure. The algorithm is revised (never the hypotheses), the revision is logged in `logs/decisions.md` with its reason, and the affected phase is re-run on the same calibration seeds. Every attempt is reported in the calibration report.

---

## Revision v2 (2026-10-01). Committed before any v2 calibration result

**Trigger:** the C4 v1 methods failure (decisions D26). The PI approved a methods redesign and required this revision to be versioned and committed before C4 is re-run.

**Reasons and full rationale:** `decisions.md` D27–D31.

**Unchanged:** all hypotheses, endpoints, monitors, the read set, analyses and claim boundaries.

### Seeds

| Seeds | Role |
|---|---|
| 9001–9003 | Development |
| 9004–9005 | Diagnostic only (not fresh) |
| 9011–9013 | Fresh validation, used ONCE |

All of these are permanently excluded from confirmatory data.

### Order (stop at the first failed acceptance criterion)

1. **C3r:** corrected P-family rule (D29).
2. **C4v2:** T-FORGET v2 development grid on 9001–9003 (D30). If no eligible cell remains after contingencies A' and B, **stop**.
3. **C2val:** train 9011–9013 with E_store = 60. Store QC must pass.
4. **C4val:** the frozen T-FORGET recipe once on 9011–9013. If it fails, **stop**.
5. **C7:** T-INTERF.
6. **C8:** T-NEW (v2).
7. **C9:** T-FAM (v2).
8. **C10:** sham fingerprint diagnostics and caliper selection on 9001–9005.
9. **VAL:** the complete frozen recipe once on 9011–9013.

### T-FORGET v2 gates (every dev seed; thresholds unchanged from v1 except the new categories)

| Gate | Criterion |
|---|---|
| X lost | ≥ 0.50 |
| Y lost | ≤ 0.03 |
| Collateral | Lost ≤ 0.05 in **each** of Z_random, Z_near_entity_X, Z_near_entity_Y, Z_near_repr (top 20% read-state similarity to X), Z_far (bottom 20%) |
| Displacement match | ≥ 8/10 site ratios D̄_X/D̄_Y in [0.8, 1.25] |
| Fluency | \|Δfluency(X ∪ Y entities)\| ≤ 0.1 SD. Held out: X/Y entities are excluded from the mention retain |
| Fingerprint | Some caliper in {0.25, 0.5, 0.75, 1.0} gives ≥ 100 matched pairs and G1–G3 matched-pair CV AUROC ≤ 0.90 on every seed |

### Grid and selection

- **Grid:** see D30 (Stage A, contingency A', contingency B).
- **Selection:**
  1. Minimal worst-category collateral, with ties within 0.01.
  2. Then best displacement match.
  3. Then smaller lr × steps.
  4. Then smaller λ_R.
- Monitor scores are never computed.

### Downstream

- C7 is unchanged.
- C8 and C9 follow D31.
- C10 caliper rule: smallest c with FORGET pairs ≥ 100, INTERF pairs ≥ 60 and FORGET G1–G3 matched AUROC ≤ 0.90 on every QC-passing seed.
- VAL pass condition: every intervention passes its QC (within the fallback ladder) on every validation seed, at the selected caliper.

### If any acceptance criterion fails

- Record it, stop the chain (exit code 3), and report to the PI.
- Thresholds are never relaxed to make a phase pass.
- A new method version would require a new committed revision and an untouched validation set (9021+).

---

## Revision v3 (2026-10-02). Committed before any v3 store or intervention run

**Basis:**

- the PI's v3 decisions (2026-10-02);
- `experiments/stage1/v3_design_memo.md`;
- `experiments/stage1/protocol_v3_PROPOSED.md`;
- `decisions.md` D36–D41.

**Unchanged:** hypotheses' substance, monitors, read set, store, world.

**Changed:**

- the primary identification (within-target);
- the statistic (simulation-selected);
- the intervention QC (continuous retention, diversity, identifiability, binary feasibility);
- the role of Y (negative control).

### Seeds

| Seeds | Role |
|---|---|
| 9031–9033 | Development |
| 9021–9023 | Fresh validation, used once per component |
| 9001–9005, 9011–9013 | Historical only |

### Order (stop at the first failed acceptance criterion; exit code 3)

1. **C2dev:** train 9031–9033 (E_store = 60). Store QC must pass.
2. **C4dev:** the T-FORGET mechanism family V0–V3 (36 cells; D40) on 9031–9033.
   - **Eligible:** every dev seed passes every per-seed gate (D38) **and** F6 on every dev seed **and** median identifiability within thresholds.
   - **Select:** smallest worst-seed retention load; ties within 0.05 → mean X lost closest to 0.5 → smaller lr × steps.
   - **If none is eligible: STOP** (report; Option B pathway).
3. **C2val:** train 9021–9023.
4. **C4val:** the frozen recipe once on 9021–9023 (fallback ladder allowed). Per-seed gates + F6 per seed + identifiability medians. **Failure: STOP** (no retuning on 9021–9023).
5. **C7:** T-INTERF on dev seeds.
   - Grid: lr {3e-4, 1e-3, 3e-3}, ≤ 3,000 steps, stop at new-fact accuracy ≥ 0.95.
   - Eligible: lost ∈ [0.10, 0.50] on every dev seed **and** P2 identifiability medians within thresholds.
   - Choose the mean lost closest to 0.25. max_steps = ceil100(1.5 × max steps used).
6. **C8:** T-NEW on dev seeds.
   - Grid: lr {5e-4, 1e-3, 2e-3} × steps {50, 100, 200} × λ_R {1, 10}.
   - Eligible: v3 QC, including the continuous collateral anchor.
   - Choose the smallest lr × steps; ties → smaller λ_R.
7. **C9:** T-FAM on dev seeds.
   - Grid: lr {5e-4, 1e-3, 2e-3} at 16 presentations, then 32, 64.
   - Eligible: v3 QC. Choose the smallest passing.
8. **C10:** on dev seeds.
   - The Y negative-control fingerprint (reported) and INTERF diagnostics.
   - Caliper for the *descriptive* matched contrasts: the smallest c with ≥ 100 FORGET and ≥ 60 INTERF pairs on all dev seeds, else 1.0. Not a gate.
9. **VAL:** the complete frozen recipe once on 9021–9023. Every per-seed QC + F6 + P1 and P2 identifiability medians.
10. **STOP.** Return the calibration package to the PI. **No monitor training, no confirmatory seed, no public registration** before PI approval.

### If any acceptance criterion fails

- Record it, stop, and report to the PI.
- Thresholds are never relaxed after results.
- No variant outside V0–V3 is added in this cycle.
- A failure of retention + diversity or of identifiability is the pre-declared trigger for considering Option B.

---

## Revision v4 (2026-10-03). Option B1 feasibility ladder, committed before any v4 run

**Basis:**

- `experiments/stage1/v4_design_memo.md` (approved);
- the PI's refinements of 2026-10-03;
- `decisions.md` D45–D48.

**Unchanged:** the v3 within-target estimand, statistic, analysis and claim boundaries.

### Seeds

| Seeds | Role |
|---|---|
| 9041–9043 | Development (all selection) |
| 9051–9053 | Fresh validation, used ONCE after freezing |
| 9021–9023 | Retired unused |

### Ladder (stop at the first failure; exit code 3)

1. **V4-F0** (9041, p_rd = 0.35, 60 epochs):
   - integrated trained-fact accuracy ≥ 0.98;
   - unknown accuracy ≤ 0.10;
   - fluency AUROC ≥ 0.80;
   - own-slot attention ≥ 0.90.

   If the trained-fact gate is missed at 60 epochs, apply the C1 rule on 9041: 200 epochs, E = min(200, ceil10(1.5 × E99)).
2. **V4-F1** (9041 × p_rd ∈ {0.2, 0.35, 0.5}). Eligibility:
   - store QC passes;
   - ablation A accuracy on covered base-correct EV ∈ [0.30, 0.70], with IQR(A margin) ≥ 1 nat;
   - M1 ≥ 0.90 or M2 ≥ 0.90 (a disagreement is reported);
   - integrated accuracy on parametric-only trained facts ≥ 0.90.

   **Select** the A accuracy closest to 0.5, then **confirm** on 9042 and 9043 with the same gates.
3. **V4-F2** (9041–9043; T-DELETE(X) + Y transplant):
   - lost ≤ 0.05 in each Z category: Z_random, Z_param, Z_near_entity_X/Y, Z_near_key, Z_far;
   - continuous retention at 0.10 (margin, log p, KL; |Δmargin| ≤ 0.10 D_C);
   - read-site displacement of each Z category ≤ 0.10 × X's;
   - |Δfluency| ≤ 0.1 SD;
   - retrieval leakage μ_X reported.

   **Y QC** (lost ≤ 0.03; continuous retention): a failure means stop and report (K-7).
4. **V4-F3:**
   - X lost ∈ [0.35, 0.65];
   - IQR(C_post) ≥ 1 nat;
   - each of retained-strong, weakened and lost ≥ 15%;
   - exposure alone: R² ≤ 0.50 and AUROC ≤ 0.80;
   - approved F6 binding (F6′ recorded).
5. **V4-F4:**
   - F3–F5 medians: R²_gen < 0.90, R²_pre < 0.90, R²_joint < 0.95;
   - trivial-decoder audit, with a kill if within-X R²(C_post | bookkeeping) ≥ 0.90;
   - the R² decomposition and incremental terms are reported.
6. **P2** (only if F0–F4 pass): parametric-route interference with the memory frozen.
   - lr grid {3e-4, 1e-3, 3e-3}; ≤ 3,000 steps; stop at new-fact accuracy ≥ 0.95.
   - Population: parametric-only base-correct EV facts.
   - Eligible: lost ∈ [0.10, 0.50] on every development seed, plus P2 identifiability.
   - Choose mean lost closest to 0.25; max_steps = ceil100(1.5 × max steps used).
7. **VAL:** the frozen recipe once on 9051–9053 (stores; F0/F1 gates; F2–F4; P2). Then **STOP**: no monitor training before PI approval.

### Rules

- No thresholds are relaxed after results.
- No v3 parametric variants are reintroduced.
- Validation seeds are never used for any choice.

## Revision v4.1 (2026-10-03). Gradient-isolated learning routes; committed before any v4.1 development run

**Basis:**

- the PI's approval of O1 (2026-10-03);
- `decisions.md` D52–D55;
- `experiments/stage1/calibration_stop_report_v4_F1.md` (v4 stopped at V4-F1, K-2).

**Changed:**

- The learning routes (D52). Covered facts with their slot available train only the memory group; all other LM loss trains only the parametric group; retrieval CE trains only addressing.
- The F1 p_rd grid and the dose-coherence rules (D53).
- The inferential status of label B (D54).

**Unchanged:** everything else in Revision v4, including:

- the architecture;
- F0, the F1 eligibility gates and the selection rule;
- F2, F3, F4 and the audit;
- P2;
- the VAL procedure.

### Seeds

| Seeds | Role |
|---|---|
| 9061 | Development: the full p_rd grid and mechanism selection |
| 9062, 9063 | Development: confirmation of the single frozen p_rd; then F2–F4 and P2 together with 9061 |
| 9051–9053 | Fresh validation, still untouched. Used ONCE, only after F0–F4 and P2 pass and the recipe is FROZEN |
| 9041–9043 | v4 history only. Never used for v4.1 |

### Ladder (stop at the first failure; exit code 3)

0. **Pre-run (done before 9061):**
   - code;
   - the exact parameter partition;
   - gradient-isolation tests;
   - config (version 0.6-v4.1);
   - this revision and the D54 label policy;
   - the full unit suite;
   - an engineering dry run on burned seed 12345 only;
   - commit.
1. **V4.1-F0** (9061, p_rd = 0.35, E = 60):
   - the same store QC as V4-F0;
   - the isolation invariants asserted during training;
   - the C1 rule if trained accuracy is below 0.98.
2. **V4.1-F1 grid** (9061 × p_rd ∈ {0.05, 0.10, 0.20, 0.35, 0.50}, fixed; nothing added after results):
   - the F1 gates and the selection rule as in v4 (A ∈ [0.30, 0.70] closest to 0.50, IQR ≥ 1 nat, memory sufficiency, store QC, parametric-only answerability);
   - the dose report for every store.

   Dose coherence:
   - **DC-1:** for all p_i < p_j, acc_A(p_j) ≥ acc_A(p_i) − 0.05, and acc_A(0.50) − acc_A(0.05) ≥ 0.20;
   - **DC-2** at the selected p_rd: Spearman ρ(route-drop count, route-A margin) > 0, one-sided p < 0.01.

   Stops:
   - none eligible → **K-2 STOP**;
   - eligible but DC-1 or DC-2 fails → **K-2d STOP**.
3. **V4.1-F1 confirmation** (9062, 9063 at the frozen p_rd): the F1 gates and DC-2. Any failure → **STOP**.
4. **V4.1-F2, F3, F4** (9061–9063): unchanged from Revision v4 (locality and Y; diversity; identifiability and the trivial-decoder audit).
5. **P2** (development seeds; only after F0–F4 pass): unchanged from Revision v4.
6. **FREEZE:** record the recipe (E_store, p_rd, P2 lr and max_steps) and the code commit; then commit.
7. **VAL** (9051–9053, ONCE): stores, F0 QC, F1 gates and DC-2, F2–F4, P2 at the frozen recipe. Then **STOP before any monitor training.**

### Additional kill conditions (v4.1)

- Gradient isolation cannot be implemented cleanly, or memory-present covered examples alter P (unit tests and the runtime audit).
- No meaningful aggregate dose-response (DC-1), or no item-level dose-response at the selected p_rd (DC-2).
- No fixed grid value is F1-eligible (K-2).
- The selected p_rd does not confirm on 9062 or 9063.
- Any existing F2–F4 or P2 gate fails.

**If v4.1 fails K-2 (or K-2d):** no automatic move to O2 or O3. Stop and report.

### Rules

- No thresholds are relaxed after results.
- No grid values are added.
- Validation seeds are never used for any choice.
- No monitor is trained.
- No confirmatory seed is run.


## Revision v4.2 (2026-10-03). Sequential development, the FINAL B1 revision. Committed before any v4.2 development seed

**Basis:**

- the PI's approval of O4 (2026-10-03);
- `decisions.md` D57–D60;
- `experiments/stage1/calibration_stop_report_v41_F0.md` (v4.1 stopped at V4.1-F0, K-1).

**Changed:** the development procedure only.

- **Stage A:** parametric route alone; memory off; covered-fact sequences included per epoch with probability p_rd; E_A = 60 fixed.
- **Stage B:** every P tensor frozen; only the memory group trains; E_B = 200 fixed.
- **NULL value:** fixed at exactly zero.
- **Injection cap:** κ = 10 × the frozen network's median ‖hA‖.

**Unchanged:**

- architecture and read set;
- T-DELETE and Y;
- every F2–F4 and P2 gate;
- the estimand, statistic, unit and monitor restrictions;
- the D54 label policy (A = possible affirmative support; B = ambiguous, not support; C = no evidence).

### Seeds

| Seeds | Role |
|---|---|
| 9071 | F0A grid and selection; F0B |
| 9072, 9073 | F1 confirmation of the frozen p_rd (no reselection); then F2–F4 and P2 with 9071 |
| 9051–9053 | Fresh validation, untouched. Used ONCE after FREEZE |
| 9041–9043, 9061 | History only |
| 9062, 9063 | Retired unused |

### Ladder (exit code 3 = stop; any v4.2 stop ends the B1 line)

1. **V4.2-F0A** (9071, Stage A only, p_rd ∈ {0.05, 0.10, 0.20, 0.35, 0.50}). For each p_rd:
   - Route-A accuracy on covered trained EV facts ∈ [0.30, 0.70] and IQR ≥ 1 nat;
   - parametric-only accuracy ≥ 0.90;
   - unknown ≤ 0.10;
   - fluency ≥ 0.80.

   Select the eligible p_rd closest to 0.50; then DC-1 across the grid and DC-2 at the selected value. Any failure → **B1 TERMINATED**.
2. **V4.2-F0B** (the selected 9071 Stage-A checkpoint → Stage B):
   - store QC;
   - F1 redundancy (`f1_eligible`);
   - covered integrated accuracy ≥ 0.98;
   - rescue ≥ 0.95 (Stage-A-wrong covered facts);
   - no damage ≥ 0.98 (Stage-A-correct covered facts), and parametric-only accuracy drop ≤ 0.01;
   - INT-1 (Spearman of Stage-A vs integrated margin > 0, p < 0.01);
   - SC-1 (≤ 10, by construction);
   - SC-2 (NULL-route injection ≤ 0.10 × ‖hA‖);
   - bit-identical Route-A equivalence.

   Any failure → **B1 TERMINATED**.
3. **V4.2-F1C** (9072, 9073 at the frozen p_rd): the F0A per-p gates, DC-2 and every F0B gate. Any failure → **STOP B1**.
4. **F2, F3, F4** (9071–9073): unchanged.
5. **P2** (development): unchanged.
6. **FREEZE:** record the recipe and commit.
7. **VAL** (9051–9053, ONCE): Stage A (F0A gates and DC-2), Stage B (F0B gates), F2–F4 and P2 at the frozen recipe. Then **STOP before any monitor training.**

### Rules

- No thresholds are relaxed.
- No grid values are added.
- No epoch extension of Stage A.
- No reselection at confirmation.
- No validation seed is used for any choice.
- No monitor is trained.
- No confirmatory seed is run.

**If B1 stops:** a B1 FINAL REPORT plus a PIVOT MEMO (three substantially different directions), with no new implementation and no v4.3.
