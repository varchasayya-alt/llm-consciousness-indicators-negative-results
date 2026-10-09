# Calibration STOP report: v3 C4dev (within-target design). No T-FORGET mechanism variant is eligible

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Status | **Calibration stopped under the pre-declared rule** (`calibration_plan.md` Revision v3; D38–D40). No eligible cell in the V0–V3 family on development seeds 9031–9033. This is the pre-declared trigger for the Option B pathway, which happens after a report to the PI |
| Not run | C2val, C4val, C7–C10, VAL. **Validation seeds 9021–9023 remain untouched.** No monitor or controller trained; no confirmatory data; nothing frozen or registered |
| Pre-run commit | `119a3c5` (v3 methods revision, statistic simulation, tests, fake-data validation) |
| Raw results | `results/raw/calibration/calibration_results.json` (key `v3`), `log_v3_*.txt` |
| Rendered tables | `results/processed/calibration/calibration_tables.md` § "Revision v3" (auto-generated) |

## 1. What ran

| Step | Result |
|---|---|
| C2dev: stores 9031–9033 (E_store 60) | All pass store QC: trained accuracy 1.00; unknown 0.028–0.030; fluency AUROC 1.00 |
| C4dev: 36 pre-declared cells × 3 dev seeds | **0 eligible.** Every cell fails F1 (continuous retention). Every cell that produces meaningful forgetting also fails F2 (outcome diversity) |

The 36 cells are V0 (6), V1 (18: anchor 0.01 / 1 / 100), V2 (6) and V3 (6).

## 2. Findings (store-only; no monitor involved)

### 2a. F1 continuous retention fails everywhere, and the failure has one clear mechanism

- **Worst-case retention load** (1 = at the 0.10 gate): minimum over all 36 cells **2.04**. The best in-range cell per variant scores 3.08 (V1), 3.12 (V0), 3.14 (V2) and 4.61 (V3).
- **Binding metric: margin.** Example: V0, lr 1e-4, 200 steps, seed 9031 (the pattern is the same in every variant).

| Set | Δmargin | Δlog p(v\*) | Δ runner-up log p | Lost |
|---|---|---|---|---|
| X (targets) | −6.31 | −2.56 | +3.75 | 0.63 |
| Y (negative control) | −1.91 | −0.07 | +1.84 | 0.00 |
| Z_random | −1.42 | −0.03 | +1.39 | 0.00 |
| Z_near_entity_X | −1.67 | −0.05 | +1.62 | 0.00 |
| Z_far | −1.34 | −0.03 | +1.30 | 0.00 |

  - Untargeted facts keep their correct-answer probability, so their log p and KL ratios to X mostly pass (0.03–0.08).
  - Their **runner-up** answer gains 1.3–1.9 nats. That shrinks margins by 20–40% of X's change (0.15–0.26 × D_C, against a gate of 0.10).
- **Mechanism.** The uniform-target forgetting objective raises *all 31 wrong-value logits* for every targeted query. The value outputs are shared across facts, so the tail flattens for every held-out fact of the same relations. The KL retain term protects exactly the facts it supervises (R_sibling Δmargin ≈ 0), not held-out facts.
- **None of the pre-declared variants changes this:**
  - V1 anchor at λ = 1 almost stops forgetting (X lost 0.01–0.11) yet still leaves load ≥ 2.0. At λ = 100 X barely changes and the ratios explode.
  - V2 (enlarged retain pool) is indistinguishable from V0.
  - V3 (localised MLP edit) is worse: the Y negative control loses more.
- This matches reports that knowledge edits in synthetic transformers distort non-target items [nishi2025shattering].

### 2b. F2 outcome diversity fails for the same reason

- Within X, the IQR of post-intervention margin is **0.22–0.83 nats** in every cell with meaningful forgetting. The gate is ≥ 1 nat.
- Uniform-target forgetting drives every forgotten item to the same near-uniform endpoint (margin ≈ 0), so graded competence variation among targets is compressed.
- IQR ≥ 1 occurs only where almost nothing is forgotten (strong anchor; V3 at the lowest lr).
- The X lost fraction itself is in range [0.35, 0.65] for many cells.

### 2c. Identifiability is good (a positive result for the within-target logic)

| Diagnostic | Range over cells with meaningful forgetting |
|---|---|
| Median cross-fitted R²(ΔC \| 40 generic features) | −0.03 to 0.17 |
| R²(ΔC \| pre-intervention information) | ≤ 0.56 |
| R²_joint | ≤ 0.55 |
| Spearman(fingerprint score, ΔC) within X | −0.58 to 0.03 |

- F3–F5 would pass. Within-target competence variation is **not** determined by the generic change signature or by pre-intervention information in this store.
- The identification strategy is therefore viable in principle. **The intervention mechanism is the binding problem.**

### 2d. F6 vs F6′ (D43)

| Rule | Seed-runs passing |
|---|---|
| Approved F6 | 32% |
| Pre-declared alternative F6′ | 58% |

F6 was not the reason for failure: every cell already fails F1. `selected_under_F6alt_recorded_only` = none.

## 3. Pre-declared consequence

By D38–D40 and the plan, no mechanism meets retention plus diversity, so: **STOP, report, and consider Option B.** Thresholds were not relaxed, no variant was added, and 9021–9023 were not used.

## 4. Options for the PI (decision needed; nothing implemented)

### B1 (recommended): dual-route store with an explicit, locally editable fact memory

- **Store.**
  - The same 4-layer transformer, plus a key–value fact memory read at the [A] position. Keys come from (entity, relation) and each stored fact has a learned value vector.
  - Facts are learned both in the memory and, with item-varying strength, in the parametric route. The redundancy follows naturally from the exposure manipulation and training dynamics.
- **Intervention (T-DELETE).** Delete the memory slots of the X facts.
  - **Identical procedure for every target.**
  - **No shared-output flattening and no weight change**, so collateral on Z is limited to retrieval re-normalisation. Continuous retention becomes testable at the approved 0.10 gate.
- **Within-target variation.** Whether a deleted fact survives, and how strongly, depends on its parametric backup, which varies by item. This yields graded C_post that the edit magnitude does not determine.
  - This addresses F2 *and* avoids the trap of a dose-graded deletion: there, ΔC and the generic change would both be set by the dose, giving F3 non-identifiability.
- **Y negative control.** A matched-norm change at the read sites that leaves the item's own slot intact, for example deleting unrelated slots that the item's query attends to weakly.
- **Interference analogue (P2).** Write new facts with no replay to the parametric route, or overwrite slots.
- **Checks to re-establish.** F3/F4 identifiability must be re-checked: the backup strength may be visible in pre-states.
- **Cost.**
  - New store calibration and a new pre-declared cycle (v4) on fresh development seeds (9041+) and fresh validation seeds (9051+).
  - Roughly a day of CPU.
  - The primary statistic, analysis, fake-data suite and claim boundaries carry over.
- **Scientific cost.** A more synthetic memory and narrower generalisation claims.

### P (not recommended without explicit approval): stay parametric with a different forgetting objective

- The mechanism in §2a is specific to the **uniform** target, the PI-approved D2 objective. Objectives that lower p(v\*) without raising every alternative (for example, correct-answer suppression with a floor) might pass F1/F2.
- This would be a new variant family outside the approved V0–V3, in a new cycle on fresh seeds. It also changes D2.

### Not proposed

- Relaxing the 0.10 margin gate, for example to log p/KL only: rejected by the PI's rule, and the margin loss is real.
- Reusing 9031–9033 for selection in a new cycle.

## 5. State

| Item | State |
|---|---|
| Within-target estimand, statistic, analysis, tests, fake-data validation | Done (`119a3c5`) and reusable for B1 |
| Intervention for the current parametric store | **Fails F1/F2** for all pre-declared variants |
| 10-item calibration package | Cannot be completed until an intervention passes |
