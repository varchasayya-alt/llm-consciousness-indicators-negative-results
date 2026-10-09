# Stage-1 D2 PROTOCOL: self-generated competence change through continued learning (store-only phase)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | Written and committed **before any D2 store run**. Proposed; not frozen or registered |
| Basis | The PI's D2 approval (2026-10-03); `memo/stage1_pivot_memo.md` (D2); `d2_S0_satisfiability_note.md`; decisions D62–D65 |
| Scope of this phase | **Store-only kill test (D2-S0 to S3). No monitor, no controller.** The phase stops when a binding gate fails or the store-only package is complete |

## 1. Question

When a neural system's competence changes as a consequence of its own continued learning, can a separately developed monitor track those item-level changes beyond what was predictable beforehand and beyond generic model drift?

**This phase asks only whether continued learning produces a usable competence-change distribution.** "Usable" means enough change, graded, identifiable and not trivial.

## 2. System (no B1 component)

- **Store:** the plain v1–v3 parametric store (`s1/store.py`), trained 60 epochs on the unchanged synthetic world (the v1–v3 recipe), from `derive_seed(seed, "storeA")`. Checkpoints at the configured fractions are saved for a later developmental monitor.
- **Forbidden:** explicit memory, deletion, unlearning, targeted parameter edits, transplanted shams, and every B1 component. `run_d2.py` and `s1/d2.py` may not import `memstore`, `v4qc`, `monitors`, `controller` or `pipeline` (unit-tested).

## 3. Conditions (each development seed)

| Condition | Definition |
|---|---|
| **INTERF** (primary) | `interventions.interference`: continue training the base store on the 1,000 reserved interference entities (2 facts each, plus 8 mentions per entity). Constant-lr Adam, batch 512, random batches, **no replay** of original knowledge. Stop when new-fact accuracy ≥ 0.95 (checked every 50 steps) or at max_steps. Seed `derive_seed(seed, "interf")` |
| **CTRL** (matched-development control; not a sham) | Continue the same base store with the same optimizer, lr, batch size and **exactly the same number of steps** as INTERF on that seed and lr. Batches are drawn from already-known material **excluding the evaluated (EV-split) facts**: MT/CT known-fact sequences plus all original mention sequences. Seed `derive_seed(seed, "ctrl")`. It estimates generic drift from continued optimisation. No matched loss is required |
| **Familiarity** (report-only) | Name-fluency change of each evaluated item's entity under INTERF and CTRL, and its association with ΔC. A familiarity-only exposure arm (mentions only) is designed for the monitor stage (§9) and is not part of the kill test |

**Population:** every base-correct EV fact of the base store; all are equally exposed to the same continued learning.

**Input identity:** the evaluation tokens [Q s1 s2 r A] are identical before and after. This is asserted per run and unit-tested.

## 4. Fixed interference grid (frozen here)

- lr ∈ {3e-4, 1e-3, 3e-3};
- max_steps 3,000; eval_every 50; stop when new-fact accuracy ≥ 0.95;
- batch 512; mentions per entity 8.

This is the previously designed P2/C7 logic, never run before D2. No values may be added after results.

## 5. Ladder

| Stage | Content | Stop if |
|---|---|---|
| D2-S0 | Analytical satisfiability (`d2_S0_satisfiability_note.md`) | n/a (done) |
| D2-S1 | Base stores on 9101, 9102, 9103; store QC (G0) | any base store fails G0 |
| D2-S2 | INTERF for each (lr, seed) in the grid, then the yoked CTRL; store-only quantities only | — |
| D2-S3 | Per (lr, seed): G1, G2, G4, G5 and the report-only diagnostics; per lr: G3 (medians). Selection rule (§6) | **no lr is eligible → D2 kill report + D3 protocol proposal** |

## 6. Gates and selection rule (store-only; no monitor quantity is ever used)

Gates G0–G5 are as in the S0 note:

| Gate | Requirement |
|---|---|
| G1 | lost ∈ [0.10, 0.50] on every development seed |
| G2 | IQR(ΔC) ≥ 1 nat on every seed |
| G3 | median R²_pre < 0.90, R²_gen < 0.90, R²_joint < 0.95 (cross-fitted nested-CV ridge; outcome ΔC; existing `identifiability`) |
| G4 | R²(C_post \| bookkeeping) < 0.90 on every seed |
| G5 | input identity |

An lr is **eligible** iff G1, G2 and G4 hold on all three seeds and G3 holds for its medians.

**Selection:** the eligible lr with mean lost closest to 0.25. The frozen recipe's max_steps = ceil100(1.5 × the largest step count used at that lr across the development seeds). This is the P2 rule.

## 7. Reported for every (lr, seed) (D2 package items)

- original accuracy (population and all known facts);
- post-learning accuracy;
- lost fraction;
- the ΔC and C_post distributions (quantiles), IQR(ΔC) and IQR(C_post);
- generic state displacement (per-site mean, quantiles) and the generic-40 summary;
- relation and exposure stratification (lost, mean ΔC; exposure R² and AUROC);
- training-step history (new-fact accuracy every 50 steps; steps used);
- **CTRL:**
  - lost fraction, ΔC distribution and displacement;
  - interference-specific loss (lost_INTERF − lost_CTRL);
  - cross-fitted R²(ΔC_INTERF | ΔC_CTRL) and Spearman correlation: how much item-level change is generic optimisation drift;
- familiarity: Δfluency (INTERF and CTRL) and Spearman(ΔC, Δfluency);
- the trivial-decoder audit: cross-fitted R² of C_post and ΔC from pre, bookkeeping, generic, output, full, full+pre and full+output, with incremental terms;
- **D50:**
  - pre-state dimension (1,280 + 7 covariates), n per seed, p/n;
  - cross-fitted R²_pre, R²_gen and R²_joint;
  - the residual SD of ΔC after the cross-fitted pre-information prediction (the usable within-target variation, in nats);
- F6 / F6′ (binary-secondary feasibility): **report-only.** F6 is not among the PI's D2 requirements and is not a gate here.

## 8. Statistic, labels and claims (unchanged; no reselection)

- θ_pre (`estimands.theta_post_dml`, residualised-post association, symmetric cross-fitted pre-state step) and θ_gen;
- nested-CV ridge;
- seed-level inference;
- Holm and MEI 0.10;
- the D54 label policy: **A is the only affirmative result; B is ambiguous and not evidence; C is no evidence.**
- Level-1 claims only. Never: consciousness, awareness or self-awareness.

## 9. After a pass (not executed in this phase; requires PI approval)

- the developmental monitor design;
- the familiarity-only exposure arm (mentions of a random half of the evaluated entities, matched steps);
- the acquisition analogue (sign-reversed: initially unknown or weak facts learned later; designed, not run);
- fresh validation 9111–9113 (proposed; untouched);
- the preregistration changes.

## 10. Seeds

| Seeds | Role |
|---|---|
| 9101, 9102, 9103 | **D2 development** (store-only kill test) |
| 9111, 9112, 9113 | **Proposed D2 validation.** Untouched; used once, only after PI approval of a frozen D2 recipe |
| 9001–9073 (all), 9051–9053, 12345 | Not used |
| 1001–1040 | Confirmatory; not used |

## 11. Prohibited

- tuning other intervention families;
- adding grid values;
- relaxing any gate after results;
- using validation seeds;
- training or evaluating any monitor;
- downloading any model (D3 needs explicit approval).
