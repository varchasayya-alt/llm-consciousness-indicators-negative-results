# Stage-1 v4 DESIGN MEMO: dual-route store with locally editable fact memory (Option B1)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | **For PI approval. Nothing implemented, nothing run.** |
| Basis | PI decision (2026-10-03) approving the move to B1 |
| v3 record | `calibration_stop_report_v3_C4dev.md`, D44 |

**The v3 negative methods result, recorded as such:**

- All 36 pre-declared parametric T-FORGET cells failed continuous retention.
- Meaningful forgetting also collapsed outcome diversity.
- The F3–F5 identifiability checks passed.
- So the binding problem is the **intervention mechanism**, not the within-target identification.
- The v3 grid is not reopened, and no new parametric objective family is introduced.

**Purpose of B1:** a cleaner causal intervention, not a manufactured positive result.

---

## 1. Exact dual-route architecture

```
QUERY  [Q s1 s2 r A]
  │
  ├─► PARAMETRIC ROUTE: blocks 1–2 (pre-LN transformer, d = 128, 4 heads, MLP 512; unchanged)
  │        │  h2[A] = residual at [A] after block 2
  │        ▼
  │   retrieval query  q = W_q · LN(h2[A])                   (W_q: 128 → 64)
  │        │
  ├─► EXPLICIT FACT MEMORY: slots j = 1..S (one per memory-covered trained fact) + 1 learned NULL slot
  │        keys k_j ∈ R^64, values v_j ∈ R^128 (free parameters); mask m_j ∈ {0,1} (1 = present)
  │        a = softmax( (q·k_j)/τ + log m_j ) over j ∪ {null}     (τ learned scalar; the null slot always present)
  │        r = Σ_j a_j v_j + a_null v_null
  │        ▼
  │   INTEGRATION: h2[A] ← h2[A] + W_o · r                   (W_o: 128 → 128; [A] position only)
  ▼
blocks 3–4 operate on the integrated residual  →  LN → unembedding → ANSWER (value tokens of r)
MONITOR READ SET (unchanged, frozen on architectural grounds): residual at s2 and [A] after layers 0–4
```

**Notes:**

- **Injection point.** Memory is injected at the [A] position after block 2.
  - Blocks 1–2 form the retrieval query, by gathering s1, s2 and r into [A].
  - Blocks 3–4 integrate the memory contribution with the parametric evidence before readout.
  - This is fixed on architectural grounds, not by any monitor quantity.
- **Read set, unchanged.** Under the causal mask, s2 sites and [A] layers 0–2 are computed **before** memory enters, so they are parametric-only. The integrated sites are [A] layers 3–4.
- **Twin store B.** Same architecture, different initialisation and data order (unchanged design).

## 2. How facts are learned into each route

Training data are as before: fact sequences `[Q s1 s2 r A v E]` and mention sequences, with the same world and exposure manipulation. The memory is read only at an `[A]` position.

**Memory coverage.**

- A random fraction **c = 0.7** of trained facts receive a slot. These are the *memory-covered* facts.
- The remaining 30% of trained facts are **parametric-only**: they never have a slot.
- Unknown (untrained) facts have no slot.
- Assignment is random per fact, independent of exposure, drawn with the world seed.

**Rationale for c < 1:**

- In natural data, "memory absent" must not equal "fact unknown". Otherwise any monitor could learn the bookkeeping cue (§7).
- Parametric-only facts are also the population for P2 (§15).

**Retrieval supervision.** An auxiliary cross-entropy on the retrieval distribution `a` (weight α_ret = 1) points to:

- the fact's own slot, when it is present;
- the NULL slot, for parametric-only facts, unknown facts, and route-dropped presentations.

This makes addressing explicit and sharp, which is what makes deletion local. It is a property of the memory system, not of the monitor.

**Route dropout.** During store training, each *presentation* of a memory-covered fact has its slot masked with probability **p_rd**. This uses exactly the masking operation of T-DELETE.

- The answer must then come from the parametric route.
- This is how memory-covered facts acquire parametric backup.
- It also makes "memory unavailable" an in-distribution state for the store.

**Training procedure:**

- next-token CE on all sequences, plus α_ret · retrieval CE;
- AdamW with the store's existing schedule (lr 2e-3 → 2e-4 cosine), batch 512;
- E_store = 60 epochs (the C1 rule is reapplied only if F0 fails; §17).

## 3. How parametric-backup variation arises

**No fact is assigned a backup strength.** Variation emerges from:

1. **Stochastic route-dropout counts.**
   - Fact i is trained parametrically on Binomial(60 · n_i, p_rd) presentations, where n_i is the presentations per epoch (1 for facts).
   - With a common p_rd, counts vary continuously across facts.
2. **Data order and optimisation dynamics.**
   - Parametric learning in this store has a steep but item-dependent transition: v1 C1 shows 17–20% of facts learned at epoch 20, 62–86% at 30, and 100% at 40.
   - At reduced parametric exposure, facts land at different points of that transition.
3. **Exposure (familiarity) effects.**
   - High-exposure entities have better entity representations, so their facts may be learned parametrically more easily.
   - This **influences** the distribution, but a pre-declared gate (§10) requires that exposure alone does not determine the outcome.
4. **Capacity competition** with the parametric-only facts, which are trained parametrically on every presentation.

**Calibrating p_rd** (single pre-declared rule, §17):

- Grid p_rd ∈ {0.2, 0.35, 0.5}.
- Choose the value whose parametric-only (route A) accuracy on memory-covered base-correct EV facts is closest to 0.5, subject to an IQR of the route-A margin ≥ 1 nat.
- This is set on development store 9041 and confirmed on 9042–9043.

## 4. Exact deletion operation: T-DELETE(X)

For every x ∈ X (300 memory-covered, base-correct EV facts; 150 high / 150 low exposure; random): set **m_{j(x)} := 0**. Its retrieval logit becomes −∞.

- **Identical rule and nominal magnitude for every target:** one slot fully removed.
- **No per-item strengths.**
- **No parameter of the network, of the NULL slot, or of any other slot changes.**
- Post-deletion, an X query retrieves what a route-dropped presentation retrieved in training (mostly the NULL value). Its answer depends on parametric backup integrated in blocks 3–4.

## 5. Why deletion is local

An untargeted query z changes only through renormalisation of its retrieval distribution:

- ‖Δr(z)‖ ≤ 2 · μ_X(z) · max_j ‖v_j‖, where μ_X(z) = Σ_{j∈X} a_j(z) is z's **pre-deletion attention mass on deleted slots**.
- With supervised, sharp retrieval, μ_X(z) ≈ 0, so collateral ≈ 0.
- μ_X is measured and reported for every Z category ("retrieval leakage").
- Unlike v3, there is **no gradient update**, so **no shared-output tail flattening** (the v3 failure mechanism, D44) can occur.

## 6. What the monitor can and cannot observe

| Monitor MAY read | Monitor may NOT read (prohibited; enforced in code and tested) |
|---|---|
| The frozen read set: residual stream at s2 and [A] after layers 0–4, i.e. ordinary neural states of the integrated computation | Slot masks / occupancy bits; deletion masks; "found / not found" flags; slot addresses or indices |
| OUT: sorted restricted output distribution + entropy (as before) | Retrieval weights `a`, a_null, max-weight, entropy of retrieval, τ, ‖r‖ |
| IN: its own input embeddings (as before) | The memory table (keys, values); W_q, W_o internals |
| | Any intervention identifier; whether T-DELETE / Y-transplant was applied |

**Enforcement:**

- The monitor feature builder takes only `probe(...)["states" / "probs"]`.
- A unit test asserts that no memory-internal quantity reaches monitor features.
- An AST guard keeps calibration code away from monitor code (as in v1–v3).

## 7. How explicit deletion cues are kept from trivialising the task

| Risk | Design response |
|---|---|
| Deleted slot produces an out-of-distribution zero vector | Deletion falls back to a **learned NULL slot**. NULL retrieval is **in-distribution** for the store (route dropout, parametric-only and unknown facts all retrieve NULL in training) |
| "Memory absent ⇒ wrong" is learnable from natural data | **c = 0.7:** in natural data, memory-absent facts include **known parametric-only facts** (mostly correct) as well as unknown ones. Memory absence is therefore not a competence label in the monitor's training data |
| Deletion cue predicts outcome | Within X the cue is **constant** (every target deleted), so it cannot explain within-X competence variation |
| Cue interacts with susceptibility | Handled by the pre-state step of θ_pre and quantified by the trivial-decoder audit (§12). Kill condition: bookkeeping features determine C_post within X (R² ≥ 0.90) |

None of these choices uses monitor performance.

## 8. Proposed Y negative control: same-answer value transplant

**Construction.**

- Y = 300 memory-covered, base-correct EV facts (disjoint from X).
- For each y, copy into slot j(y) the value vector of a randomly chosen **donor**: another memory-covered fact with the same relation and the **same answer value**, not in X, Y or any QC set. The donor keeps its own slot.

**Properties:**

- One rule and one nominal magnitude (one slot's value replaced) for every y.
- No optimisation and no monitor input.
- The integrated [A] state changes (donor-specific features), while the answer is expected to be preserved.

**Role.** A **negative control** for "the memory route changed, competence did not" (K2 equivalence, ±0.10 D_nat). It is not the primary identification.

**Y QC:**

- binary lost ≤ 3%;
- continuous retention (§9) for Y;
- the displacement ratio Y/X at [A] layers 3–4 is reported, not matched.

**Exploratory alternative Y2:** delete one weakly attended unrelated slot for each y (attention mass < 0.01).

X deletion and the Y transplant are applied together in one intervened store, as in v1–v3.

## 9. Continuous collateral gates

The approved v3 §7 criteria translate unchanged. X's change is a large, well-defined margin drop, so the ratios are well-posed.

**Gate, for every seed:**

- **Categories:** Y, Z_random, Z_param (parametric-only base-correct EV), Z_near_entity_X/Y (CT facts of X/Y entities), Z_near_key (top 20% of Z by key cosine to deleted keys; the memory-space analogue of Z_near_repr), Z_far (bottom 20%).
- **For each category:**
  - |Δmargin|, |Δlog p(v\*)| and KL(p₀‖p₁) are each ≤ **0.10** × X's;
  - |Δmargin| ≤ 0.10 × D_C;
  - binary lost ≤ 5% (Y: ≤ 3%).
- **Added (B1-specific, locality):** mean read-site displacement of each Z category ≤ 0.10 × X's.
- **Sensitivity:** reported at 0.05 / 0.10 / 0.20.
- **Reported:** the retrieval-leakage distribution μ_X(z).

## 10. Outcome-diversity gates (pre-declared)

Over X after deletion, every development seed must satisfy all of:

1. Lost fraction (C_post < 0) ∈ [0.35, 0.65].
2. IQR(C_post) ≥ 1 nat.
3. **Three outcome tiers each ≥ 15% of X:**
   - retained-strong: C_post ≥ 0.5 · C_pre;
   - weakened: 0 ≤ C_post < 0.5 · C_pre;
   - lost: C_post < 0.
4. **No deterministic exposure mapping:** cross-fitted R²(C_post | exposure class) ≤ 0.50, and AUROC(exposure → lost) ≤ 0.80.
5. **Binary-secondary feasibility:** the approved F6 is binding; F6′ is recorded (D43).

**Reported:** pre- and post-deletion accuracy and margin distributions (histograms), IQR, and the relationship with exposure and with pre-state features.

## 11. F3–F5 identifiability (retained unchanged)

Within X, cross-fitted median R² over seeds:

| Code | Quantity | Threshold |
|---|---|---|
| F3 | R²(ΔC \| 40 generic change features) | < 0.90 |
| F4 | R²(ΔC \| pre-intervention information: W0, C_pre, pre-states) | < 0.90 |
| F5 | R²(ΔC \| both) | < 0.95 |

All three are also reported for C_post.

**F4 is the critical gate for B1.**

- The parametric-only read sites (s2 and [A] layers 0–2) are identical before and after deletion. If they already reveal the backup almost perfectly, θ_pre has little to test.
- The B1 dissociation requires that the post-deletion outcome depend partly on integration in blocks 3–4 that is **not exercised while memory is present**, and so is not readable from pre-deletion states.
- **θ_pre credits only competence information not already available before the intervention.** This is by design.

## 12. Trivial-decoder audit (store-only, before any monitor)

**Method.**

- Cross-fitted ridge (R² of C_post and ΔC) and logistic (AUROC for lost).
- Run within X; for source 1, also across X ∪ Z, where deletion status varies.
- Same fold and ridge specifications as F3–F5.

| # | Information source | Features | Purpose |
|---|---|---|---|
| 1 | Deletion / memory bookkeeping (*prohibited for monitors*) | Deletion status; post a_null, max a_j, retrieval entropy, ‖W_o r‖, ‖W_o Δr‖ | Does an explicit memory cue determine outcome? |
| 2 | Pre-intervention | W0, C_pre, pre-states | = F4 |
| 3 | First-order output | Post top-1 prob, entropy, top-1 margin (what OUT sees) | Output-readout availability |
| 4 | Generic integrated-state change | 40 generic features | = F3 |
| 5 | Complete post-store representation | 1,280 post read states | Is current competence decodable at all? |

**Interpretation (pre-declared):**

- 5 high and 1 (within X) low ⇒ "the competence state of the integrated computation" is available, not a memory-absence cue.
- 5 ≫ 3 ⇒ room for an E2 (internal beyond output) result.
- **Kill:** within-X R²(C_post | source 1) ≥ 0.90.

## 13. INT versus OUT

- **E1** (the Stage-1 success criterion): does the monitor track current competence at all?
  - Tested by θ_pre for INT-S among identically targeted X (P1), with labels A/B/C.
- **E2** (reported; **not required** for Stage-1 success): does internal-state access add information beyond first-order output?
  - SEC_INTvsOUT_z = z(θ_pre INT-S) − z(θ_pre OUT) > 0, on P1 and P2, with seed-bootstrap CI, plus the INT+OUT exploratory monitor.
- **Why E2 is not required:** in B1 the answer-proximal [A] layer-4 state is close to the output. A monitor that reconstructs answer confidence satisfies the narrow Level-1 claim, and the report says which result (E1 only, or E1 + E2) was obtained.

## 14. Architecture ablations (store-only; establish route contributions)

| Ablation | Operation | Pre-declared requirement (F1) |
|---|---|---|
| **A. Parametric-only** | All slots masked (NULL only) | On memory-covered base-correct EV: accuracy ∈ [0.30, 0.70] and IQR(margin) ≥ 1 nat (partial, graded backup). Parametric-only facts: accuracy ≥ 0.90 |
| **B. Memory-only** | Retrieval as normal; the parametric residual at [A] after block 2 is replaced by its mean over items before blocks 3–4 (memory term kept) | Memory-covered facts: accuracy ≥ 0.90 (memory reliably answers) |
| **C. Integrated** | Normal | Trained facts ≥ 0.98; unknown ≤ 0.10; fluency AUROC(high vs low) ≥ 0.80 |
| Retrieval sharpness | Supervised retrieval | Mean attention on own slot ≥ 0.90 for covered facts; mean μ_X(z) ≤ 0.01 |

**Redundancy is genuine** only if A and B are each substantially competent on covered facts and C exceeds both. That is required above.

## 15. Future P2 (designed now, not implemented)

**Primary proposal: interference in the parametric route.**

- After store training, fine-tune **only the parametric weights** on new facts about the 1,000 reserved entities, with no replay. The memory table is frozen.
- **P2 population:** all parametric-only base-correct EV facts. All are equally exposed, and their competence degrades naturally and variably. This is the v3 P2 estimand (θ_pre within exposed items).
- **Why it is qualitatively different from P1:**
  - a different route (parametric vs memory);
  - a different mechanism (interference vs removal);
  - naturally arising, often confident errors.
- **Built-in control:** memory-covered facts are mostly protected by their slots, giving an untargeted comparison population.

**Exploratory alternative: contradictory slot overwrite.** Write a conflicting value into existing slots, giving confident wrong answers from memory.

**Calibration rule, as in v3 C7:**

- lr grid {3e-4, 1e-3, 3e-3}; stop at new-fact accuracy ≥ 0.95;
- lost fraction ∈ [0.10, 0.50]; F3–F5 apply.

## 16. Seed plan

| Seeds | Role |
|---|---|
| 9041, 9042, 9043 | v4 development (all v4 selection) |
| 9051, 9052, 9053 | v4 fresh validation, used **once** per component |
| 9021–9023 | **Retired unused.** Recorded as the abandoned v3 validation set, kept for the audit trail and not reused |
| 9001–9005, 9011–9013, 9031–9033 | Historical evidence only |
| 1001–1040 | Confirmatory (unchanged) |

## 17. Smallest feasibility ladder (early stopping at the first structural failure)

| Step | Store(s) | Check | Stop if |
|---|---|---|---|
| **V4-F0** | 9041 (one store, p_rd = 0.35) | Trains; integrated trained acc ≥ 0.98 at 60 epochs (else the C1 rule decides E); unknown ≤ 0.10; fluency AUROC ≥ 0.80; retrieval sharpness | Architecture cannot learn or address reliably |
| **V4-F1** | 9041 × p_rd {0.2, 0.35, 0.5} → select (§3); confirm on 9042, 9043 | Ablations A / B / C (§14): genuine redundancy and graded backup | No p_rd yields A ∈ [0.30, 0.70] with IQR ≥ 1 and B ≥ 0.90 |
| **V4-F2** | 9041–9043 | T-DELETE(X) + Y-transplant: collateral gates (§9), leakage, Y QC | Collateral fails (deletion not local) |
| **V4-F3** | 9041–9043 | Outcome diversity (§10) | Diversity or exposure gate fails |
| **V4-F4** | 9041–9043 | F3–F5 (§11) + trivial-decoder audit (§12) | F3–F5 fail, or bookkeeping R² ≥ 0.90 |

**After F0–F4 pass:**

1. P2 calibration (§15) on 9041–9043.
2. Freeze the v4 recipe.
3. Fresh validation **once** on 9051–9053: F0–F4 gates + P2.
4. **STOP** with the calibration package. No monitor training before PI approval.

**Before any v4 run, write and commit:**

- the architecture, deletion, Y and audit code;
- unit tests: monitor-input guard; deletion touches only masked slots; Y-transplant preserves donors; ablation operators;
- config, calibration-plan revision v4, decisions.

**Carry-over:**

- The statistical machinery from v3 is unchanged (§20 note).
- One B1-specific fake-data world is added to the dry-run suite: deletion with pre-visible plus hidden backup. This **verifies** S9 rather than reselecting a statistic.

## 18. CPU / RAM estimate

| Component | Estimate |
|---|---|
| Memory table | ~4,600 slots × (64 + 128) ≈ 0.9 M parameters (about the size of the transformer). Retrieval is a 512 × 4,601 × 64 matmul per step: negligible |
| One store (60 epochs, + retrieval loss) | ~10–15 min |
| F0 + F1 | 3 stores on 9041 + 2 confirmations ≈ 1.25 h |
| F2–F4 | Deletion is instant; QC, ridge and audit ~3–5 min per seed ≈ 15 min |
| P2 calibration | ~45 min |
| Validation (3 stores + gates + P2) | ~1 h |
| **Methods phase total** | **≈ 3.5–4.5 CPU-h** |
| RAM | < 2 GB |
| Later confirmatory run (2 stores per seed + monitors) | ~40–60 min per seed, so ~15–20 CPU-h for 20 seeds (as before) |

## 19. Conditions that kill B1 before monitor training

| Code | Condition |
|---|---|
| K-1 | V4-F0 fails: the integrated store cannot reach ≥ 0.98 trained-fact accuracy, or retrieval is not sharp (own-slot attention < 0.90) |
| K-2 | **No genuine redundancy** for every pre-declared p_rd: memory-only < 0.90, or parametric backup on covered facts outside [0.30, 0.70], or IQR < 1 |
| K-3 | **Deletion not local:** any §9 collateral gate fails |
| K-4 | **No outcome diversity:** any §10 gate fails, including deterministic exposure mapping |
| K-5 | **Non-identifiable:** F3, F4 or F5 fails (F4 is the critical one) |
| K-6 | **Trivial cue:** within-X R²(C_post \| memory bookkeeping) ≥ 0.90 |
| K-7 | Y negative control fails its QC. **Stop and report** (Y is secondary, so this is not an automatic kill) |
| K-8 | Fresh validation on 9051–9053 fails any of K-1 to K-6 |

**On any kill:** stop and report; thresholds are never relaxed after results. A failure of B1 would mean that this synthetic system cannot provide the required dissociation without a further redesign of the experiment.

## 20. Claims permitted if B1 eventually succeeds

**Strongest (P1 label A or B):**

> "A separately developed monitor can track changes in the current competence of an integrated neural memory system after an input-invisible internal intervention, beyond what can be inferred from unchanged inputs and measured pre-intervention susceptibility."

**If P1 and P2 are both label A, add:**

> "The tracking cannot be readily reduced to the measured generic signatures of the intervention and generalises across qualitatively different competence changes, namely memory-route removal and parametric-route interference."

**If E2 holds**, add that internal-state access carried competence information beyond first-order output. **Otherwise state explicitly:** "consistent with reconstruction of answer confidence".

**Relevance:** functional metacognition in a synthetic system, Level 1. It is consistent with the developmental / decomposition programme (primary cognition + memory + monitoring + control as separately testable capacities, before recurrence / workspace integration).

**Not claimed:** phenomenal consciousness, subjective experience, human-like self-awareness, a conscious LLM, or generalisation to LLMs or brains.

---

**Statistical framework: carried over unchanged.**

- The unit is the store seed.
- θ_pre is the simulation-selected residualized-post statistic with the symmetric pre-state step; θ_gen is robustness; A/B/C labels; continuous C; within-target; no pseudo-replication.
- No assumption is invalidated:
  - items are identically treated;
  - pre-intervention information is available;
  - C is continuous.
- Near-ceiling C_pre / M_pre within X (memory present) is handled by the rank RCS baseline.
- Therefore statistic selection is **not** re-run. The added B1 fake-data world (§17) is a verification only.
