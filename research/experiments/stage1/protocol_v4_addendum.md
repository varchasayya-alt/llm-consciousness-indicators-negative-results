# Stage-1 PROTOCOL ADDENDUM v4 (Option B1). Proposed; NOT frozen, NOT registered

This addendum modifies `protocol_v3_PROPOSED.md` for the dual-route store (`v4_design_memo.md`; D45–D48). Everything not changed here carries over unchanged:

- the estimand θ_pre (simulation-selected) and θ_gen;
- the A/B/C labels;
- families P, G, K, F and S;
- the store seed as the unit;
- the claim boundaries.

## Changes

1. **Store.** The dual-route store: a 4-block transformer plus an explicit key–value fact memory read and injected at [A] after block 2. It has a learned NULL slot, 70% random memory coverage, supervised retrieval, and route dropout p_rd ⟦CAL⟧. The query, key and temperature are as implemented (D47).
2. **P1 intervention.** T-DELETE(X): every memory-covered target's slot becomes unavailable. The rule and the full-slot magnitude are identical for every target; there are no strengths.
3. **Y (negative control).** A same-relation, same-answer donor value transplant into Y's slots (`memstore.transplant`).
4. **Per-seed QC.** The v4 F2/F3 gates (`calibration_plan.md` Revision v4) replace the v3 T-FORGET gates for P1. The fallback ladder does not apply, because T-DELETE has no tunable magnitude.
5. **P2.** Parametric-route interference (memory frozen) on parametric-only base-correct EV facts. Its calibration is in Revision v4.
   - **Recorded limitation (D46):** P1 and P2 use different item populations. A same- or overlapping-population secondary is to be investigated before P2 is frozen.
6. **Monitor training (frozen requirement, D46).** Monitor training data must include ordinary states whose integrated computation goes through the NULL route (parametric-only known facts and unknown facts in the MT split). Monitors read only `probe()` states and output probabilities.
   - **Never available to monitors:** deletion labels, slot masks or occupancy, retrieval weights or success, a_null, retrieval entropy, memory addresses or table, route-dropout state, intervention identity.
7. **E1 / E2.**
   - **Stage-1 success requires E1:** θ_pre(INT-S) and the labels.
   - **E2 is reported, not required:** z(θ_pre INT-S) − z(θ_pre OUT) and the pre-monitor ΔR²_full|output diagnostic.

## Seeds

| Seeds | Role |
|---|---|
| 9041–9043 | v4 development |
| 9051–9053 | v4 validation (once) |
| 9001–9005, 9011–9013, 9021–9023, 9031–9033 | Retired |
| 1001–1040 | Confirmatory (unchanged) |

## Before freezing

The confirmatory pipeline still has to be ported to the dual-route store (store A and twin B, T-DELETE / Y, prestates, P2). T-NEW and T-FAM need a v4 design. These items appear in the calibration package as remaining work.

## v4.1 amendment (2026-10-03; D52–D55). Proposed; NOT frozen, NOT registered

1. **Learning routes (D52).** Architecture and inference are unchanged. During store training:
   - a memory-covered fact presented with its slot available trains only the memory group M (keys, values including NULL, q_ln, W_q, W_o, temperature);
   - all other LM loss trains only the parametric group P (embeddings, all four blocks, ln_f, unembedding): route-dropped covered facts, parametric-only facts and mentions;
   - retrieval supervision trains only addressing (keys, q_ln, W_q, temperature) from a detached query input.

   Parametric backup of a covered fact therefore develops only through its route-dropped presentations (dose ≈ p_rd × E).
2. **p_rd** is selected from the fixed grid {0.05, 0.10, 0.20, 0.35, 0.50} on 9061, under the unchanged F1 rule. It must also show a coherent dose-response (DC-1 across the grid; DC-2 item level on every development seed and at validation).
3. **Seeds.**
   - Development: 9061–9063.
   - Validation: 9051–9053 (untouched until FREEZE).
   - 9041–9043 are v4 history.
4. **Label status (D54).**
   - Only label A is affirmative (confirmatory) support.
   - Label B is reported as ambiguous, change-structure-sensitive tracking and is not support.
   - Label C is no evidence.

   The estimand, statistic and label definitions are unchanged.
5. **M1/M2 (D55).** These are store-side sufficiency diagnostics only. M2 can carry parametric answer information through the query.

## v4.2 amendment (2026-10-03; D57–D60). Final B1 revision. Proposed; NOT frozen, NOT registered

1. **Development procedure (D58).** The store is developed in two stages; architecture and inference are unchanged.
   - **Stage A:** the parametric route alone, with the memory off. A memory-covered fact's LM sequence is included in an epoch with probability p_rd (the developmental dose of parametric backup). E_A = 60 is fixed.
   - **Stage B:** every parametric tensor is frozen, and only the explicit memory learns to write into the fixed computation. E_B = 200.
   - **NULL value:** exactly zero.
   - **Injection norm:** capped at 10 × the frozen network's median pre-injection residual norm.
2. **Seeds.**
   - Development: 9071–9073.
   - Validation: 9051–9053 (untouched until FREEZE).
   - 9061 is v4.1 history; 9062 and 9063 are retired unused.
3. **Additional pre-declared store gates (D59):**
   - F0A dose feasibility;
   - F0B integration: rescue, no damage, INT-1 non-overwrite, NULL-route scale, bit-identical Route-A equivalence;
   - F1C replication.
4. **B1 stopping rule (D60).** v4.2 is the final B1 revision. Any stop ends the B1 line and triggers a final report and pivot memo.
