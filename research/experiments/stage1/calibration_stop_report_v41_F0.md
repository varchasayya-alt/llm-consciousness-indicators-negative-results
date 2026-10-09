# Calibration STOP report: v4.1 (gradient-isolated learning) at V4.1-F0, kill condition K-1

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | **Methods ladder stopped under the pre-declared rule.** V4.1-F0 failed the store QC after the pre-declared C1 rule. This is K-1 (`v4_design_memo.md` §19; `calibration_plan.md` Revision v4.1) |
| Pre-run commit | `3539d3d`: code, partition, 11 isolation tests (54 tests pass), config 0.6-v4.1, plan Revision v4.1, D52–D55. Committed before any development seed |
| Seeds touched | **9061 only, for F0** (store-sanity training at p_rd = 0.35). No p_rd selection was made |
| Seeds untouched | 9062, 9063; **validation 9051–9053**; confirmatory 1001–1040 |
| Not run | F1 grid, F1 confirmation, F2–F4, P2, FREEZE, VAL. No monitor trained; nothing frozen or registered |
| Raw results | `results/raw/calibration/calibration_results_v41.json`, `log_v41_F0.txt`, `diag_v41_9061_*.json`, `diag_v4_9041_*_reference.json`, `engineering_dryrun_v41_burned_seed.txt` |

## 1. The isolation itself was implemented cleanly

- **Exhaustive P/M partition** (D52):
  - P: tok, pos, all four blocks with their LayerNorms, ln_f, unembed.
  - M: keys, values (including NULL), q_ln, W_q, W_o, temperature.
- **Exact invariants** (unit tests plus a runtime audit on the first batch of every epoch):
  - a memory-present covered batch gives exactly zero gradient to every P tensor, and one optimizer step leaves P bit-identical;
  - route-dropped and parametric-only batches train every P tensor and give zero answer gradient to M;
  - NULL supervision reaches only addressing.

  No violation occurred in any epoch.
- **Equivalence check.** The parametric route matches v4 when no row is memory-present (burned seed, p_rd = 1).

So the kill condition is **not** "isolation cannot be implemented". The cleanly isolated store fails the integrated-store sanity gate.

## 2. V4.1-F0 results (9061, p_rd = 0.35)

| Training | Trained-fact accuracy every 10 epochs | Final | Gate |
|---|---|---|---|
| 60 epochs | 0.065, **0.325**, 0.116, 0.140, 0.145, 0.163 | 0.163 | ≥ 0.98: missed → C1 rule |
| C1 rule, 200 epochs | 0.06, 0.30, 0.25, 0.25, 0.37, 0.47, 0.62, 0.65, 0.79, 0.84, **0.918**, 0.912, 0.860, 0.790, 0.736, 0.729, 0.723, 0.735, 0.714, 0.701 | 0.701 | E99 never reached, so E = 200 |
| Retrain at E = 200 (pre-declared; deterministic, identical to the C1 run) | identical, number for number | **0.701**: covered 0.580, parametric-only 0.998, unknown 0.035, fluency AUROC 1.00, own-slot attention 0.898 (< 0.90), NULL 0.993 | QC FAIL (trained accuracy and own-slot attention) → **K-1 STOP** |

Both schedules rise and then collapse: 0.325 at epoch 20 of 60, and 0.918 at epoch 110 of 200. The integrated store never comes close to 0.98.

## 3. Failure mechanism (store-only diagnostic, `diag_v41_store.py`)

Samples are 3,000 covered and 3,000 parametric-only trained facts. "Route A" means all slots masked.

| Quantity | v4 reference (9041, 60 ep) | v4.1 9061, 60 ep | v4.1 9061, 200 ep (C1) |
|---|---|---|---|
| Covered: own-slot attention | 0.988 | 0.932 | 0.896 |
| Covered: answer linearly decodable from memory term u (M2) | 0.998 | 0.984 | 0.981 |
| Covered: **integrated** accuracy (memory present) | 1.000 | **0.046** | **0.575** |
| Covered: route-A accuracy (weights only) | 0.792 | 0.104 | **0.987** |
| Parametric-only: integrated / route-A accuracy | 1.000 / 0.989 | 0.450 / 0.441 | 0.998 / 0.998 |
| Parametric-only: NULL attention | 0.991 | 0.985 | 0.993 |
| ‖hA‖ (pre-injection [A] state) | 6.9 | 1,341 | 4,518 |
| ‖u‖ on covered rows (own slot) | 21.3 | 336 | 3,031 |
| ‖u‖ on parametric-only rows (NULL term) | 2.1 | 2,417 | 11,050 |
| ‖W_o‖ (Frobenius); ‖v_NULL‖ | 63.5; 0.22 | 136; 24.0 | 398; 36.7 |

**Reading.**

1. **Addressing works and the memory holds the answer.** Own-slot attention is about 0.9 and M2 is about 0.98.
2. **The downstream does not decode the memory.** With its slot present, a covered fact is answered *worse* than with every slot masked: 0.575 vs 0.987 at 200 epochs, and 0.046 vs 0.104 at 60. The memory term disrupts the parametric answer instead of supplying it.
3. **Runaway scales.** The residual, the NULL injection and W_o grow by two to three orders of magnitude relative to v4, and the accuracy curves collapse.

**Why this follows from the approved routing.**

- **No example trains the integrated computation end to end.**
  - Memory-present rows train only M, through blocks 3–4, ln_f and unembed, which they cannot adapt.
  - Every other row trains P, with the injected memory term treated as an unadaptable input.
- **The downstream is never trained to read the memory channel.** Blocks 3–4 and the readout learn only from rows where the injection is the NULL term. The memory must therefore chase a moving readout that it cannot influence.
- **The memory-interface scale is anchored by nothing but that frozen-for-it downstream.** That covers W_o, the slot values and the NULL value. The NULL term enters every parametric row as an exogenous offset that P cannot reduce, and P can only out-scale it.
- The result is the observed mutual escalation and non-decodable memory.

**My implementation choice contributes.** D52 blocks answer gradients from parametric-route rows to *all* of M, including the non-fact-specific NULL value and W_o. That is stricter than the PI's minimum, which only forbids *fact-specific* shortcuts. It leaves the NULL injection uncontrolled, which feeds part of the escalation. It does not explain the second component: a downstream that is never trained to read memory. That component follows from the PI's explicit requirement that blocks 3–4 not train on memory-present examples.

**Side observation.** At p_rd = 0.35 with E = 200 (about 70 route-dropped presentations per covered fact), the weights alone answer 0.987 of covered facts. Under the C1 rule, the dose scale p_rd × E can saturate backup well above the [0.30, 0.70] window.

## 4. Options for the PI (decision needed; nothing implemented)

**O4 (recommended if B1 engineering continues): sequential development.**

- **Stage A:** train the parametric route alone. The memory is absent, or a fixed zero NULL term. Covered facts receive parametric presentations only with probability p_rd per epoch (the same dose logic). Parametric-only facts and mentions train normally.
- **Stage B:** freeze **all** of P and train only M, on memory-present covered rows plus NULL supervision (keys, values, NULL, W_q, W_o, temperature). Parametric-only and unknown rows may train the NULL value, which is not fact-specific, so the NULL term stays benign.
- **What this changes.** P never receives a memory-present answer gradient, as in v4.1. Unlike v4.1, the downstream is stationary while the memory learns to write a code it decodes, and P can never respond to the memory. The escalation loop is impossible by construction.
- **What stays the same:** the architecture, T-DELETE, Y, every gate, the fixed grid, DC-1 and DC-2.
- **Requirements:** a new committed revision and fresh development seeds (proposed 9071–9073; 9061 is now used). 9051–9053 stay untouched.
- **Risk:** the memory may still fail to write decodable codes into a fixed readout, but that is now a stationary optimisation problem. F0 would tell.
- **Cost:** about 0.5 day.

**O1-N (minimal variant of v4.1; not recommended alone).** Allow parametric-route answer gradients into the NULL value only (or the NULL value and W_o). This addresses the NULL escalation but not the undecoded memory channel.

**O5: stop engineering the synthetic B1 store.** Reconsider the Stage-1 design (the PI's own criterion after a second failure). Two consecutive B1 variants have failed for different structural reasons:

- in v4, shared training makes backup uncontrollable;
- in v4.1, isolated co-training makes the integrated store non-viable.

**Not proposed:**

- relaxing the F0 threshold or the C1 rule;
- letting blocks 3–4 train on memory-present examples (this reopens the v4 leak the PI excluded);
- reusing 9061 for a new revision's selection;
- touching 9051–9053.
