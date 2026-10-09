# C16 Stage 0 / Stage 1 report: Stage 0 STOP (instrument); Stage 1 finds two structural design failures

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Plan | `memo/c16_s0s1_preregistration_FROZEN.md` and `c16_s0s1_thresholds_FROZEN.json` (D79) |
| Pre-run commit | **aecfe0c** (package, prereg, thresholds, manifest, audit). S0 results: 1902ce7. S1 results and this report: see D80 |
| Scope honoured | Cached Qwen2.5-0.5B-Instruct, forward only (plus the E4 timing probe, which asserts no parameter update). Synthetic and planted Stage 1. **No Stage-2 training. No downloads. Sealed X_select / X_confirm and eval names never used** |
| Claims | None about workspaces, LMs or GWT. Stage 0/1 validate instruments only |

## 1. Verdict

1. **Stage 0: STOP at V1 under F1 and under the pre-declared F2 fallback.** The frozen rule then forbids any model or domain change without the PI.
   - Qwen2.5-0.5B is **not competent in the numbers domain**.
   - Producer competence (copy): add 0.55, sub 0.24 under F2 (needs ≥ 0.80). mul is 1.00.
   - With X written in the text, only one of five pool consumers reaches 0.80. Untrained consumer U (verbalization): 0.20.
2. **Stage 1 shows the failure is not only the model's.** A planted 4-layer transformer was built to be competent and to have the native gap; it passes V1 completely. It still fails two instrument gates:
   - **V0 (eager computation):** X is only weakly decodable where the producer sentence ends (add 0.58, sub 0.43). The model computes X *when asked*, not when the statement appears.
   - **V2 (oracle reachability):** transplanting the text-run residual at the shared prefix gives ≈ 0 gain (≤ 0.06), so the oracle's swap transport is ≈ 0. The source run does not carry X at the prefix positions; consumers read the digits in the sentence directly.

   Workspaces trained on this system (A, C1, C3; 3 seeds each) do not reach the transport positive control. A's trained-pairing accuracy rises only +0.007 over native; its CT on trained pairings is 0.10 (needs ≥ 0.5); CT_U ≈ 0 for every arm.

   The engineering checks pass: zero-gate identity, A's cue invariance, NC neutrality, and **V6** (the fixed C1 transfers to paraphrased cues: 1.21× its trained-cue CT).
3. **Interpretation.** Two of C16's design assumptions fail even in a system built to satisfy the competence and gap requirements:
   - (a) a **blind writer reading the producer region** has content available to write (eager producer computation);
   - (b) the **prefix-position oracle** delivers that content.

   So Stage 0's STOP on the LM is uninterpretable in two directions: the model lacks competence, *and* the instrument's write window and oracle are structurally invalid. No workspace hypothesis has been tested.
4. **S1b passes.** The rank estimator recovers planted ranks exactly (4 / 16 / 64; bundles 16 / 64).
5. **S1a.**
   - Core classification is FPR-controlled (≤ 0.048), with power 0.70–0.86.
   - H-CD cannot confirm the native-copy account (P = 0.00), so it fails the pre-declared centrality rule.
   - The planted pilot grid collapsed because r_X was undefined; it contributes only σ_seed ≈ 0.26 (§5).

## 2. Stage 0 detail (Qwen2.5-0.5B-Instruct)

| Measure | F1 (zero-shot) | F2 (1-shot primer, pre-declared fallback) | Gate |
|---|---|---|---|
| V0 decodability at L_w = 8 (add / sub / mul) | 0.675 / 0.622 / 1.00 | **0.761 / 0.706** / 1.00 | ≥ 0.70 (add and sub) |
| Copy = producer competence (add / sub / mul) | 0.41 / 0.29 / 0.98 | **0.55 / 0.24** / 1.00 | ≥ 0.80 (add and sub) |
| Text-consumer accuracy: succ / plus10 / parity / mag / lookup / **verb (U)** | 0.09 / 0.02 / 0.00 / 0.00 / 0.00 / 0.00 | **0.89** / 0.79 / 0.61 / 0.56 / 0.65 / **0.20** | ≥ 0.80 for 4 of 5 pool consumers and for U |
| Eligibility (copy ∧ text), pooled add/sub | — | 0.06 (verb) to 0.39 (copy) | ≥ 0.50 |
| Native κ on eligible items: succ / plus10 / parity / mag / lookup / verb | — | 0.01 / 0.04 / 0.11 / 1.00 / 0.33 / 0.00 (eligible n are small) | ≤ 0.35 |

**Gates:** F1 fails V0 and V1a/b/d; F2 passes V0 but fails V1a/b/c/d → **STOP** (prereg §4).

**Descriptive observations** (no decision rests on these):
- **F1's failure is mostly answer format.** With X in text, the zero-shot raw format elicits the expected answer token for copy (1.00) but almost never for the other consumers. The primer lifts succ to 0.89.
- **Real incompetence remains under F2.**
  - Arithmetic producers: add 0.55, sub 0.24.
  - Binary judgments with X written out are near chance: parity 0.61, magnitude 0.56.
  - Verbalization: 0.20.
- **mul (times-table) producers are fully competent** (1.00). With mul as producer, CoT (copy ∧ text) reaches 1.00 for succ and 0.92 for plus10.
- **Decodability peaks early and declines** (add: 0.84 at layer 3, 0.76 at layer 8). L_w is restricted to layers ≥ 8 by the prereg.
- **V0 has an identification weakness.** X's digits can be partly predicted from operand identity without X being computed; mul decodability is 1.00 from layer 3 onward, over only 13 distinct values. A valid V0 needs decoders tested on **held-out operand combinations** (new a, b for the same X) or a causal test (lesson R9 below).

**Engineering:**
- zero-patch and hook identity exact (0.0);
- forward ≈ 244 tok/s;
- E4 forward+backward ≈ 0.47–0.86 s/example (median of 3), with no parameter update (asserted);
- S0 wall time 40 min.

## 3. S1b rank recovery: PASS

| Case | True rank | r* | Within ×2 |
|---|---|---|---|
| content | 4 / 16 / 64 | 4 / 16 / 64 | pass |
| bundle | 16 / 64 | 16 / 64 | pass |

The estimator itself is sound on planted linear-Gaussian data.

## 4. S1c planted system (pipeline outputs only; not evidence about LMs or H-CD)

**Base model.** 4 layers, d = 128, trained 4000 steps (final loss 0.003).

**Planted Stage 0:**

| Gate | Result |
|---|---|
| V0 | **FAIL**: add 0.575, sub 0.431 at L_w = 3 |
| V1a–d | **all pass**: all five pool consumers competent; native κ ≤ 0.35; eligibility ≥ 0.50 |
| V2a / V2b | **FAIL**: full content-oracle gain ≤ 0.06 at every layer (0, 1, 2) for every consumer; oracle CT 0.00–0.01 |
| V2c | pass |
| W1–W4 | fail: no rank defined because gains are ≈ 0 |

**Arms** (capacity 2r_X, falling back to d because r_X was undefined; 3 seeds; evaluation on planted eval values):

| Arm | TR acc | TR CT | HO CT | U CT |
|---|---|---|---|---|
| A | 0.32 / 0.44 / 0.52 | 0.10 / 0.09 / 0.10 | 0.15 / 0.18 / 0.10 | −0.01 / −0.01 / −0.01 |
| C1 (cue-derived) | 0.50 / 0.47 / 0.55 | 0.14 / 0.13 / 0.12 | 0.26 / 0.22 / 0.13 | ≈ 0 |
| C3 (unrestricted) | 0.57 / 0.61 / 0.58 | 0.02 / 0.08 / −0.12 | 0.21 / 0.16 / 0.05 | ≈ 0 |

**S1c-2:**

| Check | Result |
|---|---|
| Zero-gate identity | pass |
| A cue invariance | pass (exact) |
| **V6** (C1 paraphrase / trained-cue CT) | **1.21 → pass** |
| NC neutrality | pass |
| Training reach | **+0.007 → FAIL** |
| Transport positive control (A, CT_TR) | **0.10 → FAIL** |

S1c-2 overall: **FAIL**.

**Reading.** The planted model computes X lazily, at the query. The blind writer reads the producer sentence at L_w, which holds operands, not X, so there is little content to carry. The prefix oracle carries nothing because no source position before the consumer-specific tokens holds X. Trained arms then learn small, generic adjustments (CT ≤ 0.26 on held-out pairings, ≈ 0 for U), and none passes the positive control.

## 5. S1a and the planted pilot

### Planted pilot (variance only)

- **The grid collapsed.** The planted r_X was undefined (oracle gain ≈ 0), so the pre-declared fallback r_X = d = 128 applied. Capacities 1, 2, 4 and 8 × r_X were all capped at d_w = 128 and are identical runs; only 0.5 × r_X (d_w = 64) differs.
- **Results:** every cell gives CT_U between −0.017 and +0.005 across all seeds. TR CT is 0.00 for S1 = {succ}, up to 0.12 for S4, and up to 0.25 for Spm.
- **Output:** the only usable quantity is σ_seed ≈ 0.26 on the logit scale, a single estimate from the core held-out cell. Per the prereg, none of this is evidence about H-CD.

### S1a (synthetic)

Inputs: σ_seed = 0.26. p_nc = 0.05 and r_B/r_X = (1, 2, 4, 0.5, ∞) are pre-declared defaults, because Stage 0 produced no valid r_X or r_B.

**Core classification** (memo §4.7 with A1):
- Complete-gate over-claim FPR is **0.035** (σ_X = 0.3) and **0.048** (σ_X = 0.6), both ≤ 0.05, using the pre-declared margins (no adjustment step needed).
- Power for the true class is **0.86** (σ_X = 0.3) and **0.70** (σ_X = 0.6).
- At high cluster variance, the planned 3 seeds × 22 values × 4 instances is **under-powered**; a Stage 2 would need more items per cell or more seeds.

**H-CD three-account discrimination:**
- The geometry and IB worlds are identified with P ≥ 0.92 at every design size.
- The **native-copy world is never identified** (P = 0.00–0.01, even at 8 seeds × 200 items per cell). Confirming copy requires all three contrasts (D, Q, I) to be *equivalent* to 0 within ±0.10. The difference CIs between independently trained cells are about ±0.085, so the intersection of three equivalence tests almost never passes. The copy world is reported as "undetermined", not misassigned (wrong-account rate ≤ 0.043).
- So **H-CD fails the pre-declared centrality rule** (power ≥ 0.80 for every account). As designed it can discriminate geometry from IB but cannot confirm the null-like copy account. Confirming it would need a different estimand (e.g. a pooled slope test) or a much larger n.

## 6. Lessons (added to R1–R8)

| Code | Lesson | Evidence |
|---|---|---|
| **R9** | **Eager-computation validity.** A blind (write-before-cue) channel can only carry content that is computed before the write window closes. Both the LM and a planted transformer compute latent results *lazily*, at the query. A valid design needs a write site where the producer's result is demonstrably computed, e.g. the position whose next-token prediction is X. And V0 must use decoders tested on **held-out operand combinations**, because operand identity alone predicts X's digits | Planted V0 0.58 / 0.43 with full V1 competence; LM decodability peaks at layer 3 and mul reaches 1.00 from operand identity |
| **R10** | **Oracle-carriage validity.** A transplant oracle must use source positions that are shown to carry the content (a decodability check at the transplant site). Causal precedence alone ("the prefix precedes every consumer-specific token") does not make a position a carrier | Planted content-oracle gain ≤ 0.06 at every layer, with competent consumers |
| **R11** | **Grid units need a defined anchor.** A capacity grid in units of r_X collapses when r_X is undefined. Gate the grid on a defined anchor before scheduling it | Planted pilot: four of five capacity levels identical |
| **R12** | **Confirming a null-like account needs its own power.** An intersection of three equivalence tests is far less powerful than directional tests at the same n | S1a copy-world P = 0.00 |

## 7. Integrity and deviations

- **Pre-run commit:** aecfe0c, made before any model saw C16 material. Guards verified at every phase.
- **Data hygiene:** sealed X_select / X_confirm values and eval names were never used.
- **No training on the HF model:** E4 asserted no parameter or LM-weight update. No downloads.
- **Engineering before freezing** (recorded in D79): the HF smoke and timing used only out-of-domain single-digit numbers; the planted/synthetic smoke used tiny untrained models.
- **Deviations:** none in method. One operational note: a draft of this report was first written inside the guarded package directory while S1C ran and was moved out before S1A started (the guard would otherwise have refused S1A). No effect on any result.
- **Wall time:** S0 40 min (F1 14 min + F2 27 min); S1B 5 s; S1C 2 h 38 min; S1A 4 min.

## 8. What this means and the decision for the PI

**No workspace hypothesis has been tested.** The kill sequence did its job in about 3.5 CPU-h. It shows that:
- (1) the cached Qwen2.5-0.5B is not competent in the chosen domain;
- (2) more fundamentally, C16's blind-write design rests on two assumptions — eager producer computation and prefix carriage — that fail even in a planted transformer built to have competence and a native gap.

A Stage 2 on the current design would test nothing.

**Options (nothing started):**
- **A. Close C16** (my recommendation). Fold C15 (A-stage + R2) and C16 S0/S1 into one methods / negative-results note:
  - instrument-validity failures in workspace assays of small LMs;
  - lessons R1–R12;
  - the fair addressed-control design (A1);
  - the forward-citation audit.
- **B. One bounded C16-R redesign** (design memo first, then a fresh S0/S1 of about 6 CPU-h). It would use:
  - *eliciting* producers (the write site is the position that predicts X);
  - V0 on held-out operand combinations;
  - an oracle at a carriage-validated site;
  - a competence-screened domain (e.g. times-table producers, which the 0.5B model handles at 1.00, with consumers re-screened);
  - possibly the cached Qwen3-1.7B, which needs your approval.

  These failures are instrument failures, not outcome data, so the forking-path risk of redesigning is modest. But the adversarial point stands: forcing the producer to predict X at the write site makes the "content" a pre-verbal next-token representation. Transporting it is close to soft-token re-entry (Coconut-like). The scientific interest of the integration test shrinks accordingly, and novelty is already narrow (≈ 0.5, Marincat 2026).
- **C. Pivot** to a different question.

**Significance confidence for C16 as designed: LOW.** After a redesign it would be LOW-to-MODERATE at best; P(Level 5) is well below 0.2.
