# C15-R A-stage protocol: task-independent workspace assay (pre-registered, committed before data)

| Field | Value |
|---|---|
| Authorization | PI, 2026-10-04 (D70). Scope: implementation/tests → throughput benchmark → A-stage workspace assay → report to PI |
| Not authorized | B / C / H stages, SAT data, F, rescue routes, any C15 behaviour, further downloads (Qwen3.5-4B, Gemma), the Gemma licence |
| Source designs | `memo/c15r_final_preregistration_design_memo.md` (D69), as amended by the PI (D70). Pre-data operational refinements are listed in §11 |
| Code | `experiments/c15/c15a/` (package), `run_astage.py` (runner), `tools/make_splits.py`, `tests/` |
| Constants | `c15a/config.py` (single source of truth; its SHA-256 `config_hash` is written into every result and the FREEZE record) |

## 1. Candidates and artifacts (pinned; `artifacts_astage.json`)

| Key | Checkpoint (revision) | Lens (neuronpedia/jacobian-lens @ b25d72a9…) | L × d |
|---|---|---|---|
| qwen3-1.7b | Qwen/Qwen3-1.7B @ 70d244cc… | `qwen3-1.7b/…/Qwen3-1.7B_jacobian_lens.pt`, sha256 6fcc7901… | 28 × 2048 |
| qwen3.5-2b | Qwen/Qwen3.5-2B @ 15852e8c… | `qwen3.5-2b/…/Qwen3.5-2B_jacobian_lens.pt`, sha256 69e02a97… | 24 × 2048 |
| qwen3-4b | Qwen/Qwen3-4B @ 1cfa9a72… | `qwen3-4b/…/Qwen3-4B_jacobian_lens.pt`, sha256 9c94a43a… | 36 × 2560 |

**Exact identity of the Qwen3.5-2B pair.**
- The public `qwen3.5-2b-pt` lens was fitted on **Qwen/Qwen3.5-2B-Base**, a distinct checkpoint. It is not used.
- The candidate is the post-trained **Qwen/Qwen3.5-2B** with the `qwen3.5-2b` lens. That lens's config and embedded provenance both name Qwen/Qwen3.5-2B, with target layer 23 = final block.

**Lens convention** (reference code github.com/anthropics/jacobian-lens):
- h_l is the output of decoder block l.
- J_l maps h_l to the output basis of the final block.
- Readout is `lm_head(final_norm(J_l h_l))`.
- The atom for token t is j_{l,t} = J_lᵀ(γ ⊙ W_U[t]), unit-normalised, where γ is the final-norm gain (measured by passing a ones-vector through the norm, so both w and 1 + w conventions are handled).

**Z0 (fails on any mismatch):**
- SHA-256 of every weight shard, tokenizer and lens file against the manifest;
- lens `config.yaml` `hf_model_name` equals the checkpoint ID, and `target_layer` is null;
- lens checkpoint `d_model`, `source_layers` = 0..L−2, J shapes and finiteness;
- embedded provenance `model_id` and `target_layer` (when present);
- model config layer count and width;
- lm_head rows ≥ tokenizer size;
- lens-convention sanity check on SMOKE text: lens top-1 at L−2 agrees with the model's top-1 ≥ 0.30.

Z0 results: `results/raw/c15a/z0_<model>.json`.

## 2. Materials and isolation

**Material.**
- All material is author-written and task-independent: generic paragraphs, concrete nouns with templates, countries with capital/language/currency, and two-hop factual items.
- It is in `materials/source_*.py`.
- A unit test asserts that it contains none of: satisf, clause, valid, verif, assignment.

**Splits.**
- `tools/make_splits.py` (seed 9300) applies tokenizer-only filters: paragraphs ≥ 64 tokens, and nouns single-token with a leading space, under all three tokenizers.
- It then splits each item type 50/50 into **G_select** and **G_confirm**. Two-hop items are split by bridge entity.
- It also writes a disjoint **SMOKE** set.
- Every trial design is drawn at split time and hashed into `materials/split_manifest.json`: W1 contexts and distractors, W2 pairs, W3 concepts/contexts/scalars, dose paragraphs.

**Counts per split:** 63 paragraphs, 20 concepts, 20 countries (20 pairs), 74 two-hop items (37 bridges).

**What G_select is used for:**
- candidate comparison and layer selection;
- S_J construction;
- the V_gen frequency filter;
- the KL matrix;
- the dose rule;
- matching calibration;
- the SI_iso impact matching (s\*).

**Rules for G_confirm (`guards.py`):**
- It is loaded only by CONFIRM.
- CONFIRM requires a FREEZE record that is git-tracked and identical to HEAD, with matching config, material and frozen-tensor hashes.
- It runs **once**: an existing confirm result blocks re-runs or redirection to another model.

**SMOKE and BENCH** use SMOKE material only. Their outputs (`results/raw/c15a/engineering/`) are marked non-evidential and are never read by selection.

## 3. Workspace construction (task-independent)

- **Band layers:** l from round(0.3 L) to round(0.6 L), step 2. That gives 1.7B {8,10,12,14,16}; 3.5-2B {7,9,11,13}; 4B {11,13,15,17,19,21}.
- **V_gen:** tokens of the form 'Ġ' + ≥ 2 ASCII letters, minus the 200 most frequent in G_select.
- **GP:** non-negative gradient pursuit, k = 25, over the V_gen atoms.
  - Each state's search is screened to its 512 atoms with the largest initial correlation, but **only for models where screening validated** on SMOKE against exact GP (`tools/validate_engineering.py`). The criteria are:
    - residual ratio ≤ 1.02;
    - the ablated (removed) component has cosine ≥ 0.99 and norm ratio within 2%.
  - Validation results:
    - Qwen3-1.7B: cosine 0.999, so screened.
    - Qwen3-4B: cosine 0.991, so screened.
    - **Qwen3.5-2B: cosine 0.927, so exact GP is used.**
  - Top-10 *index* overlap is not used: near-tied, near-collinear atoms make indices unstable while the removed component is unchanged.
- **S_J(l):**
  - Take G_select paragraphs truncated to 64 tokens, positions 16..63 (the lens fit skipped early positions).
  - Compute the GP₂₅ reconstructions, centre them, and run PCA.
  - r = min(r₉₀, ⌊d/8⌋); P_J = QQᵀ.
- **ĥ(l)** = the median residual norm at those positions.
- **Not used to define S_J:** SAT, validity or verdict words, C15 outcomes, policies (none exist in this stage).

## 4. Matched non-workspace controls (memo §6, task-independent rows)

For a content vector c (concept vector, or country-vector difference) at layer l:

- **Workspace direction:** u_J = P_J c / ‖P_J c‖.
- **Control family:**
  - u ⊥ S_J, unit norm; same layer, same position, rank 1, same perturbation norm;
  - optimised to be closest to c's own non-workspace component (P_⊥ c);
  - analytic constraints, each within ×/÷ 1.15:
    - lens (output-head) gain ‖W_eff J_l u‖;
    - final propagation ‖J_l u‖.
- **Empirical generic-disruption match** (the binding KL criterion):
  - A pool of up to 12 candidates is generated: proxy-KL targets ×{1, 0.5, 2, 0.25, 4, 8} × 2 seeds, each seed adding its own random ⊥ basis directions. The linearised KL proxy only diversifies the pool.
  - Candidates are taken in decreasing cosine to P_⊥ c.
  - The first whose **measured** mean next-token KL is within ×/÷ 1.15 of u_J's is chosen.
  - KL is measured for a single-position injection at the dose actually used, on 4 generic calibration paragraphs of the current split, over the subsequent positions 25..40. This is the same window as the dose rule. A prefix cache is used, and validated as exact.
  - Direct output-head leverage at the injected position is matched separately, through lens gain.
  - If both KLs are < 1e-3, the candidate counts as matched.
- **Unmatched contents** are dropped from both arms. If more than 20% of a test's contents are unmatched, that test fails: the control could not be constructed.

## 5. Dose rule (G_select; frozen for confirmation)

- The dose a\* is the largest a ∈ {0.25, 0.5, 1, 2} × ĥ(l) such that **every grid dose up to and including it** satisfies both conditions below (monotone admissibility).
- The test is a single-position injection at position 24 of 8 generic paragraphs (48 tokens), along the J direction of 4 concepts.
- Conditions over the subsequent positions 25..40:
  - mean next-token KL ≤ 0.05 nats/token;
  - top-1 agreement ≥ 0.95, counted only at robust positions (clean top-1 leads top-2 by ≥ 1 nat), so that near-tie flips under negligible perturbations are not counted as disruption.
- This is the memo's non-disruptive standard (J arm only), applied to ongoing ordinary text. The injected position's own next-token prediction is the locus of the intended effect, so it is excluded.
- If no grid dose qualifies, a\* = 0.25 and the result is flagged.
- **Where the dose applies:** W1 and W3 injections use a\* · ĥ. W2 uses the natural magnitude of the J-component swap.

## 6. Assays and gates (per layer unless stated)

| Gate | Test | Pass criterion |
|---|---|---|
| **W0** (model-level) | (a) Lens top-1 at L−2 vs model top-1 on the split's paragraphs (positions 16..63). (b) Two-hop items: clean top-1 accuracy on the first answer token; among correct items, does an intermediate-entity alias's first token appear in the lens top-20 at the last prompt position at any band layer? | (a) ≥ 0.60 and (b) ≥ 0.40. Two-hop accuracy ≥ 0.60 is also required for W4 to be assessable |
| **W1** reportability | Concept vectors = per-concept mean of h_l at the noun token over 8 templates, minus the grand mean. Inject a\*ĥ · u (J or matched ⊥) at the last token of a 24-token generic context, followed by "One single word that comes to mind right now:". Readout: 10-way forced choice (target + 9 distractors) among the option tokens. 20 concepts × 5 contexts | hit_J ≥ 0.30 (chance 0.10) and hit_J ≥ 2 × hit_⊥ |
| **W2** cross-function broadcast | Country vectors from 4 templates (last token). Pair A→B: at the last token of A in three queries (capital / language / currency), add P_J(c_B − c_A) (natural magnitude), or the matched ⊥ direction at the same norm. A function is eligible if the answers' first tokens differ and the clean model prefers A's. A switch means log p(B) > log p(A). A pair succeeds with ≥ 2 switches; pairs with < 2 eligible functions are excluded | rate_J ≥ 0.25 and rate_J ≥ 2 × rate_⊥ |
| **W3** cross-position transport | 4 concepts × 40 generic contexts (44 tokens). Inject s · a\*ĥ · u at position 23, with s ∈ {±0.5, ±1} balanced. Sites: positions 24..43 (step 2) × layers l+2..L−2 (step 2). Cross-fitted dual ridge (5 folds over contexts, λ = 0.1 × mean diagonal of the centred kernel) decodes s at each site. BB = fraction of sites with R² ≥ 0.25, averaged over concepts. CI: 200-replicate context bootstrap | BB_J − BB_⊥ ≥ 0.15, lower CI bound > 0 |
| **W4** causal relevance | Ablation window {l−1, l, l+1}. **J arm:** at every position ≥ 1, remove the projection of h onto the span of the top-10 active GP₂₅ atoms. **Random arm:** displacement −s · n · r̂, where n is the J-arm removed norm at the same (layer, item, position) and r̂ is a fixed random unit vector per (layer, item, position), identical across s. Two-hop accuracy impairment in points | impairment_J − impairment_rand(s=1) ≥ 10 pp, with two-hop accuracy ≥ 0.60 |
| **W5** selectivity | Same ablations. Damage = next-token top-1 change rate on the split's paragraphs (positions 16..63). **SI_norm** = damage_J / damage_rand(1) (always reported). **SI_iso** = damage_J / damage_rand(s\*), where s\* is the random-arm scale reaching the J arm's two-hop impairment. s\* is found on G_select by linear interpolation in log s over s ∈ {1, 1.5, 2, 3, 4, 6, 8}; per-sequence damage is interpolated the same way. 95% CI: 2,000-replicate paired bootstrap over sequences | **Binding:** SI_iso ≤ 1.0 and CI upper bound ≤ 1.25. Certifiable only if s\* is interpolated, or is an unreached upper bound at s = 8 (a conservative SI_iso). If the random arm already matches J's impairment at s = 1, SI_iso is a lower bound and W5 cannot pass |

**Mandated wording.** If SI_iso passes but SI_norm > 1, the result is described as **"selective at matched functional impact but anti-selective at matched intervention norm"**. SI_norm is never a kill criterion and is never omitted.

## 7. Selection, freeze, confirmation (PI amendments 2–3)

1. **SELECT:** all three models on G_select, every band layer. A (model, layer) passes if W0 and W1–W5 all pass.
2. **CHOOSE:**
   - Exactly one (model, layer) is chosen: the one with the highest minimum normalised margin across W1–W5.
   - Margins:
     - W1, W2: min((hit − thr)/thr, (min(ratio, 10) − 2)/2);
     - W3: (diff − 0.15)/0.15;
     - W4: (diff − 10)/10;
     - W5: 1 − SI_iso.
   - Ties go to the smaller model, then the lower layer.
   - The best passer of a *different* model is recorded as the designated replication candidate (amendment 9). It is not confirmed now.
3. **FREEZE:**
   - Frozen items: model; layer; S_J basis Q (tensor, hashed); r; the V_gen ids (hashed); the KL matrix (hashed); ĥ; a\*; s\* and its status; all thresholds (`config_hash`); G_confirm hash.
   - The record `results/raw/c15a/freeze_astage.json` is **committed before CONFIRM**.
4. **CONFIRM:**
   - W0–W5 are run **once** on G_confirm for the frozen model and layer.
   - Everything above is reused unchanged. **s\* is not re-matched**: the random arm runs at the frozen s\* (and at s = 1 for SI_norm and W4). The confirm-split impairment at s\* is reported for transparency only.
   - Content vectors and matched controls are recomputed on the confirm material by the frozen procedure.
5. **Outcome:**
   - The workspace-like label requires all of W0–W5 on G_confirm.
   - If confirmation fails, the A-stage **STOPS**: no other model is selected after G_confirm is seen, and no model-level validation hierarchy is defined.

## 8. Stop conditions (A-stage)

| Code | Condition | Action |
|---|---|---|
| ST-0 | Z0 fails for a candidate | Candidate excluded |
| ST-1 | Candidate fails W0–W5 at every band layer on G_select | Candidate excluded |
| **ST-2** | No (model, layer) passes on G_select | **STOP; report to the PI.** No further downloads (amendment 10) |
| **ST-2c** | The chosen model/layer fails confirmation | **STOP; report.** No re-selection |
| ST-G | Any guard trips (dirty tree, unauthorized phase, hash mismatch, repeated confirmation) | Run refused |

**Success** means one model/layer selected and independently confirmed. The full A-stage report then goes to the PI. B/C/H require a new PI decision.

## 9. Provenance

Every result JSON records:
- git HEAD;
- `config_hash`;
- artifact-manifest and split-manifest hashes;
- library versions and thread count.

SELECT and CONFIRM refuse to run on a dirty `experiments/c15` tree. Select-stage tensors (Q per layer, KL matrix, V_gen) are hashed in `select_<model>.json`; frozen tensors are hashed in the FREEZE record.

## 10. Compute (BENCH, fp32, this CPU; estimates)

| Model | Tokens/s (batch 8) | GP states/s | Projected SELECT |
|---|---|---|---|
| Qwen3-1.7B | 59 | 39 | ≈ 3–4.5 h |
| Qwen3.5-2B | 67 | 25 | ≈ 3–4 h |
| Qwen3-4B | 22 | 30 | ≈ 9–12 h |

The projections include the empirical matching pool, which BENCH did not measure. CONFIRM (one layer) takes ≈ 1–3 h.

## 11. Pre-data operational refinements (relative to D69; no A-stage data existed)

These were decided on engineering evidence from SMOKE material only, and all of them tighten or clarify the protocol.

1. **Dose rule.** The approved memo's non-disruptive standard (KL ≤ 0.05, agreement ≥ 0.95, J arm), measured on the 16 positions after the injection, replaces a looser draft (agreement ≥ 0.80).
   - Reason: on SMOKE the draft rule did not bind and allowed a 2× residual-norm dose, where single-position KL grows steeply (0.009 → 0.049 → 0.37 nats for 0.5 → 1 → 2 × ĥ).
   - A variant that included the injected position could never pass: that position's own top-1 flips even at 0.25 × ĥ, costing 1/17 of the window.
2. **Generic-KL matching is empirical.** The linearised KL proxy did not track measured KL on SMOKE (proxy ≈ 1e-4 with no rank relation; measured 0.12–0.42). The proxy now only diversifies the candidate pool, and the binding match is on measured KL.
3. **Dose-rule details.** Agreement is counted at robust positions, and admissibility is monotone. Reason: on SMOKE, agreement sat at about 0.94 at every dose including 0.25 × ĥ, because of near-tie flips, and a "largest passing dose" rule then jumped non-monotonically.
4. **Matching window, pool diversity, speed.**
   - KL matching uses the dose-rule window (subsequent positions).
   - Pool seeds add random basis directions.
   - GP is screened to the top-512 atoms (validated).
   - Empirical-KL calls reuse a prefix cache (validated exact).
   - Reasons: on SMOKE every pool candidate for some concepts was 2–3× more disruptive than u_J, and the projected SELECT time was about 40 h.
5. **W2 magnitude.** W2 uses the natural J-component swap magnitude (Gurnee-style), with the ⊥ control at the same norm.
6. **SI_iso certification.** W5 cannot pass when SI_iso is only a lower bound (random arm at s = 1 already reaches the J impairment). When it is an upper bound (unreached at s = 8), W5 may pass only if that bound is ≤ 1.
