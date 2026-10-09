# C15-R2 FROZEN PRE-RUN PLAN (preregistration for PI approval)

| Field | Value |
|---|---|
| Date | 2026-10-07 |
| Supersedes | `c15r2_workspace_assay_repair_memo.md` (D72) wherever they differ. This document incorporates the PI's final amendments 1–7 (D73) |
| Status | **Frozen design (approved in principle by the PI; D74 details in §12). No model run before the pre-run commit.** Nothing in `experiments/c15/` or `results/raw/c15a/` is modified |
| Machine-readable thresholds | `memo/c15r2_thresholds_FROZEN.json`. The R2 package must assert equality with these values |
| Simulations | `memo/c15r2_power/` (synthetic only): `r2_power_sim.py` (A–C) and `r2_power_sim_v2.py` (final W3′) |
| Claims discipline | Functional organisation only; no outcome bears on phenomenal consciousness |

## 1. Integrity structure (amendment 1)

- **Fresh material.** Entirely fresh G_select2 and G_confirm2, with no item or entity overlap with any A-stage material.
- **A-stage material.** The original G_select remains spent. The original G_confirm remains sealed (sha256 `3f0911cb…03b0dc`; never run).
- **Gates.** W1, W2, W4 and W5 pass criteria are unchanged.
- **Prior results.** No A-stage result counts as evidence. The A-stage pattern only generates the hypotheses H-F1/H-F2 (§7.3), tested on fresh data.
- **Separate package.** R2 is implemented as `c15a2` under `experiments/c15r2/`. The A-stage code (git tree `31f6d6e7…`) and results (tree `54a3f123…`) are read-only. Their hashes are re-verified at the start and end of R2.

## 2. Candidates, artifacts, unchanged procedures

- **Candidates:** Qwen3-1.7B, Qwen3.5-2B and Qwen3-4B, with the pinned checkpoints and lenses of `experiments/c15/artifacts_astage.json`. No downloads.
- **Unchanged procedures:**
  - S_J (generic-PCA of GP₂₅ reconstructions, r = min(r₉₀, ⌊d/8⌋));
  - V_gen; per-model GP screening; band layers [0.3 L, 0.6 L] step 2;
  - dose rule (subsequent-text KL ≤ 0.05; robust agreement ≥ 0.95; monotone);
  - matched-⊥ construction (empirical KL within ×/÷1.15; lens gain; propagation);
  - W1, W2, W4 and W5 exactly as in `experiments/c15/astage_protocol.md`.

## 3. W3′: identity-specific transport (amendment 2)

### 3.1 Estimand

> Does the downstream state carry the **identity** of injected content? Downstream states must identify *which* concept was injected, against references built from paraphrase mentions that never contain the concept word.

**Merely changing later computation is not enough.** Generic downstream influence that is common to all injected concepts cancels exactly, by within-context centring.

### 3.2 Material (per split; authored and hashed before any model run)

**Slot contexts.** 48 contexts: generic prefix (≥ 20 tokens) + slot phrase + generic continuation (≥ 24 tokens) that does not reveal the slot content. Two context folds (A, B) of 24.

**Concepts.** 8 per split, drawn from that split's 20 W1 concepts and from distinct semantic categories.

**Paraphrases.** 3 per concept (p1, p2, p3).
- No paraphrase may contain the concept word, its plural or morphological variants, or any token from the concept word's tokenization in any of the three tokenizers (checked automatically at build time).
- Example for "apple": "the round fruit that grows in an orchard".

**Slot versions:**
- (i) paraphrase p_j of concept k;
- (ii) the word itself, " the w_k" (lexical diagnostic only);
- (iii) the neutral " the thing" (injection carrier).

### 3.3 Construction

Sites are at offsets +1…+23 (step 2) after the slot phrase end, crossed with layers ℓ+2…L−2 (step 2). The continuation is identical across versions, so offsets align.

- **Reference.**
  > F̂_k(site) = mean over c ∈ A and p ∈ {p1, p2} of [ h(c, p_k) − mean_{k′} h(c, p_k′) ]
  - The reference is centred across concepts within each context.
  - It is built from paraphrases only.
- **Natural benchmark.**
  - acc_nat(site) = identification accuracy of the held-out paraphrase p3 in fold B against F̂.
  - Identification uses argmax_{k′} cos(·, F̂_k′), with each sample centred the same way.
- **Injection.**
  - In fold B with the " the thing" slot, add a\*·ĥ·u_k^R to the " thing" token at layer ℓ, for R ∈ {J, ⊥}.
  - u^J = P_J c_k / ‖P_J c_k‖ and u^⊥ is the matched control (both unchanged).
  - The injection effect is centred across the 8 injected concepts within each context:
  > D_ck^R = h(c, inj_k^R) − mean_{k′} h(c, inj_k′^R)
- **Identification accuracy.** acc_R(site) = the fraction of the 8 × 24 = 192 (c, k) trials with argmax_{k′} cos(D_ck^R, F̂_k′) = k. Chance is 0.125.
- **Site criterion: significance and effect-size floor, both pre-registered.**
  1. one-sided binomial p < 0.05 against chance; **and**
  2. acc_R(site) ≥ max(**0.30**, **0.5 × acc_nat(site)**).
- **Gate (form unchanged from W3):** BB_J − BB_⊥ ≥ 0.15, with the 95% bootstrap lower bound > 0 (bootstrap over contexts × concepts, 1,000 replicates). BB_R is the fraction of sites meeting the criterion.
- **Why the reference cannot be passed by trivial lexical carryover.**
  - The reference contains no surface token of the concept word.
  - A downstream signal that reflects only the word's surface identity therefore cannot be identified against it.
  - In simulation, pure surface carryover passes 0% of sites against the paraphrase reference and 100% against a word reference (§8.2).
  - What remains detectable is concept-level content shared with paraphrase mentions. That is the intended target.

### 3.4 Validity controls (W3′ is *not assessable* in a cell if any of PC1–PC3, NC1 or NC2 fails)

| Control | Requirement |
|---|---|
| PC1 reference reliability | Median over concepts of cos(F̂_k from p1, F̂_k from p2) ≥ 0.5, at ≥ 50% of sites |
| PC2 natural identity transport | acc_nat ≥ 0.30 and binomial p < 0.05, at ≥ 50% of sites. Without it, real paraphrase mentions do not carry identity downstream, so transport cannot be tested |
| PC3 injection-site identity | At offset 0 (the " thing" token), layer ℓ+2: identification ≥ 0.30, p < 0.05, for at least one route. Shows the injected content is identity-bearing locally |
| PC4 planted transport (engineering, before any G data) | Tiny-model unit test with a planted copy mechanism: planted content gives BB ≥ 0.8; a null gives BB ≤ 0.1. **SMOKE2:** PC3 must pass on each real model |
| NC1 random-direction sham | Norm-matched random directions as "concepts": BB ≤ 0.10 |
| NC2 label-permutation sham | J-arm effects scored with deranged concept labels: BB ≤ 0.10 |
| NC3 lexical sham (diagnostic, not gating) | Inject the normalised input embedding of w_k (same norm and site), and score the J arm against word references. Reports a *lexical-equivalence index*. Interpretation only |

**W3b** (behavioural delayed recall; secondary, not gating) is retained as in D72.

## 4. W0b′: intermediate readout at justified positions

| Element | Rule |
|---|---|
| Positions | t₁ = last token of the first-hop entity (Biran et al. 2024); t_d = last token of the bridge's descriptive mention (Yang et al. 2024; jlens example); t₂ = final token. Character spans are annotated at authoring time |
| Layers / readout | Band layers (unchanged); J-lens top-20 (unchanged) |
| Foils | 5 same-type entities per item, pre-assigned at authoring time, scored identically |
| Gate | Two-hop accuracy ≥ 0.60; intermediate rate ≥ 0.40 (unchanged thresholds); **and** foil lift ≥ 0.20 with one-sided 95% lower bound > 0 (added; stricter) |

## 5. Material isolation and hashes

| Item | Plan |
|---|---|
| Location | `experiments/c15r2/materials2/`: authored `source_*.py`, `make_splits2.py`, `g_select2.json`, `g_confirm2.json`, `smoke2.json`, `split_manifest2.json` (SHA-256 of every file). Results go to `results/raw/c15r2/` |
| Splits / seeds | Seed 9500 (splits), 9501 (designs), 9502 (rng). Two-hop items are split by bridge; slot contexts by context id |
| Disjointness (unit tests) | T2: empty intersection with all A-stage material (paragraph texts, concept words, country names, bridges, intermediates, templates; case-insensitive, including substrings of multi-word names). T3: select2 ∩ confirm2 ∩ smoke2 = ∅. A vocabulary audit (no SAT/validity terms), as before |
| Sealing | T1: no `c15a2` file references `experiments/c15/materials/g_confirm.json` or `g_select.json` (static scan). G_confirm2 loads only after a committed, unmodified FREEZE2 record, and only once. T4: R2 never writes under `experiments/c15/` or `results/raw/c15a/` |
| Provenance pins (verified at R2-Z and R2-REPORT) | `g_confirm.json` `3f0911cb006cbffc0b87a59e263ddc7d320f9f33807d12c2bbf185447f03b0dc`; `g_select.json` `740e55b9…61b9c2`; `select_qwen3-1.7b.json` `e4646d8d…5b6a`; `select_qwen3.5-2b.json` `61d88e94…c884`; `select_qwen3-4b.json` `85fd4a3f…3724`; `choose_astage.json` `2effe74b…aa`; `astage_report.md` `b18b5bb9…efaf`; trees `31f6d6e7…` / `54a3f123…` |
| Pre-run commit | All material hashes, the `c15a2` config hash (asserted equal to the frozen JSON), and the code tree hash. **Committed before any model sees G_select2** |

## 6. Function groups and powered failures (amendment 3)

- **Groups:**
  - **A** (report/broadcast) = W1 ∧ W2;
  - **B** (selective causal relevance) = W4 ∧ W5;
  - W3′ and W0b′ are reported; both are required only for P1.
- **Validity for any claim in a cell:**
  - W0a passes for the model;
  - the failing gate's own validity controls pass (W1/W2: ≤ 20% unmatched; W2: ≥ 15 included pairs; W3′: PC1–PC3, NC1–NC2; W4: two-hop accuracy ≥ 0.60; W5: s\* certifiable).
- **Powered failure** (the only kind of failure that may support absence). Validity holds **and** the 95% CI lies entirely outside the pass region **and** the point estimate misses by a margin:

| Gate | Powered failure if (either necessary condition) |
|---|---|
| W1 | hit_J CI upper < 0.30 and point ≤ 0.25; **or** ratio CI upper < 2.0 and point ≤ 1.5 |
| W2 | rate_J CI upper < 0.25 and point ≤ 0.15; **or** ratio CI upper < 2.0 and point ≤ 1.5; with ≥ 15 included pairs |
| W3′ | diff CI upper < 0.15 and point ≤ 0.05 |
| W4 | diff CI upper < 10 pp and point ≤ 5 pp |
| W5 | SI_iso CI lower > 1.0 and point ≥ 1.10 |

**W2 caveat.** With about 20 pairs the W2 CI half-width is about 0.2. A W2 point estimate between roughly 0.05 and 0.25 is therefore an **unpowered miss** and can never support absence.
- Under these rules, the A-stage's Qwen3-4B W2 values of 0.11–0.18 would *not* have counted.

## 7. Outcome logic

### 7.1 Patterns

| Pattern | Criterion on G_select2 (assay-valid cells only) | Confirmation on G_confirm2 (frozen cells; run once) |
|---|---|---|
| **P4 assay failure** | W3′ PC1/PC2/PC3/NC fail in all three models, **or** fewer than 50% of cells are assay-valid, **or** PC4 cannot be satisfied | None. STOP |
| **P1 unified** | ≥ 1 cell passes W0a, W0b′, W1, W2, W3′, W4 and W5. Chosen by the unchanged CHOOSE rule | The chosen cell passes all seven |
| **P2 fragmented (strong)** | **A genuine double dissociation across two models.** Cell X passes A and has a powered failure in B. Cell Y (another model) passes B and has a powered failure in A. Each failing gate's validity controls pass | **Both** X and Y reproduce **both** halves: X passes A with a powered B failure; Y passes B with a powered A failure |
| P2-weak (reported, not fragmentation) | The same pattern within one model, across layers | Reported as "layer-distributed" only |
| **P3 generic leverage** | In some model, at every assay-valid layer: W4 passes; W1, W2 and W3′ each have powered failures; W5 has a powered failure | The representative cell (largest W4 margin) reproduces it |
| **P0 no workspace-like route** | Assays valid; no cell passes A or W3′, and the content-specific failures (W1, W2, W3′) are powered in every model | A representative cell reproduces it |
| IND indeterminate (residual) | Anything else, e.g. complementary passes whose failures are not powered | None. STOP and report, with no claim |

- **Precedence:** P4 → P1 → P2 → P3 → P0 → IND.
- **Complementary failures alone never establish P2.** Both halves of the dissociation must be powered, assay-valid and replicated.

### 7.2 What each verdict licenses (amendments 4–5)

| Confirmed verdict | Licensed statement | Next step |
|---|---|---|
| P1 | "A workspace-like route exists in model m at layer ℓ" | **Report to the PI.** The PI decides whether to authorize the original verification-routing experiment (C15 B/C/H). **Not automatic** |
| P2-strong | "Workspace-like functions dissociate across lens-defined routes in these models" | **A C16 design/novelty memo only** (including the literature audit of §9). No C16 implementation without PI approval |
| P3 | "The J subspace is a high-leverage computational subspace, not a content-broadcasting workspace" | Report; PI decision |
| P0 | "No evidence of workspace-like routing in these models" | Report; PI decision |
| P4 / IND / not confirmed | No architectural claim | Report. A methods note on the A-stage plus R2 is the default proposal |

### 7.3 Pre-registered hypotheses from the A-stage (tested on fresh data only)

- **H-F1:** in Qwen3.5-2B, group A passes at ≥ 1 band layer, and W5 has a powered failure there.
  - Note: A-stage W0a for Qwen3.5-2B was 0.545. If W0a fails again, H-F1 is **not assessable**.
- **H-F2:** in Qwen3-4B, group B passes at layer 11 or 13, and group A has a powered failure there.
- **H-F1 ∧ H-F2, confirmed, = P2-strong.**

## 8. Simulation results (synthetic; no models)

### 8.1 Old W3 (diagnosis)

- Null mean R² is −0.02 / −0.16 / −0.37 for spectra α = 0.5 / 1 / 1.5.
- P(R² ≥ 0.25) is about 0.5 at a transported signal of 10% of context SD, and about 1.0 at 20%.
- **All 30 observed A-stage values (−0.152 to −0.083) lie in the null.**

### 8.2 Final W3′ site criterion (`r2_sim_v2.json`; 8 concepts × 24 contexts; d = 2048; 200 reps per cell)

**Pass probability by alignment φ** (fraction of the injected effect aligned with the conceptual footprint):

| Context heterogeneity γ | φ = 0 | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 1 | **0.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 | 0.00 | 0.00 | 0.91–1.00 | 1.00 | 1.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.53–0.84 |

- The effect-size floor binds as intended. With acc_nat ≈ 1, the floor is 0.5, so a site passes only if injected content is identified at least half as reliably as a real paraphrase mention.
- **Generic influence only:** pass rate 0.00, mean accuracy 0.128 (chance 0.125).

**Lexical-carryover check.** Pure lexical carryover (φ_c = 0, φ_lex = 0.5); ρ is the shared variance between the surface and conceptual footprints:

| ρ | Word-mention reference | Paraphrase reference (frozen) |
|---|---|---|
| 0 (pure surface carryover) | **1.00** (hazard) | **0.00** |
| ≥ 0.1 (semantically processed) | 1.00 | 1.00 |

So the frozen reference excludes surface carryover. Semantically processed content is detected, which is the intended content transport.

### 8.3 Breadth gate (correlated sites, design effect 3)

| Site pass rate J / ⊥ | P(gate passes) |
|---|---|
| 0.05 / 0.05 (false positive) | 0.07 |
| 0.2 / 0.05 | 0.45 |
| 0.4 / 0.1 | 0.75 |
| 0.6 / 0.2 | 0.82 |

The gate detects moderate-to-large J/⊥ breadth differences. Small ones may yield IND rather than a false claim.

### 8.4 Other gates (`r2_sim_c.json`)

- **W0b′:** P(pass) is 0.96–0.97 at a true intermediate/foil rate of 0.55/0.20, and 0.25–0.29 at 0.40/0.25.
- **W1 (n = 100):** P(hit ≥ 0.30) is 0.54 / 0.88 / 0.99 at true rates 0.30 / 0.35 / 0.40.
- **W2 (n = 20):** 0.59 / 0.76 at 0.25 / 0.30. **This is the least powered gate**, hence the §6 rule.
- **W5:** A-stage CI widths were 0.13–0.40.

**Caveat.** The noise models are isotropic; real activations are anisotropic. PC1–PC3 and NC1–NC2 are the empirical safeguards.

## 9. C16 novelty language (amendment 6: corrected)

**C16 novelty is UNRATED.** No novelty rating will be given until a dedicated current-literature audit has been done, as part of the C16 design/novelty memo, which is triggered only by confirmed P2.

**Known adjacent prior work:**
- Goyal et al. (ICLR 2022), shared global workspace among neural modules;
- Chateau-Laurent & VanRullen (2025), workspace routing for chained operations;
- Shang (2026, arXiv 2604.08206), "Theater of Mind" / Global Workspace Agents (an agent-level broadcast hub);
- CTM-AI (2026, arXiv 2605.04097);
- VanRullen & Kanai (2021);
- the 2026 J-space work (Gurnee et al.; jspace-validity; jspace-4b; looped-transformer J-lens).

**The audit must cover:**
- pretrained-model retrofitting;
- latent/internal shared-memory modules (memory tokens, recurrent memory transformers, memory-augmented LMs);
- recurrent and workspace adapters;
- modular routing (mixture-of-experts, router-based modular LMs);
- shared and latent scratchpads (including continuous-thought methods);
- workspace-inspired LLM architectures.

**Candidate novelty claim (narrow, to be audited):** *retrofitting a capacity-limited internal shared workspace into an already-pretrained monolithic LM, and causally testing whether it integrates and generalises previously fragmented functions.*

## 10. Expected CPU cost (same machine; from A-stage timings, with old W3 replaced by W3′ + W3b)

| Stage | 1.7B | 3.5-2B | 4B | Total |
|---|---|---|---|---|
| R2-ENG (PC4 unit tests + SMOKE2) | 0.5 h | 0.5 h | 1.0 h | ≈ 2 h |
| R2-SELECT (all band layers) | ≈ 8.0 h | ≈ 6.4 h | ≈ 15.9 h | **≈ 30 h** |
| R2-CONFIRM (1–2 frozen cells) | | | | ≈ 2–6 h |
| **Total** | | | | **≈ 34–38 h** (range 32–42) |

- W3′ per layer: 8 concepts × 4 arms (J, ⊥, NC1, NC3) × 24 contexts ≈ 38k tokens. References: ≈ 47k tokens once per model. W3b: ≈ 35k tokens per layer.
- Long jobs are launched via WMI. No downloads.

## 11. STOP/GO decision table

| Stage | Condition | Decision |
|---|---|---|
| R2-0 | PI approves this plan | GO: implement `c15a2`, write the materials, tests (T1–T4, PC4, threshold-equality, classification logic); **pre-run commit** |
| R2-0 | Any test fails | STOP until fixed. No model data |
| R2-Z | Pinned artifacts or provenance pins mismatch | Exclude that model (ST-0). If any A-stage pin changed: **STOP and report** |
| R2-ENG | PC4 fails, or SMOKE2 PC3/NC1 fails on a model | Fix the implementation only (no G data). Unfixable → **STOP (P4-engineering)** |
| R2-SELECT | Completed for all three models | GO → CLASSIFY |
| R2-CLASSIFY | P4 | **STOP.** Report; propose the methods note |
| R2-CLASSIFY | IND | **STOP.** Report; no confirmation, no claim |
| R2-CLASSIFY | P1 / P2-strong / P3 / P0 candidate | GO → FREEZE2: freeze the pattern's cells, predicted gate values and all frozen parameters; **commit** |
| R2-CONFIRM | Frozen pattern reproduced on G_confirm2 | Verdict = confirmed pattern → REPORT |
| R2-CONFIRM | Not reproduced | Verdict "not confirmed". **No re-selection, no second confirmation** → REPORT |
| R2-REPORT | P1 confirmed | Report to the PI. **No automatic C15 B/C/H** |
| R2-REPORT | P2-strong confirmed | Write the **C16 design/novelty memo** (with the audit of §9). **No automatic C16** |
| R2-REPORT | P3 / P0 / not confirmed / P4 / IND | Report; PI decision |

**Not authorized under R2:** B/C/H, SAT data, F, rescue routes, C16 implementation, downloads, opening the original G_confirm, any modification of A-stage files.

## 12. D74 addendum: final preregistration details (frozen before any R2 model run)

The PI approved R2 in principle and asked for two details to be resolved and frozen. Both were resolved **on synthetic data only**. The null family, the materiality rule and the adjustment ladder were committed (3d243a9) before the simulation ran.
- Script: `c15r2_power/r2_null_calibration.py`, sha256 `998859ce…d53e`.
- Results: `r2_null_calibration.json`, sha256 `e6d31212…03b2`.
- Machine-readable rules: `c15r2_thresholds_FROZEN.json` (D74 fields).

### 12.1 W0b′: one gate decision from the three positions

**Rule: one pre-declared primary position, with no multiplicity.** The gate is decided at t_d alone (last token of the bridge's descriptive mention). t₁ and t₂ are computed and reported, but have **no role in any gate, pattern or claim**.
- **Why t_d.** It is the measurement position of Yang et al. (2024). It is also the position at which the J-lens reference example reads the bridge: position −2, the end of the bridge descriptor. It is therefore the position the instrument itself is known to read.
- **Gate (all required).** Computed among correctly answered items:
  1. two-hop accuracy ≥ 0.60;
  2. intermediate rate at t_d ≥ 0.40;
  3. paired foil lift at t_d ≥ 0.20, with its one-sided 95% item-bootstrap (2,000) lower bound > 0.
- **Readout.**
  - A hit means the first token of any alias is among the J-lens top-20 at any band layer at t_d. The band-layer union is the unchanged A-stage procedure, and it is applied identically to the 5 foils.
  - lift_i = hit_i − mean over the 5 foils of foil-hit_ij.
- **Rejected alternatives.** An uncorrected "any of three positions passes" rule is rejected. So is an item-level union, whose 0.40 floor would be looser than the single-position floor.
- **Synthetic calibration.** Under the null (bridge no more readable than same-type foils; item heterogeneity; positions correlated or not; n_correct 30–60; base rates 0.10–0.55):

| Rule | Maximum false-positive rate |
|---|---|
| **Primary-position rule (frozen)** | **0.039** |
| Uncorrected any-of-3 (rejected) | 0.096 |

- **Power of the primary rule:**
  - 0.76–0.81 at a true 0.45 vs 0.10 (bridge vs foil);
  - 0.93–0.99 at 0.55 vs 0.20;
  - 0.23–0.29 at 0.40 vs 0.25 (near the floor, as designed).

### 12.2 W3′: null calibration of the complete gate

The complete gate requires all of:
- the controls PC2 (simulated) and PC1, PC3, NC1, NC2 (treated as passing, a conservative upper bound);
- the site criterion (binomial p < 0.05 on 192 trials and acc ≥ max(0.30, 0.5 · acc_nat), with acc_nat re-estimated in every bootstrap replicate);
- the breadth criterion (diff ≥ 0.15 and a 2.5th-percentile bootstrap bound > 0; 1,000 replicates over contexts × concepts).

All of these were simulated exactly as frozen.

**Null family (672 configurations).** J and ⊥ are generated by identical processes, so true breadth is equal. Noise is independent across arms, which is least favourable. The factors crossed:
- site accuracy 0.28–0.92;
- uniform or offset-decaying transport;
- 48 or 132 sites;
- site correlation 0.5 or 0.9;
- arm-specific context- and concept-level heterogeneity σ_m ∈ {0, 0.5, 1.0};
- natural benchmark ≈ 0.55 (floor 0.30) or ≈ 0.98 (floor ≈ 0.49);
- isotropic or anisotropic noise.

Screening used 400 repetitions per configuration. The 15 highest configurations were re-estimated with 10,000 fresh repetitions each.

**Result.** The worst-case false-positive rate of the complete gate at the frozen thresholds (rung R0) is **FPR\* = 0.046** (SE 0.002; 95% MC lower bound 0.042 ≤ 0.05).
- The worst case is offset-decaying transport, 132 sites, σ_m = 1.0 and floor 0.30.
- The mean over the null family is 0.010.

By the pre-declared rule this is **acceptably controlled**, so the frozen W3′ thresholds are **preserved unchanged** (rung R0). No adjustment was made.

**Context.** For information only:
- the 1st-percentile bound would give 0.024;
- the diff floor is never the binding element at the worst case.

**Power (reporting only).** The gate tests a breadth **difference**.
- When ⊥ also clears the site floor everywhere, the gate correctly does not pass: power 0.00–0.12 at a ⊥ accuracy of 0.37–0.59 with floor 0.30.
- When J transports identity where ⊥ does not, power is 0.43–0.86 (e.g. 0.73–0.84 at J 0.80–0.92 vs ⊥ 0.37, floor 0.49).

**Multiplicity across cells.** FPR\* is per cell. The protection against selecting a false cell among the ~15 cells is the single, frozen confirmation on G_confirm2. That protection is unchanged.

### 12.3 Implementation specifications

These are frozen before any data. They resolve terms that the frozen text left implicit, and none of them loosens a gate.

| Item | Specification |
|---|---|
| CIs for powered-failure classification | Two-sided 95% percentile bootstrap, 2,000 replicates, seed 9502 + layer: <br>• W0a over paragraphs; <br>• W0b′ over correct items; <br>• W1 **cluster bootstrap over concepts** (wider than trial-level, so stricter for claiming absence); <br>• W2 over included pairs; <br>• W4 over two-hop items (paired J vs random-scale-1 impairment). <br>W3′ uses its own 1,000-replicate contexts × concepts bootstrap. W5 uses its unchanged ratio CI |
| Group powered failure | A group (A = W1 ∧ W2; B = W4 ∧ W5) has a powered failure iff at least one member gate has a powered failure |
| Assay-valid cell | W0a passes for the model **and** the W3′ controls PC1, PC2, PC3, NC1, NC2 pass **and** W1 and W2 matching coverage is ≥ 80% (unmatched ≤ 20%) |
| Model coverage for P3/P0 | A model counts only if W0a passes and ≥ 50% of its band layers are assay-valid. "At every assay-valid layer" refers to that model's assay-valid layers |
| W3′ matched-control coverage | ⊥ controls are the W1 directions (empirical-KL matched at a\*·ĥ). If ≤ 1 of the 8 W3′ concepts lacks a match, that concept is dropped from both arms (K′ = 7, chance 1/7, binomial vs 1/7). If ≥ 2 lack one, W3′ is not assessable |
| W3′ references and controls | <br>• **References:** F̂_k = centred mean over fold A × {p1, p2}. <br>• **PC1:** median over concepts of cos(F̂_k^{p1}, F̂_k^{p2}) ≥ 0.5 at ≥ 50% of downstream sites. <br>• **PC2:** acc_nat (p3, fold B) ≥ 0.30 and p < 0.05 at ≥ 50% of downstream sites. <br>• **PC3:** offset 0 (the " thing" token) at layer ℓ+2 against the offset-0 references (last token of the paraphrase NP), J or ⊥ acc ≥ 0.30 and p < 0.05. <br>• **NC1:** deterministic norm-matched random unit directions at a\*·ĥ; BB ≤ 0.10. <br>• **NC2:** J effects scored under a fixed derangement (design seed); BB ≤ 0.10 |
| NC3 lexical-equivalence index (reported only) | <br>• LEI = mean over downstream sites of [acc_J vs word references − acc_J vs paraphrase references]. <br>• Also reported: identification of the injected normalised input embedding of w_k (same norm and site) against both kinds of reference |
| W3b (reported only) | Fold-B contexts with " the thing", followed by "\n\nIn the text above, the hidden object was the". 10-way forced choice (concept + 9 distractors from the split's 20 concepts, design seed). Arms none / J / ⊥. Not assessable if natural-word accuracy < 0.80 |
| P1 choice | The unchanged A-stage CHOOSE rule, with W3 replaced by W3′ (margin (diff − 0.15)/0.15) and W0 = W0a ∧ W0b′ |
| P2-strong cell pair | Over cell pairs (X in model m₁ passes A with a powered B failure; Y in model m₂ ≠ m₁ passes B with a powered A failure), maximise min(min normalised margin of A at X, of B at Y). Ties: smaller model for X, then lower layers |
| P3 / P0 representative cell | P3: in the qualifying model, the assay-valid layer with the largest W4 margin. <br>P0: the assay-valid cell with the most negative max(margin_W1, margin_W2, margin_W3′). Ties: smaller model, lower layer |
| SMOKE2 (R2-ENG) | The full R2 pipeline on SMOKE2 material at the middle band layer. PC3 and NC1 must pass on each model. SMOKE2 is never evidence |
| Seeds | Splits 9500; designs 9501; RNG 9502. The reused A-stage code's internal RNG seed is set to 9502 in memory; no A-stage file is changed |

### 12.4 Pre-run implementation notes (frozen in the pre-run commit, before any model sees G_select2)

| Item | As implemented (`experiments/c15r2/`) |
|---|---|
| Package | `c15a2` (separate). The unchanged procedures (S_J, V_gen, GP, matching, dose rule, two-hop pass, ablations) run through the A-stage code itself (`experiments/c15/c15a`, tree pinned), imported read-only, with byte-code writing disabled. W1, W2 and W4/W5 are the A-stage method bodies copied verbatim plus per-unit outcome logging for the CIs. A unit test asserts that their gate outputs equal the A-stage methods' outputs |
| Fresh material | `materials2/` (sources, `tools/make_splits2.py`, built `g_select2.json`, `g_confirm2.json`, `smoke2.json`, `split_manifest2.json`). Per split: <br>• 63 paragraphs; <br>• 20 concepts (10 categories × 2); <br>• 20 W2 countries; <br>• 74 two-hop items, from 37 bridges = 25 country + 6 animal + 6 fictional-character bridges; <br>• the W3′ block (8 concepts from 8 distinct categories; 48 slot contexts, folds 24/24; 3 paraphrases each). <br>Countries are split by country, so intermediates never cross splits. Foils are 5 same-type entities from the split's own pool. Dose-rule directions come from the first 4 W3′ concepts, as in the A-stage, where the dose concepts were the W3 concepts |
| Build checks (tokenizer only) | <br>• lengths; <br>• single tokens; <br>• paraphrase word and token exclusion; <br>• context tokenization identical across slot versions; <br>• the W3b suffix leaves every W3′ site token unchanged; <br>• unique, prefix-stable two-hop spans with t₁ ≤ t_d < t₂; <br>• answer and foil first tokens distinct from the intermediate's; <br>• T2 entity containment and shared 8-word sequences vs the A-stage source files; <br>• T3; <br>• vocabulary audit. <br>Two-hop prompts are exempt only from the 8-gram check, because they share the unchanged query frames by design; they are entity-checked |
| P4 | PC4 failed, **or** W3′ is assessable at no layer of any model, **or** fewer than 50% of all cells are assay-valid |
| P0 | Requires **every** candidate model to count (W0a passes and ≥ 50% of layers assay-valid). If any model is excluded or not assessable, P0 cannot be declared (residual: IND) |
| P2 | X and Y must both be assay-valid cells. P2-weak (same model) is reported as a flag only |
| P3 | Only qualifying models can show P3. If several do, the representative is the cell with the largest W4 margin (ties: smaller model) |
| Confirmation | Every frozen cell runs once (`CONFIRM --model --layer`). The guard refuses a second run of a cell, any cell not in FREEZE2, a changed config or thresholds hash, or a FREEZE2 that is not committed. Verdict: the pattern is confirmed only if every frozen cell reproduces its role |
| Execution order | R2-Z (all models) → SMOKE2 (all models; PC3 and NC1 must pass on each) → SELECT (all models). If SMOKE2 fails on any model, no SELECT runs until the implementation is fixed and recommitted |
