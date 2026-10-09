# C15-R2 WORKSPACE-ASSAY REPAIR MEMO (design only): fragmentation or assay failure?

| Field | Value |
|---|---|
| Date | 2026-10-07 |
| Responds to | PI acceptance of the ST-2 stop and request for a design-only repair memo |
| Amended | 2026-10-07 (D73): superseded where they differ by `c15r2_preregistration_FROZEN.md`. C16 novelty is now UNRATED pending an audit; P1/P2 no longer advance automatically |
| Status | **Design only.** No model run, no change to `experiments/c15/`, no access to G_confirm, no SAT data, no downloads |
| Preserved unchanged | The A-stage record (D70/D71, `astage_report.md`, all `select_*.json`): the negative result and the SI_iso/SI_norm finding. G_select is not reinterpreted. **G_confirm stays sealed** for provenance |
| New evidence used here | Only synthetic power simulations (`memo/c15r2_power/`, no models) and published literature |
| Claims discipline | Functional organisation only. No outcome bears on phenomenal consciousness |

---

## 0. Bottom line

1. **W3 should be treated as non-informative (an assay failure), not as evidence against transport.**
   - Every observed W3 value (mean cross-fitted R² −0.15 to −0.08 in both routes, all 15 layers) lies inside the *null* distribution of that estimator (simulated null mean −0.16 for a 1/i spectrum).
   - W3 was sensitive only to a context-invariant linear signal of roughly ≥ 10% of the context SD, and it had no positive control.
   - Meanwhile, W1 and W2 show behaviourally that injected J content reaches later positions.
   - **Replacement W3′:** a paired, interventional "footprint" test. It measures whether injected content reproduces the downstream signature of *actually mentioning* the concept. It is free of context variance by construction, anchored to a natural-mention benchmark, and has ≥ 0.95 simulated power at small effects.
2. **W0b's single readout position was not justified by the literature it was meant to follow.**
   - The two-hop literature locates bridge-entity resolution at the end of the first-hop span (Biran et al., EMNLP 2024) or of the bridge's descriptive mention (Yang et al., ACL 2024), not at the final prompt token.
   - The jlens reference example also reads at the descriptor end.
   - **W0b′:** reads three pre-annotated structural positions, keeps the 0.40 threshold, and **adds** a foil-controlled specificity requirement. That makes it stricter, not looser.
3. **W1, W2, W4 and W5 are unchanged.** Failures caused by an unconstructible control are reported as "not assessable" (pass/fail is unchanged), and confidence intervals are added for reporting.
4. **Fresh material.** All new material goes into G_select2 / G_confirm2, with no entity overlap with any A-stage material. Fragmentation can only be claimed if a **frozen double-dissociation pattern** replicates on G_confirm2.
5. **If fragmentation is confirmed,** the candidate next project is **constructing an explicit, validated shared workspace** that integrates the fragmented functions, *not* searching larger pretrained models (§10). *Amended (D73):* confirmed P2 triggers only a C16 design/novelty memo, and C16 novelty is UNRATED pending a dedicated literature audit.
6. **Recommendation (§11):** approve C15-R2 implementation on the three already-downloaded models. That means about 30–38 CPU-hours, no new downloads, and a decision point after confirmation.

---

## 1. W3 diagnosis: why R² ≤ 0 in both routes

### 1.1 What W3 measured

- At each downstream site (position p+1..p+20, layer ℓ+2..L−2), W3 decoded the injected scalar s (s ∈ {±0.5, ±1}, one value per context, 40 contexts).
- The decoder was a cross-fitted dual ridge regression, fitted **across contexts** on raw residual states (d = 2048–2560, n = 40).
- The signal therefore had to stand out against the full between-context variance of the residual stream.

### 1.2 The observed values are the estimator's null distribution

Synthetic simulation (`c15r2_power/r2_sim_a.json`; Gaussian context variance with spectrum λ_i ∝ i^−α, n = 40, d = 2048, identical estimator):

| Transported signal β (norm / total context SD) | mean R², α = 1.0 | P(R² ≥ 0.25), α = 0.5 / 1.0 / 1.5 |
|---|---|---|
| 0 (no signal) | **−0.155** (α = 0.5: −0.02; α = 1.5: −0.37) | 0.00 / 0.01 / 0.00 |
| 0.05 | −0.03 | 0.00 / 0.02 / 0.07 |
| 0.10 | 0.25 | 0.67 / 0.48 / 0.95 |
| 0.20 | 0.69 | 1.0 / 1.0 / 1.0 |

**Observed A-stage values:** mean R² from −0.152 to −0.083 in *both* routes, at all 15 model-layers, over 40–110 sites each. These are exactly the values expected under **no** linearly decodable, context-invariant transport.

### 1.3 Three explanations, not separable post hoc

Behavioural transport demonstrably exists:
- In Qwen3.5-2B, W1 injections at position 23 change a report about 12 tokens later (hit rate 0.37–0.64 vs 0.11–0.15 without injection).
- W2 swaps change answers several tokens downstream.

So W3's null must come from one or more of the following:

| Cause | Mechanism | Why W3 could not see it |
|---|---|---|
| **(a) Small transported magnitude** | Later positions read position p through attention, diluted by sink and local heads; the downstream image of the injection is small relative to between-context variance | Unpaired design: the signal must exceed about 10% of total context SD |
| **(b) Context-dependent transport** | Which heads read p, and how strongly, varies with context, so the downstream image has no single direction across contexts | A single linear decoder across contexts averages heterogeneous images toward zero |
| **(c) Undetected implementation or protocol defect** | Possible in principle | **W3 had no positive control** (e.g. decoding at the injection site itself), so a defect cannot be excluded after the fact |

**Conclusion.** The A-stage W3 result is classified as **assay non-informative**. It is evidence neither for nor against cross-position transport in either route. This classification follows from the estimator's null behaviour and the missing positive control, which are properties of the assay, not of any desired outcome.

## 2. W3′: paired interventional footprint test (replacement)

### 2.1 Causal estimand

> Does injecting concept k's route-R component at a neutral slot reproduce, at later positions, the **content-specific downstream footprint** that actually mentioning concept k produces?

This is an interventional comparison within each context, so between-context variance cancels exactly.

### 2.2 Construction (per model, band layer ℓ, split)

**Material.** 48 author-written *slot contexts* per split:
- a generic prefix of ≥ 20 tokens;
- the slot phrase " the {w}", grammatical for any concrete noun;
- a generic continuation of ≥ 24 tokens that does not reveal the noun.
- The baseline word is " thing", which is single-token in all three tokenizers (verified at authoring time).

**Concepts.** 8 per split, single-token nouns drawn from the split's 20 W1 concepts. This doubles the A-stage's 4, for power.

**Sites.** Relative offsets +1, +3, …, +23 after the slot, crossed with layers ℓ+2, ℓ+4, …, L−2. Every site lies **downstream** of the slot.

**Natural footprint (cross-fitted).** Split contexts into folds A and B (24 + 24).

> F̂_k^A(site) = mean over c ∈ A of [ h(c, w_k) − h(c, " thing") ]

and likewise F̂_k^B. Here h(c, w) is the block-ℓ′ output at the site, in context c, with word w in the slot.

**Injection effect.** In the baseline-word version of context c ∈ B:

> D_ck^R(site) = h(c, " thing" + a\*·ĥ·u_k^R at the slot, layer ℓ) − h(c, " thing")

- u_k^J = P_J c_k / ‖P_J c_k‖, with c_k the W1 concept vector (unchanged).
- u_k^⊥ = the unchanged matched control: empirical-KL, lens-gain and propagation matched.
- a\* comes from the unchanged dose rule.
- Roles of folds A and B are swapped and the results averaged.

**Specificity score.**

> x_ck^R(site) = cos(D_ck^R, F̂_k) − mean_{k′≠k} cos(D_ck^R, F̂_k′)

**Natural-mention benchmark.**

> x_ck^nat(site) = cos(N_ck, F̂_k^{other fold}) − mean_{k′≠k} cos(N_ck, F̂_k′^{other fold})

where N_ck = h(c, w_k) − h(c, " thing") is the actual mention, in held-out contexts.

**Site criterion.** A site counts as *content-transporting* for route R if **both**:
1. the mean over concepts of x^R is > 0 by a one-sided t-test across the 8 concept means (df = 7, α = 0.05); and
2. mean x^R ≥ 0.5 × mean x^nat. That is, the injection reproduces at least half the content-specificity of a real mention.

Element 2 anchors the criterion to an interpretable natural benchmark. Without it, the very high sensitivity of the paired design (§8) would count trivially small alignments and saturate both routes.

**Statistic.** Breadth BB′_R = the fraction of downstream sites that are content-transporting.

**Gate (threshold form unchanged from W3):** BB′_J − BB′_⊥ ≥ 0.15, with the 95% bootstrap CI lower bound > 0. The bootstrap resamples contexts within folds and concepts.

### 2.3 Mandatory validity controls (assay failure, not route failure, if any fails)

| Control | Requirement | Failure means |
|---|---|---|
| PC1 footprint reliability | Median over concepts of cos(F̂_k^A, F̂_k^B) ≥ 0.5 at ≥ 50% of sites | Natural footprints are not estimable: W3′ not assessable |
| PC2 natural content-specificity | Mean x^nat > 0 (one-sided t, df 7, α = 0.05) at ≥ 50% of sites | No content-specific natural transport to compare against: not assessable |
| PC3 injection-site content | At the slot position itself (layer ℓ+2), x^R > 0 significantly for at least one route | The injections carry no concept content: not assessable |
| NC random direction | A norm-matched random-direction injection gives BB′ ≤ 0.10 | False-positive control. If exceeded, the site criterion is miscalibrated: not assessable |
| ENG planted transport | Unit test: in a tiny model with a planted copy-head, a planted content direction must yield BB′ ≥ 0.8, and a null direction BB′ ≤ 0.1. **SMOKE2 run on each real model:** PC3 must pass | Implementation defect: fix before any G data (the A-stage W3 had no such check) |

### 2.4 Behavioural companion W3b (secondary, reported, not gating)

- Same slot contexts with " thing" plus the injection.
- Append "\n\nIn the text above, the hidden object was the" and take a 10-way forced choice among the split's concepts.
- Arms: J, ⊥, none.
- Natural-mention positive control: with the real word in the slot, accuracy must be ≥ 0.80. Otherwise W3b is not assessable.
- Reported as hit rates with CIs. It is not added as a gate, to avoid stacking new pass criteria.

### 2.5 Why this is not loosening

- The breadth threshold (0.15, CI > 0) is unchanged.
- The site criterion changes from "context-invariant linear decodability R² ≥ 0.25" to "content-specific interventional transport ≥ half a real mention". The new criterion is a valid measure of the construct; the diagnosis (§1) shows the old one was not.
- Four validity controls are added that the old W3 lacked.
- Nothing in W3′ was tuned on A-stage outcomes: no A-stage per-site data exist, and G_select is not reused.

## 3. W0b′: intermediate-entity readout at justified positions

### 3.1 Diagnosis

- The A-stage read the bridge entity only at the **final prompt token** (rates 0.02 / 0.04 / 0.17).
- The literature measures it elsewhere:
  - **Yang et al. 2024** (ACL; arXiv 2402.16837) measure internal entity recall at "the last token of the bridge entity's descriptive mention".
  - **Biran et al. 2024** (EMNLP; arXiv 2406.12775) find the bridge entity resolved "in early layers" at the last token of the first-hop entity (t₁), then propagated to the last prompt token (t₂), where the second hop happens "in the later layers".
  - The **jlens reference example** reads at position −2, which is the end of the bridge descriptor (" …shaped like a boot | is").
- The final position is where the *answer* is formed; by then the intermediate may already have been transformed into it.

### 3.2 Pre-registered readout set

Positions are fixed by item structure, annotated in the material at authoring time, before any model run:

| Position | Definition (character span annotated per item; last token by tokenizer rule) | Source |
|---|---|---|
| t₁ | Last token of the first-hop source entity mention (e.g. " Tower" in "the Eiffel Tower") | Biran et al. 2024 |
| t_d | Last token of the bridge's descriptive mention (e.g. " stands" in "the country where the Eiffel Tower stands") | Yang et al. 2024; jlens example |
| t₂ | Final prompt token (the A-stage position) | Retained for continuity |

- **Layers:** the unchanged band layers [0.3 L, 0.6 L] (step 2). I do **not** widen the layer window, even though Biran et al. place first-hop resolution in early layers, so that W0b′ is not loosened.
- **Readout:** J-lens top-20, unchanged.
- **Hit:** any alias first token appears at any (position, layer) in the set.

### 3.3 Foil control (added stringency, for multiplicity)

- Each item has 5 foils: same-type entities not mentioned in the item and not its answer or intermediate (e.g. other countries), pre-assigned at authoring time.
- The foil hit rate is computed with the identical procedure.

**Gate W0b′ (all required):**
1. two-hop accuracy ≥ 0.60 (unchanged);
2. intermediate hit rate among correct items ≥ 0.40 (unchanged threshold);
3. paired lift (intermediate − mean foil) ≥ 0.20, with the one-sided 95% lower bound > 0.

Three positions instead of one raise the chance of spurious hits. Element 3 controls for that. It is stricter than the A-stage, not looser.

**W0a (lens agreement ≥ 0.60) is unchanged.** Qwen3.5-2B's W0a failure (0.545) therefore remains a likely blocker for it, and that is left as data.

## 4. Unchanged components

| Component | Status | Note |
|---|---|---|
| S_J construction, V_gen, GP (per-model screening), band layers | Unchanged | Recomputed on G_select2 by the frozen procedure |
| Dose rule; matched-⊥ construction (empirical KL ±15%, lens gain, propagation) | Unchanged | |
| W1 reportability (hit_J ≥ 0.30, ratio ≥ 2) | Unchanged | The no-injection baseline (0.11–0.20) exceeds nominal chance. This does not threaten the J-vs-⊥ comparison, so it is no validity defect. A baseline-corrected rate is added as a **reported** secondary |
| W2 broadcast (rate_J ≥ 0.25, ratio ≥ 2; natural-magnitude swap) | Unchanged | |
| W4 causal relevance (diff ≥ 10 pp), W5 selectivity (SI_iso ≤ 1, CI upper ≤ 1.25; SI_norm reported with mandated wording) | Unchanged | |
| Matching coverage | Unchanged rule | If > 20% of contents lack a matched control, the test is recorded as **not assessable** (still "not passed" for the label) instead of "fail". This affects classification in §6 only |
| CIs for W1/W2/W4 | Added (reporting) | Bootstrap over trials/pairs/items. Needed to label failures "powered" in §6. Gates unchanged |

## 5. Fresh material: G_select2 and G_confirm2 (original G_confirm stays sealed)

- **Authoring.** New, author-written, task-independent material, with **no item and no entity overlap** with any A-stage material (G_select, G_confirm, SMOKE).
  - The prohibition on SAT, validity and verification vocabulary is retained (unit-tested).
  - Material is written and hashed before any model run.
- **Splits.** Seed 9500; designs seed 9501. G_confirm2 is sealed by the existing guard mechanism (FREEZE2 record committed; run once).

| Item type | Per split | Notes |
|---|---|---|
| Generic paragraphs | 63 | New topics; ≥ 64 tokens in all three tokenizers |
| Concepts | 20 (8 of them for W3′) | New single-token nouns |
| Concept templates | 8 | New |
| Countries | 20 | New countries absent from all A-stage material, e.g. Nepal, Senegal, Uruguay, Croatia, Mongolia… Same W2 procedure and eligibility rule |
| Country templates / queries | 4 / 3 | Templates new; query frames unchanged (the measurement instrument) |
| Two-hop items | 74 (37 bridges) | New bridges, each annotated with spans for e₁ and the descriptive mention, plus 5 pre-assigned foils |
| W3′ slot contexts | 48 | §2.2 |
| SMOKE2 | small | Engineering only, disjoint |

- **Why not reuse G_confirm.** It has never been opened. But it shares authoring lineage (templates, entity style) with the material on which the A-stage was selected, and preserving it sealed keeps the A-stage's confirmation option intact as provenance. There is no strong methodological reason to spend it.

## 6. Pre-declared outcome patterns

### 6.1 Definitions

- **Cell** = a (model, band layer) pair.
- **Gate vector** = (W0′, W1, W2, W3′, W4, W5), each pass / fail / not-assessable.
- **Powered failure** = a fail whose 95% CI excludes the pass threshold (e.g. CI upper of hit_J < 0.30; CI lower of SI_iso > 1.0; CI upper of the W4 difference < 10 pp).
- **Assay-valid cell** = all applicable validity controls pass (W3′ PC1–PC3 and NC; W0b′ foil discrimination; W3b natural-mention accuracy; matching coverage ≥ 80%).
- **Function groups:**
  - **A** = report and broadcast: W1 ∧ W2 (∧ W3′ where assessable);
  - **B** = selective causal relevance: W4 ∧ W5.

### 6.2 Patterns

| Pattern | Criterion on G_select2 | Required on G_confirm2 (run once) | Licensed statement |
|---|---|---|---|
| **P1 unified workspace-like route** | ≥ 1 assay-valid cell passes all six gates | The chosen cell (the A-stage CHOOSE rule, unchanged) passes all six | "A workspace-like route exists in model m at layer ℓ." C15 B/C/H could be re-proposed to the PI |
| **P2 fragmented workspace-like functions** | No cell passes all six, **and** a double dissociation exists among assay-valid cells: one cell passes A with a powered failure of B, another passes B with a powered failure of A. **P2-strong:** the two cells are in different models. **P2-weak:** same model, different layers (reported as "layer-distributed", not fragmentation) | Both frozen cells reproduce their pattern (A-pass/B-fail and B-pass/A-fail) | "Workspace-like functions dissociate across routes. No single lens-defined route carries them jointly" |
| **P3 generic high-leverage routing** | In some model, at every assay-valid layer: W4 passes; W1, W2 and W3′ fail with powered failures; and W5 fails (powered) | The same pattern in the frozen cell | "The J subspace is a high-leverage computational subspace, not a content-broadcasting workspace" |
| **P4 assay failure** | The validity controls for W3′ or W0b′ fail in all three models, or < 50% of cells are assay-valid | — | No architectural inference; the workspace-assay line stops |
| **P0 no workspace-like route** | Assays valid; content-specific groups (A) fail with powered failures everywhere; B mixed | Confirm one representative cell | "No evidence of workspace-like routing in these models" |

### 6.3 Exploratory A-stage hypotheses (H-F)

These are pre-registered for testing on **fresh** data only.
- **H-F1:** Qwen3.5-2B passes A at ≥ 1 band layer, and W5 fails (powered) at all its layers.
- **H-F2:** Qwen3-4B passes B at layer 11 or 13, and A fails (powered) at all its layers.

**H-F1 ∧ H-F2 on both G_select2 and G_confirm2 = P2-strong.** The A-stage numbers themselves are never counted as evidence.

## 7. Proposed pre-registration (stages and stops)

| Stage | Content | Stop |
|---|---|---|
| R2-0 | PI approval of this memo. Implementation as a **new versioned package** (`c15a2`), leaving `c15a` and all A-stage results untouched. Unit tests, including the planted-transport W3′ test, the foil logic, span-to-token annotation and outcome classification. Pre-run commit | — |
| R2-Z | Re-verify the pinned artifacts (no downloads); Z0 recheck | ST-0 |
| R2-ENG | SMOKE2 only: PC3 and NC on each model; timing | Fix the implementation if PC3/NC fail on SMOKE2. Never touches G data |
| R2-SELECT | All three models, all band layers, on G_select2. Gates W0′–W5 plus validity controls | — |
| R2-CLASSIFY | Apply §6 mechanically → P1 / P2 / P3 / P4 / P0 candidate | P4 → STOP (methods note) |
| R2-FREEZE | P1: freeze the chosen cell, as in the A-stage. P2/P3/P0: freeze the pre-declared representative cells and their predicted gate patterns. Commit FREEZE2 | — |
| R2-CONFIRM | Run W0′–W5 once on G_confirm2 for the frozen cells only | Mismatch → report "not confirmed"; no re-selection |
| R2-REPORT | Report to the PI with pattern verdict and tables | Stop; await a PI decision |

**Not authorized under R2:** B/C/H, SAT data, F, rescue routes, new downloads, opening G_confirm, any change to A-stage files.

## 8. Power and sensitivity (synthetic; `memo/c15r2_power/`)

**Old W3** (§1.2). It had no power below β ≈ 0.05 and about 0.5 power at β = 0.10 (1/i spectrum). Its null mean R² ranges from −0.02 to −0.37 depending on spectrum, so the observed values are uninformative.

**W3′ significance component.**
- Simulated with paired injections (40 contexts, 8 or 16 concepts, d = 2048). Footprint estimates come from 40 contexts with noise κ; context heterogeneity is γ × the consistent effect; φ = the fraction of the injected effect aligned with the natural footprint; ρ = the component shared across concepts.

| | φ = 0 (false-positive rate) | φ = 0.05 | φ = 0.10 | φ ≥ 0.20 |
|---|---|---|---|---|
| 8 concepts, ρ = 0, κ = 1, γ = 1 / 3 / 10 | 0.08 / 0.04 / 0.07 | 1.00 / 1.00 / 0.95 | 1.00 | 1.00 |
| 8 concepts, ρ = 0.5, κ = 5, γ = 10 (hardest) | 0.05 | 0.33 | 0.87 | 1.00 |
| 16 concepts, ρ = 0.5, κ = 5, γ = 10 | 0.07 | 0.66 | 0.99 | 1.00 |

- **Consequence.** Significance is essentially guaranteed for any real alignment ≥ 10%. The binding element is therefore the pre-declared effect-size criterion (≥ 0.5 × the natural mention), which is the intended design.
- False-positive rates of 0.03–0.09 against a nominal 0.05 are within Monte-Carlo error (150 reps). The NC control guards this empirically.
- **Caveat.** The noise is isotropic; real residual noise is anisotropic. PC1/PC2 and NC are the empirical safeguards.

**W0b′.** For n ≈ 40–50 correct items:
- P(rate ≥ 0.40) is 0.92–0.94 at a true rate of 0.50, and 0.55 at 0.40 (the threshold is a coin flip at its own value).
- Joint pass probability of the gate (rate ≥ 0.40 ∧ lift ≥ 0.20 ∧ lower bound > 0):
  - true intermediate 0.55 vs foil 0.20 → 0.96–0.97;
  - 0.45 vs 0.10 → 0.78–0.81;
  - 0.40 vs 0.25 → 0.25–0.29.
- So the gate passes when the intermediate is readable and item-specific, and fails near-threshold or non-specific profiles.

**W1 (n ≈ 100 trials).** P(hit ≥ 0.30) is 0.54 at a true rate of 0.30, 0.88 at 0.35, 0.99 at 0.40. The gate is well powered except at its threshold.

**W2 (n ≈ 20 pairs).** P(rate ≥ 0.25) is 0.59 at a true rate of 0.25 and 0.76 at 0.30. **This is the least precise gate.**
- Because W2 is unchanged by instruction, its decisions within about ±0.1 of the threshold should be read as *unpowered*.
- The §6 "powered failure" rule handles this: a W2 fail is not counted toward a dissociation unless its CI excludes the threshold.

**W5.** A-stage CI widths for SI_iso were 0.13–0.40 (median ≈ 0.2). Pass/fail is decisive whenever |SI_iso − 1| ≳ 0.15. The A-stage borderline case (Qwen3-4B layer 15, 1.04 [0.97, 1.11]) shows the expected unpowered zone.

## 9. Compute (same machine; estimates from A-stage timings)

| Stage | Estimate |
|---|---|
| R2-ENG (SMOKE2, 3 models) | ≈ 1.5 h |
| R2-SELECT | ≈ 26–32 h. As in the A-stage (6.6 / 5.6 / 11.0 h), with W3 replaced by W3′: footprints are computed once per model (≈ 40k tokens), injections ≈ 20k tokens per layer, plus W3b ≈ 35k tokens per layer |
| R2-CONFIRM (2–4 frozen cells) | ≈ 4–8 h |
| Total | **≈ 32–42 CPU-hours.** No downloads. Long jobs launched via WMI (A-stage lesson) |

## 10. If fragmentation (P2-strong) is confirmed: novelty, significance and the next project

**The finding itself.**
- A cross-model double dissociation of workspace-like functions in lens-defined routes: report/broadcast in one model, selective causal relevance in another, and neither unified.
- To my knowledge this is not in the literature:
  - Gurnee et al. study frontier models;
  - the jspace-validity and jspace-4b studies test ablation selectivity only;
  - the looped-transformer study reports co-occurrence, not co-localisation.
- **Novelty: MEDIUM-HIGH (provisional; to be re-audited at report time). Significance: MEDIUM.** It constrains the generalisation "verbalizable representations form a unified workspace" to scale or training regime. It is still a descriptive (Level-1) result.

**Option (i): search larger pretrained models** (Qwen3.5-4B/9B, Gemma-3-4b-it, …).

| | Assessment |
|---|---|
| Expected information | When does unification emerge? Scale trends in published selectivity (SI_norm 3.38 → 1.16 from 1.7B to 32B) suggest gradual change |
| Feasibility | ≥ 9B is infeasible in fp32 on 31 GB RAM; 4B already took 11 h per assay pass |
| Novelty / scoop risk | High scoop risk (active J-lens community, larger GPU studies) |
| Significance | Descriptive. It would answer "where", not "what makes a workspace work" |

**Option (ii): construct an explicit shared workspace that integrates the fragmented functions (proposed C16).**

| | Assessment |
|---|---|
| Idea | Retrofit a small, capacity-limited **workspace module** into a pretrained model that showed fragmentation (Qwen3.5-2B or Qwen3-4B). For example: a learned rank-r bottleneck written at a mid layer by a gated, competitive write, and broadcast (read) at all later positions and layers. Train it on generic text plus multi-function auxiliary objectives, with the base model frozen |
| Decisive tests | (1) Does the **unchanged** W0′–W5 battery validate the module's subspace as a *unified* route (P1 profile) where the native routes were fragmented? (2) Does it generalise to **held-out functions** (consumers never seen in training), i.e. broadcast rather than task-specific wiring? (3) Then the original C15 question: does routing a verification signal through the constructed workspace make it flexibly usable, where native routes do not? |
| Prior art | Goyal et al., ICLR 2022 (shared global workspace among neural modules; not in pretrained LMs; no workspace assay; no metacognition). Chateau-Laurent & VanRullen 2025 (global-workspace routing for chained operations in a toy architecture). VanRullen & Kanai 2021 (design principles). Gurnee et al. 2026 (implicit J-space) |
| Novelty | **UNRATED (amended D73).** Adjacent prior work also includes Shang 2026 "Theater of Mind" / Global Workspace Agents (arXiv 2604.08206), CTM-AI (arXiv 2605.04097) and the 2026 J-space work. A dedicated audit (retrofitting, internal shared-memory modules, recurrent/workspace adapters, modular routing, shared scratchpads, workspace-inspired LLM architectures) must precede any rating. The candidate claim is narrow: retrofitting a capacity-limited internal shared workspace into an already-pretrained monolithic LM and causally testing integration/generalisation of previously fragmented functions |
| Significance | **HIGH if positive.** It directly tests the construction principle the PI cares about: integrating fragmented functions into one broadcast medium yields flexible use. Ladder level 4–5 (constructive/developmental) |
| Feasibility ($0, CPU) | Adapter-scale training on a frozen 2B model: forward ≈ 60 tok/s, training ≈ 15–20 tok/s → 1M tokens ≈ 15–20 h. Feasible but slow. The data are generated locally |
| Main risks | (a) The module learns task-specific shortcuts → the held-out-function test is decisive. (b) The capacity limit is too small or too large → a pre-declared rank ladder. (c) The battery must be applied to an engineered subspace without circularity → train/assess on disjoint materials and functions |

**Verdict.** If P2-strong is confirmed, **option (ii) is the stronger next project**: higher significance, lower scoop risk, CPU-feasible, and it addresses mechanism rather than location. Option (i) adds mainly descriptive scale information at high compute cost.
- Under P3 (generic leverage) or P0, option (ii) is still the most direct route to the construction principle, but would start without a native "fragment" baseline to integrate.
- Under P1, report to the PI, who decides on C15 B/C/H. *Amended (D73):* there is no automatic advance.

## 11. Recommendation

1. **Approve C15-R2 as specified** (W3′ and W0b′ repaired; W1/W2/W4/W5 unchanged; fresh G_select2/G_confirm2; original G_confirm sealed; outcome patterns §6; stages §7).
   - The three already-downloaded models only; about 32–42 CPU-hours.
   - Implementation as a new versioned package, with the planted-transport validation required before any G data.
2. **Decision rule after R2-CONFIRM:**
   - P1 → report to the PI, who decides on C15 B/C/H (not automatic; D73);
   - P2-strong → write a C16 design/novelty memo with a literature audit (not automatic C16; D73), not larger-model search;
   - P3 / P0 → drop the workspace framing for native routes; C16 remains the candidate constructive project;
   - P4 → stop the workspace-assay line and write the methods/negative note (A-stage plus R2).
3. **Until then:** no further downloads. The A-stage record and the SI_iso/SI_norm finding stand as reported.
