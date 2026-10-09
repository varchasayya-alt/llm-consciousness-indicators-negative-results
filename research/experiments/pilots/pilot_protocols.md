# Pilot protocols (DRAFT preregistrations, not yet frozen)

> **2026-10-01 (v0.2) status.**
> - **Pilot A is SUPERSEDED.** P1-original was rejected after the closest-competitor review (memo §10.2), and 7–8B models are not $0-feasible on the available hardware. Its lesion and control definitions are reused in P1\* (`memo/decision_document_v2.md`).
> - Pilots B and D are deferred or parked.
> - Pilot C is optional.
> - The next preregistration to be written is **P5\* Stage 1**.
> - Kept for the record. Do not run.

Status: **draft v0.1, 2026-10-01.** Nothing has been run.

Freezing procedure: before any data collection for a pilot:

1. Copy its section to `pilot_X_prereg_frozen.md`.
2. Record the file's SHA-256 in `logs/decisions.md`.
3. Do not edit the frozen file afterwards. Deviations go in `logs/experiment_log.md` with reasons.

Common standards (all pilots):

- Raw generations, logits and probe outputs are saved as JSONL in `results/raw/<pilot>/`, with a config dump (model revision hash, prompts, seeds, library versions).
- Analysis code reads only from `results/raw/`; processed tables go to `results/processed/`.
- Uncertainty: item-level bootstrap (10,000 resamples) 95% CIs; mixed-effects models with random intercepts for item template and entity type.
- Holm correction within each pilot's pre-specified confirmatory family. Everything else is labelled exploratory.
- Every condition is reported, including failed lesions.

---

## Pilot A: lesion-tracking (P1 core)

### A.1 Question

Does a language model's *prospective* feeling-of-knowing (FOK) about a factual question change when its ability to answer *that* question is selectively impaired, with the input held fixed?

### A.2 Models

- Primary: `Qwen2.5-7B-Instruct`.
- Replication: `Llama-3.1-8B-Instruct`.
- Debug only (CPU, local): `Qwen2.5-0.5B-Instruct`.
- Exact HF revisions will be recorded at freeze time.

### A.3 Items

- ~2,000 entity–attribute questions generated from Wikidata. Relations: birthplace, birth year, country of, director of, author of, capital, team.
  - Templated, 3 paraphrase templates per relation.
  - Popularity (Wikipedia page views) recorded to span familiarity.
- ~200 fictitious entities (name-like strings checked absent from Wikidata). These give the unknown-entity FOK baseline.
- Answer scoring: normalised exact match plus alias list. A secondary continuous score is log p(gold answer tokens | prompt).

### A.4 Readouts (dependent variables)

| ID | Readout | Timing | Definition |
|---|---|---|---|
| R1 | Verbal prospective FOK | Before any answer is generated | Prompt: question + "Before answering: do you know the answer? Reply Yes or No." FOK = logit(Yes) − logit(No) at the next token. A second template with the order reversed ("No or Yes") is averaged in to cancel position bias |
| R2 | Retrospective P(True) | After the greedy answer | "Is the proposed answer correct? Yes/No" [kadavath2022know] |
| R3 | Answer confidence | During the answer | Mean token log-prob of the greedy answer |
| R4 | Frozen correctness probe | Residual stream at the last prompt token, best layer chosen on a held-out split *of the base model only* | Logistic probe, frozen before any lesion |
| R5 | Familiarity signal | Subject's last token, mid layers | Projection on a difference-of-means direction (known vs fictitious entities), computed on the base model and frozen |
| R6 | Abstention | Generation with an explicit "If you don't know, say 'I don't know'" | Rate of "I don't know" |

Accuracy is measured by greedy generation without the abstention instruction, plus R3/log p(gold).

### A.5 Lesions (independent variables). Input tokens are identical in all conditions.

| ID | Lesion | Target | Strengths |
|---|---|---|---|
| L0 | None (base) | — | — |
| L2 | **Attention knock-out of attribute extraction.** Block attention from *all positions after the subject span* to subject-token positions, in an upper layer band. Applied identically in the answer, FOK, P(True) and abstention prompts | Recall/extraction stage [geva2023dissecting] | Layer-band width ∈ {2, 4, 6, 8, 10} layers, starting above the layer where FOK-familiarity at the last token has saturated (determined from R5 traces on held-out items) |
| L4 | **Familiarity lesion.** Project out the R5 direction at subject positions in mid layers | Familiarity pathway [ferrando2025entity] | 1 strength (full projection) |
| L5a | Sham: knock-out of attention to *non-subject* tokens of matched count, same layers as L2 | Controls for generic disruption | Matched to each L2 strength |
| L5b | Sham: matched-norm random-direction perturbation at the subject positions | Controls for generic anomaly | Matched to the L4 norm change |
| L6 | 4-bit weight quantisation (bitsandbytes NF4) | Global degradation, for comparison with [cohen2026source] | 1 |

Caveat noted ex ante: L2's knock-out may partly affect FOK computation if FOK reads subject information late. The layer-band start is chosen to minimise this, and R5 at the last position is reported for every condition as a manipulation check.

### A.6 Primary analysis (confirmatory)

Let *E_lesion* be the items that are base-correct and become incorrect under L2 at a given strength. Let *S_lesion* be the items that are base-correct and stay correct.

1. **Tracking index (TI)** for R1 under L2, at the strength with the closest-to-50% lesion-induced error rate among base-correct items:

   TI = [mean R1(S_lesion) − mean R1(E_lesion)] (under lesion) ÷ [mean R1(base-correct) − mean R1(base-incorrect)] (base model, natural errors).

   - TI = 0: FOK does not separate lesion-induced errors at all (complete anosognosia).
   - TI = 1: separates them as well as natural errors.

   Bootstrap CI over items.
2. **Dose–response slope β:** mixed-effects regression of ΔR1 on Δ log p(gold), pooled over L2 strengths, with random intercepts (template, relation).
3. **Sham contrast:** ΔR1 under L5a (matched) vs L2 on items whose accuracy did not change under either. This tests generic anomaly sensitivity.

**Confirmatory family (Holm):** TI(R1, L2) in each model; β(R1); sham contrast.

### A.7 Secondary analyses (exploratory unless stated)

- TI and β for R2–R6. Hypothesis H2: R2/R3 > R1.
- Double dissociation (H3): L4 effect on R1 vs on forced-answer accuracy and log p(gold), in standardised units.
- Popularity moderation: does TI differ by familiarity bin?
- L6 vs L2 comparison ("danger zone" replication).
- Observer baseline: a probe on layer-0 (embedding) representations and a separate small LLM, both trained to predict base correctness. Report their base AUROC; by construction their predictions do not change under lesions.

### A.8 Go / kill criteria for scaling P1

- **Go:**
  - base R1 type-2 AUROC ≥ 0.65 in ≥ 1 model;
  - an L2 strength exists where ≥ 30% of base-correct items fail;
  - matched sham (L5a) accuracy change ≤ 2 points;
  - general language damage is small: perplexity increase ≤ 10% on 200 held-out Wikipedia passages, with the same layer-band knock-out applied to a random noun-phrase span of matched length.
- **Kill / redesign:** any go criterion fails in both models. Redesign options: per-item activation patching lesions; 14B models in 4-bit; a different relation set.
- **Scientific outcome is not a go/kill criterion.** Both anosognosia and tracking are publishable.

### A.9 Power (to be refined)

- With ~1,000 base-correct items and ~50% lesion-induced error, each TI arm has ~500 items.
- Assuming within-group SD of R1 ≈ 1 base-separation unit, the TI standard error ≈ 0.09, so a 95% CI half-width ≈ 0.18.
- This is sufficient to distinguish TI < 0.3 from TI > 0.6.
- Re-estimate after the first 200 items (pre-specified; does not change the stopping rule).

### A.10 Compute

≈ 2,000 items × ~12 conditions × ~15 forward passes (short prompts, ≤ 16 generated tokens) ≈ 0.4 M short passes, plus layer selection and probes. Estimate: 8–15 GPU-h on one A100-class GPU.

---

## Pilot B: C2 in the workspace + mini ignition test (P1 × P2 bridge)

- **Model:** an open Qwen model with officially released J-lens matrices. Identify it first; if none fits, fit a lens with the released code on a ~7–9B model and validate against the reported J-lens sanity checks.
- **B1 (C2):**
  - At the question's final token, compute a J-space "uncertainty score": loading on J-lens vectors for a pre-specified token list (*unknown, unsure, not, ?, maybe, uncertain, don't*). The list is frozen before analysis.
  - DVs: AUROC of the score for known vs fictitious entities; for base-correct vs base-incorrect; and Pilot A's TI under L2.
  - Go: known-vs-fictitious AUROC ≥ 0.7.
- **B2 (mini ignition):**
  - Replicate the ambiguous-mixture paradigm: interpolate between two country concepts with mixture ratio r ∈ [0, 1] in 21 steps.
  - Trial variability from three pre-specified sources: (i) 20 paraphrased contexts, (ii) Gaussian activation noise at layer 2 with σ ∈ {0.5%, 1%, 2%} of residual norm, (iii) sampling at T = 0.7 for the downstream report.
  - DVs: sigmoid vs linear fit of J-space loading vs r (ΔAIC); Hartigan dip test at the empirical threshold r*; the same analysis on a random 25-direction subspace and on early-layer (pre-workspace) coordinates.
  - Ignition-like pattern = steep sigmoid **and** significant bimodality at r* in the J-space under ≥ 2 noise sources, **but not** in the matched controls.
  - Informative in either direction.
- **Compute:** 5–8 GPU-h.

---

## Pilot C: does merging produce superadditive, untrained self-monitoring? (P5 seed falsification)

- **Model:** `Qwen2.5-1.5B-Instruct`.
- **Adapter A ("monitor"):** LoRA (r = 16) trained to output a calibrated 0–9 confidence digit after answering trivia in domain X (e.g., geography). Labels: the model's own correctness on held-in items.
- **Adapter B ("skill"):** LoRA trained on a new domain-Y skill (e.g., unit-conversion word problems in a fixed format), with no confidence outputs.
- **Conditions:** base; A; B; A⊕B (task arithmetic, TIES, learnable concatenation); joint (A ∪ B data); sequential (B then A). 5 seeds each.
- **Primary DV:** confidence quality on held-out domain-Y items (type-2 AUROC; Brier on the 0–9 scale rescaled). Confidence was never trained in Y. Secondary: Pilot A's TI applied to the merged model on domain X.
- **Analysis:**
  - 2 × 2 factorial (A present × B present) on the continuous DV.
  - Interaction tested against an additive null and a multiplicative null [okawa2023multiplicative].
  - Merged vs joint vs sequential compared descriptively.
- **Kill (for merging as an integration mechanism):** best merge ≤ best single adapter on Y-confidence and interaction CI ∋ 0. P5 would then use architectural modules only.
- **Compute:** 6–10 GPU-h.

---

## Pilot D: implicit attention schema feasibility (P4)

- **Models:** `Qwen2.5-1.5B-Instruct`, `Qwen2.5-7B-Instruct`.
- **Stimuli:** prompts with K = 4 labelled short documents and a question answerable from exactly one (target position counterbalanced). Plus "split" items answerable from two documents, to create graded allocation.
- **Allocation vector a ∈ Δ⁴:** attribution of the first answer token to each document's tokens. Two methods: attention rollout and gradient × input, aggregated per document. We report agreement between the methods.
- **D1 (existence):** ridge probe from the residual stream at the final prompt token (best layer on a held-out split) to *a*, vs a content-only baseline (probe from mean-pooled document embeddings + question embedding).
  - Go: ΔR² ≥ 0.10 on held-out items.
- **D2 (causal use):** steer along the probe's direction toward a non-target document (α sweep); measure the shift in *a* vs sham random directions of equal norm.
  - Go: Cohen's d ≥ 0.5 at the α where accuracy drop ≤ 10 points.
- **D3 (exploratory):** ask "Which document did you mainly use?" under steering. Does the report follow the steered schema or the actual allocation?
- **Compute:** 3–5 GPU-h.
