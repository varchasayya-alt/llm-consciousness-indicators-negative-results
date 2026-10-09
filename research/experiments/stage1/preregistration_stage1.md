# Stage-1 Preregistration (DRAFT, NOT FROZEN)

## Does an independently trained monitor track item-specific changes in another network's competence, beyond perturbation, familiarity and input difficulty, and does composed control generalise to held-out internal interventions?

| Field | Value |
|---|---|
| Status | **DRAFT v0.1 (2026-10-01).** Not frozen. **No experimental seed has been run.** No monitor has been trained on any store. No hypothesis-related quantity has been computed |
| Freezing | When the open decisions in §25 are resolved: (i) intervention calibration (§25-D7) is completed on calibration seeds only; (ii) code passes the dry-run on synthetic fake results (§24). This file is then copied to `preregistration_stage1_FROZEN.md`, committed, and its SHA-256 and commit hash recorded in `logs/decisions.md`. Any later deviation goes to `logs/experiment_log.md` with justification and is reported in the paper |
| Program context | P5\* + N1 (`memo/decision_document_v2.md`). Level A, the current paper: causal artificial metacognitive monitoring and developmental integration in a controlled neural system. Level B, the long-term program: integration of several separately developed capacities. **Not attempted here** |
| Citation keys | `literature/bibliography.bib` |

---

## 1. Research question

> Can an independently trained metacognitive monitor learn to detect **item-specific changes in the competence** of another neural system from that system's internal computational state, and can a controller that uses the monitor's output adapt downstream behaviour (information-seeking) under **internal interventions never encountered during training**? And are these effects explained instead by (a) perturbation detection, (b) entity familiarity, (c) input difficulty, or (d) generic, store-independent decoding?

**Target phenomenon (operational).** For the same item *q* and the same input tokens:

1. intact store knows *q* → monitor reports competence;
2. a targeted intervention makes the store lose *q* → the monitor's competence estimate falls;
3. a matched intervention that does **not** impair *q* → the estimate stays stable;
4. a monitor reading an undamaged twin does not report the lesion;
5. input corruption causing an equivalent behavioural failure is distinguishable, in where it shows up, from an internal loss of competence.

**Definitions.**

- Store parameters θ; item *q* with true answer v\*(q).
- **Competence:** c_θ(q) = 1[argmax_{v∈V_r} p_θ(v|q) = v\*(q)].
- **Competence margin:** m_θ(q) = log p_θ(v\*|q) − max_{v≠v\*} log p_θ(v|q).
- **Monitor score:** s_M(q; θ) ∈ (0,1), the monitor's estimate of P(c_θ(q) = 1).
- **Score change:** Δs(q) = s_M(q; θ_post) − s_M(q; θ_pre).

---

## 2. Exact hypotheses

All hypotheses are Level 1 (computational). "INT-S" is the internal-access monitor trained only on the intact final store (§7).

**Primary**

- **H1 (held-out competence tracking).** Under the held-out targeted-forgetting intervention (T-FORGET, §8), INT-S's score decreases more for items whose competence was lost (X_lost) than for items given a displacement-matched, competence-preserving sham (Y).
  - Measure: TI_FORGET(INT-S) = AUROC(−Δs; X_lost vs Y) > 0.5.
  - Minimum effect of interest: 0.55.

**Key secondary (confirmatory family K)**

- **K1 (unrelated-knowledge specificity).** AUROC(−Δs; X_lost vs Z) > 0.5 for INT-S, where Z = untouched base-correct items in the same intervened store that are still correct after the intervention.
- **K2 (no response to perturbation magnitude).** For INT-S, mean Δs(Y) − mean Δs(Z) lies within ±0.05 (equivalence, TOST).
- **K3 (second novel mechanism).** TI_INTERF(INT-S) = AUROC(−Δs; lost vs retained base-correct EV items) > 0.5 under later-learning interference (T-INTERF).
- **K4 (developmental history).** d′(TI_FORGET, INT-TP) > d′(TI_FORGET, INT-S), where d′ = √2·Φ⁻¹(AUROC) and the test is paired over seeds.
- **K5 (composed behaviour).** For the composed system INT-S → controller C, AUROC(increase in P(LOOKUP); X_lost vs Y) > 0.5 at the pre-registered operating cost c = 0.3.

**Familiarity dissociation (confirmatory family F, two-sided)**

- **F1 (familiar-but-lost).** TI_FORGET(INT-S) restricted to high-familiarity X and Y ≠ 0.5.
- **F2 (familiarity rise without competence).** Under T-FAM (familiarity boost, competence unchanged), mean Δs(F_boost) − mean Δs(U2) ≠ 0 for INT-S. The TOST within ±0.05 is reported alongside.
- **F3 (unfamiliar but newly learned).** Under T-NEW, AUROC(+Δs; newly-learned vs familiarity-exposure-matched still-unknown) ≠ 0.5 for INT-S, low-familiarity stratum.

**Self-specificity (confirmatory family S, two-sided)**

- **S1.** For INT-S: d′(TI_FORGET, B̃→B̃) − d′(TI_FORGET, A→B̃) ≠ 0.
  - B̃ = store B's read-set states mapped into store A's coordinates by a linear map fit on intact states only.
  - A→B̃ = monitor trained on A, reading aligned B states while B is intervened.
  - B̃→B̃ = monitor trained on B's aligned states (same input space, different developmental pairing).
- **S2.** The same contrast for INT-TP.

**No directional prediction is made for F or S.** Both outcomes are interpretable (§21).

## 3. Null hypotheses

| Hypothesis | Null |
|---|---|
| H1 | TI_FORGET(INT-S) ≤ 0.5 |
| K1 | AUROC(X_lost vs Z) ≤ 0.5 |
| K2 | \|mean Δs(Y) − mean Δs(Z)\| ≥ 0.05 (the TOST null) |
| K3 | TI_INTERF ≤ 0.5 |
| K4 | d′(INT-TP) − d′(INT-S) ≤ 0 |
| K5 | AUROC(lookup increase) ≤ 0.5 |
| F1 | TI = 0.5 |
| F2 | difference = 0 |
| F3 | AUROC = 0.5 |
| S1, S2 | gap = 0 |

---

## 4. Architecture

```
            ┌──────────── interventions (store only) ────────────┐
            │ T-FORGET(X)+sham(Y) | T-INTERF | T-NEW | T-FAM |   │
            │ [expl.] T-ACT, REPLACE                              │
            ▼                                                     │
 q ──► STORE A (causal transformer) ──► p(v|q) ──► answer         │
 │        │ read set R: residual @ {s2, [A]} × layers 0..4        │
 │        ▼                                                       │
 │   MONITOR M ── p̂ = s_M(q) ──► CONTROLLER C ──► ANSWER | LOOKUP │
 │   IN   (reads q tokens only)                                   │
 │   OUT  (reads sorted p(v|q) + entropy)                         │
 │   INT-{S,T,P,TP} (reads R; developmental history differs)      │
 └──► (IN reads q directly)                                       │
 TWIN STORE B (same world/data, other init) ── linear map B→A (fit on intact states)
 input-corruption control acts on q, never on the store
```

The full diagram is in `architecture/diagrams/stage1_system.svg`.

**Separability.**

- Store, monitor and controller are separate modules with separate parameters and separate training.
- Each can be intervened on, ablated, clamped or swapped independently.
- Gradients never flow from monitor or controller into the store.

**Store (first-order system).**

- Pre-LN causal transformer: d_model = 128, 4 layers, 4 heads, MLP width 512, GELU, learned positional embeddings (max length 8), no dropout.
- About 0.9M parameters (the embedding count depends on vocabulary).
- Output at the [A] position is read only over the value tokens of relation r (restricted argmax).

**Monitor read set R (all INT monitors).**

- Residual stream after the embedding (layer 0) and after each of the 4 blocks, at two positions: the subject's last name token (s2) and the answer position ([A]).
- 10 sites × 128 = 1,280 features.
- Features are standardised per store with means and SDs from the **intact final store on the MT split** (frozen per store; identical across monitor conditions).

**Monitors.**

| Condition | Input | Network | Training data (all labels = correctness of the store state that produced the features) |
|---|---|---|---|
| IN | Own embeddings (dim 64) of s1, s2, r | 192 → 256 → 256 → 1 | Intact final store, MT split |
| OUT | Sorted softmax over V_r (32) + entropy (1) | 33 → 64 → 64 → 1 | Intact final store, MT split |
| INT-S | Standardised R (1,280) | 1280 → 256 → 256 → 1 | Intact final store, MT split |
| INT-T | R | same | MT split × 9 store-training checkpoints |
| INT-P | R | same | MT split × final store under P-family perturbations |
| INT-TP | R | same | 50/50 mixture of the T and P pools |
| *OUT-TP (expl.)* | as OUT | as OUT | as TP |
| *INT+OUT-S (expl.)* | R ⊕ OUT features | 1313 → 256 → 256 → 1 | as S |

- **Common monitor training:**
  - GELU; dropout 0.1; sigmoid output;
  - class-balanced binary cross-entropy (weights from each pool's label base rate);
  - AdamW, lr 1e-3, weight decay 1e-4;
  - **exactly 8,000 steps of batch 256 sampled with replacement from the condition's pool**, so the optimisation budget is identical across conditions.
- **Matching:** INT conditions differ only in their training pool. IN and OUT differ in input by design.

**Controller C.**

- MLP 2 → 16 → 1 on (logit p̂, c). Output P(LOOKUP).
- Utilities:
  - ANSWER = +1 if the store is correct, −λ if wrong (λ = 1);
  - LOOKUP = 1 − c (the external memory always returns the correct answer).
- Trained to maximise expected utility on the **CT split of the intact final store only**, with c ~ U(0.05, 0.6), 3,000 steps, AdamW lr 3e-3.
- One controller is trained per monitor condition.
- Test operating point: c = 0.3.
- **Variant C+in (secondary):** additionally receives a 64-d input embedding of q (the IN monitor's frozen embedding of s1, s2, r). This tests whether a controller with access to an input/familiarity shortcut relies on it instead of the monitor.

**Twin store B.** Same world, same training data and schedule; different initialisation and data-order seed. It has its own monitors (INT-S and INT-TP), its own controller, and its own T-FORGET run (its own X/Y/Z draw).

## 5. Synthetic task / environment

**World generator** (one world per store seed; the twin shares it).

- **Entities:**
  - 3,000 entities with 2-syllable names from an 80-syllable inventory (6,400 possible names);
  - 1,000 further names are reserved for T-INTERF;
  - the remaining 2,400 are never used in training (input-corruption pool).
- **Relations and values:** R = 4 relations; |V_r| = 32 values per relation (128 relation-specific value tokens). Answers are drawn uniformly, so facts cannot be inferred.
- **Familiarity:**
  - 50% of entities are "high" (8 mention sequences `[M] s1 s2 [E]` per epoch);
  - 50% are "low" (1 per epoch).
- **Knownness (coupled world):**
  - P(fact (e,r) trained | high) = 0.8; P(trained | low) = 0.3, independently per (e,r);
  - trained facts appear **once per epoch** regardless of familiarity, so memorisation exposure is equal per fact;
  - this mirrors the natural familiarity–knowledge correlation in LLM data.
- **Training sequences:** facts `[Q] s1 s2 r [A] v [E]` (known facts only) plus mention sequences.
- **Splits per entity** (relations randomly permuted per entity):
  - 2 relations → **MT** (monitor-train);
  - 1 → **CT** (controller-train);
  - 1 → **EV** (evaluation).

  Interventions target only EV items. MT/CT/EV share entities, so the input-only monitor *can* learn entity-level knownness (the external-observer familiarity shortcut) from MT.

**Expected EV counts per store:**

| | Known (base-correct) | Unknown |
|---|---|---|
| High familiarity | ≈ 1,200 | ≈ 300 |
| Low familiarity | ≈ 450 | ≈ 1,050 |

**Item sets for interventions** (drawn from EV, stratified by familiarity, with the store-specific seed):

| Set | Composition | Use |
|---|---|---|
| X | 300 base-correct trained facts (150 high / 150 low) | T-FORGET targets |
| Y | 300 base-correct (150 / 150) | Displacement-matched sham |
| Z | all remaining base-correct trained EV facts | Untouched (unrelated-knowledge control) |
| N | 240 unknown (120 / 120) | T-NEW learning targets |
| U | 240 unknown (120 / 120) | Exposure-matched still-unknown control |
| F | 300 unknown, low familiarity | T-FAM boosted entities |
| U2 | 300 unknown, low familiarity | T-FAM control |
| C | 240 base-correct | Input-corruption control (name → unused name); count matched to the expected X_lost |

Lucky-guess "correct" unknown items are never used as base-correct.

## 6. Training procedure

**Store.**

- Next-token cross-entropy on all tokens of all sequences.
- AdamW, lr 2e-3 with cosine decay to 2e-4, batch 512, weight decay 0.
- **E_store epochs:** fixed before freezing by the calibration rule in §25-D7 (expected 80–150).
- Checkpoints saved at {5, 10, 15, 20, 30, 40, 60, 80, 100}% of training.
- **Store QC** (store-level only; no monitor quantities):
  - trained-fact accuracy ≥ 98%;
  - unknown-fact accuracy ≤ 10% (chance ≈ 3%);
  - familiarity manipulation check: a logistic probe on the layer-2 residual at s2 separates high- vs low-familiarity entities with held-out AUROC ≥ 0.80.
- Twin B is trained identically with seed + 5000.

**Monitors:** §4, trained after store QC.

**Controller:** §4, trained after its monitor is frozen.

**Order:**

1. store A / B;
2. QC;
3. extract states;
4. train monitors;
5. train controllers;
6. apply interventions to copies of the frozen final store;
7. evaluate.

**No component is retrained after any intervention.**

## 7. Developmental conditions (monitor histories)

| Code | History | What varies within item q (fixed input) |
|---|---|---|
| S (static) | Intact final store only | Nothing. Competence is fixed per item |
| T (trajectory) | States of the store's own 9 training checkpoints | Competence acquired during learning (natural developmental change) |
| P (perturbation) | Final store under random activation dropout: each sample independently picks layers (each of 4 blocks w.p. 0.5; at least one) and applies Bernoulli dropout to the residual stream at all positions with rate ~ U(0, 0.6) | Transient, untargeted competence loss |
| TP | 50/50 mixture of T and P pools | Both |

- The 2×2 (T ∈ {no, yes} × P ∈ {no, yes}) is fully crossed **within internal access**. No cell is floored by construction.
- **Input-only access is deliberately not crossed with development.** Under input-invisible interventions IN gives TI = 0.5 identically, so an access × development interaction would be manufactured. It appears only as a baseline.
- **Answer-observing / second-pass access:**
  - OUT (output-distribution readout) is the answer-observing condition.
  - A second-pass "re-entry" condition is **not included in Stage 1.** In a feedforward causal store, re-embedding the store's own answer adds only the answer's identity. The theoretically motivated re-entry test needs a store with recurrent dynamics and is deferred to Stage 2.

## 8. Intervention types

All store interventions start from a copy of the intact final store. The input tokens of evaluated items are unchanged; this is verified per item.

| Code | Class | Mechanism | Role |
|---|---|---|---|
| **T-FORGET (+sham)** | Weight-level, item-targeted, objective-driven loss | **Joint run.** L = CE(uniform_{V_r}, p(·\|x)) over X + CE(v\*, p(·\|y)) over Y + γ·Σ_{sites∈R}(D̄_Y,site − sg[D̄_X,site])² + CE over a retain batch (all trained facts except X, plus all mention sequences). D̄ = mean L2 displacement of read-set states from their pre-intervention values. sg = stop-gradient. Adam, K_forget steps, lr_forget (calibrated, §25-D7) | **Primary held-out** |
| T-INTERF | Weight-level, untargeted collateral loss | Fine-tune on 1,000 new entities (reserved names; 2 facts + 8 mentions each), **no replay**, until new-fact accuracy ≥ 95% or max steps. Lost/retained EV items emerge naturally | Secondary held-out |
| T-NEW | Weight-level acquisition | Fine-tune on N's answers + retain batch. U's entities receive mention sequences matched in number of presentations to N's entities, so familiarity exposure is matched | Familiarity dissociation (F3) |
| T-FAM | Familiarity-only change | Fine-tune on 16 extra mention presentations for F's entities + retain batch. No fact training | Familiarity dissociation (F2) |
| *T-ACT (expl.)* | Activation-level, item-targeted | Inference-time knock-out of attention from [A] to s1/s2 in layers 3–4 for X. Y gets a matched-norm perturbation at the same sites, orthogonal to the gradient of the correct-answer logit | Near-transfer for P-trained monitors |
| *REPLACE (expl.)* | Weight-level, confident replacement | Train X′ toward a new random value (confident wrong answer) with a displacement-matched sham Y′ | **Limit case.** A pure retrieval-state monitor should *not* flag replaced items. If it does, it detects edit fingerprints |
| Input corruption | Input-level (store intact) | Replace the entity name of items in C with an unused name | Control #5 |

**Intervention QC** (store-level only; computed without any monitor):

- **T-FORGET:**
  - ≥ 50% of X incorrect;
  - ≤ 3% of Y incorrect;
  - ≤ 5% of Z incorrect;
  - D̄_X/D̄_Y ∈ [0.8, 1.25] at ≥ 8 of 10 read sites;
  - familiarity-probe scores of X and Y entities change by ≤ 0.1 SD.
- **T-INTERF:** 5–60% of base-correct EV lost.
- **T-NEW:** ≥ 70% of N correct; ≤ 5% of U correct.
- **T-FAM:**
  - boosted entities' familiarity-probe score rises ≥ 0.5 SD relative to U2;
  - ≤ 5% of F correct.
- **Fallback ladder** (pre-specified; applied in order until QC passes): lr × {1, 0.5, 2}; then steps × {1, 2}. If all rungs fail, that intervention is excluded for that store (§16).

## 9. Held-out interventions: novelty status by monitor history

| | Natural | T-FORGET | T-INTERF | T-NEW | T-FAM | T-ACT | REPLACE |
|---|---|---|---|---|---|---|---|
| S | trained | **novel** | **novel** | **novel** | **novel** | novel | novel |
| T | trained | **novel** (loss by targeted objective; T only saw acquisition) | partly novel (T saw learning dynamics) | *seen-type* (acquisition) | novel | novel | novel |
| P | trained | **novel** (weight-level, targeted) | **novel** | novel | novel | *near* (activation-level) | novel |
| TP | trained | **novel** | partly novel | seen-type | novel | near | novel |

- Confirmatory claims about held-out generalisation are made **only** for cells marked **novel**.
- "Seen-type" and "near" cells are reported but labelled as such.

## 10. Baselines

- **IN** (external observer with the familiarity shortcut available);
- **OUT** (first-order output readout);
- **chance** (0.5);
- **"pre-intervention level" baseline:** the monitor's own pre-intervention scores predicting post-intervention loss. This quantifies item-level predictability of loss.
- **Oracle controller:** the true post-intervention competence, giving the utility ceiling.
- **Always-answer** and **always-lookup** controllers.

## 11. Negative controls

Each item below gives the control, its implementation, and the expected result if the monitor tracks competence.

1. **Matched-magnitude irrelevant perturbation:** Y in the T-FORGET run (displacement matched at the read sites, competence preserved). Expected: Δs(Y) ≈ Δs(Z) (K2) and X_lost ≫ Y (H1).
2. **Sham procedure (same procedure, competence preserved):** X_retained (X items still correct after forgetting). Expected: secondary AUROC(X_lost vs X_retained) > 0.5. Within X, Δs tracks Δm.
3. **Unrelated-knowledge intervention:** Z (untouched base-correct items in the same intervened store). Expected: K1 > 0.5, and no global drop in Δs(Z) (descriptive).
4. **Twin-system control:**
   - (a) Δ-based: a monitor reading the undamaged twin has Δs ≡ 0, so TI = 0.5. This is a sanity check, true by construction.
   - (b) Level-based: the A-monitor reading **aligned undamaged-B** states predicts A's X_lost vs Y. Expected ≈ 0.5, because X/Y are randomly assigned. If it is > 0.55, loss is predictable from item properties, and only Δ-based endpoints are interpretable.
5. **Input corruption:** set C (name → unused name; store intact). Expected:
   - IN detects it (level AUROC corrupted vs intact > 0.5) while giving TI = 0.5 on internal lesions;
   - INT detects both;
   - mechanistic: corruption lowers the store's familiarity signal, lesions do not (M6).
6. **Novel intervention class:** T-FORGET and T-INTERF for all histories (§9).
7. **Familiarity controls:** T-FAM (familiarity ↑, competence unchanged) and T-NEW with exposure-matched U (competence ↑, familiarity change matched). High-familiarity stratum of T-FORGET (familiar but lost).
8. **Fingerprint limit case (exploratory):** REPLACE.
9. **Shuffled-label monitor** (INT architecture trained on permuted labels). Expected: TI ≈ 0.5 and natural AUROC ≈ 0.5.

## 12. Primary endpoint

**TI_FORGET(INT-S):** per store seed A, the Mann–Whitney AUROC (ties count 0.5) of the score *decrease* −Δs.

- Positives: X_lost (X items with c_θpost = 0).
- Negatives: Y items with c_θpost = 1.
- Seed-level values are analysed across the 20 seeds.

## 13. Secondary endpoints

**Confirmatory (families K, F, S):** as in §2.

**Planned secondary (estimation with 95% CIs; no confirmatory claims):**

- **Factorial decomposition** of TI_FORGET and TI_INTERF over T × P within INT, on the d′ scale and on the raw AUROC scale (§14).
- **Development × intervention-class** interaction (FORGET, INTERF, NEW) on d′.
- **Continuous competence-change sensitivity:** per seed, Spearman ρ between Δs and Δm across all base-correct EV items under each intervention.
- **AUROC(X_lost vs X_retained)**, and the X-internal Spearman(Δs, Δm).
- **Natural-data metrics on intact EV** for all monitors: AUROC, AUPRC, Brier, ECE (15 bins), selective risk at 80% coverage. These form the observational-equivalence panel.
- **Global false-positive rate:** fraction of Z whose controller decision flips ANSWER → LOOKUP. Sham false-positive rate: the same for Y.
- **Behaviour at c = 0.3:**
  - lookup hit rate on X_lost and false-alarm rate on Y/Z, pre vs post;
  - **lookup efficiency** = (post-intervention utility − always-answer utility) / (oracle utility − always-answer utility);
  - accuracy conditional on answering;
  - C vs C+in (reliance on the input shortcut).
- **Twin level-based control (11.4b)** and the pre-intervention level baseline.
- **Input-corruption** level AUROCs for IN, OUT, INT-S.
- **OUT vs INT-S** on each intervention, stratified by whether the store's post-intervention error is low- or high-confidence (max p < or ≥ 0.5).
- **Self-specificity matrix:** A→A, B→B, A→B (raw), A→B̃, B̃→B̃, for natural AUROC and TI.

**Exploratory (labelled as such):**

- T-ACT; REPLACE; OUT-TP; INT+OUT-S;
- mechanistic analyses M1–M6 (§13a);
- a decoupled world (P(trained|high) = P(trained|low) = 0.55; 10 seeds; primary pipeline only).

### 13a. Mechanistic analyses (planned, exploratory)

- **M1 Site importance.** Replace each read site with its training mean, one at a time; record the change in natural AUROC and TI_FORGET.
- **M2 Geometry.** On the intact store, logistic probes per site for:
  - (a) recall success (known vs unknown),
  - (b) familiarity class,
  - (c) entity identity (ridge decodability).

  Report the angle between the (a) and (b) directions, and (a)'s accuracy within each familiarity stratum (does knowledge state separate from familiarity?).
- **M3 Causal.**
  - (a) Steer intact states along the recall direction and measure monitor response.
  - (b) Project the X_lost state changes onto the recall and familiarity directions.
  - (c) Cosine between the monitor's input-gradient and those directions.
- **M4 Monitor → controller.** Clamp p̂ at its pre-intervention value: the lookup change must vanish (verification of the design). Sweep p̂ to plot the controller response curve.
- **M5 Cross-relation generality.** Train the monitor on 2 relations' MT items; test natural AUROC and TI on the other 2 relations.
- **M6 Corruption vs lesion signature.** Change in the familiarity-probe score for C (corrupted) vs X_lost (lesioned).

## 14. Statistical tests

- **Unit:** store seed (world instance + store A + twin B). All item-level quantities are reduced to one value per seed before inference.
- **Directional one-sample** (H1, K1, K3, K5): one-sided one-sample t-test of (seed value − 0.5). Reported with the Wilcoxon signed-rank (robustness) and a seed-bootstrap 95% CI (10,000 resamples).
- **Paired directional** (K4): one-sided paired t-test on d′ differences.
- **Equivalence** (K2): TOST with bounds ±0.05 on the seed-level mean difference.
- **Two-sided** (F1–F3, S1–S2): two-sided one-sample / paired t-tests.
- **d′ transform:** d′ = √2·Φ⁻¹(AUROC), with AUROC clipped to [0.001, 0.999].
- **Superadditivity** (planned secondary): the interaction contrast (TP − T − P + S) on d′ and on raw AUROC, seed-paired. "Superadditive" may be used **only if**:
  - the 95% CI excludes 0 on both scales, **and**
  - TP exceeds both T and P;
  - multiplicative-composition null: also report the interaction on the logit-AUROC scale.
- **Interpretation thresholds:** **support** requires p (Holm-adjusted where applicable) < 0.05 **and** the point estimate ≥ the minimum effect of interest (AUROC 0.55; d′ difference 0.10).

## 15. Number of seeds and justification

- **20 analysed store seeds**, taken as the first 20 seeds from the sequence 1001, 1002, …, 1040 that pass store QC. Each seed's twin uses seed + 5000. Monitor, controller and intervention seeds are derived deterministically from (store seed, condition code).
- **Calibration seeds** 9001–9005: used only for §25-D7. They never have monitors trained or evaluated.
- **Burned seed:** 12345 (feasibility run; never used).
- **Sensitivity** (simulation, `results/statistics/stage1_power_sensitivity.json`). With n = 20, 80% power is reached for seed-level standardised effects of:
  - dz ≥ 0.58 (one-sided α = .05; H1);
  - dz ≥ 0.79 at α = .0083, a conservative bound for family K (its worst-case Holm α is .05/5 = .01).
- **Translated to AUROC:** seed SD 0.03–0.06 implies detectable differences of ≈ 0.02–0.05. Within-seed AUROC standard error at the planned set sizes is ≈ 0.023–0.029 (Hanley–McNeil), so seed SD is unlikely to fall below ≈ 0.025.
- **TOST (K2) power:** ≥ 0.9 when the true difference is 0 and seed SD ≤ 0.05 (conservative α = .0083), dropping to ≈ 0.54 when the true difference is 0.02 and SD is 0.05.
- **Why 20, not 10:** with n = 10 the minimum detectable dz under the K-family correction is ≈ 1.2, which is too coarse for K4 (developmental differences are expected to be modest).
- **Why not 30:** n = 30 would lower the minimum detectable dz to ≈ 0.62 at about 1.5× the CPU time. This is an open decision (§25-D9).

## 16. Exclusion criteria (all decided without monitor or controller outputs)

1. **Store A or B fails store QC (§6):** the seed is excluded and the next seed in the sequence is used. All exclusions are reported.
2. **An intervention fails QC after the full fallback ladder:**
   - T-FORGET: the seed is excluded and replaced (it carries the primary endpoint).
   - Other interventions: that intervention's data for that seed is excluded (not replaced). Analyses use the available seeds; the count is reported.
3. **Item-level exclusions:**
   - lucky-guess unknown items are never in X/Y/Z/C;
   - an item whose input tokens differ between pre and post (should be impossible) is excluded and flagged as a bug.
4. **No exclusions based on monitor or controller performance.** Monitors with poor natural AUROC are retained.

## 17. Failure criteria (study-level; if met, no confirmatory claims are made)

- **(a)** Fewer than 20 of seeds 1001–1040 pass store and T-FORGET QC. Design failure; report.
- **(b)** Median natural-EV AUROC of INT-S < 0.65. The monitors failed to learn even natural competence, so tracking tests are uninterpretable.
- **(c)** Twin level-based AUROC (11.4b) median > 0.60. Item-level loss is too predictable from item properties. Δ-based endpoints remain reportable but are flagged.
- **(d)** Median ratio of read-site displacement X/Y outside [0.67, 1.5] on the evaluation seeds (despite calibration). The sham is not matched; H1 and K2 are reported as unmatched and not confirmatory.

## 18. Multiple-comparison handling

- **Primary H1:** α = 0.05, single test.
- **Family K** (K1–K5; K2 counts as one test with both TOST sides at the adjusted α): Holm, FWER 0.05.
- **Family F** (F1–F3): Holm, FWER 0.05, two-sided.
- **Family S** (S1–S2): Holm, FWER 0.05, two-sided.
- **Planned secondary and exploratory:** no correction. Reported with CIs and explicitly labelled. No claim of confirmation.

## 19. Planned figures and tables

| Item | Content |
|---|---|
| Fig. 1 | System + intervention schematic (§4 diagram; X/Y/Z design) |
| Fig. 2 | Observational-equivalence panel: natural AUROC / Brier / ECE for IN, OUT, INT-S/T/P/TP on intact EV |
| Fig. 3 | **Primary:** TI_FORGET by monitor (seed dots, mean ± 95% CI); Δs distributions for X_lost, X_retained, Y, Z |
| Fig. 4 | Familiarity quadrants: familiar-but-lost (high-familiarity TI); unfamiliar-newly-learned (F3); familiarity-boost (F2); controls |
| Fig. 5 | Developmental 2×2 × intervention class (FORGET, INTERF, NEW) on d′, with interaction contrasts |
| Fig. 6 | Behaviour: lookup rates pre/post for X_lost / Y / Z; efficiency; C vs C+in |
| Fig. 7 | Self-specificity matrix (A→A, B→B, A→B raw, A→B̃, B̃→B̃) for natural AUROC and TI |
| Fig. 8 | Mechanistic: site importance; recall vs familiarity directions; Δ-state projections; corruption vs lesion signatures |
| Table 1 | Manipulation checks per intervention per seed (accuracy changes, displacement ratios, familiarity changes, ladder rung used) |
| Table 2 | All confirmatory tests: estimate, 95% CI, p, Holm-adjusted p, decision |
| Table 3 | Planned secondary and exploratory results (labelled) |

## 20. Exact falsification conditions

| Hypothesis | Supported | Falsified (strong form) | Otherwise |
|---|---|---|---|
| H1 | One-sided p < .05 **and** mean TI ≥ 0.55 | Upper 95% CI bound of mean TI_FORGET(INT-S) < 0.55 | Inconclusive |
| K1 | Holm-adjusted p < .05 | Upper CI < 0.55 | |
| K2 | Equivalence shown | Δs(Y) − Δs(Z) significantly < −0.05 (the monitor responds to perturbation magnitude) | |
| K3 | Holm-adjusted p < .05 | Upper CI of TI_INTERF < 0.55 | |
| K4 | Holm-adjusted p < .05 | Upper CI of the d′ difference < 0 (developmental exposure hurts) | |
| K5 | Holm-adjusted p < .05 | Upper CI < 0.55 | |

- A cross-cutting reading for H1: if mean AUROC(X_lost vs Y) ≈ AUROC(X_lost vs Z) while K2 fails, the monitor's tracking is explained by perturbation magnitude, and H1's interpretation as competence tracking is falsified even if H1 is statistically supported.
- **Familiarity explanation:** **supported** (bad for the competence interpretation) if F2 shows a significant positive false rise ≥ 0.05 **and** F1 (high-familiarity TI) is not significant.
- **Generic decoding:** **supported** if S1/S2 gaps have 95% CIs within ±0.10 d′ (TOST, reported descriptively).

## 21. Interpretation allowed under each major result

| Result pattern | Allowed interpretation (Level 1) | Allowed Level-2 remark (conditional) |
|---|---|---|
| H1 ✓, K1 ✓, K2 ✓ (equivalent), K3 ✓, F2 null | A monitor trained only on an intact system tracks item-specific competence loss under two held-out internal mechanisms, not perturbation magnitude, not familiarity. Its information comes from the observed system's internal state (input unchanged) | Satisfies an E3-type criterion (counterfactual self-dependence) for a HOT-2/GNW-C2-relevant monitoring indicator **in this synthetic system** |
| H1 ✓ but K2 ✗ (Y ≈ X_lost) | Monitor responds to representational perturbation magnitude: anomaly detection, not competence tracking | None beyond "perturbation-sensitive monitor" |
| H1 ✓ but F2 positive and F1 ✗ | Monitor tracks entity-familiarity-correlated features. Competence tracking is partial or confounded | Familiarity-based (world-tracking) metacognition, analogous to cue-familiarity FOK [reder1992fok] |
| H1 ✗, K4 ✓ (TP > S) | Tracking under held-out interventions depends on developmental exposure to the observed system's competence variation | Consistent with SOMA-style learned metarepresentation [cleeremans2020learning]; **not** evidence for it in brains |
| H1 ✗, K4 ✗, OUT tracks | Competence change is available only via the first-order output readout; internal monitors learned non-tracking (shortcut) features | First-order readout account of confidence [fleming2024review] suffices here |
| K5 ✓ (with H1 ✓) | Composed modules generalise information-seeking to a held-out internal change without any training on it ("compositional zero-shot transfer of metacognitive control") | Functional analogue of metacognitive control (C2) **in this system** |
| C+in < C | A controller given an input shortcut under-uses the monitor; integration architecture matters | — |
| S1/S2 gap ≈ 0 | Competence decoding is generic across matched stores (linearly alignable) | No self-specific coupling |
| S1/S2 gap > 0 (B̃→B̃ > A→B̃) | Monitors become coupled to idiosyncratic, non-linearly-alignable features of the system they developed with | "System-specific monitoring coupling." **Not** "identity" or "self" |
| Superadditive T×P (both scales, TP > T, P) | Combined developmental histories produce more held-out tracking than predicted additively | Narrow, pre-defined sense of "emergent" (§21a) |

### 21a. Pre-registered meaning of "emergent"

1. **"Compositional zero-shot transfer"** is the term used for K5. The composed system shows held-out selective information-seeking when:
   - no component was trained on any data from the evaluated intervention class;
   - the controller was trained only on the intact store;
   - and the effect is selective (X_lost vs Y).
2. **"Emergent"** is used **only** for a superadditive developmental interaction meeting §14's criteria. It is not used for performance gains from combining modules, nor for K5.

## 22. Claims explicitly NOT permitted

- That any component or the system is conscious, sentient, aware, experiencing, or has a self or identity (Level 3). Also any implication of this via wording ("knows it doesn't know", "feels", "self-aware").
- That LLMs (or brains) use the same mechanism, or that the result transfers to models of other scales or architectures.
- That the monitor implements a higher-order *thought* in the philosophical sense. Only "satisfies an E3-type operational criterion" is allowed.
- That developmental exposure (T/P) "causes self-monitoring in general". Claims are restricted to the tested intervention classes and world type.
- "Emergent" outside §21a; "self-specific" or "identity" for S-family results (use "system-specific coupling").
- That replaced (REPLACE) or familiarity-boosted items' results reflect "beliefs".
- Any claim from exploratory analyses presented as confirmatory.

## 23. CPU runtime estimate

**Basis:** measured on an Intel Core Ultra 7 255U (CPU only): `results/raw/feasibility/`.

| Step | Time |
|---|---|
| Store training (3,000 entities; ~20k sequences/epoch; ~100 epochs), ×2 (A, B) | ≈ 10–20 min per seed |
| State extraction (MT × 9 checkpoints + P draws; EV pre/post for all interventions) | ≈ 3–5 min |
| Monitor training (8 A-conditions + 2 B + 2 aligned; MLPs on cached features) | ≈ 10–15 min |
| Controllers | ≈ 1–2 min |
| Interventions (6 A-runs + 1 B-run of fine-tuning on a 0.9M-parameter model) | ≈ 5–10 min |
| Evaluation and alignment | ≈ 2 min |
| **Per seed** | **≈ 30–55 min** |

| Total | Time |
|---|---|
| 20 seeds | ≈ 10–18 CPU-h |
| Calibration (5 seeds, stores + interventions only) | ≈ 1–2 h |
| Exploratory decoupled world (10 seeds) | ≈ 4–7 h |
| **Overall** | **≈ 15–27 CPU-h** (overnight batches; < 2 GB RAM) |

## 24. Reproducibility procedure

1. **Code:** single package `research/experiments/stage1/src/` with modules `world`, `store`, `monitors`, `controller`, `interventions`, `evaluate`, `analysis`, and unit tests:
   - input tokens unchanged under interventions;
   - set disjointness;
   - QC functions;
   - AUROC with ties.
2. **Config:** all values in `stage1_config.yaml` (drafted from `stage1_config_DRAFT.yaml`). The frozen config hash is recorded.
3. **Environment:** `pip freeze` saved to `stage1_environment.txt`; Python 3.12, torch 2.14.1+cpu. `torch.use_deterministic_algorithms(True)`; fixed thread count (recorded); seeds for Python, NumPy and torch per module.
4. **Dry run before freeze:** the analysis pipeline is executed on **synthetic fake result files** (random numbers with planted effects) to verify tests, Holm logic and figures. **No real seed is analysed before freezing.**
5. **Execution:** a single command runs calibration (stores and interventions only), then the evaluation seeds. Logs, timings and QC outcomes are written per seed.
6. **Raw outputs:** per-seed JSONL, one record per (item, intervention, monitor condition). Fields: item id, sets, familiarity class, pre/post correctness and margins, Δs, controller P(lookup) pre/post, read-site displacements. Store checkpoints are saved locally (git-ignored) with SHA-256 recorded.
7. **Analysis:** `analysis.py` reads only `results/raw/stage1/`. Outputs go to `results/processed/`, `results/statistics/`, `results/figures/`. It is run once after all seeds complete. Any rerun is logged.
8. **Blinding of decisions:** QC and exclusion decisions are computed by code from store-level quantities before any monitor evaluation is read.

---

## Appendix A. Experiment matrix (per store seed)

**Monitor/system condition × test condition.** Legend: P = primary; K / F / S = confirmatory families; s = planned secondary; e = exploratory; — = not run or not meaningful.

| | Natural EV | T-FORGET (X/Y/Z) | T-INTERF | T-NEW | T-FAM | Corruption | T-ACT | REPLACE |
|---|---|---|---|---|---|---|---|---|
| IN | s | s (=0.5 by construction) | s | s | s | s | e | e |
| OUT | s | s | s | s | s | s | e | e |
| **INT-S (A)** | s | **P**, K1, K2, F1 | K3 | F3 | F2 | s | e | e |
| INT-T (A) | s | s (factorial) | s | s (seen-type) | s | s | e | e |
| INT-P (A) | s | s (factorial) | s | s | s | s | e (near) | e |
| INT-TP (A) | s | **K4**, s | s | s | s | s | e | e |
| INT-S → C | s | **K5** | s | s | s | s | e | e |
| INT-S → C+in | s | s | s | — | s | s | — | — |
| OUT-TP, INT+OUT-S | e | e | e | e | e | e | e | e |
| Shuffled-label INT | s | s | — | — | — | — | — | — |
| B-side INT-S / INT-TP, B̃→B̃, A→B̃, A→B raw | s | **S1, S2**, s | — | — | — | — | — | — |
| Twin level control (A-monitor on aligned undamaged B) | — | s (11.4b) | — | — | — | — | — | — |

**Developmental factorial (within INT):** T ∈ {0, 1} × P ∈ {0, 1}, crossed with intervention class ∈ {FORGET, INTERF, NEW} as a within-seed repeated measure.

## Appendix B. Confound → control table

| Confound / alternative explanation | Control(s) | Diagnostic pattern |
|---|---|---|
| Monitor detects *that something changed* (anomaly / fingerprint [youssef2025edits; chen2025unlearntrace]) | Within-store Δ-AUROC (global shifts cancel); displacement-matched sham Y; X_retained; REPLACE (expl.) | X_lost ≫ Y ≈ Z; X_lost > X_retained; REPLACE not flagged |
| Monitor uses perturbation *magnitude* at its read sites | Displacement matched per read site (QC); per-site displacement covariate (s) | AUROC(X_lost vs Y) remains > 0.5 within displacement-matched strata |
| Entity familiarity, not competence [ferrando2025entity] | Familiar-but-lost stratum (F1); T-FAM (F2); exposure-matched T-NEW (F3); familiarity-probe manipulation checks | F1 > 0.5; F2 ≈ 0; F3 > 0.5 |
| Input-level difficulty / question features [marina2025llmindep; moran2026individuated] | Inputs identical under store interventions; IN baseline; input-corruption control | IN: TI = 0.5 but corruption detected; INT tracks lesions |
| Item-level susceptibility to forgetting (selection) | Random X/Y assignment; Δ-based endpoint; twin level control; pre-intervention level baseline | Twin/pre-level AUROC ≈ 0.5; Δ-based TI > level-based |
| Global calibration drift after fine-tuning | Within-store AUROC; Z global-shift and false-positive rates | No large mean Δs(Z) |
| Generic decoding vs system-specific coupling | Twin store B; linear alignment fit on intact states; B̃→B̃ vs A→B̃ (same input space) | Gap ≈ 0 → generic; gap > 0 → coupling |
| Basis mismatch makes cross-store transfer fail trivially | Alignment [bansal2021stitching]; raw A→B reported only as a reference | — |
| Training-budget differences across developmental conditions | Identical steps/batches; class-balanced loss; same architecture | — |
| Label-balance differences across histories | Class-balanced BCE | — |
| Metric non-linearity manufacturing interactions [schaeffer2023mirage] | Interactions on d′ **and** raw AUROC (and logit) scales; continuous Spearman(Δs, Δm) | Superadditivity only if consistent across scales |
| First-order readout vs internal monitoring | OUT baseline; stratification by post-error confidence | INT > OUT only where errors are confident |
| Controller memorises item-level policies | Controller trained on CT split only; C+in variant tests shortcut reliance | — |
| Leakage of held-out intervention data | No component trained after interventions; §9 novelty table; code test | — |
| Researcher tuning on evaluation | Only intervention hyperparameters calibrated, on separate seeds, using store-level QC only; dry run on fake data | — |
| Store memorisation imbalance across familiarity | Equal fact exposure per epoch; QC trained-fact accuracy ≥ 98% in both strata (reported) | — |

## Appendix C. Power / sensitivity analysis

See §15 and `results/statistics/stage1_power_sensitivity.json` (`experiments/stage1/power_sensitivity.py`, simulation only).

- **Unit:** store seed.
- **Within-seed precision:** with |X_lost| ≈ 150–270 and |Y| = 300, the SE of each seed's AUROC is ≈ 0.019–0.029 (AUROC 0.6–0.8).
- **The high-familiarity stratum (F1)** has about half the items, so its SE is ≈ 0.035.
- **Minimum detectable effects, n = 20, 80% power:**

| α | dz | AUROC difference at seed SD 0.04 | AUROC difference at seed SD 0.06 |
|---|---|---|---|
| One-sided .05 | 0.58 | ≈ 0.023 | ≈ 0.035 |
| Conservative family-K bound (.0083; worst case is .01) | 0.79 | ≈ 0.032 | ≈ 0.047 |

- **Limitation:** no real variance estimate exists (by design: none of the hypothesis-related pilot quantities were computed). If the realised seed SD exceeds 0.08, K-family tests are underpowered for differences < 0.06. This will be stated.

## Appendix D. Researcher degrees of freedom to freeze

All values below go into `stage1_config_DRAFT.yaml` and are frozen before data collection.

1. **World:**
   - entity count;
   - syllable inventory;
   - name length;
   - reserved / unused name pools;
   - relations;
   - values per relation;
   - familiarity proportions and mention multiplicities;
   - knownness probabilities;
   - split assignment (2/1/1).
2. **Store:**
   - d_model, layers, heads, MLP width, positional scheme;
   - optimiser, lr schedule, batch size, **epochs (calibrated)**;
   - checkpoint fractions;
   - QC thresholds;
   - familiarity-probe specification (layer, site, held-out split).
3. **Read set:** sites, standardisation source.
4. **Monitors:**
   - architectures, dropout, optimiser, steps, batch size;
   - class balancing;
   - P-family (layer-selection probability, dropout range, target: residual stream);
   - T-family checkpoint list;
   - TP mixing ratio.
5. **Controller:** architecture; λ; c range; training steps; operating point; C+in feature definition.
6. **Interventions:**
   - set sizes and stratification;
   - **T-FORGET:** objective, γ, steps, lr (calibrated); retain-batch composition;
   - QC thresholds; fallback ladder;
   - **T-INTERF:** new-entity count, facts, mentions, stopping rule, maximum steps, lr;
   - **T-NEW:** steps, lr, exposure matching;
   - **T-FAM:** extra exposures;
   - **T-ACT:** layers, sham construction;
   - **REPLACE:** target sampling;
   - input corruption: name source, count.
7. **Endpoints:** AUROC on −Δs with ties = 0.5; positive/negative set definitions; d′ transform and clipping; minimum effects of interest (0.55; d′ 0.10); equivalence bound (±0.05); confidence stratification threshold (0.5); ECE bins (15).
8. **Statistics:** tests; one- vs two-sided; families; Holm; α; bootstrap count; seed counts and sequences; exclusion and failure rules.
9. **Software:** versions, thread count, determinism flags.

## Appendix E. Relationship to the closest prior work (novelty boundary)

- **Already established (not claimed):**
  - error prediction from hidden states by meta-models / observer networks [chen2019whitebox; corbiere2019confidnet];
  - detection of edits and unlearning traces from hidden states [youssef2025edits; chen2025unlearntrace];
  - familiarity directions gating refusal [ferrando2025entity];
  - post-unlearning confabulation [gu2026unlearners];
  - editing → confidence [hasegawa2025underconf];
  - global degradation danger zone [cohen2026source; full text pending];
  - output-consistency vs accuracy tracking in trained LLM metacognition [yax2026forms];
  - question-only adequacy of retrieval decisions [marina2025llmindep];
  - cross-model probe transfer under alignment [srey2026probes];
  - self-interventional learning [tomaszewski2026sil];
  - structural integration of self-monitoring [xie2026structural];
  - toy multi-theory ablations [phua2025ablations].
- **What this study adds (if results warrant):** we have **not yet identified** prior work that combines, in one controlled system with known ground truth:
  - (i) within-item competence-change tracking under **held-out** internal mechanisms, measured with a Δ-based endpoint against a **read-site displacement-matched** competence-preserving sham, a same-procedure retained set, and unrelated items;
  - (ii) the familiarity × competence quadrant dissociation (familiar-but-lost, familiarity-boost, exposure-matched new learning);
  - (iii) developmental history as a crossed factor (trajectory × perturbation);
  - (iv) composition with a controller trained only on intact data;
  - (v) system-specific coupling tested in a common aligned space (B̃→B̃ vs A→B̃).
- This statement remains provisional pending full-text checks: [cohen2026source; yax2026forms; tomaszewski2026sil; xie2026structural; phua2025ablations; youssef2025edits; chen2025unlearntrace].

---

## 25. Open decisions before freezing (to be resolved with the PI)

| ID | Decision | Recommendation | Why it matters |
|---|---|---|---|
| D1 | Primary monitor: INT-S (static) vs INT-TP | **INT-S** | Strictest "never encountered any change" test. Development enters as K4. Choosing TP as primary would make H1 easier and its claim weaker |
| D2 | Forgetting objective: KL-to-uniform vs gradient ascent (GA) | **KL-to-uniform** for primary; GA as an optional extra exploratory class | Uniform-target errors are low-confidence, so OUT will track them easily and INT-vs-OUT is uninformative for FORGET. T-INTERF supplies naturally arising (possibly confident) errors. GA gives confident errors but is unstable and mixes "loss" with "replacement" |
| D3 | Equivalence bound for K2/F2 | **±0.05** (probability units) | Smaller bounds need lower seed SD than we can guarantee (Appendix C) |
| D4 | Coupled-world parameters (0.8 / 0.3) and the decoupled exploratory arm | Keep; decoupled arm exploratory | The shortcut-availability manipulation (ρ) becomes a Stage-1b confirmatory study if the exploratory arm is interesting |
| D5 | P-family: dropout only vs dropout + Gaussian noise | **Dropout only** | Keeps T-ACT (knock-out) and Gaussian-type perturbations as distinct held-out classes |
| D6 | Read set: 2 positions × 5 layers vs all positions | **2 × 5** | Smaller, interpretable; matches the displacement-matching sites |
| D7 | **Intervention calibration run** (stores + interventions only, seeds 9001–9005, *no monitors*) to fix E_store, lr/steps/γ for T-FORGET, T-INTERF max steps, and to verify that read-site displacement matching is achievable | **Required before freeze** | If displacement matching is infeasible, the primary control must be redesigned *before* freezing. This run examines no hypothesis |
| D8 | Write code + unit tests + fake-data dry run | Required before freeze | Prevents post-hoc analysis changes |
| D9 | Seeds: 20 vs 30 | **20** (30 if the PI prefers the extra ~6–9 CPU-h) | Sensitivity in Appendix C |
| D10 | Public preregistration (e.g., OSF) vs git-timestamp only | PI's choice | External timestamp strengthens credibility. Publishing is outward-facing, so it needs the PI's explicit decision |
| D11 | Include the C+in controller variant | **Yes** (secondary) | Tests whether integration architecture determines whether the monitor is used |
| D12 | Cohen & de Melo full text | Not blocking; read when available | May sharpen Appendix E |
