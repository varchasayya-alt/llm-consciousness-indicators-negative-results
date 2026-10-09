# Instrument Validity Before Hypothesis Tests: A Pre-Registered, CPU-Only Search for Consciousness-Relevant Computation in Small Language Models

**Technical report.**

| | |
|---|---|
| Date | 2026-10-09 |
| Written against | commit `6dde7eb` |
| Status | All four research lines are closed or stopped. No positive scientific claim is made |
| Claims discipline | Level 1 = computational result; Level 2 = conditional mapping to a theory of consciousness; Level 3 = phenomenal consciousness, **never asserted**. Everything below is Level 1 or about instruments. Nothing here bears on whether any system is conscious |

**How to read citations.**
- A number tagged `[tag@commit]` comes from the file(s) listed for that tag in Appendix A, as committed at that commit.
- Literature is cited by its key in `research/literature/literature_db.json`, e.g. [butlin2025indicators].
- **Public snapshot.** The commit hashes refer to the development history, which is not published. Every cited file is included at its final version; see the root `README.md`.

---

## Abstract

We report a pre-registered research program, run between 2026-10-01 and 2026-10-08 on a single CPU-only laptop at zero compute cost [hw@f378121]. Its aim was to measure *indicator properties*: functional properties that scientific theories of consciousness associate with conscious processing [butlin2025indicators]. We looked for them in small language models and synthetic transformers. Every experiment was committed before any model saw its material, with frozen thresholds, sealed confirmation splits and pre-declared stopping rules.

Four lines ran:

1. **Self-competence monitor ladder.** A calibration ladder tried six ways to induce an input-invisible change in a synthetic fact store's own competence. All six failed a binding validity gate, so no monitor was ever trained [b1@5432243; d2@2c00545].
2. **Global-workspace assay (C15).** A task-independent assay based on the Jacobian lens (gates W0–W5) ran on Qwen3-1.7B, Qwen3.5-2B and Qwen3-4B. No (model, layer) cell passed all gates (0/15) [astage@1d72cd3]. A pre-registered repair on fresh material was then classified as an assay failure (0/15 cells assay-valid) [r2@d17691b].
3. **Retrofitted workspace (C16).** A capacity-limited workspace for a frozen Qwen2.5-0.5B stopped at Stage 0 because the model was not competent in the task domain (e.g. subtraction 0.24) [c16s0@1902ce7]. A planted transformer then showed that the design itself was invalid:
   - the content was not computed at the write site before the query (decodability 0.43–0.58) [c16s1@29c86dc];
   - nothing was transported to an untrained consumer (transport CT between −0.017 and −0.002) [c16s1@29c86dc].
4. **Self-model selection theorem (SM).** The proposed theorem reduced to existing selection theorems [nayebi2026selection; richens2024robust]. Counterexamples showed that it does not force a separate self-representation. The pre-registered regret gap was negligible (median 2.1e-4) [sm0@6dde7eb].

No hypothesis about workspaces, metacognition or self-models was confirmed or refuted. Every line stopped at an instrument-validity gate or a novelty check. Our contribution is methodological: twelve lessons (R1–R12) about instrument validity, each tied to a measured failure.

---

## 1. Motivation and scope

**The indicator approach.** The indicator approach derives computational properties from scientific theories of consciousness and asks whether a given AI system has them [butlin2025indicators; chalmers2023llm]. Two theory families shaped this program:
- **global workspace theory:** a capacity-limited stage whose contents are broadcast to many consumers [baars1988; dehaene2017science];
- **higher-order and metacognitive theories:** the system represents its own states or competence [dehaene2017science]. Shea and Frith argue that the two must interact [shea2019workspace].

**Two design decisions came first.**
- **Measure; do not build.** We chose to *measure* indicator signatures by causal intervention, not to build an architecture and assess it by inspection (decision D1). A scaffold can satisfy an indicator checklist without realising the function [goldstein2024case].
- **IIT is excluded.** We excluded integrated information theory as an engineering target, because under that theory software on conventional hardware is irrelevant [findlay2024dissociating] (D2).

**Process discipline.** These rules apply to every line:
- Each experiment has a **pre-run commit** containing its code, material hashes and frozen thresholds, made before any model saw its material.
- Every stage has **binding gates**. When a gate fails, the line stops and reports to the PI. Thresholds are never relaxed after results.
- **Select/confirm splits** have code guards: the confirmation split loads only after a committed FREEZE record and runs once.
- **Claims** are labelled Level 1/2/3, and Level 3 is never asserted.

**Hardware.** Intel Core Ultra 7 255U, 31 GB RAM, no CUDA GPU [hw@f378121].

**Program overview.**

| Line | Question | Pre-run commit(s) | Result commit(s) | Outcome |
|---|---|---|---|---|
| Monitor ladder (Stage 1) | Does a monitor track the system's *own* item-level competence, beyond input familiarity? | `f1e8efb`, `119a3c5`, `b15be7a`, `3539d3d`, `b606db3`, `7742d07` | `6f5ea82`, `8f24e8e`, `2bcef69`, `79b63d2`, `0dc1261`, `2c00545` | Six interventions failed binding gates; no monitor trained |
| C15 workspace assay | Does any small Qwen layer have a selective, broadcast, reportable workspace subspace? | `7e73b5d`; R2 `02201df` | `1d72cd3`; R2 `d17691b`, `25a21f9` | ST-2 (0/15 cells pass); R2 = P4 (assay failure) |
| C16 retrofit | Does a retrofitted capacity-limited workspace transport *content* to untrained consumers? | `aecfe0c` | `1902ce7`, `29c86dc` | S0 STOP; Stage 1 found two structural instrument failures |
| SM theory | Is a self-model *forced* when a self-change must be projected across contexts? | `879e615` | `6dde7eb` | NO-GO: novelty kill criterion K1 fired |

---

## 2. The metacognitive-monitor calibration ladder (brief)

### 2.1 Question

**Two kinds of monitor.**
- A *self-tracking* monitor reads the system's own competence.
- A *world-tracking* monitor reads input familiarity or difficulty.

On natural data the two are observationally equivalent. They come apart only under an intervention that changes the system's correctness while leaving its input tokens identical [pivot@5432243].

**Plan.**
1. Train a small from-scratch transformer "fact store" with complete ground truth.
2. Engineer such an input-invisible competence change.
3. Only then train a monitor and test it with a cross-fitted within-target estimand.

The estimand machinery passed its simulation validation [b1@5432243]. The ladder never reached step 3.

### 2.2 The six attempts

| Cycle | Intervention | Binding gate that failed | Measured value (gate) | Source |
|---|---|---|---|---|
| v2 | Targeted forgetting (T-FORGET) with a matched sham | Sham "fingerprint" classifier AUROC ≤ 0.90 | 0.845 / **0.903** / 0.858 on fresh seeds, while every selectivity gate passed (targets lost 0.66–0.73; sham lost 0.000) | [v2@6f5ea82] |
| v3 | Within-target forgetting: 36 pre-declared cells | Continuous retention; outcome diversity | **0/36** cells eligible; minimum retention load 2.04 (gate 1.0); IQR of post-change margin 0.22–0.83 nats (gate ≥ 1) | [v3@8f24e8e] |
| v4 (B1) | Dual-route store (explicit memory + parametric backup), co-trained | Backup accuracy in [0.30, 0.70] | 0.865 / 0.803 / 0.849 at every route-dropout rate (0.2 / 0.35 / 0.5) | [v4@2bcef69] |
| v4.1 | Gradient-isolated routes | Integrated-store QC ≥ 0.98 | 0.701 (memory-covered facts 0.580) | [v41@79b63d2] |
| v4.2 | Sequential development (parametric first, then memory) | Negative-control neutrality; outcome-tier diversity | Y control changed graded competence by 0.38–0.50 × X's change (gate ≤ 0.10); "retained-strong" tier 0.00 on 3/3 seeds (gate ≥ 0.15) | [v42@0dc1261; b1@5432243] |
| D2 | Self-generated change through continued learning (no engineered edit) | Lost fraction in [0.10, 0.50] | Mean lost 0.749 / 0.901 / 0.963 at lr 3e-4 / 1e-3 / 3e-3 | [d2@2c00545] |

### 2.3 Why it stopped

- **Who stopped it.** The PI's absolute stopping rule closed the dual-route line after v4.2 (D60, D61). D2 then failed its pre-declared store-only kill test, and the PI ended intervention tuning (D67).
- **Where every cycle failed: engineering, not statistics.** All six cycles failed at producing a valid intervention, never at the statistics. Locality, identifiability and dose control all worked in v4.2:
  - Z-category loss 0.000;
  - predictability from pre-state R²_pre ≈ −0.003 [b1@5432243].

  The *conjunction* of gates was unattainable. Two clear mechanisms:
  - **v3:** uniform-target forgetting flattens the answer-distribution tail of untargeted facts. Their margins shrink while their answers survive (0.15–0.26 × D_C against a 0.10 gate) [v3@8f24e8e]. This is consistent with the edit-induced distortions reported for synthetic transformers [nishi2025shattering].
  - **v4.2:** a strong explicit memory makes the intact margin about 3× the largest backup margin compatible with partial backup. A tier defined *relative to* intact competence was therefore unattainable by construction [b1@5432243].
- **D2's change was clean but too large.** The self-generated change was graded and identifiable:
  - IQR(ΔC) 3.72–4.43 nats;
  - R²_pre 0.00–0.02 [d2@2c00545].

  But it was near-catastrophic in magnitude. Accuracy on old facts fell from 1.00 to 0.23–0.27 even at the lowest learning rate [d2@2c00545].
- **Nothing was spent.** No monitor or controller was trained, and the validation and confirmatory seeds were never used [b1@5432243; d2@2c00545].

---

## 3. C15: a task-independent workspace assay on three Qwen models

### 3.1 Design

**Motivation.** C15 was meant to test whether metacognitive information reaches control *through* a workspace-like subspace (D68, D69). That required, first, a (model, layer) with a demonstrably workspace-like subspace on task-independent material: the **A-stage**.

**The workspace subspace S_J.** Built at each band layer from the Jacobian lens [gurnee2026workspace; anthropic_jlens_code]:
- run non-negative gradient pursuit (k = 25) over generic-token lens atoms;
- take a PCA of the reconstructions of generic text;
- set the rank r = min(r₉₀, d/8) [aproto@7e73b5d].

**Models and layers.** Qwen3-1.7B, Qwen3.5-2B and Qwen3-4B, with 15 band layers in total. Checkpoints and lenses were pinned by SHA-256 [aproto@7e73b5d].

**Gates.**

| Gate | What it measures | Criterion |
|---|---|---|
| W0 | Lens prerequisite | Lens agreement with the model at layer L−2 ≥ 0.60; two-hop accuracy ≥ 0.60; intermediate-entity readout ≥ 0.40 |
| W1 | Reportability of injected J-content | — |
| W2 | Cross-function broadcast of J-component swaps | — |
| W3 | Cross-position transport | — |
| W4 | Causal relevance | J ablation vs norm-matched random ablation on two-hop reasoning |
| W5 | Selectivity | SI_iso ≤ 1.0 with CI upper bound ≤ 1.25, i.e. ordinary next-token damage relative to random *at matched functional impact*. SI_norm (matched intervention norm) is always reported, as the PI required (D70) |

[aproto@7e73b5d]

**Controls (⊥).** Each J direction has a control orthogonal to S_J.
- *Matched within ×/÷1.15 on:* layer, position, rank, norm, lens gain, propagation, and the *measured* generic next-token KL.
- *Coverage rule:* if more than 20% of a test's contents had no matched control, the test failed [aproto@7e73b5d].

**Material.** Author-written and task-independent. Each split holds 63 paragraphs, 20 concepts, 20 countries and 74 two-hop items, divided into G_select and G_confirm [aproto@7e73b5d]. G_confirm could be loaded only after a committed FREEZE.

### 3.2 A-stage result: no cell passes (ST-2)

| Model | W0: lens agreement / two-hop / intermediate readout | Gates passed (best layers) | SELECT CPU time |
|---|---|---|---|
| Qwen3-1.7B | 0.674 / **0.581** / **0.023** | W2 only, at layers 10–16 (switch rate 0.32–0.45 vs ⊥ 0.00) | 6.6 h |
| Qwen3.5-2B | **0.545** / 0.622 / **0.043** | W1, W2 and W4 at layers 9 and 13. W2 switch rate 0.92–1.00 vs ⊥ 0.00; W1 hit rate J 0.37–0.64 vs ⊥ 0.13–0.21 | 5.6 h |
| Qwen3-4B | 0.720 / 0.703 / **0.173** | W4 and W5 at layers 11 and 13 | 11.0 h |

Bold marks a failing value. All values: [astage@1d72cd3].

**Verdict.** No (model, layer) combined reportability, broadcast, transport, causal relevance and selectivity. 0 of 15 cells passed W0–W5, so the A-stage stopped (ST-2) [astage@1d72cd3].

**Exploratory observations** (G_select only; never confirmed):
- **Matched-norm and matched-impact comparisons give opposite selectivity verdicts.** At Qwen3-4B layers 11 and 13:
  - ablating the active J atoms impaired two-hop reasoning by 20.3 and 25.7 points, against 1.4 and 0.0 for random ablation;
  - at matched functional impact, J ablation damaged ordinary prediction *less* than random (SI_iso 0.75 and 0.69; CI upper bounds 0.82 and 0.74);
  - at matched norm it damaged *more* (SI_norm 2.22 and 2.60) [astage@1d72cd3].

  The matched-norm direction agrees with public reports that J-space ablation is anti-selective in Qwen3 models [jspacevalidity_repo; jspace4b_repo]. A potent subspace is not thereby non-selective; the verdict depends on the matching rule.
- **W3 looked insensitive, not informative.** Its decoders reached cross-fitted R² ≤ 0 in both arms at every site. Yet W1 showed that injected J-content changed the model's report about 12 tokens later (Qwen3.5-2B hit rate 0.37–0.64, vs 0.12–0.15 without injection) [astage@1d72cd3].

### 3.3 R2: a pre-registered repair, classified as assay failure (P4)

**What changed.**
- A new identity-specific transport test (W3′), with:
  - positive controls PC1–PC3;
  - negative controls NC1 (random-direction sham) and NC2 (label permutation).
- A primary-position rule for the intermediate readout (W0b′).
- Fresh material, G_select2 and G_confirm2.
- A frozen STOP/GO table [r2pre@02201df].

**Null calibration.** Before any model run, simulation showed that W3′'s complete gate has a worst-case false-positive rate of 0.0456 at the frozen thresholds [r2null@7131249].

**Result: P4.** The frozen classifier returned P4 on two independent criteria:
- W3′ was assessable at **no** layer of any model;
- **0/15** cells were assay-valid, against a required ≥ 50% [r2@d17691b].

| Failure (frozen rule) | Extent |
|---|---|
| W3′ matched-⊥ coverage | Failed in 13/15 cells (2–8 of 8 concepts unmatched) |
| PC1 reference reliability | Failed in all 15 cells (fraction of passing sites 0.00–0.29) |
| W1/W2 matching coverage | Failed in 15/15 cells |
| W0a lens agreement, Qwen3.5-2B | 0.539 [0.520, 0.557]: a powered failure |
| Two-hop accuracy (≥ 0.60) | 0.47 / 0.36 / 0.57, against 0.58 / 0.62 / 0.70 on the A-stage material |

Extent values: [r2@d17691b; astage@1d72cd3].

**What still worked.** The W3′ pipeline itself ran cleanly where its controls did not depend on ⊥ coverage:
- PC2 natural-identity transport passed at 0.88–1.00 of sites;
- the NC1 sham gave breadth 0.00–0.02 [r2rep@25a21f9].

**Diagnostics** (descriptive only):
- **The matched controls were material-sensitive.**
  - With S_J construction unchanged, the same dose rule made J injections 3–6× more disruptive on the new material. Example: Qwen3-1.7B layer 8, generic KL 0.085 vs 0.015.
  - The selected dose therefore fell from mostly 2.0 to 0.5.
  - ⊥ coverage collapsed from 85–95% to 10–65% [r2rep@25a21f9].
- **The intermediate entity is readable earlier in the prompt.** Readout rate by position: t₁ 0.64–0.81 > t_d 0.31–0.52 > t₂ 0.09–0.24 [r2rep@25a21f9].

**Cost and closure.** R2 SELECT took about 27 CPU-h (5.9 + 6.4 + 12.1 h) [r2rep@25a21f9]. The PI then closed C15 (D78). Neither G_confirm nor G_confirm2 was ever loaded.

### 3.4 What failed, plainly

Two pre-registered attempts failed to produce a valid lens-defined workspace assay on all three CPU-feasible models. The failing component moved with the material:
- **A-stage:** W3 was uninformative, and W0 failed.
- **R2:** control coverage, reference reliability and task competence failed.

No architectural claim about any of these models is licensed.

---

## 4. C16: a retrofitted capacity-limited workspace (Stage 0/1)

### 4.1 Question and design

**The native integration gap.** In a decoder-only transformer, late-layer content computed at earlier positions can reach the early and middle layers of later positions only through emitted tokens.

**The C16 question.** Can a retrofitted, capacity-limited, *blind* workspace carry **content**, not answer-specific codes, to consumers it was never trained with? C16 was a design to test this [c16memo@437e861].

**Architecture A: the Re-entrant Slot Workspace, fitted to a frozen Qwen2.5-0.5B-Instruct.**
- **Write.** It writes at layer L_w from the producer's positions, before the query cue exists, into a few slots.
- **Read.** One shared reader injects slot content into the read layers of later positions.
- **Initialisation.** Zero-initialised gates make the model bit-identical at the start.

**Task domain (synthetic numbers).**
- A producer sentence, e.g. "{N}'s number is {a}+{b}.", defines a latent value X.
- Consumers then apply a function f to X: copy, successor, +10, parity, magnitude and lookup.
- An untrained consumer **U** (verbalisation) tests generalisation [c16pre@aecfe0c].

**Primary endpoint: content transport.**
- **Definition:** CT = P(answer = f(X′) | content swap from X to X′) − P(answer = f(X′) | norm-matched control).
- **Where it is measured:** on trained pairings, held-out producer–consumer pairings, and U.

**Controls at equal capacity.** Each control shares A's capacity budget.

| Control | Description |
|---|---|
| C1 | Cue-addressed writer |
| C2 | Private lines |
| C3 | Unrestricted re-entry of the Back-Attention type [yu2025backattention] |
| C4 | Self-recurrence |
| C5 | Parameter-matched LoRA |
| C6 | Native transplant |
| C7 | Late read |

[c16memo@437e861]

**Novelty audit.**
- *Scope:* a forward-citation crawl of 14 seed papers returned 805 citing records. 99 were kept by a screening rule fixed before reading [c16pre@aecfe0c].
- *Result:* the exact cell survived, with novelty revised to ≈ 0.5 [c16pre@aecfe0c].
- *Closest precedent:* LM "societies" on the same frozen Qwen2.5-0.5B. They found that restricted visibility yields value-indexed codes and that new-operator zero-shot transfer fails [marincat2026masking; marincat2026portable; marincat2026sixty].
- *Also related:* a causal audit of latent multi-agent channels [zhang2026latentaudit].

**PI amendments before freezing** [c16pre@aecfe0c]:
- **A1. A fair addressed control.** C1 is conditioned on a cue-derived representation (the frozen model's mean residual over the cue text), so it *can* represent U. A's U advantage would not be true by construction.
- **A2. The "capacity window" reformulated.** The peaked capacity × consumer-diversity prediction was not defensible in bits: a capacity lower bound had precedent [resnick2020capacity], but no upper bound did. It was reformulated as a three-account discrimination: format geometry vs information bottleneck vs native copy.
- **A3. A naturalistic extension.** It was specified but deferred until after a strong positive result.

### 4.2 Stage 0 (cached Qwen2.5-0.5B, forward passes only): STOP

| Measure | F1, zero-shot | F2, one-shot primer (pre-declared fallback) | Gate |
|---|---|---|---|
| X decodable at L_w = 8 (add / sub) | 0.675 / 0.622 | 0.761 / 0.706 | ≥ 0.70 |
| Producer competence, copy (add / sub / mul) | 0.41 / 0.29 / 0.98 | **0.55 / 0.24** / 1.00 | ≥ 0.80 |
| Consumers with X in the text: succ / +10 / parity / mag / lookup / **U** | 0.09 / 0.02 / 0.00 / 0.00 / 0.00 / 0.00 | 0.89 / 0.79 / 0.61 / 0.56 / 0.65 / **0.20** | ≥ 0.80 for 4 of 5 pool consumers, and for U |

All values: [c16s0@1902ce7].

**Verdict: STOP.** F2 passes decodability but fails competence, so the frozen rule stops the line and forbids any model or domain change without the PI.

**A second problem: decodability was partly an operand artifact.** For multiplication, X was decodable at 1.00 from layer 3 onward, over only 13 distinct values. Operand identity alone predicts X's digits there [c16s1@29c86dc].

### 4.3 Stage 1: the design fails even on a system built to satisfy it

**The planted system.** A 4-layer transformer (d = 128, 4,000 steps) was trained from scratch to be competent and to have the native gap. It passed every competence gate (V1a–d) [c16s1@29c86dc]. It still exposed two structural failures.

**1. The information is not at the write site before the query (V0).**
- X was only weakly decodable where the producer sentence ends: 0.575 (add) and 0.431 (sub), against a 0.70 gate [c16s1@29c86dc].
- The planted model computes X *lazily*, when the question arrives, not when the statement appears.
- A blind writer therefore reads operands, not content.

**2. Nothing transports to the untrained consumer.**
- The transplant oracle gained at most 0.06 at every layer: the prefix positions do not carry X [c16s1@29c86dc].
- Workspaces trained on this system (3 seeds per arm) learned only small, generic adjustments:

| Arm | Trained-pairing CT | Held-out-pairing CT | CT on untrained consumer U |
|---|---|---|---|
| A (blind, shared) | 0.090–0.099 | 0.099–0.183 | −0.015 to −0.007 |
| C1 (cue-addressed) | 0.120–0.138 | 0.133–0.263 | −0.017 to −0.002 |
| C3 (unrestricted re-entry) | −0.117 to 0.078 | 0.047–0.211 | −0.012 to −0.002 |

Ranges are across 3 seeds per arm: [c16s1@29c86dc].

- A's transport positive control was 0.096, against a gate of ≥ 0.5 [c16s1@29c86dc].
- Training reach was +0.007, against a gate of ≥ 0.20 [c16s1@29c86dc].

**What passed** [c16s0@1902ce7; c16s1@29c86dc]:
- zero-gate identity;
- A's cue invariance;
- negative-control neutrality;
- **V6:** the fair addressed control transferred to paraphrased cues at 1.21× its trained-cue CT (gate ≥ 0.80);
- rank recovery on planted data: exact (4/16/64; bundles 16/64).

**Power simulation (S1a).**
- The complete-gate over-claim false-positive rate was 0.035 and 0.048, with power 0.86 and 0.70 (σ_X 0.3 and 0.6) [c16s1@29c86dc].
- The three-account discrimination identified the geometry and information-bottleneck worlds with P ≥ 0.92. It never confirmed the native-copy world (P = 0.00–0.01), because confirming a null-like account needs three equivalence tests to pass together [c16s1@29c86dc].

**Cost.** The whole S0/S1 sequence took about 3.5 CPU-h (S0 40 min; S1C 2 h 38 min) [c16s1@29c86dc].

### 4.4 What failed, plainly

No workspace hypothesis was tested, and Stage 2 was never run. Two failures stand:
- The cached model was not competent in the chosen domain.
- More fundamentally, the blind-write design assumed that content is computed at the write site before the query, and that the prefix oracle carries it. Both assumptions fail even in a transformer built to satisfy the competence and gap requirements.

---

## 5. SM: a selection theorem for self-models (theory only)

### 5.1 The claim

**Origin.** A theory-only direction search screened 16 first-principles candidates and eliminated 9 outright [moon@ace2c5b]. It recommended **SM**:

> A self-model is *forced* if and only if an agent must project a change in itself across contexts.

**Its parts.**
- **SM-A (extraction/necessity):** low regret on cross-context queries forces a recoverable, context-invariant self-attribution.
- **SM-B (boundary):** within-context queries alone never force self/world attribution.
- **Extended-self corollary:** without consensus evidence, the self cannot be separated from global world factors [sm0pre@879e615].

**Pre-declared kill criteria** [moon@ace2c5b]:

| Criterion | Name |
|---|---|
| K1 | Novelty |
| K2 | Identifiability |
| K3 | Vacuity |
| K4 | Significance |

### 5.2 Reduction to existing theorems (K1 fires)

The reduction was pre-registered in `879e615` [sm0pre@879e615].

| SM claim | Existing result it reduces to |
|---|---|
| SM-A, extraction | Nayebi's Thm 3 (threshold bets recover test probabilities, E[(p̂ − p)²] ≤ 2δ̄ + 1/(4K²)), plus finite-mixture identifiability [nayebi2026selection] |
| SM-A, "must carry" | Nayebi's Cor 4 (regime tracking) with regime = shift type. Aliasing probability ≤ 2δ̄/c(γ), where c(γ) = 4γ/(1 + 2γ) [nayebi2026selection] |
| SM-B, boundary | The zero-witness case of Nayebi's Thm 5 / Cor 5. Nayebi states it in words [nayebi2026selection] |
| Told-shift variant | Richens & Everitt Thm 1–2, with the self parameter as a chance variable [richens2024robust] |
| Action-dependent self-change | Controlled and partially observable world-model results [richens2025general; cifuentes2026partial] |

The bounds are as transcribed in [sm0pre@879e615], §2.1.

**Verdict.** SM is regime tracking or causal-model identification with the latent variable relabelled "self". K1 fired [sm0@6dde7eb].

### 5.3 Counterexamples (K2 fires for a separate self-state)

[sm0pre@879e615]

- **C1: a minimal optimal memory with no self-coordinate.**
  - *Construction:* the memory holds the two per-context log-likelihood ratios (ℓ₁, ℓ₂). It is Bayes-optimal and minimal.
  - *Result:* the self-attribution T is a nonlinear function of both coordinates, so no coordinate, and no linear readout, equals T.
  - *Also:* a raw-history memory (C0) aliases nothing yet has no self-variable.
- **C2: cross-context queries need not force self-attribution.**
  - *Setting:* K = 3 contexts.
  - *Result:* two histories with opposite self-attribution give identical predictions for the unvisited context. What the queries force is "is the target context degraded?", not "did I change?".
- **C3: self and an agent-indexed world factor are observationally equivalent.**
  - *Construction:* a shared competence variable can be relabelled as, e.g., the calibration of the agent's own workbench. The relabelling leaves every history distribution, optimal policy and regret unchanged, even with consensus data from other agents.

### 5.4 Regret gap (K4: PASS-weak)

**Protocol.**
- **Model.** A single-shift model M0: n trials in each of two observed contexts, then an opt-out query in an unvisited context with c ~ U[0, 1].
- **Grid.** 288 cells: K ∈ {3, 6, 12}, ρ ∈ {0.1, 0.25, 0.5, 0.75}, a ∈ {0.75, 0.9}, Δ ∈ {0.1, 0.2, 0.4}, n ∈ {1, 3, 10, 30}.
- **Primary statistic.** R_blind: the utility lost by the better attribution-blind policy, relative to Bayes.
- **Decision rules.** The declared rule fires if the maximum is < 0.02. A robust rule passes if the median is ≥ 0.02. Both were fixed in `879e615` [sm0pre@879e615].
- **Run.** Exact arithmetic, 1.9 s [sm0@6dde7eb].

| Statistic over the grid | Median | Max | Cells ≥ 0.02 |
|---|---|---|---|
| M0 R_blind (primary, 288 cells) | 2.1e-4 | 0.028 | 6/288 (2.1%) |
| M0′ R_blind (independent causes, 192 cells) | 4.6e-4 | 0.0188 | 0/192 |
| M0 fixed-threshold R_blind (favourable to SM) | 1.3e-3 | 0.143 | 15.3% |

All values: [sm0@6dde7eb].

**Reading.**
- All six cells at ≥ 0.02 have K = 3, Δ = 0.4 and n = 30.
- K = 3 is the counterexample-C2 regime. There the Bayes advantage comes from *excluding* the observed contexts, which is world attribution to the target, not self-attribution.
- The maximum falls with K: 0.028, 0.0156 and 0.0082 at K = 3, 6 and 12 [sm0@6dde7eb].

**Label.** PASS-weak (regime-dependent): the declared rule did not fire, and the robust rule failed.

### 5.5 Verdict

**NO-GO** [sm0@6dde7eb]: K1 and K2 fire, K3 is moot, and K4 is weak.

**The strongest surviving claim is a methods remark.**
- Within-context self-knowledge tests cannot, in principle, detect self-models.
- Cross-context transfer tests can reveal predictive attribution, but cannot certify a *self* representation without structural or interventional knowledge of the system.

That bears on how introspection results in LLMs can be read [lindsey2025introspection; song2025privileged; singh2026reality]. It is a clarification, not a discovery.

---

## 6. Lessons R1–R12 as methodological contributions

Each lesson was written down after a measured failure, and later designs were required to satisfy it.

| Code | Lesson | Evidence | Origin |
|---|---|---|---|
| R1 | A self-tracking test needs a competence change that is **input-invisible**: identical tokens before and after | The N1 identification argument | [pivot@5432243] |
| R2 | Require **graded** item-level variation among identically treated items | v3 tail flattening compressed post-change margins to IQR 0.22–0.83 nats [v3@8f24e8e] | [pivot@5432243] |
| R3 | The effect must be **identifiable**: not determined by pre-state or by generic change | Under diffuse susceptibility, simulation produces false "label B" support, so only label A counts (D50, D54) | [pivot@5432243] |
| R4 | Negative controls must be neutral **at the graded level** the gates use, not only in binary accuracy | v2 sham fingerprint AUROC 0.903 [v2@6f5ea82]; v4.2 Y control at 0.38–0.50 × X's graded change with ≤ 0.011 binary loss [b1@5432243] | [pivot@5432243] |
| R5 | Define a replication unit (seed or model) before running | Statistical framework | [pivot@5432243] |
| R6 | Check analytically, before any run, that gates are **jointly satisfiable** | v4.2: the redundancy window and the relative-tier gate were incompatible by construction [b1@5432243] | [pivot@5432243] |
| R7 | **Minimise intervention engineering**: every engineered intervention produced its own artifact | v2, v3, v4, v4.1, v4.2 | [pivot@5432243] |
| R8 | **Reachability**: show that the pre-declared dose lever can actually reach the satisfiable region | D2 lost 0.749–0.963 at every lr; the lever (lr under stop-on-learning) never spanned the [0.10, 0.50] window [d2@2c00545] | [d2@2c00545] |
| R9 | **Eager-computation validity**: a write-before-query channel can carry only content computed before the write window closes. Test decodability on **held-out operand combinations** | Planted V0 0.575 / 0.431 despite full competence; mul decodability 1.00 from operand identity alone [c16s1@29c86dc] | [c16s1@29c86dc] |
| R10 | **Oracle carriage**: validate that the transplant source actually carries the content. Causal precedence is not carriage | Planted oracle gain ≤ 0.06 at every layer [c16s1@29c86dc] | [c16s1@29c86dc] |
| R11 | A grid in units of a measured quantity needs that quantity to be **defined** before scheduling | Planted pilot: 4 of 5 capacity levels collapsed to identical runs when r_X was undefined [c16s1@29c86dc] | [c16s1@29c86dc] |
| R12 | **Confirming a null-like account needs its own power** | Copy-world identification P = 0.00–0.01 even at 8 seeds × 200 items per cell [c16s1@29c86dc] | [c16s1@29c86dc] |

**Two further methodological observations from C15:**
- **Empirically matched controls are material-sensitive.** Coverage fell from 85–95% to 10–65% on fresh material [r2rep@25a21f9].
- **Matched-norm and matched-impact selectivity can disagree in sign:** SI_norm 2.22–2.60 vs SI_iso 0.69–0.75 [astage@1d72cd3].

**Cross-cutting pattern.**
- Across the four lines, every stop came before the primary hypothesis was tested: at a validity or qualification gate (intervention validity, assay validity, model competence) or at a novelty/identifiability check (SM K1, K2).
- The binding gates worked as intended: no stopped line has outcome-dependent analysis to unwind, and no sealed confirmation split was ever opened.

---

## 7. Limitations

**Scale.**
- Everything ran in fp32 on one CPU laptop [hw@f378121]. Models were ≤ 4B parameters.
- Several assays used small item counts: 74 two-hop items per split, and W2 counts ≤ 20 [astage@1d72cd3].
- Larger models might pass gates these models failed. Published J-lens results improve with scale [astage@1d72cd3].

**Toy systems.** The Stage-1 fact stores and the C16 planted transformer are synthetic. Their failures show instrument and design problems; they say nothing about pretrained LMs.

**Prior-guess thresholds.**
- Some thresholds were prior guesses, e.g. W0b's 0.40 readout gate and W1's 0.30 hit-rate gate [astage@1d72cd3].
- Strict gates may have stopped lines that a better-calibrated assay would have continued. By design we did not repair assays after observing their outcomes, except through the pre-registered R2.

**Unconfirmed findings.** The exploratory C15 findings (§3.2) come from the selection split only. G_confirm and G_confirm2 were never used.

**Literature depth.**
- Coverage was uneven. Many entries were verified at abstract level, and some are flagged `partial` in the database.
- The full-text reading of Nayebi 2026 was done through an automated summariser [sm0pre@879e615].
- The SM reduction is a written argument, not a machine-checked proof.

**Speed and oversight.**
- The program ran in eight days, from the first commit `5ff787a` (2026-10-01) to `6dde7eb` (2026-10-08). Fast stopping rules favour early termination over persistence.
- There was no external review. One PI made every decision, and all implementation and analysis was done by an AI system (§8). Errors in code or reasoning could go uncaught. The safeguards were unit tests, guards and pre-run commits, not independent replication.

**Out of scope.** No result here bears on phenomenal consciousness, moral status or welfare.

---

## 8. Contributions

- **Principal investigator** (Varchas Yogesh Hebbale): set the research direction; approved, amended or rejected each design; set the claims discipline, constraints and stopping rules; and made every go/no-go decision recorded in `research/logs/decisions.md`.
- **Claude Opus 5.5** (Anthropic), working as an AI research assistant:
  - did the literature review and maintained the literature database;
  - wrote all code and tests;
  - ran the experiments and performed the analyses;
  - drafted the memos, pre-registrations, reports and this technical report.

---

## 9. Reproducibility

See the repository `README.md` for the environment, per-experiment commands and the commit map. Re-run each experiment at its result commit: the runners verify frozen hashes and clean trees.

---

## References

Keys refer to `research/literature/literature_db.json`.

- [anthropic_jlens_code] Anthropic (2026). *jacobian-lens: companion code for the global workspace interpretability paper.* GitHub, github.com/anthropics/jacobian-lens.
- [baars1988] Baars, B. J. (1988). *A Cognitive Theory of Consciousness.* Cambridge University Press.
- [butlin2025indicators] Butlin, P., Long, R., Bayne, T., Bengio, Y., Birch, J., Chalmers, D., et al. (2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences.* doi:10.1016/j.tics.2025.10.011.
- [chalmers2023llm] Chalmers, D. J. (2023). Could a Large Language Model be Conscious? arXiv:2303.07103.
- [cifuentes2026partial] Cifuentes, S. (2026). General Agents Contain World Models, even under Partial Observability and Stochasticity. arXiv:2602.03146.
- [dehaene2017science] Dehaene, S., Lau, H., & Kouider, S. (2017). What is consciousness, and could machines have it? *Science.* doi:10.1126/science.aan8871.
- [findlay2024dissociating] Findlay, G., Marshall, W., Albantakis, L., et al. (2024). Dissociating Artificial Intelligence from Artificial Consciousness. arXiv:2412.04571.
- [goldstein2024case] Goldstein, S., & Kirk-Giannini, C. D. (2024). A Case for AI Consciousness: Language Agents and Global Workspace Theory. arXiv:2410.11407.
- [gurnee2026workspace] Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026). Verbalizable Representations Form a Global Workspace in Language Models. Transformer Circuits Thread; arXiv:2607.15495.
- [jspace4b_repo] pgrindehollevik-harvard (2026). *jspace-4b: Pre-registered J-space ablation study on Qwen3-4B.* GitHub.
- [jspacevalidity_repo] pgrindehollevik-harvard (2026). *Worse Than Random: J-Space Ablation Is Anti-Selective at Every Measured Qwen3 Scale.* GitHub.
- [lindsey2025introspection] Lindsey, J. (2025). Emergent Introspective Awareness in Large Language Models. Transformer Circuits Thread; arXiv:2601.01828.
- [marincat2026masking] Marincat, N. (2026). What You Can't See Is What You Learn: Slot-Selective Evidence Masking Favors Compositional Generalization in Shared-Genome Language-Model Societies. arXiv:2608.20054.
- [marincat2026portable] Marincat, N. (2026). Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells. arXiv:2609.11365.
- [marincat2026sixty] Marincat, N. (2026). What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization. arXiv:2609.17637.
- [nayebi2026selection] Nayebi, A. (2026). What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty. UAI 2026; arXiv:2603.02491.
- [nishi2025shattering] Nishi, K., Ramesh, R., Okawa, M., Khona, M., Tanaka, H., & Lubana, E. S. (2025). Representation Shattering in Transformers: A Synthetic Study with Knowledge Editing. ICML 2025; arXiv:2410.17194.
- [resnick2020capacity] Resnick, C., Gupta, A., Foerster, J., Dai, A. M., & Cho, K. (2020). Capacity, Bandwidth, and Compositionality in Emergent Language Learning. AAMAS 2020; arXiv:1910.11424.
- [richens2024robust] Richens, J., & Everitt, T. (2024). Robust agents learn causal world models. ICLR 2024; arXiv:2402.10877.
- [richens2025general] Richens, J., Everitt, T., & Abel, D. (2025). General agents need world models. ICML 2025; arXiv:2506.01622.
- [shea2019workspace] Shea, N., & Frith, C. D. (2019). The Global Workspace Needs Metacognition. *Trends in Cognitive Sciences* 23(7). doi:10.1016/j.tics.2019.04.007.
- [singh2026reality] Singh, S., Linzen, T., & Ravfogel, S. (2026). Can LLMs Introspect? A Reality Check. COLM 2026; arXiv:2605.26242.
- [song2025privileged] Song, S., Lederman, H., Hu, J., & Mahowald, K. (2025). Privileged Self-Access Matters for Introspection in AI. arXiv:2508.14802.
- [yu2025backattention] Yu, Z., Belinkov, Y., & Ananiadou, S. (2025). Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models. EMNLP 2025; arXiv:2502.10835.
- [zhang2026latentaudit] Zhang, H., & Emu, M. (2026). Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM. arXiv:2607.26773.

---

## Appendix A. Source key

Paths are relative to the repository root.

| Tag | File(s) | Commit |
|---|---|---|
| hw | `research/logs/decisions.md` (D12: hardware and the $0 constraint) | `f378121` |
| v2 | `research/experiments/stage1/calibration_stop_report_C4.md`; `research/results/raw/calibration/calibration_results.json` | `6f5ea82` |
| v3 | `research/experiments/stage1/calibration_stop_report_v3_C4dev.md`; `research/results/raw/calibration/calibration_results.json` (key `v3`) | `8f24e8e` |
| v4 | `research/experiments/stage1/calibration_stop_report_v4_F1.md`; `research/results/raw/calibration/calibration_results_v4.json` | `2bcef69` |
| v41 | `research/experiments/stage1/calibration_stop_report_v41_F0.md`; `research/results/raw/calibration/calibration_results_v41.json` | `79b63d2` |
| v42 | `research/results/raw/calibration/calibration_results_v42.json` | `0dc1261` |
| b1 | `research/experiments/stage1/B1_final_report.md` | `5432243` |
| pivot | `research/memo/stage1_pivot_memo.md` | `5432243` |
| d2 | `research/experiments/stage1/d2_kill_report.md`; `research/results/raw/calibration/d2_results.json` | `2c00545` |
| aproto | `research/experiments/c15/astage_protocol.md` | `7e73b5d` |
| astage | `research/experiments/c15/astage_report.md`; `research/results/raw/c15a/astage_tables.md`. Raw: `select_qwen3-1.7b.json` and `select_qwen3.5-2b.json` (`8823228`), `select_qwen3-4b.json` (`e69d2b7`) | `1d72cd3` |
| r2null | `research/memo/c15r2_power/r2_null_calibration.json` | `7131249` |
| r2pre | `research/memo/c15r2_preregistration_FROZEN.md`; `research/memo/c15r2_thresholds_FROZEN.json` | `02201df` |
| r2 | `research/results/raw/c15r2/r2_tables.md`, `classify2.json`, `verdict2.json`, `select2_*.json` | `d17691b` |
| r2rep | `research/experiments/c15r2/r2_report.md` | `25a21f9` |
| c16memo | `research/memo/c16_novelty_design_decision_memo.md` | `437e861` |
| c16pre | `research/memo/c16_s0s1_preregistration_FROZEN.md`; `research/memo/c16_s0s1_thresholds_FROZEN.json`; `research/literature/c16_audit/audit_report.md` | `aecfe0c` |
| c16s0 | `research/results/raw/c16/s0_F1.json`, `s0_F2.json`, `s0_final.json`, `s1b.json` | `1902ce7` |
| c16s1 | `research/results/raw/c16/s1c.json`, `s1a.json`, `verdict_s0s1.json`; `research/experiments/c16/s0s1_report.md` | `29c86dc` |
| moon | `research/memo/moonshot_research_decision_memo.md` | `ace2c5b` |
| sm0pre | `research/memo/sm0_proof_reduction_memo.md` (§1–8); `research/memo/sm0_regret_gap/sm0_regret_gap.py` | `879e615` |
| sm0 | `research/memo/sm0_proof_reduction_memo.md` (§7.7, §9, §10); `research/memo/sm0_regret_gap/sm0_regret_gap_results.json` | `6dde7eb` |
