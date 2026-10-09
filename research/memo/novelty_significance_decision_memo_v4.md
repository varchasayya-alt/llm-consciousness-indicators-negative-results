# NOVELTY + SIGNIFICANCE DECISION MEMO v4: broadcast-gated metacognitive control

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Supersedes | `novelty_decision_document_v3.md` §4–§5 (C3 is no longer the recommended project; it is subsumed) |
| Status | For PI decision. **No code, no downloads, no model run** |
| Gates applied | Novelty, **significance (developmental leverage)**, rigour, $0-feasibility, all as hard gates |
| Claims discipline | Every outcome below concerns *functional* organisation: access-like broadcast, metacognitive monitoring and integrated control. **None bears on phenomenal consciousness, sentience or subjective experience** |

## 0. Verdict in one paragraph

- **C3 is novel but, on its own, only significance level 2** (necessity). It is demoted.
- **The synthesis the PI asked about clears both gates:** must metacognitive information be globally broadcast to drive adaptive control, and does routing it into the workspace create that control? Call it **C15**. It is a broadcast-gated metacognitive control test, built around the three levels **F (information) / A (global access) / C (control)**.
- It admits a necessity test, a route-specific sufficiency (rescue) test, and a preregistered **content × route (× controller)** interaction. That gives significance ladder level **4**, with a defined path to 5.
- **Its closest competitor supplies the motivating observation but not the result.** Kawada & Kellis (2026) show the F ⇏ C phenomenon, and locate the inert verification state *outside* the workspace in Llama and Qwen. They explicitly did **not** attempt to route it into the workspace.
- **Generic sufficiency is already published:** steering uncertainty improves abstention. So C15's novelty rests on the **route-specific** necessity, sufficiency and interaction. That is also exactly what carries the engineering lesson.

---

## 1. Final scientific question

> **Must a language model's internal error/confidence information be broadcast through its global workspace in order to drive adaptive control? And does routing that information into the workspace, and not elsewhere, create calibrated self-regulation where local representation alone fails?**

## 2. Targeted consciousness-associated function

**Globally available metacognitive information for adaptive self-regulation.** That is the integration of a monitoring signal (C2-type, "knowing whether one knows") into access-like global availability (C1-type) so that it can steer behaviour: abstain, seek information, or reject a candidate.

Formally, at three levels:
- **F:** correctness/error information exists locally (it is decodable).
- **A:** that information is present in the globally broadcast workspace subspace.
- **C:** behaviour is adaptively governed by it.

The target relation is

> **F ⇏ C,  F + A ⇒ C,** and possibly **F + A + controller ⇒ C**.

## 3. Why this matters for developing consciousness-relevant architecture

- **Workspace and metacognition are the two dominant functional accounts of conscious access.** They have been treated as rival or separable (Shea & Frith 2019), and AI indicator frameworks score them as separate indicators (Butlin et al. 2026).
- **A builder needs to know whether a monitoring signal must be broadcast to be usable.** If it must, then adding a monitor is not enough: monitor outputs must be routed into the broadcast medium that controllers read. If it need not, workspace and metacognition are independent design modules.
- **That is a construction principle, not an indicator check.**
- **The phenomenon is real and unexplained in current models.** Decodable verification states are causally inert in Llama and Qwen (Kawada & Kellis), and in looped models a predictive signal written back does not produce control (proto-introspection 2026). C15 tests whether **lack of broadcast access** explains that F ⇏ C gap.

## 4. Theory connection (and whether it discriminates)

| Account | Architectural prediction under C15 | Discriminating outcome |
|---|---|---|
| **GNW + Shea & Frith** (metacognitive tags must be broadcast; control reads the workspace) | Native control depends on the workspace component of the error signal. Inert information is outside the workspace. Routing into the workspace rescues control; equal-norm routing elsewhere does not | Outcome A |
| **Higher-order theories** (HOT / HOSS / PRM: a higher-order representation of first-order states suffices; broadcast not required) | A non-workspace higher-order confidence representation can drive control. Blocking the workspace component spares control; non-workspace routing also rescues | Outcome B or C |
| **First-order / "no metacognition" accounts** (Carruthers; output-entropy readout) | Control tracks the first-order output distribution. Manipulating confidence components that leave the output distribution unchanged leaves control unchanged | Outcome B with output-entropy dependence (tested by conditioning on the first-order output distribution) |
| **Recurrent-processing accounts** | Not directly tested: single-pass transformers. A looped-model extension is listed as future work (CPU-infeasible now) | — |
| **Attention schema / self-model** | Not addressed | — |

**This genuinely discriminates GNW-style from HOT-style routing predictions** at the level of function. It does not merely borrow their terms.

## 5. Novelty and significance confidence

- **NOVELTY CONFIDENCE: MEDIUM.**
  - The route-specific necessity, sufficiency and interaction are not found in any of the closest works read in full (§13).
  - Scoop risk is real. Kawada & Kellis's natural follow-up is a rescue, and the J-lens community is active.
- **SIGNIFICANCE CONFIDENCE: MEDIUM.**
  - A positive result with rescue and interaction reaches ladder level 4 and changes an architecture roadmap.
  - The main threat is assay validity at CPU scale: a lens-defined workspace in ≤ 4B models (§15). A pre-declared manipulation check converts that risk into a stop rule, not an ambiguous result.

## 6–9. Design: necessity, sufficiency/rescue, factorial integration, control endpoint

### Models and baseline

- Models with published pre-fitted J-lenses: Qwen3-4B (also in Kawada & Kellis's set), and Qwen3.5-2B in base and post-trained form.
- Replication in a second family: Gemma-3-1B/4B (pt/it, gated) or Llama-3.1-8B(-it) (gated, CPU-slow, small N).
- **Baseline per item:**
  - the first-order answer from the intact model, which fixes correctness;
  - a cross-fitted correctness probe at the control-decision position;
  - its decomposition into a **J (workspace) component** and a **⊥ (non-workspace) component**, following Gurnee et al.'s concept-vector decomposition.

### 10. Control endpoint (cleanest CPU-feasible)

**Primary: ANSWER/PASS (or LOOKUP) choice.**
- The model sees the question with an explicit payoff (+1 correct, −k wrong, 0 pass). The decision is read from the next-token log-odds, in one forward pass.
- **DVs:** control discrimination (AUROC of pass-propensity for incorrect vs correct items) and realised expected payoff. Pass *rate* alone is not a DV: a bias is not adaptive control.
- Because the first-order answer is elicited separately with the intact model, **manipulations applied during the decision pass cannot change the first-order answer.**

**Secondary: rejecting an invalid external candidate** (the Kawada & Kellis paradigm).
- Accept/reject a tool-supplied answer that is valid or invalid. The baseline failure is 93–100% acceptance despite internal verification. That makes it the natural rescue target and Shea & Frith's own case: confidence-weighting of broadcast contents.

**Considered but not primary:** extra computation, revision, error correction. They need generation, so they are slower on CPU and harder to score. They remain a later generalisation test.

### 7. Necessity experiment (block access, keep the information)

At the workspace-band layers and the decision position, project out the **J-component of the correctness subspace**.

**Verification that the block is specific (each pre-declared):**
1. **F preserved:** correctness is still decodable from the ⊥ component (cross-fitted AUROC ≥ 0.9 × baseline).
2. **First-order answer unchanged:** by construction.
3. **Generic proposition evaluation preserved:** a TRUE/FALSE judgement of externally stated facts the model knows. This separates *self*-monitoring from general truth evaluation.
4. **Workspace identifiability:** a selective-ablation check (J-ablation impairs two-hop more than automatic continuation, versus a matched-norm random-subspace control).

**Contrasts, as a 2 × 2 of content × route:**
- equal-norm removal of the **⊥ correctness component**;
- equal-norm removal of a **non-correctness J direction**.

**Necessity result:** control discrimination collapses only under J-correctness removal, while the four checks pass.

### 8. Sufficiency / rescue experiment (route the information in)

1. In the condition where control is weak or absent but F is present (natively inert models or items, or after the necessity block), read item-specific error information from the ⊥ subspace with a cross-fitted probe.
2. **Write it into the workspace**: the J-projection of the native correctness direction, or the J-lens vector of an uncertainty token, scaled by the item's predicted error.

**Controls:**
- the same item-specific signal written into an **equal-norm ⊥ direction** (route control);
- the same J write with **item-permuted** magnitudes (content control: same marginal pass bias, no information);
- an injection-strength grid whose range is shown in S0 to span no effect to saturation (lesson R8, reachability).

**Rescue result:** control discrimination and payoff increase only for true content written into J.

### 9. Factorial / integration test (pre-registered interaction)

**Content (true item signal vs permuted) × Route (J vs ⊥) × Controller (post-trained vs base model of the same family).**

| Cell | Hypothesis C15 prediction |
|---|---|
| Neither (permuted, ⊥) | No adaptive control |
| Content only (true, ⊥) | No or weak control (F ⇏ C) |
| Route only (permuted, J) | Bias without discrimination |
| Content + route (true, J) | **Adaptive control**: the interaction term |
| … × base model | If no learned controller reads the workspace: no control even in the true-J cell, i.e. **F + A ⇏ C without a controller** |

**Primary statistic:** the content × route interaction on control AUROC (item-paired, entity-clustered bootstrap; models as replication units).

**Secondary:** the three-way interaction with the controller. This is the developmental-integration factor: post-training as the acquisition of a controller that reads the workspace.

## 11. Exact positive-result architectural implication (Outcome A)

> "Representing uncertainty is not sufficient for self-regulation. An error/confidence signal drives adaptive control only when it is written into the globally broadcast subspace that downstream control reads, and only in a system that has learned a controller reading that subspace. Consciousness-relevant architectures should therefore include an explicit **monitor → workspace routing pathway plus a workspace-reading controller**, rather than a stand-alone monitor."

**Concrete design change:** add or route confidence and error estimates into the broadcast medium. Evaluate monitors by their *effect when broadcast*, not by decodability.

## 12. Exact negative-result architectural implications

| Outcome | Result | Implication |
|---|---|---|
| **B** | Control survives the J-block, and ⊥ routing rescues as well as J routing | Metacognitive control is **workspace-independent**. Global workspace and metacognition are separable design modules, so architectures should **not assume one yields the other**. A dedicated higher-order monitor → controller link suffices (HOT-consistent) |
| **C** | Information is inert locally, but **any** routing to the controller rescues (J ≈ ⊥) | The bottleneck is **routing to control, not broadcast specifically**. Design principle: direct monitor–controller coupling; the workspace is not special for metacognitive control |
| **D** | The block also destroys generic evaluation or first-order use, or the workspace is not identifiable at this scale | **Assay insufficient.** Stop under the pre-declared rule; methods report. This is the only non-informative outcome, and it is caught *before* interpretation |
| **E** | Nothing rescues control, even in post-trained models | Inference-time routing cannot create control. Integration must be **developmental (training-time)**. That points to the constructive follow-up (C12′), as also suggested by the proto-introspection paper's "training-time integration" conclusion |

## 13. Closest competing literature (full text or primary page read)

| Work | What it establishes | What it does not do |
|---|---|---|
| **Kawada & Kellis 2026** (Evidence Integration) | **F ⇏ C:** verification decodable but causally inert. In Llama/Qwen, "the verbalizable workspace holds none of the causal arbitration state"; in Gemma the direction moves decisions | **No rescue or routing**; no necessity block; no content × route design; no abstention endpoint |
| **Kumaran et al. 2026** (*NMI*) | Steering a confidence vector at the pre-answer token shifts abstention | No route (workspace vs other) contrast; accuracy not held fixed; no workspace |
| **PANL 2026** (error detection and correction) | A post-answer confidence state is sufficient but not necessary (redundant) for error detection | No workspace; no control routing |
| **Closing the confidence–faithfulness gap 2026** | Orthogonal accuracy vs verbal-confidence subspaces; a probe → steer pipeline improves calibration | Verbal confidence, not control; no workspace route contrast |
| **Verbal-uncertainty feature 2025; TRAPSBench 2026; CORAL 2026** | Injecting uncertainty reduces hallucination or induces abstention | **Generic sufficiency, not route-specific** |
| **Proto-introspection (looped) 2026** | Readout without control: predictive directions written back do not produce success | Write-back not targeted at the workspace; no route contrast |
| **Gurnee et al. 2026** (J-space) | J vs non-J components differ hugely for concept swaps (59% vs 5%); ablation selectivity | No metacognition or control under ablation or routing |
| **LAP 2026; inverted steering vectors 2026** | Output-aligned directions steer better; detection and control directions can dissociate | Not metacognitive control; no workspace routing |
| **Shea & Frith 2019** | Theory: broadcast representations need metacognitive tags | No machine test |
| **Phua 2026** (toy agents) | Synthetic blindsight via self-model lesion | Toy only; no routing; no factorial |

## 14. Does any existing paper already supply the necessity or sufficiency result?

- **Necessity** (workspace access needed for metacognitive control): **no.** Kawada & Kellis supply the *correlational-plus-inertness* observation (information outside the workspace is inert), which is supportive but not a necessity test.
- **Generic sufficiency** (injecting uncertainty changes abstention or hallucination): **yes, already established** (Kumaran; the verbal-uncertainty feature; closing-the-gap; TRAPSBench). **C15 does not claim it.**
- **Route-specific sufficiency** (J-routing rescues and ⊥-routing does not) and the **content × route (× controller) interaction**: **not found.** These are C15's novel claims.

## 15. Zero-budget feasibility

- **Pre-fitted J-lenses** exist (neuronpedia/jacobian-lens) for Qwen3-4B, Qwen3.5-2B and Qwen3.5-2B-pt, Gemma-3-1B/4B (pt/it), Gemma-4-E4B and Llama-3.1-8B(-it).
- **Computation:** each manipulation is a projection or addition in the residual stream during *one* decision forward pass. Probes and routers are cross-fitted linear maps (closed-form ridge).
- **Estimate per model:** about 1,500 items × (1 answer generation + ~10 decision-pass conditions + checks). That is about 20k short forward passes:
  - 2B: about 1–2 s each → about 6–12 h;
  - 4B: about 3–5 s each → about 1–1.5 days.
- **RAM:** ≤ 16 GB (bf16/fp32 for 2–4B).
- **The 8B replication** is feasible only at reduced N. Runs must be detached processes (the background-task cap is 2 h).
- **Risks (each a pre-declared S0 or manipulation gate):**
  1. A lens-defined workspace at ≤ 4B may be weak: open models showed much lower capacity estimates, and one replication found mixed signatures. → identifiability gate.
  2. Small-model baseline control may be at floor. → that is acceptable for the *rescue* arm, but the *necessity* arm needs a native-control model with AUROC ≥ 0.65 (Qwen3-4B post-trained is the candidate). → baseline gate.
  3. Injected uncertainty-token vectors act partly as semantic cues. This is consistent with a functional broadcast account and disclosed.
  4. Inference-time routing is not an architecture. → the architectural inference is stated as a design hypothesis, and the constructive follow-up (C12′) tests it by training.
- **Downloads needed (each needs explicit PI approval):**
  - Qwen3-4B (about 8 GB);
  - Qwen3.5-2B and Qwen3.5-2B-pt (about 4–5 GB each);
  - their lens folders (size checked first);
  - a closed-book QA subset (PopQA, about 20 MB);
  - optional: the Gemma or Llama licences (gated; accepted by the PI) plus their lenses.

## 16. Re-ranked landscape under the NEW scoring

Each dimension is scored 0–5. "Ladder" is the highest significance level the design can reach.

| Candidate | Novelty | Theory centrality | Necessity test | Sufficiency / rescue | Integration / synergy | Developmental leverage | Roadmap change | Generalisation | Rigour | CPU | **Σ/50** | **Ladder** | Novelty conf. | **Significance conf.** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C15** broadcast-gated metacognitive control (C3 + C13 + C11 synthesis) | 4 | 5 | 4 | 4 | 5 | 4 | 5 | 3 | 4 | 3 | **41** | **4** (5 if it generalises + C12′) | MEDIUM | **MEDIUM** |
| C12′ constructive integration: train small models with vs without a monitor → workspace route (architecture is the hypothesis) | 3 | 4 | 5 | 5 | 5 | 4 | 3 | 1 | 3 | 5 | 38 | 4–5 (toy) | MEDIUM | LOW-MEDIUM (external validity; B1-type degrees of freedom) |
| C3 workspace-dependence of metacognitive report | 4 | 5 | 4 | 1 | 3 | 2 | 3 | 3 | 4 | 4 | 33 | 2 | MEDIUM | LOW-MEDIUM → **subsumed** |
| C11 post-training integration (standalone, 7B ladder) | 4 | 4 | 3 | 2 | 4 | 4 | 4 | 3 | 3 | 1 | 32 | 3 | MEDIUM | MEDIUM (→ folded into C15 as the controller factor) |
| C13 control routing alone | 3 | 4 | 2 | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 32 | 3 | MEDIUM-LOW | MEDIUM-LOW → **subsumed** |
| C9 recurrence dose (loop count as IV) | 4 | 5 | 4 | 2 | 3 | 4 | 4 | 2 | 3 | 0 | 31 | 2–3 | MEDIUM | MEDIUM, but **CPU-infeasible** |
| C4 causal ignition → report | 3 | 5 | 3 | 3 | 1 | 2 | 2 | 3 | 3 | 3 | 28 | 3 | MEDIUM-LOW | MEDIUM-LOW |
| C7 self-model × control | 4 | 3 | 2 | 2 | 3 | 2 | 2 | 2 | 2 | 4 | 26 | 1–2 | MEDIUM | LOW |
| C5 workspace bottleneck | 2 | 4 | 3 | 1 | 2 | 2 | 2 | 3 | 3 | 4 | 26 | 2 | LOW-MEDIUM | LOW |
| C1 D3/P1\* lesions | 1 | 2 | 3 | 1 | 1 | 1 | 1 | 3 | 4 | 3 | 20 | 2 | LOW + UNRESOLVED | LOW |

**Rejected for low significance despite novelty:**
- **C3 alone:** ladder 2; it describes a dependence without a construction lesson.
- **C7:** weak identification.
- **C8, C14** (v3): ladder 0–1.

**Top 3 under the new scoring:**
1. **C15.** Recommended.
2. **C12′.** The constructive follow-up, *only after* C15. If C15 finds route-specific necessity and sufficiency, a pre-registered architecture factorial (monitor present/absent × broadcast route present/absent, compute-matched) tests whether *building* the route *creates* calibrated control. That would be a level-5 developmental principle. Alone it fails the external-validity bar.
3. **C4.** Theory-central, but ladder 3, with low developmental leverage and the highest scoop risk.

## 17. Recommended project

**C15: Broadcast-gated metacognitive control.**

**One-sentence contribution:** "We show that a language model's internal error information [does / does not] need to enter its global workspace to govern adaptive control. Blocking its workspace component while preserving the information removes calibrated abstention, and routing the same information into the workspace (but not elsewhere) restores it. This is a causal test of whether metacognition must be broadcast to be used: the architectural claim that separates global-workspace and higher-order accounts of conscious access."

**What we would build differently afterwards:**
- **If A:** monitor → workspace routing plus a workspace-reading controller as a core module (then C12′).
- **If B or C:** a direct higher-order monitor → controller link; stop treating the workspace as the route to metacognition.
- **If E:** move integration into training (C12′).

**Minimal decisive experiment, in order (each step gated; not an implementation plan):**

| Step | Content | Gates |
|---|---|---|
| **S0** | Satisfiability and **reachability** note (R6, R8) | — |
| **S1** (Qwen3-4B post-trained) | Baseline checks | F-decodability AUROC ≥ 0.75; native control AUROC ≥ 0.65; workspace identifiability (selective J-ablation vs matched control) |
| **S2** | Necessity 2 × 2 (content × route block) | The four specificity checks |
| **S3** | Rescue 2 × 2 (content × route write), with a strength grid proven in S0 to span its range | — |
| **S4** | Controller factor: Qwen3.5-2B base vs post-trained, repeating S3 | — |
| **S5** (only if S2–S4 are decisive) | Second family (Gemma or Llama, gated) and the second endpoint (invalid-candidate rejection) | — |

**Approvals needed:**
1. The scientific question (C15).
2. Writing and committing a pre-registered C15 protocol (S0 note, gates, seeds in a fresh namespace) before any model is loaded.
3. The downloads listed in §15.
4. Optional: accept the Gemma or Llama licences yourself.

**Open items:**
- Cohen & de Melo: still **NOVELTY UNRESOLVED**, but it bears on D3, not C15.
- The Dehaene–Naccache commentary text was not machine-readable here. It reportedly lists self-monitoring (C2) as not yet shown; that would *support* C15's relevance, not pre-empt it.
