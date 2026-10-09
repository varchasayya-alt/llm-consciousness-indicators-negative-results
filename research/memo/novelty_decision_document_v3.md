# NOVELTY DECISION DOCUMENT v3: choosing the next Stage-1 question (novelty as a hard gate)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | For PI decision. **No code, no downloads, no experiment run.** D3 not approved and not implemented |
| Scope | A fresh novelty search across consciousness-related capacities in AI (literature through early October 2026), a missing-cell matrix, 14 candidate programs, scoring, top 3 and a recommended #1 with a minimal decisive experiment |
| Preserved, not presupposed | N1 machinery, within-target estimand, simulation framework, B1 and D2 negative results, process safeguards. **Used as tools, not as the research question** |

## Method and limits

**Searches** covered: GW/J-space, ignition, recurrence, dual-task, metacognition, introspection, self-models, attention schema, HOT/reality monitoring, temporal continuity, error monitoring and control, developmental emergence, modular integration and merging, indicators, toy-agent theory testing.

**Full-text reading.** For each close competitor I read the full text through arXiv HTML or the repository page, using structured questions: what exactly was manipulated, measured, identified and concluded. The arXiv-HTML reading was done through a summarising fetch tool, so key claims are quoted but this is not line-by-line reading.

**NOVELTY UNRESOLVED items:**
- Cohen & de Melo 2026 (OpenReview verification wall);
- the Dehaene–Naccache external commentary PDF (not machine-readable here);
- several 2026 author lists.

---

## 1. The 2026 landscape (what is now crowded, what is not)

1. **Global workspace in LLMs is established and heavily mined.**
   - **Gurnee et al. 2026 (J-space)** tested, among other things:
     - verbalisability and injection reportability;
     - directed modulation (focus/ignore);
     - causal swaps of intermediate concepts;
     - cross-function broadcast;
     - selectivity of ablation (reasoning impaired, classification spared);
     - ignition-like switching under concept mixing;
     - capacity (about 25 concepts) and persistence (autocorrelation);
     - concurrent covert tasks (interleaving, A.17);
     - base vs post-trained differences ("Assistant POV");
     - self-monitoring tokens ("BUT", "damn");
     - training-implanted reflections whose ablation reverts behaviour;
     - three model sizes.
   - **Their stated open points:** recurrence; whether competition is *sharp* ignition; specialised processors; "the dissociation between … conscious access and … selfhood"; the single-token lens.
   - **Open-model follow-ups:**
     - looped models: the workspace survives recurrence, but loop count is not an independent variable [loopedjlens2026];
     - probe-only Ignition Index [ignitionindex2026];
     - JGateBench: entry is by attention transport, not a gate [innerj_repo];
     - an OLMo-3 replication: the workspace predicts the model's own errors observationally [m9h_jlens_repo];
     - a preregistered Gemma-4 error-monitor study: answer-identity confound, failed prospective transfer [jspace_gemma_repo].
   - **Pre-fitted J-lenses for 38 open models** are published (neuronpedia/jacobian-lens), **including small ones:** Qwen3.5-0.8B/2B (+pt), Gemma-3-270M/1B (+it), Gemma-2-2B (+it), Qwen3-1.7B, Pythia-70M. Workspace experiments are now CPU-feasible at $0.
2. **Introspection and metacognition are crowded at the behavioural, probing and "steer-then-report" levels:**
   - Lindsey; Singh et al.; Song et al.; Binder et al.; Ferrara et al.; Ackerman; Blandfort & Pawar; Ashuach et al.; Yax et al.;
   - mechanistic introspection-detection circuits [mechintrospect2026];
   - causal confidence → abstention [kumaran2026causal];
   - post-answer confidence sufficient (not necessary) for error detection [pangl2026errors];
   - orthogonal accuracy vs verbal-confidence subspaces [closinggap2026].
3. **Monitoring → control has been examined, with negative results at the boundary:**
   - "readout-control boundary" in a looped model [protointrospection2026];
   - verification states decodable but causally inert for answers [kawada2026integration];
   - steering confidence does control abstention [kumaran2026causal].
4. **Self-models and valence:** assistant/persona axes, a self-referential direction, a self-directed "pain axis" with steering and costly relief choices [painaxis2026].
5. **Methods and indicators:**
   - Butlin et al. 2026 (*TiCS*) indicator framework [butlin2026tics];
   - the Cross-Substrate Access Assay [crosssubstrate2026] specifies what an indicator test must declare, plus an ignition estimand. **This partly pre-empts D1's checklist contribution.** Its model-side ignition test is designed but not run.
6. **Integration:**
   - merging cannot reach a superadditive capability of joint training (LoRA) [emergentmerging2026];
   - toy agents with single ablations: Phua 2026 (synthetic blindsight via self-model lesion, workspace capacity necessity; no factorial test, no LLMs) [phua2025ablations]; ReCoN-Ipsundrum [recon2026ipsundrum].

**What is NOT crowded** (see §2):
- *factorial* causal interactions between consciousness-related mechanisms in real pretrained models;
- *necessity* tests linking the workspace to metacognition;
- *causal* (not probe-defined) ignition → report;
- recurrence as a manipulated variable;
- developmental (pretraining) emergence of *causal* workspace signatures.

---

## 2. Missing-cell matrix (filled from papers read in this audit)

**Legend:** ● = substantive evidence; ◐ = partial, observational or toy-only; ○ = nothing found. Citations are abbreviated.

| Property \ Evidence | Observational behaviour | Hidden-state decoding | Causal intervention | Necessity ablation | Sufficiency intervention | Developmental emergence | Cross-mechanism generalisation | **Factorial interaction** |
|---|---|---|---|---|---|---|---|---|
| Reportability | ● Lindsey; Binder; Song | ● J-space; Ferrara | ● J-space swaps / injection | ◐ J-space (experiential self-report ↓) | ● J-space injection | ◐ base vs post; OLMo ladder | ◐ J-space multi-function | ○ |
| Global availability / broadcast | ● J-space | ● J-lens | ● cross-function swap | ● task-battery ablation | ● swaps | ◐ base vs post | ◐ | ○ (◐ Phua toy, sequential) |
| Flexible (directed) control | ● J-space focus/ignore; attnctrl2026 | ● | ◐ | ◐ (flexible > automatic under ablation) | ○ | ◐ size effect | ○ | ○ |
| Ignition | ◐ J-space mixing; looped widths | ◐ Ignition Index (probes) | ○ **causal ignition → report** | ○ | ○ | ◐ Pythia changepoint (probe) | ◐ Huginn iteration axis | ○ |
| Recurrence | ◐ looped J-lens | ◐ | ◐ within-loop writes / ablations | ○ **loop count as IV** | ○ | ○ | ○ | ○ |
| Persistence | ◐ J-space autocorrelation; Huginn | ◐ | ◐ Huginn sliding window | ○ | ○ | ○ | ○ | ○ |
| **Metacognitive access** | ● Kadavath; Ackerman; Moran | ● Ashuach; PANL; Kumaran; m9h (workspace → error AUROC 0.69) | ● Kumaran steering; CAA (closinggap) | ◐ PANL not necessary (redundant); **workspace necessity untested** | ● Kumaran; PANL restore | ◐ OLMo ladder (verbal ↑ with SFT, observational) | ◐ Yax; solarkyle transfer miss | **○** |
| Self-model | ● Binder; Betley; Blandfort | ● assistant axis; pain axis | ● steering | ◐ Phua toy self-model lesion | ● steering | ◐ J-space post-training POV | ○ | ○ |
| Error monitoring | ● Ackerman | ● PANL; solarkyle; m9h | ● PANL patching | ◐ (not necessary) | ● | ○ | ◐ (fails to transfer) | ○ |
| Adaptive control | ● Kumaran; Ackerman | ◐ proto-introspection | ◐ Kumaran ✓ / proto-introspection ✗ / Kawada ✗ | ○ | ◐ | ○ | ○ | ○ |
| Temporal continuity | ◐ multi-turn consistency; proactive interference | ◐ | ○ | ○ | ○ | ○ | ○ | ○ |

**Empty or weak cells, which seed the new hypotheses:**
1. The **factorial-interaction column is empty** in pretrained models. Only toy agents with sequential single ablations exist.
2. **Metacognitive access × necessity of the workspace** is empty: the workspace has been ablated, but its effect on knowing-whether-one-knows has never been measured.
3. **Ignition × causal intervention or necessity** is empty (probe- or mixing-defined only).
4. **Recurrence × necessity** (loop count as an independent variable) is empty.
5. Developmental emergence of *causal* workspace and metacognition signatures during *pretraining* is weak.

---

## 3. Candidates (14) with novelty audits

### Field key for every candidate

- **Q:** the question.
- **Closest work:** what was manipulated / measured / found.
- **Gap sentence.**
- **Evidence:** the full-text check.
- **N1 / N2 / N3:** question / method / empirical novelty.
- **Confidence:** novelty confidence.
- **Sig.:** scientific significance.
- **Feas.:** $0 / CPU feasibility.
- **Falsifier:** the strongest falsification.
- **+ / −:** what a positive / negative result would establish.
- **Contribution:** one-sentence paper contribution.

### C1. D3/P1\*: input-fixed lesions in small pretrained LMs; does self-monitoring track lesion-induced competence loss? (category A)

**Closest work:**
- Gu et al. 2026: unlearning, then the same question re-asked; models confabulate rather than refuse.
- Hasegawa et al. 2025: editing → underconfidence.
- Binder et al.: fine-tuning changes behaviour; self-prediction follows (35.4% vs 21.7%).
- Ferrando et al.: entity-recognition latents gate refusal.
- Singh et al.: input-only probes match.
- Ferrara et al.: sham-controlled internal interventions; reports at chance.
- Cohen & de Melo: competence-source regimes. **UNRESOLVED.**

**Gap sentence:** prior work established that unlearning and editing degrade item-level knowledge on fixed inputs and that models mostly fail to report it. It has not combined a sham-controlled within-target estimand with a matched input-corruption contrast.

**Aggressive audit:**
- **Phenomenon:** largely covered, predictably "anosognosia".
- **Attention knock-out vs editing:** a *methodological substitution*, not a new scientific question.
- **Within-target estimand:** a cleaner analysis of a known phenomenon, which is moderate methodological novelty.
- **Distinctness from Singh, Gu, Hasegawa, Ferrando, Cohen & de Melo:** not strong enough for a main-venue paper.

**Ratings:**
- N1 1 · N2 2 · N3 2.
- **Confidence: LOW**, and **NOVELTY UNRESOLVED** for Cohen & de Melo.
- Sig. medium-low. Feas. good (needs Qwen download).

**Falsifier:** monitor tracking ≥ sham under a valid lesion.

**+ / −:**
- \+ "small LMs track self-change" (unexpected).
- − anosognosia (expected; incremental).

**Contribution:** "We measure self-monitoring after input-fixed lesions with a within-target estimand." **Incremental → DEMOTED.**

### C2. Metacognition under self-generated capability change in a pretrained LM (continued fine-tuning; within-target) (A)

**Closest work:**
- Binder (behaviour change tracked);
- Blandfort & Pawar (fine-tuning on own behaviour changes the target);
- continual calibration (aggregate coverage drift) [contcalib2026];
- our D2 (toy: change is all-or-nothing).

**Gap sentence:** prior work established aggregate calibration drift and behaviour-level self-prediction after fine-tuning. It has not tested item-level monitoring of interference-induced competence change beyond pre-change information.

**Ratings:**
- N1 2 · N2 3 · N3 3.
- **Confidence: MEDIUM-LOW.**
- Sig. medium. Feas. marginal (CPU LoRA on ≤ 0.5B; the D2 reachability risk).

**+ / −:** \+ natural-change monitoring; − uninformative if forgetting is catastrophic again.

**Contribution:** "Monitors track [or fail to track] self-generated forgetting in a pretrained LM."

### C3. ★ Is metacognitive access causally dependent on the global workspace? Factorial dissociation in pretrained LMs (A × B, interaction)

**Q:** does ablating the workspace (J-space) abolish a model's ability to know whether it knows, while its first-order answer is held fixed? Is that dependence additive with, or redundant to, a dedicated confidence subspace? And does post-training change the dependence?

**Closest work (full text read):**

| Work | Question | Manipulation | DV | System | Identification | Conclusion |
|---|---|---|---|---|---|---|
| Gurnee et al. 2026 | Is there a workspace? | Lens swaps, injections, top-10 J-direction ablation | Report, reasoning, 14-task battery, experiential self-report | Claude 4.5 family | Matched-norm ablation controls | Workspace exists; ablation spares classification, impairs reasoning. **Confidence, abstention and known/unknown under ablation not reported** |
| m9h OLMo-3 replication | Does the workspace carry error signals? | None (readout) | AUROC of own-error prediction | OLMo-3 ladder | Observational | Workspace 0.69 > verbal 0.51; SFT raises verbal self-evaluation |
| solarkyle Gemma-4 study | Do workspace monitors add over logprob? | None | Error-catching AUROC | Gemma-4-12B | Preregistered, observational | Answer-identity confound; prospective transfer miss |
| Kumaran et al. 2026 (*NMI*) | Do LMs use confidence for behaviour? | Steering a confidence vector at the pre-answer token | Abstention | Gemma-3-27B and others | Steering | Confidence causally drives abstention. **No workspace link; accuracy not held fixed** |
| PANL 2026 | How are errors detected? | Patching / ablation of the post-answer state | Error-detection d′ | Gemma-3-27B, Qwen-2.5-7B | Patching | Sufficient, not necessary (redundant). **No workspace** |
| Closing the gap 2026 | Calibration vs verbal confidence | CAA steering | ECE, verbal confidence | 7–8B base and instruct | Probes + steering | Orthogonal subspaces. **No workspace** |
| Phua 2026 | Can theories be tested on AI? | Self-model lesion; workspace capacity | Type-2 AUROC; access markers | **Toy gridworld agents** | Single ablations, n = 20 seeds | Synthetic blindsight (toy). **No factorial, no LLMs** |
| Kawada & Kellis 2026 | Evidence integration | Mechanistic interventions; J-lens decomposition | Acceptance of candidates | 12 LLMs | Causal + J-lens | Verification decodable but causally inert for answers (control side, not metacognitive sensitivity) |

**Gap sentence:**
- **Established:** LLMs have a causally potent verbalisable workspace (Gurnee et al.); workspace readouts observationally predict the model's own errors (OLMo-3, Gemma-4 replications); a pre-answer confidence representation causally drives abstention and error detection (Kumaran et al.; PANL).
- **Not tested:** whether *metacognitive sensitivity causally depends on workspace access*. That is, whether workspace ablation, with matched-norm controls and the first-order answer held fixed, abolishes knowing-whether-one-knows. Nor has its *interaction* with ablation of the confidence subspace, or with post-training, been tested.

**Evidence:**
- The J-space full text reports no ablation effect on confidence, calibration, known/unknown discrimination or abstention.
- The m9h and solarkyle studies are observational.
- Kumaran, PANL and "closing the gap" have no workspace link.
- Phua is toy-only, with no factorial.

**Ratings:**
- N1 4 · N2 4 (within-item ablations with the answer held fixed + a 2×2 interaction) · N3 4 (first test in pretrained LMs).
- **Confidence: MEDIUM.** The gap survives full-text checks of the 8 closest works, but this is a fast-moving area with active groups.
- **Sig.: high.**
  - It tests GNW's claim that C2 (self-monitoring) operates on C1 (globally broadcast) content, against HOT/PRM-style independent monitors.
  - Indicator frameworks (Butlin et al.) score workspace and metacognition as separate indicators. A dependence or dissociation result tells them whether they are separable in real models.

**Feas.:** high. Pre-fitted lenses exist for Qwen3.5-2B (+pt), Gemma-3-1B (+it), Gemma-2-2B (+it) and Qwen3-4B; the work is CPU inference only (§6).

**Falsifier:** with manipulation checks passed (workspace ablation impairs a workspace-dependent task more than the matched control, and spares an automatic one), metacognitive AUROC on answer-preserved items is **equivalent** (TOST ± 0.05) between workspace ablation and matched control.

**+ / −:**
- \+ (dependence): metacognitive access in LMs is workspace-routed, the GNW-consistent architecture: C2 built on C1. Workspace ablation yields a "reverse blindsight": answers intact, knowing-that-one-knows gone.
- − (independence, validated assay): LM metacognition is computed outside the workspace, for example from first-order fluency or logit signals. So global availability and metacognitive access are **dissociable capacities** in real models: a publishable negative for GNW-based indicator reasoning.
- **Interaction:** tells whether the workspace and confidence subspaces are redundant or serial routes.

**Contribution:** "We give a factorial causal test of whether a language model's metacognitive access depends on its global workspace. Within-item ablations hold the first-order answer fixed, and we find that [it does / does not], separating two capacities that global-workspace accounts link hierarchically and that AI-consciousness indicator frameworks score separately."

### C4. Causal ignition: does trial-wise workspace entry at fixed input determine report? (B)

**Closest work:**
- J-space concept mixing (sharp switch at the workspace onset; deterministic per α);
- Ignition Index (probe sigmoids; no causal test);
- looped J-lens (ignition widths);
- JGateBench (entry by transport, not a gate);
- Lindsey (detection variable at a fixed injection strength; no workspace readout);
- **the Cross-Substrate Access Assay (a model-side two-state vs graded ignition protocol designed, not run).**

**Gap sentence:** prior work established ignition-like switching under input mixing and probe-defined sharpness. It has not shown that, at *identical input*, trials whose content ignites into the workspace are reported and non-ignited ones are not, with a causal nudge at the onset layer converting one into the other.

**Ratings:**
- N1 3 · N2 4 · N3 4.
- **Confidence: MEDIUM-LOW.** Scoop risk is high: the cross-substrate protocol is ready, and Anthropic and JGateBench are active.
- Sig. high (GNW's core signature). Feas. good (same small lensed models; needs natural near-threshold stimuli, avoiding the injection/anomaly confound).

**Falsifier:** report tracks injection strength or context but not trial-wise entry; nudges at onset do not change report more than matched off-workspace nudges.

**+ / −:** \+ access is ignition-gated in LMs; − access is graded, which is evidence against GNW-style access in transformers.

**Contribution:** "Ignition causally gates report in language models at fixed input [or does not]."

### C5. Workspace load × task type: a causal central-bottleneck test (B)

- **Closest work:** J-space A.17 (concurrent covert tasks interleave or split tokens; "moderate cost"; observational); capacity about 25 (Claude), 0.7–2.4 words (open models).
- **Gap sentence:** not tested causally as a 2×2 (load vs matched control load × workspace-dependent vs automatic task) with an interaction test.
- **Ratings:** N1 2 · N2 3 · N3 3. **Confidence: LOW-MEDIUM.** Sig. medium-high. Feas. good.
- **+ / −:** \+ a bottleneck exists; − no bottleneck (consistent with the J-space capacity estimate). Both are modest.

### C6. Access vs report ("inattentional gap"): is unreported, task-irrelevant information inside the workspace? (B / G)

- **Closest work:** the inattentional gap (behavioural, 7 models); J-space alignment auditing (strategic and evaluation-awareness concepts present in the workspace but unreported; ablation surfaces behaviour).
- **Gap sentence:** that accessed-but-unreported content exists is **already shown** (J-space auditing).
- **Ratings:** N1 2 · N2 2 · N3 3. **Confidence: LOW.**

### C7. Access without selfhood: is the self-model workspace-resident, and does ablating it dissociate self-attribution from access? (C)

- **Closest work:** J-space (Assistant POV acquired with post-training; "dissociation between … access and … selfhood" flagged as open); assistant axis; self-referential direction; pain axis.
- **Gap sentence:** open by the authors' own statement.
- **The weakness:** operationalising "selfhood" in an LM is contested, and identification is weak.
- **Ratings:** N1 4 · N2 3 · N3 3. **Confidence: MEDIUM.** Sig. medium-high. Falsifiability low-medium.

### C8. An implicit attention schema in pretrained LMs (C)

- **Closest work:** ASAC (an explicit schema module); Wilterson & Graziano (agents); Testing AST components (other people's attention); "LMs control their own attention" (a prompted protocol, not a schema).
- **Gap sentence:** no test of an *internal* model of the model's own attention that is used for control.
- **The weakness:** attention is computed from the residual stream, so "the residual stream predicts attention" is nearly tautological. That is an identification problem.
- **Ratings:** N1 4 · N2 3 · N3 4. **Confidence: MEDIUM.** But **falsifiability and identification are weak**, and a negative result would be ambiguous.

### C9. Recurrence necessity: loop count as an independent variable for workspace signatures (D)

- **Closest work:** looped J-lens (fixed loops: Ouro 4, Huginn r = 16); the Ignition Index iteration axis; proto-introspection (fixed loops).
- **Gap sentence:** the dose-response of persistence, capacity, ignition or metacognitive readout to the number of recurrences (Huginn supports test-time variation) is untested.
- **Ratings:** N1 4 · N2 4 · N3 4. **Confidence: MEDIUM.** Sig. high (RPT/GNW recurrence).
- **Feas.: poor on CPU.** Huginn is 3.5B × 16–32 recurrences, and no pre-fitted lens exists. **Not $0-credible.**

### C10. Workspace persistence → delayed report (temporal continuity necessity) (D)

- **Closest work:** J-space autocorrelation; Huginn persistence across recurrences; proactive-interference studies.
- **Gap sentence:** whether ablating maintained workspace content at intermediate tokens selectively impairs *delayed* report, against immediate report, is untested.
- **Ratings:** N1 3 · N2 3 · N3 3. **Confidence: LOW-MEDIUM.** Feas. good. Sig. medium.

### C11. Developmental integration: does post-training couple metacognitive report to the workspace? (E)

- **Closest work:** m9h OLMo-3 ladder (post-training moves the J-space 31%; verbal self-evaluation rises with SFT; observational); J-space base vs post.
- **Gap sentence:** causal stage × workspace-ablation interaction on metacognition is untested.
- **Ratings:** N1 4 · N2 4 · N3 4. **Confidence: MEDIUM.**
- **Feas.:** a standalone run on the 7B ladder is too slow on CPU. **Folded into C3 as the base/instruct factor**, using small pt/it pairs with lenses.

### C12. The original seed: separately developed modules integrated, factorial interaction on a held-out capability (E, toy)

- **Closest work:** Phua toy agents (single ablations); emergent merging (superadditivity not reachable by merging); ReCoN-Ipsundrum.
- **Gap sentence:** factorial integration of consciousness-related modules with an interaction test is untested.
- **The weakness:** B1 showed bespoke engineering creates large researcher degrees of freedom, and external validity is low.
- **Ratings:** N1 3 · N2 3 · N3 3. **Confidence: MEDIUM.** **Demoted** (rule 13: architecture engineering).

### C13. Monitoring × control routing: must a metacognitive signal be workspace-resident to drive control? (F, interaction)

- **Closest work:**
  - Kumaran (steering confidence → abstention works);
  - proto-introspection (predictive direction written back does not produce control);
  - Kawada & Kellis (verification states causally inert);
  - J-space (J-component swaps 59% vs 5% for non-J components, for *concept* outputs).
- **Gap sentence:** whether metacognitive *control* (abstain / revise) depends on the confidence signal being expressed inside vs outside the workspace (a 2×2: signal × site) is untested.
- **Ratings:** N1 3 · N2 4 · N3 3. **Confidence: MEDIUM-LOW.** Sig. medium-high. Feas. good (same infrastructure as C3). The natural **follow-up to C3** (output side vs input side).

### C14. Continuous-to-discrete collapse at workspace entry (G)

- **Closest work:** Yang 2026 (theory: access as continuous-to-discrete translation; predicts a geometric collapse); J-space switching.
- **Gap sentence:** the predicted collapse from graded similarity to categorical equivalence classes at workspace onset is untested.
- **Ratings:** N1 3 · N2 3 · N3 3. **Confidence: MEDIUM.** Sig. medium (a single new theory). Identification is correlational.

---

## 4. Scores (0–5; for "scoop risk", 5 = low risk) and novelty confidence

| # | Candidate | Question nov. | Exper. nov. | Theory sig. | Consc.-science link | Falsif. | Causal ID | $0 | CPU | Meaningful negative | Scoop (5 = low) | Ext. validity | Paper | **Σ** | **Novelty conf.** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C3 | Workspace × metacognition (factorial) | 4 | 4 | 5 | 5 | 5 | 4 | 5 | 3 | 4 | 2 | 3 | 4 | **48** | **MEDIUM** |
| C4 | Causal ignition → report | 3 | 4 | 5 | 5 | 4 | 4 | 5 | 3 | 3 | 1 | 3 | 4 | **44** | MEDIUM-LOW |
| C13 | Monitoring × control routing | 3 | 4 | 4 | 4 | 4 | 4 | 5 | 3 | 3 | 3 | 3 | 3 | **43** | MEDIUM-LOW |
| C9 | Recurrence dose (loop IV) | 4 | 4 | 5 | 5 | 4 | 4 | 3 | 1 | 4 | 2 | 2 | 4 | 42 | MEDIUM (infeasible) |
| C11 | Post-training integration (standalone) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 1 | 3 | 3 | 3 | 4 | 42 | MEDIUM (→ in C3) |
| C12 | Modular factorial integration (toy) | 3 | 3 | 4 | 4 | 4 | 5 | 5 | 5 | 2 | 4 | 1 | 2 | 42 | MEDIUM |
| C10 | Persistence → delayed report | 3 | 3 | 3 | 4 | 4 | 4 | 5 | 3 | 3 | 3 | 3 | 3 | 41 | LOW-MEDIUM |
| C5 | Load × task bottleneck | 2 | 3 | 4 | 4 | 4 | 4 | 5 | 3 | 3 | 2 | 3 | 3 | 40 | LOW-MEDIUM |
| C14 | Continuous→discrete collapse | 3 | 3 | 3 | 3 | 4 | 2 | 5 | 4 | 3 | 4 | 2 | 2 | 38 | MEDIUM |
| C7 | Access without selfhood | 4 | 3 | 4 | 4 | 2 | 2 | 5 | 3 | 2 | 3 | 2 | 3 | 37 | MEDIUM |
| C8 | Implicit attention schema | 4 | 4 | 4 | 4 | 2 | 2 | 5 | 3 | 1 | 4 | 2 | 2 | 37 | MEDIUM |
| C6 | Access vs report (inattentional) | 2 | 2 | 3 | 4 | 3 | 3 | 5 | 3 | 2 | 3 | 3 | 3 | 36 | LOW |
| C2 | Self-generated change, pretrained LM | 2 | 3 | 3 | 2 | 4 | 4 | 4 | 2 | 3 | 3 | 3 | 3 | 36 | MEDIUM-LOW |
| C1 | **D3/P1\*** lesions | 1 | 2 | 2 | 2 | 4 | 4 | 5 | 3 | 2 | 2 | 3 | 2 | 32 | **LOW + UNRESOLVED** |

**Ranking: top 3**
1. **C3** (workspace × metacognition, factorial; C11 folded in).
2. **C4** (causal ignition → report). Strong, but most exposed to scooping.
3. **C13** (monitoring × control routing). The natural companion to C3, reusing its infrastructure.

**D3/P1\* ranks last.** It is incremental, and its closest competitor is unresolved.

---

## 5. Recommended #1: C3. Minimal decisive experiment (not an implementation plan)

**Question:** in a pretrained LM, does *knowing whether it knows* causally depend on the global workspace, with the first-order answer held fixed? Is the dependence redundant with a confidence subspace, and does post-training change it?

**System:**
- Primary: Qwen3.5-2B, base (`-pt`) and post-trained, using pre-fitted neuronpedia J-lenses.
- Replication: Gemma-3-1B pt/it, or Gemma-2-2B pt/it (gated licence).
- Scale check: Qwen3-4B (lens available).

**Items:** about 1,500 closed-book factual questions (e.g. a PopQA subset), stratified by popularity so the known/unknown mix is balanced.

**Step 1: first-order.** Generate a greedy answer with the intact model. Its correctness is now fixed per item.

**Step 2: metacognition, the primary measure.** Kadavath-style self-evaluation of *that same answer string*: P("True" | question, proposed answer), one forward pass. It works for base and post-trained models.
- **Secondary measures** (post-trained models only): abstain/answer choice and verbal confidence.

**Step 3: four ablation conditions, applied during the self-evaluation pass** (so the answer and its correctness are identical across conditions):

| Condition | Definition |
|---|---|
| **R** | Matched-norm random-subspace ablation (control) |
| **W** | Workspace ablation: project out the top-k (k = 10, as in Gurnee et al.) J-lens directions at workspace-band layers |
| **C** | Confidence-direction ablation: a cross-fitted correct-vs-incorrect direction at the final self-evaluation token |
| **W + C** | Both |

**First-order check under each ablation:** re-ask the question under the same ablation. The **analysis set** is the items whose answer is unchanged (first-order competence demonstrably intact). This handles the answer-identity and first-order confounds by construction.

**Manipulation checks (pre-declared kill gates; failing any one ends the study with a methods report):**
1. Intact metacognitive sensitivity AUROC ≥ 0.65 in at least one model (otherwise floor).
2. W impairs a workspace-dependent task (two-hop questions) significantly more than R, and spares an automatic one (corpus next-token), replicating Gurnee et al.'s selectivity at this scale. Otherwise the workspace is not identifiable here.
3. C shifts self-evaluation more than R.
4. The analysis set is at least 40% of items.

**Primary estimand:** the within-item change in metacognitive sensitivity (AUROC of P(True) for correct vs incorrect answers) on the analysis set, W vs R. This is paired by item, with an entity-clustered bootstrap. The models are replication units, with a pre-declared decision per model plus a pooled summary.

**Secondary analyses:**
- the W × C interaction (2×2; additive vs redundant routes);
- W × post-training (pt vs it);
- the N1 within-target estimand: does the post-ablation P(True) track correctness beyond the pre-ablation P(True)?

**Pre-declared decision rules:**

| Outcome | Rule |
|---|---|
| **Dependence** | ΔAUROC(W − R) ≤ −0.10, 95% CI excluding 0 |
| **Independence** | TOST equivalence within ± 0.05, with checks 2–3 passed |
| **Otherwise** | Inconclusive |

The interaction is reported on the same scale.

**Compute ($0):**
- Per model: about 1,500 items × (1 generation + 4 self-evaluations + 4 re-asks + manipulation-check items) ≈ 15–20k short forward passes.
- At about 1–2 s each on CPU for a 2B model: roughly **5–10 h per model**.
- Lens projections are cheap. RAM is under 16 GB.

**Why both outcomes are publishable:**
- **Dependence:** the first causal evidence, in real pretrained models, for GNW's C2-on-C1 architecture (knowing-that-one-knows needs global availability). It also gives a "reverse-blindsight" lesion.
- **Independence, with validated assays:** global availability and metacognitive access are dissociable in LMs. That bears on how indicator frameworks aggregate GWT and HOT indicators, and on which theory's architecture LMs instantiate.
- **The W × C interaction** adds mechanism either way.

**Needs PI approval before anything runs (nothing downloaded yet):**
1. **Downloads from Hugging Face:**
   - Qwen3.5-2B and Qwen3.5-2B-pt weights (about 4–5 GB each in bf16);
   - their neuronpedia J-lens folders (size to be checked before download);
   - a PopQA subset (about 20 MB).
   - Optional: the Gemma licence (gated; you must accept it yourself) plus Gemma lenses.
2. Writing a pre-registered C3 protocol (S0 satisfiability and reachability, i.e. R6 and R8, gates, seeds), committed before any model is run.

---

## 6. Direction-specific audits the PI requested

**D3/P1\* (§3, C1):**
- input-preserving lesion → competence loss → (failed) self-monitoring is largely covered by unlearning, editing and introspection work;
- attention knock-out vs editing is a method substitution;
- the within-target estimand adds rigour, not a new phenomenon.

**Verdict: incremental; demoted. NOVELTY UNRESOLVED pending Cohen & de Melo.**

**Global-workspace direction.** "Does an LLM have a workspace?" is done (Gurnee et al. and replications). Status of the remaining causal predictions:

| Prediction | Status |
|---|---|
| Recurrence necessity | Open, but CPU-infeasible (C9) |
| Causal ignition | Open, scoop-exposed (C4) |
| Refractory / blink-type temporal bottlenecks | No precedent found, but a weak theoretical mapping for per-token transformers (folded into C5) |
| Capacity / dual-task | Observational only (C5) |
| Higher-order access to workspace contents | Partly done (injection reportability) |
| **Workspace × metacognition** | **Open and feasible (C3)** |
| Developmental emergence | Post-training only, observational; pretraining-stage causal signatures open (in C3 via the pt/it factor) |

**D1 (methods note) update.** The Cross-Substrate Access Assay (2609.22300) already argues that an indicator test must declare its predictors, fitting, sampling unit, uncertainty target and decision rule. That overlaps our R1–R8 / preregistration-checklist contribution. **D1 novelty is lower than the previous audit stated.** It survives only as a negative-results and case-study note, best attached to an empirical paper (e.g. C3) rather than standing alone.

---

## Sources (read in this audit)

- Gurnee et al. 2026: <https://transformer-circuits.pub/2026/workspace/index.html> · <https://arxiv.org/html/2607.15495v1>
- Looped J-lens: <https://arxiv.org/html/2609.01924v1> · Ignition Index: <https://arxiv.org/html/2608.05160>
- Cross-Substrate Access Assay: <https://arxiv.org/html/2609.22300> · Evidence Integration: <https://arxiv.org/pdf/2609.04290>
- innerJ: <https://github.com/parsa-mz/innerj> · m9h replication: <https://github.com/m9h/jacobian-lens> · solarkyle: <https://github.com/solarkyle/jspace> · lens list: <https://huggingface.co/api/models/neuronpedia/jacobian-lens/tree/main>
- Kumaran et al. 2026: <https://arxiv.org/html/2603.22161> · PANL: <https://arxiv.org/html/2604.22271> · Closing the gap: <https://arxiv.org/html/2603.25052>
- Phua 2026: <https://arxiv.org/html/2512.19155> · Mechanisms of introspective awareness: <https://arxiv.org/html/2603.21396v1>
- Proto-introspection (looped): <https://arxiv.org/html/2607.18553v2> · Emergent capabilities & merging: <https://arxiv.org/html/2609.24504v1>
- Pain axis: <https://arxiv.org/html/2609.16247v1> · LMs control own attention: <https://arxiv.org/html/2609.02737>
- Inattentional gap: <https://arxiv.org/abs/2606.26529> · Yang 2026: <https://arxiv.org/abs/2608.20723>
- ReCoN-Ipsundrum: <https://arxiv.org/abs/2602.23232> · Butlin et al. 2026: <https://cris.tau.ac.il/en/publications/identifying-indicators-of-consciousness-in-ai-systems/>
- Plus the D1 audit sources (`memo/d1_novelty_audit.md`).
