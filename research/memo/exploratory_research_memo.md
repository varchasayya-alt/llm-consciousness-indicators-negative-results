# Exploratory Research Memo

## Computational pathways toward consciousness-related properties in LLM-based systems: problem decomposition, theory and literature landscape, candidate programs, and decisive pilots

**Version** 0.2 · **Date** 2026-10-01 · **Phase** 0 → 0.5 (novelty audit). No preregistered experiments run.

> **v0.2 note.**
> - §10 (Novelty Audit After Closest-Competitor Review) and `memo/decision_document_v2.md` **supersede §6 (preferred direction) and §7 Pilot A**.
> - P1 as originally stated is rejected. The recommended direction is now P5\* + N1, with P1\* as a bridge. P2 is deferred under the $0 constraint.
> - §0–§9 are kept unchanged as the historical record of v0.1 reasoning.

**How to read citations.** Keys in [brackets] point to `research/literature/bibliography.bib`. Full records (claim, relevance, limitations, verification status) are in `research/literature/literature_matrix.csv`, generated from `literature_db.json`.

- `memory-verify`: the entry was written from memory and must be checked before it is cited in a paper.
- `partial`: some details were not retrieved.
- **Important limitation:** most 2025–2026 preprints were assessed from abstracts and fetched summaries, not full-text reading. Any claim of novelty below is provisional until the closest papers are read in full (list in §9).

---

## 0. Executive summary

**0.1 The field moved a lot in 2025–2026, and the research gap is not where the original brief assumed.**

- **LLM introspection via concept injection** [lindsey2025introspection] is now crowded and contested. The paradigm has been:
  - replicated in open models [pearsonvogel2026latent; lederman2026content];
  - dissected mechanistically (detection emerges from preference post-training, through an "evidence-carrier → gate" circuit) [macar2026mechanisms];
  - critiqued as explicable by global logit shifts [hahami2025disturbance];
  - shown to be content-agnostic [lederman2026content];
  - shown to fail to distinguish internal from input-level manipulations [singh2026reality];
  - found to be at chance against sham interventions across 8 open models, even though linear probes recover the intervention [ferrara2026owmi].
- **Anthropic reported an emergent global workspace ("J-space") in Claude** [gurnee2026workspace]. It shows reportability, broadcast, selectivity and limited capacity. Dehaene & Naccache [dehaene2026commentary] call it a landmark. They also list what is not yet shown:
  - ignition;
  - a dual-task bottleneck;
  - autonomous recurrence;
  - self-monitoring (their "C2"), i.e., whether the workspace encodes confidence, error detection, and the boundary between known and unknown.

  Fitted J-lens matrices for open Qwen models are reported (secondary sources, to confirm) to have been released, and independent open-model replications exist on GitHub.
- **The metacognition evidence points toward "world-tracking," not self-tracking:**
  - Across 20 frontier models, confidence is roughly rank-one: one shared item-difficulty factor [moran2026individuated].
  - Self-representations carry privileged correctness information for factual recall but not for math [ashuach2026consensus].
  - Degraded or stale-knowledge systems fall into a "danger zone" of high confidence with low accuracy [cohen2026source; full text not yet read].

**0.2 The central methodological gap.** For every capacity whose object is the system itself (metacognition, introspection, self-model, attention schema), we have not found designs where these two accounts make different predictions:

- a *genuine self-monitoring* account;
- a *generic difficulty/world-model* or *first-order readout* account.

We also have not found mechanistic evidence of a dissociable second-order process. Singh et al. [singh2026reality] call for exactly this. Dehaene & Naccache [dehaene2026commentary] call for the C2 probes.

**0.3 Proposed criterion: counterfactual self-dependence.** A self-representation counts as self-tracking only if it changes when the system's *own* state is intervened on while the input is held fixed. The experiment that follows is **lesion-tracking**:

- selectively impair a model's competence on specific items, without changing the input;
- ask whether its prospective feeling-of-knowing, retrospective confidence, internal monitor signals and workspace contents update;
- or whether instead it shows **artificial anosognosia** (keeps high confidence on items it can no longer answer).

This generalises Lindsey's "grounding" criterion from injected foreign contents to the system's own competence.

**0.4 Seven candidate programs (P1–P7)** were generated and compared (§4–§5).

- Preferred direction (provisional): **P1, counterfactual self-tracking ("artificial anosognosia")**.
- It is staged so that it bridges to:
  - P2 (workspace / GNW tests: is C2 in the J-space?);
  - P3 (recurrence: does looping improve self-tracking?);
  - P5 (the human researcher's decomposition/integration idea, reformulated as trained monitors tested on *held-out lesion types they were never trained on*).

**0.5 Decisive pilots (A–D)** are designed to falsify cheaply (§7).

- Total ≈ 30–45 GPU-hours on one rented 24–80 GB GPU.
- **This machine has no CUDA GPU.** Locally we can only debug code on ≤1.5B models on CPU.

**0.6 What we will not claim.** Nothing here bears on whether any system *experiences* anything (Level 3, see below).

### Claims discipline

| Level | Form of claim | How we may support it | Status in this project |
|---|---|---|---|
| 1 Computational | "System S has measurable property X" (e.g., its feeling-of-knowing tracks lesions to its own recall) | Experiments, ablations, statistics | Target of all experiments |
| 2 Theory-mapping | "X corresponds to indicator I predicted by theory T" | Argument from literature, stated *conditional on T* (and on computational functionalism) | Stated explicitly and conditionally in every write-up |
| 3 Phenomenal | "S experiences something" | Nothing currently available | Never asserted; listed as unresolved |

Paper rule: every sentence mentioning consciousness must say which level it is at. "Indicator" never means "evidence of experience" without the word "conditional".

---

## 1. Problem decomposition

### 1.1 What could "developing a conscious LLM" mean?

| # | Interpretation | What success would mean | Approachable now? | Level |
|---|---|---|---|---|
| I1 | **Phenomenal**: the system has experiences | Something it is like to be the system | **No.** No theory-neutral test exists. Some versions (Type-A biological naturalism) are untestable in principle [klatzmann2026biology] | 3 |
| I2 | **Indicators by design**: the architecture contains the components a theory names | The components exist on a diagram | **Trivially yes, so weak.** Scaffolds can tick GWT boxes [goldstein2024case] without the function being realised | 2 |
| I3 | **Indicators realised and measured**: the computational *signatures* a theory predicts are present and causally do the work | Measured signatures, with causal ablations | **Yes.** Best target | 1→2 |
| I4 | **Contrastive structure**: the system reproduces the *pattern of dissociations* that defines the conscious/unconscious contrast in humans (threshold bimodality, blindsight, masking, automatic vs controlled, anosognosia) | Dissociations, not just capacities | **Yes**, partly started [gurnee2026workspace] | 1→2 |
| I5 | **Self-report validity**: the system's reports about itself are causally grounded and privileged, so they could serve as evidence | Reports pass privileged-access and counterfactual tests | **Yes**; contested area | 1 |
| I6 | **Substrate / dynamical requirements** (Φ, biological properties, autonomous dynamics) | Physical/dynamical property present | **Mostly no.** IIT holds that software on conventional hardware is irrelevant [findlay2024dissociating]. Complexity proxies are unvalidated in ANNs [phua2025ablations] | 2–3 |
| I7 | **Developmental pathway**: the properties *arise through learning* rather than being hand-built | Emergence under training, with generalisation | **Yes**, in fine-tuning and small models [cleeremans2020learning; gurnee2026workspace] | 1 |
| I8 | **Moral-status-relevant** (valence, suffering) | Valenced states | **Excluded** on ethical grounds [metzinger2021suffering; butlin2025principles] and because the theory is immature | 3 |

**Conclusion.** I3, I4, I5 and I7 are experimentally approachable. I2 is insufficient on its own. I1, I6 and I8 are outside what we can claim. The most informative work combines I4 and I5: *dissociation tests applied to self-reports and self-assessments.*

### 1.2 Conceptual distinctions (working definitions, kept separate throughout)

| Concept | Working (functional) definition | LLM-relevant operational signature | Must not be conflated with |
|---|---|---|---|
| Phenomenal consciousness | There is something it is like [block1995confusion] | None available. All signatures below are at most theory-conditional indicators | Any of the below |
| Access consciousness (C1) | Content available for report, reasoning and control [block1995confusion; dehaene2017science] | One content causally usable by *many* downstream consumers; selective; capacity-limited (J-space) [gurnee2026workspace] | Mere decodability (information present but not used) [ferrara2026owmi; pearsonvogel2026latent] |
| Self-awareness | Representing oneself *as oneself*, including one's own states and traits | Self-specific information, beyond what an observer has; self/other discrimination | Self-description learned from text. J-space ablation reduced experiential language for self and third-person descriptions *equally* [gurnee2026workspace] |
| Metacognition (C2) | Monitoring and control of one's own cognition: confidence, error detection, feeling of knowing (FOK) | Type-2 sensitivity *beyond an input-only observer*; tracks one's own changes (§1.3) | Calibration from a generic difficulty model [moran2026individuated] |
| Agency | Goal-directed selection of actions, sensitive to learned action–outcome contingencies | Flexible goal pursuit; a model of output→input contingencies | Instruction following |
| Self-modelling | An internal model of one's own processes or dispositions, used for prediction and control | Predicts own behaviour/states; updates when the self changes; used for control | Self-description; persona role-play [shanahan2024exotica] |
| Global integration (GW sense) | Contents combined across sources and broadcast | Broadcast breadth plus competition for a bottleneck | IIT's Φ (intrinsic causal integration of the physical substrate) |
| Recurrent processing | Later representations modulate earlier ones *over time* (re-entry) | See note below on decoder transformers | Autoregression in general |
| Persistent identity / temporal continuity | A self-model and dispositions maintained across time and change; memory of one's own past | Stable self-representation across sessions and updates; correct self-attribution of past states | A long context window |
| Attention vs attention schema | Selection mechanism vs a *model of* that mechanism used for control [graziano2015ast] | Representation that predicts the system's own allocation and is used to steer it | Attention weights themselves |
| Source / reality monitoring | Telling self-generated content from external content | Detects inserted vs self-generated tokens beyond surprisal [wang2026prefill; ranjan2026reality] | Style detection |
| Uncertainty awareness | A *represented and used* estimate of one's own uncertainty | Second-order signal, distinct from output entropy | First-order output entropy [stolfo2024confidence] |

**Note on recurrence in decoder transformers.** Within one forward pass, computation is strictly feedforward. Across tokens, layer ℓ at step t+1 can read layer ℓ's keys/values from earlier positions (lateral, same depth). The *only* path from a deep layer at step t to a shallower layer at step t+1 is the sampled token, which is a discrete, low-bandwidth bottleneck. So standard LLMs have top-down re-entry only through their own output stream.

Looped, recurrent-depth and feedback architectures [geiping2025huginn; zhu2025ouro; fan2020feedback; hao2024coconut] add continuous top-down re-entry. This is a precise, manipulable architectural variable.

**Dissociations that make these distinctions empirical:**

- blindsight: performance without awareness or metacognition [lau2006relative];
- anosognosia: a deficit without awareness of the deficit;
- cue-familiarity FOK without access to the target [reder1992fok] vs accessibility-based FOK [koriat1993fok];
- confabulated self-explanation [turpin2023unfaithful];
- access without a special self-model [gurnee2026workspace];
- **a dissociation specific to transformers:** broadcast, selectivity and capacity limits *without recurrence* [gurnee2026workspace].

That last case is a natural experiment *for* theory. GNW treats recurrent ignition as part of the mechanism of access. If feedforward systems implement access-like functions without it, the theory's components come apart.

### 1.3 The evidential problem, and a hierarchy of evidence for self-directed capacities

The **gaming problem** [birch2024edge]: systems trained on human-generated data can display human markers without the underlying process. This is sharpest for capacities whose object is the system itself. We propose five tiers of evidence:

| Tier | Criterion | What it rules out | Prior work using it |
|---|---|---|---|
| E0 Behavioural | Outputs that sound self-aware ("I'm not sure") | Nothing | Self-referential report studies [berg2025selfref] |
| E1 Accuracy | Self-assessments correlate with ground truth (calibration, type-2 AUROC, meta-d′ [maniscalco2012]) | Random reporting | Most metacognition work [kadavath2022know; steyvers2026metacog] |
| E2 Privileged access | Beats any equally cheap observer that has the same inputs/outputs [song2025privileged] | A generic "typical solver" model | [binder2024looking; song2025fail; ashuach2026consensus; singh2026reality] |
| **E3 Counterfactual self-dependence** | Tracks interventions on the system's *own* internal state or competence **with input fixed**. Moves in the right direction, item-specifically, and does **not** move under sham interventions of matched magnitude | Difficulty models; generic anomaly detection; input-cue heuristics | Lindsey's *grounding* criterion for injected contents [lindsey2025introspection]. Behaviour changes via fine-tuning [binder2024looking; guo2026coupling]. **We have not yet found it applied to item-specific lesions of competence (see §3.10)** |
| E4 Mechanistic second-order structure | A representation that (a) is causally upstream of the self-assessment, (b) is separable from first-order content (e.g., answer identity is not decodable from it [jspace_gemma_repo]), and (c) can be lesioned independently, giving a **double dissociation** | "First-order readout" accounts of confidence [fleming2024review] | Toy agents [phua2025ablations]; called for in [singh2026reality] |

- E3 entails E2 for the property intervened on. No observer with only input access can predict a lesion it cannot see.
- **Theory mapping (Level 2, conditional):**
  - E3 is a *necessary* condition for a higher-order representation *of first-order states* (HOT). A HOT that does not covary with its target state at best generates misrepresentations.
  - E3 is also necessary for GNW's C2 self-monitoring [dehaene2017science], for AST's schema-tracks-attention requirement [graziano2015ast], and for using self-reports as evidence [perez2023selfreports].
  - E4 is what HOT adds beyond first-order-readout theories of confidence.

### 1.4 What is experimentally approachable

**Approachable now, on open-weight models with limited compute:**

- E1–E4 tests of self-directed representations;
- dissociation paradigms imported from human consciousness science:
  - threshold bimodality;
  - dual-task bottlenecks;
  - blindsight- and anosognosia-like dissociations;
  - automatic vs controlled processing;
- architectural manipulations of recurrence (within-model loop counts; small from-scratch models);
- developmental manipulations: what a monitor was trained on, and in what order.

**Not approachable:** phenomenal claims; substrate claims; valence.

---

## 2. Theory landscape

### 2.1 State of the evidence (why we do not privilege one theory)

- **No theory is empirically privileged.**
  - The preregistered GNWT-vs-IIT adversarial collaboration found results that "align with some predictions" of both and "substantially challenge key tenets" of each [cogitate2025].
  - Across the field, methodological choices predict which theory a study supports [yaron2022contrast]; see also [seth2022theories] for a comparative review.
- **IIT is disputed:**
  - its scientific status [iitconcerned2025, response iitresponse2025];
  - the unfolding argument [doerig2019unfolding];
  - by IIT's own lights, AI consciousness depends on the hardware's causal structure, not on software [findlay2024dissociating].
- **Biological naturalism** rejects computational functionalism [seth2025bbs; aru2023feasibility]. Its Type-B form (biology matters *because* it affords particular computations) is testable [klatzmann2026biology].
- **The indicator method** [butlin2025indicators; butlin2023report] is the most operational approach. It assumes functionalism as a working hypothesis and has mostly been applied by *inspecting architectures*, not by measuring.

**Implication for this project:**

- Prefer predictions that **discriminate between theories**, or that are **shared by several**.
- Report **separate indicator measurements**, never a composite "consciousness score".
- Keep every Level-2 statement conditional.

### 2.2 Theory profiles

**Global Workspace / Global Neuronal Workspace (GWT/GNW)** [baars1988; mashour2020gnw; dehaene2017science]

- **Central claim:** conscious access is the selection of a content into a limited-capacity workspace that broadcasts it to many specialised processors. In brains this is marked by non-linear *ignition* and sustained, long-range recurrent amplification.
- **Computational requirements:** specialised modules; a limited-capacity workspace; competition for entry; global broadcast; ignition; maintenance; and for C2, self-monitoring.
- **Implementable:**
  - Designed workspaces [goyal2022workspace; vanrullen2021gw; devillers2024gw; chateaulaurent2025chain].
  - LLM agent scaffolds [goldstein2024case].
  - **An emergent workspace in pretrained transformers** [gurnee2026workspace].
- **Evidence:** strong correlational and causal evidence for access-related signatures in humans (ignition, bifurcation at threshold [sergent2021bifurcation]). It was challenged on prefrontal content and offset predictions [cogitate2025].
- **Criticisms:** conflates access with phenomenal consciousness [block1995confusion]; report confounds; prefrontal role disputed.
- **AI relevance: highest.** It has the richest set of operational predictions, and these are now directly testable in LLMs. Risk: it is easily satisfied by scaffolds (I2).
- **Our view:** the most useful theory for *measurement*. Its open questions in LLMs are ignition, the central bottleneck, the role of recurrence, and C2.

**Recurrent Processing Theory (RPT)** [lamme2006]

- **Central claim:** local re-entrant processing in sensory cortex suffices for (phenomenal) consciousness, independently of access and report. The feedforward sweep is unconscious.
- **Requirements:** feedback within a processing hierarchy that binds and organises representations.
- **Implementable:** looped, recurrent-depth and feedback models. In text-only LLMs, though, it is unclear what the "perceptual stage" is.
- **Evidence:** masking and recurrence studies; recurrence needed for hard object recognition [kar2019recurrent]. The phenomenal-without-access claim is hard to test by construction.
- **Criticisms:** the unfolding argument [doerig2019unfolding] applies to any structural criterion; it is also hard to falsify.
- **AI relevance:** standard transformers fail RPT-1, and recurrent-depth models now satisfy it cheaply, so *structural* presence is a weak indicator. We need **functional signatures of re-entry** (e.g., masking analogues), and we must accept that those can be unfolded.

**Higher-Order Theories (HOT; including perceptual reality monitoring and higher-order state-space)** [brown2019hot; lau2022trust; fleming2020hoss]

- **Central claim:** a state is conscious when a suitable higher-order representation represents it, e.g., marks it as a reliable representation of the present.
- **Requirements:** metarepresentation of first-order states; a monitor that separates signal from noise; belief formation and action guided by the monitor (Butlin HOT-1 to HOT-4).
- **Implementable:** second-order networks [pasquali2010know]; monitor heads; internal correctness representations in LLMs [ferrando2025entity; ashuach2026consensus].
- **Evidence:**
  - relative blindsight, with prefrontal involvement [lau2006relative];
  - metacognition can dissociate from performance [fleming2014measure; fleming2024review].
  - Disputed points: prefrontal lesion evidence, and the "empty higher-order thought" (misrepresentation) problem.
- **Criticisms:** misrepresentation; reduction to metacognition; prefrontal debates.
- **AI relevance: high and under-tested.** LLM metacognition is measurable, but the evidence that it is genuinely *second-order and self-directed* is weak (§3.3). The E3/E4 tests operationalise HOT's core structural commitment.

**Attention Schema Theory (AST)** [graziano2015ast; graziano2020standard]

- **Central claim:** awareness is the system's simplified, descriptive model of its own attention, used to control attention. Reports of experience are read off this model (an illusionist-friendly view [frankish2016illusionism]).
- **Requirements:** an attention mechanism; a model of attention-state; that model used for endogenous control; reports drawing on the model.
- **Implementable:** explicit schema modules [wilterson2021ast; liu2023ast; farrell2024ast; asac2025]. In LLMs, "attention" is multi-head and multi-layer, so the mapping is non-trivial.
- **Evidence:** behavioural attention/awareness dissociations; computational demonstrations; limited direct neural tests.
- **Criticisms:** it explains *claims about* consciousness (its proponents accept this); implementations risk being trivial.
- **AI relevance:** concrete engineering predictions. **We found no prior work probing pretrained LLMs for an *implicit* attention schema (P4).**

**Predictive processing / active inference** [clark2013whatever; friston2010fep; laukkonen2025loop; seth2021being]

- **Central claim:** brains minimise hierarchical prediction error. Consciousness is associated with particular generative models:
  - of the body and its regulation (Seth);
  - a unified world model, plus competition to enter it, plus "epistemic depth" from recursively sharing beliefs (Laukkonen et al.).
- **Requirements:** a generative world model; precision weighting; action that minimises expected free energy. For the "beautiful loop": competition into a single model, plus recurrent self-modelling.
- **Implementable:** model-based RL agents and world models. LLMs predict, but are not PP architectures in the error-unit sense.
- **Evidence:** strong as a theory of perception; weakly *discriminating* as a theory of consciousness.
- **AI relevance:** mostly framework-level. Its testable AI predictions overlap with GNW (competition) and HOT/AST (self-modelling).

**Integrated Information Theory (IIT 3.0/4.0)** [tononi2016iit; albantakis2023iit4]

- **Central claim:** consciousness *is* maximally irreducible intrinsic cause–effect structure (Φ) of a physical system.
- **Requirements:** physical integration. Feedforward structure gives Φ = 0, and software is irrelevant.
- **Implementable:** not through software on conventional hardware [findlay2024dissociating]. Attempts to compute IIT measures on LLM activations exist [iitllm2025; synergy2026core], but IIT itself would regard activation time series as the wrong object.
- **Evidence:** mixed [cogitate2025]. PCI's clinical success [casali2013pci] is often credited to IIT, but PCI is not Φ.
- **Criticisms:** intractable; testability [iitconcerned2025]; unfolding argument; panpsychist implications.
- **Our decision: excluded as an engineering target** (decisions log D2). We keep it as a contrast theory: IIT predicts every result we obtain is irrelevant to consciousness, and the paper should say so.

**Biological naturalism (BN)** [seth2025bbs; aru2023feasibility; klatzmann2026biology]

- **Central claim:** consciousness depends on properties of living systems.
- **Our response:** results here cannot address Type-A BN. Type-B claims (e.g., continual learning or homeostatic regulation as required computations) can be turned into P6-style experiments.

**Self-Organising Metarepresentational Account / radical plasticity (SOMA)** [cleeremans2020learning; pasquali2010know]

- **Central claim:** consciousness is *learned*. A system learns metarepresentations that redescribe its own first-order states.
- **AI relevance:** directly motivates *training* monitors and testing whether what they learn generalises (P1 Stage 3; P5). It is the most natural theoretical home for the human researcher's developmental intuition.

**Others, noted for completeness:**

- Conscious Turing Machine [blum2022ctm];
- information generation [kanai2019infogen];
- illusionism [frankish2016illusionism];
- unlimited associative learning [ginsburg2019soul] and the continual-learning argument [hoel2025disproof];
- dendritic integration theory [aru2020cellular];
- information-theoretic accounts of richness and ineffability [ji2024richness].

### 2.3 Comparative table

| Theory | Core requirement | Implementable in LLM systems? | Measurable LLM signature | Empirical standing (our judgment) | Use in this project |
|---|---|---|---|---|---|
| GNW | Bottleneck + broadcast + ignition (+C2) | Yes; emergent J-space | Broadcast breadth, selectivity, capacity, bimodality at threshold, dual-task interference | Strongest for *access*; key tenets challenged [cogitate2025] | P2; C2 bridge in P1 |
| RPT | Local re-entry | Yes, with looped/feedback models | Re-entry-dependent functions (masking analogue) | Moderate for perception; phenomenal claim hard to test | P3 |
| HOT / PRM / HOSS | Higher-order representation of first-order states | Plausibly | E3/E4: self-tracking, dissociable monitor | Moderate; disputed | **P1 core** |
| AST | Model of attention used for control | Plausibly | Schema probe, schema→control causality, report-follows-schema | Limited direct evidence | P4 |
| PP / active inference | Generative self/world model; competition; epistemic depth | Partly | Overlaps GNW/HOT | Strong for perception; weak for consciousness specifically | Framework only |
| IIT | Physical Φ | No (per IIT) | None valid in software | Disputed | Excluded; contrast theory |
| BN | Life-like substrate/processes | Type-A no; Type-B testable | Type-B computations | Unresolved | Conditional framing; P6 |
| SOMA | Learned metarepresentation | Yes | Trained monitors that generalise | Theoretical | P1 Stage 3, P5 |

### 2.4 Where the theories converge and diverge on AI-testable predictions

**Convergence.** Most theories demand three things:

- (i) selective global availability (GNW, AST, CTM, the beautiful loop);
- (ii) self-monitoring or self-modelling (HOT, AST, SOMA, GNW-C2, the beautiful loop);
- (iii) temporally extended or re-entrant processing (RPT, GNW ignition, the beautiful loop, BN).

**Testable divergences:**

1. **Does report need a separable higher-order representation (HOT), or only workspace entry (GNW)?**
   - Test: can workspace contents be present and broadcast while the monitor says "absent", and vice versa? This is E4 inside the J-space; P1 Stage 2 with P2.
2. **Is re-entry necessary for access-like functions?**
   - Feedforward transformers that broadcast are a natural experiment (P2/P3).
   - Unfolding constraint: we can test whether *training with re-entry* produces functions. We cannot test whether re-entrant *structure* is necessary.
3. **Is a model of attention (AST) the same thing as general self-monitoring (HOT)?**
   - Test: dissociate the attention-schema signals (P4) from FOK signals (P1).

---

## 3. Existing work: what has already been attempted

### 3.1 Indicator frameworks and position papers

| Work | Contribution | Leaves open |
|---|---|---|
| Butlin et al. 2023/2025 [butlin2023report; butlin2025indicators] | Theory-derived indicators; assessed systems by architecture | Measuring indicators; validating indicators against dissociations |
| Chalmers 2023 [chalmers2023llm] | Obstacles: recurrence, workspace, unified agency | Whether the obstacles are functional or structural |
| Dehaene, Lau & Kouider 2017 [dehaene2017science] | C1/C2 decomposition | C2 in real systems |
| Aru et al. 2023; Seth 2025 [aru2023feasibility; seth2025bbs] | Neuroscience- and biology-based scepticism | Type-B tests |
| Goldstein & Kirk-Giannini 2024 [goldstein2024case] | Language agents may satisfy GWT | Shows I2 is cheap |
| Findlay et al. 2024 [findlay2024dissociating] | IIT dissociates AI from consciousness | — |
| Hoel 2025 [hoel2025disproof] | Substitution argument; continual learning as necessary | Empirical consequences |
| Bengio & Elmoznino 2025 [bengio2025illusions]; Comsa 2026 [comsa2026tractable] | Risks of attribution; shift to perceived consciousness | — |
| Perez & Long 2023; Long et al. 2024; Butlin & Lappas 2025 [perez2023selfreports; long2024welfare; butlin2025principles] | Self-report evaluation proposals; welfare; responsible-research principles | Validity tests for trained self-reports (→ P1) |

### 3.2 Introspection in LLMs (2024–2026)

| Work | Paradigm | Result |
|---|---|---|
| Binder et al. 2025 [binder2024looking] | Fine-tuned self-prediction of hypothetical behaviour | Self beats cross-prediction; tracks behaviour changed by fine-tuning |
| Song et al. 2025a/b [song2025fail; song2025privileged] | Self vs other model on linguistic knowledge and temperature | No privileged access; defines privileged access |
| Comsa & Shanahan 2025 [comsa2025introspection] | Conceptual analysis | Introspection requires a causal link from state to report |
| Plunkett et al. 2025 [plunkett2025selfinterp] | Report fine-tuned preference weights | Accurate; training generalises to native preferences |
| Ji-An et al. 2025 [jian2025metacognitive] | Neurofeedback on activation directions | Report and control, but critiqued as input-solvable [singh2026reality; aoki2026neurofeedback] |
| Lindsey 2025 [lindsey2025introspection] | Concept injection | ~20% detection at ~0 false positives in Claude Opus 4/4.1 |
| Hahami et al. 2025 [hahami2025disturbance] | Injection in Llama-3.1-8B | Yes/no detection explained by logit shift; localisation above chance |
| Pearson-Vogel et al. 2026 [pearsonvogel2026latent] | Qwen-32B, logit lens | Latent detection suppressed before output |
| Lederman & Mahowald 2026 [lederman2026content] | Large open models | Detection without content identification |
| Macar et al. 2026 [macar2026mechanisms] | Circuits | Post-training (DPO) creates an evidence→gate circuit; detection and identification separate |
| Singh, Linzen & Ravfogel 2026 [singh2026reality] | Re-analysis with input-only controls and a "gaslight" condition | Input-only probes match; cannot tell internal from input manipulations |
| Ferrara 2026 [ferrara2026owmi] | 8 open models, sham and impact-matched controls | Reports at chance (AUROC ≈ 0.50); probes 75–96% |
| Guo et al. 2026 [guo2026coupling] | Explanation training | Explanations track behavioural change despite fixed supervision |
| Gurnee et al. 2026 [gurnee2026workspace] | J-space | Injected J-lens vectors reportable |

**Assessment:**

- Injection *detection* is real in some models but is partly explained by anomaly detection and response bias.
- Content identification is weak.
- Reports often fail sham and input-manipulation controls.
- Post-training shapes the pathway.

**Structural weakness of the paradigm:** the injected state is *foreign*, so the test mostly measures anomaly detection. It is not a test of monitoring one's *own ongoing competence*. This is the opening that P1 targets.

### 3.3 Metacognition and self-knowledge

- **Calibration and P(IK)** exist [kadavath2022know], but:
  - verbal and internal confidence diverge [zhang2026diverge];
  - frontier-model confidence is approximately rank-one across models, i.e., shared difficulty [moran2026individuated];
  - self-probes beat peer probes only on disagreement items, and only for factual recall [ashuach2026consensus].
- **Limited but growing behavioural metacognition** (opt-out, strategic use of confidence) [ackerman2026limited].
- **Mechanisms:**
  - entity-recognition latents causally gate refusal vs hallucination [ferrando2025entity];
  - a "known entity → inhibit can't-answer" circuit whose misfires produce hallucinations [lindsey2025biology];
  - hidden-state "knowing" signals track whether recall occurred, not truth [cheang2025recall];
  - familiarity and reliability are dissociated, and familiarity signals are rarely acted on [brzezinka2026bielik];
  - confidence-regulating entropy neurons [stolfo2024confidence].
- **Degradation:**
  - "danger zone" in degraded or stale-knowledge systems [cohen2026source; full text pending];
  - after unlearning, models confabulate rather than admit ignorance [gu2026unlearners].
- **Training:**
  - metacognitive alignment generalises to "newly acquired knowledge" [park2026esma];
  - LoRA metacognitive losses improve *aggregate* calibration [luo2026predictive];
  - probe-based J-space error monitors fail to transfer prospectively and can exploit answer identity [jspace_gemma_repo].
- **Human literature we can borrow:** FOK as *cue familiarity* [reder1992fok] vs *accessibility* [koriat1993fok]; meta-d′ [maniscalco2012]; relative blindsight [lau2006relative].

### 3.4 Global workspace: designed and discovered

- **Designed:**
  - shared workspace for modules [goyal2022workspace];
  - deep-learning GW roadmap [vanrullen2021gw];
  - multimodal GW [devillers2024gw];
  - routing operations through a GW gives compositional generalisation [chateaulaurent2025chain];
  - GWT/IGT/AST related to general intelligence [juliani2022link].
- **Discovered:** J-space in Claude [gurnee2026workspace], with a commentary [dehaene2026commentary] and a critical review [nanda2026review]. Open replications on Qwen and Gemma exist (GitHub, §Appendix).
- **Ignition claims:**
  - layer-wise probe sigmoids [rahbar2026ignition]. We judge this weak: representational emergence across layers is not trial-wise bistability;
  - readout-localised ignition in a 30M recurrent-depth model [lammuir2026ignition].
- **Toy multi-theory agents with ablations:** [phua2025ablations].

### 3.5 Recurrence and latent reasoning

- Open looped models: Huginn-3.5B [geiping2025huginn]; Ouro-1.4B/2.6B [zhu2025ouro].
- Related architectures: feedback memory [fan2020feedback]; Coconut [hao2024coconut]; Universal Transformers [dehghani2019universal].
- Layers as block-recurrent dynamics [jacobs2025blockrecurrent].
- Pre-answer probes in looped models predict correctness beyond surface features, but control fails [kirin2026looped].
- **Not found:** loop count used as an independent variable for *self-tracking* under lesion.

### 3.6 Attention schema and self-modelling

- **AST agents:** [wilterson2021ast; liu2023ast; farrell2024ast]; AST module in transformers [asac2025].
- **Self-modelling** as an auxiliary task regularises networks [premakumar2024selfmodel]; earlier robot self-models (Bongard/Lipson; not yet in the bibliography).
- **Not found:** probing *pretrained* LLMs for an implicit attention schema that is causally used for control.

### 3.7 Agency, authorship, persistence

- **Prefill awareness / self-authorship:** [wang2026prefill; ackerman2024selfrec; ranjan2026reality]; Lindsey's prefill experiment [lindsey2025introspection]. This area is crowded.
- **Persistence and identity:** [perrier2026identity; guo2026coupling; chen2025persona]; continual-learning arguments [hoel2025disproof].

### 3.8 Integration and complexity measures

- IIT measures on LLM activations [iitllm2025]; synergistic cores [synergy2026core].
- PCI-like measures *decreased* under workspace constraints in toy agents [phua2025ablations], a caution against transferring clinical proxies.

### 3.9 Modular, developmental and merging approaches

- **Merging:**
  - task arithmetic [ilharco2023taskarith];
  - TIES [yadav2023ties];
  - LoRA Soups, where merging beats data mixing for skill composition [prabhakar2024lorasoups];
  - evolutionary merging yields cross-domain combinations [akiba2025evolutionary].
- **Emergence caveats:** metric artefacts [schaeffer2023mirage]; multiplicative composition [okawa2023multiplicative].
- **Developmental theory:** [cleeremans2020learning].
- **We have not yet identified prior work that** trains consciousness-related capacities separately, integrates them, and tests *pre-registered superadditive interactions* on *held-out* behaviours with compute-matched nulls. Closest: toy multi-module agents with ablations [phua2025ablations].

### 3.10 Gap analysis (provisional; "not found" ≠ "does not exist")

| Gap | Description | Closest work | Confidence the gap is real |
|---|---|---|---|
| **G1** | Item-specific, **input-preserving** lesions of the recall pathway, testing whether self-assessments are self-tracking: verbal FOK, logit P(IK), frozen probes, abstention, J-space contents. Includes a *double dissociation* (familiarity lesion vs recall lesion) | [cohen2026source] (degraded systems: **must read**); [binder2024looking]; [gu2026unlearners]; [phua2025ablations]; [park2026esma] | Medium. Needs full-text check of Cohen & de Melo, Park et al., and the Liu et al. survey [liu2026metacogsurvey] |
| **G2** | Does the J-space carry C2 signals (FOK, error), and are they self-tracking? | [dehaene2026commentary] (call); [gurnee2026workspace] ("damn" tokens); [jspace_gemma_repo] | Medium–high, but Anthropic/GDM may be doing it now |
| **G3** | Proper ignition tests in open models: stimulus-strength non-linearity, across-trial bimodality, local-vs-workspace contrast. Also dual-task central-bottleneck interference | [gurnee2026workspace] §4.1.1, A.17; [rahbar2026ignition] | Medium; high risk of being scooped |
| **G4** | Loop count as an IV for self-tracking quality | [kirin2026looped] | Medium–high |
| **G5** | Implicit attention schema in pretrained LLMs | [farrell2024ast; asac2025] (explicit modules only) | Medium–high |
| **G6** | Factorial module integration with pre-registered interaction tests, held-out emergent behaviours, and developmental order as an IV | [phua2025ablations] | Medium |

---

## 4. Candidate research programs

Each program is substantially different in its hypothesis, theory and method. Compute estimates assume one rented GPU (24–80 GB) unless stated.

### P1. Counterfactual self-tracking: do LLM self-assessments track lesions to their own competence? ("Artificial anosognosia")

- **Scientific hypothesis.** Two competing hypotheses, with a third for later stages:
  - **H1-world (cue familiarity):** an LLM's *prospective* feeling-of-knowing (FOK) is computed mostly from input/cue familiarity, by a pathway separable from recall. So lesioning recall while sparing familiarity produces artificial anosognosia: confidence stays high while accuracy collapses, item by item.
  - **H1-self:** FOK reads out the state of the recall process. Confidence drops on exactly the items whose recall was lesioned, and not under sham lesions of matched magnitude.
  - **H1-HOT (Stage 2):** there is a monitor representation that is dissociable from answer content and can be lesioned independently (E4).
- **Theoretical motivation:**
  - E3/E4 are necessary conditions for HOT-style higher-order representation, GNW-C2, AST-style schemas, and evidential self-report (§1.3).
  - Human FOK has both cue-familiarity and accessibility components [reder1992fok; koriat1993fok]. Anosognosia is a classic dissociation of self-monitoring from competence. Either LLM outcome therefore has a human analogue, and the question is *which mechanism*, not "is the model good or bad".
  - Mechanistic priors favour H1-world for prospective FOK: familiarity gating [lindsey2025biology; ferrando2025entity]; a shared-difficulty factor [moran2026individuated]; a danger zone [cohen2026source].
  - Other priors favour partial self-tracking: privileged factual self-information [ashuach2026consensus]; recall-process signals [cheang2025recall].
  - **Genuinely uncertain → informative.**
- **Architecture.** Off-the-shelf open models:
  - Qwen2.5-7B/Qwen3-8B-Instruct and Llama-3.1-8B-Instruct (primary and replication);
  - Gemma-2-9B-it (Gemma Scope SAEs and entity latents [ferrando2025entity]);
  - a Qwen model with released J-lens matrices (Stage 2 bridge);
  - Ouro-1.4B/2.6B and Huginn-3.5B (Stage 3 recurrence).
- **Training method.**
  - Stage 1: none.
  - Stage 3: a monitor (LoRA, or probe plus learned verbal readout) trained with *counterfactual* labels: post-lesion correctness from lesion family A. Tested on held-out families B–D.
- **Experiments:**
  1. **Lesion-tracking (Stage 1).** Entity–attribute questions with popularity metadata. Readouts:
     - prospective verbal FOK (logit Yes − No, answer not yet generated);
     - retrospective P(True);
     - answer log-probability;
     - frozen linear probe on pre-answer residuals;
     - familiarity-direction projection;
     - abstention.

     Lesions (input tokens identical in all cases):
     - **L2** attention knock-out of attribute extraction (last token → subject, upper layers only) [geva2023dissecting]: blocks recall, spares subject familiarity;
     - **L3** targeted unlearning of a forget set (gradient-ascent/NPO on answer tokens, *no* refusal text) [li2024wmdp]: a weight-level, learning-induced change;
     - **L4** familiarity lesion (ablate the known-entity direction at the subject): the reverse dissociation;
     - **L5** sham lesions (matched-norm random directions; knock-out to non-subject tokens);
     - **L6** global degradation (quantisation, weight noise), for comparison with [cohen2026source];
     - dose–response over lesion strength.
  2. **Mechanism (Stage 2).**
     - Locate the FOK pathway with attribution and patching.
     - Test E4: a monitor direction that mediates FOK, does not encode answer identity, and whose ablation lowers type-2 sensitivity with accuracy spared (an "artificial metacognitive blindsight").
     - J-space C2 test: are uncertainty concepts present in the J-space, and do they track lesions?
     - Tip-of-the-tongue analogue: under partial lesions, is FOK predicted by the amount of partial answer information that is decodable, as the accessibility model predicts?
  3. **Development and architecture (Stage 3).**
     - (a) Trained self-monitors: does training on lesion family A generalise to unlearning, quantisation, novel items, and novel lesion sites? Compare monitors trained on (i) unlesioned correctness (standard calibration), (ii) input-difficulty labels, (iii) counterfactual lesion labels. SOMA predicts that only (iii) yields generalising self-tracking.
     - (b) Recurrence: vary loop count in Ouro/Huginn at matched accuracy. Does self-tracking improve?
  4. **Optional Stage 4.** Perceptual "artificial blindsight" in a VLM: lesion the visual route. Does "Do you see X?" track forced-choice performance?
- **Controls:**
  - sham lesions of matched norm and site count (rules out generic anomaly detection [lederman2026content; singh2026reality]);
  - an input-only observer (another model, or a probe on layer-0 embeddings) predicting correctness: it calibrates without tracking lesions, and serves as the E1-without-E3 reference;
  - response-bias correction: type-2 measures; symmetric Yes/No prompts; forced-choice formats [hahami2025disturbance];
  - answer-identity control for probes [jspace_gemma_repo];
  - lesion selectivity: collateral accuracy on unlesioned items;
  - multiple prompt templates and models.
- **Ablations:** lesion type × layer band × strength; readout type; prompt format; base vs instruct model (post-training dependence [macar2026mechanisms]).
- **Dependent variables:**
  - *tracking index*: post-lesion type-2 discrimination of lesion-induced errors ÷ baseline type-2 discrimination of natural errors;
  - slope β of ΔFOK on Δ log p(correct) across items (mixed-effects);
  - anosognosia rate;
  - Δabstention.
- **Expected observations (prior, ~55/45).**
  - Prospective verbal FOK shows substantial anosognosia (β small) under L2.
  - Retrospective/logit confidence tracks partially, through first-order readout.
  - L4 lowers FOK with forced accuracy largely spared.
  - Unlearning (L3) shows intermediate tracking.
- **How it fails / negative evidence:**
  - Lesions cannot be made selective (collateral damage), so item-level analysis is impossible.
  - Baseline type-2 sensitivity is too low in 7–9B models for tracking to be measurable.
  - Sham lesions move FOK as much as real ones: generic anomaly, not self-tracking.
  - FOK tracks via surface changes in the generated answer rather than an internal signal (handled by prospective readouts).
  - **Each of these is a result about the method, not about the hypothesis, and is reported as such.**
- **Compute.**
  - Stage 1: ≈ 10–15 GPU-h.
  - Stage 2: ≈ 15–30 GPU-h.
  - Stage 3: ≈ 20–60 GPU-h (LoRA on 7–8B; loops on ≤3.5B models).
- **Difficulty:** moderate.
- **Novelty:** medium–high, *pending full-text checks* (§3.10 G1).

### P2. Workspace dynamics under graded evidence and competition (GNW decisive tests in open models)

- **Hypotheses:**
  - **H2a:** in feedforward transformers, J-space entry is graded with evidence strength within a pass. Apparent all-or-none behaviour comes from attractor-like competition in mid layers, or only at readout [lammuir2026ignition].
  - **H2b:** dual-task interference. Holding *k* concepts "in mind" reduces J-space entry and flexible use of a new implied concept, in proportion to *k*, while automatic processing is spared (central bottleneck).
  - **H2c:** looped models show bistability along the loop dimension.
- **Motivation:** the decisive tests explicitly requested by GNW proponents [dehaene2026commentary]. GNW predicts ignition and a bottleneck; transformers might deliver access *without* them, which would separate GNW's functional and dynamical claims.
- **Architecture:** Qwen models with released J-lens matrices; we fit our own lenses for Ouro/Huginn (companion code).
- **Experiments:**
  - graded cues (implication strength; embedding interpolation);
  - across-"trial" variability from paraphrases, small activation noise, or sampling;
  - Hartigan dip tests for bimodality;
  - load manipulation (1–4 held concepts);
  - local–global and trace-conditioning analogues [bekinschtein2009localglobal; clark1998trace].
- **Controls:**
  - random and non-J-space subspaces of equal dimension;
  - pre-softmax linear quantities (to avoid readout-induced bimodality);
  - multiple trial-noise sources (bimodality must not depend on one arbitrary noise model).
- **Ablations:** J-space ablation vs matched non-J-space ablation; layer bands.
- **Expected:** H2b moderate (the reported dual-task degradation is "moderate"); H2a uncertain.
- **Failure modes:** J-lens artefacts and single-token limits [nanda2026review]; "trials" in a deterministic network are a modelling choice.
- **Compute:** 10–30 GPU-h.
- **Difficulty:** moderate.
- **Novelty:** low–medium. **Scoop risk high:** well-resourced groups are on it. Value: open-model replication with rigorous, preregistered criteria.

### P3. Recurrence as an engineering pathway: masking analogues and self-monitoring in looped vs feedforward models

- **Hypothesis:** continuous top-down re-entry (looped/feedback) enables two things:
  - (a) *masking susceptibility*: interrupting re-entry after *k* steps abolishes global availability (use by many readouts) while local decodability is spared. This is the RPT/GNW contrast; feedforward models have no re-entry to interrupt.
  - (b) better self-tracking (link to P1).
- **Motivation:** recurrence is the most-cited "missing ingredient" [chalmers2023llm; aru2023feasibility; dehaene2026commentary], and it is now cheaply manipulable.
- **Architecture:**
  - within-model loop count (Ouro, Huginn);
  - from-scratch 20–100M models matched for parameters and FLOPs: deep feedforward, looped (weight-shared), feedback transformer, looped plus bottleneck.
- **Training:** synthetic multi-hop/pointer tasks with "mask" inputs; ≥5 seeds per architecture.
- **Experiments:** masking-onset curves; global-availability index (number of distinct readout tasks solvable from the state); P1 lesion-tracking as a function of loops.
- **Controls:**
  - compute- and parameter-matched deeper feedforward models;
  - an **unfolding control**: untie the looped model's weights after training. It is I/O-equivalent, so any functional signature must persist. The honest conclusion is therefore about re-entry as a *trained inductive bias*, never about necessary structure [doerig2019unfolding].
- **Failure:** effects reduce to "more compute = more information"; synthetic tasks don't transfer.
- **Compute:** 20–100 GPU-h.
- **Difficulty:** high.
- **Novelty:** medium.

### P4. Implicit attention schemas in transformers

- **Hypothesis:** pretrained LLMs contain a compact representation of their *own current allocation of attention over context sources* that:
  - predicts allocation beyond content;
  - is causally used for subsequent allocation (endogenous control);
  - drives verbal reports of "what I relied on".

  When schema and attention are decoupled by steering, reports follow the schema and control degrades (AST). The alternative: "attention reports" are inferences about which document is relevant (world-tracking).
- **Motivation:** AST makes concrete engineering predictions [graziano2015ast; wilterson2021ast]. Existing work builds explicit modules [farrell2024ast; asac2025]; implicit schemas in pretrained LLMs appear untested.
- **Experiments:**
  - (1) multi-document prompts; allocation vector *a* over segments from attribution; probes for *a* at the query position vs a content-only baseline;
  - (2) steer the schema direction and measure shifts in allocation;
  - (3) schema–attention decoupling and reports;
  - (4) attention lesions (head ablation): does the schema update? (E3 for attention);
  - (5) training arm: auxiliary self-attention-prediction loss in small transformers vs predicting *another* network's attention.
- **Controls:** content-only probes; other-model attention; sham steering.
- **Failure:** "attention" has many heads and layers, and attribution methods disagree; the schema may be indistinguishable from content.
- **Compute:** 5–20 GPU-h.
- **Difficulty:** moderate–high.
- **Novelty:** medium–high.

### P5. Decomposition → integration, reformulated (the human researcher's seed)

**Assessment of the seed.**

| Element | Verdict | Reason |
|---|---|---|
| Decompose consciousness into separately measurable capacities | **Keep** (strongly) | Matches best practice: separate indicators, no composite score [butlin2025indicators] |
| Train capacities in specialised environments | **Modify** | In a shared pretrained base, capacities share representations and are not independent. Verbal training targets invite imitation (gaming problem). Train in environments where the *target is the model's own variation* (e.g., counterfactual lesion labels), not human-written self-talk |
| Integrate via LoRA merging / parameter averaging / distillation | **Mostly reject as a consciousness pathway** | Merging edits the weights of the *same* feedforward computation. It adds no re-entry, no broadcast bottleneck and no new information-flow topology, which are what theories care about. Merging does compose *skills* [prabhakar2024lorasoups; akiba2025evolutionary], but that is compositional generalisation, not a consciousness-relevant property |
| Integrate via workspace / recurrent communication / routing | **Keep** | These change information flow in theory-relevant ways (GNW, RPT, AST) |
| "More than the sum of parts" | **Keep the question, change the test** | Must be (i) pre-registered target behaviours, (ii) continuous metrics [schaeffer2023mirage], (iii) interaction terms in a factorial design against a multiplicative-composition null [okawa2023multiplicative], (iv) compute-matched nulls. Theories predict *specific* dependencies (HOT: monitor ← first-order states; GNW: C2 needs C1; AST: control needs schema). They do not predict generic synergy |
| The capacities listed | **Narrow** | Working memory, episodic memory and predictive modelling are general cognitive functions. Their co-presence is not what any major theory identifies with consciousness. What matters is the *relations* among them |

- **Hypothesis (reformulated):** for capacity pairs where a theory predicts a dependency, integration produces superadditive gains on pre-registered, *never-trained* target behaviours. For theory-unrelated pairs, effects are additive. Falsified if interactions are null, or equally large for unrelated pairs (= generic capability composition).
- **Architecture:** a small controllable agent or LM (10–100M), or a 1–3B LM with modules:
  - M1 workspace bottleneck (routing among modules);
  - M2 self-model (predict own hidden states [premakumar2024selfmodel]);
  - M3 second-order monitor [pasquali2010know];
  - M4 episodic memory.
- **Design:** 2⁴ = 16 configurations × ≥5 seeds, with parameter- and FLOP-matched "null modules".
- **Developmental IV:** separate-then-integrate vs joint-from-start vs curriculum orderings (SOMA [cleeremans2020learning]).
- **Held-out target behaviours:**
  - confidence-guided information-seeking on novel tasks;
  - **E3 self-tracking of lesions to its own modules** (shares P1's test);
  - blindsight-like dissociations under monitor lesion;
  - reports of internal conflict.
- **Failure:** toy results judged irrelevant to LLMs; design arbitrariness; small effects.
- **Compute:** low–moderate (toy) or moderate (1–3B).
- **Difficulty:** high (environment design).
- **Novelty:** medium (closest: [phua2025ablations]).
- **Cheapest falsification:** Pilot C (merging test).

### P6. Continual learning and persistent self-models

- **Hypothesis:** continual weight updates without self-model maintenance cause *self-model drift*: self-prediction privileged access (E2/E3) decays. A self-model objective maintained during continual learning preserves E2/E3 and stabilises identity-related directions [chen2025persona].
- **Motivation:** this tests, at Level 1, a functional consequence of continual-learning theories [hoel2025disproof; ginsburg2019soul] and Type-B BN claims [klatzmann2026biology].
- **Experiments:**
  - sequential fine-tunes altering behaviours [binder2024looking] and knowledge (P1 L3);
  - with vs without a self-prediction objective;
  - track self-vs-cross prediction and persona-vector drift.
- **Controls:** cross-model predictors; shuffled-history training.
- **Failure:** tracking appears regardless (cf. introspective coupling [guo2026coupling]), making the objective irrelevant.
- **Compute:** 20–60 GPU-h.
- **Novelty:** low–medium.

### P7. Self/world boundary: efference-copy-like authorship signals

- **Hypothesis:** models represent their intended continuation in a way that lets them detect externally inserted tokens *beyond* surprisal and style.
- **Status:** prefill awareness and self-authorship are already well studied [wang2026prefill; ackerman2024selfrec; ranjan2026reality; lindsey2025introspection].
- **Novelty:** low. **Parked**, but its "matched-surprisal" control is reusable.

---

## 5. Comparison

Ordinal judgments (H = high, M = medium, L = low), with "↑" marking the favourable end. They are judgments, not measurements.

| | Importance ↑ | Testability ↑ | Novelty ↑ | Feasibility ↑ | Compute need ↓ | Confound risk ↓ | Interpretability ↑ | Paper strength ↑ |
|---|---|---|---|---|---|---|---|---|
| **P1** Self-tracking lesions | H | H | M–H (pending) | H | L | M (designed controls) | H | **H** |
| P2 GNW decisive tests | H | M–H | L–M (scoop risk) | H | L | M–H (J-lens, trial-noise model) | M | M |
| P3 Recurrence pathway | M–H | M | M | M | M | H (compute; unfolding) | M | M |
| P4 Implicit attention schema | M | M | M–H | M | L | H (defining "attention state") | M | M |
| P5 Factorial integration | M–H | M–H | M | L–M | L–M | M | H in toy / L external validity | M (H if combined with P1 tests) |
| P6 Continual self-models | M | M | L–M | M | M | H | L–M | L–M |
| P7 Authorship | L–M | H | L | H | L | M | M | L |

**Narrative:**

- **P2 is the most consciousness-central test**, but it is crowded and depends on a debated tool. Its signatures may also be trivially present or absent because of transformer architecture.
- **P1** addresses the evidential bottleneck that limits interpretation of *all* self-directed indicators:
  - it is cheap and falsifiable in both directions;
  - its negative outcome (anosognosia) is as publishable as its positive one;
  - it supplies the test that P3, P4, P5 and P6 all need (E3).
- **P5** is the right home for the seed idea, but on its own it risks a "toy" verdict. Attaching P1's E3 test as its held-out target behaviour fixes that.

---

## 6. Preferred direction (provisional)

### 6.1 Choice

**P1 is the core program, with three bridges:**

- P2 (is C2 in the workspace?);
- P3 (does recurrence improve self-tracking?);
- P5 (do *integrated, trained* monitors generalise self-tracking to held-out lesion types?).

**Working research question.** *Do language models' self-assessments depend counterfactually on their own internal states, i.e., do they track interventions on their own competence when the input is held fixed? If so, through what mechanism, and can training or architecture (recurrence, an integrated monitor) produce self-tracking that generalises to kinds of self-change never seen in training?*

### 6.2 Why this direction

1. **It targets the field's actual bottleneck.** Introspection and metacognition claims are stalled on privileged-access and second-order-computation objections [singh2026reality; song2025privileged; ferrara2026owmi]. Lesion designs satisfy privileged access *by construction*, because the input is identical, and they pit self-monitoring against difficulty-model and readout accounts.
2. **It is theory-plural:**
   - HOT, GNW-C2 and AST all require E3;
   - SOMA motivates the developmental arm;
   - human FOK and anosognosia research supplies priors and analogues.
3. **It is cheap, fast and falsifiable in both directions.** It runs on 7–9B open models and gives clear outcome tables (§6.4).
4. **It produces a reusable instrument** (the tracking index and the E-tier battery) that applies to attention schemas (P4), workspace C2 (P2) and continual self-models (P6).
5. **Its relevance extends beyond the consciousness community.** Reliability of models that know their limits after quantisation, unlearning and fine-tuning [gu2026unlearners; cohen2026source] widens the paper's audience without changing its claims.
6. **Level discipline is easy.** Results are Level 1. The Level-2 mapping (to HOT-2 / GNW-C2 indicators) is explicit and conditional.

**Strongest objection.** "FOK about facts is not perceptual consciousness. HOT concerns perceptual states."

- Response 1: E3 is a general criterion. We start with factual recall because circuits are mapped there [geva2023dissecting; meng2022rome] and because privileged self-information exists there [ashuach2026consensus].
- Response 2: Stage 4 moves to perceptual detection in a VLM (artificial blindsight).
- Response 3: the paper will state the scope limitation plainly.

### 6.3 Hypotheses, stated so they can fail

- **H1 (Stage 1):** under recall lesions that spare familiarity (L2), the tracking index of prospective FOK is < 0.3. That is anosognosia, supporting H1-world. *Falsified if* the tracking index is ≥ 0.6 and exceeds sham in both models.
- **H2:** retrospective confidence tracks more than prospective FOK (first-order readout).
- **H3 (double dissociation):** the familiarity lesion (L4) lowers FOK while forced-choice accuracy is spared (≤ 25% of the FOK drop in standardised units).
- **H4 (E4, Stage 2):** a monitor direction exists whose ablation lowers type-2 AUROC by ≥ 0.1 with accuracy change ≤ 2 points, and from which answer identity is not decodable above a matched control.
- **H5 (Stage 3, developmental):** monitors trained with counterfactual lesion labels on family A track held-out lesion families better than monitors trained on natural correctness or input difficulty, at matched in-distribution calibration.
- **H6 (Stage 3, recurrence):** in looped models, the tracking index rises with loop count at matched accuracy.

Thresholds are provisional and will be fixed in the preregistration after Pilot A's variance estimates.

### 6.4 Outcome → interpretation

| Outcome | Interpretation (Level 1) | Level-2 consequence (conditional) |
|---|---|---|
| Anosognosia (low tracking) + L4 dissociation | Prospective FOK is a cue-familiarity signal, not self-monitoring of recall | LLM confidence/FOK should not count toward HOT-2/C2 indicators. Self-report-based assessments need E3 tests |
| Tracking > sham, item-specific | Self-assessments depend counterfactually on the system's own recall state | Necessary condition for C2/HOT-style monitoring met for this domain. E4 is next |
| Tracking only for retrospective readouts | Monitoring is a first-order readout of output strength | Meets E3, fails E4: "metacognition without higher-order representation" |
| Sham ≈ real | Generic anomaly sensitivity | Consistent with [lederman2026content; singh2026reality] |
| Trained monitor generalises to held-out lesions (H5) | Self-tracking can be *learned as a general capacity* | Supports a SOMA-style developmental pathway (I7) |
| No generalisation | Monitors learn lesion-specific readouts | Against "train capacities, integrate, get general self-monitoring" |

### 6.5 Pivot criteria

- **If** full-text reading shows that [cohen2026source] or [park2026esma] already did item-level, input-preserving, mechanism-targeted lesion tracking: drop Stage 1 novelty claims, build on their data, and lead with Stage 2 (E4) and Stage 3 (generalisation).
- **If** Pilot A's kill criteria trigger (§7): move to the strongest alternative (P4, or P5 with E3 as the target).
- **If** a major lab publishes C2-in-workspace results first: keep the lesion methodology and frame it as an independent open-model test.

---

## 7. Decisive pilot experiments

All pilots will be pre-registered in `research/experiments/pilots/pilot_protocols.md`, frozen before data collection. Raw outputs will be saved as JSONL in `research/results/raw/`. Full protocols are in that file; summaries follow.

**Pilot A — lesion-tracking (P1 core). ~10–15 GPU-h. Answers: does prospective FOK track input-preserving recall lesions?**

- **Models:** Qwen2.5-7B-Instruct (primary); Llama-3.1-8B-Instruct (replication).
- **Items:** ~2,000 Wikidata entity–attribute questions across popularity bins, plus ~200 fictitious entities.
- **Conditions:** base; L2 at 5 strengths; L4; L5 sham × 2; L6 4-bit.
- **Primary DV:** tracking index for prospective FOK under L2; β of ΔFOK on Δ log p(correct).
- **Go criteria:**
  - baseline FOK type-2 AUROC ≥ 0.65 in at least one model;
  - L2 makes ≥ 30% of base-correct items fail;
  - matched sham (L5a) changes accuracy by ≤ 2 points;
  - general language damage from a matched-span knock-out is small (perplexity increase ≤ 10%).
- **Kill / redesign:** any go criterion fails in both models → redesign lesions or move to 14B in 4-bit before scaling.
- **Either H1 outcome counts as success** of the pilot.

**Pilot B — C2 in the workspace, plus a mini ignition test (P1×P2 bridge). ~5–8 GPU-h.**

- **Model:** a Qwen model with released J-lens matrices (to be identified).
- **Part 1:**
  - Do uncertainty or "don't know" concepts appear in the J-space at the question's end for unknown vs known entities?
  - Do they change for L2-induced errors?
  - Go if a J-space uncertainty score separates known from unknown entities (AUROC ≥ 0.7).
- **Part 2:**
  - Graded two-concept mixtures (replicating the snapping result) with three trial-noise sources.
  - Dip test for bimodality at threshold, J-space vs matched non-J-space subspace.
  - Informative either way.

**Pilot C — does integration by merging produce superadditive, untrained self-monitoring? (Cheap falsification of the seed's mechanism.) ~6–10 GPU-h.**

- **Model:** Qwen2.5-1.5B-Instruct.
- **Adapters:**
  - A: trained to give confidence calibrated to the model's own correctness on trivia domain X;
  - B: trained for a new skill on domain Y.
- **Conditions:** base, A, B, merged (task arithmetic / TIES / CAT), joint, sequential; 5 seeds.
- **DV:** confidence calibration on Y, which was never trained for confidence (type-2 AUROC, Brier); plus Pilot A's tracking test on the merged model.
- **Test:** 2×2 interaction against a multiplicative null.
- **Kill:** merged ≤ best single adapter and interaction CI includes 0 → drop merging as an integration mechanism. Keep the factorial logic for architectural modules (P5).

**Pilot D — implicit attention schema feasibility (P4). ~3–5 GPU-h.**

- **Model:** Qwen2.5-1.5B and 7B.
- **Task:** multi-segment prompts; allocation vector from attribution.
- **Go if:**
  - a residual-stream probe at the query position predicts allocation beyond a content-only baseline (ΔR² ≥ 0.10);
  - steering the probe direction shifts allocation (d ≥ 0.5) versus sham.
- **Otherwise:** park P4.

**Sequencing:**

1. Read the closest papers in full (§9 step 1).
2. Pilot A.
3. Pilots B and D (cheap, parallel).
4. Pilot C.

Decisions after each pilot are recorded in `logs/decisions.md`. **Budget:** ≈ 30–45 GPU-h, roughly $50–150 at typical single-GPU rental rates. All code is first debugged locally on 0.5–1.5B models on CPU.

**Statistical standards (all pilots and later experiments):**

- preregistered DVs and thresholds;
- mixed-effects models with item and template random effects;
- item-bootstrap 95% CIs;
- ≥ 2 models for any claim;
- ≥ 5 seeds wherever training is involved;
- Holm correction across the pre-specified family;
- every condition reported, including failed lesions;
- raw generations and activations summaries preserved;
- code and config hashes logged.

---

## 8. Ethics and risk

- The experiments manipulate small open models in ways (lesions, unlearning) that we judge to raise no credible welfare concern at this scale. We nevertheless follow the five principles of [butlin2025principles], document that judgment, and do not build valence/affect systems (decision D8).
- Communication risk: results about "self-monitoring" are easily over-read as consciousness claims [bengio2025illusions]. Every write-up uses the claims table (§0).
- Dual use: lesion-tracking tools could help detect or hide model degradation. Low risk; we will release them openly.

---

## 9. Immediate next steps (in order)

1. **Full-text novelty check (G1):** [cohen2026source], [park2026esma], [liu2026metacogsurvey], [singh2026reality], [macar2026mechanisms], [ferrara2026owmi], [gurnee2026workspace] (C2-relevant sections), [phua2025ablations]. Update `literature_db.json` and the gap table.
2. Verify all `memory-verify` and `partial` bibliography entries that will be cited.
3. Identify which open Qwen models have released J-lens matrices; confirm licences of Ouro/Huginn/Gemma Scope.
4. Freeze the Pilot A preregistration; set up a cloud GPU; implement the harness and debug on Qwen2.5-0.5B on CPU.
5. Run Pilot A; decision point.

---

## 10. Novelty Audit After Closest-Competitor Review (added v0.2, 2026-10-01)

### 10.1 What was done

- **Git provenance initialised.** Commit `5ff787a` holds the v0.1 state.
- **Full-text retrieval of Cohen & de Melo (ICML 2026) [cohen2026source] was attempted through every legitimate route available:**
  - OpenReview forum and API: blocked by a Cloudflare browser check, which we do not bypass;
  - PMLR/ICML virtual pages: not listed in the session pages fetched;
  - arXiv: not found under the title or in de Melo's 2025–26 arXiv listing;
  - Semantic Scholar API: rate-limited;
  - papers.cool: not indexed;
  - author pages.

  The best available source is a structured secondary summary (Lacuna) plus search-engine snippets. **The verdict in §10.2 is therefore provisional until the PDF is read.**
- **~35 targeted searches** across the topic combinations requested (§10.8).
- **CPU feasibility measurements** on the researcher's laptop: Intel Core Ultra 7 255U, 31 GB RAM, no CUDA. See `decision_document_v2.md`.
  - Measurement only: throughput benchmarks, plus training-time-only memorisation of a throwaway synthetic world (seed 12345, never to be reused).
  - No monitor, lesion or hypothesis was examined.

### 10.2 Cohen & de Melo vs the stronger causal design (Q1)

**Reconstructed design** (secondary source; to be confirmed):

- **Six "source of competence" regimes:**
  - strong parametric memory;
  - evidence-grounded (RAG);
  - deliberative reasoning (CoT);
  - partial/stale memory (rare or outdated facts, e.g., CEO changes);
  - heuristic pattern matching (shortcuts / reduced-model approximations);
  - computation-degraded (4/8-bit quantisation, or noise injected into internal states).
- **Data:** TriviaQA, NQ, LAMA-style probes, GSM8K, StrategyQA, CommonsenseQA, SQuAD, partitioned by prominence.
- **Confidence channels:** verbalised, token log-probability, self-consistency.
- **Key results:**
  - matched-accuracy divergence (~61% accuracy: partial memory has a +27-point confidence gap vs +5 points for evidence-grounded);
  - stale facts: +33 points;
  - a "danger zone" under degradation.
- **Self-described scope:** behavioural, not internal, metacognition.

| Element of the stronger design (user's list) | Cohen & de Melo (reconstructed) | Other prior work found | Status |
|---|---|---|---|
| 1. Model knows item X | Item regimes chosen by prominence (between-item) | [gu2026unlearners] (known WMDP items), [hasegawa2025underconf] | Done elsewhere |
| 2. Measure accuracy, explicit, token, internal | Accuracy, verbal, token, self-consistency; **no internal** | [gu2026unlearners] (refusal, first-token entropy) | Partly done |
| 3. Selectively damage competence for X | **Global** degradation (quantisation/noise), not item-selective | **[gu2026unlearners]: item-selective unlearning; [hasegawa2025underconf; hasegawa2026jnlp]: item-selective editing** | **Done (unlearning/editing)** |
| 4. Input unchanged | Yes for degradation | Yes in Gu, Hasegawa | Done |
| 5. Verify knowledge reduced, neighbours preserved, same difficulty | Not item-level | Gu: forget accuracy + retain/MMLU utility | Done (behavioural) |
| 6. Re-ask the same question | Yes (degraded) | Yes (Gu, Hasegawa) | Done |
| 7. Self-assessment changes specifically for X? | Aggregate confidence gap | Gu: rejection on forget set (mostly **fails**: confabulation); Hasegawa: token confidence partially tracks edits (underconfidence) | **Done behaviourally** |
| 8a. Sham lesions | — | — | **Not found** |
| 8b. Unrelated lesions | — | (retain-set utility ≠ unrelated-lesion control) | **Not found** |
| 8c. Matched input-corruption errors | — | — | **Not found** |
| 8d. Global degradation | **Yes** | — | Done |
| 8e. Naturally difficult items | Yes (prominence regimes) | — | Done |
| Identification question (self-change vs observable difficulty), internal monitors, control endpoints | No | Familiarity gating of refusal is mechanistically shown [ferrando2025entity; lindsey2025biology]; "no-answer" signal not routed to refusal [du2026recognition] | **Not found as an identification study** |

**Verdict (Q1).**

- **P1 in its current form is rejected.** "Selectively damage competence, re-ask the same question, check self-assessment" has been done behaviourally:
  - unlearning → confabulation rather than honest refusal [gu2026unlearners];
  - editing → token-confidence underconfidence [hasegawa2025underconf; hasegawa2026jnlp];
  - globally degraded systems → a confidence danger zone [cohen2026source];
  - the likely mechanism (familiarity-gated refusal) is causally demonstrated [ferrando2025entity].

  Changing the endpoint from verbal to token confidence, or adding more models, would be a cosmetic rescue. We decline to do that.
- **What survives:** the *identification contrast set* (8a–8c + 8e at matched error rates) and the question of *which* control signal tracks self-change. This is genuinely missing, but incremental. It survives as **P1\*** (a bridge study) and, more importantly, as the measurement backbone **N1** (decision document).

**Checklist to confirm against the PDF** (any "yes" further weakens P1\*):

- (a) item-selective interventions on model weights or activations?
- (b) sham or matched-norm controls?
- (c) input-corruption controls?
- (d) internal-representation or probe analyses?
- (e) control endpoints such as abstention or retrieval decisions?
- (f) any synthetic/trained-from-scratch system?

### 10.3 Other closest competitors identified in this audit

| Work | What it does | Relevance |
|---|---|---|
| Gu et al. 2026, ACL [gu2026unlearners] | 9 unlearning methods (WMDP-Bio): rejection, multi-turn stability, IDK-choice with position control; ReVa aligns forget-set activations with a refusal direction | Behavioural "anosognosia" after selective lesions is established |
| Hasegawa et al. 2025/2026 [hasegawa2025underconf; hasegawa2026jnlp] | Within-item editing → token-confidence calibration (10 LMs) | Within-item intervention + confidence is established (insertion direction) |
| Yax, Palminteri & Oudeyer 2026 [yax2026forms] | Training LLMs to predict own accuracy yields output-consistency tracking (generalises) vs accuracy tracking (local) | Key for P5\*: a trained "monitor" may be a first-order readout. Lesion tests separate the two |
| Li et al. 2026 [li2026functionalmeta] | Steerable functional-metacognition directions (self-assessed capability, effort) | Internal metacognitive variables causally control behaviour, but never tested against competence changes |
| Zhao et al. 2026, COLM [zhao2026wired] | Circuits inflating verbal confidence | Verbal confidence is a poor E3 endpoint |
| Moskvoretskii et al. 2025, ACL; Marina et al. 2025 [moskvoretskii2025adaptive; marina2025llmindep] | Retrieval-trigger "self-knowledge" is weak; question-only features match LLM uncertainty | World-tracking is observationally sufficient on natural data. **Motivates N1** |
| Du & Hu 2026 [du2026recognition] | "No admissible answer" encoded but orthogonal to refusal | Monitoring present but not routed to control (anosodiaphoria-like) |
| Xie 2026 [xie2026structural] | Self-monitoring helps only on the decision pathway; parameter-matched null comparable | Closest to P5\* integration; strong reason for matched nulls |
| Tomaszewski 2026 [tomaszewski2026sil] | Networks perturb themselves to learn predictive self-models that generalise to unexecuted interventions | Closest to P5\*'s self-variation regime; different target (structural consequences, not knowledge monitoring) |
| Nishi et al. 2025; Zucchet et al. 2025; Allen-Zhu & Li 2024 [nishi2025shattering; zucchet2025facts; allenzhu2024physics31] | Synthetic knowledge worlds for transformers; editing shatters representations; new facts corrupt old ones | Validated substrate and known lesion side-effects for P5\* |
| Community J-lens work [anthropic_jlens_code; bcywinski_jlens_qwen9b; idhan_jlens4b; hasin_jspace_qwen; precommit_lens] | Open lenses (4B–27B+); ignition and dual-task experiments planned on Kaggle GPUs; internal readouts often no better than surface baselines | P2 is crowded; smallest J-lens target is 4B |

### 10.4 Upgrading P1 to functional self-monitoring (Q2)

Candidate endpoints for "downstream cognitive control changes selectively after an input-invisible competence intervention":

| Endpoint | Prior use | Lesion-tested before? | $0 feasibility | Assessment |
|---|---|---|---|---|
| Hidden-state correctness/familiarity probes | Many [ferrando2025entity; ashuach2026consensus; cheang2025recall] | No (as far as found) | Yes (0.5–1.5B) | Necessary for E4. Must control for answer-identity leakage |
| J-space contents | [gurnee2026workspace] | No | Marginal (≥4B) | P2\* |
| Abstention | [gu2026unlearners] (after unlearning) | **Yes** (behavioural) | Yes | No longer novel alone |
| Information seeking / tool or retrieval request | Adaptive RAG [moskvoretskii2025adaptive; yao2024seakr] | **No** | Yes (prompted LOOKUP option; native in P5\* controller) | **Best behavioural endpoint** |
| Allocation of reasoning compute | Looped-model exit gates [zhu2025ouro]; effort directions [li2026functionalmeta] | **No** | Yes (Ouro-1.4B, ~5 s/prompt) | Elegant non-verbal endpoint; requires `trust_remote_code` review |
| Error correction / willingness to revise | Self-correction literature | No | Partly (needs multi-turn generation) | Secondary |
| Before-vs-after differential behaviour | Gu (rejection); Hasegawa (confidence) | Yes (those endpoints) | — | The *contrast set*, not the endpoint, is what's new |

**Conclusion.** The strongest form is "after an input-invisible competence lesion, the system *requests external information* or *allocates more recurrent compute* specifically for lesioned items, and not under sham or input-matched controls."

This is best established **first in P5\***, where:

- the controller and its information access are designed;
- ground truth about what changed is known.

It is then tested in small pretrained LMs (P1\*).

### 10.5 P2 after J-space (Q3)

| Remaining GNW prediction | Status in literature | Strongest open test | $0 feasible on this laptop? |
|---|---|---|---|
| Sharp/non-linear ignition with graded stimulus strength | Snapping for ambiguous mixtures [gurnee2026workspace]; weak probe-sigmoid index [rahbar2026ignition]; readout-localised ignition in a tiny looped model [lammuir2026ignition]; planned on Qwen-9B [hasin_jspace_qwen] | Stimulus-strength sweep + across-trial bimodality vs matched non-J-space subspace | Marginal (4B, small N) |
| Dual-task / central bottleneck | "Moderate" cost [gurnee2026workspace] A.17; planned [hasin_jspace_qwen] | Load × entry curves | Marginal |
| Recurrence | Absent in standard transformers (acknowledged) | J-space in looped models (Ouro) | Requires fitting a lens on Ouro: expensive on CPU |
| Competition for access | Partial | Two-concept competition at matched strength | Marginal |
| Error monitoring / confidence / known-vs-unknown in workspace | Anecdotal ("damn" tokens) | **Does J-space content *govern* abstention/lookup?** (vs matched non-J-space ablation) | Marginal (4B); this is the most novel P2 test |

**Verdict.** P2\* (workspace → metacognitive control) is scientifically strong. It is **not robustly executable at $0** on this hardware, and ignition/dual-task tests face high scoop risk. **Deferred.**

### 10.6 P5 reassessment (Q4)

The seed survives once reformulated as A/B/C levels:

- **A.** Capacities developed separately: store, monitor, controller, memory, self-variation.
- **B.** Genuine information-flow integration: access, re-entry, broadcast bottleneck, decision-pathway placement.
- **C.** A pre-registered emergent property: selective tracking of *never-experienced self-changes* in control behaviour.

The factorial includes controls that block reduction to parameters, compute, data, ensembling, prompt length or leakage, plus a **yoked-twin self-specificity control**. Full specification is in `decision_document_v2.md` §P5\*.

Parameter merging is dropped as an integration mechanism. Pilot C remains available as a cheap side-test but is not needed.

### 10.7 New direction found in the audit: N1 (identification framework)

- **The gap.** The audit's strongest general finding: in natural data, world/difficulty tracking is *observationally sufficient*:
  - question-only features match LLM "self-knowledge" in retrieval decisions [marina2025llmindep];
  - cross-model confidence is rank-one [moran2026individuated];
  - internal readouts often add nothing over surface text [precommit_lens; singh2026reality].
- **N1 formalises why** with an observational-equivalence result.
- **It derives the minimal interventional design that identifies self-tracking:** sham, input-corruption, unrelated-lesion and yoked-twin controls.
- **It argues that privileged access (E2) can arise from internally computed familiarity**, and is therefore not sufficient for self-tracking.

N1 is merged into the P5\* paper as its measurement theory.

### 10.8 Adjacent-work search log (Q5, 2026-10-01)

- **Topics searched:**
  - selective knowledge lesion + confidence;
  - unlearning + metacognition / self-knowledge;
  - knowledge editing + confidence;
  - adaptive retrieval + self-knowledge;
  - artificial / machine anosognosia;
  - metacognitive dissociation / blindsight in ANNs;
  - confidence after ablation;
  - consciousness indicators + causal ablation;
  - modular consciousness-inspired architectures;
  - self-model + workspace;
  - structural integration of self-monitoring;
  - self-interventional learning;
  - synthetic knowledge transformers;
  - J-space follow-ups (lenses, replications, ignition, dual-task);
  - why unlearned models hallucinate;
  - recognition–refusal misalignment.
- **Main finds:** tabulated in §10.3; all are in `literature_db.json`, which now has 165 entries.
- **No hits** for "artificial/machine anosognosia" as a term in ML. The nearest is a neuroscience computational model [andrade2023dualpath], useful for the anosognosia vs anosodiaphoria distinction: monitor failure vs routing-to-control failure.

### 10.9 Zero-budget reassessment (summary)

| Direction | $0? | Min. model | RAM | CPU runtime | Training? | Scientific meaning at small scale |
|---|---|---|---|---|---|---|
| P1\* | Yes | 0.5B (1.5B preferred) | 3–7 GB | 5–30 h | No | Limited (floor effects); bridge only |
| P2\* | Marginal | ~4B (lens availability) | 8–10 GB (bf16) | 20–40 h, fragile | No (if lens valid) | Single model, third-party lens |
| **P5\* + N1** | **Yes** | 0.5–5M (toy) | < 2 GB | 30–70 h total incl. bridge | Tiny | **Full** (the science is the toy-scale mechanism) |

### 10.10 Novelty statements (careful wording)

- We have **not yet identified** prior work that tests whether metacognitive *control* (lookup/compute allocation) tracks input-invisible competence lesions against sham, unrelated-lesion and matched input-corruption contrasts.
- We have **not yet identified** prior work that manipulates shortcut availability (familiarity–knowledge correlation) and information-flow architecture factorially to map when counterfactually self-dependent monitoring emerges, with held-out self-changes as the emergent test.
- Both statements are pending full-text reading of: [cohen2026source; yax2026forms; tomaszewski2026sil; xie2026structural; phua2025ablations; gu2026unlearners; liu2026metacogsurvey].

---

## Appendix A. Search log and limitations of this review

- **Searches:** ~70 web searches and fetches on 2026-10-01. Topics:
  - indicators;
  - introspection and concept injection, including replications and critiques;
  - metacognition, privileged access and confidence;
  - unlearning and editing vs self-knowledge;
  - anosognosia and lesion analogues;
  - global workspace / J-space and its commentary and critiques;
  - ignition in LMs;
  - PCI/IIT in ANNs;
  - attention schema;
  - GW agents;
  - recurrence (Huginn, Ouro);
  - merging and skill composition;
  - continual learning and identity;
  - prefill awareness;
  - theory-status papers (Cogitate, IIT dispute, BN).
- **Code and replications noted (not peer reviewed):** github.com/solarkyle/jspace (Gemma 4, preregistered); github.com/Hasin-ai/jspace-qwen (Qwen3.5-9B replication); github.com/MRIIOT/jspace-scope.
- **Limitations:**
  - many 2026 items are read only at abstract level;
  - OpenReview pages could not be fetched (Cohen & de Melo details unverified);
  - some author lists are incomplete (flagged `partial`);
  - classic references are entered from memory (`memory-verify`);
  - search engines under-index very recent work, so the novelty assessments are provisional.
