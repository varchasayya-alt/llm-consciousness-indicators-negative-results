# Moonshot Research Decision Memo

**Date:** 2026-10-08
**Author:** Claude (lead research scientist role), for the PI
**Scope:** Research discovery only. Nothing was implemented, downloaded, trained or run. No prior experiment, protocol, result or decision was changed. C15 and C16 are treated as retired; their records were read only for evidence, failure modes and literature coverage.
**Decision requested:** see §H.

**Evidence conventions used in this memo:**
- **F**: full text read. Fetched pages were condensed by an automated summariser, so wording should be re-checked by direct reading before any citation in a paper.
- **P**: partial full text read (setup, theorem statements, discussion).
- **A**: abstract or landing page only.
- **S**: search snippets or third-party summaries only.
- **M**: entered from memory; must be verified.

Keys refer to `research/literature/literature_db.json`.

---

## 0. Bottom line

1. **No Category C (breakthrough) opportunity was found that has credible odds under our constraints.** I will not claim one.
2. **One direction is a genuine Category B candidate and appears open on current evidence: SM, a selection theorem for self-models.**
   - **The claim:** an agent must carry a self-model exactly when it must carry a change in *itself* across contexts.
   - **The flip side:** within a single context, "self-model" and "world-model" are observationally equivalent and nothing forces the former.
   - **The test:** this gives a principled, falsifiable behavioural test for self-models, the *cross-context self-change transfer test*.
   - **The prediction:** agents trained without self-non-stationarity, such as static LLMs, face no pressure to form self-models.
3. **SM has one specific, unresolved novelty risk.** Its core theorem may be a direct instance of Nayebi's (2026) regime-tracking corollary, with self-shift vs world-shift as the regimes.
4. **Decision: CONDITIONAL.** A bounded, pen-and-paper milestone (§F) can kill SM before any code is written.

---

## A. Executive verdict

**Is there a credible breakthrough-level opportunity under our constraints?** Not at a level I can defend.

The fields most relevant to consciousness are now heavily mined: workspace, metacognition, introspection, attention schema, latent reasoning, and the theory of out-of-context reasoning (OOCR). Two passes of targeted screening covered 16 first-principles candidates:
- **Nine were eliminated outright.** Each was already published, explicitly pre-empted by the theory's own authors, an application of known theory, or a measure without a target phenomenon.
- **Three more were absorbed into the leading candidate** as corollaries or predictions.

**What survives is a Category B theory-plus-toy-system programme (SM).** Its potential is real but bounded.

- **What it offers:**
  - A necessity-and-identifiability account of *when* a cognitive system must model itself.
  - A crisp boundary condition explaining why within-task introspection or metacognition tests cannot distinguish self-models from world-models. That boundary is also the structural root of this project's own P5\*/B1 identifiability failures.
  - A new behavioural test.
  - A constructive recipe for building agents that must form self-models.
- **Why it suits our constraints:** the decisive parts are mathematics plus tiny recurrent agents with exact Bayesian ground truth. That fits a CPU laptop, and it sidesteps the model-competence and measurement-site failures that sank C15 and C16.

**Confidence (qualitative; any numbers are subjective):**

| Question | Confidence | Reason |
|---|---|---|
| No self-model selection theorem exists in the literature | Moderate | Targeted searches through Oct 2026 found none. Every theorem in the line covers world models, belief states or protocol regimes. On Wentworth's LessWrong survey of selection theorems (2021), a reader asked about self-modelling theorems; Wentworth replied "None that I know of; it's a topic ripe for exploration". |
| SM's mathematical core is more than a re-labelling of Nayebi Cor. 4 or Richens & Everitt | Low–moderate | This is the unresolved risk. Subjective probability that the §F milestone passes: about 0.4–0.5. |
| SM reaches Category C impact even if everything works | Low | Subjective: about 0.05. A skeptic can fairly call it a formal consolidation of Kelley's covariation principle (1967/1973) and Berniker & Kording's (2008) body/world attribution model. |

---

## B. Candidate landscape (Pass 1 → Pass 2)

### B.1 What the project history teaches, in one paragraph

Across B1, D2, C15, C15-R2 and C16, every stop had the same structure:
- the system lacked competence (C16 S0);
- the hypothesised variable was absent at the measurement site (C16 R9/R10, C15 W3);
- an intervention created its own artefact (B1 R7);
- or an observational equivalence made the estimand unidentifiable (N1, D1; Smith et al. 2016).

The common cause was asking hidden-variable questions of systems whose ground truth was unknown. The lesson for selection:
- prefer questions whose answer is a **theorem** or a **ground-truth toy system**;
- require a **positive-control instrument** before any measurement;
- reject any primary effect that is **guaranteed by construction**.

### B.2 Sixteen candidates

Each entry gives:
- **Q:** the scientific question;
- **Why:** why it matters broadly;
- **Best:** the strongest plausible original contribution;
- **Prec.:** the closest precedent;
- **Test:** the smallest decisive experiment or result;
- **Alt.:** the strongest alternative explanation;
- **Feas.:** feasibility on our hardware;
- **Verdict:** the decision, with evidence level.

**1. SM: when must a system model itself?**
- **Q:** Which task demands *force* an agent to carry a variable that represents its own (changing) capacities, as distinct from the world?
- **Why:** Self-model theories (AST, Metzinger, HOT-style metacognition, interoceptive predictive processing) assert that self-models matter, but none has a necessity condition. AI introspection debates lack one too.
- **Best:**
  - a selection theorem showing that a self-model is forced iff the agent must project a self-change across contexts;
  - a within-context observational-equivalence boundary;
  - an "extended self" corollary;
  - toy agents testing internal realisation.
- **Prec.:** Richens & Everitt 2024 (P); Richens et al. 2025 (P); Nayebi 2026 (P); Cifuentes 2026 (A); Berniker & Kording 2008 (F); Kelley 1973 (M); Kwiatkowski & Lipson 2019 (A); Xing et al. 2026 (P).
- **Test:** a minimal-case proof (§F), then GRU agents with exact Bayes targets.
- **Alt.:** generic hierarchical-Bayes shared latents (it is "just" multi-task learning); inductive-bias reuse.
- **Feas.:** high.
- **Verdict:** **Top-1 (§D.1).**

**2. IR: when are self-reports faithful?**
- **Q:** Under what training regimes does a report come to depend causally on the internal state it reports, rather than on input proxies?
- **Why:** This bears on the validity of every LLM introspection claim.
- **Best:** a training-regime principle: reports track internal states only when those states vary independently of input during training.
- **Prec.:** Smith et al. 2016; Singh et al. 2026; Song et al. 2025; Zeng et al. 2026 (A); this project's N1.
- **Test:** toy reporters with and without input-independent internal variance; causal patching.
- **Alt.:** reuse bias makes reports internal anyway.
- **Feas.:** high.
- **Verdict:** **Folded into SM** as prediction P-IR. It is the same principle seen through a report head, and it has low novelty on its own.

**3. AN: are self-model blind spots inevitable?**
- **Q:** Are some perturbations of a self-monitoring agent necessarily undetectable from inside (anosognosia, misattribution)?
- **Why:** This links neurology and AI self-evaluation failure.
- **Best:** an observability-style limitative theorem.
- **Prec.:** the comparator account of anosognosia (Frith, Blakemore & Wolpert 2000, A); Breuer 1995 (M); Andrade 2023.
- **Test:** formal observability analysis; lesioning a toy agent's evaluation pathway.
- **Alt.:** a trivial "you can't check a checker with itself".
- **Feas.:** high.
- **Verdict:** **Folded into SM** as Conjecture C. Its conceptual core is known.

**4. CL: is the capacity limit a learning device?**
- **Q:** Does a selective, capacity-limited broadcast speed up the *learning* of cross-module coordination under local (non-backprop) credit assignment, but not under backprop?
- **Why:** It would rationalise workspace capacity limits as adaptations for learning, and say when artificial systems need them.
- **Best:** a learning-rule × capacity crossover interaction plus an analytic variance law for the optimal k.
- **Prec.:** AGREL (Roelfsema & van Ooyen 2005, A); Niv et al. 2015 (A); Goyal et al. 2022; Musslick et al. 2017, Sagiv et al. 2020, Petri et al. 2021, Musslick & Cohen 2021 (A/S); Clark et al. 2021 (A); Werfel et al. 2005 (M).
- **Test:** a linear-module node-perturbation analysis plus small simulations.
- **Alt.:** generic dimensionality reduction; Musslick's interference trade-off.
- **Feas.:** high.
- **Verdict:** **Top-3 (§D.2).** Partly anticipated.

**5. TA: toy models of access.**
- **Q:** For many consumers reading a shared, bandwidth-limited medium, when is the *optimal* code winner-take-all and serial (workspace-like), and when is it parallel superposition?
- **Why:** It is a normative test of whether workspace signatures are optimal at all.
- **Best:** a phase diagram over content sparsity, consumer count and noise.
- **Prec.:** Elhage et al. 2022 (M); Goyal et al. 2022; Kanai 2026 (A; diagnostic, not normative); SchedNet-style learned scheduling (M).
- **Test:** small autoencoder-style optimisation.
- **Alt.:** the outcome is predictable from superposition theory.
- **Feas.:** high.
- **Verdict:** **Top-5.** High risk of being true by construction.

**6. UN: what makes two controllers one agent?**
- **Q:** Does unity of agency require communication bandwidth, or can a shared body plus mutual prediction suffice, while unity of *information* still needs bandwidth?
- **Why:** It is a computational handle on the split-brain "unity with split perception" debate (Pinto et al. 2017), which bears on GW and IIT assumptions.
- **Best:** a two-transition phase diagram.
- **Prec.:** Pinto et al. 2017 (M/S); Schechter 2018 (M); bihemispheric NN models (S); MARL implicit coordination (S).
- **Test:** a two-controller agent with a bandwidth sweep.
- **Alt.:** shared outputs unify trivially.
- **Feas.:** high.
- **Verdict:** **Top-4 (§D.3).**

**7. DR: discreteness as error correction.**
- **Q:** Does the seriality and discreteness of access exist to correct errors in long serial computation?
- **Prec.:** Zou et al. 2026 (2602.01148, S) explicitly frames token selection as error-correcting discretisation against noise accumulation in latent CoT; 2510.14095 (S) shows discretise-and-re-embed anchoring.
- **Verdict:** **Eliminated.** Already established; only the relabelling to consciousness would remain (Category A).

**8. RR: implicit→explicit (behavioural self-awareness) in toy transformers.**
- **Prec.:** Huang et al. 2025 (NeurIPS; OOCR from factorised OV matrices, A); Bozoukov et al. 2025 (rank-1 LoRA / single steering vector suffices, A); Betley et al. 2025.
- **Verdict:** **Eliminated.** The mechanism is substantially explained and the field is crowded.

**9. EM: narrow updates re-organise a global persona latent.**
- **Prec.:** 2607.21356 (an "inference" account, A); persona features controlling emergent misalignment (ICLR 2026, S); a SPAR 2026 listing cites an existing HMM toy model (S).
- **Verdict:** **Eliminated.** Crowded, with high scoop risk.

**10. UAL: a minimal machine passing Unlimited Associative Learning.**
- **Prec.:** Birch, Ginsburg & Jablonka 2020 (F) explicitly concede that engineered systems may show UAL-like learning without the hallmarks, and declare them "derivative". Halina 2022 (S) treats UAL as a null hypothesis. Herzog et al. 2007 make the small-network argument (M).
- **Verdict:** **Eliminated.** The predictable positive result is pre-empted by the theory's own authors.

**11. UF: the unfolding argument fails under plasticity.**
- **Prec.:** O'Reilly-Shah, Selvitella & Schurger 2025/2026 (bioRxiv; Neuroscience of Consciousness, S) prove exactly this.
- **Verdict:** **Eliminated.**

**12. JS: is J-space "verbalizability" just shared-readout geometry?**
- **Prec.:** Gurnee et al. 2026 §9.1 concede the construction point; 2608.25347 gives a mathematical account of the J-lens readout (S); public replications are mixed (S).
- **Verdict:** **Eliminated.** It is acknowledged and workspace-anchored (C15-adjacent).

**13. MT: do transformers show the shared-representation multitasking limit?**
- **Prec.:** Musslick et al. 2017; Petri et al. 2021 (Nature Physics); Musslick & Cohen 2021.
- **Verdict:** **Eliminated.** It would apply known theory to a new architecture (Category A).

**14. SLT: integration as sub-additivity of the local learning coefficient.**
- **Q:** Is joint-model LLC smaller than the sum of single-task LLCs?
- **Prec.:** none found (A/S; Lau et al. 2025; Wang et al. 2024).
- **Verdict:** **Eliminated.** It is a measure with no target phenomenon, and LLC estimates are noisy at small n. Low significance.

**15. CC: integration as cross-bipartition communication complexity (unfolding-invariant).**
- **Verdict:** **Eliminated.** It is a definition with little testability. I did not screen it deeply, so this judgement is based on the significance assessment only.

**16. PRM: is reality monitoring forced in systems that share perception and imagination codes?**
- **Prec.:** Dijkstra, Kok & Fleming 2023 (empirical "reality threshold", S); Gershman 2019 (S).
- **Verdict:** **Folded into the SM family.** Source attribution is the self/world attribution problem applied to signals; it is a later theorem in the same framework.

### B.2b The seven fields for candidates 7–16

Candidates 1–6 give all seven fields above. For candidates 7–16, the remaining fields follow.

| # | Why it would matter | Strongest plausible contribution | Smallest decisive test | Strongest alternative | Feasibility |
|---|---|---|---|---|---|
| 7 DR | A normative reason for the seriality and discreteness of conscious access | A law for the optimal discretisation interval vs noise and chain length | Noisy recurrent chain with and without periodic quantisation | Generic error-correction theory (von Neumann; Sarpeshkar), already applied to latent CoT | H |
| 8 RR | How implicit (procedural) knowledge becomes reportable | A toy theory of behaviour→self-description generalisation | Small transformer pre-trained on trait-describing agents, then behaviour-only fine-tuning | Steering-vector or OOCR accounts (already shown) | H |
| 9 EM | Unity of self-models: local updates propagate globally | Bayesian persona-posterior theory of narrow-to-broad generalisation | HMM-persona toy transformer | Optimisation-accumulation account; existing toy model | H |
| 10 UAL | Validity of a policy-relevant consciousness marker (animal welfare) | A ~10³-parameter agent passing all five UAL criteria without the eight hallmarks | Meta-RL agent on a UAL battery | "Derivative AI" clause (the authors') | H |
| 11 UF | Testability of causal-structure theories (IIT, RPT) | A non-equivalence theorem under plasticity | Proof | Already proven (O'Reilly-Shah et al.) | H |
| 12 JS | Whether the LLM "workspace" is architectural or emergent | Shared-vs-untied readout ablation in toy transformers | Toy transformers with varied readout heads | Construction already conceded; mixed replications | H |
| 13 MT | Whether transformers inherit the multitasking/learning trade-off | Petri/Musslick prediction confirmed in attention models | Concurrent-task interference in small transformers | Known theory; Category A | H |
| 14 SLT | A geometry-based integration measure | LLC(joint) < ΣLLC(single) as "integration" | SGLD LLC estimates on toy multitask nets | Additivity of non-interacting blocks makes it near-definitional | M (estimator noise) |
| 15 CC | An integration measure invariant to unfolding | Communication-complexity lower bounds across all bipartitions | Proof for toy function classes | A definition, not a discovery | H (theory) |
| 16 PRM | Why generative perceivers need reality monitoring | Selection theorem for source attribution | Proof plus toy perception/imagination network | Vividness-threshold account (Dijkstra et al.) suffices | H |

### B.3 Pattern in the eliminations

- **Ideas framed around a consciousness theory** (workspace, ignition, attention schema, UAL) were either already tested in AI, conceded by the theory's authors, or reducible to known computational results.
- **What survives is a question about *selection pressure*:** which task demands force which internal organisation. That is a genre where theory is cheap, ground truth is exact, and the 2024–2026 world-model theorems have left a visible gap at the self.

---

## C. Top-five ranking

All ratings are qualitative: H = high, M = medium, L = low. Novelty, significance and feasibility are assessed independently.

| Rank | Candidate | Originality | Potential impact | Feasibility ($0 CPU) | Identifiability | Main risk |
|---|---|---|---|---|---|---|
| 1 | **SM**: selection theorem for self-models, via cross-context self/world attribution | M–H for the theorem as a package; L–M if it reduces to Nayebi Cor. 4 | M–H: necessity condition for self-model theories, a new behavioural test, an explanation of why within-task introspection tests are uninformative | H: theory plus GRU agents, ≈10–25 CPU-h | H for the theory; M–H for the empirical part (exact Bayes ground truth, planted positive control) | The math is a re-labelling of known selection theorems; the "self" definition is challenged (global world factors) |
| 2 | **IR**: faithful self-report requires self-non-stationarity | L–M | M for AI-evaluation practice | H | M–H | Subsumed by SM; informal precedents |
| 3 | **CL**: capacity-limited broadcast as a learning device | L–M | M | H | M (dimensionality confound) | Reduces to known variance scaling (Werfel et al.) and attention-for-RL (Niv; AGREL) |
| 4 | **UN**: unity transition (bandwidth vs shared body) | M | L–M | H | M | Unity holds trivially by construction through shared outputs |
| 5 | **TA**: toy models of access | L–M | L–M | H | M–L | Outcome predictable from superposition theory |

**Why IR is not analysed separately below:** it shares its core principle with SM, so §D.1 analyses it as SM prediction P-IR. Positions 2 and 3 of the deep analysis therefore go to CL and UN.

### C.1 Adversarial novelty audit of the three finalists

| Audit question | SM | CL | UN |
|---|---|---|---|
| Has the underlying question already been answered? | Not found. World-model necessity is answered (Richens & Everitt 2024; Richens et al. 2025; Cifuentes 2026; Nayebi 2026). Self-model necessity is not; it was explicitly noted as open (Wentworth survey comments) | Partly. Capacity limits are rationalised as learning-efficient via shared representations (Musslick et al.); attention aids RL credit assignment (AGREL; Niv et al.) | Not found computationally. It is debated conceptually (Pinto et al.; Schechter) |
| Has the mechanism already been demonstrated? | Body/world attribution in human motor control (Berniker & Kording 2008, normative model); self-model *benefit* in robots (Kwiatkowski & Lipson). Not as a necessity result, and not with a within-context boundary | Selection-gated plasticity, yes (AGREL). A learning-rule × capacity *interaction*, no | Implicit coordination through the environment in MARL, yes. A unity dissociation, no |
| Does novelty exist only by combining known components? | **Risk.** Kelley + Berniker & Kording + Nayebi Cor. 4 could jointly imply SM. The §F milestone decides | Largely yes, except the interaction prediction | Partly: known MARL components, new framing and measures |
| Does prior work already imply the result? | Possibly Nayebi Cor. 4 for Theorem A (regimes = self vs world). Theorem B and the extended-self corollary are not obviously implied | Werfel et al.'s variance scaling plausibly implies the local-learning half | Rate-distortion implies the information-unity half; the agency-unity half is not implied |
| Would a positive finding teach something new? | Yes if the boundary and the only-if dissociation hold: when self-knowledge is forced, and why within-context tests fail | Modestly: that capacity limits are learning-rule-specific | Modestly: that "unity" is not one property |
| Theoretically important, or merely an unusual implementation? | Theoretical (a necessity condition) | Mixed | Mostly an unusual implementation, with conceptual payoff |
| Exact result distinguishing it from its closest predecessors | An iff characterisation of the query families that force a self/world split (cross-context rank plus consensus), shown not to be derivable from Nayebi Cor. 3/4 or Richens & Everitt Thm 2 without new assumptions; plus SM+ vs SM-within internal dissociation at matched competence | Crossover: top-k beats full broadcast under node perturbation but not under backprop, *and* selective beats random-fixed-k | Two distinct critical bandwidths (agency-unity at C = 0 with body coupling; information-unity at C ≈ task rate) |

---

## D. Deep analysis of the top three

### D.1 SM: a selection theorem for self-models ("the self is what must be carried across contexts")

#### Precise hypothesis

**The setting:**
- **Contexts:** an agent acts in contexts k = 1…K, such as tasks, skills or places.
- **Outcomes:** in each context, outcomes depend on world parameters w_k (local to that context) and self parameters s. The self parameters are shared by every context that uses a faculty: effector reliability, perceptual acuity, memory retention, a skill's competence.
- **Shifts:** interventions hit s (self-shift) or some w_k (world-shift). The agent is **not told** which.
- **Queries:** the agent faces queries (h, j, c): act in context j after history h, with an outside option of value c. This is betting or opt-out, as in metacognition paradigms and Nayebi's betting reduction.

**Theorem A (necessity and extraction; conjectured, to be proved):**
- **Premise:** a policy has δ-bounded regret on the *cross-context* query family Q_X, i.e. queries in contexts j not visited since the last possible shift.
- **Conclusion:** from the policy alone one can recover two things, each within γ(δ), with γ → 0 as δ → 0:
  - T(h) = P(self-shift | h);
  - the posterior over s.
- **Context-invariance:** the recovered object is the same for every unvisited j. That is a self-model in the functional sense.
- **Identifiability condition:** a rank condition on how contexts load on s, analogous to Kelley's "distinctiveness".

**Theorem B (identifiability boundary):**
- **Premise:** the agent only faces the *within-context* family Q_W, i.e. queries in the context just observed.
- **Conclusion:** self-shifts and world-shifts that produce the same local outcome distribution are observationally equivalent. There exist zero-regret policies that carry no self/world attribution. No self-model is forced.

**Corollary (extended self):**
- **What cross-context structure cannot do:** it cannot separate s from *global* world factors that affect all of the agent's contexts.
- **What can separate them:**
  - "consensus" evidence (other agents' outcomes), or
  - relocation (the factor stays behind when the agent moves).
- **Consequence:** without such evidence, the forced self-variable is an *extended self*: the agent plus whatever reliably travels with it. This formally echoes the extended-mind thesis, and it is a falsifiable prediction.

**Conjecture C (blind spot):**
- **Setting:** attribution is computed from the agent's own outcome-evaluation channel.
- **Claim:** context-global perturbations of that channel are indistinguishable from genuine self-shifts. Perturbations that null the comparator yield undetected deficits (anosognosia-like confident failure). The only exception is an independent channel, such as external feedback or consensus.

**Prediction P-IR:**
- **Claim:** a report head reads a faithful (causally dependent) self-state only in agents trained under self-non-stationarity with cross-context demands.
- **Consequence:** for systems trained without either, including static pretrained LMs, self-reports should be expected to track input proxies and population priors, not internal states.

**Extension, not in scope now (SM+, self-models of plasticity):**
- **Premise:** the agent's own actions change s, as with practice or fatigue.
- **Claim:** combining Richens et al.'s (2025) multi-step extraction with self-dynamics would force a model of the agent's *own learning curve*.

#### What is genuinely new (if the milestone passes)

1. **A necessity result for self-models.** Every existing selection or regulator theorem I could find concerns models of the *environment*:
   - Richens & Everitt 2024 (P) keeps the agent's decision outside the modelled variables, restricts to unmediated tasks, and tells the agent the shift. The authors argue that inferred shifts are covered by the same argument.
   - Richens et al. 2025 (P) assumes a stationary environment and a fixed agent.
   - Cifuentes 2026 (A) extends this to partial observability.
   - Nayebi 2026 (P) forces "regime-tracking variables", but the regime is an index of the evaluation protocol. The paper does not discuss self-models or agent-vs-environment change.
   - Virgo et al. 2025 (S) cover belief-based regulation of the environment.
   - Xing et al. 2026 (P) give a *sufficiency* (dominance) theorem for fast self-model updates, under an assumption (A1) that sidesteps attribution.
   - On Wentworth's 2021 LessWrong survey of selection theorems (A), a reader comment asked about self-modelling theorems. Wentworth replied "None that I know of; it's a topic ripe for exploration".
2. **An identifiability boundary (Theorem B).** It says *when self-knowledge is not forced*, which connects to Smith et al.'s (2016) associative/metacognitive isomorphism and Singh et al.'s (2026) privileged-access condition.
3. **A behavioural test with a theoretical guarantee:** the cross-context self-change transfer test. I did not find it in any AI introspection or self-modelling benchmark I checked (e.g. Zeng et al. 2026); this is not an exhaustive survey.
4. **The extended-self corollary**, which gives a principled, testable boundary of the functional self.

#### Closest competing literature, and what distinguishes SM

| Work | What it shows | What SM adds |
|---|---|---|
| Richens & Everitt 2024 (P); Richens et al. 2025 (P); Cifuentes 2026 (A) | Robust or general agents must contain world models (extractable from the policy) | Self vs world. The agent's own channel parameters are untold and shared across contexts, and the boundary result says when they are *not* forced |
| Nayebi 2026 (P), Cor. 4 | Low regret forces memory to separate latent regimes on tests that separate them | **Main novelty risk.** SM may be Cor. 4 with regimes = {self-shift, world-shift-in-k}. SM must add (i) an iff characterisation of *which* test families separate self from world (cross-context rank, consensus), (ii) the extended-self corollary, (iii) Conjecture C, (iv) internal-realisation evidence |
| Berniker & Kording 2008 (F) | A normative Bayesian model of how humans attribute motor errors to body vs world. Transfer patterns are the identifying evidence | Turns a modelling hypothesis about humans into a necessity result for any low-regret agent. Adds the boundary (within-context equivalence) and artificial-agent tests |
| Kelley 1967/1973 (M) | Covariation principle: consensus, distinctiveness and consistency drive person vs situation attribution | A regret-theoretic foundation, with distinctiveness and consensus as identifiability conditions |
| Kwiatkowski & Lipson 2019 (A); Bongard et al. 2006 (M); Kwiatkowski et al. 2022 (A) | Self-models *help* robots transfer across tasks and recover from damage. The benefit grows with degrees of freedom | The converse: when a self-model is *necessary*, and when it is useless |
| Tomaszewski 2026, SIL (A) | Networks learn self-models by self-intervention. Model-guided action is *not* better than empirical memory | Consistent with Theorem B: with no cross-context projection demand, a self-model brings no advantage. SM predicts when it would |
| Rapid Motor Adaptation (M); hidden-parameter MDPs (M); ReMAP / GraphOp-WM 2026 (S) | Agents infer body or extrinsic parameters from history; morphology-invariant factorisation | Empirical machinery exists. SM supplies the necessity theory and the only-if control |

#### Why the result matters

- **Self-model theories of consciousness.** AST (Graziano), self-model theory (Metzinger), metacognitive and HOT accounts, and interoceptive predictive processing (Seth) all assume a self-model is functionally important. None states when a system must have one. SM would give a Level-2 bridge: a *necessary computational condition* that makes those theories' functional claims checkable.
- **AI evaluation.** The current debate (Lindsey 2025; Song et al. 2025; Singh et al. 2026; Zeng et al. 2026) runs within-context tests. Theorem B, and its report-head analogue P-IR, say those tests cannot in principle separate self-access from world-access when the internal states are input-determined. The cross-context transfer test is a principled replacement.
- **Construction.** SM gives a recipe for building agents that must form self-models: train them under self-non-stationarity with cross-context demands.

#### The dream positive finding

1. Theorems A and B, plus the extended-self corollary, are proven in a general form, with a clean iff characterisation, and are shown not to be a one-line instance of Nayebi Cor. 4.
2. In ground-truth toy agents:
   - agents trained under self-shifts with cross-context queries develop a low-dimensional, context-invariant internal variable. It decodes s on held-out shift magnitudes and context combinations, and *causally* carries adaptation into unvisited contexts when patched;
   - matched agents trained with world-shifts only, or with self-shifts but only within-context queries, reach the same in-context performance *without* such a causal variable;
   - removing consensus evidence turns the variable into an "extended self" that absorbs global world factors;
   - lesioning the evaluation pathway produces the predicted anosognosia-like signature.

#### The most informative negative finding

**Theoretical negative:** the self/world split is exactly an instance of existing regime-tracking theorems, with nothing beyond re-labelling. That is still useful: it means self-models need no special theory. It also says the consciousness-relevant content of "self-model" theories is not computational necessity.

**Empirical negative:** within-context-trained agents develop the same causal, factorised self-variable anyway. Then inductive bias toward reuse, not task demand, explains self-models. That would cut against selection-based accounts of self-modelling, including AST's control-theoretic rationale.

#### Decisive strategy

**Stage 0 (theory; the §F milestone):**
- prove A and B in a minimal model;
- write out the reduction to Nayebi Cor. 4 and Richens & Everitt explicitly;
- decide whether a non-trivial core remains.

**Stage 1 (ground-truth agents, only after Stage 0 passes):**
- **Task family:** K ≈ 8 betting/opt-out contexts, loading on 2 overlapping self faculties plus context-local world difficulty.
- **Training signal:** self-supervised next-outcome prediction in queried contexts (log loss). The agent never receives self/world labels, which avoids building the effect in by construction.
- **Architecture:** GRU, hidden 64–128.
- **Conditions:**
  - SM+ (self- and world-shifts; cross-context queries);
  - W-only (world-shifts only);
  - SM-within (self- and world-shifts; only within-context queries);
  - SM+ with and without consensus evidence (other agents' outcomes).
- **Gates, in the R1–R12 spirit:**
  - competence gate: regret relative to the exact Bayes-optimal predictor below a pre-declared bound before any representational analysis;
  - a planted positive control: a hand-built Bayes-filter RNN whose self-variable is known, used to validate probes and patching before they touch trained agents;
  - decoders tested on held-out shift magnitudes and context combinations (R9);
  - norm-matched world-direction and random-direction patch controls (R4/R10).
- **Regime design:** attribution must require a *nonlinear* covariation pattern (degradation in ≥2 contexts vs 1), so that a running cross-context average cannot pass as a self-model.

**Stage 2 (optional, separate approval; the LLM bridge):**
- **Design:** a cross-context self-change test in a frozen small LM. A component ablation hits a faculty shared by tasks A and B; in-context failure feedback is given on A; confidence or abstention is measured on B. The world-change control makes inputs to A harder instead.
- **SM's prediction for static LMs:** no selective transfer.
- **Risk:** this stage carries C15/C16-type risks (competence, intervention validity), so it is explicitly not part of the current proposal.

#### Strongest confound or alternative explanation

1. **"It is just hierarchical Bayes."** Any good multi-task learner shares latents across tasks, so calling the shared latent "self" may be a relabelling.
   - Reply: the content lies in the boundary (B), the extended-self corollary and the only-if dissociation. If those add nothing, SM fails §F.
2. **The behavioural vs internal gap** (Thobani 2024, A; Virgo et al. 2025, S). Selection theorems show a model is *extractable from behaviour*, not *represented*. Stage 1 exists to test internal realisation. The theorem alone licenses only the functional claim.
3. **History-statistic shortcuts.** A "self-variable" might just be a running average of recent outcomes. This is controlled by nonlinear attribution regimes and causal patching on held-out combinations.

#### Falsification conditions

- **F-T1 (novelty):** the minimal-case result and its iff characterisation follow by direct instantiation of Nayebi Cor. 4 or Richens & Everitt Thm 2, with no additional assumptions or conclusions. **Stop.**
- **F-T2 (identifiability):** even with cross-context queries satisfying the rank condition, generic parameterisations leave T(h) unidentifiable from δ-optimal policies. **Stop.**
- **F-T3 (vacuity):** γ(δ) does not vanish, or vanishes only in measure-zero regimes. **Stop or reframe.**
- **F-S (significance):** the exact Bayes regret gap between attribution-aware and attribution-blind policies on Q_X is negligible over plausible parameter ranges, so the selection pressure is too weak to matter. Pre-declared threshold to be set in Stage 0.
- **F-E1:** SM+ agents pass the competence gate but no context-invariant causal self-variable passes the planted-control-validated patch test.
- **F-E2:** SM-within agents show the same causal self-variable at matched competence. The only-if claim fails; report as an inductive-bias finding.

#### Approximate cost

| Stage | CPU time | RAM | Calendar time |
|---|---|---|---|
| Stage 0 | 0 compute; optionally under 1 CPU-minute of exact Bayes arithmetic if approved | — | One to two focused sessions |
| Stage 1 | About 4 conditions × 5 seeds × 2 sizes (≈40 runs), roughly 15–40 min each on CPU: **≈10–27 CPU-h** | < 4 GB | — |
| Stage 2 | ≈10–30 CPU-h | — | Not proposed now |

No downloads are needed for Stages 0–1.

#### What success would and would not justify

- **Would justify:**
  - *Level 1:* a necessary computational condition for self-models in adaptive agents, plus evidence on whether gradient training realises them internally.
  - *Level 2:* a principled test of self-model claims in AI and animals, and a formal grounding for the functional claims of self-model theories.
- **Would not justify:** that self-models are sufficient for consciousness, that any agent has phenomenal selfhood, or anything at Level 3.

---

### D.2 CL: capacity-limited broadcast as a learning device

#### Precise hypothesis

In modular systems trained with local credit assignment (node perturbation or three-factor rules), a *selective* top-k broadcast of module messages speeds up learning of cross-module coordination. It beats both:
- full broadcast, and
- a non-selective fixed low-rank channel of equal bandwidth.

Under end-to-end backprop, the advantage vanishes or reverses. The prediction is a **learning-rule × capacity interaction**.

#### What is new

- **The discriminating interaction.** Musslick and Cohen's learning-efficiency vs multitasking trade-off (2017–2021) arises in backprop-trained networks. Their account therefore does not predict a learning-rule dependence; CL does.
- **An analytic optimal-k law** based on how the variance of the perturbation-gradient estimate scales with the number of simultaneous contributors.

#### Closest literature

- AGREL (2005; attention feedback gates plasticity for credit assignment);
- Niv et al. 2015 (attention solves dimensionality in RL);
- Goyal et al. 2022 (top-k workspace helps coordination);
- Musslick et al. / Petri et al. (shared-representation trade-off);
- Clark, Abbott & Chung 2021 (global error broadcast);
- Werfel, Xie & Seung 2005 (perturbation learning slows with N).

#### Why it matters

If true, workspace capacity limits are best understood as adaptations for learning in substrates without backprop. That gives a principled reason why backprop-trained AI systems need not, and perhaps do not, develop sharp workspace bottlenecks. The claim stays Level 1–2.

#### Dream finding

- a clean crossover interaction across module counts M ∈ {4, 8, 16};
- an analytic k\*(M, noise, task sparsity) that predicts the simulated optimum;
- selectivity matters: random fixed projections do not reproduce the benefit.

#### Most informative negative finding

The bottleneck helps (or hurts) equally under both learning rules. Then capacity limits are not credit-assignment adaptations, and accounts based on interference or inference cost win.

#### Strategy

1. Analytic: linear modules with node-perturbation estimator variance as a function of k.
2. Small simulations with matched parameter counts.
3. Three channel conditions: selective top-k, random-fixed-k, full broadcast.
4. Two learning rules: node perturbation and backprop.
5. Pre-declared M, k grid and seeds.

#### Strongest confound

Selection may help merely by restricting *what is learned about* (feature selection, Niv) rather than *what is shared* (broadcast capacity). Separating the two requires:
- a condition with selective plasticity gating but full sharing, and
- the converse condition.

The gate's own learning problem is a second confound.

#### Falsification

- no top-k advantage under local learning at any M; or
- an equal advantage under backprop; or
- random-fixed-k matches selective top-k.

#### Cost

Under 10 CPU-h, under 2 GB RAM.

#### What it would and would not justify

- **Would justify:** a sufficiency demonstration that selective capacity limits aid local credit assignment.
- **Would not justify:** that brains have workspaces for this reason, or any claim about consciousness.

#### Novelty assessment

L–M. The interaction prediction is the only clearly new element.

---

### D.3 UN: unity transition (communication bandwidth vs shared body)

#### Precise hypothesis

Consider a two-controller agent: two controllers, each with a private sensory field and its own effector, joined by a channel of capacity C.

- **Unity of agency** (one coherent goal, no inter-manual conflict, consistent choices) can survive at C = 0. The mechanism is a shared body/environment loop (cross-cueing) plus learned mutual prediction.
- **Unity of information** (one decision integrating both private fields) collapses once C falls below the task's information rate.
- The two unities therefore have **distinct transitions**. Pinto et al.'s "unity with split perception" pattern would then be a generic outcome of embodied coupling, not decisive evidence about GW or IIT.

#### What is new

I found no computational model that dissociates the two senses of unity as a function of bandwidth and body coupling. Schechter's "single body" argument is conceptual (M).

#### Closest literature

- Pinto, de Haan & Lamme 2017 (M/S);
- Schechter 2018 (M);
- bihemispheric NN lateralisation models (S);
- MARL implicit communication and bandwidth-limited coordination (S).

#### Why it matters

"Unity" is a core explanandum of consciousness science, and many theories assume a single notion of it. A computational dissociation would show which unity each theory needs. The significance is modest.

#### Dream finding

A two-transition phase diagram that replicates under architecture and training changes.

#### Most informative negative finding

Both unities collapse at the same C, even with body coupling. Unity would then be unitary, and Pinto's pattern would need residual (e.g. subcortical) channels.

#### Strategy

- separate controllers with a bandwidth sweep;
- body coupling either on or off;
- joint vs separate training;
- unity metrics pre-declared: conflict rate, choice transitivity across controllers, and cross-field integration accuracy.

#### Strongest confound

Any shared output stage unifies by construction. Effectors must be strictly separate, with only environmental coupling. "Cross-cueing" is a hidden channel by design, so it must be measured and reported, not hidden.

#### Falsification

Agency-unity metrics collapse at C = 0 with body coupling present.

#### Cost

5–20 CPU-h.

#### What it would and would not justify

- **Would justify:** that the two unity notions dissociate in artificial agents.
- **Would not justify:** anything about patients' experience.

#### Novelty assessment

M novelty, L–M significance.

---

## E. Recommended direction

**Primary recommendation: SM.**

**Why SM rather than CL or UN:**
- It is the only candidate whose central claim is a **necessity theorem** about a construct that consciousness theories care about, the self-model.
- Its ground truth is exact, and its alternatives make different predictions:
  - *task-demand account:* a self-variable only under SM+;
  - *inductive-bias account:* a self-variable under SM-within too;
  - *generic shared-latent account:* identical structure for global world factors, which the consensus manipulation tests.
- It turns the project's most persistent failure, observational equivalence within a single target, into a *result* (Theorem B) and points to the identifying lever (cross-context structure).

**The trade-off against CL:**
- CL is cheaper and has a cleaner single experiment, but its novelty is weaker. Its interaction prediction is new; its mechanism is largely anticipated by AGREL, Niv et al. and Musslick et al.
- SM has higher ceiling and higher novelty risk.
- **If SM fails §F on novelty**, CL is the fallback I would recommend. Only its interaction test would be run, as a short Category-B study.

**Not recommended:** any further LLM workspace, introspection-assay or retrofit study (C15/C16 lineage). Nothing found in this audit changes the C15/C16 assessments.

### E.1 Consciousness claims for SM, kept at three levels

| Level | What SM could establish |
|---|---|
| **1. Computational phenomenon** | Under stated conditions, any low-regret agent must carry a context-invariant self/world attribution variable (Theorem A, conjectured). Within-context demands alone do not force one (Theorem B, conjectured). Trained toy agents do, or do not, realise it internally |
| **2. Consciousness-theory relevance** | A necessary condition that the functional claims of self-model theories (AST, Metzinger's self-model theory, metacognitive and HOT accounts, interoceptive predictive processing) must respect. A principled behavioural test for self-model indicators in AI and animals. A formal reason why static training regimes give no selection pressure for self-models (cf. Hoel 2025's continual-learning argument, which reaches a related conclusion by a different route) |
| **3. Phenomenal consciousness** | Nothing. Establishing when a functional self-model is necessary says nothing, by itself, about subjective experience |

---

## F. First decisive milestone (not implemented)

**Milestone SM-0: a minimal-case proof attempt plus a written reduction check.** Theory only; no code.

### The minimal model M0

- contexts K = 3;
- one self parameter s ∈ {nominal, degraded};
- one binary world parameter per context;
- success probability q(s, w_k), with self-degradation and world-difficulty chosen to be **within-context indistinguishable** (q(degraded, easy) = q(nominal, hard));
- shift prior ρ (self) vs (1 − ρ)/K per context (world);
- n Bernoulli outcomes observed in contexts 1–2;
- an opt-out query in unvisited context 3 with outside value c.

### Tasks

1. **Prove Theorem A in M0.**
   - Show the policy's switch-points in c, over histories, recover T(h) = P(self | h).
   - Derive γ(δ) using Nayebi's betting lemma or a direct argument.
   - State the rank/"distinctiveness" condition precisely. With only context 1 observed, T(h) collapses to a prior-weighted quantity; with contexts 1 and 2, covariation identifies it.
2. **Prove Theorem B in M0.** Exhibit two change processes that are indistinguishable on Q_W and a zero-regret self-free policy.
3. **Prove the extended-self corollary in M0.** Add a global world factor g, show it is unidentifiable from s without a consensus channel, and show identifiability with one.
4. **Reduction check (the novelty kill).** Write the most direct derivation of tasks 1–3 from:
   - Nayebi 2026 Cor. 3/4 (regime tracking, informational modularity);
   - Richens & Everitt 2024 Thm 2;
   - Richens et al. 2025 Thm 1.

   Record exactly what extra assumptions or conclusions SM needs. Read the unread parts of Nayebi (characters 100k–157k), Richens & Everitt (characters 100k–190k) and the Cifuentes full text first.
5. **Significance check.** Write the closed-form regret gap between the Bayes-optimal and an attribution-blind policy on Q_X as a function of (ρ, n, q, c). Evaluating it numerically would need a few lines of arithmetic, which needs PI approval in this no-code phase.

### Kill criteria (any one stops SM)

- **K1 (novelty):** task 4 yields complete derivations of tasks 1–3 from existing theorems with no new assumption or conclusion.
- **K2 (identifiability):** task 1 fails generically in M0.
- **K3 (vacuity):** γ(δ) is non-vanishing.
- **K4 (significance):** the regret gap is below a threshold pre-declared before computing it (proposed: < 2% of the maximum achievable utility in the best-case regime).

**Pass condition:** a non-trivial iff characterisation (cross-context rank plus consensus) and the extended-self corollary survive task 4. In that case, return to the PI with a Stage-1 preregistration memo. Nothing is run before that approval.

---

## G. Red-team verdict (arguing against my own recommendation)

### 1. "It is unoriginal."

**The argument:**
- Self/world attribution from covariation is Kelley (1967/1973).
- Body/world attribution with transfer as the identifying signal is Berniker & Kording (2008), a Nature Neuroscience paper.
- "Low-regret agents must track latent regimes that tests separate" is Nayebi (2026, Cor. 4).
- Hierarchical Bayesian shared parameters across tasks are textbook multi-task learning.
- So SM is three known results with a consciousness-flavoured label.

**Assessment:** this is the strongest objection, and it may be right. It is exactly what K1 tests.

### 2. "It is insignificant."

**The argument:** even if novel, a necessity theorem in a betting framework says little about real self-models. Thobani (2024) shows that "contains a model" claims of this kind can be trivially satisfied, and Virgo et al. (2025) concede models are observer-relative. Consciousness theorists care about phenomenal or transparent self-models, not about extractable posteriors.

**Assessment:** partly right. SM's significance for consciousness science is Level 2 at most. Its clearer value is methodological, for AI evaluation (Theorem B plus the transfer test).

### 3. "It is theoretically misguided."

**The argument:** defining the self as "what is shared across the agent's contexts" mislabels global environmental constants as self. The extended-self corollary concedes this. So the theorem may be about *agent-indexed invariants*, not selves.

**Assessment:** a fair point. The reply is that the functional self of any agent *is* agent-indexed and invariant, and that consensus evidence is how humans also draw the line (Kelley). Even so, the "self" label must be argued for, not assumed.

### 4. "It is experimentally unidentifiable."

**The argument:**
- In Stage 1, any low-regret agent will behave as the theorem says, so the behavioural result is guaranteed by construction.
- Internal "self-variables" may be found by probes simply because history statistics correlate with s, which is the old N1 problem.

**Assessment:** partly right. The behavioural transfer is indeed *not* a primary empirical claim. The primary empirical claims are:
- internal causal factorisation validated against a planted control;
- the only-if dissociation (SM-within vs SM+);
- the consensus-dependent extended self.

Each can come out either way.

### 5. Scoop risk

The Richens/Everitt and Nayebi lines are active, and a self-model extension is an obvious next step for them. A preprint could appear within months.

### What evidence would change my decision

- **Toward NO-GO:**
  - K1, K2, K3 or K4 fires;
  - or a self-model selection theorem or a cross-context self-change test is found in prior work (I would re-search before Stage 1).
- **Toward GO (dropping the condition):**
  - task 4 shows SM needs a genuinely new proof idea. Plausible candidates: the self-parameter sits on the action channel, so the agent's own policy changes the evidence it receives about s (a coupling that, as far as I have read, Nayebi's regime corollary does not address); or self-dynamics are driven by the agent's own actions (SM+);
  - and the regret gap is substantial in plausible regimes.

---

## H. Final decision

**CONDITIONAL.** SM is potentially important but depends on one unresolved novelty question: whether its core reduces to Nayebi 2026 Cor. 4, Richens & Everitt 2024 or Richens et al. 2025. A secondary significance question is the size of the regret gap.

**For the PI to decide:**
1. Approve milestone SM-0 (theory only; §F), or decline and record NO-GO.
2. Separately, decide whether the K4 regret-gap check may use a few lines of exact arithmetic (seconds of CPU), or must stay symbolic.
3. If SM-0 passes, approve or decline a Stage-1 preregistration memo. Nothing will be run without explicit approval.

**Fallback if SM-0 fails on novelty:** CL's interaction test (§D.2), after its own short audit of the selective-plasticity vs selective-sharing literature. Otherwise NO-GO.

---

## Appendix 1. Search protocol and limits

**Passes:**
- Pass 1 was first-principles ideation of 16 candidates.
- Pass 2 screened each candidate's novelty with targeted web search (standard mode; extended mode for the self-model gap).
- Pass 3 read full text, or partial full text, of the closest SM competitors:
  - Richens & Everitt 2024 (setup, theorems, assumptions, limitations);
  - Richens et al. 2025 (Thm 1);
  - Nayebi 2026 (Cor. 3–4, discussion);
  - Berniker & Kording 2008;
  - Xing et al. 2026 §4.2;
  - Birch, Ginsburg & Jablonka 2020.

**Limits:**
- **Unread portions:** Nayebi (beyond 100k characters), Richens & Everitt (beyond 100k characters), Cifuentes (full text) and Thobani (full text) remain unread. Task 4 of §F requires them.
- **Citation tracing:** forward and backward tracing was not run systematically, given the bounded process. I did not trace reference lists; only works surfaced by targeted searches were checked. Re-running the self-model prior-art search with citation tracing from Richens & Everitt 2024 and Nayebi 2026 is part of §F task 4.
- **Coverage:** searches were in English and web-indexed, and may miss very recent workshop papers.
- **Summariser:** all fetched texts were condensed by an automated summariser, so exact statements must be re-read before citation.

## Appendix 2. Key sources (all added to the literature DB with verification flags)

**SM core and competitors:**
- richens2024robust (P)
- richens2025general (P)
- cifuentes2026partial (A)
- nayebi2026selection (P)
- berniker2008sources (F)
- kelley1973attribution (M)
- xing2026agentmodel (P)
- kwiatkowski2019taskagnostic (A)
- kwiatkowski2022origins (A)
- bongard2006resilient (M)
- tomaszewski2026sil (A; existing entry)
- zeng2026selfmodeling (A)
- virgo2025goodregulator (S)
- thobani2024triviality (A)
- frith2000abnormalities (A)
- breuer1995selfmeasurement (M)
- kording2007changing (M)
- kumar2021rma (M)
- wentworth2021selectiontheorems (A)

**CL:**
- roelfsema2005agrel (A)
- niv2015attention (A)
- musslick2017multitasking (S)
- sagiv2020multitasking (A)
- petri2021topological (S)
- musslick2021rationalizing (S)
- clark2021gevb (A)
- werfel2005learning (M)

**UN:**
- pinto2017splitbrain (M)
- schechter2018split (M)

**Eliminations:**
- zou2026latentcot (S)
- huang2025oocr (A)
- bozoukov2025selfaware (A)
- em2026personasubspace (A)
- birch2020ual (F)
- halina2022ual (S)
- herzog2007smallnetwork (M)
- oreillyshah2026unfolding (S)
- kanai2026gmw (A)
- jlensreadout2026 (S)
- elhage2022superposition (M)
- dijkstra2023reality (S)
- gershman2019gab (S)

**Existing entries used:**
- smith2016formal
- singh2026reality
- song2025privileged
- lindsey2025introspection
- hoel2025disproof
- graziano2015ast
- goyal2022workspace
- gurnee2026workspace
- andrade2023dualpath
- betley2025tell
