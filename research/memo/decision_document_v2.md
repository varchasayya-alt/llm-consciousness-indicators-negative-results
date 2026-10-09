# Decision Document v2: strongest surviving directions under a $0 compute budget

**Date:** 2026-10-01

**Status:** supersedes memo §6–§7 (preferred direction and Pilot A).

**Inputs:**
- the closest-competitor review (memo §10);
- ~35 targeted searches;
- CPU feasibility measurements on the researcher's laptop (`results/raw/feasibility/`).

**Hard constraints:**
- $0 compute;
- no paid GPU;
- hardware: Intel Core Ultra 7 255U (15 W mobile CPU, 12 cores), 31 GB RAM total (~5 GB free during normal use), no CUDA GPU;
- the design must not depend on free cloud GPUs.

**Measured feasibility on this laptop** (benchmark only; no hypotheses tested):

| Workload | Measurement | Implication |
|---|---|---|
| Qwen2.5-0.5B-Instruct, fp32, 64-token prompt | 0.49 s/prompt (batch 1); 0.34 s/prompt (batch 8); 8-token greedy generation 1.0 s | ≤1.5B pretrained models are usable for thousands of prompts |
| Same model, bf16 | 2.0–2.2 s/prompt (4–6× *slower*: no native bf16 on this CPU) | Use fp32; 4B+ models need bf16 for RAM and are therefore slow |
| Tiny transformer training (d=128, 2 layers, 0.9M params) | 11.5k tokens/s | |
| Tiny transformer training (d=256, 4 layers, 4.2M) | 3.0k tokens/s | |
| Tiny transformer training (d=384, 6 layers, 12M) | 1.3k tokens/s | |
| Synthetic fact store (d=128, 3 layers, 4,000 facts) | 97.6% train-fact accuracy in **~20 s** | Developmental and factorial experiments with many seeds are cheap |

**Extrapolated:**
- 1.5B fp32: ~1–1.5 s/prompt, ~7 GB RAM.
- 4B bf16: ~15–20 s/prompt, ~9 GB RAM.
- 9B bf16: ~40 s/prompt, ~18 GB RAM (impractical).

---

## Verdicts in brief

| Candidate | Verdict |
|---|---|
| **P1 as originally proposed** (damage competence → does confidence fall?) | **Rejected.** The observation is established: global degradation [cohen2026source]; editing → token-confidence change on the same question [hasegawa2025underconf; hasegawa2026jnlp]; selective unlearning → models confabulate rather than refuse on the same questions [gu2026unlearners]; familiarity gating of refusal causally shown [ferrando2025entity]. Verbal or token confidence as the endpoint adds nothing |
| **P1\*** (input-invisible lesion → selective downstream *control*, with the full identification contrast set) | **Survives in reduced form.** Not done as such (§P1\*.2–3). But at $0 it is limited to 0.5–1.5B models, where floor effects are likely. Best role: **measurement bridge**, not flagship |
| **P2\*** (workspace → metacognitive control; ignition; dual-task) | **Survives scientifically; fails the $0 constraint in strong form.** The smallest validated J-space targets are 4–9B. 9B is infeasible on this laptop; 4B is marginal. Ignition and dual-task tests are being run by others (scoop risk). **Deferred** |
| **P5\*** (separately developed capacities + information-flow integration; held-out self-change tracking as the pre-registered emergent property) | **Survives and is the strongest under $0.** Fully CPU-feasible with many seeds. Ground truth is known. Directly addresses the original engineering question and the researcher's seed |
| **N1** (causal-identification framework for machine self-monitoring) | **New, from this audit.** Not stronger than P5\* alone, but it is the measurement theory P5\* and P1\* both need. **Merge into the P5\* paper** |
| N2 (conditions for workspace *emergence* in tiny transformers) | Considered; weaker than P5\* (construct validity of "workspace" in tiny models is unclear). Parked |

---

## P1\*: causal self-monitoring after an input-invisible competence intervention (pretrained small LMs)

1. **Exact scientific claim.** In a pretrained LM, after an intervention that impairs recall of specific items while leaving the input token-identical, downstream *control* behaviour changes selectively for exactly those items. Control behaviour here means requests for external lookup, abstention and (in a looped model) allocation of recurrent compute. "Selectively" is measured relative to five contrasts:
   - (a) matched sham interventions;
   - (b) lesions to unrelated items;
   - (c) matched-rate errors induced by *input* corruption;
   - (d) global degradation;
   - (e) naturally unknown items.

   *Or* it does not change (artificial anosognosia), and the failure can be localised: the controlling signal reads features upstream of the lesion.
2. **Closest prior work.**
   - Gu et al. 2026 (ACL) [gu2026unlearners]: selective unlearning on WMDP-Bio, same questions re-asked, rejection rate measured. Models mostly confabulate.
   - Hasegawa et al. 2025/2026 [hasegawa2025underconf; hasegawa2026jnlp]: knowledge editing, same question, token-probability confidence → underconfidence.
   - Cohen & de Melo 2026 [cohen2026source]: global quantisation/noise degradation and item regimes; behavioural confidence channels.
   - Ferrando et al. 2025 [ferrando2025entity]; Lindsey et al. 2025 [lindsey2025biology]: familiarity directions gate refusal.
   - Du & Hu 2026 [du2026recognition]: an encoded "no answer" signal is not routed to refusal.
   - Marina et al. 2025; Moskvoretskii et al. 2025 [marina2025llmindep; moskvoretskii2025adaptive]: question-only features match "self-knowledge" signals for retrieval decisions.
   - Yax et al. 2026 [yax2026forms]: trained monitors track output consistency (which generalises) rather than accuracy (which stays local).
3. **Precise difference.** None of these jointly:
   - (i) contrasts self-change vs input-change at matched error rates;
   - (ii) includes sham and unrelated-lesion controls;
   - (iii) uses *control* endpoints (lookup, compute allocation) rather than stated confidence;
   - (iv) localises the controlling signal relative to the lesion site (E4).

   This is a genuine but **incremental** difference. The likely main result (anosognosia under extraction lesions) is predictable from [ferrando2025entity; lindsey2025biology; gu2026unlearners]. The identification design, not the phenomenon, is the contribution.
4. **Falsifiable hypothesis (H1\*).**
   - For extraction lesions placed after familiarity is computed, the lookup-request tracking index for lesion-induced errors is ≤ 0.3, while it is ≥ 0.6 for input-corruption errors at a matched error rate. In words: the system tracks input difficulty, not its own change.
   - The *strongest* version: no endpoint, behavioural or internal, tracks lesion-induced errors beyond sham.
5. **Smallest decisive experiment.**
   - Model: Qwen2.5-0.5B-Instruct, replicated on Qwen2.5-1.5B-Instruct.
   - Items: ~3,000 PopQA/Wikidata questions → screen for ~500 base-correct items.
   - Conditions: base; attention knock-out of attribute extraction (3 strengths); sham knock-out; lesions to unrelated subjects; entity-name corruption calibrated to the same error rate; global weight noise; naturally unknown items.
   - Endpoints: prompted LOOKUP/ANSWER choice; abstention; frozen pre-lesion correctness probe; answer log-probability.
   - Optional: exit-gate depth allocation in Ouro-1.4B [zhu2025ouro].
6. **Main confounds.**
   - Lesions change answer-distribution concentration, so control may follow a first-order readout (Yax et al.'s "output consistency"). This is not self-monitoring of recall per se; it is handled by also reporting tracking conditional on matched output entropy.
   - Response-bias shifts [hahami2025disturbance].
   - Weak instruction following at 0.5B (floor effects).
   - Lesion non-selectivity.
   - Probe answer-identity confound [jspace_gemma_repo].
7. **Negative controls.**
   - Sham lesions (matched norm, matched number of knocked-out edges).
   - An input-only observer (must track input corruption and must *not* track lesions).
   - Lesions on unrelated subjects (must not change control on target items).
8. **What falsifies or weakens it.**
   - Lookup tracking for lesion errors ≥ 0.6 and ≫ sham: anosognosia hypothesis falsified (self-tracking present).
   - No endpoint tracks *input-corruption* errors either: the model lacks metacognitive control at this scale. The study is uninformative; abandon P1\* at this scale.
9. **What constitutes a meaningful paper.**
   - A methods/measurement paper (workshop level on its own): "Input-invisible lesions dissociate self-tracking from difficulty-tracking in LM metacognitive control".
   - Main-venue only with mechanistic localisation across ≥ 2 models, which is hard at $0.
10. **Directness to the original question.** Indirect. It *measures* whether existing LMs have counterfactually self-dependent monitoring (a C2/HOT-2-relevant property). It does not engineer it.
11. **$0?** Yes.
12. **Zero-budget execution plan.**
    1. Use public PopQA or a Wikidata-generated question set.
    2. Load fp32 models with PyTorch hooks for attention knock-out.
    3. Screen 3,000 items on 0.5B (~1.5 h).
    4. Run 8 conditions × 500 items × ~3 passes (~4 h).
    5. Replicate on 1.5B (~12–15 h, overnight).
    6. Optional Ouro arm (~6–10 h). Requires `trust_remote_code`; inspect the code before running.
13. **Minimum model size.** 0.5B (needs instruction following plus some factual knowledge); 1.5B preferred.
14. **Hardware / RAM.**
    - 0.5B fp32 ≈ 2–3 GB;
    - 1.5B fp32 ≈ 7 GB;
    - Ouro-1.4B fp32 ≈ 6 GB (4 loops → ~4× compute).
15. **CPU-only realistic?** Yes.
16. **CPU runtime.** 0.5B: ~5–10 h. 1.5B: ~15–30 h. Ouro arm: ~6–10 h.
17. **Training required?** No. An optional unlearning lesion (LoRA gradient ascent on 0.5B) takes ≤ 1 CPU-h.
18. **Ideal vs zero-budget: substantially different.**
    - Ideal: 7–70B models, where metacognitive signals are stronger, J-space exists, and Pilot A as originally drafted is meaningful.
    - At 0.5–1.5B, floor effects are a real risk.
19. **Valid / invalid conclusions from the $0 version.**
    - **Valid:** whether 0.5–1.5B instruction-tuned LMs' control behaviour tracks self-changes vs input changes, and through which internal signal.
    - **Invalid:** any claim about frontier LLMs; "LLMs lack self-monitoring" in general; any consciousness claim.

**Abandon if:** base lookup/abstention behaviour fails to discriminate natural known vs unknown items (AUROC < 0.6) in both small models.

---

## P2\*: post-J-space discriminating tests: does workspace content *govern* metacognitive control?

1. **Exact scientific claim.** In an open LM with a validated J-space, known/unknown status is carried by J-space contents, and *those* components (not matched non-J-space components) causally govern abstention and lookup.

   Secondary claims:
   - J-space entry is threshold-like with bimodality at threshold (ignition);
   - holding *k* concepts reduces J-space entry and flexible use of a new concept (central bottleneck).

   These test GNW's C1→C2 linkage and its dynamical predictions [dehaene2017science; dehaene2026commentary].
2. **Closest prior work.**
   - Gurnee et al. 2026 [gurnee2026workspace]: J-space; "damn"/failure tokens; ambiguous-mixture snapping; moderate dual-task cost (A.17).
   - Dehaene & Naccache 2026 (explicit calls).
   - Li et al. 2026 [li2026functionalmeta]: self-assessed-capability directions steer behaviour, not located relative to J-space.
   - Ferrando 2025 (familiarity directions).
   - Du & Hu 2026.
   - Community replications planning ignition (E15) and dual-task (E17) tests [hasin_jspace_qwen].
   - Rahbar 2026; Lam-Muir 2026.
3. **Precise difference.** Localising the *metacognitive control* signal with respect to the workspace (is C2 control workspace-mediated or automatic?), using matched non-J-space ablations; plus ignition tested with stimulus-strength and across-trial criteria rather than layer-wise probes.
4. **Falsifiable hypothesis.**
   - H2\* (GNW C1→C2): ablating J-space uncertainty components reduces abstention on unknown entities by ≥ 2× the effect of norm- and dimension-matched non-J-space ablations.
   - Alternative ("automatic gating"): familiarity-gated refusal operates outside the J-space; the effects are equal.
5. **Smallest decisive experiment.**
   - Model: Qwen3.5-4B with a community J-lens [idhan_jlens4b], validated first against the paper's sanity checks.
   - Items: 300 known + 300 unknown entities.
   - Conditions: J-space uncertainty-concept ablation; matched random-subspace ablation; familiarity-direction ablation.
   - Endpoint: abstention.
6. **Main confounds.**
   - Third-party lens validity;
   - single-token vocabulary restriction;
   - ablation collateral damage;
   - bf16 numerics on CPU;
   - "uncertainty concepts" chosen post hoc (must be frozen).
7. **Negative controls.** Random-subspace ablations; ablation of a non-metacognitive J-space concept of matched loading.
8. **What falsifies or weakens it.** J-space ablation effect ≤ matched non-J-space effect → C2 control is not workspace-mediated in this model. That is an informative negative. If the lens fails validation, the test cannot be run.
9. **What constitutes a meaningful paper.** A clean positive or negative on C2-in-workspace with strong controls would be significant. However, large labs and active replicators target exactly these tests, so scoop risk is high.
10. **Directness to the original question.** It tests a structural prediction of GNW in an existing model (theory-testing). It does not engineer anything.
11. **$0?** Borderline / no for the strong version.
12. **Zero-budget execution plan.**
    1. Check the 4B lens and its fidelity metrics.
    2. Run bf16 on CPU with all other applications closed.
    3. Keep N small (≈ 600 items × 3 conditions ≈ 1,800 passes ≈ 8–10 h).
    4. If the 4B lens fails validation, stop. Fitting a lens on CPU is ~25–250 CPU-h, and whether a workspace exists in sub-billion models is unknown.
13. **Minimum model size.** ~4B (smallest known J-lens target). The 9B lenses that exist are impractical on this laptop.
14. **Hardware / RAM.** 4B bf16 ≈ 8–10 GB free RAM, which requires closing most apps. 9B needs ≈ 18–20 GB and is impractical.
15. **CPU-only realistic?** Marginal.
16. **CPU runtime.** ~20–40 h for the core plus secondary tests (ignition: ~2 h; dual-task: ~4 h). Fragile under memory pressure.
17. **Training required?** No, if a valid lens exists.
18. **Ideal vs zero-budget: very different.**
    - Ideal: 9–27B models on a GPU with larger N.
    - Zero-budget: one 4B model, small N, third-party lens.
19. **Valid / invalid conclusions from the $0 version.**
    - **Valid:** properties of one 4B model's J-space under one lens.
    - **Invalid:** general conclusions about workspaces in LLMs or verdicts on GNW.

**Abandon / defer if:** the 4B lens fails validation, or ignition and dual-task results from better-resourced groups appear first.

**Decision: deferred.** Revisit only if the researcher *chooses* to accept a free-GPU dependency (e.g., Kaggle's weekly GPU quota), documented as a limitation.

---

## P5\*: developmental decomposition + architectural integration: when does genuine self-monitoring emerge?

### The researcher's seed, decomposed into three levels (as requested)

**A. Trainable capacities**, each developed or trained separately and evaluated on its own before integration:

| Capacity | Instantiation | Separate evaluation |
|---|---|---|
| A1 Knowledge / recall ("first-order competence") | Tiny causal transformer *store* trained on a synthetic entity–relation–value world [allenzhu2024physics31; zucchet2025facts; nishi2025shattering] | Fact accuracy; held-out "unknown" facts |
| A2 Metacognitive monitoring | Small monitor trained to predict the store's correctness | Calibration / type-2 AUROC on natural data |
| A3 Agency / control | Controller choosing ANSWER vs LOOKUP (external memory, cost c), trained to maximise reward | Reward on natural data |
| A4 Temporal continuity / memory (Stage 2) | External episodic memory receiving facts learned after deployment; continual fine-tuning of the store | Retention, interference |
| A5 Self-modelling via self-variation (developmental) | Monitor training under random structural perturbations of the store ("experience of one's own variation", SOMA [cleeremans2020learning]; cf. self-interventional learning [tomaszewski2026sil]) | Prediction of perturbed-store correctness on *trained* perturbation families |

**B. Architectural mechanisms** (the integration manipulations):

| Mechanism | Levels |
|---|---|
| B1 Information access of monitor/controller | (i) input-only (an external observer of the query); (ii) store's subject-position state (pre-recall, "familiarity level"); (iii) store's final-position state (post-recall); (iv) **re-entrant**: the store's own answer is re-embedded and passed back through the store, and the monitor reads the second pass ("what did I just retrieve?", RPT/GNW-style re-entry) |
| B2 Bottlenecked broadcast (Stage 2) | The monitor reads the store only through a k-dimensional workspace shared with other consumers (GNW C1→C2 dependency) |
| B3 Integration topology | Monitor output on the controller's decision pathway vs beside it [xie2026structural] |

**C. Emergent integration property (pre-registered, never trained).**

*Selective self-tracking under held-out self-changes:* the integrated system increases LOOKUP specifically for items whose answers it has *lost*, through changes it never experienced in training:

- **T1** targeted unlearning of specific facts (gradient ascent on those facts);
- **T2** catastrophic interference from continual learning of new facts [zucchet2025facts];
- **T3** newly acquired facts: the system should *stop* looking up items it has since learned;
- plus a **monitor lesion** test (metacognitive "blindsight": store accuracy intact, selectivity lost).

**Why the obvious reductions fail by design:**

| Reduction | Control |
|---|---|
| More parameters | All monitor/controller variants are parameter-matched. The input-only observer gets the same depth and width |
| More inference compute | Re-entry is compared with an equal-FLOP single-pass monitor (extra layers) and with a "second pass without re-embedding the answer" control |
| More training data | Identical monitor training items across cells. Self-variation uses the same items under perturbation, with label count matched to the natural cells |
| Ensembling / voting | An ensemble of non-integrated input-only monitors cannot see self-changes (negative control) |
| **Self-specificity ("yoked twin")** | A monitor reads an *unlesioned twin* store trained identically on another seed. If it "tracks", the effect is not self-tracking |
| Prompt length | Fixed-length queries |
| Leakage | Lesions are applied after monitor/controller training. Held-out lesion families are disjoint from self-variation families. Query embeddings are verified unchanged by lesions |
| Metric artefacts [schaeffer2023mirage] | Continuous DVs (tracking index, AUROC, reward). Interactions tested against additive and multiplicative nulls [okawa2023multiplicative] |

### 19-point specification

1. **Exact scientific claim (Level 1).** In a fully observable synthetic system whose knowledge, monitoring and control capacities are developed separately and then integrated:
   - counterfactually self-dependent monitoring (E3: tracking never-experienced changes to the system's *own* knowledge) emerges only under specific information-flow and developmental conditions;
   - under natural training with a familiarity–knowledge correlation, an **observationally equivalent but interventionally distinct** world-tracking monitor emerges instead.

   **Level 2 (conditional):** these conditions bear on the HOT-2 / GNW-C2 indicator ("metacognitive monitoring that distinguishes reliable from unreliable first-order representations") and on SOMA's claim that higher-order monitoring is learned. **Level 3:** none.
2. **Closest prior work.**
   - Toy agents with GWT/HOT/IIT modules and self-model ablation [phua2025ablations];
   - structural-integration self-monitoring with a parameter-matched null [xie2026structural];
   - self-interventional learning [tomaszewski2026sil];
   - second-order networks [pasquali2010know];
   - forms of trained metacognition in LLMs [yax2026forms];
   - synthetic knowledge substrates [allenzhu2024physics31; zucchet2025facts; nishi2025shattering];
   - self-modelling regularisation [premakumar2024selfmodel];
   - GW routing [chateaulaurent2025chain];
   - SOMA theory [cleeremans2020learning].
3. **Precise difference.** We have not yet identified prior work that combines:
   - (i) held-out *self-change tracking* as the pre-registered emergent integration property (E3);
   - (ii) a parametric manipulation of *shortcut availability* (familiarity–knowledge correlation ρ);
   - (iii) a factorial over information access × re-entry × developmental regime with interaction tests;
   - (iv) behavioural control (lookup) under lesions;
   - (v) the yoked-twin self-specificity control.

   Phua (2025) engineers dissociations by construction. Xie (2026) studies performance benefits. Tomaszewski (2026) predicts structural consequences, not knowledge self-monitoring. Yax et al. (2026) study fine-tuned LLMs without lesions.
4. **Falsifiable hypotheses** (thresholds provisional until the Stage-1 variance estimate):
   - **H5a (observational equivalence):** in a high-ρ world, monitors with access (ii) and (iii) reach natural-data type-2 AUROC within 0.03 of each other, yet differ in lesion tracking index (TI) by ≥ 0.3.
   - **H5b (shortcut):** under natural training, the best single-pass internal-access monitor's TI on held-out lesions decreases with ρ; at ρ ≥ 0.9, TI < 0.3.
   - **H5c (re-entry):** re-entrant access yields higher TI than FLOP-matched single-pass access.
   - **H5d (development):** self-variation training raises held-out TI, and the gain requires internal access (access × regime interaction).
   - **H5e (integration):** controller lookup selectivity under T1–T3 shows a superadditive access × regime interaction, exceeding the multiplicative null.
5. **Smallest decisive experiment (Stage 1, ≈ 2–4 CPU-h).**
   - Two worlds (ρ = 0.95, ρ = 0.5) × 5 seeds.
   - Train stores (~1–5 min each).
   - Cache activations; train access-(i/ii/iii) monitors on natural correctness.
   - Apply one held-out lesion family (targeted unlearning of 20% of known facts) plus a sham.
   - Compute TI and natural AUROC.
   - This decides H5a and H5b, and whether the paradigm is non-degenerate.
6. **Main confounds.**
   - Lesion collateral damage ("representation shattering" [nishi2025shattering]);
   - monitors exploiting generic lesion artefacts (activation-norm shifts) rather than recall content (handled by norm-matched shams and by lesions that shift norms without affecting recall);
   - capacity or compute differences across access conditions;
   - hyperparameter sensitivity;
   - synthetic-world idiosyncrasy (vary world size, relation structure, name length).
7. **Negative controls.**
   - Input-only observer (TI must be ≈ 0 on lesions and high on input corruption);
   - yoked-twin monitor (TI ≈ 0);
   - sham lesions;
   - unrelated-fact lesions;
   - shuffled-label monitors.
8. **What falsifies or weakens it.**
   - TI ≥ 0.8 for internal-access monitors at all ρ under natural training: H5b false. Self-tracking then comes "for free" with internal access, and the developmental and shortcut thesis collapses. This is still a publishable negative, but a weaker paper.
   - Self-variation gives no held-out gain: H5d false.
   - No superadditive interaction: H5e false. The integration claim reduces to additive effects.
   - **Abandon** if:
     - lesions cannot be made selective (collateral loss > 30% of target loss);
     - natural-data monitors stay at chance;
     - conclusions flip across seeds or reasonable hyperparameters.
9. **What constitutes a meaningful paper.**
   - Working title: "Observationally equivalent, interventionally distinct: conditions for genuine self-monitoring in artificial systems".
   - Contents:
     - (1) a formal identification framework (N1) proving that calibration and privileged-access tests cannot separate self-tracking from world-tracking monitors when a familiarity statistic is sufficient on natural data;
     - (2) controlled experiments mapping when E3 self-monitoring emerges;
     - (3) a small-LM bridge (P1\* at 0.5–1.5B, plus Pythia training checkpoints) testing whether pretrained LMs resemble the natural-trained toy regime.
   - Suitable venues: ICLR/NeurIPS (methods), CogSci, *Neuroscience of Consciousness* (theory).
10. **Directness to the original question.** High. It is an engineering and developmental study of *which architectures and training regimes produce a consciousness-theory-relevant property* (counterfactually self-dependent monitoring). It operationalises the researcher's decomposition → integration idea with a falsifiable emergence criterion. **Scope caveat:** one property, toy scale.
11. **$0?** Yes, with large margin.
12. **Zero-budget execution plan.**
    1. Write the world generator, store, monitors, controller and lesion library in PyTorch (CPU).
    2. Freeze the preregistration (Stage 1 hypotheses H5a–H5b; thresholds after a 1-seed pipeline test on a throwaway world).
    3. Stage 1 (~3 CPU-h).
    4. Stage 2: full factorial: 2–3 ρ levels × 4 access conditions × 2 regimes × 10 seeds; T1–T3 tests; controller (~15–30 CPU-h, run overnight).
    5. Stage 3: robustness grid (world size, depth) and B2 workspace factor (~10–20 CPU-h).
    6. Bridge: P1\* on Qwen2.5-0.5B/1.5B and Pythia-160M/410M checkpoints (~15–30 CPU-h).
    7. All raw outputs in `results/raw/`, seeds logged.
13. **Minimum model size.** Store 0.5–5M parameters; monitors and controllers < 1M.
14. **Hardware / RAM.** < 2 GB RAM; any laptop CPU.
15. **CPU-only realistic?** Yes (measured).
16. **CPU runtime.** Stage 1: ~2–4 h. Full program: ~30–70 h total including the bridge, which is the slowest part.
17. **Training required?** Yes, but tiny: each store trains in minutes; monitors train in seconds on cached activations.
18. **Ideal vs zero-budget: moderately different.**
    - The ideal adds a *pretrained-LLM replication of the developmental intervention*: self-variation fine-tuning of 7B monitors with held-out lesion tests.
    - The $0 version substitutes a small-LM consistency bridge.
    - The core synthetic science is *identical* in both, because the toy study does not need more compute.
19. **Valid / invalid conclusions from the $0 version.**
    - **Valid:**
      - existence proofs and boundary conditions for when E3 self-monitoring emerges in this class of systems;
      - the identification theorem and its empirical illustration;
      - whether small pretrained LMs behave like the natural-trained regime.
    - **Invalid:**
      - that LLMs implement the same mechanism;
      - that self-variation training would work in large LMs;
      - that the toy system has "consciousness" or any property beyond the measured Level-1 one;
      - that monitoring of *facts* generalises to perceptual or phenomenal monitoring.

---

## N1: causal-identification framework for machine self-monitoring (new; merged into P5\*)

- **Claim.** For any monitor M of a system S, define:
  - **natural-data observational equivalence:** two monitors M_self and M_world induce identical joint distributions of (input, report, correctness) whenever a familiarity statistic F(input) is sufficient for correctness under the natural data distribution;
  - **interventional identifiability:** M_self and M_world differ under interventions do(S → S′) that change correctness while holding the input fixed.
- **Consequences:**
  - calibration (E1) and privileged access (E2) are insufficient for self-tracking (E3);
  - privileged access can come from *internally computed* familiarity, which is still world-tracking.
- **The minimal control set needed for identification:**
  - sham interventions (rule out generic anomaly);
  - input-corruption contrasts (rule out difficulty);
  - unrelated lesions (rule out global state);
  - yoked twins (rule out non-self information).
- **Closest prior.** Privileged-access definitions [song2025privileged]; necessary conditions [singh2026reality]; grounding criterion [lindsey2025introspection]; [comsa2025introspection; kammerer2023introspective].
- **Difference.** A formal *identification* statement (what data can and cannot settle) plus a minimal sufficient design, rather than criteria lists.
- **$0.** Pure theory plus simulation.
- **Alone, it is weaker than P5\*;** combined, it is P5\*'s measurement backbone and is validated in a system with known ground truth. That validation is something LLM studies cannot provide.

---

## Evaluation of the proposed combination ("P1 = measurement, P5 = engineering")

**The combination is correct in spirit but needs one amendment.**

- The *measurement* role belongs mainly to **N1 plus the E3 battery, validated in P5\***. P5\* is the only setting where ground truth is known: which facts were lesioned, and what each monitor reads. A measurement framework that has never been validated against ground truth would be weak. Applying it first to LLMs (P1\*) would leave every negative or positive result ambiguous.
- **P1\*** then becomes the **external-validity bridge**: does a small pretrained LM behave like the toy's natural-trained (world-tracking) regime or its self-tracking regime?
- **P2\*** would be the natural third paper (C2 inside a discovered workspace) once compute allows.

**Why not lead with P1\*?** Three reasons:

- its core phenomenon is anticipated by prior work;
- at $0 it is restricted to models where floor effects are likely;
- its mechanistic claims cannot be checked against ground truth.

**Why not P2\*?** It fails the $0 constraint in its decisive form, and it is heavily contested territory.

---

## Final recommendation

**Adopt P5\* + N1 as the primary research direction, with P1\* as a pre-specified bridge study. Defer P2\*.**

Working research question:

> *Under what architectural (information access, re-entry, broadcast) and developmental (self-variation, shortcut availability, separate-vs-joint training) conditions does an artificial system develop monitoring that is counterfactually dependent on its own internal states, rather than an observationally equivalent model of task difficulty? And do small pretrained language models resemble the self-tracking or the world-tracking regime?*

Why this beats the alternatives on the stated criteria:

| Criterion | P5\* + N1 | P1\* | P2\* |
|---|---|---|---|
| Scientific importance | High: resolves an identification problem blocking all self-report/metacognition indicator claims | Medium | High |
| Genuine novelty | Medium–high (combination not found; components have neighbours) | Low–medium | Medium, contested |
| Falsifiability | High (5 pre-registerable hypotheses, each can fail) | High | High |
| Interpretability | Very high (ground truth known) | Medium | Medium (lens validity) |
| Causal identification | Very high | High | Medium–high |
| Connection to consciousness theory | Medium–high (HOT-2, GNW-C2, SOMA, RPT re-entry), explicitly Level 2 | Medium | High (GNW) |
| $0 feasibility | Excellent | Good, with floor-effect risk | Poor–marginal |
| Paper potential | Main-venue plausible if H5a/H5b hold or fail cleanly | Workshop | High if feasible, scoop risk |

**What would make us switch:**

- Stage 1 shows degenerate behaviour (all internal-access monitors trivially self-track at every ρ, or nothing tracks anything). Then publish N1 + negative toy result as a short paper, and move effort to P1\* at 1.5B.
- Full-text reading of Cohen & de Melo (or another paper) reveals a synthetic-system lesion-tracking study.
- The researcher accepts a free-GPU dependency. Then P2\* becomes viable as a second paper.

**Immediate next steps (no preregistered experiment is run until the user approves):**

1. **User action requested:** obtain the Cohen & de Melo PDF. OpenReview's Cloudflare check blocks automated access, and the paper is not on arXiv. Either complete the check in the browser pane, or save the PDF to `research/literature/papers/`. I will then confirm or revise the §10 verdict against the checklist in memo §10.2.
2. Read in full: [yax2026forms; tomaszewski2026sil; xie2026structural; phua2025ablations; gu2026unlearners].
3. Draft the P5\* Stage-1 preregistration and code. Validate the pipeline only on throwaway worlds (seed ≠ preregistered seeds).
4. Freeze the preregistration (commit hash recorded in `logs/decisions.md`), then run Stage 1.
