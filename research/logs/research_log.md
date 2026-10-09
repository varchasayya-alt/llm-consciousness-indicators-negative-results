# Research log

## 2026-10-01: Phase 0 (exploratory) begins

**Setup**

- Created the repository skeleton under `research/`.
- Environment: Windows 11, Python 3.12, no CUDA GPU, 31 GB RAM.
- Consequence: experiments need a rented GPU; code debugging on ≤1.5B models on CPU.

**Literature search (~70 searches/fetches).** Main surprises relative to the starting brief:

- **Concept-injection introspection is crowded and contested** after Lindsey (2025). There are mechanistic, replication and critique papers through Sept 2026:
  - Macar et al.;
  - Lederman & Mahowald;
  - Pearson-Vogel et al.;
  - Hahami et al.;
  - Singh, Linzen & Ravfogel (COLM 2026);
  - Ferrara;
  - Aoki et al.
- **Anthropic's "Verbalizable Representations Form a Global Workspace" (Gurnee et al., July 2026)** reports J-space in Claude. The Dehaene & Naccache commentary lists open tests: ignition, dual-task bottleneck, C2 self-monitoring, local–global, trace conditioning. Open-model J-lens tooling and replications exist.
- **Metacognition evidence favours "world/difficulty tracking":**
  - Moran & Whiting 2026 (rank-one cross-model confidence);
  - Ashuach et al. ACL 2026 (privileged knowledge only for facts);
  - Cohen & de Melo ICML 2026 ("danger zone" in degraded systems; full text not yet read).
- **Toy multi-theory agents with ablations already exist** (Phua 2025). Closest to the seed idea.

**Key conceptual move**

- Proposed an evidence hierarchy for self-directed capacities (E0–E4) whose core is **counterfactual self-dependence (E3)**: self-assessments must track interventions on the system's own competence with input fixed.
- This generalises Lindsey's grounding criterion.
- It leads to the **lesion-tracking / artificial anosognosia** paradigm.

**Outputs**

- Memo `memo/exploratory_research_memo.md` (v0.1).
- Literature DB (138 entries; 57 `memory-verify`, 10 `partial`).
- Pilot protocol drafts.
- Directions / rejected / open-questions files.
- Decisions D1–D9.

**Next:** full-text novelty check of the closest works (memo §9), then freeze Pilot A.

## 2026-10-01 (later): v0.2 targeted novelty audit

- **Git:** repository initialised; Phase 0 committed (`5ff787a`).
- **Cohen & de Melo (ICML 2026):**
  - The full text could not be retrieved: OpenReview's Cloudflare check (not bypassed); the paper is not on arXiv; the Semantic Scholar API was rate-limited.
  - Reconstruction from the Lacuna summary and search snippets: six competence regimes; global degradation (quantisation / noise in internal states); three behavioural confidence channels; matched-accuracy divergence; no internal analyses.
  - **Not** item-selective lesions.
- **But item-selective lesion + re-ask designs exist elsewhere:**
  - Gu et al. 2026 (ACL): unlearning → confabulation, rejection metrics;
  - Hasegawa et al. 2025/2026: editing → token-confidence underconfidence.
  - **→ P1 original rejected (D11).**
- **New important neighbours:**
  - Yax/Palminteri/Oudeyer 2026 (two forms of trained metacognition);
  - Marina et al. 2025 (question-only adaptive retrieval matches LLM uncertainty);
  - Xie 2026 (structural integration; parameter-matched null comparable);
  - Tomaszewski 2026 (self-interventional learning);
  - Du & Hu 2026 (recognition–refusal misalignment);
  - Li et al. 2026 (functional metacognition directions);
  - Zhao et al. 2026 (overconfidence circuits);
  - community J-lens resources (4B–27B) and replications planning ignition / dual-task tests.
- **Feasibility (no hypotheses examined):**
  - CPU benchmark → `results/raw/feasibility/cpu_benchmark.json`;
  - synthetic store memorisation → `results/raw/feasibility/store_training_feasibility.json`. 97.6% train-fact accuracy in 0.34 min.
- **Outputs:**
  - memo v0.2 §10 (Novelty Audit);
  - `memo/decision_document_v2.md`;
  - literature DB now has 165 entries;
  - directions tracker, rejected ideas and decisions D10–D16 updated.
- **Blocking item for the user:** obtain the Cohen & de Melo PDF.

## 2026-10-01 (later): Stage-1 design / preregistration (v0.3)

- **Focused review.** Added 7 references (172 total):
  - edit detection from hidden states (Youssef et al., NAACL 2025);
  - unlearning traces detectable at >90% (Chen et al., 2025);
  - whitebox meta-models (Chen et al., AISTATS 2019);
  - ConfidNet (2019);
  - model stitching (Bansal et al., 2021);
  - probe transfer (Srey et al., 2026);
  - MetaErr (2026).
- **Conclusion:** "a monitor reading hidden states predicts errors" and "edits are detectable from hidden states" are both established. Stage-1 novelty must come from the dissociation design (memo/prereg Appendix E).
- **Drafted:**
  - preregistration (24 sections + appendices A–E + open decisions);
  - config draft;
  - system diagram;
  - simulation-only power analysis.
- **Not done (by design):** no stores trained for the experiment, no monitors, no calibration runs, no hypothesis-related quantities.

## 2026-10-02: Calibration revision v2 → STOP at C4val

- **Redesign succeeded on its target.** Selective forgetting is achieved (collateral ≤ 1% in every retained-fact category on fresh seeds) with KL-to-frozen-original retain.
- **C4val failed the pre-declared sham-fingerprint gate** (matched G2 0.903 on one validation seed; 0.85–0.90 everywhere). The cause is a procedure fingerprint (direction of change at the answer sites).
- **Stopped and reported to the PI**: `experiments/stage1/calibration_stop_report_C4.md`. Recommended next step: a v3 direction-matched sham.

## 2026-10-02: Stage-1 v3 (within-target identification). Methods revision committed before any v3 store run

- **PI decisions:** approved the within-target design with refinements:
  - θ_pre is primary, adjusting only for pre-intervention information;
  - θ_gen is the robustness estimand, with labels A/B/C;
  - residualized-post formulation;
  - continuous P2;
  - fresh dev seeds 9031–9033 and validation seeds 9021–9023;
  - mechanism variants V0–V3 only.
- **Simulation-only statistic comparison** (`v3_statistic_simulation/`): the symmetric pre-state residualized-post statistic (S9) was selected. It is the only candidate clean under null, regression-to-the-mean and state-visible susceptibility in both loss regimes. H2 worlds get label B and never A; H3 worlds get A. The change-score and P_i-only alternatives fail.
- **Fake-data dry run:**
  - all nine worlds give the expected labels;
  - K1 was corrected after the dry run exposed a susceptibility leak (D42);
  - F6 was found to be noise-limited (D43); the approved F6 stays binding and the alternative F6′ is recorded only.
- **Next:** C2dev → C4dev (V0–V3 grid) → C2val → C4val → C7–C10 → VAL → STOP with the calibration package.

## 2026-10-02 (evening): v3 C4dev → STOP

- **C2dev:** the stores pass.
- **C4dev:** none of the V0–V3 cells passes continuous retention or outcome diversity.
  - The cause is tail flattening from the uniform-target objective: held-out runner-up answers gain 1.3–1.9 nats.
  - Uniform forgetting also compresses forgotten items to a near-uniform endpoint.
- **Identifiability is good**, so the design itself is viable; the parametric intervention is not.
- **Report:** `calibration_stop_report_v3_C4dev.md`. Recommended next step: Option B1 (dual-route store with a locally editable fact memory). Awaiting the PI.

## 2026-10-03: v4 (B1) implemented, verified, then STOP at V4-F1

- **Code and verification.** Code, tests (42 pass), config and plan were committed before any development seed. The engineering dry run on a burned seed found and fixed two issues: retrieval trainability (D47) and the ridge GCV failure when features outnumber items (D49). The B1-world and high-dimensional verifications ran; one open issue is logged (D50).
- **V4-F0** passed.
- **V4-F1 (K-2):** the parametric backup is too strong (route-A accuracy 0.80–0.87 for every p_rd), and route dropout does not control it.
- **Next step:** awaiting the PI. Recommended O1, gradient-isolated parametric learning for memory-available presentations.

## 2026-10-03 · v4.1 approved (O1) and implemented before any development run

- **PI decisions:**
  - O1 approved;
  - O2 and O3 not pursued;
  - a fixed p_rd grid {0.05, 0.10, 0.20, 0.35, 0.50} on fresh development seeds 9061–9063;
  - D50 resolved by restricting affirmative support to label A (D54);
  - M1/M2 read as sufficiency diagnostics only (D55).
- **Implemented:** the exact P/M partition and routing (D52), with invariants unit-tested and asserted at runtime.
- **Added:** pre-declared dose-coherence checks DC-1 and DC-2 (D53), so p_rd cannot be selected merely because it happens to hit 50% while the mechanism is uncontrolled.
- **Engineering note** (burned seed): isolation slows early memory-route learning, because the downstream readout no longer co-adapts to memory. This is a property of the approved design. V4.1-F0 and the pre-declared C1 rule decide whether the store converges.

## 2026-10-03 · v4.1 stops at F0 (K-1)

- The gradient-isolated store is cleanly isolated but not viable as an integrated store.
- The memory addresses correctly and carries the answer, but the parametric downstream, never trained on memory-present examples, does not decode it. The co-training then escalates residual and memory scales by two to three orders of magnitude, and accuracy collapses after an early peak.
- Lesson: separating *who learns from which presentation* also separates *who learns to integrate*. A cleanly separable dual-route store probably needs sequential development (a frozen parametric readout before memory training) rather than interleaved isolated co-training.
- Decision pending with the PI (O4 / O1-N / O5).

## 2026-10-03 · v4.2 (final B1 revision) approved and implemented before any development run

- **v4.1 recorded as a legitimate negative methods result (D57):** isolation was correct, the memory carried the answer and retrieval worked, but two separately updated pathways could not co-adapt under simultaneous training.
- **v4.2 develops the store sequentially:**
  - a parametric stage whose covered-fact dose is the per-epoch inclusion probability p_rd;
  - then a memory stage on the frozen network, with a zero NULL and a bounded injection.
- **Design property:** the final store with all slots masked is bit-identical to the Stage-A network. The developmental dose is therefore measured exactly where it is used.
- **New pre-declared checks:** memory rescue, no damage, and non-overwrite (INT-1).
- **If any gate fails, the B1 line ends** (D60).

## 2026-10-03 · B1 closed; pivot memo written

- **v4.2 (sequential development)** solved what v4 and v4.1 could not:
  - controllable dose;
  - usable integration on a frozen network (rescue without damage, no overwrite);
  - perfect locality;
  - good identifiability.
- **It failed on the two remaining requirements:**
  - outcome-diversity tiers defined relative to intact competence are unattainable when a strong memory sets the intact margin;
  - the value-transplant negative control is not neutral at the graded level.
- **Main lesson across five cycles:** Stage 1's bottleneck is constructing a valid, controllable, input-invisible competence change, not measuring monitoring. The statistical machinery held up throughout.
- **Pivot options:**
  - D1: N1 methods paper;
  - D2: self-generated competence change through the store's own continued learning;
  - D3: P1\* lesions in small pretrained LMs.
- **Recommendation:** D1 now; then the D2 store-only kill test.

## 2026-10-03 · Post-B1 pivot: D2 (self-generated competence change) designed; D1 audit begun

- **D2 makes development itself the intervention.** The plain store keeps learning new entities without replay, and every known fact undergoes the same process.
- **The S0 analysis caught a scale issue before any run.** Intact margins are tightly concentrated (IQR about 0.87 nats), so the meaningful spread requirement is on the change ΔC, not on post competence.

## 2026-10-03 · D2 stops: self-generated change is graded and unpredictable, but all-or-nothing in magnitude

- Continued learning without replay forgets 73–96% of known facts by the time the new facts are learned.
- The change has the identification profile Stage 1 wants: graded, essentially unpredictable from pre-states, only partly generic, not a familiarity effect. But its magnitude cannot be brought into a partial-loss window with the pre-declared lever.
- **Lesson R8:** check lever reachability, not only gate satisfiability, before a run.
- **Next, per the pre-specified plan:** D3 (pretrained-LM lesions), pending PI approval and download permission. The D1 note continues in parallel.

## 2026-10-03 · C15-R: routing/rescue design memo

- **The question is now exact.** Is the model's own verification verdict, decodable but inert in the use pass, made to govern candidate adoption by writing it into the lens-defined workspace subspace? And not by writing the same content, at the same norm, into the orthogonal complement?
- **Key design moves:**
  - a rank-1 bridge separates content (what is read) from route (where it is written);
  - minimal-pair candidates;
  - a verbal-routing reachability control (lesson R8);
  - a report-access gate, so that the workspace is validated as the route for report;
  - rank-1 subspace LEACE replaces broad J ablation for necessity.
- **Honest limit:** linearly, routing is item-signed steering. The science is which subspace the controller reads, established by the interaction and the semantic, direct-path and disruption controls.
- **Status:** awaiting PI approval and download permission. Nothing implemented.

## 2026-10-03 · C15-R final pre-registration design

- **Five structural changes:**
  - F becomes a same-pass signal read before the policy is known;
  - the workspace subspace becomes task-independent;
  - the "workspace" label must be earned on generic assays, including selectivity;
  - the model is chosen by assay;
  - rescue must show broadcast and flexible, policy-reversible use, not just a stronger push on one decision.
- **Why reversed policy P3 matters:** information about invalidity should *increase* keeping when the instruction is to collect faulty outputs; an action-steering vector cannot do that.
- **Honest bottom line:** the workspace assay is the binding risk. Published Qwen3 ablations are anti-selective at matched norm. Significance stays MEDIUM, and a STOP at the A-stage is plausible.

## 2026-10-05 · C15-R A-stage built; pre-run commit

- **Pipeline built.** The task-independent workspace assay is implemented and tested (19 unit tests). Z0 passed for all three candidates. The Qwen3.5-2B checkpoint/lens identity is resolved: the "-pt" lens belongs to the Base model and is not used.
- **SMOKE engineering found three things the design memo did not anticipate:**
  1. The linearised KL proxy is useless for matching real disruption.
  2. Near-tie token flips swamp a raw agreement criterion.
  3. Greedy-pursuit screening is exact enough for Qwen3 but not for Qwen3.5-2B.
- **Fixes, all before any G data:**
  - empirical KL matching;
  - robust-position agreement with monotone dose admissibility;
  - per-model exact GP where screening failed validation.
- **Next:** SELECT on G_select for all three (detached), CHOOSE, FREEZE (committed), CONFIRM once.

## 2026-10-06 · C15-R A-stage: no workspace-like route in Qwen3-1.7B, Qwen3.5-2B or Qwen3-4B

- **No candidate passes.** None of the three small open models shows the full pre-registered workspace-like profile on generic material.
- **The signatures come apart across models.**
  - Qwen3.5-2B's lens-defined J subspace is route-specifically reportable and broadcast across functions, but anti-selective.
  - Qwen3-4B's is causally relevant and selective at matched functional impact (anti-selective at matched norm), but barely reportable or broadcast.
- **Two assays failed for every model:** the intermediate-entity readout and the cross-position transport decoder. These look like assay-sensitivity limits.
- **Protocol consequence:** C15 stops as a workspace study (ST-2); G_confirm remains untouched.
- **Most useful methodological lesson:** matched-norm and matched-impact selectivity can reverse.
