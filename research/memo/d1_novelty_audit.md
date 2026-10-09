# D1 novelty audit: is the N1 identification argument or the within-target estimand already published?

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Purpose | PI requirement before drafting D1: a targeted full-text novelty audit on eight topics. If novelty is weaker than expected, reframe D1 as a methods / negative-results note |
| Independence | Written in parallel with D2. **No D2 threshold, gate or choice was influenced by this audit;** all D2 gates were committed in `7742d07` before the audit began |

**Method.**
- Targeted web searches on the eight topics.
- Each closest paper was read from its full text through arXiv HTML or PMC, with structured questions:
  - Is there a formal identification or observational-equivalence argument?
  - Is the intervention input-fixed, acting on the system's own state or competence?
  - Is there an estimand that conditions on pre-change or difficulty information?
  - Are generic-change and anomaly alternatives separated?
  - What controls are used?

**Limitations of the method.**
- Full texts were read through an automatic summariser that answered the structured questions. The key passages are quoted below, but this is not a line-by-line reading.
- **Cohen & de Melo (ICML 2026) is still inaccessible:** OpenReview's verification wall blocks access, and I did not attempt to bypass it.
- The classic cognitive-science sources marked `memory-verify` in the bibliography must be checked before citation.

## 1. Closest works found

| Work | What it does | Formal identification / estimand? | Input-fixed intervention on own state or competence? | Overlap with N1 |
|---|---|---|---|---|
| **Smith, Zakrzewski & Church 2016** [smith2016formal], *Psychon. Bull. Rev.* | Associative (first-order) and signal-detection metacognitive models fit macaque uncertainty data equally well; "a perfect mathematical correspondence" | **Yes, a formal observational equivalence** (in animals) | Discusses discriminating designs (deferred reinforcement, judgements without stimulus-linked cues, working-memory load) | **High.** Anticipates N1's observational-equivalence claim |
| **Hampton 2001** [hampton2001monkeys]; **Carruthers 2008** [carruthers2008skeptical] | No-sample probe trials manipulate the animal's own memory state with cues and delay held fixed. Carruthers argues for first-order explanations | Conceptual | **Yes** (memory-state manipulation) | **High, conceptually.** The input-fixed competence-change idea is about 25 years old in comparative metacognition |
| **Singh, Linzen & Ravfogel 2026** [singh2026reality], COLM | Two necessary conditions (privileged access, second-order computation). States "first- and second-order accounts can give rise to identical model self-reports". Three-way design: activation steering vs input-level "gaslight" vs control | No theorem or estimand (construct-validity argument) | Yes (activation steering; input-only probes as control) | **High.** Informal observational equivalence; separates generic anomaly detection from state-specific sensitivity, close to N1's θ_gen concern |
| **Song et al. 2025** [song2025privileged] | Defines introspection as more reliable than any equal-or-cheaper third-party process; temperature self-reports equal cross-model prediction | No (conceptual definition) | No (self vs third-party) | Medium (privileged-access criterion) |
| **Binder et al. 2024** [binder2024looking] | Self- vs cross-prediction. After fine-tuning changes the model's behaviour, its self-prediction follows the new behaviour (35.4% vs 21.7%) | No | **Yes** (behaviour changed by fine-tuning; prediction tracked) | **Medium–high.** A "meta-model under capability change" precedent, without pre-change adjustment or an item-level within-target estimand |
| **Ackerman 2026** [ackerman2026limited], ICLR | Delegate and second-chance games. "Introspective ability" = **partial correlation after removing variance explained by surface difficulty cues** | **Partial-correlation estimand** | No (observational) | **Medium–high.** Residualising monitor behaviour on difficulty information is established |
| **Blandfort & Pawar 2026** [blandfort2026strangers] | Self-reports are generic. "Net of shared knowledge, self-report adds little" (a partial correlation beyond the cross-model behaviour mean) | Partial-correlation estimand (between-model) | Fine-tuning on own behaviour (changes the target) | Medium |
| **Ashuach et al. 2026** [ashuach2026consensus], ACL | Self vs peer probes; disagreement subsets isolate "private" correctness signal (factual yes, math no) | Decomposition z_public ⊕ z_private; correlational | No | Medium |
| **Ferrara et al. 2026** [ferrara2026owmi] | Input-fixed internal interventions with **matched sham**. Reports at chance (AUROC ≈ 0.50), but probes recover the intervention | SDT estimands (d′, AUROC), a leakage decomposition | **Yes** (with sham) | Medium (intervention detection, not competence tracking) |
| **Lindsey 2025** [lindsey2025introspection] | Accuracy / grounding / internality / metacognitive-representation criteria. Grounding = counterfactual dependence of the report on the state | No estimand | Yes (concept injection) | Medium (counterfactual-dependence criterion) |
| **Kumaran et al. 2026** [kumaran2026causal], *NMI* | Steering a confidence representation shifts abstention | No | Intervenes on confidence, **not competence** | Low–medium |
| Gu et al. 2026 [gu2026unlearners]; Hasegawa et al. 2025 [hasegawa2025underconf] | Same question re-asked after unlearning or editing; refusal and confidence measured | "Unlearning honesty" definition; no identification estimand | Yes (unlearning, editing) | Medium (phenomenon, not estimand) |
| Yax et al. 2026 [yax2026forms] | Trained confidence tracks output consistency far from training data and accuracy near it | No (Δr comparisons) | No competence change | Low–medium (rival "consistency" monitor) |
| Tomaszewski 2026 [tomaszewski2026sil] | A network intervenes on itself and learns to predict the functional consequences; permuted-mapping negative control | No ("firewalls" rather than identification) | Self-lesions | Low (self-model of structure, not competence monitoring) |
| Continual calibration 2026 [contcalib2026] | Confidence and coverage degrade under sequential fine-tuning (aggregate) | No | Self-generated capability change, aggregate only | Low (D2-adjacent phenomenon) |

## 2. Per-topic verdicts

1. **Causal identification of machine self-monitoring.** The need for designs where first- and second-order accounts diverge is stated explicitly (Singh et al.), and counterfactual-dependence criteria exist (Lindsey; Comșa & Shanahan). A *formal* observational-equivalence result exists in animal metacognition (Smith et al. 2016). **I found no formal identification statement written for machine monitors of item-level competence.** The idea is nonetheless well anticipated.
2. **Competence vs difficulty dissociations.** These are established:
   - Ackerman 2026 residualises on difficulty cues;
   - the human literature on metacognitive efficiency and cue utilisation controls first-order performance and difficulty [fleming2014measure; maniscalco2012; koriat1993fok];
   - Hampton separates memory strength from cues.
3. **Interventional metacognition.** Present: Hampton; Lindsey; Singh (three-way); Ferrara (sham); Kumaran (steering confidence); Gu and Hasegawa (unlearning, editing).
4. **Privileged-access tests.** Present: Song; Binder; Ashuach; Blandfort & Pawar; Singh.
5. **Machine introspection identification.** Present as conceptual criteria (Singh, Song, Lindsey, Comșa & Shanahan, Kammerer & Frankish). No formal identification theorem was found.
6. **Counterfactual self-knowledge.** This is the content of Lindsey's grounding criterion. Binder's behaviour-change experiment tests it behaviourally.
7. **Meta-models under capability change.** Present:
   - Binder (self-prediction after fine-tuning);
   - Blandfort & Pawar (fine-tuning changes the target);
   - continual calibration (aggregate).

   **I found no item-level, within-target analysis adjusted for pre-change information.**
8. **Causal tests of confidence and self-knowledge.** Present: Kumaran (steering confidence); Ferrara (sham-controlled detection).

## 3. Overall verdict

- **The formal identification argument: largely anticipated.** Observational equivalence between first-order (difficulty, familiarity, cue) accounts and self-monitoring is formalised in animal metacognition (Smith et al. 2016) and stated for LLMs (Singh et al. 2026). Writing it out for machine monitors is a **useful synthesis, not a new theorem.**
- **The estimand: partly anticipated.** Residualising a monitor on difficulty or shared information exists (Ackerman 2026; Blandfort & Pawar 2026), and the statistic is a standard partially-linear / partial-correlation DML construct. **Not found in this audit** is the specific combination:
  1. the outcome is *post-intervention* item competence among **identically treated items**, after an input-fixed intervention on the system's own competence;
  2. a symmetric, cross-fitted adjustment for **high-dimensional pre-intervention internal states** (θ_pre);
  3. a separate adjustment for measured generic change (θ_gen), with a pre-declared A/B/C label policy whose conservativeness (only A is affirmative) is justified by a documented finite-sample failure mode (D50);
  4. simulation validation against explicit rival data-generating processes (difficulty, susceptibility, regression to the mean, generic change, null, high-dimensional).
- **The engineering lessons (R1–R7) and the five documented calibration failures:** no equivalent found. They are an honest negative-results contribution about how hard it is to construct valid input-fixed competence changes in an artificial system.

**Recommendation (PI rule): reframe D1 as a methods / negative-results note, not a large-paper novelty claim.**

- **Working title:** "Measuring machine self-monitoring under input-fixed competence change: a within-target estimand, its validation, and five design failures."
- **Positioning:** a synthesis that brings the animal-metacognition identification logic (Hampton; Carruthers; Smith et al.) and the LLM introspection controls (Song; Binder; Singh; Ferrara; Ackerman) into one item-level estimand, with simulation validation and design requirements.
- **Venue tier:** a workshop or methods note (e.g. a methods track or a workshop on interpretability or metacognition). A main-venue paper would need an empirical Stage-1 result (D2 or D3).

**Permitted claims:** "We formalise …", "we combine …", "we validate by simulation …", "we document …".

**Not permitted:** "the first identification framework", "a new theory of machine introspection", or any claim implying the observational-equivalence insight is new.

## 4. Open verification items (before any D1 draft)

1. **Cohen & de Melo 2026 (ICML), the closest competitor per `decision_document_v2.md`.** It is still unread: OpenReview requires browser verification. **The PI needs to save the PDF to `research/literature/papers/`**, or authorise me to open it in the browser pane where they complete the check.
2. **Classic sources to verify** (marked `memory-verify` or partial):
   - Hampton 2001 (PNAS);
   - Carruthers 2008;
   - Koriat 1993 / 1997;
   - Fleming & Lau 2014;
   - Maniscalco & Lau 2012;
   - Hampton 2009 (converging-evidence review, not yet in the database).
3. **Read in full:** Hasegawa et al. 2025 (editing → underconfidence); Comșa & Shanahan 2025 (causal criterion); Kammerer & Frankish 2023.
4. **Re-run the audit** before any submission. The field moves monthly: several key items are from August–September 2026.
