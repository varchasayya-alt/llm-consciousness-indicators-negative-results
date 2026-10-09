# Open questions

## Conceptual

1. **Self-directed vs world-directed.** Is a feeling-of-knowing that reads *partial retrieval* (accessibility) self-monitoring in HOT's sense, or "first-order readout"? Where exactly is the line between E3 and E4?
2. **Familiarity.** Is entity "familiarity" a representation about the *world* (how common the entity is) or about the *self* (whether *I* have stored it)? Proposed test: they come apart when the model is trained on a rare entity, or when an entity is unlearned. Does the familiarity signal follow the model's knowledge or the entity's frequency?
3. **Recurrence vs unfolding.** Can any functional signature distinguish re-entrant structure from an unfolded equivalent? Given the unfolding argument, what is the strongest *non-structural* claim about recurrence that P3 can make?
4. **Feedforward access.** If transformers implement broadcast, selectivity and capacity limits without recurrence, does this count against GNW's dynamical claims, or only against its implementation claims?
5. **Scope of HOT.** Do higher-order theories about *perceptual* states extend to self-monitoring of *memory*? (Stage 4 VLM "blindsight" addresses this.)

## Methodological

6. **Trial-to-trial variability.** What is the right model of noise in a deterministic network for bimodality tests (paraphrase, activation noise, sampling)? Results must be robust across all three.
7. **Lesion selectivity.** How selective can input-preserving recall lesions be in 7–9B models? Attention knock-out vs per-item patching vs unlearning.
8. **Probe confound.** How to stop probe-based readouts from exploiting answer identity [jspace_gemma_repo]? Candidate: train probes only on items whose answers are balanced across classes; test on swapped-answer items.
9. **Readout format.** Verbal FOK in instruction-tuned models may be dominated by post-training priors (e.g., always "Yes"). What readout format minimises response bias? Forced choice? Comparative judgements ("Which of these two questions are you more likely to answer correctly?")?

## Empirical (to resolve by reading)

10. What exactly did Cohen & de Melo (ICML 2026) do in the "computationally degraded systems" condition? Item-level? Input-preserving?
11. Does Park et al. (2026) test generalisation to *removed* (not only newly acquired) knowledge?
12. Which open models have released J-lens matrices, and which layers form the workspace in them?
13. Are Ouro/Huginn usable at untrained loop counts without collapse (needed for H6)?

## Added v0.2 (2026-10-01)

14. **Cohen & de Melo, full text.** Did they use any item-selective intervention, sham or input-corruption controls, internal analyses, control endpoints, or a synthetic system? See the checklist in memo §10.2.
15. **N1, the identification theorem.** What is the weakest sufficiency condition on a familiarity statistic F under which E1 and E2 cannot separate self-tracking from world-tracking monitors? Can it be stated for type-2 AUROC and meta-d′ simultaneously?
16. **Lesion artefacts.** In P5\*, can a monitor detect lesions through generic activation-statistics shifts (norm, entropy) rather than recall content? Design norm-matched shams, and lesions that shift norms without affecting recall.
17. **Re-entry vs compute.** Is "re-entry" (re-embedding the store's own answer) distinguishable from extra compute? Planned controls: FLOP-matched single pass; second pass without the answer.
18. **Bridge validity.** At what minimal pretrained-LM scale (0.5B? 1.5B?) do lookup and abstention behaviours discriminate natural known vs unknown items well enough (AUROC ≥ 0.6) for P1\* to be informative?
19. **Developmental checkpoints.** Do public training checkpoints (e.g., Pythia) let us observe the familiarity-shortcut regime emerging during pretraining, at zero training cost?
