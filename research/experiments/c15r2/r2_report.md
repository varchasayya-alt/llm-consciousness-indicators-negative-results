# C15-R2 report: G_select2 classified **P4 (assay failure)**. STOP.

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Decision | **R2-CLASSIFY = P4 → STOP** (frozen STOP/GO table, prereg §11). No FREEZE2. **G_confirm2 was never loaded (still sealed).** No architectural claim |
| Plan | `memo/c15r2_preregistration_FROZEN.md` (§1–12) and `memo/c15r2_thresholds_FROZEN.json` (D73, D74) |
| Code / pre-run commit | `experiments/c15r2/` (package `c15a2`); pre-run commit **02201df** (D75); engineering fixes f415bb1 (D75a) and 86f506a (D75b), both before any G data |
| Results | `results/raw/c15r2/`: `select2_*.json`, `classify2.json`, `r2_tables.md`, `verdict2.json` (commit d17691b) |
| Claims discipline | Functional organisation only. No outcome bears on phenomenal consciousness. Under P4 there is no architectural inference at all |

## 1. Verdict

The frozen classifier returns **P4**. Both P4 criteria hold independently:
- W3′ is assessable at **no** layer of **any** model;
- **0 of 15** cells are assay-valid, against the required ≥ 50%.

By the frozen table, the line stops here: no freeze, no confirmation, report to the PI. The default proposal is a methods note.

## 2. Why P4 (each item suffices for some cells; together they cover all 15)

| Failure (frozen rule) | Where | Values |
|---|---|---|
| **W3′ matched-⊥ coverage** (≤ 1 of 8 W3′ concepts may lack an empirical-KL-matched ⊥) | **13 / 15 cells** | 2–8 of 8 concepts unmatched: 1.7B L8–14; 3.5-2B L7, L11, L13; 4B all six layers |
| **W3′ PC1 reference reliability** (median split-half cos ≥ 0.5 at ≥ 50% of sites) | Fails at **every** cell where it was computed (all 15, via the main block or the diagnostics) | Fraction of sites 0.00–0.29; mean median cos 0.23–0.43 |
| **W1/W2 matching coverage** (unmatched ≤ 20%; required for an assay-valid cell) | **15 / 15 cells** fail at least one | W1 unmatched 0.10–0.90; W2 unmatched 0.20–1.00 |
| **W0a** (lens agreement ≥ 0.60) | Qwen3.5-2B | 0.539 [0.520, 0.557], a powered failure. Qwen3-1.7B passes (0.675); Qwen3-4B passes (0.712) |
| **Two-hop accuracy ≥ 0.60** (required for W0b′; W4 assessability) | All three models | 0.47 (1.7B), 0.36 (3.5-2B), 0.57 (4B). So W0b′ cannot pass and no W4 failure can be powered anywhere |

The W3′ controls that do not depend on ⊥ coverage behave as designed in every cell:
- PC2 natural-identity transport passes at 0.88–1.00 of sites;
- PC3 injection-site identity passes;
- NC1 random-direction sham: BB 0.00–0.02;
- NC2 label permutation: BB 0.00.

The W3′ pipeline therefore runs and its shams are clean. The validity failures are coverage and reference reliability.

**Pre-registered hypotheses.**
- **H-F1** (Qwen3.5-2B: A passes, W5 has a powered failure): **not assessable**, because W0a fails again.
- **H-F2** (Qwen3-4B: B at L11/13, A has a powered failure): **not assessable**, because two-hop accuracy is 0.57 < 0.60, so W4 cannot be evaluated.

## 3. Gate table (G_select2; from `results/raw/c15r2/r2_tables.md`)

| Model | Layer | a\* | W1 hit J/⊥ | W2 rate J/⊥ (n) | W3′ | W4 diff pp [CI] | W5 SI_iso [CI] | Gates passed |
|---|---|---|---|---|---|---|---|---|
| 1.7B | 8 / 10 / 12 / 14 / 16 | 0.5 / 0.5 / 0.5 / 1 / 2 | 0.07–0.22 / 0.05–0.22 | 0.00–0.31 / 0.00 (0–13) | NA | 4.1–8.1 (not assessable) | 1.48–2.10, all CIs > 1 | W0a only |
| 3.5-2B | 7 / 9 / 11 / 13 | 1 / 2 / 2 / 2 | 0.27–0.39 / 0.11–0.20 | 1.00 / 0.00 (3–6) | NA | 2.7–5.4 (not assessable) | 1.95–2.17, all CIs > 1 | W1 at L9 only (W0a fails) |
| 4B | 11 / 13 / 15 / 17 / 19 / 21 | all 0.5 | 0.00–0.27 / 0.00–0.27 | 0.00–0.10 / 0.00 (6–16) | NA | 0.0–13.5 (not assessable) | 0.99–2.16 | W0a; W5 at L15 (0.99 [0.91, 1.08]) |

- W0b′ fails in all models, at the primary position t_d and on two-hop accuracy.
- W2 counts are small (n ≤ 16) and its matching coverage is poor, so W2 is uninformative.
- None of these gate values is evidence under P4.

## 4. Diagnostics (descriptive only; used for no decision; not evidence)

1. **Material sensitivity of the unchanged dose rule and matching.**
   - **Unchanged construction.** S_J rank, median norm ĥ and J-fraction of the concept vectors are essentially identical to the A-stage. For example, 1.7B r = 51–157 vs 49–156, and ĥ is within 1–2%.
   - **More disruptive J injections.** At the same dose they are 3–6× more disruptive than in the A-stage. For example, 1.7B L8 at a = 1: generic KL 0.085 vs 0.015.
   - **Lower a\*.** The KL ≤ 0.05 limit therefore binds at a = 1, and a\* fell from mostly 2.0 to 0.5.
   - **Coverage collapse.** The ⊥ candidate pool's empirical-KL ratio (⊥/J) has a median of 0.4–0.5, rarely inside the ×/÷1.15 window. Coverage fell from 85–95% (A-stage) to 10–65%.
   - **Same pattern on smoke.** It was already visible on SMOKE2 (D75a).
2. **PC1 / PC2 dissociation.**
   - References from single paraphrases have low mutual cosine (0.23–0.43).
   - Yet the pooled reference identifies a held-out paraphrase at accuracy 0.67–0.81, and PC2 passes almost everywhere.
   - PC1 as specified measures the paraphrase invariance of raw mean states. That is apparently stricter than what identification needs. **It is not reinterpreted here: P4 stands.**
3. **Intermediate-entity readout by position** (W0b′ secondary positions are report-only):

   | Position | Intermediate rate | Foil rate |
   |---|---|---|
   | t₁ | 0.64–0.81 | ≤ 0.05 |
   | t_d | 0.31–0.52 | — |
   | t₂ | 0.09–0.24 | — |

   For comparison, the A-stage final-token rates were 0.02–0.17.
4. **The two cells where W3′ was computed** (not assessable: PC1 fails, and W0a fails for 3.5-2B):
   - 3.5-2B L9: BB_J 0.99 vs BB_⊥ 0.19. W3b recall J 0.82 / ⊥ 0.23 / none 0.01 / natural 1.00.
   - 1.7B L16: BB_J 0.30 vs BB_⊥ 0.55.
   - **Not evidence.**
5. **Fresh two-hop items were harder.** Accuracy was 0.47 / 0.36 / 0.57, against 0.58 / 0.62 / 0.70 in the A-stage. This is the main reason W4 and W0b′ were unassessable.

## 5. Integrity and deviations

**Provenance.**
- G_confirm2 was never loaded: no FREEZE2 and no `confirm2_*` file exist.
- The original A-stage G_confirm remains sealed.
- The A-stage provenance pins (trees `31f6d6e7…` and `54a3f123…`, and all pinned files) were verified at R2 start, at every SELECT, and at REPORT (`verdict2.json`).
- No threshold, rule or material was changed after any model saw R2 material.

**Deviations, all recorded:**
- **D75a and D75b:** engineering fixes to the SMOKE2 check wiring, made before any G data. They change no gate.
- **D76:** a Windows reboot at 22:00 killed the first Qwen3-4B SELECT. It was rerun identically. Layer 11's logged gate flags matched the interrupted run (the interrupted run wrote no result file).

**CPU.** About 27 h of SELECT (5.9 + 6.4 + 12.1 h), plus about 2.5 h of ENG, plus about 4 h lost to the reboot. No downloads.

## 6. Interpretation and recommendation

**Interpretation.** No architectural claim. Two pre-registered attempts (A-stage ST-2; R2 P4) have now failed to produce a valid lens-defined workspace assay on the three CPU-feasible candidates. The failing component moves with the material:
- in the A-stage: W3 non-informative, W0b, W0a(3.5-2B);
- in R2: matched-control coverage, PC1, two-hop accuracy, W0a(3.5-2B).

**Significance confidence for C15 as a workspace paper on these models: LOW.**

**Recommendation (PI decision; nothing is started):**
- **A (default per the frozen table):** a methods / negative-results note covering the A-stage and R2. Contents:
  - the null diagnosis of the A-stage W3;
  - the identity-specific W3′ design with its complete-gate null calibration;
  - the material sensitivity of norm- and KL-matched controls;
  - the PC1/PC2 dissociation;
  - the t₁ > t_d > t₂ intermediate-readout profile;
  - the integrity apparatus.
- **B (not recommended):** a design-only R3 that repairs control coverage and PC1. Each round spends fresh material, and repairing after observing failures raises forking-path risk.
- **C16 is not triggered:** it required a confirmed P2.
