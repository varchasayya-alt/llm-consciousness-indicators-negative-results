# D1 methods note: organisation of existing N1 material (NOT a manuscript)

| Field | Value |
|---|---|
| Status | Organisation only, per the PI (2026-10-03): "do NOT yet write a polished final manuscript" |
| Framing | Fixed by the novelty audit (`memo/d1_novelty_audit.md`): a **methods / negative-results note**, a synthesis plus a validated estimand plus documented design failures. Not a new identification theory |
| Firewall | Decisions about this note may not influence D2 thresholds or experimental choices. D2 gates were frozen in `7742d07` before this file existed |

**Guiding question:** how can we distinguish a system that monitors its own competence from one that merely predicts task difficulty, familiarity, intervention artifacts or generic internal change?

## Section plan (the PI's 11 elements, mapped to existing material)

| § | Element | Content | Existing material | Gap / to do |
|---|---|---|---|---|
| 1 | Identification problem | Monitors M of a system S; "self-monitoring" defined as counterfactual dependence on S's current competence. Rivals: difficulty / world tracking, familiarity, intervention artifact, generic change. **Credit:** Hampton; Carruthers; Smith et al. 2016; Singh et al. 2026; Song et al.; Lindsey (grounding) | `memo/decision_document_v2.md` (N1); `experiments/stage1/rival_hypotheses.md` | Write the formal setup (notation) |
| 2 | Observational equivalence | Under natural data, a sufficient familiarity or difficulty statistic makes self- and world-tracking monitors produce the same (input, report, correctness) distribution. Present as a short proposition **explicitly acknowledged as a restatement** of Smith et al. 2016 (animals) and Singh et al. 2026 (LLMs) | N1 text in the decision document | Proposition plus proof sketch; cite precedents prominently |
| 3 | Why input-fixed competence change helps | do(S → S′) with identical inputs separates the rivals. Analogues: Hampton's no-sample probes; Binder's behaviour change; Ferrara's sham-controlled interventions | `protocol_v3_PROPOSED.md`; `claim_boundaries.md` | Table of rival predictions under the intervention |
| 4 | Within-target identification principle | Among **identically treated** items, does the monitor's post-change output track post-change competence beyond pre-change information? It removes treatment-assignment artifacts (no X-vs-Y contrast) | `v3_design_memo.md`; `protocol_v3_PROPOSED.md` | Diagram; assumptions (no post-treatment adjustment, overlap, positivity of residual variation) |
| 5 | θ_pre / θ_gen | θ_pre: residualised-post association, RCS baselines and symmetric cross-fitted ridge pre-state step (nested-CV λ, D49). θ_gen adds 40 measured generic-change features. A/B/C labels | `s1/estimands.py`; `s1/analysis.py`; D36–D43, D49 | Position as standard partially-linear / DML partial correlation (Chernozhukov et al.); relate to Ackerman 2026 and Blandfort & Pawar 2026 |
| 6 | Simulation validation | Seven worlds (H3, H3 strong, H3 + generic, H2 generic, H2 intensity, susceptibility, RTM, null), the B1-like world and the high-dimensional world: support rates, false-label rates and θ distributions | `results/statistics/v3_statistic_simulation/*`; `v4_b1_simulation.md`; `sim_highdim_results.md` | One consolidated results table and figure |
| 7 | Finite-sample susceptibility limits | D50: diffuse susceptibility spread over 1,280 dimensions with n = 300 → θ ≈ 0.09, false B in 25% of studies, never false A. Report p/n and cross-fitted R² | `sim_highdim_check.py`; D50 | Sensitivity curve over p/n (simulation only; optional) |
| 8 | Label-A-only rule | D54: only A is affirmative; B is ambiguous; C is no evidence. Justified by §7 | D54; `claim_boundaries.md` | — |
| 9 | Design requirements R1–R7 | Input identity; graded within-population variation; identifiability; graded-neutral controls; replication unit; analytically satisfiable gates; minimal intervention engineering | `memo/stage1_pivot_memo.md` §0; `B1_final_report.md` §4 | Expand each with a failure example |
| 10 | Five calibration failures as case studies | Each with a pre-declared gate, the observed failure and the lesson: **v2** procedure fingerprint; **v3** tail flattening and compressed endpoints; **v4** uncontrollable redundancy; **v4.1** co-adaptation failure; **v4.2** relative-tier incompatibility and graded-non-neutral transplant control | The five stop reports and `B1_final_report.md` | Uniform case-study template (design → gate → result → mechanism → lesson) |
| 11 | Preregistration / checklist | Gates committed before data; seed namespaces; monitor-input prohibitions; satisfiability check; label policy; FREEZE/VAL guards | `logs/calibration_plan.md`; `d2_S0_satisfiability_note.md`; tests | Turn into a one-page checklist (appendix) |

## Claims discipline

- **Level 1 only.** No consciousness, awareness or self-awareness claims.
- **Novelty claims are limited to the combination** in §5–§7 and §9–§11 (see the audit verdict).
- The observational-equivalence point is credited to its precedents.

## Dependencies before drafting

1. Cohen & de Melo full text (needs the PI).
2. Verify the classic references.
3. Decide whether a D2 result (if D2 passes and a monitor study is later approved) belongs in this note or in a separate empirical paper. **This decision must not affect D2's design.**
