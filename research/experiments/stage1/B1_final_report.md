# B1 FINAL REPORT: the dual-route ("explicit memory + parametric backup") Stage-1 store. Line closed

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | **The B1 line is closed** under the PI's absolute stopping rule (D60). The final revision, v4.2, stopped at F2–F4 on its development seeds |
| Commits | v4 `b15be7a` / `2598e6f` / `2bcef69`; v4.1 `3539d3d` / `79b63d2`; v4.2 `b606db3` (pre-run), `8a5170f` (chain-script fix), `0dc1261` (results) |
| Raw results | `results/raw/calibration/calibration_results_v4.json`, `_v41.json`, `_v42.json`, plus `log_v4*_*.txt`, `diag_v41_*.json` and the engineering logs |
| Untouched | **Validation 9051–9053**, confirmatory 1001–1040 and retired-unused 9062/9063. No monitor or controller was ever trained on any B1 store. Nothing frozen, registered or claimed |

**Not run:** P2, FREEZE, VAL. P2 requires F0–F4 to pass first.

---

## 1. Exact stopping gate (v4.2, development seeds 9071–9073, selected p_rd = 0.20)

The runner's first failing ladder item is "V4-F2 locality (K-3)". Read precisely, there are **two independent failures**, each fatal under D60.

**(a) Y negative control, continuous retention (K-7), on all three seeds.**

- The F2 `retention` gate inherits the v3 definition, so it covers Y **and** the six Z categories. It fails *only* through Y.
- Y binary loss: 0.011 / 0.004 / 0.007 (passes ≤ 0.03).
- Y continuous change:

  | Seed | Y margin change (nats) | Ratio to X's change (gate ≤ 0.10) | Fraction of D_C (gate ≤ 0.10) |
  |---|---|---|---|
  | 9071 | −4.29 | 0.50 | 0.37 |
  | 9072 | −3.45 | 0.38 | 0.33 |
  | 9073 | −3.81 | 0.42 | 0.35 |

**(b) F3 outcome diversity (K-4), on all three seeds.**

| Seed | Retained-strong (gate ≥ 0.15) | Binding F6 (≥ 50 pairs and max SMD ≤ 0.10) | X lost (gate [0.35, 0.65]) |
|---|---|---|---|
| 9071 | **0.00** | **fails**: 62 pairs, max SMD 0.12 | 0.36 |
| 9072 | **0.00** | **fails**: 67 pairs, max SMD 0.16 | **0.68** |
| 9073 | **0.00** | **fails**: 74 pairs, max SMD 0.16 | 0.64 |

- F6′ (recorded only) is feasible on all three seeds.
- IQR(C_post) is 2.33–2.58 (passes), and exposure passes (R² ≤ 0, AUROC ≤ 0.53).

**Everything else at F2–F4 passed:**

- **Z locality** is essentially perfect in all six categories on all seeds:
  - Z lost 0.000;
  - retention load 0.00 (|Δmargin| ≤ 1e-4 nats);
  - read-site displacement ratio ≤ 1.4e-4;
  - retrieval leakage ≤ 1.5e-4;
  - name-fluency change 0.0 SD.
- **F4:**
  - medians R²_gen 0.42, R²_pre −0.003, R²_joint 0.0005;
  - bookkeeping R²(C_post | memory metadata) 0.24–0.28, against a kill threshold of 0.90.

## 2. Failure mechanism

**F3 (structural; not a noise or near-miss failure).**

- In v4.2 the explicit memory answers every covered fact with high confidence, whatever its parametric status. Integrated margins are 8.4–9.4 nats in all Stage-A tiers (F0B, 9071).
- The "retained-strong" tier (C_post ≥ 0.5 · C_pre) therefore needs a *parametric* backup margin of about 4.5 nats or more.
- The developmental dose that gives a partially competent backup (Route-A accuracy in [0.30, 0.70], as F1 requires) produces backup margins whose 95th percentile is 3.0 nats (p_rd 0.20).
- Backup margins of about 4.5 appear only at p_rd ≥ 0.35. There Route-A accuracy is 0.96–0.999, outside the redundancy window.

| p_rd (9071 Stage A) | Route-A accuracy | Route-A margin quantiles (5 / 25 / 50 / 75 / 95%) |
|---|---|---|
| 0.20 | 0.642 | −2.8 / −0.7 / 0.8 / 1.8 / 3.0 |
| 0.35 | 0.960 | 0.3 / 2.0 / 2.75 / 3.4 / 4.5 |
| 0.50 | 0.999 | 2.3 / 3.1 / 3.7 / 4.3 / 5.1 |

So in this store family the F1 redundancy gate and the F3 tier gate are **jointly unattainable**. The tier is defined relative to the intact (integrated) margin, and a strong explicit memory makes that margin about 3× the largest backup margins compatible with partial backup.

**Y (K-7).**

- The same-answer donor-value transplant keeps the *answer* (lost ≤ 1.1%) but not its *strength*.
- In Stage B each slot value is optimised, through the frozen downstream, *together with its own fact's pre-injection state*. A donor value written into a different entity's context is a weaker write.
- The likely interpretation: in this architecture a value is a context-specific code, not a portable "answer token". This is not tested further.
- Y therefore changes graded competence by about 40–50% of X's change, so it is not a competence-neutral, matched-magnitude edit.

**F6.** The binding balance criterion fails by small margins (max SMD 0.12–0.16, against 0.10), while the noise-calibrated F6′ passes. This is the same noise-limited pattern documented in D43.

**Not the cause:**
- locality (perfect);
- identifiability (good);
- dose control (F0A passes; dose-response coherent);
- integration (F0B and F1C pass);
- training stability.

## 3. Summary of v4, v4.1 and v4.2

| | v4 (D45–D51) | v4.1 (D52–D56) | v4.2 (D57–D60) |
|---|---|---|---|
| Training rule | Memory and parametric routes co-trained, shared gradients; route dropout p_rd | Simultaneous, gradient-isolated: memory-present examples train memory only | **Sequential**: Stage A parametric alone with dose p_rd; Stage B memory only on the frozen network (zero NULL, bounded injection) |
| Seeds used | 9041 (F0, F1 grid) | 9061 (F0 only) | 9071 (F0A grid, F0B), 9072/9073 (F1C), 9071–9073 (F2–F4) |
| Store sanity (F0) | Pass (trained 1.00, own-slot 0.988) | **Fail (K-1)**: integrated store unusable (0.70 after C1; covered 0.58 < 0.987 weights-only); scales ran away 200–1,000× | Pass: Stage A param-only 1.00; F0B trained 1.000, own-slot 0.998 |
| Backup control (F1) | **Fail (K-2)**: Route A 0.80–0.87 at every p_rd, non-monotone; backup learned through memory-present presentations | Not reached | **Pass**: coherent dose-response (0.20 → 0.16 → 0.64 → 0.96 → 1.00; DC-1, DC-2); 0.642 / 0.300 / 0.394 on 9071 / 9072 / 9073 |
| Memory integration | Pass (M1 = M2 = 1.00) | Fail (memory carried the answer, M2 0.98, but was not decoded) | **Pass**: rescue 1.00, no damage 1.00, INT-1 ρ 0.49–0.67, ‖u‖/‖hA‖ 0.9–1.15, route-A bit-identical |
| Locality (F2 Z) | Not reached (engineering dry run: very local) | Not reached | **Pass, essentially perfect** |
| Y control | Not reached | Not reached | **Fail (K-7)**: graded change 0.38–0.50 × X |
| Diversity (F3) | Not reached (would have failed: about 80–87% retained) | Not reached | **Fail (K-4)**: retained-strong 0%; F6 binding fails; 9072 lost 0.68 |
| Identifiability (F4) | Not reached | Not reached | Pass (R²_pre ≈ 0, R²_gen 0.42, bookkeeping 0.25) |
| Report | `calibration_stop_report_v4_F1.md` | `calibration_stop_report_v41_F0.md` | this report |

**Process issues, all logged.**

- **v4.2 chain script:** a variable-scoping bug crashed the chain after the Stage-A stores were trained and before any evaluation. It was fixed, committed (`8a5170f`) and resumed from cached stores, with no result affected.
- **9072:** passed the Route-A ≥ 0.30 gate by a single fact (346/1152 = 0.3003). It is reported as observed.

## 4. What the three B1 failures teach us

1. **Each component worked on its own; the joint specification did not.** v4.2 achieved:
   - controllable developmental dose;
   - sharp addressing;
   - full memory sufficiency, with rescue and no damage;
   - perfect locality;
   - no exposure confound;
   - good identifiability;
   - near-zero pre-state predictability of the backup (R²_pre ≈ 0).

   The failure sits in the *conjunction* of a strong-redundancy requirement with an outcome-diversity tier defined *relative to intact competence*, and in the negative control.
2. **Co-trained routes share what they learn** (v4), and **separately trained routes do not learn to talk to each other** (v4.1). Only a frozen, stationary readout (v4.2) gives a separable, controllable redundant system. Redundancy and separability pull against each other unless development is staged.
3. **Margin scales are set by the strongest route.** In any redundant store the intact margin reflects the dominant route, so "how much of the intact competence survives" is mostly a statement about the dominant route's confidence, not about the backup. Gates that mix the two scales can be unattainable by construction. This should have been checked analytically when the tiers were carried over from v3 (single-route) to v4 (two routes); it was not.
4. **"Matched but competence-neutral" controls are fragile, and each architecture breaks them differently:**
   - v2 sham: procedure fingerprint;
   - v4.2 Y: graded change without binary change.

   A control defined by a *binary* invariance (still correct) does not guarantee *graded* invariance, and the approved gates were graded.
5. **Five calibration cycles (v2, v3, v4, v4.1, v4.2) all failed at the engineering of the intervention, never at the statistics.** The within-target estimand, its simulation validation and the identifiability machinery behaved as designed throughout. The scientific bottleneck of Stage 1 is producing a valid, controllable, input-invisible competence change in a synthetic system, not measuring monitoring once such a change exists.
6. **Pre-declared, binding gates worked as intended.** They stopped every variant before any monitor existed. There is no outcome-dependent analysis to unwind, and the validation and confirmatory seeds are pristine.

## 5. What remains reusable

**Statistical and measurement machinery (N1 core):**

- **`s1/estimands.py`:**
  - the within-target estimand θ_pre (`theta_post_dml`: residualised-post association, symmetric cross-fitted pre-state adjustment, RCS baselines);
  - θ_gen;
  - nested-CV ridge (D49), cross-fitting, propensity matching and balance.
- **`s1/analysis.py`:** families P, G, K, F and S; Holm; seed-level t-tests with MEI; identifiability checks; A/B/C labels and the **D54 inferential status** (only A is affirmative).
- **`s1/sim_worlds.py`, `sim_statistic_comparison.py`, `sim_b1_check.py`, `sim_highdim_check.py`:** the simulation-validation suite. It covers the H3, H2, susceptibility, regression-to-the-mean, null, B1-like and high-dimensional worlds, and documents the finite-sample limit (D50).
- **`s1/interventions.py` helpers:**
  - `retention_continuous`, `generic40`, `identifiability`, `natural_gap`, item pre-covariates;
  - `v4qc.audit` (trivial-decoder audit: R²_pre / book / generic / output / full and incremental terms);
  - DC-1 / DC-2 dose-coherence checks.
- **Process infrastructure:**
  - seed namespaces and their tests;
  - the monitor-input prohibition and AST import guards;
  - pre-declared calibration plans committed before runs;
  - the FREEZE/VAL guards;
  - the claim boundaries (Level 1/2/3).

**Systems** (reusable as objects; *not* proposed for further B1 engineering):

- the v1–v3 parametric store and synthetic world;
- the v4.2 sequential dual store, with exact P/M partition, Stage A/B training and bit-identical route-A equivalence;
- deletion and transplant operators.

The v4.2 store is a validated object with known properties (sections 1–3). Any reuse would need a new, separately approved preregistered question. It is not a continuation of B1.

**Not reusable as is:** the B1 intervention specification (T-DELETE plus Y plus the relative-tier diversity gate) as a Stage-1 primary design.

## 6. Claims

**No Stage-1 scientific claim is made.** No monitor was trained, and B1 produced no evidence for or against competence monitoring.

The B1 results are **Level-1 methods results** about synthetic stores:

- under shared training, explicit-memory redundancy produces uncontrollable backup;
- under simultaneous isolated training, the routes do not co-adapt;
- under sequential development, redundancy, control, locality and identifiability are achievable, but a strong explicit memory makes relative outcome tiers unattainable within the redundancy window, and value transplants are not graded-neutral.

The pivot options are in `research/memo/stage1_pivot_memo.md`. No new implementation has been done.
