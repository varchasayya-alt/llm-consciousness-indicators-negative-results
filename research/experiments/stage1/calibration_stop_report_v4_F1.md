# Calibration STOP report: v4 (Option B1) at V4-F1, kill condition K-2 (parametric backup too strong)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | **Methods ladder stopped under the pre-declared rule** (`calibration_plan.md` Revision v4; memo §19 K-2) |
| Not run | F1 confirmation on 9042/9043, F2–F4, P2, VAL. **Validation seeds 9051–9053 untouched.** No monitor trained; no confirmatory data; nothing frozen or registered |
| Pre-run commits | `b15be7a` (code, tests, config, plan); `2598e6f` (D49 ridge correction; verifications) |
| Raw results | `results/raw/calibration/calibration_results_v4.json`, `log_v4_*.txt`, `engineering_dryrun_v4_burned_seed.txt` |

## 1. Results

### V4-F0 (9041, p_rd = 0.35, 60 epochs): PASS

| Check | Result |
|---|---|
| Trained-fact accuracy | 1.00 |
| Unknown accuracy | 0.026 |
| Fluency AUROC | 1.00 |
| Own-slot retrieval attention (covered facts) | 0.988 |
| NULL attention for parametric-only facts | 0.991 |
| Slots | 4,636 |
| Training time | 8.8 min |

### V4-F1 (9041, p_rd ∈ {0.2, 0.35, 0.5}): K-2

Route A is all slots masked, measured on 1,145 memory-covered base-correct EV facts.

| p_rd | Route-A accuracy (gate [0.30, 0.70]) | IQR of route-A margin (≥ 1) | Route-A margin quantiles (5 / 25 / 50 / 75 / 95%) | M1 | M2 | Integrated (covered) | Parametric-only facts (integrated / A) | Route A by exposure (high / low) |
|---|---|---|---|---|---|---|---|---|
| 0.2 | **0.865** | 2.10 | −1.58 / 1.07 / 2.27 / 3.17 / 4.36 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 | 0.864 / 0.865 |
| 0.35 | **0.803** | 3.13 | −2.96 / 0.58 / 2.52 / 3.71 / 4.84 | 1.00 | 1.00 | 1.00 | 1.00 / 0.989 | 0.794 / 0.824 |
| 0.5 | **0.849** | 3.17 | −2.34 / 1.26 / 3.26 / 4.43 / 5.57 | 1.00 | 0.997 | 1.00 | 1.00 / 0.983 | 0.845 / 0.859 |

**Gate outcome.**

- Every p_rd fails only the A-range gate: the backup is too strong.
- Passing everywhere: store QC, memory sufficiency (M1 and M2 agree, with no ablation-artifact discrepancy), graded backup (IQR 2.1–3.2 nats), and parametric-only answerability.
- Per K-2, the ladder stops.
- The approved F3 gate (35–65% lost) would also have failed: about 80–87% of deleted targets would remain correct.

## 2. What this shows (store-only)

1. **The architecture works as designed.**
   - The memory is explicit, sharply addressed (own-slot attention about 0.99) and fully sufficient (M1 = M2 = 1.00).
   - Parametric-only facts route to NULL (0.99).
   - Backup competence is **graded and continuous**: route-A margins span about −3 to +5 nats.
   - Exposure barely matters (high vs low ≈ equal), so outcome would not be fixed by exposure.
2. **Backup strength is not controlled by route dropout.**
   - Route-A accuracy is 0.865 / 0.803 / 0.849 at p_rd 0.2 / 0.35 / 0.5, which is not monotone.
   - So the parametric weights learn covered facts mainly through ordinary **memory-present** presentations: the shared LM gradient keeps training the parametric route even when memory already supplies the answer.
   - Not through the route-dropped presentations.
3. **Engineering dry run** (burned seed, p_rd 0.35): route A was 0.92. Deletion of X slots was highly local (Z displacement ratios ≈ 0.001; retrieval leakage ≈ 1e-4), but nothing was lost.
   - This is consistent: locality is excellent, but with this backup strength the deletion leaves almost all targets answerable.
4. **Diagnostic note (M2 control).** The M2 readout on parametric-only facts reaches 0.42–0.51, far above chance (about 0.03), even though retrieval goes about 99% to NULL.
   - The retrieval query, built from block-2 states that already encode parametric knowledge, apparently steers the residual about 1% of retrieval mass toward same-answer slots.
   - So the memory contribution u carries some parametric answer information.
   - This does not affect K-2, but it matters for interpreting M2 and the bookkeeping features in the trivial-decoder audit.

## 3. Problems / deviations in this cycle (all logged)

| Code | Description |
|---|---|
| D47 | Retrieval trainability: query from block-2 residual at s1, s2, r; cosine retrieval; self-key initialisation. Found on a burned engineering seed before any dev seed |
| D49 | GCV ridge failed when p > n. Replaced by nested-CV ridge before any dev seed. Some v3 identifiability R² values were probably understated by the old ridge (the v3 conclusion is unaffected) |
| D50 | Open issue: in the high-dimensional simulation, diffuse state-visible susceptibility is partly unrecovered by the pre-state adjustment (θ ≈ 0.09; false label B in 25% of studies, never A). Statistic unchanged; needs a PI decision before freezing |
| B1 fake-data verification | Clean once the backup reader uses state-encoded information (run 1, with only a latent reader, is kept and explained) |

## 4. Options for the PI (decision needed; nothing implemented)

**O1 (recommended): gradient-isolated parametric learning for memory-available presentations ("v4.1").**

- Keep the architecture.
- During store training, a covered fact's presentation **with its slot available** updates only the memory route (keys, values, query/readout interface) and the integration path. Its gradient into the parametric weights is stopped.
- Covered facts are then learned parametrically **only** in route-dropped presentations. That makes p_rd a direct, monotone dose of parametric exposure, which is the lever this cycle lacked.
- Parametric-only facts are unaffected.
- **Requirements:** a new committed revision, the same F0–F4 gates, and **fresh** development seeds (proposed 9061–9063). Validation 9051–9053 stays untouched.
- **Cost:** about 1–2 h to F1.

**O2: reduce parametric capacity** (MLP width or d_model) so the weights cannot memorise all covered facts.

- Simpler, but it changes the approved store, also weakens parametric-only facts, and its effect on backup is less controllable.

**O3: larger world** (more facts per parameter). Same logic as O2. Cost: slower stores.

**Not proposed:** relaxing the A-range or diversity gates (binding per the PI), or reusing 9051–9053.

**If O1 also fails K-2, or later gates fail:** report that this synthetic store cannot provide the required partial-backup regime under the approved constraints, and reconsider the Stage-1 design.
