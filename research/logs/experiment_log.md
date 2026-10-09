# Experiment log

One entry per run, including failed and aborted runs. Template:

```
## <date> · <pilot/experiment id> · <run id>
- Prereg file + SHA-256:
- Code commit / config hash:
- Model + revision:
- Hardware / provider / GPU-hours:
- Conditions run:
- Deviations from preregistration (and why):
- Raw output path:
- Outcome summary (no interpretation beyond prereg criteria):
- Problems / anomalies:
```

---

*No experiments have been run yet (Phase 0, 2026-10-01).*

## 2026-10-01 · FEASIBILITY (not an experiment) · cpu_benchmark
- Prereg: n/a (no hypothesis examined).
- Code: `research/tools/cpu_feasibility_bench.py`.
- Model: Qwen/Qwen2.5-0.5B-Instruct (HF), fp32 and bf16; tiny random-token transformers.
- Hardware: Intel Core Ultra 7 255U, 12 threads, CPU only, torch 2.14.1+cpu.
- Raw output: `results/raw/feasibility/cpu_benchmark.json`.
- Outcome:
  - 0.5B fp32: 0.49 s/prompt (b1), 0.34 s/prompt (b8), 64 tokens;
  - bf16: 2.0–2.2 s/prompt;
  - tiny transformer training 11.5k / 3.0k / 1.3k tokens/s at 0.9M / 4.2M / 12M params.

## 2026-10-01 · FEASIBILITY (not an experiment) · store_training_feasibility
- Code: `research/experiments/prototypes/store_training_feasibility.py`. Throwaway world seed 12345 (burned; never reuse).
- Outcome: 4,000 facts, d=128, 3 layers. Train-fact accuracy 0.16 / 0.71 / 0.977 at 10 / 20 / 30 epochs (0.34 min total).
- No monitor, lesion or confidence analysis performed.

## 2026-10-01/02 · STAGE-1 CALIBRATION (methods only) · revision v2 chain

| Field | Value |
|---|---|
| Prereg | Not frozen (calibration phase). Rules: `logs/calibration_plan.md` Revision v2; code `f1e8efb` (+ reporting-only pipeline changes `ead3fab`) |
| Hardware | Intel Core Ultra 7 255U, CPU only, 12 threads, torch 2.14.1+cpu |
| Seeds | Dev 9001–9003; validation 9011–9013 (now retired) |
| Monitors / controllers trained | **None** (unit-tested guard) |

**Phases:**

| Phase | Time | Outcome |
|---|---|---|
| C3r | 22:24–22:25 | Rate max 0.6 |
| C4v2 | 22:25–03:09 | 27 cells × 3 seeds. Selected lr 5e-4 / 100 steps / λ_R 100 / γ 10 |
| C2val | 03:09–03:30 | 3 stores, QC pass |
| C4val | 03:30–03:33 | **FAIL** on the fingerprint gate: G2 0.903 on 9012 |

**Diagnostics:** `diag_fingerprint.py` on 9001 and 9002 (lr 1e-4 / 200 / λ_R 1) and on 9001 and 9012 (selected config); `diag_intact_margins_dev.json`.

**Deviations:** none from the pre-declared rules.

**Anomaly:** per-seed run time doubled while diagnostic and test processes ran concurrently (CPU contention only; results deterministic per seed).

**Raw output:** `results/raw/calibration/`.

## 2026-10-02 · STAGE-1 CALIBRATION v3 (methods only)

| Field | Value |
|---|---|
| Code | `119a3c5` |
| Rules | `calibration_plan.md` Revision v3 |
| Seeds | Dev 9031–9033 (validation 9021–9023 not used) |
| Monitors trained | None |

**Phases:**

| Phase | Time | Outcome |
|---|---|---|
| C2dev | 18:31–18:59 | Stores pass QC |
| C4dev | 18:59–22:19 | 36 cells × 3 seeds; **0 eligible → STOP (exit 3)** |

**Deviations:** none.

**Raw output:** `results/raw/calibration/` (`log_v3_*`, `calibration_results.json` key `v3`).

## 2026-10-03 · STAGE-1 v4 (B1) FEASIBILITY (methods only)

| Field | Value |
|---|---|
| Code | `b15be7a` + `2598e6f` |
| Rules | `calibration_plan.md` Revision v4 |
| Seeds | 9041 only (9042–9043 not reached; 9051–9053 untouched) |
| Monitors trained | None |

**Phases:**

| Phase | Time | Outcome |
|---|---|---|
| V4-F0 | 03:44–03:53 | PASS |
| V4-F1 | 03:53–04:12 | 9041 × p_rd {0.2, 0.35, 0.5} → **K-2 STOP** (route-A accuracy 0.803–0.865 > 0.70) |

**Raw output:** `results/raw/calibration/calibration_results_v4.json`, `log_v4_*`, `engineering_dryrun_v4_burned_seed.txt`.

## 2026-10-03 · v4.1 pre-run (code, tests, config, plan; burned seed 12345 only)

- **Implemented** the gradient-isolated learning routes (D52), the fixed p_rd grid and the dose-coherence rules (D53), the D54 label status and the D55 M2 note.
- **Tests:** `tests/test_v41_isolation.py`, 11 tests, all exact-zero / bit-identical invariants. Full suite: see the commit.
- **Burned-seed engineering** (`results/raw/calibration/engineering_dryrun_v41_burned_seed.txt`):
  - code path and timing: about 10 min per 60-epoch store at 12 threads;
  - the P route is equivalent to v4 when no row is memory-present;
  - the memory route learns more slowly early under isolation (expected from the routing; F0 decides);
  - F2–F4 and P2 plumbing OK.
- **No development or validation seed has been touched.**

## 2026-10-03 · V4.1-F0 on 9061: K-1 STOP

- **Training (p_rd 0.35):**
  - 60 epochs → 0.163 trained-fact accuracy;
  - C1 200 epochs → 0.701 (non-monotone; peak 0.918 at epoch 110);
  - E = 200 retrain identical → QC fail.
- **Diagnostics:** `results/raw/calibration/diag_v41_9061_prd0.35_{E60,C1E200}.json`, with the v4 reference `diag_v4_9041_prd0.35_E60_reference.json`.
- **Report:** `experiments/stage1/calibration_stop_report_v41_F0.md`.
- **Nothing beyond F0 was run.**

## 2026-10-03 · v4.2 pre-run (code, tests, config, plan; burned seed 12345 only)

- **Implemented:**
  - Stage A / Stage B training (`memstore.train_stage_A`, `train_stage_B`), the fixed-zero NULL value and the injection cap;
  - the F0A and F0B reports (`v4qc.stageA_report`, `stageB_report`);
  - runner `run_calibration_v42.py` and chain `_calib_chain_v42.sh`.
- **Tests:** `tests/test_v42_sequential.py` (11 tests: Stage A, Stage B and cross-stage invariants), plus the updated guard and seed tests.
- **Burned-seed engineering:** `results/raw/calibration/engineering_dryrun_v42_burned_seed.txt`.
- No development or validation seed touched.

## 2026-10-03 · V4.2 ladder (9071–9073): F0A, F0B and F1C pass; F2–F4 STOP; B1 closed

- **F0A** (9071, Stage A, 5 p_rd values): Route A 0.196 / 0.163 / **0.642** / 0.960 / 0.999. Presentations match expectation. Parametric-only accuracy 1.00. Selected 0.20.
- **F0B** (9071): all gates pass.
- **F1C** (9072, 9073): pass.
- **F2–F4** (9071–9073): Z locality and F4 pass; Y continuous retention (K-7) and F3 diversity (K-4) fail on every seed.
- **Not run:** P2.
- **Engineering:** a chain-script variable bug crashed after Stage-A training; fixed (`8a5170f`) and resumed from the cached stores.
- **Results:** `results/raw/calibration/calibration_results_v42.json`, `log_v42_*.txt`.
- **Reports:** `experiments/stage1/B1_final_report.md`, `memo/stage1_pivot_memo.md`.

## 2026-10-03 · D2 pre-run package (no D2 data)

- **Written:** protocol `experiments/stage1/d2_protocol.md`; S0 note `d2_S0_satisfiability_note.md`.
- **Code:** `s1/d2.py`, `run_d2.py`, `_d2_chain.sh`; config 0.8-D2; tests `tests/test_d2.py`.
- **Engineering dry run:** burned seed 12345 only (`_eng_dryrun_d2.py`).
- **Intact-margin facts for S0** come from retired stores 9031–9033 (no interference run on them).

## 2026-10-03 · D2 store-only kill test (9101–9103): STOP at G1

- **S1:** base QC passed (trained 1.00, unknown ≈ 0.03, fluency 1.00; populations 1,657 / 1,688 / 1,639).
- **S2:** interference reached the stop rule in 250–400 steps. CTRL was yoked.
- **S3:** lost 0.73–0.96 at every lr → G1 fails everywhere; G2–G5 pass.
- **SELECT:** no eligible lr → STOP. The SELECT step was run by hand after the chain parent hit the 2-hour tool cap.
- **Results:** `results/raw/calibration/d2_results.json`, `d2_S3_*.json`, `log_d2_*.txt`.
- **Deviations:** D66.
