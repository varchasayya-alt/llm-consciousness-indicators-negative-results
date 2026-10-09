# Consciousness-relevant computation in small language models: a pre-registered, CPU-only research record

This repository is the complete record of a short research program, run from 2026-10-01 to 2026-10-08. It tried to *measure*, by causal intervention, functional properties that scientific theories of consciousness treat as indicators:
- a self-competence monitor;
- a global-workspace-like subspace;
- a retrofitted shared workspace;
- a forced self-model.

It covered small open language models and synthetic transformers, on one laptop CPU, at zero compute cost.

**Start with the technical report: [`research/report/technical_report.md`](research/report/technical_report.md).**

## About this public release

- **All research lines are closed.** No further experiments are planned in this repository.
- **The former confirmation sets are now public.** `research/experiments/c15/materials/g_confirm.json` (C15 A-stage) and `research/experiments/c15r2/materials2/g_confirm2.json` (C15 R2) were sealed during the program and never loaded by any model. Now that they are public, **they can no longer serve as held-out data** for any future confirmation.
- **This is a single-commit snapshot** of a private development repository, taken at its final state (development commit `8cbeb43`). The development history is not published.
  - Commit hashes cited in the technical report, the decision log and this README refer to that history.
  - Every file they cite is included here at its final version.

## Status and claims

**Every line is closed or stopped. No positive scientific claim is made.** Each line stopped at a pre-declared validity, qualification or novelty gate before its main hypothesis could be tested.

The contribution is methodological: twelve lessons (R1–R12) about instrument validity, each tied to a measured failure.

Claims are labelled by level:

| Level | Meaning |
|---|---|
| 1 | Computational result |
| 2 | Conditional mapping to a theory of consciousness |
| 3 | Phenomenal consciousness: **never asserted** |

Nothing here bears on whether any system is conscious.

| Line | Question | Outcome | Report |
|---|---|---|---|
| Stage 1: monitor calibration ladder | Can we induce an input-invisible change in a synthetic store's *own* competence, so that a monitor's self-tracking can be tested? | Six interventions failed binding gates; no monitor was trained | [`B1_final_report.md`](research/experiments/stage1/B1_final_report.md), [`d2_kill_report.md`](research/experiments/stage1/d2_kill_report.md) |
| C15: workspace assay | Does any layer of Qwen3-1.7B / Qwen3.5-2B / Qwen3-4B have a selective, broadcast, reportable Jacobian-lens workspace subspace? | A-stage: 0/15 cells pass. R2 repair: assay failure (P4) | [`astage_report.md`](research/experiments/c15/astage_report.md), [`r2_report.md`](research/experiments/c15r2/r2_report.md) |
| C16: retrofitted workspace | Does a capacity-limited workspace added to a frozen Qwen2.5-0.5B transport *content* to untrained consumers? | Stage 0 STOP (model not competent in the domain); Stage 1 shows the blind-write design is invalid | [`s0s1_report.md`](research/experiments/c16/s0s1_report.md) |
| SM: self-model selection theorem | Is a self-model forced when a self-change must be projected across contexts? | NO-GO: reduces to existing theorems; counterexamples; negligible regret gap | [`sm0_proof_reduction_memo.md`](research/memo/sm0_proof_reduction_memo.md) |

## Repository layout

```
README.md                       this file
LICENSE                         MIT (code); see "License" for documentation (CC BY 4.0)
research/
  report/technical_report.md    the write-up (start here)
  logs/decisions.md             every decision (D1-D80) with alternatives and reasons
  logs/public_release_redactions.md   what was redacted for public release, and how to verify it
  memo/                         design memos, pre-registrations (*_FROZEN.md / .json), theory memos
  experiments/
    stage1/                     monitor calibration ladder (package s1/, runners, chain scripts, stop reports)
    c15/                        C15 A-stage workspace assay (package c15a/, run_astage.py, materials, tests)
    c15r2/                      C15 R2 repair (package c15a2/, run_r2.py, fresh materials, tests)
    c16/                        C16 Stage 0/1 (package c16/, run_c16.py, materials, tests)
  results/raw/                  committed result JSONs, tables and logs (weights and tensors are git-ignored and hashed)
  literature/                   literature_db.json (source of truth), generated matrix and BibTeX, C16 citation audit
  tools/                        build_literature.py, CPU feasibility benchmark
```

## How to reproduce

### Environment

The original runs used Windows 11 and a CPU-only virtualenv at `.venv/`.

| Package | Version |
|---|---|
| Python | 3.12.10 |
| torch | 2.14.1+cpu |
| transformers | 5.18.0 |
| huggingface_hub | 1.33.0 |
| tokenizers | 0.23.2 |
| safetensors | 0.8.0 |
| accelerate | 1.15.0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python -m pip install transformers==5.18.0 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0 accelerate==1.15.0 numpy==2.5.3 scipy==1.18.1 pandas==3.0.6 matplotlib==3.11.2 PyYAML==6.0.3 pytest==9.1.1
```

On Linux or macOS, use `.venv/bin/python`. The Stage-1 chain scripts and the `.cmd` launchers hard-code the Windows venv path `.venv/Scripts/python.exe`; edit their `PY` variable for other systems.

### Model weights (not committed)

Weights live in a git-ignored Hugging Face cache at `hf_cache/`. Runs use `HF_HOME=hf_cache` and, where set, `HF_HUB_OFFLINE=1`.

- **C15:**
  - *Models:* Qwen/Qwen3-1.7B, Qwen/Qwen3.5-2B and Qwen/Qwen3-4B.
  - *Lenses:* their neuronpedia Jacobian lenses.
  - *Pins:* revisions and SHA-256 for every file are in [`research/experiments/c15/artifacts_astage.json`](research/experiments/c15/artifacts_astage.json).
  - *Download:* `research/experiments/c15/tools/download_astage.py`, about 17.5 GB.
  - *Verification:* the `Z0` phase verifies every hash before any run.
- **C16:** Qwen/Qwen2.5-0.5B-Instruct, the locally cached snapshot at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
  - **Caveat:** this revision is not pinned in the frozen C16 package. That is a reproducibility gap.
- **Stage 1 and SM-0** need no downloaded models.

### Re-running an experiment

The runners enforce the pre-registration:
- they refuse unauthorised phases;
- they require a clean tree for their own code;
- they verify frozen hashes.

C15-R2 also checks git-tree pins of the A-stage results.

**What can be re-run from this snapshot.**
- **Runnable as is.** For these experiments the snapshot's code is byte-identical to the code that produced their results, apart from the one-line `ROOT` change in the `.cmd` launchers (checked with `git diff` in the development repository):
  - C15 A-stage, including the CHOOSE fix;
  - C16 S0/S1;
  - SM-0;
  - Stage-1 D2.
- **C15 R2.** Its code is also identical, but `run_r2.py` refuses to run here by design: its guards verify git-tree pins of the A-stage results, and the public-release redaction changed that tree.
- **Stage-1 cycles v2 to v4.2.** These ran older versions of the Stage-1 runners. Reproducing them exactly needs the development history.

The table gives, for each experiment, the development commit at which its code was frozen and where its results were committed.

| Experiment | Development commit (code) | Run (from the directory shown) | Compare with |
|---|---|---|---|
| Stage 1 v2 / v3 | `f1e8efb` / `119a3c5` | `research/experiments/stage1`: `bash _calib_chain.sh` | `results/raw/calibration/calibration_results.json` at `6f5ea82` / `8f24e8e` |
| Stage 1 v4 / v4.1 | `2bcef69` (code identical to `2598e6f`; the chain script was committed with the results) / `3539d3d` | `research/experiments/stage1`: `bash _calib_chain_v4.sh` / `bash _calib_chain_v41.sh` | `calibration_results_v4.json` at `2bcef69`; `calibration_results_v41.json` at `79b63d2` |
| Stage 1 v4.2 | `8a5170f` (includes the chain-script fix) | `research/experiments/stage1`: `bash _calib_chain_v42.sh` | `calibration_results_v42.json` at `0dc1261` |
| Stage 1 D2 | `04b9a4e` (includes the SVD fallback) | `research/experiments/stage1`: `bash _d2_chain.sh` | `d2_results.json` at `2c00545` |
| C15 A-stage | `7e73b5d` (CHOOSE needs the bookkeeping fix in `e69d2b7`) | repo root: `python research/experiments/c15/run_astage.py Z0 --model qwen3-1.7b`, then `SELECT --model <key>` for each model, then `CHOOSE` and `REPORT` (or `bash research/experiments/c15/_astage_select_chain.sh`) | `results/raw/c15a/` at `1d72cd3` |
| C15 R2 | `fe59dd3` (after engineering fixes D75a/b, before any G data) | repo root: `python research/experiments/c15r2/run_r2.py PINS`, then `Z`, `SMOKE2` and `SELECT` with `--model <key>`, then `CLASSIFY` and `REPORT` | `results/raw/c15r2/` at `d17691b` |
| C16 S0/S1 | `aecfe0c` | repo root: `python research/experiments/c16/run_c16.py S0`, then `S1B`, `S1C`, `S1A`, `REPORT` (or `research\experiments\c16\_c16_chain.cmd`) | `results/raw/c16/` at `1902ce7` (S0, S1B) and `29c86dc` (S1C, S1A, REPORT) |
| SM-0 regret gap | `879e615` | repo root: `python research/memo/sm0_regret_gap/sm0_regret_gap.py` (exact arithmetic, about 2 s) | `sm0_regret_gap_results.json` at `6dde7eb` |

**Unit tests.** Each experiment directory has a `pytest.ini`. For example:

```bash
python -m pytest research/experiments/c16
```

**Literature files.** Regenerate the matrix and BibTeX from the database:

```bash
python research/tools/build_literature.py
```

**Approximate CPU cost on the original laptop:**

| Run | CPU time |
|---|---|
| C15 A-stage SELECT | about 23 h |
| C15 R2 SELECT | about 27 h |
| C16 S0/S1 | about 3.5 h |
| Each Stage-1 cycle | minutes to a few hours |

### Integrity conventions

- **Pre-run commits.** Every experiment has a pre-run commit (code, material hashes, frozen thresholds) made before any model saw its material. Deviations are logged as decisions (`D66`, `D75a/b`, `D76`, …).
- **Sealed confirmation splits.** The split guards worked: G_confirm and G_confirm2 were never loaded. They are now public and no longer held out (see "About this public release").
- **Redaction.** Development commit `8cbeb43` redacted the local checkout path from logs and launchers; see [`research/logs/public_release_redactions.md`](research/logs/public_release_redactions.md). Because 12 A-stage logs changed, the C15-R2 tree pin no longer matches.

## Commit map

These hashes refer to the unpublished development history (see "About this public release").

### Phase 0 and direction choice (2026-10-01)

| Commit | What |
|---|---|
| `5ff787a` | Exploratory memo, literature database, draft pilot protocols |
| `f378121` | v0.2 novelty audit; $0 decision document (hardware constraint D12) |

### Stage 1: monitor calibration ladder (2026-10-01 to 10-03)

| Commit | What |
|---|---|
| `e750091`, `b13fc54` | Draft pre-registration; methods code, tests, fake-data dry run |
| `9a13a0a` | v1 calibration: C4 methods failure |
| `f1e8efb`, `ead3fab` → `6f5ea82` | v2 pre-run → **v2 STOP** (sham fingerprint gate) |
| `9aeabe6`, `119a3c5` → `8f24e8e` | v3 design and pre-run → **v3 STOP** (no eligible cell) |
| `1e7bb2b`, `b15be7a`, `2598e6f` → `2bcef69` | v4 (dual-route store) → **STOP** at F1 |
| `3539d3d` → `79b63d2` | v4.1 → **STOP** at F0 |
| `b606db3`, `8a5170f` → `0dc1261` | v4.2 → **STOP** at F2–F4 |
| `5432243` | B1 line closed: final report and pivot memo |
| `7742d07`, `177837d`, `04b9a4e` → `2c00545` | D2 pre-run, novelty audit, engineering fix → **D2 STOP** (kill test) |

### C15: workspace assay (2026-10-03 to 10-08)

| Commit | What |
|---|---|
| `7148a8a`, `81c001d` | Novelty decision documents v3 and v4 (C15 chosen) |
| `db0a3d4`, `79a55c7` | C15-R design and final pre-registration design memos |
| `7e73b5d` | A-stage pre-run commit |
| `8823228`, `e69d2b7` | A-stage SELECT results |
| `1d72cd3` | **A-stage report: ST-2 STOP** |
| `6c615a5`, `ba26a66`, `3d243a9`, `7131249` | R2 repair memo, frozen plan, null calibration (FPR 0.046) |
| `02201df` | R2 pre-run commit |
| `f415bb1`, `86f506a`, `fe59dd3` | R2 engineering fixes (before G data); SELECT started |
| `027813c`, `ba5dca5`, `5bffcb5` | R2 SELECT results (Qwen3-1.7B, Qwen3.5-2B); reboot restart of Qwen3-4B |
| `d17691b` | R2 SELECT complete; **CLASSIFY = P4** |
| `25a21f9` | **R2 report: STOP** |

### C16: retrofitted workspace (2026-10-08)

| Commit | What |
|---|---|
| `437e861` | Novelty and design memo; C15 closed by the PI |
| `aecfe0c` | Citation audit, amendments A1–A3, frozen Stage 0/1 pre-registration and package |
| `1902ce7` | **S0 STOP** (model not competent); S1b passes |
| `29c86dc` | Stage 1 results and S0/S1 report: two structural instrument failures |

### SM: self-model selection theorem (2026-10-08)

| Commit | What |
|---|---|
| `ace2c5b` | Moonshot direction memo (SM selected, with kill criteria K1–K4) |
| `879e615` | SM-0 pre-registration: reduction analysis and frozen regret-gap protocol |
| `6dde7eb` | **SM-0 results: NO-GO** |

### Write-up (2026-10-09)

| Commit | What |
|---|---|
| `8cbeb43` | Technical report, this README, public-release redactions |
| (public snapshot) | This repository's single commit: the files at `8cbeb43`, plus `LICENSE` and these release notes |

## Contributions

- **Principal investigator** (Varchas Yogesh Hebbale): set the research direction and constraints, approved or amended every design, and made every go/no-go decision.
- **Claude Opus 5.5** (Anthropic): did the literature review, wrote the code, ran the experiments and analyses, and drafted the memos and reports.

## License

| Content | License |
|---|---|
| **Code**: Python, shell and `.cmd` scripts, tests, configuration | MIT; see [`LICENSE`](LICENSE) |
| **Report and documentation**: the technical report, READMEs, memos, pre-registrations, decision and experiment logs, and the project's own results files, task materials and literature notes | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| **Third-party material**: bibliographic records retrieved from the Semantic Scholar API (`research/literature/c16_audit/`), and the cited works themselves | Not covered by either license. Rights remain with their owners |

**Abstracts removed.** The 765 paper abstracts that the citation crawl retrieved were removed from this public copy (`"abstract": null`), because they belong to their authors and publishers. Titles, authors, years, IDs and links remain. To re-screen from abstracts, re-run `research/literature/c16_audit/crawl_citations.py`, then `screen_citations.py`.
