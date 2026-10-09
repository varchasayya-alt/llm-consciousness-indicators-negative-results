# D3 PROTOCOL PROPOSAL (short; proposed, NOT approved): input-fixed lesions in small pretrained LMs (P1\*)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | Proposal only (PI item 16). **Nothing downloaded, nothing run** |
| Basis | `memo/decision_document_v2.md` (P1\*); `memo/stage1_pivot_memo.md` (D3); `d2_kill_report.md`; lessons R1–R8 |

**Downloads needed. Each needs explicit PI approval; all from Hugging Face; sizes approximate:**

| Item | Size | Use |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` (safetensors) | ~1.0 GB | Primary model |
| `Qwen/Qwen2.5-1.5B-Instruct` (safetensors) | ~3.1 GB | Replication |
| PopQA (`akariasai/PopQA`) | ~20 MB | Factual items. Alternative: a Wikidata-derived list from an approved source |

## 1. Question

In a pretrained LM with *natural* knowledge and natural redundancy, when an **input-fixed** lesion changes item-level recall competence, does a separately trained monitor track the change beyond pre-lesion information and generic change? And does it do so differently from errors caused by input corruption at a matched error rate (the central N1 contrast)?

## 2. Changes relative to P1\*, from B1 and D2

1. **Primary estimand: within-target θ_pre among identically lesioned items, and θ_gen.** D54 applies: only A is affirmative.
2. **S0 before any run, covering satisfiability (R6) and reachability (R8, new from D2).**
   - **Binding gates:**
     - lost ∈ [0.10, 0.50];
     - IQR(ΔC) ≥ 1 nat;
     - identifiability (R²_pre < 0.90, R²_gen < 0.90, R²_joint < 0.95);
     - binary collateral on unrelated-subject items ≤ 0.05, with continuous retention ≤ 0.10 of the lesion's change;
     - sham graded-neutral (R4).
   - **Lever:** a single lesion-strength parameter s ∈ [0, 1], where s = 0 is no lesion and s = 1 is the full knock-out. A fixed grid of five values is declared in advance.
   - **Reachability:** lost(0) = 0 by construction, and lost(1) must be shown ≥ 0.50 on *screening* items (not the evaluation items) before the grid is frozen. The lost window is then reachable on [0, 1] (continuity in s), which D2's stop-on-learning lever lacked.
3. **One literature-standard lesion (R7):** attention knock-out of the last-token→subject-position edges at pre-declared layers (attribute extraction), scaled by s. No new lesion families.
4. **Controls:**
   - matched-norm sham (graded-neutral, R4);
   - unrelated-subject lesion;
   - input corruption (entity-name corruption calibrated to the same error rate);
   - global noise;
   - naturally unknown items.
5. **Replication unit (R5):** model (0.5B, 1.5B) × two pre-declared lesion sites as blocks, with entity-clustered inference. This is weaker than store seeds and stated as a limitation.

## 3. Ladder (model-only kill test first; **no monitor**)

| Stage | Content | Stop if |
|---|---|---|
| D3-S0 | Satisfiability and reachability note | Not demonstrable |
| D3-S1 | Screening: base-correct items; abstention/lookup separates natural known from unknown items (AUROC ≥ 0.60 in at least one model, the decision-document criterion) | Both models < 0.60 |
| D3-S2 | Strength grid plus all control arms (fixed in advance) | — |
| D3-S3 | The gates above, the trivial-decoder audit, D50 reporting | No strength setting passes → stop; return to the PI (the next step would be the D1 note only) |

**STOP before any monitor.** Return the package for PI approval.

## 4. Seeds and compute

- **Seeds** (proposed fresh namespaces for every stochastic component: item draws, sham directions, folds):
  - D3 development 9201–9203;
  - D3 validation 9211–9213.
- **Compute:** CPU inference only.
  - 0.5B: about 5–10 h.
  - 1.5B: about 15–30 h.
  - Runs must be detached processes (the background-task cap is 2 h).
- **Cost:** $0.
