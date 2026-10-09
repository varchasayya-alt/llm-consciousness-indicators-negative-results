# C15-R A-stage report: task-independent workspace assay

| Field | Value |
|---|---|
| Date | 2026-10-06 |
| Authorization | PI, 2026-10-04 (D70): A-stage only |
| Protocol | `experiments/c15/astage_protocol.md` (pre-run commit `7e73b5d`) |
| Result files | `results/raw/c15a/`: `select_<model>.json`, `choose_astage.json`, `astage_tables.md` |
| Outcome | **ST-2: no (model, layer) passed W0–W5 on G_select.** The A-stage STOPS |
| Not done (by rule) | FREEZE, CONFIRM, any B/C/H step, any SAT data, F, rescue routes, further downloads. **G_confirm is sealed and unused** |
| Claims discipline | Functional organisation only. Nothing here bears on phenomenal consciousness |

---

## 1. Verdict

1. **None of the three approved CPU-feasible candidates qualifies.** Qwen3-1.7B, Qwen3.5-2B and Qwen3-4B were assessed on 15 band layers in total. None shows the full pre-registered workspace-like profile. Under PI amendment 4, the "workspace-like" label is **not licensed for any of them**, and C15 cannot proceed as a workspace-routing study on these models.
2. **The workspace signatures are fragmented across models:**
   - **Qwen3.5-2B:** route-specific reportability (W1) and route-specific cross-function broadcast (W2) are strong. But J ablation is anti-selective even at matched functional impact (W5 fails), and W0 fails.
   - **Qwen3-4B:** J ablation is causally relevant (W4) and, at layers 11 and 13, **"selective at matched functional impact but anti-selective at matched intervention norm"** (SI_iso 0.75 and 0.69; SI_norm 2.2–2.6). But reportability and broadcast are weak or absent (W1/W2 fail), and W0 fails.
   - **Qwen3-1.7B:** only W2 passes (layers 10–16).
3. **Two gates fail everywhere, and I believe both reflect assay sensitivity as much as model properties (§5):**
   - W0's intermediate-entity readout: 0.02 / 0.04 / 0.17 against a threshold of 0.40.
   - W3's cross-position transport: R² ≤ 0 in both arms, in every model and layer.
   - They are reported exactly as pre-registered. **Neither has been revised.**
4. **Significance confidence for C15 as a workspace paper on CPU-feasible models drops from MEDIUM to LOW.** The selectivity risk you flagged as binding did bind for two of the three models. The full profile was absent in all three.

## 2. What ran

- **Artifacts:** pinned revisions and SHA-256 (`artifacts_astage.json`). Z0 passed for all three.
  - Qwen3.5-2B is the post-trained checkpoint with lens `qwen3.5-2b`. The `-pt` lens belongs to Qwen3.5-2B-Base and was not used.
- **Material:** author-written and task-independent (`materials/`). Per split: 63 paragraphs, 20 concepts, 20 countries, 74 two-hop items. Trial designs were hashed before any model run. Only **G_select** was used.
- **SELECT runtimes (fp32 CPU):** 6.6 h (1.7B), 5.6 h (3.5-2B), 11.0 h (4B).
  - The first Qwen3-4B worker was killed by an app-session restart before writing any result. It was re-run with identical committed code and data, launched via WMI so that it ran outside the app's process tree.
- **Pre-data refinements** (protocol §11, D70; all decided on SMOKE engineering evidence before any G data):
  - dose rule on subsequent text (KL ≤ 0.05, agreement ≥ 0.95 at robust positions, monotone admissibility);
  - empirical generic-KL matching from a diversified pool;
  - natural-magnitude W2 swaps;
  - per-model GP screening (exact GP for Qwen3.5-2B);
  - SI_iso certification rules.
- **Bookkeeping fix after SELECT:** CHOOSE crashed parsing "inf" ratios from JSON. This was fixed and tested (`selection.py`, commit `e69d2b7`); no measurement code changed.

## 3. Results on G_select (per pre-registered gate)

Columns:
- W1 = forced-choice report hit rate, J / matched ⊥ / no injection (chance 0.10);
- W2 = consistent cross-function switch rate, J / ⊥;
- W3 = transport breadth, J / ⊥;
- W4 = two-hop impairment, J vs norm-matched random;
- W5 = SI_iso, with SI_norm always reported.
- In the gate column, ✓/✗ is listed in the order W0–W5.

#### qwen3-1.7b (G_select; 6.6 h CPU)

W0 — lens agreement @ L−2 **0.674** (≥ 0.60), two-hop accuracy **0.581** (≥ 0.60; n = 74), intermediate readout **0.023** (≥ 0.40; n correct = 43) → **FAIL**

| Layer | r | a\* | W1 hit J / ⊥ / none | W2 rate J / ⊥ (n) | W3 BB J / ⊥ | W4 impairment J / rand₁ (pp) | W5 SI_iso [95% CI] | SI_norm | W0–W5 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 49 | 2.0 | 0.31 / 0.34 / 0.18 | 0.06 / 0.00 (16) | 0.014 / 0.019 | 4.1 / -4.1 | 1.32 [1.20, 1.46] | 5.14 | ✗✗✗✗✗✗ (W4 not assessable) |
| 10 | 75 | 2.0 | 0.36 / 0.27 / 0.15 | 0.45 / 0.00 (20) | 0.037 / 0.028 | 14.9 / -4.1 | 1.12 [1.03, 1.22] | 5.32 | ✗✗✓✗✗✗ (W4 not assessable) |
| 12 | 86 | 2.0 | 0.28 / 0.24 / 0.16 | 0.32 / 0.00 (19) | 0.025 / 0.029 | 6.8 / -2.7 | 1.56 [1.44, 1.67] | 6.16 | ✗✗✓✗✗✗ (W4 not assessable) |
| 14 | 110 | 2.0 | 0.25 / 0.24 / 0.17 | 0.35 / 0.00 (20) | 0.008 / 0.017 | 8.1 / -5.4 | 2.31 [2.16, 2.48] | 6.63 | ✗✗✓✗✗✗ (W4 not assessable) |
| 16 | 156 | 2.0 | 0.23 / 0.24 / 0.17 | 0.37 / 0.00 (19) | 0.000 / 0.010 | -2.7 / -2.7 | 7.47 [6.39, 8.97] | 7.47 | ✗✗✓✗✗✗ (W4 not assessable) |

#### qwen3.5-2b (G_select; 5.6 h CPU)

W0 — lens agreement @ L−2 **0.545** (≥ 0.60), two-hop accuracy **0.622** (≥ 0.60; n = 74), intermediate readout **0.043** (≥ 0.40; n correct = 46) → **FAIL**

| Layer | r | a\* | W1 hit J / ⊥ / none | W2 rate J / ⊥ (n) | W3 BB J / ⊥ | W4 impairment J / rand₁ (pp) | W5 SI_iso [95% CI] | SI_norm | W0–W5 |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 162 | 2.0 | 0.64 / 0.21 / 0.15 | 0.92 / 0.00 (12) | 0.004 / 0.043 | 12.2 / 2.7 | 1.12 [1.04, 1.21] | 2.91 | ✗✓✗✗✗✗ (W2 ⊥ unmatched 40%) |
| 9 | 178 | 2.0 | 0.48 / 0.19 / 0.12 | 1.00 / 0.00 (18) | 0.000 / 0.042 | 12.2 / 0.0 | 1.50 [1.39, 1.63] | 3.30 | ✗✓✓✗✓✗ |
| 11 | 208 | 2.0 | 0.47 / 0.13 / 0.11 | 1.00 / 0.00 (20) | 0.000 / 0.020 | 13.5 / 4.1 | 1.99 [1.80, 2.22] | 3.35 | ✗✗✓✗✗✗ (W1 ⊥ unmatched 25%) |
| 13 | 256 | 2.0 | 0.37 / 0.17 / 0.12 | 1.00 / 0.00 (20) | 0.000 / 0.000 | 16.2 / 2.7 | 1.55 [1.45, 1.67] | 3.43 | ✗✓✓✗✓✗ |

#### qwen3-4b (G_select; 11.0 h CPU)

W0 — lens agreement @ L−2 **0.720** (≥ 0.60), two-hop accuracy **0.703** (≥ 0.60; n = 74), intermediate readout **0.173** (≥ 0.40; n correct = 52) → **FAIL**

| Layer | r | a\* | W1 hit J / ⊥ / none | W2 rate J / ⊥ (n) | W3 BB J / ⊥ | W4 impairment J / rand₁ (pp) | W5 SI_iso [95% CI] | SI_norm | W0–W5 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 73 | 1.0 | 0.26 / 0.16 / 0.17 | 0.13 / 0.00 (15) | 0.002 / 0.007 | 20.3 / 1.4 | 0.75 [0.69, 0.82] | 2.22 | ✗✗✗✗✓✓ (W2 ⊥ unmatched 25%) |
| 13 | 81 | 2.0 | 0.31 / 0.23 / 0.18 | 0.11 / 0.00 (18) | 0.005 / 0.028 | 25.7 / 0.0 | 0.69 [0.63, 0.74] | 2.60 | ✗✗✗✗✓✓ |
| 15 | 73 | 1.0 | 0.29 / 0.20 / 0.18 | 0.00 / 0.00 (19) | 0.000 / 0.000 | 21.6 / 0.0 | 1.04 [0.97, 1.11] | 2.26 | ✗✗✗✗✓✗ (W1 ⊥ unmatched 35%) |
| 17 | 99 | 1.0 | 0.33 / 0.21 / 0.19 | 0.18 / 0.00 (17) | 0.000 / 0.009 | 17.6 / 8.1 | 1.53 [1.39, 1.67] | 3.01 | ✗✗✗✗✗✗ |
| 19 | 119 | 2.0 | 0.29 / 0.21 / 0.20 | 0.15 / 0.00 (20) | 0.000 / 0.000 | 13.5 / 1.4 | 1.62 [1.49, 1.78] | 2.86 | ✗✗✗✗✓✗ (W1 ⊥ unmatched 30%) |
| 21 | 126 | 2.0 | 0.20 / 0.19 / 0.17 | 0.05 / 0.00 (19) | 0.000 / 0.000 | 14.9 / 1.4 | 1.56 [1.43, 1.72] | 2.61 | ✗✗✗✗✓✗ (W1 ⊥ unmatched 25%) |

## 4. Gate-by-gate synthesis

| Gate | 1.7B | 3.5-2B | 4B | Reading |
|---|---|---|---|---|
| W0a lens agreement | 0.67 ✓ | **0.55 ✗** | 0.72 ✓ | Lens readout is weaker in the hybrid-attention Qwen3.5 |
| W0b two-hop accuracy | **0.58 ✗** | 0.62 ✓ | 0.70 ✓ | 1.7B is just below; W4 is not assessable there |
| W0b intermediate readout | **0.02 ✗** | **0.04 ✗** | **0.17 ✗** | Fails everywhere (see §5) |
| W1 reportability | ✗ (J ≈ ⊥) | **✓ at 7, 9, 13** (J 0.37–0.64 vs ⊥ 0.13–0.21) | ✗ (J/⊥ < 2) | Route-specific report only in Qwen3.5-2B |
| W2 cross-function broadcast | **✓ at 10–16** (0.32–0.45 vs 0.00) | **✓ at 9–13** (0.92–1.00 vs 0.00) | ✗ (≤ 0.18 vs 0.00) | J swaps switch capital/language/currency; matched ⊥ swaps never do |
| W3 transport | ✗ (≈ 0 / ≈ 0) | ✗ (≈ 0 / ≈ 0) | ✗ (≈ 0 / ≈ 0) | Neither route decodable downstream (see §5) |
| W4 causal relevance | not assessable (diff 19 pp at layer 10) | ✓ at 9, 13 | **✓ at 11, 13, 15, 19, 21** (13.5–25.7 pp vs ≤ 8 pp) | J ablation hurts two-hop reasoning far more than norm-matched random |
| W5 selectivity (SI_iso) | ✗ (1.12–7.47) | ✗ (1.12–1.99, CI > 1) | **✓ at 11, 13** (0.75, 0.69; CI < 1) | Only Qwen3-4B is selective at matched functional impact. SI_norm > 1 everywhere (2.2–7.5), consistent with published anti-selectivity at matched norm |

**No single (model, layer) combines reportability, broadcast, transport, causal relevance and selectivity.** The best partial profiles are:
- Qwen3.5-2B layers 9 and 13: W1, W2 and W4 pass;
- Qwen3-4B layers 11 and 13: W4 and W5 pass.

## 5. Assay-validity concerns (flagged for the PI; nothing was revised)

1. **W3 looks insensitive, not informative.**
   - Cross-fitted R² ≤ 0 in *both* arms at every site, in every model.
   - Yet W1 shows that J-injected content changes the model's report about 12 tokens later (hit rate 0.37–0.64 vs 0.12–0.15 without injection, in Qwen3.5-2B). Behaviourally, cross-position influence exists.
   - The pre-registered decoder (n = 40 contexts, d = 2048–2560, single-position scalar) most likely lacked power against context variance.
   - As specified, W3 cannot pass for any model. **Treat it as an assay failure, not as evidence against transport.**
2. **W0b's readout site may be wrong.**
   - The intermediate entity was read only at the last prompt position. The jlens reference example reads at another position, and intermediates may sit at the bridge tokens.
   - The 0.40 threshold was a prior guess.
   - The steep size trend (0.02 → 0.17) suggests the measure tracks scale.
3. **W1 option bias.** The no-injection hit rate is 0.11–0.20 against a nominal chance of 0.10. The J-vs-⊥ comparison is unaffected, but the absolute 0.30 threshold is partly prior-driven.
4. **Matching coverage.** At several layers, more than 20% of contents had no ⊥ control within ±15% empirical KL, which forced test failures there (e.g. Qwen3-4B W1 at layers 15, 19, 21). The binding empirical-KL match is strict by design.
5. **Dose.** a\* = 1–2 × the median residual norm at one position. This is non-disruptive to subsequent text by the memo's rule, but locally large.

Because G_select was used for selection, none of these observations is confirmatory. **Changing W0b, W3 or W1 now would be post hoc with respect to G_select.**

## 6. What can be said (Level-1, exploratory, G_select only)

1. **Qwen3-4B, layers 11–13.**
   - Ablating each position's top-10 active J-lens atoms impairs two-hop factual reasoning by 20–26 points (random ≈ 0–1).
   - At matched functional impact, it damages ordinary next-token prediction **less** than random perturbation (SI_iso 0.69–0.75).
   - At matched norm it damages more (SI_norm 2.2–2.6).
   - So matched-norm and matched-impact comparisons give opposite selectivity verdicts. That is a methodological point for the J-space literature: a potent subspace is not thereby non-selective.
2. **Qwen3.5-2B.** J-subspace components of concept vectors are reportable and drive consistent multi-function attribute switches, while empirically matched non-J controls of equal norm, lens gain, propagation and generic KL almost never do.
3. **Neither result licenses the "workspace-like" label**, and neither has been confirmed on held-out material.

## 7. Implications and options (PI decision; nothing proceeds automatically)

Per the pre-registration, C15 stops as an artificial-consciousness/workspace paper on these candidates. No further models are downloaded (amendment 10). Options:

| Option | What it would mean | My assessment |
|---|---|---|
| **A. Stop C15; write up the A-stage** | Methods / negative-results note. Contributions: the task-independent assay battery; empirically matched non-workspace controls; the matched-impact vs matched-norm selectivity contrast; the fragmentation finding; strict select/confirm discipline | **Recommended.** It is the honest reading of the pre-registered result, and the SI_iso/SI_norm dissociation is a genuine methodological contribution |
| B. Reframe as a generic routing/control study | Drop the workspace label; use a "lens-verbalisable route" (e.g. Qwen3.5-2B layer 9/13, or Qwen3-4B layer 11/13) for the B/C/H rescue design | Possible, but it lowers the claim. Needs your explicit decision and a new claim tier |
| C. Revise W0b/W3 and re-assess | New pre-registration of revised assays. Fresh generic material would be needed, because G_select is spent for selection. G_confirm is still sealed and could serve once as the confirmation split | Scientifically defensible only with frozen revisions and untouched confirmation data. Selection would still use spent G_select unless new select material is written |
| D. Assess larger or other candidates | Qwen3.5-4B, Gemma-3-4b-it (download / licence approval) | Published Qwen3 results improve with scale. A larger model is CPU-heavy (4B already took 11 h); Gemma needs your licence acceptance |

## 8. Files and commits

| Item | Location |
|---|---|
| Pre-run commit | `7e73b5d` |
| Results commits | `8823228` (1.7B, 3.5-2B); `e69d2b7` (4B + CHOOSE fix); this report |
| Select results | `results/raw/c15a/select_qwen3-1.7b.json`, `select_qwen3.5-2b.json`, `select_qwen3-4b.json` |
| Select tensors | `results/raw/c15a/select_tensors/` (S_J bases, KL matrices, V_gen; untracked, hashed) |
| Choice | `results/raw/c15a/choose_astage.json` (0 passers; no replication candidate) |
| Tables | `results/raw/c15a/astage_tables.md` |
