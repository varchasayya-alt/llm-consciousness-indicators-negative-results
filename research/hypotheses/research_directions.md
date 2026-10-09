# Research directions tracker

Living document. Current rationale is in `memo/decision_document_v2.md` and memo §10 (v0.2). Status values: **active** / **bridge** / **deferred** / **parked** / **rejected** / **closed** / **candidate**.

## Status update (2026-10-08)

This table supersedes the v0.2 table below wherever the two differ.

| ID | Direction | Status (2026-10-08) | Record | Next action |
|---|---|---|---|---|
| P5\* Stage 1 / B1 | Developmental decomposition in a synthetic store | **closed** | D61 (B1 closed), D67 (D2 stopped at its kill test) | None |
| C15 (A-stage + R2) | Native lens-defined workspace routing on Qwen3-1.7B / 3.5-2B / 4B | **closed by the PI (2026-10-08)**; preserved as negative / methodological results | D71, D77, D78 | Methods / negative-results note (A) if approved. No R3. G_confirm and G_confirm2 stay sealed |
| **C16** | Engineered capacity-limited re-entrant workspace retrofitted into a frozen small LM | **S0 STOP (D80)**: Qwen2.5-0.5B incompetent in the numbers domain. A planted system shows the blind-write instrument is structurally invalid (lazy producer computation; prefix oracle carries no content). No hypothesis tested | `experiments/c16/s0s1_report.md`, D78–D80 | PI decision: A close and write a combined methods note (recommended); B one bounded C16-R redesign; C pivot |

## v0.2 table (2026-10-01; historical)

| ID | Direction | Status (2026-10-01, v0.2) | Core hypothesis (one line) | Next action |
|---|---|---|---|---|
| **P5\* + N1** | Developmental decomposition + information-flow integration in a synthetic system; causal-identification framework | **active (provisionally approved by PI, 2026-10-01)** | Counterfactually self-dependent monitoring emerges only under specific access / developmental conditions; otherwise an observationally equivalent world-tracking monitor emerges | Stage-1 prereg DRAFTED (`experiments/stage1/`); resolve open decisions D1–D12, run calibration (stores + interventions only), write code + fake-data dry run, then freeze |
| P1\* | Input-invisible competence lesion → selective *control* (lookup, compute allocation) in small pretrained LMs, with full contrast set | **bridge** | Small LMs' control tracks input difficulty, not self-change (artificial anosognosia) | After P5\* Stage 1; Qwen2.5-0.5B/1.5B, Pythia checkpoints |
| P1 (original) | Damage competence → does (verbal/token) confidence fall? | **rejected** | — | Prior art: Gu et al. 2026; Hasegawa et al. 2025/26; Cohen & de Melo 2026 (see memo §10.2) |
| P2\* | Workspace → metacognitive control; ignition; dual-task (post-J-space) | **deferred** ($0 constraint; scoop risk) | C2 control is mediated by J-space contents | Revisit only if a free-GPU dependency is explicitly accepted |
| P3 | Recurrence pathway | folded into P5\* (re-entry factor) and P1\* (Ouro compute-allocation endpoint) | — | — |
| P4 | Implicit attention schema in pretrained LLMs | **parked** | LLMs carry a causally used model of their own attention | Revisit later; could reuse the N1 identification logic (does the schema track attention lesions?) |
| P6 | Continual learning and persistent self-models | partly folded into P5\* (T2 interference, T3 new learning) | — | — |
| P7 | Authorship / efference copy | **parked** (low novelty) | — | — |
| N2 | Conditions for workspace *emergence* in tiny transformers | **parked** (construct validity) | Workspace-like subspaces emerge when tasks demand flexible recombination | Possible later paper |

## Cross-cutting instrument

N1's identification battery is the shared test:

- E1 calibration;
- E2 privileged access;
- E3 counterfactual self-dependence;
- E4 dissociable second-order structure.

The required contrasts are sham, input-corruption, unrelated-lesion and yoked-twin. The battery is validated first in P5\* (known ground truth), then applied to small pretrained LMs (P1\*).
