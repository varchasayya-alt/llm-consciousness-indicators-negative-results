# C16 Stage 0 / Stage 1 preregistration (FROZEN)

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Status | **FROZEN at the pre-run commit** (decision log D79). This file, `c16_s0s1_thresholds_FROZEN.json`, the package `experiments/c16/` and the material manifest are committed and hashed before any model sees C16 material. Nothing is changed after a result. Deviations are logged as D79a, D79b, … with reasons |
| Authorization | PI, 2026-10-08: the full-text and citation audit; C16 **Stage 0 on the cached Qwen2.5-0.5B-Instruct**; then **synthetic / planted Stage 1**. **No Stage-2 training.** Stop and report after Stage 1 |
| Parent documents | `c16_novelty_design_decision_memo.md` (D78); audit `literature/c16_audit/audit_report.md` |
| Claims discipline | Stage 0/1 make **no** hypothesis claim. They validate instruments, measurement, satisfiability and power. Level 3 is never claimed |

---

## 1. Audit outcome (precondition for freezing)

**The exact C16 cell survives** (`audit_report.md` §1). The closest precedent is Marincat 2026a–c: language-model "societies" on the same frozen Qwen2.5-0.5B, using packet-swap tests and visibility regimes. It narrows the conceptual margin but differs on every C16-defining element:
- within-pass re-entry;
- frozen native consumers;
- broadcast to qualitatively different consumers;
- blind vs addressed writing at equal capacity;
- a capacity × consumer-diversity manipulation;
- an untrained native consumer.

The memo's novelty rating for the exact cell is revised from ≈ 0.6 to **≈ 0.5**.

## 2. PI amendments (2026-10-08)

### A1. Fair addressed-channel and private-line controls

The addressed control C1 must be able to represent the untrained consumer U. A learned consumer-ID table could not. It is replaced by a **cue-derived consumer representation**:

- **Cue representation.**
  - e(cue) = mean, over the cue tokens, of the frozen LM's residual at layer L_c (= L_w), computed in a separate forward pass of the **cue string alone** (e.g. "Q: Take the number. Is it even or odd?").
  - The same frozen computation gives e for every cue, including U's, so no consumer-specific parameter exists.
- **Conditioning module.** Writer slot queries q_k ← q_k + W_q^c·LN(e). Writer candidates c_t ← c_t + W_c^c·LN(e). Both W_q^c and W_c^c map d → d_w.
- **A receives the same module.** A's input is the constant ē = mean of e over the *training* consumers' cues, frozen. A and C1 therefore have **identical parameters**, and the only difference is whether the writer knows the consumer's cue.
- C1 can express A's solution exactly (W^c = 0), so it is never structurally weaker.
- **C2 (private lines) gets the same treatment.** Read routing over its P lines is ρ = softmax(W_ρ·LN(e)), so U receives a cue-determined mixture of lines rather than "no line".
- **Validity gate V6 (a positive control for C1's generalization).** On *paraphrased* cues of trained consumers (never used in workspace training), C1 must reach ≥ 0.80 of its trained-cue CT. This shows that a C1 failure on U is not an inability to read new cues.
- V6 is exercised in the Stage-1 planted system now and becomes a Stage-2 gate.

### A2. The capacity × consumer-diversity hypothesis (H-CD)

The audit (§4) found:
- the **peaked** window is not defensible in information terms (an answer bundle never needs more bits than the content);
- its upper limb was looked for and not found in two related settings.

A defensible, preregisterable version exists only as a **three-account discrimination** in **format-geometry** coordinates measured before any training:
- r_X = the residual rank needed to deliver content X in native format (content oracle);
- r_B(S) = the rank needed to deliver the native-format answer bundle of a consumer set S (bundle oracle).

| Account | Prediction on the grid c ∈ {0.5, 1, 2, 4, 8}·r_X × S |
|---|---|
| **Geometry** | U-transfer only for r_X ≤ c < r_B(S): a window that widens with diversity; decline for c ≥ r_B(S) |
| **Information bottleneck** | U-transfer only if S jointly determines U's input; no upper decline; tighter c never helps |
| **Native copy ("lingua franca")** | U-transfer whenever c ≥ r_X; no effect of diversity or of S's identifiability |

**Discriminating contrasts** (Stage-2; frozen there):
- **D** (diversity) = T(2, S4) − T(2, S1);
- **Q** (upper limb) = T(2, S4) − T(8, S4);
- **I** (identifiability) = T(2, Spm) − T(2, S4);
- here T = CT_U for arm A.

**Stage 0/1 role.** Stage 0 measures r_X and r_B(S) and gates whether the window is measurable (W1–W4). Stage 1 sets the seeds and items needed for power ≥ 0.80 with complete-gate FPR ≤ 0.05.

**Whether H-CD becomes Stage 2's central hypothesis is decided at the PI review after Stage 1, by this pre-declared rule:** central iff W1–W4 pass and S1a power ≥ 0.80 within ≤ 120 CPU-h of Stage-2 training. Otherwise H-CD is secondary: the lower bound and D only, or not run.

### A3. Conditional naturalistic extension (no Stage-0/1 cost; not part of any initial claim)

Run **only after a confirmed W-BC or W-GW** result, under a separate preregistration:
- **Producer:** verification of a stated claim, e.g. "Ana says 31+6 is 38." The content is the verdict plus the corrected value. This is the native verification-without-integration gap (Kawada & Kellis 2026).
- **Qualitatively different consumers:**
  - **revision** ("What should Ana have said?" → corrected value);
  - **retrieval keyed on the verdict** ("If Ana was right take the red card, otherwise the blue card." → card);
  - **keyed lookup on the corrected value**.
- **Untrained consumer:** abstention or confidence report.
- Same arms, endpoints and gates as the core.

---

## 3. Materials (numbers domain, format F1)

**Numbers.**
- X ∈ {10, …, 99} \ {50}, which gives 89 values.
- Split with seed 16001, stratified by parity × (X > 50): **X_train 45**, **X_select 22**, **X_confirm 22**.
- **Stage 0/1 use X_train only.** The select and confirm value lists are written and hashed in the manifest. The runner refuses to build items from them in any S0/S1 phase.

**Names** (single-token with and without a leading space, verified):
- train: Ana, Ben, Dan, Jon, Sam, Tom, Max, Leo, Ian, Ray;
- eval (sealed): Ted, Amy, Joe, Kim, Lou, Roy, Tim, Val;
- "Bo" is reserved for the F2 primer.

**Producer sentences** (X never appears on the surface):

| Code | Sentence | Constraints |
|---|---|---|
| P_add | "{N}'s number is {a}+{b}." | b ~ U{2..9}; a = X − b ≥ 2 |
| P_sub | "{N}'s number is {a}-{b}." | b ~ U{2..9}; a = X + b |
| P_mul | "{N}'s number is {k}*{m}." | 2 ≤ k ≤ m ≤ 9 with km = X; only the X_train values that factor this way |

Paired source sentences (same name):
- **text:** "{N}'s number is {X}.";
- **bundles:**
  - B1 "{N}'s number plus one is {X+1}.";
  - B2 "{N}'s number plus one is {X+1}, and plus ten is {X+10}.";
  - B4 "{N}'s number plus one is {X+1}, plus ten is {X+10}; it is {even|odd} and {larger|smaller} than 50.";
  - Bpm "{N}'s number is {even|odd} and {larger|smaller} than 50.".

**Episode:** `{sentence}\nQ: Take {N}'s number. {question}\nA:` followed by the teacher-forced answer `" {answer}"`. The **prefix** "Q: Take {N}'s number." is 7 tokens. It is identical across paired runs and causally precedes every consumer-specific token (checked by test).

**Consumers** (questions):

| Code | Question | Answer |
|---|---|---|
| copy | "What is it?" | X |
| succ | "What is it plus one?" | X+1 |
| plus10 | "What is it plus ten?" | X+10 |
| parity | "Is it even or odd?" | even / odd |
| mag | "Is it larger or smaller than 50?" | larger / smaller |
| lookup *(parametric)* | "The codes are: {x1} is {c1}, {x2} is {c2}, {x3} is {c3}. Which colour is its code?" | X's colour. X sits at a random one of 3 positions; distractor values from X_train \ {X}; colours from {red, blue, green, black, white, pink} |
| verb = **U** | "How is it written in words?" | English words, e.g. "thirty-seven" |

**Cue strings** (for e(cue)): "Q: Take the number. {question}". For lookup, the question with the code list replaced by "The codes are listed. Which colour is its code?".

**Instances (Stage 0):**
- Main set: for every X ∈ X_train and each producer, **4** instances (names from the train names; addends sampled; seed 16010).
- Decodability set: **8** instances per X per producer.

**F2 fallback format** (pre-declared; used only if F1 fails V0 or V1a/b/d): each episode is preceded by one fixed primer of the same consumer, "Bo's number is 4.\nQ: Take Bo's number. {question}\nA: {answer(4)}\n\n". For lookup the primer codes are 4 / 6 / 8 → red / blue / green. F2 is evaluated once. If it also fails: **STOP** (no model or domain change without the PI).

---

## 4. Stage 0 (Qwen2.5-0.5B-Instruct, fp32, CPU, forward only)

All correctness is **greedy correctness by teacher forcing**: the argmax at every answer position must equal the target token. This is exactly equivalent to greedy decoding producing the target. "Layer l" means the residual stream entering decoder layer l (`hidden_states[l]`; l = 0 is the embedding output). Patching adds a vector to that residual at specified positions through a forward pre-hook. A zero patch must reproduce unpatched logits exactly (engineering check E2).

**M0 Engineering (non-gating, required to proceed).**
- E1: tokenization and prefix-identity checks on every paired item.
- E2: zero-patch identity.
- E3: throughput.
- E4: a backward-timing probe. A zero-initialized dummy adapter is attached at three layers; forward and backward run on 5 batches; **no optimizer step and no parameter update**. This projects Stage-2 cost.

**M1 Decodability (V0).**
- Data: the decodability set (add, sub, mul).
- Features: residuals at the last 3 producer-sentence tokens, for every layer.
- Model: ridge one-vs-rest classifiers (λ = 1, standardized, PCA-256 fitted in-fold) for the **tens and ones digits** of X. An item counts as decoded iff both digits are right.
- Validation: 5-fold CV, folds stratified by X.
- For each producer and layer, take the max over the 3 positions.
- **L_w** = argmax over layers 8–23 of the mean over add and sub.

**M2 Competence and native gap (V1).** For every main-set instance i and consumer j, compute:
- native correctness (latent sentence);
- text correctness (text sentence with the same name and X);
- copy correctness (latent sentence, copy consumer).

Derived quantities:
- **Eligibility:** E(i, j) = copy(i) ∧ text(i, j).
- **CoT accuracy** = the rate of E (the external-scratchpad route).
- **Native κ_j** = (acc_native − chance_j) / (1 − chance_j) on eligible add/sub items. Chance is 0.5 for parity and mag, 1/3 for lookup, and 0 for the others.

**M3 Content-oracle layer sweep (choose l_o and R).**
- Split: X_train is split into halves H_A and H_B by seed 16011.
- Sweep: on eligible add/sub items in H_A (≤ 100 per consumer), apply the full content transplant at the prefix: Δ(i, p) = h_text(i)[l][p] − h_latent(i)[l][p], for l ∈ {1, 2, 3, 4, 6, 8, 10, 12, 14} with l ≤ L_w − 3.
- gain_j(l) = acc_patched − acc_native.
- **l_o** = argmax of the mean gain over {succ, plus10, parity, mag, lookup, verb}.
- **R** (Stage-2 read layers) = the top 3 layers by that mean.

**M4 Oracle transport, negative control, ranks (at l_o, cross-fitted H_A ↔ H_B; ≤ 100 eligible items per half per consumer).**
- **Swap oracle.**
  - Transplant the full Δ from the text run of X′ (same name, same producer type; X′ drawn from the same half with f(X′) ≠ f(X)).
  - **CT_oracle,j** = P(answer = f_j(X′) | swap) − P(answer = f_j(X′) | NC).
- **NC (negative control).** Δ_NC(i, p) = μ_p + ‖Δ(i, p) − μ_p‖ · z/‖z‖, with z ~ N(0, I). This keeps the generic text-vs-latent mean shift and replaces the item-specific part with a random direction of equal norm.
- **Rank sweep.**
  - Δ_r = μ_p + U_r U_rᵀ (Δ − μ_p). μ_p and U_r (top-r right singular vectors of the pooled Δ − μ_p over prefix positions) are fitted on one half and evaluated on the other.
  - r ∈ {1, 2, 4, 8, 16, 32, 64, 128, full}.
  - **r\*_j** = the smallest r with gain(r) ≥ 0.9 · gain(full). Undefined if gain(full) < 0.10.
  - **r_X** = max of r\*_j over the consumers passing V2a (including U).
  - Each half's estimate is reported; the reported value is their geometric mean.
- **Bundle oracles** B1, B2, B4 and Bpm. Same procedure, with the bundle sentence as source, evaluated on the consumers each set covers plus U.
  - B1 → {succ}; B2 → {succ, plus10}; B4 → {succ, plus10, parity, mag}; Bpm → {parity, mag}.
  - **r_B(S)** = max of r\*_j over j ∈ S.
  - Also measured: source-text competence of each bundle sentence, and the **bundle gain on U**.

**Stage-0 gates** (binding; numbers in the JSON):

| Gate | Criterion |
|---|---|
| V0 | Decodability ≥ 0.70 at L_w for add and for sub (mul is reported; it enters Stage 2 only if ≥ 0.70) |
| V1a | Copy (producer competence) ≥ 0.80 for add and for sub |
| V1b | Text-consumer accuracy ≥ 0.80 for ≥ 4 of the 5 pool consumers {succ, plus10, parity, mag, lookup}, and for verb (U) |
| V1c | Native κ ≤ 0.35 for ≥ 3 pool consumers passing V1b, and for U |
| V1d | Eligibility ≥ 0.50 for those consumers |
| V2a | Full content-oracle gain at l_o ≥ 0.25 for ≥ 3 pool consumers passing V1, and for U |
| V2b | CT_oracle ≥ 0.30 for ≥ 3 of those pool consumers, and for U |
| V2c | NC neutrality on every evaluated consumer: \|P_NC(f(X′)) − P_native(f(X′))\| ≤ 0.10 and \|acc_NC − acc_native\| ≤ 0.15 |

**Core instrument-valid** iff V0–V2 all pass. Consumers failing V1b/c are excluded from Stage-2 trained sets (recorded).

**Phase-diagram measurement gates** (decide H-CD's status; not STOP gates):

| Gate | Criterion |
|---|---|
| W1 Reliability | \|log₂(r_X^{A→B} / r_X^{B→A})\| ≤ 1, and the same for r_B(B4) |
| W2 Format specificity | Bundle B4 gain on U ≤ 0.5 × content gain on U, **and** bundle B4 gain on its own consumers ≥ 0.25 (the bundle oracle is valid) |
| W3 Window measurability | r_B(B4) ≥ 4 · r_X |
| W4 Diversity monotone | r_B(B1) ≤ r_B(B2) ≤ r_B(B4), allowing one grid step |

---

## 5. Stage 1 (synthetic / planted; no LM data beyond the Stage-0 summary statistics)

### S1b Rank-recovery check (analytic synthetic, runs first)

- **Synthetic generator.** A linear-Gaussian "native format":
  - content z ∈ ℝ^{r_true};
  - h = A z + ε (d = 896; A random orthonormal);
  - consumers = argmax readouts of fixed random linear maps of the patched state.
- **Procedure.** The **same** `ranks.py` estimator recovers r_true ∈ {4, 16, 64}, and bundle ranks for r_B,true ∈ {16, 64}.
- **Gate S1b:** every estimate within ×2 of the truth.

### S1c Planted tiny transformer (end-to-end pipeline and joint satisfiability, R6)

- **Base model.**
  - Trained from scratch: 4 layers, d = 128, 4 heads, learned absolute positions, seed 16200.
  - Vocabulary: digits, + - *, 10 names, prompt tokens, op tokens with **two synonyms each**, answer tokens, colours, number words, and a filler token.
  - Training data: producer + copy; text + every consumer (both synonyms); bundle sentences + the matching consumers; lookup tables.
  - **No latent producer + non-copy consumer sequences**, so the native gap is built in.
  - Text producer sentences are padded with fillers to the latent length, so positions align.
- **Training budgets.**
  - Base model: AdamW with a one-cycle schedule (lr 3e-3), batch 256, **4000 steps**. If the base model's copy or text-consumer accuracy on its train values is < 0.80, it is retrained **once** to 8000 steps (pre-declared).
  - Arms: Adam (lr 3e-3), batch 64, **1500 steps** for the core arms and **1000 steps** for the pilot. K = 2 slots, slot noise σ = 0.1, C3 width r_v = d.
- **Continuation rule.** On the planted model, M1–M4 always run to completion (`force_continue`), and the arms run even if S1c-1 fails. An S1c-1 failure is itself the reportable result: it questions the gates or the pipeline.
- **Stage-0 pipeline on the planted model.** M1–M4 run with the planted adapter. Layer parameters are scaled to the 4-layer model:
  - decodability layer_min = 2, PCA-64;
  - oracle layers {0, 1, 2} with l ≤ L_w − 1;
  - 2 read layers;
  - all other parameters, gates and thresholds as for Stage 0;
  - P = 5 prefix tokens.
- **Gate S1c-1:** V0, V1a–d and V2a–c pass on the planted model. If the planted system cannot pass its own instrument gates, the gates or pipeline are at fault, and S0's verdict is not interpretable until this is fixed.
- **Workspace arms** A, C1 (cue-derived), C3 (Back-Attention-type, unrestricted):
  - 3 producers × 4 trained consumers (succ, parity, mag, lookup): 8 TR / 4 HO pairings (connected support); U = verb.
  - 3 seeds (16211–16213).
  - Training: CE on answer tokens, X_train values. Evaluation: X_eval values (planted split).
- **Gate S1c-2 (pipeline validity):**
  - zero-gate bit-identity at initialization;
  - I6 cue invariance for A, exact (slot state identical across cues);
  - training reach: A's TR accuracy − native ≥ 0.20;
  - transport positive control: A's CT on trained pairings ≥ 0.5. (The copy consumer is deliberately not trained: a trained copy consumer would force a content code and confound A vs C1 and H-CD);
  - NC neutrality as V2c (random slots of matched per-slot norm vs no workspace), on the TR, HO and U sets;
  - **V6 C1 paraphrase control** ≥ 0.80 × C1's trained-cue CT.
- **Planted phase-diagram pilot.**
  - A at c ∈ {0.5, 1, 2, 4, 8} · r_X(planted) × S ∈ {S1 = {succ}, S4 = {succ, plus10, parity, mag}, Spm = {parity, mag}} × 3 seeds (16221–16223).
- **Status of S1c outputs.** CT, T(c, S) and σ_seed are **reported as planted-system pipeline outputs, not evidence about LMs or about H-CD**. They are used **only** to parameterize S1a variances.

### S1a Parametric simulator (power and complete-gate FPR)

- **Core classification** (memo §4.7, with A1 and V6 added).
  - Model: item-level Bernoulli outcomes with logit-normal seed and X-cluster effects.
  - Inputs: σ_seed from S1c; σ_X ∈ {0.3, 0.6}; NC and native rates from S0.
  - Worlds: null, generic, re-entry, broadcast, full GW, lingua franca.
  - Outputs:
    - the **complete-gate FPR** of each class claim under every world in which that class is false;
    - power for the true class at n_s = 3 seeds and the planned item counts.
- **H-CD.**
  - Worlds: geometry, IB, copy, null.
  - Grid: as in A2, with S1, S2, S4, Spm and Sparam.
  - Decision rule (pre-declared): sign tests on D, Q and I with CI lower bounds against 0.10 margins → account.
  - Outputs: P(correct account) and FPR, for n_s ∈ {3, 5, 8} and items per cell ∈ {100, 200}.
- **Rule** (as D74): if any complete-gate FPR > 0.05, raise the offending margins in steps of 0.05, decided on synthetic data only, and freeze for Stage 2.
- **Output:** the minimal seeds × items reaching power ≥ 0.80, and its CPU-h cost from E4.

---

## 6. Outputs, decision point, integrity

**Outputs:**
- `results/raw/c16/s0_*.json`, `s1_*.json`;
- `experiments/c16/s0s1_report.md`.

**Decision point:** after Stage 1, **STOP and report to the PI**. No Stage-2 code path may train a workspace on the HF model. The guard allows only the E4 timing probe (no optimizer, and it asserts that no parameter changes).

**Seeds:**

| Seed(s) | Use |
|---|---|
| 16001 | Split |
| 16010 | Stage-0 sampling |
| 16011 | Halves |
| 16012 | Swap partners and NC draws |
| 16100 | S1a |
| 16150 | S1b |
| 16200 | Planted base model |
| 16211–16213 | Planted core arms |
| 16221–16223 | Planted pilot |

Stage-2 seeds stay undeclared until the Stage-2 preregistration.

**Integrity:**
- X_select / X_confirm and the eval names are sealed for S0/S1.
- Thresholds live in `c16_s0s1_thresholds_FROZEN.json`.
- The runner verifies the hashes of this file, the JSON and the manifest at start.
- Results are committed after each stage.
