# C15-R FINAL PRE-REGISTRATION DESIGN MEMO: routing otherwise-unused verification information into flexible control

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Responds to | PI approval of C15-R as the scientific direction, with 15 required revisions |
| Supersedes | `c15r_routing_rescue_design_memo.md` (D68) wherever they differ (§18 lists the changes) |
| Status | Final design for PI approval. **No code, no downloads, no model run** |
| What was read | Only public pages and metadata: Hugging Face API listings, lens `config.yaml` files, arXiv / GitHub / blog pages. The local CPU benchmark (`results/raw/feasibility/cpu_benchmark.json`) was re-read for compute figures |
| Claims discipline | Functional organisation only. No outcome of this design bears on phenomenal consciousness. The licensed claim levels are the PI's: **Level-1 mechanistic** and **Level-2 correspondence to workspace/metacognition theories** |

## 0. Verdict

- **The target construction principle can now be tested in a claim-identifiable way**, but only conditionally.
  - Two gates come before any hypothesis data: a task-independent **workspace assay** (A-stage) and a **claim-identifiability gate** (C-stage). Either can STOP the project before rescue data exist.
- **The binding risk is the workspace assay, not the rescue.**
  - Published evidence shows norm-matched J-space ablation to be *anti-selective* at every measured Qwen3 scale (selectivity index 3.38 at 1.7B, preregistered; 2.17 at 4B; 1.57 at 14B; 1.16 at 32B). J-ablation damages ordinary text more than a matched random control.
  - No CPU-feasible model has published evidence of a selective workspace.
  - So the model must be chosen **by assay**, from an inventory of five CPU-feasible, lens-equipped candidates. If none passes, C15 STOPS as a workspace paper.
- **Final novelty confidence: MEDIUM. Final significance confidence: MEDIUM.**
  - Significance stays MEDIUM for exactly the reason the PI anticipated: no small open model has a demonstrated selective workspace assay.
  - It can rise only if a candidate passes the A-stage. I do not lower the claim to protect the project.

---

## 1. Exact F definition

**Items: 3-SAT minimal pairs.**
- Each instance has n ∈ [5, 9] variables and m 3-clauses. The (n, m) ladder is set on development data so that checking competence clears E2.
- Each instance yields a valid assignment a⁺ and an invalid a⁻ that differs by one flipped variable violating ≥ 1 clause.
- Everything is generated locally from seeds, with an exact brute-force checker. There is no dataset download.
- Kawada & Kellis released no code or prompts, so this is a **re-creation** of their paradigm, not their exact computation. The F→C dissociation must be re-established (§5, MS2).

**Shared-prefix use computation (P-prefix), identical across all policies:**

```
[system] [Problem: find an assignment satisfying every clause. <formula>]
[A solver tool returned: <candidate>] ⟂  <- policy-independent site p_c
[policy segment π: instruction + response]
```

- Causal attention makes every activation at positions ≤ p_c **bitwise identical across policies**.
- F is therefore read, and routes are written, *before the model knows which control policy will apply*.
- p_c is chosen on development data from three pre-declared positions: last candidate token, the following newline, and the closing period.

**F (same-pass, primary):** F_i = wᵀ h_{ℓ_r, p_c, i}, where:
- w is an L2-logistic probe;
- the target is the model's own checking verdict y_i (see below);
- training uses development items only, with 5-fold instance-grouped CV; λ, ℓ_r and p_c are chosen by nested CV;
- the read layer ℓ_r comes from the band of workspace-validated layers (§4);
- the **write layer is ℓ_w = ℓ_r**, so the bridge (§8) is a rank-1 linear map at a single site, (ℓ_w, p_c).

**Model verdict (label only, never the F signal):**
- y_i = 1[V_i > 0], with V_i = logit P(INVALID) − logit P(VALID).
- V_i comes from a separate checking pass: formula + candidate, "Does this assignment satisfy every clause?"

**Frozen on development data:**
- direction w and its sign (oriented so that larger means "INVALID");
- read site (ℓ_r, p_c);
- probe type and λ;
- scaling μ_F, σ_F;
- content scalar **c_i = clip((F_i − μ_F)/σ_F, −3, 3)**;
- dev threshold τ_F;
- write code **v** = E[h | y = 1] − E[h | y = 0] at (ℓ_w, p_c), the class-mean difference on dev BASE states.

**Evaluation items never influence F.** w, v, μ, σ and τ_F are applied unchanged.

**Held-out validity test.** Computed on the evaluation split's BASE states *before* any rescue analysis:
- AUROC(F, y) ≥ 0.70 and AUROC(F, truth) ≥ 0.65.
- Otherwise rescue results are **not interpreted as routing of F** (stop ST-9).

**Reported separately, always:**
1. AUROC(F, model verdict);
2. AUROC(F, ground-truth validity);
3. the **error-following index**: on items where the verdict is wrong, AUROC(F, verdict) − AUROC(F, truth). Positive means F tracks the model's own verdict rather than surface validity.

**Terminology rule.**
- **TOOL arm:** F is called *internal verification information*. "Metacognitive" is reserved for the **SELF-GEN arm** (§9), where the candidate is the model's own earlier output and F is information about the model's own error.
- **F is never called metacognitive merely because it predicts the verbal verdict.**

**Strongest F-without-C population (pre-registered):**
- **D\*** = a⁻ items for which:
  - the verbal verdict is INVALID (correct);
  - F_i > τ_F (the internal verifier is correct);
  - the model nonetheless greedily **adopts** a⁻ at BASE under P0.
- Collateral set **V\*** = a⁺ items verified VALID by both verifiers and adopted at BASE.

## 2. Same-pass vs separate-pass decision (pre-declared, development data only)

**SAME-PASS is selected iff all five gates pass at a single frozen site:**

| Gate | Requirement |
|---|---|
| **F-a** predicts the verdict | Cross-fitted AUROC(F, y) ≥ 0.75 |
| **F-b** predicts validity | AUROC(F, truth) ≥ 0.70 |
| **F-c** present before the decision | Satisfied by construction: the read site p_c precedes every policy token |
| **F-d** local manipulation is weak | NATIVE steering (TRUE content written along v at κ\*) changes P0 AUROC by ≤ 0.05. **NATIVE is the only route intervention run on development data**; it is used for this gate and nothing else |
| **F-e** behaviour contradicts F often | \|D\*\| ≥ 25% of a⁻ items. On evaluation, D\* must contain ≥ 80 items |

**Otherwise:**
- If the checking-pass log-odds V_i satisfies F-a, F-b and F-e, use **SEPARATE-PASS**. Here F_i = V_i, standardised on dev.
- **The claim is then weakened throughout** to "cross-process metacognitive communication", not intrinsic within-computation routing.
- If both fail → STOP (ST-6).

**Supporting precedent.** Kawada & Kellis define their decodable-but-inert verification coordinate on use-trial residual states, H_ℓ = h_ℓ(x, e). This was read via a summarising fetch of the arXiv HTML; positions and decodability values are not stated there. A same-pass F is therefore expected to exist, but it must be shown here.

## 3. Exact task-independent S_J construction (fixed before any SAT item exists)

**Inputs:**
- the published lens J_ℓ;
- the unembedding W_U;
- generic material **G**, disjoint from C15, which contains no SAT, logic-verification or validity content:
  - preferred: the WikiText-103 validation split. Small; **requires download approval**; it matches the lens's own fitting distribution;
  - fallback: 2,000 × 64-token continuations generated by the intact model from 200 author-written, topic-diverse seed prompts (no download).

**Generic dictionary.**
- V_gen = word-initial tokens of ≥ 2 letters, minus the 200 most frequent tokens in G. Fixed from the tokenizer and G alone.
- Atoms ĵ_{ℓ,t} = normalise(J_ℓᵀ W_Uᵀ e_t), with the final-norm convention checked at Z0.

**Sparse workspace projection (Gurnee et al.):** Π_J(x) = GP₂₅(x), non-negative gradient pursuit over the V_gen atoms, k = 25.

**Primary workspace subspace.**
- Sample 5,000 residual states per candidate layer from G (all positions).
- Compute their J-reconstructions Π_J(h), centre them, and run PCA.
- **S_J(ℓ)** = span of the top r_ℓ principal components, with **r_ℓ = min(r₉₀, ⌊d/8⌋)**. Here r₉₀ is the number of components explaining 90% of the variance of the generic J-reconstructions.
- P_J = QQᵀ and P_⊥ = I − P_J.
- S_J is hashed and committed before the first SAT item is generated.

**Task enters only as content to be projected, never as a criterion for which directions count as workspace.**
- **Workspace route:** û_J = P_J v / ‖P_J v‖.
- Reliability gate: split-half cos(P_J v⁽¹⁾, P_J v⁽²⁾) ≥ 0.7 on dev.
- The fraction ‖P_J v‖/‖v‖ is reported against its chance value √(r/d). This is descriptive: native F's workspace share.

**Secondary analyses only:**
- the sparse route normalise(Π_J(v)), which is Gurnee-faithful;
- verdict-word atoms, used solely for lens readouts in the broadcast assay.

## 4. Workspace-assay gates (A-stage: generic material only; before any SAT data; per candidate model)

All interventions use the same rank-1, layer, position and norm conventions as C15. Each J condition is compared with a **matched non-workspace control** built by the §6 procedure (the task-independent rows: norm, propagation, generic KL, lens gain), applied to the generic content.

| Gate | Property (PI list) | Test (author-written generic material) | Pass criterion |
|---|---|---|---|
| **W0** | Lens prerequisite | (a) Late-layer lens top-1 agrees with the model's next token. (b) Intermediate entities of 100 generic two-hop prompts appear in the lens readout | (a) ≥ 0.60 at L − 2. (b) Intermediate in top-20 at some band layer for ≥ 40% of the prompts the model answers correctly. Baseline two-hop accuracy ≥ 0.60, else W4 is not assessable (= fail) |
| **W1** | Reportability | Inject the J component (vs the matched ⊥ component) of 40 concrete-concept vectors at a neutral position, then a free-association prompt; 10-way forced choice among concept words | J top-1 ≥ 0.30 (chance 0.10) and ≥ 2 × the ⊥ rate |
| **W2** | Broad downstream accessibility (cross-function) | Swap the J component of a country concept for another's (vs the matched ⊥ swap) at the subject token; three downstream queries (capital, language, currency) | Consistent switch on ≥ 2/3 functions for ≥ 25% of 40 pairs, and ≥ 2 × the ⊥ rate |
| **W3** | Cross-position influence | Content injected at p is decoded at positions p + 1 … p + 20 and at later layers (cross-fitted ridge decoders) | Breadth BB_J − BB_⊥ ≥ 0.15, with CI > 0 |
| **W4** | Task-independent causal relevance | Per-position ablation of the top-10 active J atoms at the band layers vs a norm-matched random ablation; generic two-hop accuracy | J impairment − random impairment ≥ 10 points (jspace-4b reported 16 vs 0 at Qwen3-4B) |
| **W5** | Selectivity vs matched controls | Same ablation; ordinary next-token change on G | **SI_iso ≤ 1.0, CI upper ≤ 1.25.** SI_iso compares against a random ablation *rescaled to equal the J ablation's two-hop impairment*. SI_norm (the published norm-matched index) is always reported |

**Why SI_iso, and the PI decision it implies.**
- A more causally potent subspace damages more at equal norm, so SI_norm conflates potency with selectivity. SI_iso asks whether, *at equal functional impact on workspace-demanding computation*, the J ablation spares automatic processing better than a random one.
- **It is the less strict of the two criteria. That is stated plainly.**
- If the PI prefers SI_norm ≤ 1, then Qwen3-1.7B and Qwen3-4B fail on published data (3.38 and 2.17) and drop from the inventory. If any model passes with SI_iso ≤ 1 but SI_norm > 1, the paper must say "selective at matched impact, anti-selective at matched norm".

**Layer rule.** Gates are evaluated on every layer of the band [0.3 L, 0.6 L] (step 2). Only layers passing W1–W5 are eligible for ℓ_r and ℓ_w.

**The "workspace-like" label requires W0–W5.**
- If W0–W4 pass but W5 fails, the route may be described only as a "lens-verbalisable route". C15 then STOPS as a workspace paper (ST-2) and returns to the PI with a reframing option (generic routing/control). Nothing continues automatically.

## 5. Model-selection gates (assay-driven; no rescue data)

### Candidate inventory (CPU-feasible: fp32 RAM ≤ about 20 GB; published lens on the exact checkpoint)

| Candidate | Licence / gate | Weights (bytes) | Lens file (bytes) | Lens fit | Prior workspace evidence |
|---|---|---|---|---|---|
| Qwen3-1.7B | Apache-2.0, ungated | 4,074,938,246 | 226,501,315 | Qwen/Qwen3-1.7B, 466 prompts | Anti-selective, SI_norm 3.38 (preregistered) |
| Qwen3.5-2B | Apache-2.0, ungated | 4,548,221,488 | 192,946,315 | Qwen/Qwen3.5-2B, 283 prompts | None. Hybrid linear/full attention; multimodal wrapper |
| Qwen3-4B | Apache-2.0, ungated | 8,045,181,920 | 458,762,977 | Qwen/Qwen3-4B, 479 prompts | SI_norm 2.17 (exploratory). Two-hop −16 vs 0 under J ablation. In Kawada's set |
| Qwen3.5-4B | Apache-2.0, ungated | 9,319,828,096 | 406,333,179 | Qwen/Qwen3.5-4B, 417 prompts | None. Hybrid; multimodal wrapper |
| Gemma-3-4b-it | Gemma licence, **manual gate (PI)** | 8,600,277,880 | 432,548,236 | google/gemma-3-4b-it, 546 prompts | None. Lens "identity distance" 0.96 vs 0.39–0.64 for the others (uninterpreted; W0 must check) |

**Excluded, with reasons:**
- **Llama-3.1-8B-Instruct** (Kawada's cleanest F-without-C). fp32 needs about 32 GB, above the 31 GB of RAM. bf16 runs 4.5× slower on this CPU (measured: 2.21 vs 0.49 s per 64-token prompt at 0.5B), so the H-stage would take about 200–400 h. **Optional reduced replication only.**
- **Gemma-4-E4B.** Base-only lens; about 32 GB fp32.
- **Sub-1B models.** Unlikely to clear checking competence E2.

### Gates

| Gate | Requirement |
|---|---|
| **MS0** | Z0 lens/convention check passes; projected H-stage wall-clock ≤ 96 h (measured throughput) |
| **MS1** | A-stage W0–W5 pass at ≥ 1 band layer (§4) |
| **MS2** | F exists, and the F→C dissociation exists: F-a, F-b, F-d, F-e (§2); P0 BASE AUROC ≤ 0.65; greedy adoption of a⁻ ≥ 0.60 |
| **MS3** | Endpoint non-degenerate: parse ≥ 95%; checking competence AUROC(V, truth) ≥ 0.85 (E2). **Verbal-routing reachability:** inserting the model's own verdict text at p_c lifts P0 AUROC to ≥ 0.70. At least two of the unseen policies P1–P3 (§9) are reachable by verbal routing (AUROC ≥ 0.70, correct sign), **including P3** |
| **MS4** | Route constructibility: §6 matching feasible for ⊥T1 and J-irr |

**Order.**
1. A-stage on all permitted candidates. It is cheap, generic-only and needs no SAT data.
2. B-stage (MS2–MS3) only for A-passers.
3. C-stage (MS4 and §12) only for B-passers.

**Choice among passers.** The highest minimum normalised margin across W1–W5 (the workspace assay is the bottleneck). Ties go to the larger \|D\*\| fraction, then to lower compute. A second passer may serve as a pre-registered replication.

**No rescue outcome on any item enters selection.**

## 6. Matched non-workspace route construction (predeclared; no C15 behavioural outcomes)

| Matching dimension | How it is matched |
|---|---|
| Perturbation norm, layer, position, dimensionality (rank 1) | Identical by construction: the same κ, ℓ_w, p_c and a unit vector |
| Downstream propagation magnitude | ‖Δh_{ℓ′}‖/‖Δ‖ at ℓ′ = ℓ_w + ⌈0.1L⌉, ℓ_w + ⌈0.2L⌉ and L − 1, same and later positions, on G and dev BASE states: within ±15% of û_J |
| Generic output disruption | Next-token KL on G at κ\*: within ±15% |
| Output-head (lens) gain | ‖W_U J_{ℓ_w} u‖: within ±15%. Removes "J simply has more leverage over the output head" |
| **First-order gain toward the decision output** (⊥T2 only) | ḡᵀu = ḡᵀû_J ± 10%, where ḡ = E_dev[∇_{h_{ℓ_w,p_c}} A_i^{P0}], the BASE adoption log-likelihood gradient |

**Construction.** u is restricted to the family:

> u(θ) = normalise( cos θ₁ · ê_v⊥ + sin θ₁ (cos θ₂ · ê_g⊥ + sin θ₂ · ê_h⊥) ),  u ⊥ S_J

- ê_v⊥: F's own non-workspace code, P_⊥ v.
- ê_g⊥: the orthogonalised P_⊥ ḡ.
- ê_h⊥: the top right-singular direction of P_⊥-restricted W_U J_{ℓ_w} (high lens gain).
- Grid-search θ (0.05π resolution) and keep the feasible u **closest to ê_v⊥**, i.e. as close as possible to "F's own non-workspace code".
- ⊥T1 meets the first four rows; ⊥T2 adds the decision-gain row.
- Verbalisability check: GP₂₅ fidelity of u inside the random-direction 95% band.

**J-irr (workspace, irrelevant content).** Among the J components of the 40 W1 concept vectors (generic, verbalisable), take the three that are best matched to û_J on rows 2–4 and orthogonal to P_J v within 0.2 cosine. Results are averaged over the three.

**Infeasibility.** If no u ⊥ S_J meets the lens-gain and KL tolerances, the workspace route cannot be separated from output-head leverage. That is **ST-4 (STOP)**. The same applies if J-irr cannot be matched.

**Why ⊥T1 (not ⊥T2) is the primary control on P0.**
- With matched decision gain, a first-order (linear) response gives TRUE-⊥T2 ≈ TRUE-J on P0 *whether or not the workspace is special*. A P0 contrast against ⊥T2 tests only effects beyond first order.
- Its decisive use is as the **H3 model** in the transfer and broadcast tests (§7, §9): a direction matched to J in leverage over the *calibration* controller.
- The PI's concern, that "workspace works better" could reduce to leverage over the output head, is addressed by three things together:
  - lens-gain matching (⊥T1);
  - decision-gain matching (⊥T2);
  - LATE (§10).

## 7. Broadcast / propagation assay (pre-specified signature)

- **Injected-content decodability under PERM.** Under PERM, the injected scalar c_{π(i)} is independent of the item's native F. Any downstream decodability of it therefore measures **transport of the injected signal**, uncontaminated by native verification.
- **Sites, per policy π ∈ {P0, P1, P3}:**
  - positions: every third policy-instruction token, plus the decision slot;
  - layers: ℓ_w + 2, ℓ_w + 4, …, L − 2.
- **Readout:** cross-fitted (5-fold, instance-grouped) ridge decoding of c_{π(i)} from residual states under PERM-R, for R ∈ {J, ⊥T1, ⊥T2}.
- **Breadth:** BB_R = the fraction of sites with R² ≥ 0.25.
- **H1 broadcast signature (both required):**
  1. BB_J − BB_⊥T1 ≥ 0.15 and BB_J − BB_⊥T2 ≥ 0.15, each with 95% CI > 0;
  2. J-injected content decodable (R² ≥ 0.25) at the decision slot of ≥ 2 policies.
- **Interpretation.** Propagation magnitude is matched in the residual stream at p_c, so a breadth difference reflects cross-position, cross-computation **information transport**, not perturbation size.
- **Secondary readouts:**
  - the TRUE-content version (native + injected F decodability vs BASE);
  - the J-lens verdict-word readout at downstream sites (the only place verdict words enter).

## 8. Primary content × route estimand (P0, evaluation split, everything frozen)

**Conditions.** κ = a\* · ‖h̄_{ℓ_w}‖, with a\* = the largest a ∈ {0.02, 0.04, 0.08, 0.16, 0.32} at which CONST-J keeps the neutral KL ≤ 0.05 nats/token and next-token agreement ≥ 95% (chosen on G only). The write at (ℓ_w, p_c) is h ← h + κ · g_i · û_R, with:
- **content g_i:** TRUE c_i; PERM c_{π(i)}; FLIP −c_i; CONST 1;
- **route R:** J, ⊥T1, ⊥T2, J-irr, NATIVE (v), LATE (the same û_J construction at ℓ_late ≥ 0.9 L, which is after integration; C-stage prerequisite 3).

**DV.** The adoption propensity A_i: the mean teacher-forced log-probability of the candidate assignment after "Give your final assignment". Discrimination = paired minimal-pair AUROC, P(A(a⁺) > A(a⁻)).

**Primary estimand (the single primary test):**

> **I = [AUC(TRUE, J) − AUC(PERM, J)] − [AUC(TRUE, ⊥T1) − AUC(PERM, ⊥T1)]**

- Instance-clustered bootstrap with 10,000 resamples; one-sided.
- **Pass: I ≥ 0.10 and the lower 95% bound > 0.**

**Power.** Simulated in the C-stage from dev BASE variances (§12). At 400 evaluation instances (800 candidates) the rough standard error of I is ≈ 0.03.

## 9. Secondary flexible-control policies (intervention unchanged; only the instructed policy changes)

All policies read the **identical** residual stream at p_c, including the identical routed write. Option labels are two arbitrary letters, counterbalanced across item halves, which removes lexical leverage.

| Policy | Instruction (abbreviated) | Policy-correct response to an INVALID candidate | Role |
|---|---|---|---|
| **P0** | Give your final assignment (evidence integration) | Do not adopt a⁻ | Primary. **Calibration policy**: only ⊥T2 gain-matching uses it |
| **P1** | SUBMIT the tool's assignment (+1 if valid / −2 if invalid) or DISCARD (0) | DISCARD | Unseen; same mapping |
| **P2** | ANSWER with it (+1 / −6) or PASS (0) | PASS | Unseen; payoff (threshold) shift. Exploratory |
| **P3** | We are collecting FAULTY outputs for a test suite: KEEP it if it violates any clause, else SKIP | **KEEP** | Unseen; **reversed mapping**. The decisive H1-vs-H3 test |

**Flexible-control estimand (antisymmetric, so it works even at floor or ceiling).**
- Δ_i^π(R) = D_i^π(TRUE, R) − D_i^π(FLIP, R).
- This cancels every sign-independent effect of the write (disruption, generic bias), leaving only the content-sign effect.
- **T^π(R)** = the policy-signed OLS slope of Δ_i^π on c_i. Positive means the response moves toward the policy-correct response for the broadcast verdict.

**Unseen-policy transfer test.** Route, layer, dose and matching are frozen on G, dev and P0. Then, on P1 and P3:

> T^π(J) − T^π(⊥T1) > 0 and T^π(J) − T^π(⊥T2) > 0, and **T^{P3}(J) > 0** (correct sign under the reversed mapping)

- A direction that steers an *action* ("reject this") predicts the wrong sign under P3.
- *Information* that the controller applies according to the instruction ("this is invalid") predicts the right sign.

**SELF-GEN arm (pre-registered secondary).**
- 200 evaluation instances. The candidate is the model's **own** earlier greedy answer from a no-candidate pass, presented as "Your earlier answer: …" (natural valid and invalid outputs, so no minimal pairs).
- Conditions: TRUE/PERM × J/⊥T1 on P0.
- This is the only arm that can license "one's own error" or "metacognitive" wording.

## 10. Rescue criteria and claim tiers

| # | Criterion (evaluation split, frozen) |
|---|---|
| R1 | Primary I passes (§8) |
| R2 | AUC(TRUE, J) − AUC(PERM, J) ≥ 0.10 and AUC(TRUE, J) ≥ 0.70. D\* rescue: P(not adopted \| D\*, TRUE-J) − P(not adopted \| D\*, TRUE-⊥T1) > 0 (paired, CI). Collateral on V\* ≤ 0.05 |
| R3 | FLIP-J lowers AUC by ≥ 0.05 below BASE. The behaviour follows the broadcast content in both directions |
| R4 | CONST-J: \|ΔAUC\| ≤ 0.03. A rate shift is allowed; that is bias, not information |
| R5 | J-irr(TRUE) content gain ≤ 1/3 of J's |
| R6 | LATE(TRUE-J) gain ≤ 1/3 of J's. The effect needs downstream computation, not a direct output path |
| R7 | Integrity under TRUE-J: no-candidate answer validity within 2 pp; parse ≥ 95% |
| R8 | ⊥T2 tier: I computed with ⊥T2 in place of ⊥T1. Positive means an advantage beyond first-order decision leverage |
| R9 | Broadcast signature (§7) |
| R10 | Unseen-policy transfer (§9), including the correct sign on P3 |

The secondary family R2–R10 is Holm-corrected.

**Claim tiers (most to least).**

| Tier | Evidence required | Licensed statement |
|---|---|---|
| **STRONG** | MS1 (W0–W5) + SAME-PASS F + held-out F validity + R1–R7 + R9 + R10 | "A local verification signal became flexibly behaviourally usable when given access to a workspace-like routing mechanism." A significant Level-1 mechanistic result with a Level-2 correspondence to workspace/metacognition theories. **Not consciousness.** "Metacognitive / own error" wording only if SELF-GEN replicates R1–R2 |
| ROUTE-PRIVILEGED | R1–R7 without R9 or R10 | "A particular internal subspace is privileged for transmitting verification information to control" (route-specific controller sensitivity, not broadcast) |
| LEVERAGE-ONLY | R1 passes; R8 ≤ 0 and R10 fails | The J advantage is explained by first-order leverage over the calibration controller (H3) |
| ANY-ROUTE | TRUE − PERM > 0 on both J and ⊥T1, but R1 fails | "Routing/connectivity matters, but J-space is not special" |
| GENERIC | R3, R4, R5 or R6 fails | No metacognitive integration claim |
| NULL | Validated assay; TRUE − PERM ≈ 0 on every route; MS3 verbal routing works | "Verification-guided control in this model is not enabled by this candidate workspace route" |
| (qualifier) | SEPARATE-PASS F | Every statement becomes "cross-process metacognitive communication" |

**Target sentence, clause by clause.** It is used **only** if every clause is supported:

| Clause | Required evidence |
|---|---|
| "causally rescue" | R1, R2 |
| "otherwise-unused" | MS2: F-d, F-e, D\* |
| "internal error information" | SAME-PASS F + held-out validity; "**own** error" requires SELF-GEN |
| "workspace-like" | MS1 (W0–W5) |
| "broadcast pathway" | R9 |
| "representing … is insufficient … unless … globally usable" | F-without-C + R10 (multi-policy, including reversed) |

## 11. Necessity criteria (secondary; not forced)

**Operator.** Rank-1 LEACE restricted to S_J (Belrose et al. 2023), at ℓ_e … ℓ_e + 2, positions from candidate onset to p_c:

> E_S(h) = h − Q W⁺ b bᵀ W (Qᵀh − μ)

**Comparison erasures, all with matched dimensionality r:**
- ERASE-⊥ inside the top-r principal subspace of P_⊥ h on dev, where F is also decodable before erasure;
- SHUF-J and SHUF-⊥, fitted to permuted labels;
- ERASE-ALL, the positive control.

**Cell.** A *naturally coupled* policy: BASE AUROC ≥ 0.70, probably P1. This is within the same model; base vs post-trained is **not** used as a controller factor.

**Counts as necessity only if every condition holds** (validated on dev before any evaluation readout):

| Condition | Requirement |
|---|---|
| F stays decodable outside S_J | AUROC from the full h′ ≥ 0.9 × pre. Linear AUROC from Qᵀh′ ≤ 0.55 (quadratic-probe AUROC reported); no re-entry ≥ 0.60 up to 0.85 L |
| Generic task behaviour intact | Next-token agreement on G ≥ 95%; generic two-hop within 3 pp |
| First-order validity reasoning intact | No-candidate answer validity within 2 pp; F decodable at downstream ⊥ sites |
| Unrelated workspace contents intact | Jaccard ≥ 0.8 of non-verdict active atoms at the decision slot; W1 report of unrelated concepts within 5 pp |
| Adaptive control selectively falls | N = [ΔAUC(ERASE-J) − ΔAUC(SHUF-J)] − [ΔAUC(ERASE-⊥) − ΔAUC(SHUF-⊥)] ≤ −0.10 (CI); ERASE-J removes ≥ 50% of the natural excess AUROC; ERASE-ALL removes it |

**If any validation condition fails, necessity is reported only as "not identifiable".** A clean rescue outranks a forced ablation.

## 12. Claim-identifiability gate (C-stage; before evaluation items exist)

**Hypotheses:**
- **H1** content-specific workspace routing;
- **H2** generic activation steering (a fixed direction-to-output map, available through any route of comparable leverage);
- **H3** direct output-controller access (leverage specific to one decision output);
- **H4** generic broadcast / global perturbation (any workspace perturbation affects behaviour regardless of content).

| Contrast | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| a. TRUE − PERM on J (P0) | + | + | + | ≈ 0 |
| b. Primary I (J vs ⊥T1) | + | ≈ 0 | + | ≈ 0 |
| c. J vs ⊥T2 on P0 (R8) | ≥ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| d. Transfer to P1: T(J) − T(⊥T2) | + | ≈ 0 | ≈ 0 or − | ≈ 0 |
| e. **Reversed P3: sign of T(J)** | **+** | fixed by output | **− or 0** | 0 |
| f. J-irr vs J | J ≫ J-irr | depends on leverage | depends | J-irr ≈ J |
| g. LATE vs J | LATE ≈ 0 | LATE > 0 | LATE > 0 | ? |
| h. Broadcast BB_J − BB_⊥T2 | + | 0 | 0 | + (content-nonspecific) |

- H1 vs H2: b, d, e.
- H1 vs H3: e, g, h, d.
- H1 vs H4: a, f and FLIP (R3).

**Gate passes only if all of the following hold** (development data plus simulation; no evaluation items):
1. ⊥T1, ⊥T2 and J-irr are constructible (§6).
2. P3 is reachable by verbal routing (MS3). This is the only clean H1-vs-H3 sign test.
3. A post-integration LATE layer exists: the P0 decision is ≥ 90% decodable from layer ≥ 0.9 L.
4. Simulated power ≥ 0.80, using dev BASE variances of A, D and c, for:
   - the primary I at Δ = 0.10;
   - contrasts d and e at a slope of 0.5 × the verbal-routing effect;
   - contrast h at a breadth difference of 0.15.

**If any fails → STOP C15 (ST-8).** This is a claim-identifiability failure, not a statistical one.

## 13. Exact stop conditions

| Code | Stage | Condition | Action |
|---|---|---|---|
| ST-0 | Z0 | Lens convention check fails (late-layer lens/logit mismatch), or the model cannot load in fp32 | Candidate excluded |
| ST-1 | A | Candidate fails any of W0–W5 at every band layer | Candidate excluded |
| **ST-2** | A | **No candidate passes A** | **STOP C15 as a workspace paper.** Report to the PI with an optional reframing (generic routing/control). Nothing continues automatically |
| ST-3 | B | No A-passer clears MS2–MS3 | STOP |
| ST-4 | C | Matched ⊥T1 / ⊥T2 / J-irr infeasible within tolerance | STOP (route contrast confounded with output leverage) |
| ST-5 | C | S_J reliability gate (split-half < 0.7) fails | STOP |
| ST-6 | B | Neither SAME-PASS nor SEPARATE-PASS F qualifies | STOP |
| ST-7 | C | Projected H-stage > 96 h, and power < 0.80 at the minimum N of 300 instances | STOP |
| ST-8 | C | Claim-identifiability gate fails | STOP |
| ST-9 | H, analysis step 1 | Held-out F validity fails | No routing interpretation; report as such |
| ST-10 | H | Parse < 95% or competence collapse under BASE | Run invalid; report |

**Rule.** No threshold changes after any stage result. Every deviation is logged before any further analysis. Evaluation items are generated only after FREEZE.

**Seeds** (proposed; checked against every used range):
- generic material G: 9300;
- development SAT: 9301–9303;
- evaluation SAT: 9311–9313;
- SELF-GEN: from evaluation instances.

## 14. Compute estimate per candidate (fp32; this CPU)

**Basis.**
- Measured: Qwen2.5-0.5B fp32 at 0.49 s per 64-token prompt (batch 1), 0.34 s batched. That is about 130–190 GFLOP/s. bf16 is 4.5× slower, so fp32 is assumed throughout.
- Projection uses 130–250 GFLOP/s at about 2 × params FLOPs per token.
- **Z0 measures the real throughput** and re-projects. ST-7 applies.

**Workload, in token-forward equivalents.**
- **A-stage:** about 0.25 M, plus gradient pursuit (≈ 0.1–0.3 PFLOP).
- **B-stage:** 600 dev candidates × (check, prefix, 3 policies BASE + verbal routing, NATIVE dose) plus P0 gradients ≈ 1.0 M.
- **C-stage:** route matching on G and dev plus simulations ≈ 0.15 M.
- **H-stage:** 800 evaluation candidates × (check 250 + prefix 250 + 15 conditions × 3 policies × ≈ 50 + necessity ≈ 550) ≈ 2.6 M, plus SELF-GEN ≈ 0.2 M.

| Candidate | fp32 RAM | Tokens/s (est.) | A | B | C | H | Total if selected |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | ≈ 8 GB | 32–62 | 2–3 h | 4–9 h | 1–2 h | 13–25 h | **≈ 20–39 h** |
| Qwen3.5-2B | ≈ 9 GB | ≈ 30–60 (linear-attention CPU kernels unverified) | 2–3 h | 5–9 h | 1–2 h | 13–26 h | ≈ 21–40 h (+ risk) |
| Qwen3-4B | ≈ 16 GB | 16–31 | 3–6 h | 9–17 h | 1–3 h | 25–49 h | **≈ 38–75 h** |
| Qwen3.5-4B | ≈ 19 GB | 15–30 | 3–6 h | 9–19 h | 1–3 h | 26–52 h | ≈ 39–80 h |
| Gemma-3-4b-it | ≈ 17 GB | 17–32 (262k-vocabulary unembedding) | 3–6 h | 9–16 h | 1–3 h | 24–46 h | ≈ 37–71 h |
| Llama-3.1-8B-it (excluded) | 32 GB fp32 ✗ / bf16 | ≈ 2–4 (bf16) | ≈ 20–40 h | — | — | ≈ 200–400 h | Infeasible; optional reduced P0 2×2 replication ≈ 20–40 h |

**A-stage for all five candidates:** about 13–24 h CPU. **Disk:** 521 GB free.

**Downloads that need approval (not requested yet):**
- **A-stage:** all five candidates plus lens folders, about 34.6 GB of weights + 1.72 GB of lenses. Gemma requires the PI to accept its licence.
- Optional: the WikiText-103 validation split.
- **Suggested minimum first step:** A-stage on Qwen3-1.7B, Qwen3.5-2B and Qwen3-4B (≈ 16.7 GB + 0.88 GB).

## 15. Final novelty confidence: **MEDIUM**

**Supported:**
- No paper read (full text or primary page) performs route-specific rescue, route-specific necessity, or **policy-flexible use of routed verification information**.
- The policy-reversal transfer test and the PERM-transport broadcast assay are, to my knowledge, new.

**Neighbours:**
- Kawada & Kellis: F ⇏ C, with no rescue.
- Gurnee et al.: J vs non-J *concept* swaps.
- Kumaran et al., VUF, TRAPSBench, VerifySteer and Li et al.: generic sufficiency.
- jspace-4b / jspace-validity: non-selective ablation.

**Why not higher:**
- the risk of a direct Kawada & Kellis follow-up;
- the active J-lens community;
- Cohen & de Melo is still NOVELTY UNRESOLVED.

## 16. Final significance confidence: **MEDIUM**

- **If the STRONG tier is reached,** the result would be a significant Level-1 mechanistic finding with a Level-2 theory correspondence, and a construction principle (monitor → broadcast route → flexible control). The design is now claim-identifiable *conditional on* the C-stage prerequisites.
- **Confidence remains MEDIUM, as the PI anticipated, because no small open model has a demonstrated selective workspace assay.**
  - Qwen3-1.7B and Qwen3-4B are anti-selective on the published norm-matched criterion.
  - Qwen3.5-2B/4B and Gemma-3-4b-it are untested.
  - Under SI_iso the outcome is unknown.
- **It is more likely than not** that either:
  - (i) no candidate passes W0–W5 (ST-2: STOP); or
  - (ii) a candidate passes only under SI_iso, with an obligatory "anti-selective at matched norm" qualifier.
- **Path to HIGH:** a candidate passes A-stage with margin, and the B/C prerequisites hold. That would be judged *before* any evaluation item exists.

## 17. PI decisions required (nothing is executed until then)

1. **Approve this design**, in particular:
   - the shared-prefix SAME-PASS F;
   - task-independent S_J (generic-PCA of J-reconstructions);
   - **W5 criterion: SI_iso, or SI_norm as well** (which excludes Qwen3-1.7B and Qwen3-4B on published data);
   - ⊥T1 as the primary control with ⊥T2 for transfer and broadcast;
   - P3 as the decisive reversed-policy test;
   - the SELF-GEN arm as the only route to "own error" wording.
2. **Download approval for the A-stage**, from huggingface.co, staged. Minimum:
   - Qwen/Qwen3-1.7B: 4,074,938,246 B + lens 226,501,315 B;
   - Qwen/Qwen3.5-2B: 4,548,221,488 B + lens 192,946,315 B;
   - Qwen/Qwen3-4B: 8,045,181,920 B + lens 458,762,977 B.

   Optional: Qwen3.5-4B (9,319,828,096 B + 406,333,179 B); Gemma-3-4b-it (8,600,277,880 B + 432,548,236 B, needs the PI's licence acceptance); the WikiText-103 validation split for G.
3. Thresholds marked in §4 (W1–W5) and §2 (F-a to F-e) are proposals. Changing them is free **now**, never after A-stage data.

## 18. Changes vs `c15r_routing_rescue_design_memo.md` (D68)

| Area | D68 | Now |
|---|---|---|
| F | Decision slot | **Same-pass** at a policy-independent site before any policy token; checker only as a fallback with a weakened claim. Held-out validity; verdict vs truth reported separately; D\* population |
| S_J | Task-dependent (active atoms on SAT items + verdict atoms) | **Generic-PCA of J-reconstructions**; verdict words secondary only |
| Workspace label | Assumed from M1–M4 | Requires **W0–W5**, including selectivity; Qwen3 not assumed |
| Model | Fixed (Qwen3-4B) | **Assay-driven selection** from five candidates |
| ⊥ route | Matched on norm and propagation | Also generic KL, lens gain, rank and decision gain (⊥T2), via a predeclared family |
| Broadcast | Not specified | **PERM-transport broadcast assay** |
| Endpoints | One consumer (C1, plus C2) | **Flexible control**: four policies incl. reversed P3; unseen-policy transfer; antisymmetric estimand |
| Controller factor | — | Base vs post-trained dropped; the within-model policy is the controller manipulation |
| Identifiability | — | **Claim-identifiability gate H1–H4** |
| "Own error" wording | — | Requires the SELF-GEN arm |
