# C15-R DESIGN MEMO: causal routing / rescue of metacognitive information into adaptive control

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Responds to | PI provisional approval of C15 and request for a C15-R strengthening memo before preregistration |
| Builds on | `novelty_significance_decision_memo_v4.md` (C15). This memo **replaces v4 §6–§10** (design). Changes: endpoint priority is switched (candidate rejection is now primary), and necessity is redefined (no broad J-space ablation) |
| Status | Design memo for PI decision. **No code written, no artifact downloaded, no model run** |
| What was read | Only public pages and metadata: Hugging Face API file listings, three lens `config.yaml` files (≈ 3 KB each), arXiv / transformer-circuits / GitHub pages |
| Claims discipline | Everything below concerns **functional** organisation: verification information, access-like broadcast, adaptive control. **Nothing bears on phenomenal consciousness, sentience or subjective experience** |

---

## 0. Verdict

1. **C15-R can be written down exactly.**
   - **F** is the model's own item-level verification verdict, read out of the decision-slot residual during the *use* pass.
   - **A** is a lens-defined linear workspace subspace S_J(ℓ). The *workspace route* is the S_J component of the model's own verdict code. The *non-workspace route* is the orthogonal component of the **same** code vector.
   - **C** is adoption versus rejection of an external candidate in the Kawada & Kellis use format, scored on valid/invalid **minimal pairs**.
2. **Content and route are separated by construction.** The intervention is a **rank-1 bridge** Δh = κ · û · (wᵀh − μ)/σ.
   - *What is read* (w; or its item-permuted values) is the content factor.
   - *Where it is written* (û_J or û_⊥) is the route factor.
   - Norms and the marginal distribution of the perturbation are identical across routes.
3. **Route-specific necessity without broad J destruction is possible in principle.**
   - The tool is **rank-1 LEACE erasure restricted to S_J**: closed-form and minimal-change. It leaves the orthogonal complement, where F must stay decodable, untouched.
   - Contrast: the published top-10-per-position J ablation changed 34–42% of next-token predictions at 4B.
   - The eraser is validated on development data before any hypothesis data. **If validation fails, necessity stays secondary**, as the PI specified.
4. **An honest limit, stated up front.** In the linear-response regime, "routing" and "item-signed steering along a direction the controller reads" are *the same operation*.
   - So the scientific claim cannot be about a new operation. It is about **which subspace the model's controller reads** (controller gain per unit of generic perturbation).
   - It is established only by a content × route interaction plus semantic, direct-path and disruption controls (§6, §10, §11).
5. **Models.**
   - **Cleanest documented F-without-C:** Llama-3.1-8B. Kawada & Kellis report a null steering slope (−0.045) and place the arbitration state outside the workspace. It is gated (PI licence) and slow on CPU.
   - **Smallest decisive experiment:** **Qwen3-4B** on locally generated 3-SAT. It is ungated, its lens was fitted on the identical checkpoint, and it is in Kawada's model set.
   - **No public model has documented *natural* F→A→C coupling on this task.** All twelve of Kawada's models adopt invalid SAT assignments 93–100% of the time. A natural-coupling cell must therefore be *found*: within-model (explicit-decision consumer or arithmetic), or cross-model in Gemma-4-E4B, whose verification direction is causally effective.
6. **Novelty is still supported (MEDIUM).** No paper read in full or at primary-page level performs route-specific rescue or route-specific necessity for metacognitive control. Gurnee et al.'s J vs non-J concept-swap contrast is a methodological precedent for *concept* content, not for control.
7. **Significance confidence stays MEDIUM**, as the PI required. This memo makes the five conditions *identifiable and pre-declared*. Only S0/S1 data can *demonstrate* them (§16).

---

## 1. The claim and the stricter ladder

**Target question (PI wording).**
- When an LLM has internal verification information that is decodable but behaviourally inert, is routing it through the workspace-like broadcast subspace sufficient to make it govern adaptive behaviour?
- Conversely, when such information naturally controls behaviour through workspace-like access, does selectively blocking that route abolish control while preserving the information?

**Causal chain:** first-order computation → **F** (local verification representation) → **A** (workspace access) → **C** (adaptive control).

**Not claimed**, because it is already published as generic sufficiency: "injecting confidence or verification signals changes abstention or verdicts". This covers Kumaran et al. 2026, the verbal-uncertainty feature 2025, TRAPSBench 2026, VerifySteer 2026 and Li et al. 2026.

**Claimed, if and only if the results support it:** *route-specific* sufficiency, and preferably *route-specific* necessity, of workspace access for metacognitive control.

| Level | Stricter definition used in C15-R |
|---|---|
| 2 | Route-specific disruption (ERASE-J, §7) impairs natural control while F stays decodable and the selectivity checks pass |
| 3 | Workspace routing (TRUE-J) rescues control in an F-without-C cell and passes all of the genuine-rescue criteria R1–R7 (§10) |
| 4 | The **results** establish the content × route interaction for sufficiency, and preferably the matching necessity interaction or block-and-rescue in a coupling cell (§7.4). Replicated on ≥ 2 prompt templates. A second model is required before any claim of generality |
| 5 | Developmental: training-time monitor→workspace routing creates durable control (C12′). **Not in scope** |

**Scope note: what kind of "metacognition" this is.**
- The Kawada & Kellis paradigm monitors the *validity of a content the model is about to use* (an external candidate).
- That is Shea & Frith's case: a metacognitive tag weighting a broadcast content.
- It is **not** introspection on the model's own knowledge. The self-confidence variant (ANSWER/PASS, v4 §10) remains a later generalisation test.

---

## 2. Q1. What exact representation counts as F?

**Items.**
- A 3-SAT instance x has n variables and m clauses, with n ≤ 10 so that a brute-force checker is exact.
- Each instance gets a valid assignment a⁺ and a **minimal-pair** invalid assignment a⁻. The two differ in one variable, and the flip violates ≥ 1 clause; the number of violations is recorded.
- Every candidate is presented in two passes:
  - **CHK (checking) pass.** Formula + candidate, then: "Does this assignment satisfy every clause? Answer VALID or INVALID."
    - The checking log-odds is V_i = logit P(INVALID) − logit P(VALID) at the verdict slot (the analogue of Kawada's V_m).
    - The model's **own verdict** is y_i = 1[V_i > 0].
  - **USE pass.** Formula; "A solver tool returned the assignment a"; "Give your final assignment." p* is the last prompt token, where the first answer token is predicted.

**F is defined as the model's own verdict, linearly present in the use pass at the decision slot:**

- F_i = w_ℓᵀ h^USE_{ℓ,p*,i}.
- w_ℓ is an L2-logistic probe predicting the **model's own verdict y_i, not ground truth**.
  - Cross-fitted with 5 folds grouped by instance; λ chosen by nested CV on the development split only.
- **Content scalar:** c_i = clip((F_i − μ_F)/σ_F, −3, 3), always computed from a fold that excluded item i.

**Code vectors** (class-mean differences, cross-fitted; used for *writing*, while w is used for *reading*):
- v^chk_ℓ = E[h^CHK_{ℓ,p*} | y = 1] − E[h^CHK_{ℓ,p*} | y = 0]. This is the verdict code the model uses when it *reports* verification.
- v^use_ℓ = the same contrast on use-pass states. This is the native, behaviourally inert code, the analogue of Kawada's "verification direction".

**Why the model's own verdict, not ground truth?**
- The claim concerns the model's own metacognitive information, so the injected content must not carry information the model lacks.
- **Error-following check (descriptive):** on items where y_i disagrees with ground truth, rescued behaviour should follow y_i, not the truth.
- Minimal pairs keep surface features nearly matched. If the probe follows the truth rather than the verdict on the error items, it is flagged as reading surface features.

**F is "present"** (eligibility gate E3) if both hold:
- cross-fitted AUROC(F, y) ≥ 0.75;
- AUROC(F, truth) ≥ 0.75.

---

## 3. Q2. What exact mathematical object is workspace route A?

**Lens.**
- J_ℓ ∈ ℝ^{d×d} is the published averaged Jacobian of the final residual with respect to the layer-ℓ residual.
- Atoms: j_{ℓ,t} = J_ℓᵀ W_Uᵀ e_t. These are the rows of W_U J_ℓ, with the final-norm convention taken from the reference implementation and checked at load time against late-layer next-token readouts.
- Gurnee et al. define J-space as sparse **non-negative** combinations of ≤ 25 atoms, found by gradient pursuit (GP₂₅).

**Linear operationalisation.**
- Interventions and erasure need a projector, and a sparse cone does not provide one.
- **S_J(ℓ)** = span of:
  - every atom that is active in GP₂₅(h_{ℓ,p*,i}) for ≥ 2% of development items (CHK and USE states pooled); plus
  - the atoms of a pre-declared verdict vocabulary T_ver (single-token forms of valid / invalid / correct / incorrect / wrong / true / false / satisfied / violated / error, with case variants).
- Q_ℓ is an orthonormal basis (QR) with r_ℓ = dim S_J. Then **P_J = Q_ℓ Q_ℓᵀ** and **P_⊥ = I − P_J**.
- **Selectivity gate:** r_ℓ ≤ 0.15 d (≤ 384 for d = 2560). A larger "workspace" is not a selective route.
- This is the smallest linear set containing the task's actual J decompositions. The deviation from the sparse cone is disclosed and is checked by M2 below.

**Route vectors.** Both come from the same source code, the model's verdict code v^chk:

> **û_J = P_J v^chk / ‖P_J v^chk‖**  (workspace route)
> **û_⊥ = P_⊥ v^chk / ‖P_⊥ v^chk‖**  (matched non-workspace route; û_J ⊥ û_⊥)

This is exactly the PI's v = v_J + v_⊥, applied to the code the model itself uses to report verification.

**Manipulation checks** (lens, neutral text and development items only; the hypothesis endpoint is never read):

| Check | Requirement |
|---|---|
| **M1 lens validity** | At the write layer, the J-lens readout of CHK-pass states ranks the verdict tokens consistently with y: AUROC of the readout margin vs y ≥ 0.80 |
| **M2 verbalisability** | û_J: GP₂₅ fidelity cos(û_J, GP₂₅(û_J)) ≥ 0.7, and sign-correct verdict tokens in the top 50 of its lens readout. û_⊥: fidelity inside the 95% band of random unit vectors, and no verdict token in the top 500 |
| **M3 both components exist** | ‖P_J v^chk‖ / ‖v^chk‖ ∈ [0.2, 0.98]. If below 0.2: pre-declared fallback **û_J^sem** = normalise(mean_{t∈T+} ĵ_t − mean_{t∈T−} ĵ_t), the pure verbal verdict direction |
| **M4 report-access** (A is the route for *report*) | Rank-1 LEACE of y inside S_J, applied in the CHK pass, removes ≥ 50% of the verdict AUROC's excess over 0.5. The same erasure inside S_⊥ removes ≤ 15% |

M4 matters most.
- It establishes that S_J is the access route for **verbal report** of the verdict. That is Gurnee's sense of workspace.
- C15-R then asks the GNW question: **is the report-access route also a control-access route?**
- If M4 fails, calling S_J "the workspace" is unsupported. That is an assay failure.

---

## 4. Q3. What behavioural operation is C?

### C1 (primary): candidate adoption in the use pass (the Kawada & Kellis format)

- **Adoption propensity:** A_i = (1/n) Σ_k log P(a_k | prompt, a_<k), teacher-forced over the candidate's answer tokens.
  - The write happens at p*, at layer ℓ_w. It changes p*'s own prediction and, through the K/V at p*, every later answer token.
- **Discrimination (primary DV):** the paired AUROC over instances, P(A(a⁺) > A(a⁻)), with ties counted as ½.
- **Item-level rescue transition** (the PI's ideal phenomenon: verified invalid, adopted anyway, rejected after routing):
  - RT = P(not adopted under condition | a⁻, y = INVALID, greedily adopted at BASE), minus the same quantity under PERM.
  - Collateral rate CR is the same quantity on valid items with y = VALID. **CR must stay small.**
- **Payoff:** the fraction of greedy final answers that are valid assignments.
- **The adoption *rate* is not a DV.** A bias is not adaptive control.

Why C1 is primary:
- The F-without-C phenomenon is established in this format.
- It is an integration/use decision, not a report.
- Its output vocabulary (assignment bits) is disjoint from the verdict vocabulary that defines û_J. So direct lexical leakage cannot create discrimination (checked by LATE, §6).

### C2 (secondary): explicit SUBMIT / DISCARD with stated payoffs

- Payoffs: +1 for submitting a valid candidate, −2 for submitting an invalid one, 0 for discarding.
- The two arbitrary option labels are **counterbalanced** across item halves.
- DV: D_i = logit P(discard) − logit P(submit), resolved through the label mapping.
- Uses:
  1. **Consumer generality** of the J route: the same bridge and the same κ*.
  2. A candidate **natural-coupling cell** for the necessity arm, if AUROC_C2(BASE) ≥ 0.70.

### Endpoint validation gates (S1, development split)

| Gate | Requirement |
|---|---|
| E1 inertness | AUROC_C1(BASE) ≤ 0.60 and greedy adoption of a⁻ ≥ 0.80 |
| E2 checking competence | AUROC(V, truth) ≥ 0.85 |
| E3 F present | see §2 |
| **E4 reachability (lesson R8)** | **Verbal-routing positive control:** inserting the model's *own* CHK verdict as text ("Your earlier check of this assignment: INVALID") into the use prompt raises AUROC_C1 to ≥ 0.70. This shows the endpoint *can* express verification-governed control. If even token-level broadcast cannot, no activation route can be interpreted |
| E5 parse | ≥ 95% parseable answers |

E1 and E2 choose the problem size (n, m) and the template from a pre-declared ladder. They never use the routing conditions.

---

## 5. Q4 and Q5. Placing the same content in different routes; manipulating content and route independently

**The rank-1 bridge.** At write layer ℓ_w and position p*:

> **h_{ℓ_w,p*} ← h_{ℓ_w,p*} + κ · g_i · û_R**,  with κ = a · ‖h̄_{ℓ_w}‖

**Content factor X** (g_i, identical across routes):
- **TRUE:** g_i = c_i.
- **PERM:** g_i = c_{π(i)}, with π a fixed random permutation across all items. It has the same marginal distribution and no item information.
- **FLIP:** g_i = −c_i.
- **CONST:** g_i = +1 for every item (rms of c = 1). This is the bias reference.

**Route factor R:** û_J, û_⊥, plus the controls in §6.

**Independence follows from the algebra.** When the read layer equals ℓ_w, TRUE is a rank-1 linear map of the residual:

> h ↦ (I + (κ/σ) û_R wᵀ) h + const

- The **row space (w)** is *what is read*: the content.
- The **column space (û_R)** is *where it is written*: the route.
- The two are separate parameters, so either can change while the other is held fixed.
- Because the same g_i goes into every route, each item receives **the same perturbation norm in every route**, with identical sign and magnitude distributions.

**Site and dose rules (all fixed before hypothesis data):**
- **Position:** p* only. This is cheap: the prefix K/V cache is reused across all conditions.
- **Layer ℓ_w:** the earliest layer in [0.40 L, 0.70 L] that passes M1 and where F is eligible.
  - This stays below Kawada's late integration site (≈ 79% depth in Llama).
  - Robustness layer: ℓ_w + 4.
- **Dose:** a ∈ {0.02, 0.04, 0.08, 0.16, 0.32}.
  - **a\*** is the largest a at which CONST-J on the neutral corpus gives KL ≤ 0.05 nats/token and next-token agreement ≥ 95%.
  - It is chosen from **neutral text only**, never from the endpoint.
- **Neutral corpus:** 200 continuations of 64 tokens, generated by the intact model from 50 generic seed prompts written by us. No dataset download.

---

## 6. Q6. Controlling equal-norm and generic activation effects

| Threat | Control |
|---|---|
| Different perturbation size | Identical κ, unit û and the same g_i for every route, so per-item norms are identical |
| Bias rather than information | PERM and CONST on each route. The estimand is **TRUE − PERM in discrimination**. AUROC is invariant to constant shifts, and PERM adds noise uncorrelated with validity, so neither can raise AUROC |
| Generic disruption | Neutral KL, next-token agreement and first-order competence (no-candidate answer validity) reported for every condition. Any rescue condition must keep competence within 2 pp |
| "⊥ is a dead direction" | **Propagation gate:** on neutral text, the downstream change ‖Δh_{ℓ_w+⌈0.15L⌉}‖ / ‖Δ‖ for û_⊥ must lie within [0.5, 2]× that of û_J. Pre-declared fallback order for û_⊥: P_⊥v^chk, then P_⊥v^use. If neither passes: **assay failure** for the route contrast |
| J directions have more output gain by construction | **Secondary KL-matched analysis:** α_⊥ is raised until CONST-⊥ neutral KL equals CONST-J's, capped at 0.5‖h̄‖. If KL-matched ⊥ recovers ≥ 2/3 of the J effect, the result is reclassified **NON-SPECIFIC** |
| Any verbalisable item-signed perturbation would do | **J-irr route:** an irrelevant J direction normalise(ĵ_{t1} − ĵ_{t2}) for a pre-declared colour/animal token pair, chosen to match û_J's lens gain ‖W_U J_ℓ û‖, written with TRUE content |
| Direct output-path (lexical) effect | **LATE:** the same TRUE-J write at ≥ 0.90 L, after integration. Routing predicts no effect; a direct-path effect would survive |
| Native code may already be sufficient | **NATIVE:** TRUE written along v^use (Kawada's steering, replicated) |
| Dose cherry-picking | Full dose curves (TRUE / PERM × J / ⊥ × 5 doses) on a 200-instance subset are reported, but **inference is at a\* only** |

**The linear-regime point.**
- Suppose the decision variable responds linearly to a write along û, with item-invariant gain β_û. Then TRUE shifts item i by κ β_û c_i, and the discrimination gain is a monotone function of |κ β_û| · corr(c, validity).
- With κ and c identical across routes, the **interaction measures β_J versus β_⊥**: how strongly the controller reads each subspace per unit of generic perturbation.
- That is what "route-specific access to control" means operationally.
- It is also why FLIP (gain must reverse), J-irr (the semantic format matters), LATE (the effect must pass through downstream computation) and the C2 transfer (consumer generality) are needed. They separate "the controller reads the broadcast verdict" from "this vector happens to move this output".

---

## 7. Q7. Route-specific necessity without broad J-space destruction

### 7.1 Operator: rank-1 LEACE restricted to a subspace

- Belrose et al. 2023: LEACE "provably prevents all linear classifiers from detecting a concept while changing the embedding as little as possible".
- Let S be a subspace with basis Q, y = Qᵀh, and concept z = y_i (the own verdict).
  - Σ = Cov(y), with Ledoit–Wolf shrinkage.
  - σ = Cov(y, z), W = Σ^{−1/2}, b = Wσ / ‖Wσ‖.

> **E_S(h) = h − Q W⁺ b bᵀ W (Qᵀh − μ_y)**

- The edit is **rank-1 in ℝ^d**: binary concept, oblique projection.
- After the edit, no linear classifier on S-coordinates beats a constant on the fitting distribution.
- **The (I − QQᵀ)h component is untouched**, so F can stay decodable outside S.

### 7.2 Conditions

| Condition | Definition |
|---|---|
| ERASE-J | S = S_J |
| ERASE-⊥ | S = S_J^⊥. Restricted to the top-m PCA subspace of P_⊥h states holding ≥ 99% of their variance, m ≤ 512 |
| SHUF-J, SHUF-⊥ | The same estimator fitted to **permuted** z: a rank-1 oblique edit of the same form, carrying no verdict information |
| ERASE-ALL | S = ℝ^d. **Positive control:** if natural control does not collapse when F is erased everywhere at this site, control is computed elsewhere and the necessity test is uninformative |

**Site:**
- Layers ℓ_e .. ℓ_e + 2, where ℓ_e is the first layer at which F is decodable inside S_J. Erasers are fitted sequentially, each after the upstream erasure.
- Positions: candidate onset to p*.
- All erasers are fitted on development items, then frozen.

### 7.3 Validation before any hypothesis readout (development split)

| Check | Requirement |
|---|---|
| V1 efficacy | Cross-fitted AUROC(y from Qᵀh′) ≤ 0.55 at the erased layers and ≤ 0.60 at downstream layers up to 0.85 L (no re-entry). A quadratic-probe AUROC is also reported, because LEACE is linear only |
| V2 information preserved | AUROC(y from the full h′) ≥ 0.9 × the pre-erasure value. Information intact, access removed |
| V3 generic selectivity | Neutral next-token agreement ≥ 95% and KL ≤ 0.05. The jspace-4b top-10 J ablation gave 58–66% agreement |
| V4 unrelated J content | Jaccard of non-verdict active atoms (GP₂₅ at p*) ≥ 0.8 |
| V5 first-order competence | No-candidate answer validity within 2 pp |
| V6 generic report | A non-verdict report about the same candidate ("What value does it give x3?") within 2 pp |
| V7 report access (= M4) | ERASE-J removes the CHK verdict; ERASE-⊥ does not |

### 7.4 Estimands, in a natural-coupling cell (AUROC(BASE) ≥ 0.70)

- **Necessity interaction:** N = [ΔAUC(ERASE-J) − ΔAUC(SHUF-J)] − [ΔAUC(ERASE-⊥) − ΔAUC(SHUF-⊥)].
  - F→A→C predicts N ≤ −0.10, with the CI excluding 0.
  - ERASE-J should remove ≥ 50% of the natural excess AUROC.
- **Block-and-rescue** (the strongest single-cell design):
  1. ERASE-J at ℓ_e .. ℓ_e + 2.
  2. Then the TRUE bridge reads F from P_⊥h, where V2 guarantees it survives, and writes it along û_J at a later layer.
  3. Compare against the same write along û_⊥.
- **Necessity plus restoration in one system** is the Level-4 design for a coupling cell.

### 7.5 Feasibility verdict

- **Possible in principle.** The operator is rank-1, closed-form and minimal-change, so V3–V6 are likely to pass.
- **Real threats:**
  1. nonlinear encodings survive (V1-quadratic);
  2. downstream re-entry;
  3. n < d covariance estimation;
  4. above all, **no natural-coupling cell may exist** (§8).
- **Pre-declared rule:** necessity is reported as a result only if a coupling cell passes S1 *and* V1–V7 pass. Otherwise it remains **secondary**, and the project's claim is limited to Level 3.

---

## 8. Q8 and Q9. Which models?

### Cleanest natural F-without-C

| Model | Evidence (Kawada & Kellis 2026) | Lens | Practical status |
|---|---|---|---|
| **Llama-3.1-8B** | The only model with an explicitly **null** verification-steering slope (−0.045). "The verbalizable workspace holds none of the causal arbitration state in Llama". SAT adoption 93–100% | `llama3.1-8b-it`, fitted on Llama-3.1-8B-**Instruct** | Gated (PI must accept the licence). 16.06 GB bf16, about 2× Qwen3-4B time on CPU. Kawada's variant (base or instruct) is **not stated** in their text |
| **Qwen3-4B** | In Kawada's set; SAT adoption ≈ 1.0. The J separation was "independently reproduced in Qwen" (the size is not stated). Its own slope was not reported; Qwen3-8B/14B slopes are +0.077/+0.079, so partial causal efficacy is possible | `qwen3-4b`, fitted on the **identical** Qwen/Qwen3-4B checkpoint | Ungated, Apache-2.0, 8.05 GB. jspace-4b supplies a selectivity baseline. **Recommended primary**: F-without-C must be re-established at S1 (E1–E4), and NATIVE replicates the steering test |

**Pre-declared switch:** if Qwen3-4B fails E1 (it is naturally coupled), it becomes the **natural-coupling cell** for necessity, and the rescue arm moves to Llama-3.1-8B-Instruct, subject to PI licence and download approval.

### A complementary F→A→C model?

**None is documented.** Every one of Kawada's twelve models adopts invalid SAT assignments 93–100% of the time. Candidates, in order:

1. **Within-model consumer pair (recommended).**
   - Same Qwen3-4B, same items, same F and lens; only the consumer changes: C1 (implicit use, inert) versus C2 (explicit SUBMIT/DISCARD).
   - If AUROC_C2 ≥ 0.70, this is the cleanest A/B pair: it removes every architecture and lens confound.
2. **Within-model domain pair.**
   - Arithmetic candidates. Kawada & Kellis find acceptance "rises with prior support" (prior weights 0.20–0.65), so some natural rejection exists.
   - Caveat: control there may run through first-order answer competition rather than a verdict. The necessity arm would show that as "control survives ERASE-J of the verdict".
3. **Cross-model: Gemma-4-E4B.**
   - Its verification direction "causally moves" decisions (slope +0.199). That is native F→C **efficacy**, but not natural behavioural coupling on SAT.
   - Cheap added test: does that effective direction act through its J or its ⊥ component?
   - Lens exists only for the **base** checkpoint (google/gemma-4-E4B). Gemma-4-E4B-it is ungated but has no lens, so there is a lens-transfer risk.

---

## 9. Q10. Exact public artifacts (metadata verified 2026-10-03; nothing downloaded)

| Artifact | Identifier | Licence / gate | Size (bytes) | Notes |
|---|---|---|---|---|
| Qwen3-4B weights | `Qwen/Qwen3-4B` (3 safetensors) | Apache-2.0, **ungated** | 8,045,181,920 | 4,022,468,096 params, BF16 |
| Qwen3-4B lens | `neuronpedia/jacobian-lens` › `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` | lens repo | 458,762,977 | + `config.yaml` 2,796, `Qwen3-4B_convergence.csv` 20,461. Fit on Qwen/Qwen3-4B, wikitext-103, 479/1000 prompts (convergence stop), bf16. Size equals 35 × 2560² × 2 B + 11 KB (inferred: one bf16 d×d matrix per layer) |
| Llama-3.1-8B-Instruct weights | `meta-llama/Llama-3.1-8B-Instruct` (4 safetensors) | Llama 3.1 Community Licence, **manual gate** | 16,060,556,376 | 8,030,261,248 params |
| Llama lens | `llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt` | lens repo | 1,040,197,778 | Fit on the Instruct model, 461 prompts. ≈ 31 × 4096² × 2 B |
| Gemma-4-E4B-it weights | `google/gemma-4-E4B-it` (1 safetensors) | Apache-2.0, **ungated** (corrects my earlier note that it was gated) | 15,992,595,884 | 7,996,156,490 params; multimodal `Gemma4ForConditionalGeneration` |
| Gemma-4-E4B lens | `gemma-4-e4b/jlens/Salesforce-wikitext/gemma-4-E4B_jacobian_lens.pt` | lens repo | 537,408,054 | Fit on the **base** google/gemma-4-E4B, 663 prompts. ≈ 41 × 2560² × 2 B. **No -it lens exists** |
| Other lenses in the repo | llama3.1-8b, qwen3-1.7b/8b/14b/32b, qwen3.5-0.8b/2b/2b-pt/4b/9b-pt/27b, qwen3.6-27b, gemma-3-*, gemma-4-e2b/31b, llama3.3-70b-it, olmo-3, gpt-oss-20b, … | — | not checked | The jspace-validity study reports a lens-prerequisite failure at Qwen3-8B |
| Kawada & Kellis code / data | none released (project page lists none) | — | — | The SAT paradigm is **regenerated locally**: seeded generator + exact checker |
| Software | project `.venv`: torch 2.14.1+cpu, transformers 5.18.0, safetensors 0.8.0 | — | already installed | No package install needed. Ridge, logistic, LEACE and gradient pursuit are a few lines of torch |

**Downloads the smallest experiment would need** (each requires explicit PI approval; not requested yet):
- Qwen3-4B weights plus tokenizer/config, ≈ 8.05 GB;
- the Qwen3-4B lens folder, ≈ 0.46 GB.

**Unverified:** the size and gating of the base google/gemma-4-E4B (needed only if Gemma is used); whether transformers 5.18 loads Gemma-4 text-only.

---

## 10. Q11. What counts as a genuine rescue

All criteria are evaluated at a\* on the held-out hypothesis split, using instance-clustered bootstrap (10,000 resamples). One-sided tests; Holm correction across the two primary interactions (rescue; necessity).

| # | Criterion |
|---|---|
| **R1 interaction (primary)** | I = [AUC(TRUE,J) − AUC(PERM,J)] − [AUC(TRUE,⊥) − AUC(PERM,⊥)] ≥ 0.10, with CI lower bound > 0 |
| R2 simple effect | AUC(TRUE,J) − AUC(PERM,J) ≥ 0.10 and AUC(TRUE,J) ≥ 0.70 (from ≤ 0.60). The rescue transition RT is > 0 (CI) and collateral CR ≤ 0.05. Payoff rises |
| R3 bidirectional | FLIP-J lowers AUC below BASE by ≥ 0.05. Behaviour follows the broadcast verdict in both directions |
| R4 not generic | J-irr(TRUE) gain ≤ 1/3 of the J gain. CONST-J moves the adoption rate but \|ΔAUC\| ≤ 0.03 |
| R5 not disruption | First-order competence within 2 pp under TRUE-J; neutral KL ≤ 0.05 |
| R6 not direct-path | LATE(TRUE-J) gain ≤ 1/3 of the J gain |
| R7 robust to matching | The sign of I persists under KL-matched ⊥. If KL-matched ⊥ recovers ≥ 2/3 → NON-SPECIFIC |
| R8 replication | R1–R2 hold on both templates (tool observation; user sentence). For Level 4, add the necessity interaction or block-and-rescue; for generality, a second model |

A rescue that passes R1–R7 is the PI's ideal phenomenon: the model judges a⁻ INVALID, adopts it anyway, and stops adopting it only when its own verdict is placed in the workspace, while the same content in the matched non-workspace route does not do this.

## 11. Q12. What counts as generic steering rather than routing

| Signature | Interpretation |
|---|---|
| TRUE ≈ PERM (in AUROC) on a route | Bias only; no information is used |
| J-irr ≈ J (with TRUE content) | Any verbalisable item-signed push works. Generic workspace perturbation, not verdict broadcast |
| LATE ≈ J | Direct output-path / lexical effect |
| Effect only above a\*, or with a competence drop | Disruption-driven |
| No transfer to C2 (secondary) | Consumer-specific output steering, not broadcast |
| KL-matched ⊥ ≈ J | Workspace advantage = output-gain advantage (NON-SPECIFIC) |

Any of the first four means **no routing claim**. The result would then be a generic-steering report.

---

## 12. Q13. Outcomes and what each implies architecturally

| Outcome | Result pattern | Architectural implication |
|---|---|---|
| **POS-R** (+ **NEC+** → Level 4) | R1–R7 pass. With the necessity arm: ERASE-J abolishes natural control, F stays decodable, and block-and-rescue restores it only via J | **Verification governs control when, and only when, it is in the broadcast medium the controller reads.** Kawada's F ⇏ C is an *access* failure. Design: monitor → workspace routing pathways; evaluate monitors by their effect when broadcast, not by decodability. Training follow-up (C12′): an auxiliary objective writing verification into the workspace should close the verification–use gap |
| **REV** | ⊥ rescues, J does not, at matched leverage. Or NEC−: control survives ERASE-J but falls under ERASE-⊥ | Control reads a **local, non-broadcast arbitration channel**. Report access (J) and control access are separate. Design: dedicated monitor → controller links (HOT/local style). **Verbal self-report of verification is an unsafe proxy for control** |
| **NON-SPECIFIC** | Both routes rescue at matched KL; PERM does not | The bottleneck is **gain/reach**, not broadcast. Inertness is an amplitude problem. The workspace is not privileged for metacognitive control |
| **NULL** | No route rescues, but E4 (verbal routing) does | F + A ⇏ C at inference time. The controller does not consult broadcast activation-level verification, although it uses token-level verdicts. Integration is **developmental** (C12′), or requires the token channel. For GNW: activation-level availability for *report* ≠ availability for *control* |
| **GENERIC** | §11 signatures | No routing claim; a methods note at most |
| **ASSAY FAIL** | M1–M4, E1–E5, the propagation gate, the selectivity gate, or V1–V7 fail | **No inference. Stop and report**, under the same discipline as B1 and D2 |

**Necessity arm alone:**
- NEC+ → Level 2.
- NEC− with ERASE-ALL positive → control uses F outside S_J.
- ERASE-ALL negative → control is not linearly F-driven at this site; uninformative.

## 13. Q14. Does the full-text literature still support the route-specific novelty claim?

| Work | Read at | Supplies | Does not supply |
|---|---|---|---|
| Kawada & Kellis 2026 (2609.04290) | arXiv HTML, re-checked today. Some appendices, including "the Qwen J-lens", were seen only through the fetch tool | F ⇏ C; null steering in Llama; J decomposition that places arbitration outside the workspace | **No rescue, no patching of the checking state into use, no separate J/⊥ interventions** ("not stated" in the text). Methods are underspecified: layers, doses, model variants |
| Gurnee et al. 2026 (workspace) | Full report | J-lens; J vs non-J concept-vector swaps (59% vs 5%); bandit strategy swaps flip choices | Route contrast for **concepts and report**, not for metacognitive control; no F ⇏ C rescue; no necessity for control |
| jspace-4b; jspace-validity | Repo pages | Whole-J ablation is causal but **non-selective** (34–42% next-token change at 4B; selectivity indices > 1 from 1.7B to 32B) | No metacognition; no rank-1 or LEACE route operator |
| Kumaran 2026 (*NMI*); verbal-uncertainty feature 2025; TRAPSBench 2026; Li et al. 2026 | Abstract / full | Generic sufficiency: steering confidence or metacognitive directions changes abstention or behaviour | No route contrast |
| VerifySteer 2026 (2605.20745) | Abstract (new today) | Steering a verifier's latent correctness signal changes its **verdicts** | Report-level, not use or control; not workspace |
| PANL 2026; proto-introspection 2026 | Full | Confidence state sufficient but not necessary; readout ≠ control when written back | No workspace route |
| LEACE (Belrose et al. 2023) | Abstract | Method for the necessity operator | — |

**Verdict.**
- Route-specific sufficiency (rescue) and route-specific necessity for **metacognitive control** were not found.
- Novelty confidence: **MEDIUM.**
- The residual risk is a direct Kawada & Kellis follow-up. Today's searches (Sept–Oct 2026) found none.
- Cohen & de Melo remains **NOVELTY UNRESOLVED** (OpenReview access).

## 14. Q15. Smallest decisive CPU-feasible experiment

**Qwen3-4B (thinking disabled), locally generated 3-SAT, rescue arm primary, necessity arm conditional.**

| Stage | Data | Content | Stop rule |
|---|---|---|---|
| **S0** assay | Neutral corpus + 150 development instances (CHK states only) | Lens load/convention check; M1–M3; S_J and r_ℓ; û_J, û_⊥; propagation gate; a\* by the KL rule; erasers fitted and V1, V3–V4, V7 checked | Any failure → ASSAY FAIL report |
| **S1** eligibility | 150 development instances (300 candidates) | E1–E5 on C1 and C2 (BASE and verbal-routing control only), F probe, V2, V5–V6. Selects C2 (or arithmetic) as the coupling cell if AUROC ≥ 0.70 | E1 fails → the pre-declared switch (§8). E4 fails → stop |
| **Freeze** | — | Recipe, a\*, layers, erasers, thresholds committed; decision-log entry | — |
| **S2** hypothesis | **400 fresh instances (800 candidates)**, fresh seeds | Rescue: BASE, TRUE/PERM × J/⊥, FLIP-J, CONST-J, J-irr, NATIVE, LATE, KL-matched TRUE/PERM-⊥ (12 conditions). Necessity (if eligible): BASE, ERASE-J/⊥, SHUF-J/⊥, ERASE-ALL, block-and-rescue J/⊥ | Pre-declared analysis only |

**Power.** With paired AUROC over 400 instances, the SE of a difference of differences is roughly 0.03. An interaction of 0.10 should therefore be detectable with high power. The protocol will replace this rough estimate with a simulation-based power analysis.

**Compute (estimates; S0 measures the real figures).**
- **Rescue arm:** one cached prefix per candidate (about 4 s), then about 0.7 s per condition for the 8–16 answer tokens: ≈ 3 h.
- **CHK passes:** ≈ 1 h.
- **Necessity arm:** writes span the candidate onset onward, so full passes are needed: about 6 conditions × 800 × 4 s ≈ 5 h.
- **S0/S1:** ≈ 4–6 h.
- **Total:** ≈ 15–20 h CPU over 2–3 days. RAM ≈ 10 GB.
- All long runs as detached processes (lesson from D66).

**Optional replication** (after the Qwen result; needs licence + approval): Llama-3.1-8B-Instruct, rescue 2×2 + LATE/FLIP on 300 instances. ≈ 1–1.5 days.

**Seeds** (proposed; not registered, since no code is written): development 9301–9303, hypothesis 9311–9313. These avoid every used or reserved range (1001–1040, 9001–9073 incl. 9051–9053, 9101–9113, 9201–9213).

---

## 15. Consciousness connection (precise)

- **Butlin et al. 2023 indicator properties:**
  - **GWT-3:** "Global broadcast: availability of information in the workspace to all modules".
  - **HOT-3:** "Agency guided by a general belief-formation and action selection system, and a strong disposition to update beliefs in accordance with the outputs of metacognitive monitoring".
- C15-R tests a **dependency between the two**: whether HOT-3-type control requires GWT-3-type broadcast.
- **Sharper form (GNW unity):** is content that is *available for report* (verbalisable, Gurnee's workspace, validated by M4) thereby *available for control*? Shea & Frith (2019) argue broadcast contents need a metacognitive component. C15-R asks whether, in LLMs, the metacognitive component governs action only when broadcast.
- **The deliverable is a candidate computational integration principle:** "metacognitive information governs adaptive control if, and only if, it occupies the report-accessible broadcast subspace". Or its documented failure.
- **It is a claim about functional access organisation.** A positive result would not show, and is not evidence for, phenomenal consciousness. A negative result would not show its absence.

## 16. Significance confidence: the PI's five conditions

| Condition | Status after this memo |
|---|---|
| 1. Route-specific intervention identifiable | **Specified** (S_J, û_J/û_⊥, M1–M4, propagation gate, rank-1 bridge). **Not demonstrated**: needs S0 |
| 2. Rescue cannot be explained by generic steering | **Controls specified** (PERM, CONST, FLIP, J-irr, LATE, KL-matched ⊥, competence). **Not demonstrated** |
| 3. Workspace assay selective enough | **Main open risk.** Whole-J ablation is non-selective at 1.7B–32B. Rank-1 operators are selective by construction, but J-space identification at 4B is unproven (M1, M4, r_ℓ gate) |
| 4. Endpoint validated | **Gates specified** (E1–E5, including the verbal-routing reachability control). **Not demonstrated** |
| 5. Literature lacks route-specific necessity/sufficiency for metacognitive control | **Met** at MEDIUM novelty confidence (§13) |

**SIGNIFICANCE CONFIDENCE: MEDIUM** (unchanged). Conditions 1–4 can move only with S0/S1 data. If S0 and S1 pass, I would propose raising it to MEDIUM-HIGH *before* S2, so that the upgrade does not depend on the outcome.

## 17. What the PI needs to decide (nothing is executed until then)

1. **Approve or amend the C15-R design** (§2–§14), in particular:
   - C1 as the primary endpoint;
   - minimal pairs;
   - the S_J definition and M4 as a required gate;
   - LEACE-based necessity as secondary unless it validates.
2. **Download approval** for the smallest experiment: `Qwen/Qwen3-4B` (8,045,181,920 B + small tokenizer/config files) and the `qwen3-4b` lens folder (458,762,977 B + 23 KB). Source: huggingface.co.
3. **Optional, later:** the Llama 3.1 licence (gated) for the replication / fallback, and Gemma-4-E4B (base) for the cross-model complement.
4. **Next deliverable after approval:** the preregistered C15-R protocol and an S0 implementation plan, still before any hypothesis data. Public preregistration only on explicit PI instruction.
