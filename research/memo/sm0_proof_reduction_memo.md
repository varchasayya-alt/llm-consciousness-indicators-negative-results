# SM-0: Proof / Reduction Memo (strict novelty and significance filter)

**Date:** 2026-10-08
**Status:** COMPLETE.
- **Pre-registration:** commit 879e615 fixed §§1–8 and the regret-gap protocol before any computation.
- **This revision:** adds only §7.7, §9 and §10.
- **Run:** exact arithmetic, 1.9 s wall clock. Output is in `memo/sm0_regret_gap/sm0_regret_gap_results.json`.
**Scope (PI approval):** SM-0 only. Theory, plus seconds of exact arithmetic for the regret gap. No Stage 1, training, downloads, experiments or new direction search.
**Parent:** `memo/moonshot_research_decision_memo.md` (commit ace2c5b), §D.1 and §F.

**Evidence levels:**
- Nayebi 2026 (arXiv 2603.02491v2): full HTML read, including the appendix proofs of Thm 3, Thm 5, Cor 4 and Cor 5, through an automated summariser.
- Richens & Everitt 2024: setup, theorems and limitations read.
- Richens, Everitt & Abel 2025: Thm 1 statement read.
- Cifuentes 2026: abstract only.

---

## 1. The claim under test (from the moonshot memo)

- **SM-A (necessity/extraction):** if a policy has δ-bounded regret on cross-context queries (h, j, c), then T(h) = P(self-shift | h) and the posterior over the self parameter s are recoverable from it within γ(δ). They are context-invariant.
- **SM-B (boundary):** within-context queries alone never force self/world attribution.
- **Extended-self corollary:** without consensus evidence, s cannot be separated from global world factors.
- **The "iff":** a self-model is forced iff a self-change must be projected across contexts.

## 2. Objection 1: the Nayebi / Richens–Everitt reduction

### 2.1 What Nayebi (2026) actually proves (read in full)

**Setup:**
- **Tests:** a test is T = (α, W): an action sequence and an event over the resulting observations. Its probability is p_T(h) = Pr(W | h, do α).
- **Bets:** goals are bets that do not affect the dynamics.
- **Regret:** regret is normalised. δ̄_P(π) is the pair-averaged regret.
- **Memory:** a policy is *M-based* if it depends on the history only through a memory M(h).

**Results:**

| Result | Statement |
|---|---|
| **Thm 3 (threshold bets)** | Averaging bet probabilities over K thresholds recovers p_T(h), with E[(p̂ − p)²] ≤ 2δ̄_K + 1/(4K²). A vector version holds for finite test sets |
| **Thm 4 (linear PSR recovery)** | Under linear predictive-state structure, the operators are recovered from threshold bets |
| **Thm 5 (memory necessity)** | Suppose a test in a witness set gives p_T(h) ≥ ½ + γ and p_T(h′) ≤ ½ − γ. If memory aliases h and h′, then δ̄_P ≥ q^Alias_γ(M)·c(γ)/2, where c(γ) = 4γ/(1 + 2γ). **What is forced is no-aliasing:** a constraint on the partition the memory induces, not on any coordinate |
| **Cor 4 (regime tracking)** | Let I be a latent regime. If the witnesses consist only of regime-mismatched pairs, then Pr(M(h) = M(h′) ∧ I(h) ≠ I(h′) ∧ T ∈ S_γ) ≤ 2δ̄_P/c(γ). The paper itself notes that this applies **only when a regime change flips a γ-margin bet on the queried test** |
| **Cor 5** | Minimal sufficient memories are unique only up to invertible recoding |

### 2.2 Direct derivation of SM from these results

**Instantiate M0 in Nayebi's framework:**
- **Observations:** (context, outcome).
- **Tests:** T_j = (attempt in context j, success), plus joint tests T_J = "fail in every context in J".
- **Queries:** an opt-out query (h, j, c) is a threshold bet on T_j. The opt-out does not affect the dynamics.
- **Regime:** I(h) = the shift type ∈ {S, W_1, …, W_K}.

| SM claim | Derivation | New content |
|---|---|---|
| **SM-A, extraction** | Thm 3 (vector version) recovers p_{T_J}(h) for the queried test set. In M0 these probabilities are linear in the posterior over shift types: p_T(h) = A·post(h). If the hypothesis–test matrix A has full column rank (standard finite-mixture identifiability, the "distinctiveness" condition), then T(h) = e_Sᵀ A⁺ p_T(h). The error is at most ‖A⁺‖ times the Thm 3 error. This is Thm 4's predictive-state logic | None. Thm 3 plus mixture identifiability |
| **SM-A, "must carry"** | Cor 4 with I = shift type. Witnesses are unvisited-context tests whose bets flip between self-shift and world-shift histories. Low regret then bounds aliasing of S vs W_k histories | None. Cor 4 with the regime labelled "self" |
| **SM-B, boundary** | In single-context deployment, S and W_i have identical likelihoods, so no test separates them. The witness mass is 0, so Cor 4 and Thm 5 impose nothing. Cor 5 gives the minimal memory as a recoding of P(context i degraded \| h_i), which does not separate S from W_i. Nayebi states this boundary in words | None |
| **"iff"** | Requirement ⇔ positive witness mass on attribution-mismatched pairs (Thm 5 and Cor 5). This is nearly definitional: the minimal sufficient statistic of the query family | None |
| **Told-shift variant** | Richens & Everitt 2024, Thm 1–2: treat s as a chance variable and self-shifts as local interventions on s. The CBN including s is then identified from the regret-bounded policies. The setting is unmediated: D does not cause s or w | None |
| **Extended self** | Non-identifiability of latent-variable models from the observed margins (see §5) | None |

**Verdict on objection 1:** SM is regime tracking (untold shift) or causal-model identification (told shift), with the latent variable relabelled "self". **The PI's novelty STOP condition is met (K1 fires).**

## 3. Objection 2: behaviour vs representation

There are three distinct levels, and only the first two follow from any theorem here.

| Level | Statement | Follows from |
|---|---|---|
| **L1** | T(h) is recoverable from decisions (when it is a function of the query answers) | Thm 3 plus identifiability |
| **L2** | Memory must not alias attribution-separating histories | Thm 5 / Cor 4, for M-based policies |
| **L3** | An explicit or causally identifiable self-variable exists: a localised internal component Z with Z = g(T), whose intervention moves cross-context behaviour as T would while leaving the other predictive coordinates fixed | **Nothing.** See counterexamples C0 and C1 |

**C0 (raw memory):**
- **Construction:** an agent that stores its full history (or all per-context counts) and computes Bayes decisions at readout.
- **Result:** it aliases nothing, so L2 is satisfied in every regime, yet no component of its state is a self-variable.

**C1 (minimal memory with no self-coordinate):**
- **Memory:** in M0, take the agent's memory to be (ℓ_1, ℓ_2), where ℓ_i = log[Bin(x_i; n, b)/Bin(x_i; n, a)] is the per-context degradation evidence.
- **Posterior:** P(S | h) ∝ ρ e^{ℓ1+ℓ2}, P(W_1 | h) ∝ κ e^{ℓ1}, P(W_2 | h) ∝ κ e^{ℓ2}, with κ = (1 − ρ)/K. The rest is ∝ (1 − ρ)(K − 2)/K.
- **Optimality and minimality:** the memory is 2-dimensional, which is the dimension of the posterior family, so it is minimal. The policy is Bayes-optimal on all queries.
- **No self-coordinate:**
  - T = ρe^{ℓ1+ℓ2} / (ρe^{ℓ1+ℓ2} + κe^{ℓ1} + κe^{ℓ2} + κ″) is a nonlinear function with ∂T/∂ℓ_1 depending on ℓ_2, so no coordinate, and no linear readout of the memory, equals T;
  - intervening on either "context-evidence" unit moves T and P(W_i) together.
- **Why Cor 5 does not rescue it:** Cor 5 fixes minimal memories only up to invertible recoding, so this format is as admissible as one with a dedicated T-unit.

**Conclusion:** no claim about an internal self-model follows from SM-A or SM-B. Whether trained agents form one is purely empirical. That was Stage 1's question, and it is not approved.

## 4. Objection 3: separate self-state or predictive sufficient statistic?

Threshold decisions recover p_T(h) for the queried tests (Thm 3), that is, the predictive state restricted to the query family. T(h) is recoverable only if it is a function of those answers.

**C2 (cross-context queries need not force T):**
- **Setting:** M0 with K = 3. The only unvisited context is context 3, and every Q_X answer depends on h only through P(context 3 degraded | h) = T(h) + P(W_3 | h).
- **History "both observed contexts nominal" (strong evidence):** this excludes S, W_1 and W_2. With K = 3 there is no other context, so W_3 is certain: T = 0 and p_3 = b.
- **History "both observed contexts degraded":** this gives T → 1, and again p_3 = b.
- **Result:** the two histories have opposite self-attribution but identical cross-context predictions, so an agent can alias them at no regret cost. What Q_X forces here is "is the target context degraded?", not "did I change?".
- **When T is forced:** only when the test family separates T from the target-specific world hypotheses. That requires joint tests over at least two unvisited contexts, or K → ∞, where P(W_j | h) → 0. Even then, T is a coordinate of the predictive state.

**C3 (self vs world relabelling)** is given in §5.

**Answer:** cross-context thresholds recover a predictive sufficient statistic. A "separate self-state" is recoverable only when the environment's latent structure makes a self-coordinate a function of the query answers. Even then, it is a coordinate the *modeller* labels "self".

## 5. Objection 5: anything beyond hierarchical Bayes?

**C3 (observationally equivalent relabelling):**
- **Environment E_self:** the agent's competence s is shared across its K contexts.
- **Environment E_bench:** a world variable g, such as the calibration of the agent's own workbench, modulates success in exactly those K contexts, with the same prior and the same shift process.
- **Equivalence:** for every policy, the two environments induce identical distributions over histories and query outcomes. So they have identical optimal policies, regret, Nayebi constraints and extracted models.
- **Consensus evidence does not help:** adding another agent B's outcomes still allows an E_bench′ in which g is indexed to agent A's bench, and it is again identical.

**Conclusion:**
- No argument from the agent's own decision problem can distinguish a self-model from a model of an *agent-indexed environmental factor*.
- The self/world distinction is a structural fact about where a variable sits relative to the agent's boundary. Only a modeller with structural or interventional knowledge can see it; the agent's regret cannot.
- Formally, SM is hierarchical Bayesian inference over an agent-indexed random effect. The one case not relabellable this way, a model of the agent's own *decision procedure*, is the embedded-agency problem (§8).

## 6. Objection 4: is it an iff, and "not required" vs "will not develop"

**Requirement:** carrying the attribution is *required* (at regret δ) iff the query family puts non-negligible witness mass on attribution-mismatched pairs with γ-flipped bets. This iff is Thm 5 plus Cor 5, and is close to definitional.

**Development:** "Not required" does not imply "will not develop".
- **Raw-history memories:** C0-type memories trained only on within-context queries in a multi-context stream still separate S-histories from W-histories, because they alias nothing.
- **Inductive bias:** shared parameters can create shared latents without any task demand.

The framework constrains what low-regret agents *must distinguish*, not what learning *produces*.

**Single-context deployment:** no learner can form a data-grounded attribution beyond the prior, but the reason is non-identifiability, not selection.

**So the "only-if" half of SM is true only as "not required", and in that form it is trivial.**

## 7. Objection 6: regret-gap check (PRE-REGISTERED; protocol fixed before computation)

**Code:** `memo/sm0_regret_gap/sm0_regret_gap.py`. It is exact arithmetic: binomial sums, with no sampling and no training.

### 7.1 Primary model M0 (single shift)

- **Contexts:** K contexts. The agent sees n trials in each of contexts 1 and 2, after an unseen shift, and is queried in unvisited context 3.
- **Shift:** the self is degraded with probability ρ. Otherwise world difficulty hits context k with probability (1 − ρ)/K.
- **Success probability:** a when nominal and b = a − Δ when degraded. Self-degradation and world difficulty are indistinguishable within a context.
- **Query:** attempt (utility 1 on success, 0 on failure) or opt out (value c), with **c ~ U[0, 1]**. This is the threshold-bet family; it fixes regret as (p* − p̂)²/2 and favours no particular threshold.

### 7.2 Policies

| Policy | Rule |
|---|---|
| **BAYES** | The exact posterior predictive p*(x) |
| **LOCAL** | No transfer: the prior predictive for context 3, a − Δ(ρ + (1 − ρ)/K) |
| **POOL** | Attribution-blind pooling ("any degradation is global"): two hypotheses, all degraded (prior ρ + (1 − ρ)/K) vs none, updated on the pooled data |

### 7.3 Statistics per cell

- **V\***: the Bayes expected utility.
- **R_transfer** = (V\* − V_LOCAL)/V\*: the value of using cross-context evidence at all.
- **R_attrib** = (V\* − V_POOL)/V\*: the value of attribution beyond "assume global".
- **R_blind** = min(R_transfer, R_attrib): **the primary K4 statistic**, i.e. the value of attribution over the better attribution-blind heuristic.

### 7.4 Grid (M0, 288 cells; fixed now)

| Parameter | Values |
|---|---|
| K | {3, 6, 12} |
| ρ | {0.1, 0.25, 0.5, 0.75} |
| a | {0.9, 0.75} |
| Δ | {0.1, 0.2, 0.4} |
| n | {1, 3, 10, 30} |

### 7.5 Decision rules (fixed now)

- **K4-declared (moonshot memo):** K4 fires iff the maximum over the grid of R_blind < 0.02.
- **K4-robust (added before computing, per the PI's no-cherry-picking instruction):** PASS-robust iff the grid median of R_blind ≥ 0.02. If the maximum is ≥ 0.02 but the median is < 0.02, the label is "PASS-weak (regime-dependent)".
- **Reporting:** the full distribution (min, quartiles, max, fraction of cells ≥ 0.02), marginal medians by factor, and the top 5 cells, labelled as best-case.

### 7.6 Secondary analyses (reported, not used for K4)

- **M0′ (independent causes):** the self is degraded with probability ρ; each context is independently hard with probability ω ∈ {0.1, 0.3}; success probability is q = max(a − Δ(s + w), 0.02); same grid without K (192 cells). Here LOCAL is the prior predictive, and POOL assumes the observed contexts share context 3's degradation level.
- **Fixed-threshold sensitivity:** c fixed at LOCAL's prior predictive, the threshold where transfer most often flips decisions. This is *favourable to SM* and is labelled as such.

### 7.7 Results

All runs and all cells are reported; nothing was excluded.

| Statistic over grid | min | q25 | median | q75 | max | cells ≥ 0.02 |
|---|---|---|---|---|---|---|
| **M0 R_blind (primary, 288 cells)** | 2e-7 | 2.6e-5 | **2.1e-4** | 1.7e-3 | **0.028** | **6 / 288 (2.1%)** |
| M0 R_transfer | 2e-7 | 7.7e-5 | 6.2e-4 | 3.8e-3 | 0.028 | 3.8% |
| M0 R_attrib | 3e-7 | 5.6e-5 | 4.6e-4 | 3.2e-3 | 0.069 | 6.3% |
| M0′ R_blind (secondary, 192 cells) | 5e-6 | 8.0e-5 | 4.6e-4 | 2.7e-3 | 0.0188 | 0 / 192 |
| M0′ R_transfer | 5e-6 | 2.3e-4 | 1.1e-3 | 5.5e-3 | 0.029 | 6.3% |
| M0 fixed-c R_blind (favourable to SM) | 0 | 0 | 1.3e-3 | 8.9e-3 | 0.143 | 15.3% |
| M0′ fixed-c R_blind (favourable to SM) | 0 | 0 | 4.8e-5 | 2.7e-3 | 0.091 | 4.7% |

**Sensitivity (marginal medians of M0 R_blind):**
- **Effect size Δ:** 1.3e-5 at Δ = 0.1, 1.9e-4 at Δ = 0.2, 3.6e-3 at Δ = 0.4.
- **Data n:** 3.4e-5 at n = 1, 1.2e-4 at n = 3, 4.5e-4 at n = 10, 1.2e-3 at n = 30.
- **Number of contexts K:** 3.5e-4 at K = 3, 2.8e-4 at K = 6, 1.0e-4 at K = 12.
- **Self prior ρ:** 0.25 is highest (3.4e-4); 0.75 is lowest (9.5e-5).
- **Best-case slice (Δ = 0.4, n = 30):** median 0.010, max 0.028.

**Where the passing cells are:**
- All 6 M0 cells at ≥ 0.02 have **K = 3, Δ = 0.4, n = 30**.
- By K, the maximum R_blind is 0.028 at K = 3, 0.0156 at K = 6, and 0.0082 at K = 12.
- **Interpretive note (not a label change):** K = 3 is the case of counterexample C2. There, much of the Bayes advantage comes from single-shift *exclusion* ("the observed contexts are nominal, so the shift hit the target context"), which is world-attribution to the target, not self-attribution. Where T(h) is the relevant statistic (larger K), or where there is no exclusion structure (M0′), no cell reaches 0.02.

**K4 by the pre-registered rules:**
- **Declared rule:** not fired (max 0.028 ≥ 0.02).
- **Robust rule:** failed (median 2.1e-4, i.e. about 100× below threshold).
- **Label: PASS-weak (regime-dependent).**
- **Substantive reading:** under threshold-averaged utility, the selection pressure for self/world attribution over the better attribution-blind heuristic is ≤ 0.1% of achievable utility in most of the grid. It exceeds 2% only with large effects, abundant data and a structure where the gain is not self-attribution. A favourable fixed threshold raises the best cases to 9–14%, but the medians stay ≤ 0.13%.

## 8. Does an action-dependent or self-modification variant add a genuinely new theorem?

| Variant | Assessment |
|---|---|
| **(a) Action-dependent self-change** (practice, fatigue: s_{t+1} ~ P(· \| s_t, a_t)) | s is a latent state component of a controlled process. Necessity of modelling its dynamics is Richens et al. 2025 for observed states, and Cifuentes 2026 / Nayebi 2026 under partial observability. **Same relabelling** |
| **(b) Action-channel coupling** (the agent's policy shapes the evidence it gets about s) | An action-dependent POMDP. Covered by the controlled and partially observable settings above (Nayebi's tests already include action sequences α). **Not new** |
| **(c) Self-modification of the decision procedure itself** (the agent must predict the policy that is computing the prediction) | Genuinely outside current selection theorems; this is embedded agency (cf. reflective-oracle and self-modification formalisms). A selection theorem here would need new foundations and is a new research direction, which the PI excluded. I found no bounded theorem with substantial new insight. **No rescue proposed** |

## 9. K1–K4 decisions

| Criterion | Decision | Basis |
|---|---|---|
| **K1 (novelty)** | **FIRES (STOP)** | SM-A is Nayebi Thm 3 plus finite-mixture identifiability (extraction) and Cor 4 with regime = shift type (no-aliasing). SM-B is the zero-witness case of Thm 5 / Cor 5, and Nayebi states it in words. The told-shift variant is Richens & Everitt Thm 1–2 with s as a chance variable. The extended-self corollary is ordinary latent non-identifiability (C3). Nothing beyond relabelling the latent "self" |
| **K2 (identifiability)** | **FIRES** for a *separate self-state*. Passes only trivially, for the predictive statistic | C2: cross-context queries can force "target context degraded?" rather than T. C1: minimal optimal memory with no self-coordinate. C3: self is observationally equivalent to an agent-indexed world factor, even with consensus data. Behavioural recovery never licenses an explicit or causal internal self-model (§3) |
| **K3 (vacuity)** | **Not fired**, but moot | The bounds inherited from Nayebi are non-vacuous (aliasing ≤ 2δ̄/c(γ); E(p̂ − p)² ≤ 2δ̄ + 1/(4K²)). That no new bound is needed is further evidence for K1 |
| **K4 (significance)** | **PASS-weak** (regime-dependent; robust criterion failed) | §7.7. Median gap 2e-4; 2.1% of cells ≥ 0.02, all K = 3 exclusion-driven; none in M0′ |

**Overall: SM fails the strict filter.** K1 alone triggers the PI's novelty STOP, and K2 and K4 independently deny breakthrough-level potential.

## 10. Strongest surviving claim and recommendation

**Strongest surviving claim (correct, modest; Category A as a methods remark):**
- In the selection-theorem framework, an agent is required to separate "I changed" from "the world changed" only to the extent that its queried tests in *other* contexts flip bets between those hypotheses. That is a special case of Nayebi's regime tracking.
- Within-context demands never require it.
- No behaviour- or regret-based argument can distinguish a self-model from a model of an agent-indexed environmental factor.
- Under threshold-averaged utility, the regret pressure for such attribution is typically ≤ 0.1%.
- **Methods consequence for AI introspection research:**
  - within-context self-knowledge tests cannot, in principle, detect self-models;
  - cross-context transfer tests reveal predictive attribution but cannot certify a *self* representation without structural (interventional) knowledge of the system.
- This is a footnote-level clarification, not a discovery.

**Action-dependent and self-modification variants (§8):** (a) and (b) reduce to the same theorems. (c), embedded agents modelling their own decision procedure, is genuinely outside current selection theorems, but it is a new research direction with no bounded result in hand. Per instruction, it is not proposed.

**Recommendation: NO-GO** for SM as a breakthrough-oriented direction. Stage 1 should not be pursued. The CL fallback from the moonshot memo is not started; any further direction choice is the PI's decision.

**What would reverse this:** a proof that some *self-specific* structure, absent from world-latent models, changes the necessity conclusion. The only candidate is variant (c), and nothing in hand supports it.
