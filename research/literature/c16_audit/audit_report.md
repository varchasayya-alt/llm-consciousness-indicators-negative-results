# C16 full-text and forward-citation audit (amendments step 0)

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Scope | Full-text reading of the five closest works (memo §2.4); forward-citation crawl of 14 seed papers; targeted prior-work audit of the capacity × consumer-diversity prediction (PI amendment 2) |
| Crawl | `crawl_citations.py` (Semantic Scholar Graph API, unauthenticated). Raw results in `citations_*.json` (14 seeds, 805 citing records). Screening rule fixed in `screen_citations.py` before reading; 99 records kept (`screened_candidates.json`, `screened_list.txt`); every kept record read at title/abstract level; the decisive ones in full text |
| Limits | Semantic Scholar did not resolve Chateau-Laurent & VanRullen 2025 (no citations retrieved), and returned 0 citations for Bertin-Johannet 2026. Its index lags arXiv by days to weeks. Full texts were read through arXiv HTML; the PDF of Goyal et al. was read through ar5iv |

## 1. Verdict

**The exact C16 cell survives, but more narrowly than the memo assumed.** The closest work is now Marincat (2026a, b, c), which I had not found by search in the memo; the crawl surfaced it. Its setting:
- societies of cells that share a **frozen Qwen2.5-0.5B-Instruct** (our model) and a trained rank-8 LoRA;
- cells communicate through two 896-dimensional continuous packets;
- the task is natural-language function composition with **sealed held-out programs**;
- **packet-swap tests** show that restricted evidence visibility yields **value-indexed (content) codes**, while global visibility yields episode-entangled codes.

So "a regime that favours reusable, content-indexed latent codes, with swap tests, in a system built on a frozen pretrained LM" is no longer open. What remains open is C16's own conjunction:

| C16 element | Marincat 2026a–c |
|---|---|
| Re-entry **within one forward pass** of one frozen LM | Separate sequential calls (relay between cells) |
| Frozen **native** consumers (the LM's own skills; no adapter on the consumer) | Shared LoRA trained to execute the operators |
| **Broadcast** of one written content to several **qualitatively different** consumers | Chain: each packet has one consumer; all operators are affine maps over ℤ₁₇ |
| Blind (consumer-agnostic) vs **consumer-addressed** writing at equal capacity | Not compared (visibility of *evidence* is manipulated instead) |
| **Capacity** manipulation, and the capacity × diversity prediction | None |
| **Untrained consumer** | New operator family fails at chance (0.04–0.08); failure localized to *executing* the new operators, while carrier packets transfer (same-value preservation 1.000) |

This contrast also sharpens C16's prediction. In Marincat 2026b, zero-shot failure came from the consumer lacking the operation (it had to be learned by the LoRA). In C16 the untrained consumer U is a native skill of the frozen LM (verbalization), so that failure mode is designed out. U is a cleaner test of whether *content* transfers.

## 2. Full-text checks of the five closest works

| Work | Checked in full text | Bearing on C16 |
|---|---|---|
| **Back Attention** (Yu, Belinkov & Ananiadou, EMNLP 2025; 2502.10835) | One read layer (layer 6) reads all positions of layers 6–top; trained back-attention projections only; no held-out task/pairing tests; no parameter-matched or content-swap controls; only a layer sweep | Mechanism pre-empted; becomes control C3. Not the integration test |
| **Maytié et al. 2024** (RLJ; 2403.04588) | GW trained over a pretrained VAE and normalized attributes, then frozen; policy trained on one modality, tested zero-shot on the other. Ablations: no cycles, CLIP-like, AVAE (all fail transfer). **No capacity sweep, no content swap, no LM, no addressed/point-to-point control** | Held-out source→consumer logic pre-empted outside LMs only |
| **Goyal et al. 2022** (ICLR; 2103.01197, via ar5iv) | All trained from scratch; Atari source→target fine-tuning is the only transfer; **one slot sweep** (SCOFF+SW, bouncing balls: ARI 0.15 / 0.49 / 0.92 / 0.89 / 0.35 at 2 / 4 / 5 / 8 / 10 slots; single run, in-distribution); no post-training ablation or swaps | Gives an in-distribution, single-run "intermediate capacity" pattern. Not a held-out-consumer test and not a theory. Cited as precedent for H-CD (§3) |
| **T²MLR** (Cai et al. 2026; 2607.15178) | Cache from layer ℓ_end at t−1 fused (gated, full width, d×d) before layer ℓ_start at t; retrofit to SmolLM2-1.7B (5,28) trained 1 epoch on OpenMathReasoning (GSM8K 35.8→39.9); no held-out task combinations, no channel ablation/swap, no workspace framing | Retrofitted re-entry precedent only |
| **Latent-channel causal audit** (Zhang & Emu 2026; 2607.26773) | Message conditions: none / current / other-example (K = 4) / self-generated. Five measurements (PS, PL, CIC, CAG, SSG). No held-out pairings, no capacity results, no broadcast; separate agents | Content-vs-generic decomposition adopted. A second audit (2608.04893) shows "a large cache effect need not be a pairing effect". Draft-KV (2609.34754) reports message replacement changes accuracy by ≤ 0.60 points across five published methods. Hence CT, not accuracy, is primary |

## 3. Forward-citation findings (recent, relevant)

| Work | Relevance | Status for C16 |
|---|---|---|
| **Marincat 2026a** "What You Can't See Is What You Learn" (2608.20054) | See §1. Restricted (masked) vs global visibility in 4-cell societies on frozen Qwen2.5-0.5B; held-out compositions; value-indexed packets in 6/6 audited restricted societies; global model's same-value transplants 0.12–0.25 | **Closest precedent.** Narrows conceptual novelty; does not fill the cell |
| **Marincat 2026b** "Portable Semantics, Private Dialects" (2609.11365) | Strict zero-shot transfer to a new operator family fails (failure in operator execution). Interfaces from different initializations are not interoperable. A globally trained interface acts as a negative-transfer prior (0.17 vs 0.86 with a fresh interface) | Cautionary prior for H2; motivates native U and fresh-interface controls |
| **Marincat 2026c** "…Still What You Learn" (2609.17637) | Preregistered 60-society confirmation of the masking advantage (median paired differences 0.85–0.86) | Shows the regime effect replicates |
| One-to-many emergent communication (2024) | Broadcasting to many listeners alone does **not** induce compositionality; listeners with **different interests** and coordination do | Precedent for the *diversity* half of H-CD (from scratch, emergent language) |
| Emergent compositional communication for latent world properties (2604.03266) | Multi-agent structure, not bandwidth, drives compositional protocols over frozen video features | Same direction: diversity of receivers matters more than bandwidth |
| From Signals to Structure (2607.00233) | LLM agents in a Lewis game: an IB argument predicted optimal capacity = number of objects; instead the bottleneck was a fragility point and **surplus capacity was generally better** | Evidence against a capacity *peak* |
| DiscoLoop (2607.00341) | Bridge entity decodable after the first loop but poorly aligned with its token embedding; a training-free realignment nearly closes the gap | Supports C16's premise that re-entered content must be in **native format** (V2 reachability; the format-geometry account in §4) |
| Transformers Stop Thinking Too Early (2609.36585) | A rank-8 LoRA at one early layer (all weights frozen) extends in-context reference-following dramatically | C5 (parameter-matched LoRA) is a **strong** competitor; keep it mandatory |
| Why Knowing Both Hops Is Not Enough (2608.07261); Loop, Think & Generalize (2604.07822) | Two-hop failures arise from cross-layer mismatch; recurrence helps systematic generalization (from scratch) | Background for the native gap |
| REST (2609.36159) | CE-only training of latent thoughts collapses thoughts and retains irrelevant information | Risk for CE-only workspace training; monitored via slot decodability (descriptive) |
| Draft-KV (2609.34754); When Does Latent Communication Pay? (2608.04893); CacheBack receiver-conditioned (2609.32046); XBridge (2608.11676) | Inter-model latent communication with message-replacement audits; receiver-conditioned (addressed) compression | Inter-model, not internal; CacheBack is an addressed-channel analogue (training-free) |

No other retrieved record combines within-pass internal re-entry in a frozen LM with held-out pairings or untrained consumers and matched non-broadcast controls.

## 4. Audit of the capacity × consumer-diversity prediction (amendment 2)

**The prediction as posed:**
- (i) capacity below the content requirement prevents transport;
- (ii) very large capacity permits consumer-specific answer bundles;
- (iii) an intermediate bottleneck plus multiple training consumers favours reusable content;
- (iv) held-out integration therefore peaks in a predictable capacity/diversity region.

**Prior work by component:**

| Component | Prior work | Status |
|---|---|---|
| (i) Lower bound | Resnick et al. 2020 (AAMAS): evidence for the bottom of a capacity/bandwidth range for compositional emergent language | Precedented in emergent language; **new** for a retrofitted channel with an *a priori measured* native-content rank |
| (ii) Upper limb | Resnick et al. 2020: "we curiously do not find evidence for the top part of the range"; models up to 1.5M parameters did not overfit as predicted. 2607.00233: surplus capacity generally better. Goyal 2022: one in-distribution slot sweep peaking at 5 slots | **Contested; two direct attempts failed to find it** |
| (iii) Diversity → reusable content | Johnston & Fusi 2023 (multi-task → abstract representations); Tripuraneni et al. 2020 (task diversity); Vafidis et al. 2025; Mu & Goodman 2021; one-to-many emergent communication (2024); 2604.03266 | **Precedented** (from scratch / emergent communication) |
| (iv) Peak region | No study found that sweeps a channel bottleneck against the number of training consumers and shows an intermediate optimum for unseen-consumer use | Open, but weakly supported |

**Theoretical defensibility. As stated, the peak is not defensible in information terms.**
- The answer bundle (f₁(X), …, f_n(X)) is a function of X, so H(bundle) ≤ H(X). In bits, a bundle never needs more capacity than the content, and an information-bottleneck pressure (Kharitonov et al. 2020, ICML: emergent channels minimize input–message information within task demands) favours bundles at **every** capacity.
- A capacity window can arise only from **format**: the reader is a linear map into the frozen model's residual stream, and frozen consumers compute f only from **native-format** inputs.
- Let r_X be the rank needed to deliver X in native format, and r_B(S) the rank needed to deliver the native-format answer bundle of consumer set S. Then content is the *only* linear solution serving every consumer in S when r_X ≤ c < r_B(S). Outside that window, nothing works (c < min) or both work (c ≥ r_B).
- This "format-geometry" account is new in this form.
- It makes a peak prediction only if **r_B(S) ≫ r_X**. That is an empirical property of the model's native geometry, measurable before any training (Stage 0, W-gates).
- Two competing accounts make different predictions on the same grid:
  - an **information-bottleneck** account: transfer depends on whether S jointly determines f_U; no upper decline;
  - a **native-copy ("lingua franca")** account: the writer copies native X whenever c ≥ r_X; no diversity effect and no decline.

**Recommendation (adopted in the S0/S1 preregistration):**
1. Do **not** adopt the peaked window as a central *directional* hypothesis.
2. Adopt **H-CD as a preregistered three-account discrimination** on a capacity × consumer-set grid, with coordinates (r_X, r_B(S)) measured a priori in Stage 0. The defensible core — the lower bound and the diversity effect — is primary. The upper limb is a secondary contrast whose prior is weak.
3. Stage 0/1 validate (a) whether r_X and r_B(S) can be measured reliably, (b) whether the window is wide enough to test (r_B ≥ 4 r_X), and (c) the seeds and items needed for power and complete-gate FPR. All of this uses no Stage-2 outcome data.

## 5. Literature database

The new entries are added to `literature/literature_db.json` (verified web-2026-10-08):
- Marincat ×3;
- Zhang & Emu 2026;
- 2608.04893; Draft-KV; CacheBack;
- DiscoLoop; Stop Thinking Too Early; Two-Hop generalization; REST;
- one-to-many emergent communication; 2604.03266; 2607.00233;
- Resnick 2020; Kharitonov 2020; Johnston & Fusi 2023; Tripuraneni 2020; Mu & Goodman 2021; Vafidis 2025;
- ICAE; xRAG; ThoughtComm.
