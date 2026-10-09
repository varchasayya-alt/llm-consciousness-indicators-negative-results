# C16 decision memo: an engineered shared workspace in a pretrained LM

| Field | Value |
|---|---|
| Date | 2026-10-08 |
| Status | **Design and novelty memo only.** No code, downloads, training or model runs. Nothing here is preregistered or frozen |
| Trigger | PI instruction, 2026-10-08: close C15; preserve the A-stage and R2 as negative / methodological results; no R3; open no sealed material; return a Phase 1–3 design memo for C16 |
| Premise | C16 does **not** assume that native LLM workspaces are fragmented or absent. C15/R2 established neither (R2 ended P4, an assay failure). C15 motivates only one choice: an explicitly built, causally isolable mechanism instead of an uncertain native one |
| Claims discipline | Level 1: mechanistic results. Level 2: correspondence to GWT/GNW, only where stated. Level 3 (phenomenal consciousness): never claimed |
| Literature | About 40 web searches and 12 page fetches on 2026-10-08, mostly at abstract level (§2.1). 40 new entries were added to `literature/literature_db.json` (CSV and BibTeX regenerated) |

---

## 0. Verdict

1. **The exact narrow cell is probably open (novelty confidence ≈ 0.6).** I found no work that does all of the following:
   - retrofits a capacity-limited internal workspace into a frozen pretrained LM;
   - tests transfer to producer–consumer pairings, or to a consumer operation, that were held out from workspace training;
   - compares against equal-capacity non-broadcast controls;
   - uses content-swap tests.
2. **The mechanism is not new.**
   - *Back Attention* (Yu, Belinkov & Ananiadou, EMNLP 2025) already trains a late-to-early, cross-position re-entry module in frozen LLMs. TwoHop accuracy goes from 11.5 to 47.8 on Llama3-8B.
   - T²MLR (2026), retrofitted recurrence (2025–26), Coconut and TransformerFAM are close neighbours.
   - All of C16's novelty would lie in **the test of an integration principle**. None lies in the module.
3. **A premise needs correcting.** In a residual transformer, the residual stream already broadcasts within a position, and attention already broadcasts across positions at the same depth.
   - The only native separation is depth × position. Content computed late at one position cannot reach early or mid layers at later positions, except through emitted tokens.
   - So what an engineered workspace adds is **capacity-limited re-entry**.
   - "Broadcast vs private" must therefore be defined against consumer-addressed and private-line channels, not against writes to the residual stream (§1.2).
4. **Significance is the weak point.**
   - The expected outcome is rung 3–4 of the significance ladder.
   - P(defensible Level 5) ≈ 0.2. P(Level 6) ≤ 0.05.
   - The one question whose answer is not built into the design: suppose a workspace is written *before* the downstream use is known (blind) and is capacity-limited. Does it learn **content** that an untrained consumer can use, or a bundle of answers for the trained consumers? And does it beat capacity-unrestricted learned re-entry?
5. **Recommendation: do not adopt C16 as the main line yet.** Approve only a cheap kill sequence:
   - Stage 0: instrument validation and reachability on the cached Qwen2.5-0.5B-Instruct (about 2 CPU-h, forward passes only).
   - Stage 1: synthetic satisfiability and complete-gate null calibration (about 2–3 CPU-h).
   - Promote C16 to the main direction only if both pass. The full study is about 55 CPU-h, with no downloads.

---

## 1. Premises (adversarial)

### 1.1 A native integration gap exists independently of C15

**The constraint.** In a decoder-only transformer, a representation formed at layer L at position t reaches later positions only at layers above L. A computation that needs it earlier (for example, fact retrieval in early or mid MLPs) can get it only through emitted tokens, i.e. the verbal loop.

**Evidence that this matters:**
- **Biran et al. 2024 ("Hopping too late").** The bridge entity is resolved, but the second hop starts too late. Patching later-layer states back into earlier layers ("back-patching") fixes up to 66% of failures.
- **Yu et al. 2025 (Back Attention).** A trained late-to-early module gives large gains on two-hop questions and on arithmetic.
- **Kawada & Kellis 2026.** Verification states can be decoded but have little causal effect on answers.
- **Kirin 2026 (looped Ouro).** The direction that predicts success does not produce success when written back.

**Consequence.** The gap is real. But "a learned re-entry channel rescues composition" has already been shown. C16 has to ask *which kind of channel* yields **general** integration.

### 1.2 What "separate functions" and "broadcast" can mean inside one transformer

There are no separable modules, so a "function" has to be defined operationally:
- **Producer:** a computation whose result X is not in the surface input and becomes decodable at late layers over a producer segment.
- **Consumer:** a computation at later positions that needs X as an input at early or mid layers.

Every additive write into the residual stream already reaches every downstream layer at that position. So "private channel" cannot mean "write into one layer"; that would be broadcast anyway. The contrasts that do mean something, and that §4 adopts, are:

| Contrast | One side | Other side |
|---|---|---|
| Who the content is for | **Consumer-agnostic** content: the writer is blind to future use, and one shared reader serves everyone | **Consumer-addressed** messages: the writer is told which consumer will read |
| Medium | One shared medium | **Private lines**: one dedicated line per consumer type |
| Capacity | **Capacity-limited** re-entry | **Unrestricted** re-entry |

### 1.3 This reverses D1, and the reversal needs a justification

**What D1 said (2026-10-01).** Prefer measured indicator signatures over architecture-by-design, because scaffolds can satisfy indicators without realising them (Goldstein & Kirk-Giannini 2024). Its revisit clause: "if a designed architecture is needed as the object of measurement".

**When C16 qualifies.** Only if the designed workspace is the **object of causal measurement**: ablation, content swaps, matched alternatives and held-out tests. Never if the design itself is the evidence.

**Excluded explicitly:** "we built a GW-like module, therefore GWT indicators are satisfied".

---

## 2. Phase 1: novelty audit

### 2.1 Scope, method and limits

**Families searched on 2026-10-08:**
- GW-inspired neural architectures (the Goyal lineage, RIMs, DVNC, modularity studies);
- the VanRullen / Chateau-Laurent GW lineage;
- Shang's Theater of Mind / Global Workspace Agents;
- CTM-AI;
- J-space (Gurnee et al. 2026) and follow-ups;
- recurrent and shared-memory transformers (RMT/ARMT, LM2, MemoryLLM, TransformerFAM, Feedback Transformer);
- memory tokens, registers and latent scratchpads (Coconut, pause tokens);
- latent bottlenecks (Perceiver, Q-Former);
- mixture-of-experts and global routers;
- adapters in pretrained LMs (LoRA, ReFT, LST, CALM);
- late-to-early re-entry retrofits (Back Attention, back-patching, T²MLR, retrofitted recurrence, CFL);
- global tokens (Longformer);
- blackboard and agent-level workspaces (LbMAS, Salemi et al., GWA, UMM, SRMT, SAF);
- latent communication between LM instances (activation communication, LatentMAS, C2C, Interlat, Group Think, Hogwild!, Latent Agents, and a causal audit of latent channels);
- integration of metacognitive signals (Kumaran et al. 2026, Kawada & Kellis 2026).

**Limits:**
- I read mostly abstracts, search snippets and fetched HTML, not full texts. Back Attention was read from its HTML version.
- **No citation crawl** (Semantic Scholar or Google Scholar) of the key papers was done. This is the main remaining risk, and it is step 0 of the proposed next stage (§6).
- The latent-communication field alone has 18 or more methods (Liu 2026 survey, cutoff 2026-07-15). I sampled it; I did not enumerate it.
- Search engines lag arXiv by days to weeks, so work from about late September 2026 onward may be missing.
- Classic references marked "memory" in the literature database were not re-verified this session.

### 2.2 Comparison matrix

Axes:
1. pretrained monolithic LM, retrofitted (vs trained from scratch, or separate modules);
2. internal neural workspace (vs external text or multi-agent);
3. capacity-limited;
4. shared broadcast (vs local or point-to-point);
5. causal ablation, rescue or content swap;
6. held-out functions or pairings;
7. same-model, matched architectural controls;
8. general coordination mechanism (vs task-specific).

Key: ✓ yes · ~ partial · ✗ no · ? cannot tell from what was read.

| Work | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | What it pre-empts |
|---|---|---|---|---|---|---|---|---|---|
| Goyal et al. 2022, shared global workspace | ✗ | ✓ | ✓ | ✓ | ~ | ✗ | ~ (vs pairwise attention) | ✓ | A limited-capacity shared workspace with competitive writes beats pairwise communication (from scratch) |
| RIMs 2021; DVNC 2021 | ✗ | ✓ | ~ / ✓ | ~ | ✗ | ~ (OOD / systematic) | ✓ | ✓ | Bottlenecked communication between modules helps systematic generalization (from scratch) |
| Mittal 2022; Schug 2024 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ (held-out module combinations, Schug) | ✓ | ~ | The held-out-combination protocol; modular architectures often fail to specialize |
| Perceiver 2021; "the Perceiver is a functional GW" (CogSci paper) | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | A latent bottleneck as GW (descriptive) |
| Devillers 2024; Maytié 2024; Bertin-Johannet 2026 | ~ (frozen unimodal encoders, not an LM) | ✓ | ✓ | ✓ | ~ | ✓ (zero-shot cross-modal policy transfer; unseen modality) | ~ (CLIP-like variants) | ~ | **A shared workspace over frozen specialists supports zero-shot transfer to a held-out source→consumer pairing** (multimodal / RL) |
| Chateau-Laurent & VanRullen 2025 | ✗ | ✓ | ✓ | ~ (gated routing through the GW) | ? | ✓ (unseen addition operations) | ✗ (vs LSTM / Transformer) | ✗ | GW routing transfers to held-out operations (toy arithmetic, from scratch) |
| Dossa 2024 GW agent; Phua 2025 toy ablations | ✗ | ✓ | ✓ | ✓ | ✓ (Phua: workspace lesion) | ✗ | ~ | ~ | An engineered workspace with lesions, in toy agents |
| SRMT 2025; SAF 2022 | ✗ | ~ (agent-level, neural) | ~ | ✓ | ~ | ~ | ~ | ✗ | GW-inspired shared memory between agents |
| Gurnee et al. 2026 (J-space) and follow-ups | native | ✓ | ✓ (native) | ✓ (broadcast heads) | ✓ | ~ | ✗ (not engineered) | ✓ | Descriptive and causal claims about a *native* workspace; no engineering |
| Shang 2026 GWA; CTM-AI 2026; UMM 2025; LbMAS / Salemi 2025 blackboards; Goldstein & Kirk-Giannini 2024 | LLMs as parts | ✗ (external) | ~ | ✓ | ✗ | ✗ | ✗ | ✓ | GWT-style scaffolds around LLMs (risk of box-checking) |
| Ramesh & Li 2025 (communicating activations) | ✓ (frozen, zero params) | ~ (between instances) | ✗ | ✗ | ~ | ✗ | ~ | ~ | **Native activations of one model are mutually readable without training**, so a native-transplant baseline is required |
| LatentMAS 2026; C2C 2025; Interlat 2026; Group Think / Hogwild! 2025 | ✓ | ~ (between instances or threads) | ✗ | ✓ (shared KV) | ~ | ✗ | ~ | ✓ | Unrestricted latent sharing between LLM processes |
| Latent-channel causal audit 2026 (2607.26773) | ✓ | ~ | ✗ | ✗ | ✓ (message replacement; content vs generic) | ✗ | ✓ | ~ | **Splitting a latent message's effect into a content-specific part and a generic part** (methodology) |
| Latent Agents 2026 | ✓ (post-trained) | ✓ | ✗ | ✗ | ✓ (steering) | ✗ | ✗ | ~ | Multi-agent debate internalised in one model |
| CALM 2024 | ✓ (two frozen LMs) | ✓ | ✗ | ✗ (the anchor model reads the augmenting model) | ~ | ~ | ~ | ~ | A small learned channel between frozen LMs enables new capabilities |
| **Back Attention 2025** | ✓ | ✓ | ✗ (full-width states) | ✗ (one read layer) | ~ (layer sweep only) | ✗ | ✗ | ~ | **Retrofitted late→early, cross-position re-entry rescues latent composition in frozen LLMs** |
| Biran 2024, back-patching | ✓ | ✓ | ✗ | ✗ | ✓ (oracle patch) | ✗ | ✗ | ✗ | Late→early re-entry fixes two-hop failures (oracle) |
| T²MLR 2026; retrofitted recurrence 2025/26 (McLeish; Shapiro) | ✓ | ✓ | ~ | ✗ | ~ | ~ (Shapiro: transfer to verbal renderings) | ✓ | ✗ | Retrofitted recurrence improves reasoning. In Shapiro, one adapter hosted one operation |
| Coconut 2024 / TransformerFAM 2024 / Feedback Transformer 2020 / CFL 2024 | ✓ / ✓ / ✗ / ? | ✓ | ✗ / ✗ / ✗ / ✓ (compact vector) | ~ (via inputs) / ✗ / ✓ / ✓ | ✗ | ✗ | ~ | ✓ | Latent re-entry or feedback memory as working memory |
| RMT/ARMT; LM2; MemoryLLM; Prometheus Mind; Bottlenecked Transformers | ~ / ✗ / ✓ / ✓ / ~ | ✓ | ✓ (fixed slots) | ~ | ✗ | ✗ | ~ | ~ | Retrofitted latent memory for long context or knowledge |
| Global tokens (Longformer), registers, memory tokens | ✓ (Longformer built from RoBERTa) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ~ | Global tokens retrofitted for long context |
| Adapters / PEFT (LoRA, ReFT, LST, AdapterFusion); Q-Former / BLIP-2 | ✓ | ✓ | ✓ (low rank / few queries) | ✗ | ✗ | ~ | ✓ | ✗ | Capacity-limited learned interfaces to frozen models (point-to-point) |
| MoE and routers (Shazeer 2017; DeepSeekMoE shared experts; Chain-of-Experts 2025; MoLE) | mostly ✗ | ✓ | n/a | ✗ (routing) | ✗ | ✗ | ~ | ✓ | Central routing: the "task-routing" alternative |
| Kumaran et al. 2026 (confidence → abstention) | ✓ | ✓ native | — | — | ✓ (steering) | ✗ | — | — | Confidence → abstention is **already natively integrated**, so it is a poor C16 function |
| Kawada & Kellis 2026 | ✓ | native | — | — | ~ | ✗ | — | — | Verification is decodable but not integrated: a native gap (possible later extension) |

**No row fills columns 1–7 together.** The nearest misses:
- Back Attention has 1 and 2 but fails 3, 4, 6 and 7.
- Maytié 2024 has 2–4 and 6 but fails 1.
- Goyal 2022 has 2–4 and 8 but fails 1 and 6.

### 2.3 Claims that C16 can no longer make

| Claim | Status | Precedent |
|---|---|---|
| A retrofitted re-entry channel lets a frozen LM compose latent results it otherwise cannot | **Already made** | Back Attention 2025; back-patching 2024; T²MLR 2026 |
| A limited-capacity shared workspace beats pairwise communication | Already made (from scratch) | Goyal 2022 |
| A shared workspace over frozen specialists supports zero-shot transfer across held-out pairings | Already made (multimodal / RL) | Maytié 2024; Bertin-Johannet 2026 |
| Latent channels carry content-specific vs generic influence (causal audit) | Methodology already exists | 2607.26773 |
| Activations of the same model are interoperable without training | Already made | Ramesh & Li 2025 |
| LLMs have a native workspace-like subspace | Already made (descriptive) | Gurnee 2026 |
| Building a GW scaffold satisfies GWT indicators | Not a contribution (box-checking) | Goldstein & Kirk-Giannini 2024; GWA; CTM-AI |
| **Inside a frozen pretrained LM, a blind capacity-limited workspace transports content into held-out producer–consumer pairings and to an untrained consumer, beating equal-capacity addressed and private channels and unrestricted re-entry** | **Not found** | — |

### 2.4 Is any of the five closest papers fatal?

1. **Back Attention (EMNLP 2025).**
   - *What it shares:* the same insertion logic. A lower layer of a frozen LLM reads higher-layer states across positions.
   - *Why it is not fatal:* full width, a single read layer; no held-out task, pairing or consumer tests; no capacity, broadcast or matched controls; no content swaps.
   - *Consequence:* a Back-Attention-type arm (C3) becomes mandatory. If C3 matches the workspace on the held-out tests, the capacity-limited workspace adds nothing.
2. **Maytié et al. 2024 (with Devillers 2024 and Bertin-Johannet 2026).**
   - *What it pre-empts:* the *logic* of held-out source→consumer transfer through a shared workspace.
   - *Why it is not fatal:* the specialists are frozen unimodal encoders feeding an RL policy, not the internal functions of one pretrained LM. The transfer is cross-modal (same content, different source), not reuse by qualitatively different consumers. There are no equal-capacity addressed or private controls.
3. **Goyal et al. 2022.**
   - *What it pre-empts:* the architectural principle, in systems trained from scratch.
   - *Why it is not fatal:* no retrofit, no held-out pairing tests, no content swaps.
4. **The latent-communication family plus the causal audit (2024–26).**
   - *What it pre-empts:* splitting a latent message's effect into content-specific and generic parts, between LM instances.
   - *Why it is not fatal:* the channels connect separate instances, are unrestricted, and are never tested on held-out pairings.
   - *Consequence:* C16 must **not** be built as parallel encapsulated streams (candidate D), or it collapses into this family.
5. **T²MLR and retrofitted recurrence.**
   - *What it pre-empts:* "retrofitted recurrence helps reasoning".
   - *Why it is not fatal:* no workspace, held-out or broadcast tests.
   - *Useful negative pointer:* in Shapiro 2026, one adapter hosted only one operation.

### 2.5 Novelty verdict

| Aspect | Rating | Notes |
|---|---|---|
| Exact cell (decision item 1) | **Probably open, confidence ≈ 0.6** | Lowered by abstract-level reading, the missing citation crawl, and how fast the field is moving |
| Mechanism | **Low (≈ 0.2)** | See §2.3 |
| Test design | **Moderate** | The conjunction in §2.3's last row is what is new |
| Durability | **Low** | Since Gurnee et al. (July 2026), several groups are building J-space tooling, and latent-communication methods appear monthly. Substantial scoop risk over 6–12 months. The most likely scoop: "Back-Attention-style re-entry with cross-task generalization" |

### 2.6 Significance audit against the ladder

| Rung | What C16 would need | Realistic? |
|---|---|---|
| 1 Descriptive | Slot contents can be decoded | Trivial |
| 2 Dissociation | The workspace helps some consumers but not others | Easy, but says little |
| 3 Causal necessity | Ablating the trained workspace removes the gain | Likely, if training works |
| 4 Sufficiency / rescue | Adding the workspace enables compositions the frozen model fails at | Likely, **but Back Attention already shows the generic version** |
| 5 Integration principle | **Blind content, written once, is causally usable by qualitatively different consumers, including held-out pairings and an untrained consumer. Content swaps redirect all of them. Equal-capacity addressed or private channels and unrestricted re-entry do not reproduce this** | Possible, P ≈ 0.2 |
| 6 Developmental / architectural principle | A predicted *condition* for such integration, confirmed out of sample. Examples: a capacity-window × consumer-diversity law, or developmental path dependence (shared from the start vs private first, then shared) | P ≤ 0.05 at $0 scale |

**Adversarial points:**
- **Parts of the outcome are predictable.** That addressed and private controls fail on an *untrained consumer* is close to true by construction. The informative unknown is whether the workspace itself succeeds: whether a blind workspace learns content rather than an answer bundle.
- **Generality may be inherited, not created.**
  - Ramesh & Li show that a model's own activations are interoperable across instances. So success on untrained consumers might come from the pretrained model's native representational "lingua franca", not from the workspace.
  - Hence the need for a zero-parameter native-transplant baseline (C6) and an unrestricted re-entry baseline (C3).
  - If C3 or C6 matches the workspace, the result supports a **re-entry** principle (closer to GNW recurrence, or recurrent processing theory), not a **limited-capacity broadcast** principle.
- **Toy-scale risk.** A 0.5B model on synthetic content tasks will be read as a toy unless the principle comes with a prediction that is verified out of sample (the capacity window, H4 in §4.7).
- **Consciousness relevance (Level 2) is modest.**
  - A positive result bears on GWT's *functional* claim: limited-capacity broadcast of content supports flexible novel combination. It says nothing about consciousness.
  - It would turn indicators GWT-2 and GWT-3 into testable functions rather than boxes to tick. That is modest but real.

**Planning probabilities (subjective):**

| Event | Probability |
|---|---|
| Stage 0 passes on Qwen2.5-0.5B | ≈ 0.6 |
| Given a pass: at least W-RE (re-entry principle) | ≈ 0.55 |
| Given a pass: at least W-BC (broadcast principle) | ≈ 0.3 |
| Given a pass: W-GW (full principle, including capacity) | ≈ 0.15–0.2 |

**Significance confidence (decision item 4): low to moderate.**

---

## 3. Phase 2: architecture search (design only)

### 3.0 Requirements derived from the audit

| Code | Requirement |
|---|---|
| R-a | **Blind write.** The write window closes before the consumer cue. This is enforced in code: a unit test shows the slot state is bit-identical whatever cue follows. It blocks the "precompute the answer" / local-controller shortcut |
| R-b | **Explicit, manipulable capacity**, in units that allow a capacity-window prediction (H4, §4.7) |
| R-c | **Reads at several sites through one shared, consumer-agnostic reader**: the same parameters at every read layer and position, with no task input |
| R-d | **Causal isolability.** Slot state can be zeroed, mean-replaced, swapped across episodes, interpolated or replaced by norm-matched noise, all without retraining |
| R-e | **Frozen base.** Every LM weight is frozen. Read gates are zero-initialised, so at initialisation the model is bit-identical to the base (tested) |
| R-f | **Matched by construction.** Workspace arms and their controls have equal total trainable parameters (±10%) and equal capacity. Never matched empirically (C15-R2 lesson: empirically KL-matched controls lost coverage when the material changed) |
| R-g | **CPU-feasible** on a cached model, with no downloads |

### 3.1 Model

- **Primary: Qwen2.5-0.5B-Instruct** (already cached; 24 layers, d = 896; Apache-2.0).
  - It is the smallest cached model.
  - Shapiro 2026 retrofitted recurrence into this exact model with a 6M-parameter adapter.
  - Measured forward cost is about 0.35–0.5 s per prompt in fp32 (D12). bf16 is 4–6× slower on this CPU, so use fp32.
  - Throughput ≈ 150–240 tok/s, extrapolated from the measured 59 tok/s of Qwen3-1.7B at batch 8.
- **Fallback and replication: Qwen3-1.7B** (cached). Used only if the 0.5B model fails Stage-0 competence (pre-declared order), or to replicate the decisive contrasts.
- **Not requested:** Qwen3-0.6B, which would need a download.

### 3.2 Candidate architectures

**A. Re-entrant Slot Workspace (RSW): the "mental blackboard"**

| Element | Specification |
|---|---|
| Insertion | Write band at a late layer L_w (expected around 14–18 in the 0.5B model; chosen in Stage 0 as the layer where X is most decodable on calibration items). Three read layers, all ≤ 12, chosen in Stage 0 by text-oracle reachability |
| Computation | Segment recurrence, as in RMT. The producer segment runs through all layers with its KV cache kept; the slots S ∈ ℝ^{K×d_w} are then fixed; the consumer segment runs with reads enabled. No second forward pass, so compute is unchanged |
| Write rule | For each write-window position t: c_t = W_in · LN(h_{L_w,t}). Positions compete for slots, as in Goyal et al.: α_{k,t} = softmax over k of (q_k · c_t / √d_w), sparsified to the top 2 positions per slot. Gated update: s_k ← (1 − z_k) · s_k + z_k · Σ_t α_{k,t} c_t, with z_k = σ(w_z · [s_k ; Σ α c]). Then layer norm and fixed Gaussian noise of variance σ², which sets the information capacity |
| Read rule | At every consumer position u and every read layer l: β = softmax(W_q · LN(h_{l,u}) · Sᵀ / √d_w); h_{l,u} += g_l · W_o (β S). W_q and W_o are **shared across read layers** (one code); g_l is a zero-initialised scalar |
| Persistence | S persists through all later segments of the episode until the next write window. This allows reuse by several consumers and survives distractor text |
| Broadcast, operationally | The same S is readable at every read layer and position, by every consumer type, through one reader with no task input. Tested by ablating the reader layer by layer and by checking content-swap effects across consumers |
| Frozen / trainable | Frozen: the whole LM. Trainable: about 3 · d · d_w parameters (≈ 86k at d_w = 32), plus slot queries and gates |
| Guards against becoming an extra MLP or shortcut | Blind write (R-a); shared reader; capacity limit; at least 3 consumers per producer in training; content swaps; the self-recurrence (C4) and LoRA (C5) controls |
| Cost | See §4.11 |

**B. Low-rank shared state (the "broadcast bus")**
- A single vector z ∈ ℝ^r (r = 4–64), written as a sparsemax-weighted sum of P · LN(h_{L_w,t}) over the write window.
- Read is query-free: h_{l,u} += g_l · U z at every read layer, with U shared across layers.
- *For:* the simplest option and the closest to a literal blackboard. Trivial to swap or interpolate. Similar to CFL.
- *Against:* every consumer receives the same additive vector, which invites a generic "mode switch" (lesson R7: engineered interventions create artifacts). Consumers cannot take up content selectively.
- *Use:* best kept as an **ablation of A** (does selective reading matter?).

**C. Specialist-competition hub (a GNW "ignition" variant)**
- Several writer heads sit on distinct layer bands (early, mid, late "processors") and propose messages.
- A low-temperature or straight-through top-1 competition decides which messages enter the hub.
- The hub broadcasts to read layers in all bands. Optionally, it re-enters across segments ("reverberation").
- *For:* explicit competition between processors, an ignition-like nonlinearity, the closest match to GNW.
- *Against:* the processors are layer bands, not functions. Many knobs. Hard-selection gradients are unstable on CPU. Causal readings are harder.
- *Use:* better kept for a later ignition study than for the integration test.

**D. Encapsulated parallel streams with global workspace tokens**
- Producer and consumer contexts run as separate streams under a block-diagonal attention mask.
- K learned workspace tokens attend to every stream at write layers, and every stream attends to them at read layers.
- *For:* perfect encapsulation, with no native route; the textbook GWT picture; natural private-line controls.
- *Against:* this is latent multi-agent communication with shared weights (LatentMAS, Group Think, the causal audit), so it is the least novel. Surgery on the attention mask and RoPE of a pretrained model adds artifacts (R7). The native gap becomes artificial. It costs 1.5–2× the tokens.

**E. Discrete re-entrant bottleneck (two passes)**
- Pass 1: a learned dictionary of 512 codes encodes the write-band state, keeping the top k (k = 2–4). The workspace is k code indices plus their magnitudes, so capacity is exactly k · log₂512 bits plus magnitude bits.
- Pass 2: re-run from the read layer at consumer positions, with the decoded codes added. This is Back-Attention-like, but discrete and capacity-limited.
- *For:* capacity measured in bits; symbolic, swappable content; the strongest operationalisation of "limited capacity".
- *Against:* 2× compute; unstable VQ / straight-through training; codebook collapse; the dictionary could memorise answers.

**Rejected:**
- *A workspace defined by the J-lens (J-space directions).* The instrument is not validated at 0.5B, and repeating C15's instrument risk is unacceptable.
- *MoE or LoRA routers.* They test routing, not broadcast.
- *Text scratchpads.* External, not internal.

### 3.3 Ranking

Criteria: isolability (R-d); whether the design answers the decisive question (blind content vs answer bundle, held-out pairings, untrained consumer); novelty distance from §2.3; artifact risk (R7); CPU cost.

| Rank | Architecture | Why |
|---|---|---|
| 1 | **A. RSW** | Meets R-a to R-g. Segment recurrence needs no second pass. Distinct from Back Attention: capacity-limited, multi-site, shared reader, blind write |
| 2 | E. Discrete bottleneck | Best operationalisation of capacity, but 2× compute and unstable VQ training. Candidate to replicate the capacity claim if A is positive |
| 3 | B. Low-rank state | Cheapest. Built into A's protocol as the query-free ablation |
| 4 | D. Parallel streams | Cleanest encapsulation, but collapses into the latent-communication family and adds mask/RoPE artifacts |
| 5 | C. Competition hub | Most GNW-like, but its processors are layer bands. Keep for an ignition follow-up |

**Recommended (decision item 6): A (RSW), with B as a built-in ablation.**

---

## 4. Phase 3: decisive experiment (design only)

### 4.1 Choosing the functions

Each candidate function was screened on five criteria:
1. exact, discrete ground truth, so a content swap has an exactly predicted consequence;
2. native competence at 1.7B or smaller;
3. a native gap (composition fails without chain of thought);
4. qualitatively distinct internal machinery across consumers;
5. not already natively integrated.

| Candidate | Verdict | Reason |
|---|---|---|
| Uncertainty / confidence → abstention | **Reject (for the core)** | Already natively integrated: Kumaran et al. 2026 show confidence causally drives abstention. Scalar content gives a weak swap test. The consumers would be answer *policies*, exactly where the local-controller objection bites |
| Verification / error signal → revision | **Defer (possible Stage-3 extension)** | A real native gap (Kawada 2026). But the content is about 1 bit, the consumers are policies (keep, revise, abstain), and small-model competence is weak |
| Planning, goal representation, information seeking, compute allocation | Reject | Not cleanly measurable at 0.5B on CPU; the consumers would be policies |
| **Retrieval and composition over discrete content** (numbers; entities as fallback) | **Adopt (core)** | Exact ground truth; the swap test predicts f(X′) exactly; consumers use distinct machinery; a documented native gap; cheap on CPU |

**Primary content domain: two-digit numbers.** X ranges over 10–99, about 90 values, which gives enough clusters for statistics.

Producers (X never appears on the surface):

| Code | Producer | Example |
|---|---|---|
| P1 | Arithmetic | "Ana has 23+14 marbles" |
| P2 | In-context binding | "Ana's code is 37 … Ben's code is 52 …", then "Ana's code" |
| P3 | Positional selection | "the third of 12, 48, 37, 66" |

Consumers:

| Code | Consumer | Machinery |
|---|---|---|
| C1 | Successor (X + 1) | Arithmetic |
| C2 | Parity (even / odd) | Feature readout |
| C3 | Keyed in-context lookup ("Codes: 37→blue, 52→red … colour for Ana's number") | Attention-based retrieval |
| C4 | Magnitude comparison (X > 50?) | Magnitude judgment |
| U | Verbalisation (X → English number word) | Lexical generation. **The held-out consumer, never used in workspace training** |
| copy | Emit X | Positive control only |

**Fallback domain: countries** (pre-declared; used only if numbers fail Stage-0 competence).
- Producers: P1 capital → country; P2 binding; P3 position.
- Consumers: capital, continent, keyed lookup, equality; U = first letter.

### 4.2 Splits and leakage guards

- **Pairing graph:**
  - 3 producers × 4 trained consumers = 12 pairings: **8 trained (TR)** and **4 held out (HO)**.
  - The held-out pairings are chosen so that every producer and every consumer still appears in at least 2 trained pairings (connected support, as in Schug 2024).
  - U is paired with all 3 producers and never trained.
- **Content split.** Disjoint X values for train / select / confirm (e.g. 50 / 20 / 20), so evaluation uses content values never seen in training.
- **Template split.** Disjoint surface templates and names in each split.
- **Sealed confirmation set.** G_confirm sits behind a runner gate, as in C15.
- **Development uses training data only.**
  - Every hyperparameter is chosen on TR-validation loss alone.
  - HO and U metrics are never computed before FREEZE; the code refuses to compute them.

### 4.3 Conditions

**Trained arms.** Three seeds each; equal trainable parameters (±10%); identical data, steps, optimiser, and read/write layers.

| Arm | What it is | What it rules out |
|---|---|---|
| **A** (RSW: blind, shared reader, capacity c = 2r\*) | The hypothesised workspace. (r\* is measured in Stage 0; see gate V2) | — |
| **C1** Addressed channel | As A, but the writer also receives a learned consumer-identity embedding (an oracle address) | A high-leverage local controller; addressed messaging |
| **C2** Private lines | Capacity split into 4 lines, one per consumer type, each with its own writer and reader heads (all blind). A line is read only when its consumer type is cued (oracle routing) | Task-specific routing; the same total capacity with no shared medium |
| **C3** Unrestricted re-entry (Back-Attention-type) | Consumer read layers attend to every write-window position at width r_v = 64 (about 4–8× A's capacity), with no slots and no competition; parameters matched | That the capacity limit is irrelevant; generic learned re-entry |
| **C4** Self-recurrence | A's machinery, but the writer reads the consumer segment's own earlier late-layer states, so it has no access to producer content | Extra compute, recurrence or parameters without shared access |
| **C5** Parameter-matched LoRA | LoRA on the read layers, with the same number of trainable parameters and the same data | That fine-tuning simply repairs the native routing |
| **C7** Late read | As A, but reading only at the last two layers | A final-layer policy switch |
| A-cap sweep | A at c ∈ {0.5 r\*, r\*, 4 r\*}, 2 seeds each | Tests the capacity window (H4) |
| A-B | A with query-free reads (candidate B) | Whether selective uptake matters |

**Zero-parameter baselines:**

| Baseline | Role |
|---|---|
| C0, native single pass | No workspace at all |
| C6, native transplant | Oracle back-patch: the late producer state is transplanted into the consumer's read layer (the logic of Ramesh & Li and of Biran). Best layer pair chosen on calibration items |
| Text chain of thought | Upper bound with an external scratchpad |
| Text oracle | X given in the text: the consumer's competence ceiling |

**Test-time interventions on trained arms (no retraining):**

| Code | Intervention | Purpose |
|---|---|---|
| I1 | Zero and mean slots | Necessity |
| I2 | **Content swap:** slot state taken from an episode with X′ ≠ X and the same producer type | Content-specific transport (primary) |
| I3 | Norm-matched random slots | Negative control |
| I4 | Writer ablation | Necessity of the write path |
| I5 | Reader ablation per layer and per slot | Selective access |
| I6 | Cue invariance: slots must be bit-identical across cues | Verifies the blind write |
| I7 | Slot interpolation between X and X′ | Dose-response |

### 4.4 Endpoints: integration, not accuracy

**Primary: content transport (CT)**, on held-out pairings and on U.
- **Definition:** CT = P(answer = f(X′) after a content swap, I2) − P(answer = f(X′) with norm-matched random slots, I3).
- **Scope:** only items where the consumer handles X′ correctly when X′ is given in the text (from Stage 0).
- **Interpretation:** a high CT means the downstream computation **follows the workspace content**, for a pairing or consumer the workspace was never trained on.

**Secondary: accuracy gains over C0.** Not primary, because they can come from generic influence (compare the latent-channel causal audit).

**Integration advantage:** Δ_X = CT(A) − CT(X), for every control X.

**Breadth:** the number of consumer types, out of 5 including U, with CT ≥ 0.3.

**Descriptive only (no decisions rest on these):**
- whether the slots encode X or a bundle of trained-consumer answers (the content-code vs answer-bundle diagnosis);
- native readability: a logit-lens readout of W_o · S at the read layers.

### 4.5 Alternative explanations and what discriminates them

| Alternative explanation | Discriminating control |
|---|---|
| Extra parameters | C5 (parameter-matched LoRA); C4 |
| Extra compute or generic recurrence | C4 (same recurrence, no shared access) |
| Additional memory | C3 (more capacity); the A-cap sweep |
| A high-leverage local controller or goal-conditioned switch | C1 (addressed); blind write, verified by I6; the untrained consumer U |
| Task-specific routing | C2 (private lines); held-out pairings |
| Simple copying of X | Consumers that transform X, so f(X′) ≠ X′. The copy consumer is only a positive control |
| A final-layer policy switch | C7 (late read); per-layer reader ablation (I5) |
| Generic perturbation or influence | CT is measured relative to I3; the swap must move answers specifically to f(X′) |
| A pretrained "lingua franca" (no workspace needed) | C6 (native transplant); C3 |
| Leakage from training to evaluation | Disjoint X values, templates and pairings; code that refuses early evaluation; sealed G_confirm |
| Seed or optimisation luck | 3 seeds per arm as the replication unit (lesson R5) |

### 4.6 The smart-local-controller objection

**The objection.** A goal-conditioned switch can look flexible by mapping (goal, cue) to a policy, without content being broadly available.

**How the design answers it:**
1. **Blind write.** The workspace is filled before the goal exists, so it cannot hold a goal-specific answer. I6 verifies this.
2. **Different computations, not different policies.** The same written content must drive qualitatively different computations: arithmetic (C1), feature readout (C2), attention-keyed lookup (C3), magnitude judgment (C4) and lexical generation (U). These are not different answer policies applied to one computation. This is also the design's answer to "does it work under different goals".
3. **Content swaps must redirect each consumer to f(X′),** the consequence specific to the swapped content. A policy switch would change behaviour generically; it would not move answers to the swapped content's answer.
4. **The objection is built in as a competitor.** C1 has oracle addresses and equal capacity. If C1 ≥ A on held-out pairings and on U, the objection wins.

Episodes with several consumers (one write, two different cues) are reported but do not gate anything.

### 4.7 Hypotheses, gates and classification

All numbers are proposals. They are frozen only after Stage 1 (§4.8).

**Validity gates.** Every gate must pass before any hypothesis is read.

| Gate | Criterion | Lesson behind it |
|---|---|---|
| V1 Instrument | Producer competence ≥ 0.80; consumer competence with X in text ≥ 0.80; chain-of-thought composition ≥ 0.70; **native single-pass composition ≤ 0.35** | Validate the instrument first; the native gap must exist |
| V2 Reachability | Text-oracle re-entry raises composition accuracy by ≥ 0.25 in at least 3 of 5 consumers. (The transplant is the residual at the read layers from a consumer run with X in the text.) Also measures **r\***: the smallest PCA rank of that transplant that keeps ≥ 90% of its effect | R8: the lever must be able to reach the region where the gates can pass |
| V3 Neutral negative control | Random slots (I3): f(X′) rate ≤ 0.03; change in log-probability of the correct answer within ±0.10 nats; accuracy change ≤ 0.03 | R4: neutrality at the graded level, not just on accuracy |
| V4 Training reach | A on trained pairings: accuracy − C0 ≥ 0.20 in at least 2 of 3 seeds | Engineering reachability |
| V5 Positive control | Copy-consumer CT ≥ 0.5 | The pipeline can show transport when it exists |

**Hypotheses:**

| ID | Statement | Gate (proposal) |
|---|---|---|
| H1 Transport into held-out pairings | CT_HO(A) ≥ 0.30 | 95% CI lower bound ≥ 0.10 |
| H2 Untrained consumer | CT_U(A) ≥ 0.20, and Acc_U(A) − Acc_U(C0) ≥ 0.10 | CI lower bounds > 0 |
| H3 Consumer-agnostic advantage | Δ_C1 ≥ 0.10 and Δ_C2 ≥ 0.10 on held-out pairings, and Δ_C1 ≥ 0.10 on U. (C2 has no line for U, so U is reported only with all its lines open) | Intersection–union test: every component's CI lower bound must exceed 0. All components must pass, so no multiplicity correction is needed |
| H4 Capacity advantage | CT_U(A) − CT_U(C3) ≥ 0.10. The cap sweep is reported against the pre-declared window prediction: peak inside [r\*, 4 r\*), and lower at 0.5 r\* | CI lower bound > 0 |
| H5 Not generic | Δ_C4, Δ_C5 and Δ_C7 each ≥ 0.15 on held-out pairings | Intersection–union test |
| H6 Necessity | Zero/mean slots (I1) remove ≥ 80% of A's held-out accuracy gain | CI |
| Qualifier | If C6 (zero-parameter transplant) ≈ A on U, then the generality claim becomes "the workspace learns to deliver content in native format", not "the workspace creates generality" | Report |

**Classification** (precedence from the top):

| Class | Condition | Action | Ladder rung |
|---|---|---|---|
| I-FAIL | Any validity gate fails | **STOP.** Report. No repair beyond the single pre-declared fallback (model or domain) at Stage 0 | — |
| W0 | H1 fails with a powered CI (upper bound < 0.30) | **STOP.** Negative report | Informative negative |
| W-GEN | Accuracy rises but H1 fails | **STOP.** Report as generic influence | 2–3 |
| W-RE | H1, H2, H5, H6 pass; H3 fails | Confirm. Report a re-entry principle; no broadcast claim | 4 |
| W-BC | H1, H2, H3, H5, H6 pass; H4 fails | Confirm. Report a consumer-agnostic broadcast principle; no capacity claim | 5 |
| W-GW | All hypotheses pass | Confirm. Then Stage 3 (developmental / phase diagram) only with PI approval | 5, with a Level-6 attempt |

**Confirmation.** One run on the sealed G_confirm (fresh X values and templates) with the frozen recipe and the same gates. A non-replication is reported, not rescued.

### 4.8 Statistics and calibration (Stage 1: synthetic only)

**Unit of analysis.** Items, nested in X values, nested in seeds. Hierarchical bootstrap over seeds × X values × items, 2000 replicates.

**Stage 1a: parametric simulator.**
- Simulates item-level outcomes in five worlds:
  1. null;
  2. generic influence;
  3. re-entry (A ≈ C3);
  4. broadcast (A beats C1, C2 and C3);
  5. "lingua franca" (A ≈ C6).
- Outputs:
  - the false-positive rate of the **complete** gate for each claim class under each null world;
  - power at the planned sample size (about 320 held-out items and 240 U items per seed, 3 seeds; clustering by X value may require more templates).
- Rule, as in D74: if a complete-gate false-positive rate exceeds 0.05, make the minimal pre-declared adjustment using synthetic data only, then freeze.

**Stage 1b: planted system** (about 1–2 CPU-h).
- *Setup:* train a tiny transformer from scratch on the single functions and on text composition. Freeze it. Then give it the A, C1, C3 and C6 arms.
- *Checks:*
  - the whole pipeline end to end;
  - that V1–V5 can pass together (R6);
  - the analytic capacity-window prediction (below).
- *The capacity-window prediction.* A blind writer serving n_c consumers can carry either the content X, which needs about r_X dimensions, or a bundle of answers, which needs about n_c · r_X. So content is *forced* only when capacity c lies in [r_X, n_c · r_X).

**Freezing.** A machine-readable threshold JSON is frozen and hashed before any Stage-2 run, as in C15.

### 4.9 Lessons from the prior program, built in

| Lesson | Where it appears in C16 |
|---|---|
| R1 / R3: identifiability; separate representation from causal usability | CT (following the swapped content) is primary; slot decodability is descriptive only |
| R4: negative controls must be neutral at the graded level | V3 checks log-probability and accuracy, not accuracy alone |
| R5: a replication unit | 3 seeds per arm; conditional replication on Qwen3-1.7B |
| R6: joint satisfiability, checked analytically or synthetically | Stage 1b planted system; the capacity-window algebra |
| R7: minimal intervention engineering | No mask or RoPE surgery (candidate D rejected); zero-initialised gates; bit-identity at initialisation is tested |
| R8: reachability | V2 (text-oracle re-entry) and r\*, measured before any training |
| C15: empirical matching collapses when the material changes | Controls are matched by construction (parameter and capacity budgets), never by empirical KL search |
| C15: sub-assays must measure what they claim (the PC1/PC2 split) | Every validity gate gets a positive and a negative control in Stage 1b |
| C15: fresh material shifted difficulty (two-hop accuracy fell from 0.57 to 0.36) | Competence gates apply per item (only competent items enter CT), and X-value splits are balanced on Stage-0 competence |
| Complementary failures are not a dissociation | No dissociation claims; contrasts use intersection–union tests with powered CIs |
| No threshold repair after outcomes | Frozen threshold JSON; code refuses held-out and U computation before FREEZE |

### 4.10 The developmental angle (Stage 3, conditional)

**What it is.** The closest faithful version of the original seed idea (D7, D15): channels developed separately and later integrated through a common bottleneck.

**Two cheap manipulations that reuse the Stage-2 machinery:**
- **D-a, phase diagram (an architectural principle).**
  - Vary the number of trained consumers n_c ∈ {1, 2, 4} against capacity c ∈ {0.5, 1, 2, 4} × r\*.
  - Prediction: transfer to U appears only when n_c ≥ 2 and c ∈ [r\*, n_c · r\*).
  - With n_c = 1, the workspace should behave like a local controller (an answer code).
- **D-b, developmental path dependence.** Compare three schedules:
  1. shared from the start;
  2. private lines first, then merged into a shared medium;
  3. capacity annealed from large to small.
  - Prediction: schedule 2 keeps consumer-specific codes and transfers to U worse than schedule 1.

**On the word "emergence": I recommend not using it.** Use **integration gain (IG)** instead.

If you want the term anyway, pre-register this operational definition of **superadditive integration**. All five conditions must hold:
1. measured only on held-out-pairing and U items with zero training exposure;
2. IG = CT(integrated) − the maximum over component-only and matched local-communication controls (C0, C1, C2, C4, C5, C6), with IG > 0 and its CI lower bound > 0;
3. the effect agrees on a continuous metric (log-odds of f(X′)) and on the thresholded metric, which guards against metric artifacts (Schaeffer 2023);
4. replicated on 3 seeds and at 2 capacities;
5. never applied beyond the Level-1 functional claim.

**Cost and value.** About 12 CPU-h. This is the only route to Level 6, and it is worth running only after W-BC or W-GW.

### 4.11 Cost (decision item 11)

Assumptions: forward throughput of 150–240 tok/s for the 0.5B model in fp32; 0.3–0.45 s per training example (about a 40-token episode, with backward passes over about 15 consumer tokens through about 20 layers). Stage 0 measures the real rate first.

| Stage | Content | CPU-h | RAM |
|---|---|---|---|
| 0. Instrument and reachability (forward only) | Competence, native gap, chain of thought, oracle re-entry, r\*, choice of L_w and read layers, timing | ≈ 2 | < 4 GB |
| 1. Synthetic | Simulator and planted tiny transformer | ≈ 2–3 | < 2 GB |
| 2. SELECT | 7 arms × 3 seeds × ≈ 1.7 h; capacity sweep (6 runs); A-B ablation; evaluation | ≈ 45–50 | < 6 GB |
| Confirmation | Evaluation only | ≈ 3 | < 6 GB |
| **Core total** | | **≈ 55** | |
| 3. Developmental (conditional) | D-a and D-b | ≈ 12 | |
| Replication on Qwen3-1.7B (conditional) | A, C1, C3, C5 × 2 seeds | ≈ 40–50 | < 16 GB |

No downloads. Long runs are launched through WMI and committed before they start, as in C15.

---

## 5. Decision items

1. **Strongest exact novelty claim that survives the audit.**
   - *The claim:* in an already pretrained, frozen decoder LM, a retrofitted, capacity-limited, re-entrant workspace causally transports content: content swaps redirect behaviour to the swapped content's consequence. This holds for producer–consumer pairings, and for a consumer operation, never used to train it.
   - *The workspace's defining properties:* it is written before the downstream use is specified, and read through one consumer-agnostic interface at several depths.
   - *What it must beat:* equal-capacity consumer-addressed and private-line channels; capacity-unrestricted learned re-entry; self-recurrence; late-layer reading; parameter-matched fine-tuning. None of these may reproduce the effect.
   - *Not claimable:*
     - that re-entry helps a frozen LM compose (Back Attention);
     - that shared workspaces beat pairwise communication (Goyal);
     - zero-shot transfer through a shared workspace over frozen specialists in general (Maytié).
2. **Closest competitors, and why none pre-empts the exact cell.**

   | Competitor | Why it does not pre-empt |
   |---|---|
   | Back Attention 2025 | Same insertion logic, but no capacity, broadcast, held-out or matched tests. It becomes control C3 |
   | Maytié 2024 / Devillers 2024 / Bertin-Johannet 2026 | Held-out transfer through a GW over frozen encoders, but not inside an LM, and cross-modal rather than cross-consumer |
   | Goyal 2022 | The principle, but from scratch and without held-out pairings |
   | Latent communication and the 2026 causal audit | Between instances; I adopt its content-vs-generic methodology |
   | T²MLR / retrofitted recurrence | Recurrence, not a workspace |
   | Gurnee 2026 | Native and descriptive |
   | CALM | Point-to-point composition |

3. **Novelty confidence.**

   | Aspect | Confidence |
   |---|---|
   | Exact cell | Moderate, ≈ 0.6 |
   | Mechanism | Low, ≈ 0.2 |
   | Durability | Low |

4. **Significance confidence: low to moderate.** Expected rung 3–4. P(Level 5) ≈ 0.2; P(Level 6) ≤ 0.05.
5. **Candidate architectures, ranked:** A (RSW) > E (discrete bottleneck) > B (low-rank state) > D (parallel streams) > C (competition hub). See §3.3.
6. **Recommended architecture:** A, the Re-entrant Slot Workspace, with B as a built-in ablation.
7. **Minimal decisive experiment.**
   - *Setup:* Qwen2.5-0.5B-Instruct; numbers domain; 3 producers × 4 trained consumers (8 trained and 4 held-out pairings), plus the untrained consumer U.
   - *Arms:* A against C1, C2, C3, C4, C5 and C7, with 3 seeds each. Zero-parameter baselines: C0, C6 and chain of thought.
   - *Primary endpoint:* content transport (following a content swap) on held-out pairings and on U.
   - *Outcome classes:* W0, W-GEN, W-RE, W-BC, W-GW.
   - *If the budget must be cut:* A, C1, C3 and C5 × 3 seeds (≈ 25 CPU-h). This drops part of H5, so it could support at most W-BC.
8. **Strongest matched controls.**

   | Control | Answers |
   |---|---|
   | C1, addressed channel | The local-controller objection |
   | C3, unrestricted re-entry (Back-Attention-type) | Whether limited capacity matters |
   | C5, parameter-matched LoRA | Whether fine-tuning alone suffices |
   | C6, zero-parameter native transplant | Whether the pretrained model's native format already gives generality |
   | Content swap I2 vs random slots I3 | Content-specific transport vs generic influence |

9. **Held-out generalisation test.** Held-out pairings with connected support, plus an untrained consumer that uses different machinery (verbalisation). Both run on disjoint X values and templates. G_confirm stays sealed.
10. **Falsification and STOP conditions.**
    - **STOP when:**
      - any validity gate fails, after the single pre-declared model or domain fallback at Stage 0;
      - a Stage-1 complete-gate false-positive rate exceeds 0.05 and the pre-declared synthetic rule cannot fix it, or power is below 0.8 at the planned sample size;
      - training reach (V4) fails after the pre-declared 3-value learning-rate grid, chosen on trained pairings only;
      - the outcome is W0 or W-GEN;
      - leakage is detected;
      - confirmation does not replicate.
    - **Which part of H_workspace each outcome falsifies:**

      | Outcome | Part falsified |
      |---|---|
      | C3 ≥ A | The limited-capacity part |
      | C1 or C2 ≥ A | The broadcast part |
      | C4, C5 or C7 ≥ A | The claim that the workspace specifically matters |

11. **Cost.** Core ≈ 55 CPU-h, RAM < 6 GB, no downloads. Optional Stage 3 ≈ 12 h; optional Qwen3-1.7B replication ≈ 45 h.
12. **What a positive result would actually establish.**
    - **Level 1.** A small, blind workspace makes late-computed content causally available to several qualitatively different early- and mid-layer computations of a frozen LM. This includes untrained pairings and an untrained consumer. Equal-capacity addressed and private channels, and unrestricted re-entry, do not achieve this.
    - **Level 2.** Consistent with GWT's functional claim: limited-capacity *broadcast of content*, rather than addressed messaging, supports flexible novel combination. It turns indicators GWT-2 and GWT-3 into testable functions in an LM.
    - **What it would not show:**
      - that LLMs natively lack a workspace;
      - that the model has conscious access;
      - anything about phenomenal experience (Level 3 is unsupported and not claimed).
13. **What a negative result would teach.**

    | Outcome | Lesson |
    |---|---|
    | I-FAIL at V2 | Frozen early layers cannot use re-entered late content, which bounds any retrofitted workspace (methods value) |
    | W-GEN | Latent channels in frozen LMs exert generic rather than content-specific influence. This extends the 2026 causal audit to channels inside one model |
    | W-RE | Integration comes from re-entry plus the pretrained "lingua franca", not from limited-capacity broadcast. Evidence against capacity-limited broadcast being *necessary* for flexibility in pretrained LMs |
    | C1 ≥ A | Addressed messaging generalises as well, which weakens the GWT-specific broadcast claim |
    | C5 ≥ A | Fine-tuning the native routes suffices; no workspace is needed |

14. **Should C16 replace C15 as the main research direction? Not yet, and only conditionally.**
    - **For it:**
      - It is the best-identified option available at $0. Because we build the mechanism, the instrument-validity failures that stopped B1, D2, the C15 A-stage and R2 can be checked at Stage 0/1.
      - Its exact cell appears open.
    - **Against it:** its realistic significance (rung 3–4, P(Level 5) ≈ 0.2) falls below your stated bar of "approximately Level 5, ideally approaching 6".
    - **Recommendation:**
      - Make C16 the main line **only if Stage 0 and Stage 1 both pass.** Together they cost about 4–5 CPU-h, need no downloads, and require approval only to run the cached 0.5B model forward.
      - If either fails, the honest fallback is the C15 methods / negative-results note, and a pause on new empirical lines rather than another variant.

---

## 6. Proposed next step (not executed; needs your approval)

**The C16-S0/S1 preregistration package (design → frozen pre-run plan):**

0. **Targeted full-text and citation audit.**
   - Read Back Attention, Maytié 2024, Goyal 2022, T²MLR and the 2026 latent-channel causal audit in full.
   - Crawl their forward citations.
   - **Kill or narrow C16** if any of them tests held-out pairings or untrained consumers with matched non-broadcast controls inside a pretrained LM.
1. **Function and item pool.**
   - Selection rule: single-function competence on a calibration split only.
   - Pre-declared domain order: numbers, then countries.
   - Pre-declared model order: Qwen2.5-0.5B, then Qwen3-1.7B.
2. **Stage-0 protocol and gates V1–V2.** Competence, native gap, chain of thought, text-oracle reachability, r\*, choice of L_w and read layers, timing. **Running it requires your approval to execute the cached Qwen2.5-0.5B-Instruct forward-only. No downloads.**
3. **Stage-1 synthetic plan.** Simulator worlds, the planted tiny transformer, and the rule for the complete-gate false-positive rate.
4. **The full Stage-2 specification, frozen and committed before any Stage-2 run.** Arms, parameter and capacity budgets, the threshold JSON, the classification table, the split builder and the sealed G_confirm gate.

Nothing in §6 has been started.

---

## Appendix: key sources (verification status in `literature/literature_db.json`)

**Re-entry and recurrence retrofits**
- Back Attention, Yu, Belinkov & Ananiadou, EMNLP 2025: <https://arxiv.org/abs/2502.10835>
- Hopping too late, Biran et al., EMNLP 2024: <https://arxiv.org/abs/2406.12775>
- T²MLR, Cai et al. 2026: <https://arxiv.org/abs/2607.15178>
- Retrofitting recurrent depth, Shapiro 2026: <https://arxiv.org/abs/2608.11233>
- Retrofitted recurrence, McLeish et al. 2025: <https://arxiv.org/abs/2511.07384>

**Global-workspace and modular architectures**
- Shared global workspace, Goyal et al., ICLR 2022: <https://arxiv.org/abs/2103.01197>
- DVNC, Liu et al. 2021: <https://arxiv.org/abs/2107.02367>
- Is a modular architecture enough?, Mittal et al. 2022: <https://arxiv.org/abs/2206.02713>
- Modular solutions that generalise compositionally, Schug et al. 2024: <https://arxiv.org/abs/2312.15001>
- Devillers et al. 2024: <https://arxiv.org/abs/2306.15711>
- Maytié et al. 2024: <https://arxiv.org/abs/2403.04588>
- Bertin-Johannet et al. 2026: <https://arxiv.org/abs/2602.08597>
- Chateau-Laurent & VanRullen 2025: <https://arxiv.org/abs/2503.01906>
- VanRullen & Kanai 2021: <https://arxiv.org/abs/2012.10390>
- The Perceiver as a functional global workspace (CogSci paper): <https://escholarship.org/uc/item/2g55b9xx>
- Phua 2025: <https://arxiv.org/abs/2512.19155>

**Native workspace in LLMs**
- Gurnee et al. 2026 (J-space): <https://arxiv.org/abs/2607.15495>
- "Small models have a global workspace" (LessWrong): <https://www.lesswrong.com/posts/L4o7efBwoiFLxBGRm/small-models-have-a-global-workspace>

**Agent-level and blackboard workspaces**
- Theater of Mind / GWA, Shang 2026: <https://arxiv.org/abs/2604.08206>
- CTM-AI, Yu et al. 2026: <https://arxiv.org/abs/2605.04097>
- UMM 2025: <https://arxiv.org/abs/2503.03459>
- LbMAS 2025: <https://arxiv.org/abs/2507.01701>
- Blackboard multi-agent system, Salemi et al. 2025: <https://arxiv.org/abs/2510.01285>
- SRMT 2025: <https://arxiv.org/abs/2501.13200>
- SAF 2022: <https://arxiv.org/abs/2210.03022>
- Goldstein & Kirk-Giannini 2024: <https://arxiv.org/abs/2410.11407>

**Latent communication between models and threads**
- Communicating activations, Ramesh & Li 2025: <https://arxiv.org/abs/2501.14082>
- LatentMAS 2026: <https://arxiv.org/abs/2511.20639>
- C2C 2025: <https://arxiv.org/abs/2510.03215>
- Interlat 2026: <https://arxiv.org/abs/2511.09149>
- Latent-channel causal audit 2026: <https://arxiv.org/abs/2607.26773>
- Latent-communication survey 2026: <https://arxiv.org/abs/2606.05711>
- Group Think 2025: <https://arxiv.org/abs/2505.11107>
- Hogwild! 2025: <https://arxiv.org/abs/2504.06261>
- Latent Agents 2026: <https://aclanthology.org/2026.acl-long.709/>
- CALM 2024: <https://arxiv.org/abs/2401.02412>

**Memory and feedback architectures**
- TransformerFAM 2024: <https://arxiv.org/abs/2404.09173>
- CFL 2024: <https://arxiv.org/abs/2412.17737>
- Feedback Transformer 2020: <https://arxiv.org/abs/2002.09402>
- Coconut 2024: <https://arxiv.org/abs/2412.06769>
- RMT 2022: <https://arxiv.org/abs/2207.06881>
- ARMT 2024: <https://arxiv.org/abs/2407.04841>
- LM2 2025: <https://arxiv.org/abs/2502.06049>
- MemoryLLM 2024: <https://arxiv.org/abs/2402.04624>
- Prometheus Mind 2026: <https://arxiv.org/abs/2601.15324>
- Bottlenecked Transformers 2025: <https://arxiv.org/abs/2505.16950>

**Metacognitive signals**
- Kumaran et al. 2026: <https://arxiv.org/abs/2603.22161>
- Kawada & Kellis 2026: <https://arxiv.org/abs/2609.04290>
- Looped proto-introspection 2026: <https://arxiv.org/abs/2607.18553>

---

## 7. Amendments after PI review (2026-10-08, D79)

The PI approved the audit and Stage 0/1 only. The following supersedes the earlier sections where they differ.

1. **Full-text and citation audit** (`literature/c16_audit/audit_report.md`).
   - **Exact cell survives.** Novelty for the exact cell is revised from ≈ 0.6 to **≈ 0.5**.
   - **New closest precedent:** Marincat 2026a–c. These are LM "societies" on the same frozen Qwen2.5-0.5B with a trained LoRA. Cells communicate through 2 × 896-d packets in a relay chain. Packet swaps show that restricted visibility yields value-indexed codes, and zero-shot transfer to a new operator family fails.
   - **Why the cell survives:** Marincat uses separate calls, LoRA-trained consumers and a chain topology, with no capacity, broadcast or addressed contrasts. C16 uses a native-skill U, which avoids Marincat's failure mode (the consumer must execute a new operator).
2. **C1 is now fair** (amendment A1).
   - The writer conditioning is cue-derived: e(cue) is the frozen LM's mean residual over the cue string. A has the identical module with a constant ē, so the parameter counts are identical.
   - C2's read routing is also cue-derived.
   - New gate **V6:** C1 must reach ≥ 0.80 of its trained-cue CT on paraphrased cues.
   - A's U advantage is therefore not true by construction.
3. **H-CD reformulated** (amendment A2).
   - The *peaked* capacity window is not defensible in bits: an answer bundle never needs more information than the content. Its upper limb was also looked for and not found in related settings (Resnick et al. 2020; 2607.00233).
   - Retained: a preregistered **three-account discrimination**: format-geometry vs information bottleneck vs native copy.
   - Its coordinates r_X and r_B(S) are measured before any training (Stage 0, W1–W4), and its power is set by S1a.
   - Whether H-CD becomes the central Stage-2 hypothesis is decided at the PI review by the pre-declared rule in the S0/S1 preregistration.
4. **Conditional naturalistic extension** (amendment A3). Verification/error content feeds revision and keyed retrieval, with abstention as the untrained consumer. It runs only after a confirmed W-BC/W-GW, under a separate preregistration, at no Stage-0/1 cost.
5. **Design adjustments** made while freezing (S0/S1 preregistration §3):
   - **Producers:** all three now keep X off the surface (add / sub / mul). The memo's binding and positional producers had X on the surface.
   - **Untrained consumer:** U stays verbalization.
   - **Consumer pool:** gains plus10, so that bundle ranks can grow with consumer diversity.
