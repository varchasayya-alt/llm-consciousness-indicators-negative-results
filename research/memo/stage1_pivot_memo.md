# Stage-1 PIVOT MEMO (after closure of the B1 line)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | For PI decision. **No new implementation.** Nothing below has been run |
| Basis | `experiments/stage1/B1_final_report.md`; the stop reports for v2, v3, v4 and v4.1; `memo/decision_document_v2.md` |
| Constraints carried over | $0 CPU laptop; Level-1/2/3 discipline; D54 (only label A is affirmative); pre-declared binding gates; monitor-input restrictions; validation 9051–9053 and confirmatory 1001–1040 untouched |
| Explicitly not proposed | B1 variants (v4.3), new p_rd grids, larger or smaller stores, larger worlds, new memory architectures, relaxed thresholds |

The decision document anticipated this branch: *"If Stage 1 shows degenerate behaviour … publish N1 + negative toy result as a short paper, and move effort to P1\* at 1.5B."* The options below refine that rule in light of what five calibration cycles taught.

## 0. What any next Stage-1 design must satisfy

These requirements come from N1 and the calibration history.

| # | Requirement | Calibration evidence behind it |
|---|---|---|
| R1 | The competence change must be **input-invisible**: identical tokens before and after | Core N1 contrast |
| R2 | **Graded item-level variation** in post-change competence among *identically treated* items | v3 tail-flattening (compressed endpoints); v4.2 relative tiers unattainable |
| R3 | **Identifiable:** not determined by pre-state or generic change; non-identifiable if R²_pre, R²_gen or R²_joint is too high | F3–F5; D50 (diffuse susceptibility → false B, so only A counts) |
| R4 | A negative control whose neutrality holds **at the graded level** the gates use | v2 sham fingerprint; v4.2 Y graded change |
| R5 | A replication unit (seed or model) for seed-level inference | Statistical framework |
| R6 | Gates that are mutually satisfiable **by construction**, checked analytically before any run | v4.2 redundancy × tier incompatibility |
| R7 | **Minimal intervention engineering.** Every engineered intervention so far produced a new artifact | v2, v3, v4, v4.1, v4.2 |

## Direction 1: N1-first methods paper (theory, simulation and calibration case studies). No new empirical system

**Question.** What can and cannot identify a monitor that tracks a system's *own* competence, as opposed to world or difficulty tracking or generic change detection? And what must an intervention satisfy for such a test to be valid?

**Content (all existing material):**
1. A formal identification statement:
   - **observational equivalence:** self-tracking and world-tracking monitors produce the same data whenever a familiarity statistic is sufficient on natural data;
   - **interventional identifiability:** they come apart under input-fixed interventions that change correctness;
   - the minimal control set needed.
2. The within-target estimand θ_pre / θ_gen, its assumptions, and the cross-fitted DML implementation.
3. **Simulation validation:**
   - H3, H2, susceptibility, regression-to-the-mean, null, B1-like and high-dimensional worlds;
   - the finite-sample limit (D50) and the resulting label-A-only policy (D54).
4. **Design constraints from five calibration cycles (R1–R7)**, each with its quantitative evidence:
   - procedure fingerprint (v2);
   - tail flattening (v3);
   - uncontrollable redundancy (v4);
   - co-adaptation failure (v4.1);
   - relative-scale gate incompatibility and graded-non-neutral controls (v4.2).
5. A reusable preregistration template and gate checklist.

| Aspect | Assessment |
|---|---|
| Strength | Uses only existing, already validated work. Immediate and zero-risk. Makes the field's implicit assumptions explicit and provides an estimand plus negative-result evidence that design matters |
| Weakness | No empirical claim about monitoring. Novelty rests on the identification statement and the validated estimand: criteria lists already exist [song2025privileged; singh2026reality; lindsey2025introspection; comsa2025introspection; kammerer2023introspective] |
| Cost | $0; writing only (about 1–2 weeks of drafting); no new runs |
| Kill criterion | A full-text reading of the closest papers shows the identification result already stated. The paper then becomes a short methods / negative-results note |

## Direction 2: self-generated competence change in a from-scratch parametric store (interference and acquisition, no engineered edit)

**Question.** Does a separately developed monitor track item-level competence changes that arise from the system's **own continued learning**, beyond pre-change information and generic change?

**Design (requires a new protocol and PI approval).**
- **System:** the plain v1–v3 parametric store and synthetic world. No explicit memory and no B1 component.
- **Competence change:**
  - later-learning interference (new facts, no replay) applied identically to all base-correct items. This is the P2 operator the PI already approved as a secondary, never calibrated because P1 failed first.
  - Ordinary acquisition (newly learned items) as a secondary, sign-reversed test.
- Inputs are token-identical (R1). The change is the system's own development (R7: no targeted edit, no locality requirement on a target set, no transplant control).
- **Controls:**
  - a twin continuation on already-known material with matched steps (natural generic-drift control);
  - θ_gen with the measured generic features;
  - the familiarity-only exposure arm.
- **Unit:** the store seed, as before (R5).

| Aspect | Assessment |
|---|---|
| Strength | Full ground truth (complete training history). Cheapest of all options (minutes per store). Nearly all code exists. Directly on the P5\*/N1 developmental theme. Avoids every engineered-intervention artifact seen so far |
| Main risk (D54) | Interference susceptibility may be largely predictable from pre-states (R²_pre high → non-identifiable), or driven by global change (θ_gen not supported → label B, which is **not** affirmative). That would make the direction uninformative |
| Second risk | Natural interference may give too few or too many losses at usable rates. The approved P2 window, lost ∈ [0.10, 0.50], applies |
| Cost | $0; CPU minutes per store |
| First decisive kill test (store-only, no monitor, about 1 day) | On fresh development seeds: the interference lr grid with lost ∈ [0.10, 0.50], IQR ≥ 1 nat, and F3–F5 identifiability plus the trivial-decoder audit. **Abandon** if no setting is identifiable or diversity fails. R6 is checked analytically first: there is no relative-to-intact tier gate here, because intact margins come from the same single route |

## Direction 3: input-invisible lesions in small pretrained LMs (P1\*, promoted from bridge to primary)

**Question.** In a pretrained LM with *natural* parametric knowledge and natural redundancy, does a separately trained monitor (or a control endpoint such as lookup or abstention) track lesion-induced, item-level competence loss beyond pre-lesion information? And does it do so differently from input-corruption errors at a matched error rate (the central N1 contrast)?

**Design (from `decision_document_v2.md` P1\*, re-specified with the within-target estimand).**
- **Models:** Qwen2.5-0.5B, replicated on 1.5B.
- **Items:** about 3,000 PopQA/Wikidata items, giving about 500 base-correct items.
- **Lesion:** an identical procedure for every target, e.g. attention knock-out of attribute extraction.
- **Contrasts:**
  - sham lesion;
  - unrelated-subject lesion;
  - matched-rate input corruption;
  - global noise;
  - naturally unknown items.
- **Endpoints:** a frozen pre-lesion correctness probe and control behaviour.

| Aspect | Assessment |
|---|---|
| Strength | Real knowledge, so backup and redundancy are natural, not engineered. External validity. The N1 self-change vs input-change contrast is testable directly. It was already the pre-specified bridge |
| Risks | (i) The lesion is again an engineered intervention, so v2/v3-type selectivity, collateral and fingerprint failures may recur under the same binding gates. (ii) Floor effects at 0.5B. (iii) No ground truth on what the monitor reads. (iv) No store seeds: the unit must be re-specified (models × lesion sites, or entity-clustered inference), so replication is weaker. (v) Novelty is incremental [gu2026unlearners; hasegawa2025underconf; ferrando2025entity; lindsey2025biology] |
| Cost | $0 CPU, about 5–30 h per model. **Requires model downloads (needs explicit PI permission; about 1–3 GB each)** |
| Pre-specified kill tests | Abandon if natural known vs unknown items are not discriminated (AUROC < 0.6) by base lookup/abstention in both models (decision document), **or** if no lesion strength gives graded, selective item loss passing the binary and continuous collateral gates at a matched input-corruption error rate |

## Comparison

| Criterion | D1 N1 methods paper | D2 self-generated change (synthetic) | D3 pretrained-LM lesions (P1\*) |
|---|---|---|---|
| Directness to the Stage-1 question | Indirect (framework) | **Direct** | Direct (bridge) |
| Ground truth | n/a (simulation) | **Full** | Weak |
| Avoids the B1 / v2–v3 failure classes | n/a | **Yes** (no engineered edit, no transplant, no relative tiers) | Partly (the engineered lesion remains) |
| Main risk | Novelty | Non-identifiable, or label B only (D54) | Lesion artifacts, floor effects, weak unit |
| $0 and time to first decisive result | Immediate | **~1 day (store-only kill test)** | ~1 week incl. downloads |
| Reuse of existing machinery | Total | Very high | Estimand and analysis only |
| What a positive result licenses (Level 1) | A validated measurement framework | Monitoring of self-generated competence change in a synthetic system (label A only) | The same for small pretrained LMs, plus the self-change vs input-change dissociation |
| What a negative result means | — | Natural interference is not identifiable in this system: an informative boundary | Small LMs show anosognosia, or lesions are not valid at this scale |

## Recommendation

1. **Start D1 now.** It captures the value already produced and needs no compute. It is the decision document's pre-specified fallback.
2. **Run D2's store-only kill test as the single next empirical step**, after PI approval of a short D2 protocol committed before any run:
   - fresh seeds;
   - no monitor;
   - existing gates only;
   - R6 joint-satisfiability checked analytically first.
3. **If D2 fails its kill test, move empirical effort to D3**, as the decision document pre-specifies. A pass at D2 keeps D3 as the external-validity bridge.

Under this ordering, B1 is not resumed in any form. Every empirical step is preceded by an analytical check that its gates can be satisfied together (lesson R6), and no positive claim is possible except label A under the unchanged rules.
