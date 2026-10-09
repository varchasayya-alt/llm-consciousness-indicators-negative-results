# Rival-hypothesis table (Stage 1, v3 within-target design). Written before any confirmatory data exist.

v3 (decisions D36–D41) replaces the v1/v2 primary contrast (X_lost vs a separately constructed displacement-matched sham Y). Calibration showed that the X-vs-Y contrast confounds competence outcome with a **procedure fingerprint**: the direction of internal change at the answer sites, which items targeted but not forgotten (X_retained) share. The table below shows the v1/v2 rows as **superseded** where that applies.

## The three explanations

| Code | Explanation | What the monitor's estimate is a function of |
|---|---|---|
| **H1 — DIFFICULTY** | Input-difficulty / item-property estimation | Properties of the query (item identity, entity familiarity / exposure, surface features), however and wherever these are encoded. It is insensitive to changes in the store that leave those properties intact |
| **H2 — ANOMALY / PROCEDURE** | Intervention / anomaly / procedure detection | Whether the item was targeted, and how much or how abnormally its representation changed relative to the intact distribution, regardless of whether the answer survives |
| **H3 — COMPETENCE** | Competence monitoring | Whether, and how strongly, the monitored system *currently* has task-relevant competence for this item |

The three explanations are not mutually exclusive. A real monitor may be a mixture, so the *pattern* across controls estimates their relative contributions.

**Notation.**

- C = answer margin. M = monitor score.
- θ_pre = within-target residualized-post association between M_post and C_post, adjusting for pre-intervention information only.
- θ_gen = the same, additionally adjusting for the 40 pre-declared generic post-change features.
- All store interventions leave the input tokens unchanged.

## Predictions

| Control (endpoint) | H1 predicts | H2 predicts | H3 predicts | What discriminates |
|---|---|---|---|---|
| **Within-target tracking** (identically treated X items; θ_pre; **P1**) | θ_pre ≈ 0 (inputs unchanged; static item properties absorbed by pre-level adjustment) | θ_pre > 0 **only through item-varying intervention intensity** (items that moved more lose more) | θ_pre > 0 | Rules out H1. Separates H3 from H2 only together with θ_gen |
| **Generic-adjusted within-target tracking** (θ_gen; **G1**; label A vs B) | ≈ 0 | **≈ 0** (intensity and change-structure absorbed by the measured generic features) | **> 0** (competence variation beyond the measured change structure) | **A** (θ_pre and θ_gen supported) ⇒ H3 component beyond measured generic change. **B** ⇒ competence tracking not separable from generic change detection |
| **Pre-existing susceptibility** (symmetric pre-state adjustment inside θ_pre; reported with/without the pre-state step and with the one-sided P_i score) | — | A monitor that responds more on fragile items: positive θ *without* pre-state adjustment, ≈ 0 with it | Unchanged by adjustment (tracks the post-intervention state) | Material change between θ_noState and θ_pre ⇒ susceptibility contribution (reported transparently) |
| **Forgotten vs retained within the same procedure** (K1: propensity-matched on pre-intervention variables only; baseline-adjusted monitor outcome) | ≈ 0.5 | ≈ 0.5 (same procedure) | > 0.5 | Binary support for H3, with residual selection on unmeasured susceptibility |
| **New intervention class** (T-INTERF; θ_pre within all equally exposed items; **P2**; G2) | ≈ 0 | > 0 only via intensity; θ_gen ≈ 0 | θ_pre > 0 and θ_gen > 0 | Generalises H3 to naturally arising (often confident) errors not forced to uniform |
| **Y negative control** (K2: TOST of ΔM(Y) − ΔM(Z) within ±0.10 D_nat) | ≈ 0 | **ΔM(Y) < ΔM(Z)** (Y moved by a matched amount) | ≈ 0 (Y keeps competence) | Non-equivalence ⇒ magnitude-sensitive (H2) component. *Limitation:* Y differs from X in direction of change, so Y cannot control direction-specific procedure detection; the within-target design does |
| **Newly acquired knowledge** (T-NEW, exposure-matched; F3) | ≈ 0 (familiarity exposure matched) | **Δs ≤ 0** (representations changed) | **Δs > 0** (competence gained) | **Opposite signs for H2 vs H3** |
| **Familiarity-only manipulation** (T-FAM; F2) | **Δs > 0** | **Δs < 0** | **Δs ≈ 0** | **Three different signs** |
| **Input corruption** (name → unused name, store intact) | Drop | Drop | Drop | A double dissociation across monitors: IN detects corruption but is constant under store interventions; INT detects both |
| **Developmental history** (K3: z(θ_pre) INT-TP − INT-S) | — | — | — | Whether exposure to the store's own competence variation improves within-target tracking |
| **Cross-store aligned monitor** (A→B̃ vs B̃→B̃ on θ_pre; S1/S2) | Transfers | Transfers iff change geometry is generic | Transfers iff competence representation is generic | Generic decoding vs system-specific coupling |
| **REPLACE** (exploratory; confident new answer) | ≈ 0 | Drop | ≈ 0 for a retrieval-state monitor | A drop marks an H2 (fingerprint) component |
| *Superseded (v1/v2):* X_lost vs displacement-matched Y as primary | X_lost ≈ Y ≈ 0 | X_lost ≈ Y | X_lost ≪ Y | **Confounded by the procedure fingerprint** (D34). Now descriptive only, labelled "fingerprint-confounded" |
| *Superseded:* X_lost vs Z (old K1) | Both ≈ 0 | X_lost < Z | X_lost < Z ≈ 0 | Confounded by targeting. Descriptive only |

## Expected joint patterns (pre-registered reading)

| Joint pattern | Reading |
|---|---|
| P1 label **A**, P2 label **A**, K1 ✓, F3 > 0.5, F2 ≈ 0, K2 equivalent | Predominantly **H3**, generalising across two mechanisms |
| P1 label **B** (θ_pre ✓, θ_gen ✗), K2 non-equivalent or REPLACE drop | Competence tracking **not separable from generic change detection or residual susceptibility** (H2 component). **Not counted as support for H3** (D54) |
| P1 label **C** with θ_noState > 0 | Apparent tracking explained by pre-existing susceptibility or baseline coupling |
| All store-intervention ΔM ≈ 0, F2 > 0, corruption detected | Predominantly **H1** |
| Mixed patterns | Reported as mixtures. No single-label claim is made |
