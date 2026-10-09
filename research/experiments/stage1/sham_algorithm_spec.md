# Frozen sham-matching algorithm (T-FORGET + sham). Specification answers the PI's nine points

> **v3 status (D36, 2026-10-02).** The construction below is unchanged, but its **role** has changed.
>
> - **Old role:** the displacement-matched sham Y was the sole primary identification control.
> - **What calibration showed (D34):** X and Y differ in the *direction* of internal change at [A] layers 3–4, which is a procedure fingerprint shared by X_retained.
> - **New role:** Y is a **negative control** (K2: does the monitor respond merely because a state moved by a matched amount, without competence loss?). The primary identification is now *within* the identically targeted set X (`protocol_v3_PROPOSED.md`).
> - **Per-seed QC for Y in v3:**
>   - binary lost ≤ 3%;
>   - ≥ 8/10 site displacement ratios in [0.8, 1.25];
>   - continuous retention (|Δmargin|, |Δlog p| and KL each ≤ 0.10 × X's; |Δmargin| ≤ 0.10 × D_C).
> - **Superseded:**
>   - the matched-pair minimum (point 9: ≥ 100 pairs);
>   - the 0.90 fingerprint *gate*. The v2 result stands on record (C4val FAILED). The fingerprint is now quantified and reported per seed, and Y-based contrasts are descriptive and labelled "fingerprint-confounded".
> - The fallback ladder (point 5) and the seed-replacement rule (point 8) still apply to the full v3 per-seed T-FORGET QC.

The algorithm uses **no monitor output at any step**. This is enforced by a unit test: `s1/interventions.py` and `s1/matching.py` cannot import monitor, controller or pipeline code.

## Construction ("forget-then-relearn with read-site displacement matching")

One joint fine-tuning run of the intact store on the two target sets X and Y (300 items each, familiarity-stratified, random from base-correct EV).

**Phase A** (steps 1 … K/2):

- X and Y **both** receive the forgetting objective, CE(uniform over V_r, p(·|q)).
- This gives identical optimisation exposure and parameter-update opportunity.

**Phase B** (steps K/2+1 … K):

- X keeps the forgetting objective.
- Y receives CE toward its **correct** answer (competence restored) plus γ·L_match, where:

  L_match = Σ_s (D̄_Y,s − sg[D̄_X,s])² / Σ_s sg[D̄_X,s]²

- D̄_·,s is the mean over the set's items of the per-item displacement at read site s (stop-gradient on X).

**Throughout (v2, decisions D27):** retain loss λ_R·L_R, one Adam optimiser, a single run. Both sets are therefore exposed to the same global weight change and the same global fingerprint.

- L_R = KL_ans + KL_tok against a **frozen copy of the original intact store**.
  - **KL_ans:** KL of the full-vocabulary answer distribution at [A], on a random batch of known MT-split facts.
  - **KL_tok:** next-token KL on mention sequences of all entities except X's and Y's.
- EV and CT facts and X/Y-entity mentions never enter the retain pool, so all retained-fact and fluency QC is held out.
- *v1 (superseded after the C4 failure, D26):* β·LM on random training sequences. This gave 50–63% collateral loss.

## The nine pre-registration points

| # | Item | Frozen specification |
|---|---|---|
| 1 | **Displacement metric** | d[i,s] = RMS over the 128 dimensions of (h_post[i,s] − h_pre[i,s]) / σ_s, where σ_s is the per-dimension SD of intact read states on the MT split (store-level statistic). Profile p[i] = log(d[i,·] + 10⁻³) |
| 2 | **Sites entering the metric** | Exactly the monitor read set R: residual stream at positions s2 and [A], layers 0–4 (10 sites). Frozen on architectural grounds before calibration |
| 3 | **Matching tolerance** | *Set level:* D̄_X,s/D̄_Y,s ∈ [0.8, 1.25] at ≥ 8 of 10 sites. *Item level:* caliper on RMS-over-sites distance between log-profiles, value fixed by calibration rule C5 |
| 4 | **Matching algorithm** | *Set level:* the L_match term above (fixed γ, lr, steps from calibration rule C4). *Item level:* greedy nearest-neighbour caliper matching without replacement. X_lost items are visited in a seeded random order (seed = derive_seed(store seed, "pairF")), and each is paired with the nearest unused post-correct Y item within the caliper |
| 5 | **Maximum optimisation attempts** | The base configuration, then the pre-registered fallback ladder: lr × {1, 0.5, 2} at steps × 1, then the same at steps × 2. **At most 6 attempts per store.** No other tuning |
| 6 | **When a sham cannot be matched** | If no ladder rung passes all T-FORGET QC criteria, the seed has a **sham-machinery failure**. The criteria are: X lost ≥ 50%; Y lost ≤ 3%; lost ≤ 5% in **each** retained-fact category (Z_random, Z_near_entity_X, Z_near_entity_Y, Z_near_repr, Z_far; v2, D28); ≥ 8/10 site ratios within [0.8, 1.25]; \|fluency change\| ≤ 0.1 SD; matched pairs ≥ minimum |
| 7 | **Are such items excluded?** | *Item level:* unmatched X_lost and Y items are excluded **only from the matched-pair endpoints** (P1, F1, K6, S1/S2). They remain in K1, K2, K3. *Seed level:* see 8 |
| 8 | **Exclusion rule** | A seed with a sham-machinery failure (on store A, or on twin B for S1/S2) is excluded from all confirmatory analyses and replaced by the next seed in the pre-registered sequence. The decision is computed by code from store-level QC only, before any monitor is evaluated. **If more than 5 of the first 25 seeds fail, the sham machinery is declared a design failure and the study stops (no confirmatory claims).** |
| 9 | **Minimum valid matched items per seed** | ≥ **100** matched (X_lost, Y) pairs for T-FORGET, and ≥ **60** matched (lost, retained) pairs for T-INTERF, at the frozen caliper. A seed below these is treated as in 8 |

## Additional sham diagnostics

These are reported for every confirmatory seed and do not use monitors:

- G1–G3 generic-statistics classifier CV AUROC (X_lost vs matched Y; X_lost vs Y);
- G4 raw-state probe (X_retained vs Y);
- Y margin retention;
- X_retained margin change.

The pre-declared thresholds and their consequences are in `logs/calibration_plan.md`.
