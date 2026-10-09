# D2-S0: analytical satisfiability of the D2 store-only gates (lesson R6)

Written and committed **before any D2 store run.** The only data used are intact-store margin distributions from *retired* historical plain stores (9031–9033, v3 development; E = 60). No interference or continued learning was run on them.

## 1. System and population

- **System:** the plain v1–v3 parametric store (4 blocks, d = 128), trained 60 epochs on the unchanged world. There is a single route: no explicit memory and no B1 component.
- **Competence change:** continued training on the reserved interference entities' facts and mentions, with no replay (`interventions.interference`). This is one process, applied identically to everything the store knows.
- **Population:** every base-correct EV fact of the base store, all equally exposed (about 1,650 per seed).
- **Competence:** C = answer margin (nats). ΔC_i = C_post,i − C_pre,i, on identical input tokens.

## 2. Binding store-only gates (from the PI's list of scientifically necessary requirements)

| # | Gate | Threshold | Unit |
|---|---|---|---|
| G0 | Base-store QC | trained ≥ 0.98, unknown ≤ 0.10, fluency AUROC ≥ 0.80 (existing v1–v3 values) | each seed |
| G1 | Enough competence change | lost fraction (C_post < 0) ∈ [0.10, 0.50] (existing P2 range) | each seed |
| G2 | Continuous spread | **IQR(ΔC) ≥ 1 nat** (scale choice in §4) | each seed |
| G3 | Identifiability (F3–F5) | median over seeds: R²_pre < 0.90, R²_gen < 0.90, R²_joint < 0.95 (existing thresholds, outcome ΔC) | across seeds |
| G4 | No trivial bookkeeping explanation | R²(C_post \| intervention-bookkeeping features) < 0.90 (existing audit kill threshold) | each seed |
| G5 | Input identity | evaluation tokens identical before and after (structural, asserted) | always |

**Not carried over from B1 (single route):**
- retained-strong tiers relative to intact competence;
- memory redundancy;
- Y-transplant neutrality;
- Z-locality and collateral sets. Interference is global by design, and the whole known population is the population.

**Report-only, so they impose no satisfiability constraint:**
- matched-development control;
- familiarity diagnostics;
- exposure and relation strata;
- F6 / F6′;
- IQR(C_post);
- residual spread;
- D50 reporting.

## 3. Facts about the intact plain store (retired seeds)

| Seed | n base-correct EV | C_pre quantiles 5 / 10 / 25 / 50 / 75 / 90 / 95% | IQR |
|---|---|---|---|
| 9031 | 1,659 | 5.08 / 5.29 / 5.69 / 6.13 / 6.55 / 7.00 / 7.32 | 0.86 |
| 9032 | 1,685 | 5.00 / 5.21 / 5.57 / 5.98 / 6.45 / 6.91 / 7.25 | 0.89 |
| 9033 | 1,643 | 4.85 / 5.03 / 5.41 / 5.81 / 6.26 / 6.68 / 7.01 | 0.85 |

Also: D_C (known − unknown mean margin) ≈ 11.9–12.5 nats on 9001–9003, and every known EV fact is base-correct. **Intact competence is concentrated:** median about 6 nats, SD about 0.64.

## 4. Joint satisfiability of G1–G4

**Single scale.** C_pre and C_post are margins of the same network on the same tokens. No gate compares two routes or uses a ratio to an intact quantity, so the v4.2 incompatibility (strong-route margin versus backup margin) cannot arise.

**G1 ∧ G2.** Model the change as ΔC_i = −(μ + σ Z_i), with Z standardised heterogeneity, and C_pre ~ (6, 0.64²).

- Then C_post has mean 6 − μ and SD √(0.41 + σ²). Lost = P(C_post < 0) varies continuously from 0 to 1 in μ for any σ, so every G1 value is reachable.
- G2 holds iff σ ≥ 1/1.35 ≈ 0.74 nats (normal Z).
- **Example:** μ = 5, σ = 1.5 gives lost ≈ 0.27, IQR(ΔC) ≈ 2.0, IQR(C_post) ≈ 2.2. Both gates pass.
- **Feasible region:** σ ≥ 0.74 nats with mean drop μ ∈ [6 − 1.28·√(0.41 + σ²), 6] nats: 4.75–6.0 at σ = 0.74, 3.9–6.0 at σ = 1.5, and 2.1–6.0 at σ = 3.
- **Degenerate case** (σ → 0, a uniform drop): G2 fails whatever the lost fraction. Lost then jumps from 0.10 to 0.50 within about a 0.8-nat window of drop size (5.2–6.0 nats). This regime is scientifically useless (no item-level variation in the change), and G2 exists to exclude it.

**Why G2 uses ΔC rather than C_post.**
- Because IQR(C_pre) ≈ 0.87 < 1, a uniform drop gives IQR(C_post) ≈ 0.87, so C_post's IQR is not inherited from intact competence.
- Whenever the change dominates, IQR(C_post) ≈ IQR(ΔC).
- The spread that matters scientifically is the spread *of the self-generated change*, so ΔC is the appropriate scale (the PI's clause allowing another scale). IQR(C_post) is reported.

**G3 against G1 and G2.**
- Identifiability concerns how predictable ΔC is from pre-states, from generic change, or from both. There is no logical relation to the lost fraction.
- G2 guarantees ΔC has variance, so the R² values are well defined.
- A high R²_pre would mean D2 merely reads out pre-existing susceptibility. That is the empirical question, not a contradiction.
- **D50 regime:** pre-state dimension p = 10 sites × 128 = 1,280, plus 7 covariates, against n ≈ 1,650 items per seed (p/n ≈ 0.78). Diffuse susceptibility may be partly unrecovered, so a low R²_pre does not prove the adjustment is complete. Dimensions, n and cross-fitted performance are reported, and the label-A-only rule (D54) already guards the inference.

**G4.** The bookkeeping features are coarse data-overlap counts between each item and the interference set (§5). There is no conflict with G1–G3. The threshold 0.90 is the existing kill value.

**G5.** This is structural.
- Evaluation tokens [Q s1 s2 r A] depend only on the world.
- Interference entities come from a reserved, disjoint name pool (asserted when the world is built).
- No original fact sequence is trained during interference, and the vocabulary is unchanged.

All of this is unit-tested.

**Replication.** The store seed is the unit. G0, G1, G2 and G4 must hold on every development seed; G3 uses medians. No conflict.

**Process identity.** The interference rule (constant-lr Adam on random batches of new facts and mentions; stop when new-fact accuracy ≥ 0.95, checked every 50 steps; cap max_steps) is the same for every item and every seed. Step counts may differ by seed and are reported.

## 5. Bookkeeping features (G4)

For item (e, r) with answer v\*, computed from the world only:

- the number of interference facts with relation r and answer v\* (the same value token);
- the number of interference facts with relation r;
- the number of interference entities sharing e's first syllable;
- the number sharing e's second syllable;
- exposure class.

## Conclusion

The binding D2 gates are **jointly satisfiable by construction**: a heterogeneous drop (SD ≥ 0.74 nats) whose mean lies in the §4 window satisfies G1 and G2, and G3–G5 impose no conflicting constraint. Whether continued learning actually lands in this region, and whether the resulting change is identifiable, is the empirical question for D2-S2 and S3.
