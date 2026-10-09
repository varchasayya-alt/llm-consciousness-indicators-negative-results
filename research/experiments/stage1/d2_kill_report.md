# D2 KILL REPORT: self-generated competence change through continued learning (store-only kill test)

| Field | Value |
|---|---|
| Date | 2026-10-03 |
| Status | **D2 STOPPED under its pre-declared rule** (`d2_protocol.md` §5–6): no interference setting satisfies the binding gates. Per the PI: no further intervention tuning; move to D3 (proposal in `d3_protocol_proposal.md`) |
| Commits | pre-run `7742d07`; engineering fix `04b9a4e` (D66); results in this commit |
| Seeds | D2 development 9101–9103 (used). **Proposed D2 validation 9111–9113 untouched.** Validation 9051–9053 and confirmatory 1001–1040 untouched |
| Not done | No monitor or controller trained; nothing frozen or registered; no model download |
| Raw results | `results/raw/calibration/d2_results.json`, `d2_S3_{9101,9102,9103}.json`, `log_d2_*.txt` |

## 1. Exact stopping gate

**G1 (enough but not excessive competence change):** lost fraction ∈ [0.10, 0.50] on every development seed. It fails at **every** lr on **every** seed.

| lr | Lost 9101 / 9102 / 9103 | Mean | Steps to new-fact accuracy ≥ 0.95 |
|---|---|---|---|
| 3e-4 | 0.725 / 0.742 / 0.780 | 0.749 | 300 / 300 / 350 |
| 1e-3 | 0.891 / 0.899 / 0.913 | 0.901 | 250 / 250 / 250 |
| 3e-3 | 0.963 / 0.961 / 0.964 | 0.963 | 350 / 400 / 400 |

**All other binding gates pass at every lr:**

- **G0 base QC:** trained 1.00, unknown about 0.03, fluency AUROC 1.00 on every seed.
- **G2 spread:** IQR(ΔC) 3.72–4.43 nats.
- **G3 identifiability:** medians R²_pre 0.00–0.02, R²_gen 0.20–0.35, R²_joint 0.11–0.27.
- **G4 bookkeeping:** R² ≤ 0.025.
- **G5 input identity:** holds.

## 2. Failure mechanism

- **What the process does.** Continued learning on the 1,000 new entities with no replay, run until the new facts are learned (≥ 95%), is **near-catastrophic for old knowledge**. Accuracy on all previously known facts falls from 1.00 to 0.23–0.27 at the lowest lr, to about 0.10 at 1e-3 and to about 0.04 at 3e-3.
- **Where the drops land.** The median ΔC is −7.6 to −8.2 nats at 3e-4 and −11.6 to −11.9 at 3e-3.
- **Why that misses the window.** The S0 note showed that lost ≤ 0.50 requires a mean drop of at most about 6 nats, because intact margins sit at about 6 ± 0.6 nats. The observed drops lie beyond that window.
- **Why no grid value fixes it.** Loss rises monotonically with lr. The stop-on-learning rule ties the amount of continued training to how long the new facts take to learn (250–400 steps), not to a dose that could be held inside the window. Within the frozen grid there is no lever that reaches it.
- **The gates were not the problem.** The S0 joint-satisfiability argument was correct: the gates do not conflict. The failure is empirical: the magnitude of self-generated change in this store lies outside the satisfiable region.

## 3. What D2 nevertheless established (store-only; no monitor; Level 1)

| Property | Observation |
|---|---|
| **Graded change** | IQR(ΔC) about 3.7–4.4 nats; ΔC quantiles at 3e-4 (5–95%) ≈ −12.4 to −3.4 nats |
| **Not pre-predictable** | R²_pre ≈ 0. The residual SD of ΔC after the cross-fitted pre-information prediction ≈ the raw SD of ΔC (e.g. 2.72 vs 2.75 nats on 9101 at 3e-4). D50 regime: p = 1,280 + 7 against n ≈ 1,650 (p/n ≈ 0.78) |
| **Partly generic** | R²(ΔC \| generic-change features) 0.19–0.38; R²(C_post \| generic) 0.20–0.40 |
| **Post-change competence not linearly decodable** | Cross-fitted ridge from the post-change read-site states: R²(C_post) ≈ 0 (−0.01 to 0.02) at every lr. From output summaries (sorted probabilities, entropy): 0.00–0.08 (0.15 at 3e-3). Same pattern as the v4.2 audit |
| **No trivial structure** | Exposure: R² ≤ 0, AUROC ≈ 0.50–0.54; lost is similar across relations |
| **Matched-development control (CTRL)** | 3e-4: lost **0.000**, median ΔC ≈ −0.1 (the change is interference-specific). 1e-3: lost 0.17–0.19. 3e-3: lost 0.76–0.80. Continued constant-lr optimisation on other material alone destroys unrehearsed facts at high lr. Item-level CTRL change does not predict INTERF change (R² ≈ 0, \|ρ\| ≤ 0.05) |
| **Familiarity** | Interference lowers the name fluency of the original entities (−0.7 nats at 3e-4; −3.9 at 3e-3), CTRL barely (−0.06 to −0.14). Spearman(ΔC, Δfluency) ≈ 0: competence loss is not a familiarity effect |
| **F6 / F6′ (report-only)** | F6 feasible on 9101/9102 at 3e-4 (455 / 435 pairs, max SMD ≤ 0.08) but not on 9103 (0.109); F6′ feasible everywhere |

**Reading.** Self-generated change in this store has an attractive identification profile: graded, unpredictable from pre-states, only partly generic, and not explained by familiarity or bookkeeping. **But it is all-or-nothing in magnitude:** the system forgets most of what it knew, rather than some of it.

## 4. Deviations (D66; no method change)

1. **Chain parent stopped.** The chain's parent shell hit the 2-hour cap for background tool tasks. The S3 workers continued unaffected, and SELECT was run by hand with the same code.
2. **LAPACK failure.** SVD non-convergence in one audit fit (9102, 3e-3; the store is numerically sound). Fixed with a gesvd fallback, identical whenever gesdd converges (tested), committed before recomputing that single cell. That cell's lost fraction (0.961) already failed G1.
3. **Wait-loop bug.** My monitoring wait-loops had a path-quoting bug. No effect on data.

## 5. What this teaches (adds to R1–R7)

**R8: reachability.** An S0 check must establish not only that the gates are jointly satisfiable, but also that the intervention's **pre-declared dose lever can reach** the satisfiable region. Here the lever was lr under a stop-on-learning rule. That needs either a lever that provably spans from no effect to full effect, or evidence on retired or burned material. D2's S0 checked the first condition and not the second.

**Not proposed (per the PI):** other lrs, step-based stopping, replay mixtures, smaller interference sets, or any other D2 variant.

## 6. Next

D3, as pre-specified. The short updated protocol proposal is in `d3_protocol_proposal.md`. It needs the PI's approval, **including explicit approval of the model and dataset downloads it lists.**
