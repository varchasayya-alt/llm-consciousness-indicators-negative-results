# High-dimensional verification of S9 with the D49 nested-CV ridge (simulation only)

20 replicate studies x 20 seeds x 300 items; pre-state dimension 1280 (> items); ~50% loss.

| world | mean theta_pre | labels A / B / C | median cross-fitted R2_pre |
|---|---|---|---|
| H3 | 0.767 | 1.00 / 0.00 / 0.00 | 0.09 |
| null | 0.001 | 0.00 / 0.00 / 1.00 | 0.10 |
| rtm | -0.001 | 0.00 / 0.00 / 1.00 | 0.13 |
| susceptibility | 0.091 | 0.00 / 0.25 / 0.75 | 0.08 |
| H2_intensity | 0.464 | 0.00 / 1.00 / 0.00 | 0.08 |
| susceptibility_latent | 0.149 | 0.85 / 0.15 / 0.00 | 0.07 |
