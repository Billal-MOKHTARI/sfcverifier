# Evaluation report

Generated 2026-10-12T22:42:24 - profile `full`.

## Protocol

* 100 synthetic block-structured SFCs (tiers small / medium / large / complex) and 1 reference case study.
* Clean test trace of 50,000 scans per graph (scan = 1.0 s, duration jitter ±20 %), independent training trace for the baselines.
* 40 episodes per attack class and graph, attack windows of 1-8 scans, 5 scans of horizon after each window; 44,040 episodes in total.

## Key results

* False alarms of the Phi verifier: **0** on 4,999,900 clean transitions (FPR 95 % upper bound 6.0e-07), without any training.
* Structure-violating attacks: **100.00 %** detected (95 % CI 99.99-100.00 %), 98.8 % on the first manipulated scan.
* Structure-preserving (stealthy) attacks: 15.7 % detected - these executions are legal for the SFC structure, which is the expected limit of the method (replays are only caught at the splice discontinuity).
* Learned transition whitelist (100,000 training scans): 14.84 false alarms per hour.

## Detection per attack (Phi verifier)

| Attack | Family | Episodes | Detected | Onset | Median latency (scans) |
|---|---|---:|---:|---:|---:|
| Deadlock | violating | 4000 | 100.0 % | 100.0 % | 0 |
| Coil manipulation | violating | 4000 | 100.0 % | 90.8 % | 0 |
| Spurious activation | violating | 4000 | 100.0 % | 100.0 % | 0 |
| Step suppression | violating | 4000 | 100.0 % | 100.0 % | 0 |
| Illegal jump | violating | 4000 | 100.0 % | 100.0 % | 0 |
| OR exclusivity violation | violating | 4000 | 100.0 % | 100.0 % | 0 |
| Partial AND divergence | violating | 3840 | 100.0 % | 100.0 % | 0 |
| Premature AND convergence | violating | 3760 | 100.0 % | 100.0 % | 0 |
| Delay (freeze) | stealthy | 4000 | 0.0 % | 0.0 % | - |
| Wrong OR branch | stealthy | 4000 | 0.0 % | 0.0 % | - |
| Replay | stealthy | 4000 | 47.1 % | 0.0 % | 5 |

Figures: `figures/fig_detection`, `fig_false_alarms`, `fig_rules`, `fig_scalability` (SVG and PDF). LaTeX tables: `tables/`.
