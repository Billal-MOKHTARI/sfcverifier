# SFCVerifier: What a Runtime Monitor Can and Cannot Detect in PLC Sequential Function Charts

Detects control-flow anomalies in a running SFC (PLC control logic) by checking every pair of
consecutive activation vectors `S(t-1) → S(t)` against the transition matrix `X`.

## Method

`X[i, j] = δ_ij^(k)`: edge `i → j`; edges leaving a step with the same `k` fire together (AND),
different `k` are exclusive alternatives (OR). Steps with several predecessors also carry
**convergence groups**: the predecessor sets that enter them together (AND-convergence = one
group with every predecessor, OR-convergence = one singleton per predecessor).

```
l   = S(t-1) ∘ (1 − S(t))          steps that left
n   = S(t)   ∘ (1 − S(t-1))        steps that entered
E(t) = X · diag(S(t))
Φ(t) = diag(l) · X · diag(n) = diag(l) · E(t) · diag(n)     fired-transition matrix
```

| Rule | Condition |
|---|---|
| R1 no deadlock | `1ᵀ S(t) ≥ 1` |
| R2 justification | `(1ᵀΦ)_j ≠ 0` for every entered `j` |
| R3 divergence | `(Φ1)_i` equals exactly one group `G_i^k` for every left `i` |
| R4 convergence | `(1ᵀΦ)_j` equals the sum over one convergence group of `j` |

Steps that stay active are in neither `l` nor `n`, so they never interfere.
`sfcverify/symbolic.py` implements these formulas literally with SymPy. `sfcverify/verifier.py`
implements the same rules with integer bitmasks (about 1 µs per transition), and the test-suite
checks that both give the same verdicts.

## Layout

```
data/graphs/          the graphs (JSON): CS_reference (the original 26-step SFC) + G001…G100
sfcverify/
  model.py            SFCModel: matrix X (labels k), convergence groups, durations, JSON I/O
  verifier.py         rules R1–R4 (+ ablation presets)
  symbolic.py         literal SymPy version of the formulas (reference, used by the tests)
  simulator.py        timed token game producing normal traces (independent of the verifier)
  generator.py        random block-structured SFC generator (Seq / Par / Alt / Loop, nested)
  attacks.py          11 attack classes injected into traces
  baselines.py        learned transition / state whitelists
  evaluation.py       experimental protocol (parallel over graphs)
  report.py           figures (SVG + PDF), LaTeX tables, REPORT.md
  traces.py           CSV traces (id, timestamp, 0 … n-1)
  cli.py              command line
gui/                  web interface (server.py + static/)
tests/                pytest suite
results/              evaluation output (summary.json, figures/, tables/, REPORT.md)
legacy/               previous code, kept for reference only
```

## Usage

```bash
python3 -m sfcverify gui                      # web interface on http://127.0.0.1:8765
python3 -m sfcverify evaluate                 # full evaluation (~30 s on 16 cores) + report
python3 -m sfcverify evaluate --profile quick
python3 -m sfcverify simulate CS_reference --scans 2000 --out trace.csv
python3 -m sfcverify verify CS_reference trace.csv
python3 -m sfcverify graphs --regenerate      # rebuild the 100 benchmark graphs (seed 2026)
python3 -m pytest                             # tests
```

GUI deep links: `#graph=G092_complex&attack=premature_join&seed=4&frame=alarm&tab=matrices`.

## Graph file format

```json
{
  "name": "CS_reference",
  "initial": [0],
  "steps": [{"id": 0, "name": "X0", "duration": 0.0}, ...],
  "matrix": [[0,1,0,...], ...],                 // k labels, 0 = no edge
  "convergence": {"9": [[8, 11]], "11": [[10], [16]], ...}
}
```

## Evaluation protocol

* **Graphs**: 100 random block-structured SFCs in four tiers. *small* (11–25 steps), *medium*
  (26–58) and *large* (62–120) graphs, plus *complex* graphs (41–96 steps) with nesting depth up
  to 8, up to 21 simultaneously active steps, and combined join/fork transitions (e.g.
  `{24,25,26} → {27,34}`). The reference 26-step SFC is reported separately.
* **Normal behaviour**: a timed token-game simulation with ±20 % duration jitter. It uses
  dead-path elimination at convergences and never reads the convergence groups or the verifier,
  so false alarms are not ruled out by construction. Every edge of every graph fires during the
  test traces.
* **Attacks** (40 episodes per class and graph, windows of 1–8 scans, plus a 5-scan horizon):
  * 8 *structure-violating* classes: deadlock, coil manipulation, spurious activation, step
    suppression, illegal jump, OR-exclusivity violation, partial AND divergence, premature AND
    convergence;
  * 3 *structure-preserving* (stealthy) classes: delay, wrong OR branch, replay. These are legal
    executions and mark the limit of any purely structural monitor.
* **Detectors**: the Φ verifier; three ablations (no divergence semantics, no convergence
  semantics, adjacency only); transition and state whitelists learned from 1k–100k scans.
* **Metrics**: detection rate with 95 % Wilson intervals, detection at onset, latency,
  false alarms per transition and per hour (95 % upper bound when zero), verification time.

Results: `results/REPORT.md`, figures in `results/figures/`, LaTeX tables in `results/tables/`.
