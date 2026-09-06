# Observational screening: bounded compute benchmark

First frozen scm_4 complete-target setting: graph multiplier 1/16, fixed target multiplier 16. Fresh workers run sequentially with two solver threads, seed 20260901, one numerical-library thread, 120-second solver limits and identical common arrays/target constraints/numerical bounds. Other original pilot work may run concurrently; this is not an isolated-machine benchmark.

| Condition | Candidate pairs | Parent sets | Solver seconds | CPU seconds (user + system) | Gap (%) | Nodes explored | Directed TP / FP | Skeleton TP / FP | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | 45 | 5120 | 120.003 | 181.175 | 10.484819 | 18667 | 23 / 13 | 23 / 13 | ok_nonoptimal |
| observational_screen | 32 | 1472 | 22.714 | 35.067 | 0.000000 | 27786 | 20 / 11 | 20 / 11 | ok |

Previously completed unscreened fit at this same setting: 2056.547 solver seconds with eight threads, gap 0.008673%, directed TP/FP 23/13. Different thread count and execution conditions mean this is context, not a matched speedup estimate.

The observational screen is fixed at alpha=sqrt(log(10)/10000), support threshold 1e-8. Its tighter numerical diagnostic converged and matched the existing synthetic helper support. Screening itself took about 0.002776874999999901 seconds in its saved audit, reported separately from the profiled-score/solve timing.

Screening removes 28.9% of candidate pairs and 71.25% of parent sets (3.48 times fewer). This does not mathematically imply an equivalent wall-time speedup. The screened objective is optimized over a different, smaller feasible set: objective gaps and certificates refer to each problem separately, not to the unscreened optimum. Three documented true edges are excluded. Recovery at a time limit is descriptive of the returned incumbent, not its converged performance.

All results including failures and nonoptimal solutions are retained. No change was made to the original pilot or the synthetic defaults. This single setting cannot predict full-path runtime.

Reproduce with `.venv-dcdi/bin/python -B experiments/benchmark_chamber_screen.py` and the four numerical-library thread variables set to 1. Existing results are validated and reused; they are not overwritten.
