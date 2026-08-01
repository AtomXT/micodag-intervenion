# Mixed-Integer Learning of DAGs with Interventions

This repository contains research code for learning a directed acyclic graph
(DAG) from observational and interventional data when the intervention targets
may be unknown.

The project is being rewritten around the mixed-integer formulation in Equation
(4.4) of the accompanying dissertation chapter. Earlier coordinate-descent and
alternating-optimization methods were removed because their performance was
sensitive to the variable update order. The current proposed method does not
depend on a supplied variable ordering.

## Current Method

`MIP.py` is the primary implementation. It estimates:

- one baseline matrix `Gamma` whose off-diagonal support represents the DAG;
- binary edge variables and continuous topological-order variables enforcing
  acyclicity;
- binary intervention indicators for each non-observational environment; and
- epigraph variables selecting the intervened or non-intervened cost for each
  variable and environment.

The objective combines the weighted observational likelihood, a DAG sparsity
penalty, and the optimized interventional costs. Sample-size weights are
computed as

```text
w_e = n_e / sum_e(n_e).
```

The command-line runners estimate an undirected superstructure from the
observational sample using graphical lasso. The only exposed graphical-lasso
setting is `--alpha` (default `0.2`). The imported optimization functions can
still accept a supplied superstructure or use the complete graph.

The formulation requires bounded `Gamma` entries. In particular, `MIP.py`
places a positive lower bound on the diagonal of `Gamma`; without this bound,
no finite big-M constant can cover the logarithmic intervention-selection
costs. The selector bounds are computed automatically from the empirical
second-moment matrices and the chosen coefficient bounds.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `MIP.py` | Current Equation (4.4) implementation and command-line runner |
| `MIP_profiled.py` | Fully profiled parent-set and intervention-pattern MILP |
| `MIP_naive.py` | Original full joint-environment formulation from Equation (4.2) |
| `GNIES.py` | GNIES rank baseline using the same synthetic-data defaults |
| `DCDI.py` | Adapter for the official DCDI-G perfect/unknown-target implementation |
| `UTIGSP.py` | Adapter for UT-IGSP from `causaldag` |
| `data/DataGeneration.py` | Python translation of the original R synthetic-data generator |
| `data/DataGeneration_vary_n.py` | Fixed 20-node graph with `n = 100, 500, 1000` in every environment |
| `data/DataGeneration.R` | Original data-generation reference |
| `data/SyntheticData/` | Main synthetic graphs and multi-environment datasets |
| `data/Vary_n/` | Fixed-graph sample-size experiment |
| `data/RealData/` | Sachs observational and interventional data |
| `src/utils.py` | Data-loading and graph-evaluation utilities |
| `experiment_results/` | Historical estimates, summaries, and plots |
| `experiments/` and notebooks | Earlier experimental and exploratory code |

The old experiment scripts and notebooks have not all been migrated to the new
formulation. Some still describe or import removed coordinate-descent code and
should be treated as historical material rather than the current runnable
pipeline.

## Installation

The current MIP requires:

- Python;
- NumPy and `causaldag`;
- Gurobi and `gurobipy`; and
- a working Gurobi license.

Using a virtual environment is recommended:

```bash
python3.9 -m venv /path/to/venvs/python39
source /path/to/venvs/python39/bin/activate
brew install libomp
python -m pip install numpy scikit-learn "causaldag==0.1a163" gnies gurobipy \
  "pgmpy==0.1.25"
```

Install `pandas` as well when using the legacy data-loading utilities:

```bash
python -m pip install pandas
```

The R dependencies are only needed for historical comparison and evaluation
scripts.

The `pgmpy` pin is required when using Python 3.9 because current `pgmpy`
releases require Python 3.10 or newer.

### Project-local DCDI installation

`DCDI.py` uses the official DCDI source pinned at commit
`594d328eae7795785e0d1a1138945e28a4fec037`. The source and its Python
environment are installed below the project and ignored by Git. The sparse
checkout avoids downloading DCDI's large example-data directory.

From the repository root:

```bash
mkdir -p external
git clone --filter=blob:none --no-checkout \
  https://github.com/slachapelle/dcdi.git external/dcdi
git -C external/dcdi sparse-checkout init --cone
git -C external/dcdi sparse-checkout set dcdi
git -C external/dcdi checkout 594d328eae7795785e0d1a1138945e28a4fec037

python3.9 -m venv .venv-dcdi
.venv-dcdi/bin/python -m pip install -r requirements-dcdi.txt
```

The adapter verifies the commit before every fit. It stages only sample values
and environment labels. Although the upstream loader requires DAG and target
files, the adapter supplies zero/blank placeholders, so simulation truth is
not exposed to DCDI's objective. Upstream R-backed reporting metrics are
disabled because this project performs its own I-CPDAG evaluation.

## Running the MIP

From the repository root, the following command fits graph 2 (`p = 20`) using
the first generated dataset:

```bash
python MIP.py \
  --graph 2 \
  --iteration 1 \
  --lambda-graph 0.01 \
  --lambda-delta 0.01 \
  --time-limit 1000
```

The graph index selects the included synthetic datasets:

| Graph | Variables |
| ---: | ---: |
| 1 | 10 |
| 2 | 20 |
| 3 | 50 |
| 4 | 100 |
| 5 | 200 |

The script reports:

1. the estimated baseline `Gamma`;
2. the estimated intervention-target indicator for each environment;
3. the final MIP gap;
4. the objective value; and
5. the solver runtime;
6. `d_cpdag`, the entrywise distance between the estimated and true I-CPDAGs;
   and
7. the environment-by-variable intervention-target Hamming error;
8. equivalence-class false discovery proportion (FDP); and
9. equivalence-class true discovery proportion (TDP), as defined by the
   nested comparisons over the estimated and true I-MECs in Taeb et al.
   (2024).

The FDP/TDP calculation is exact: it enumerates the consistent DAG extensions
of both I-CPDAGs. Its cost can therefore be exponential when either I-CPDAG
contains many reversible edges.

The optimization function can also be imported directly:

```python
from MIP import optimization

Gamma, targets, gap, objective, runtime = optimization(
    data,
    moral_graph,
    l=0.01,
    l_delta=0.01,
)
```

Here, `data[0]` must be the observational sample and the remaining entries must
be the interventional environments. Each entry may be a NumPy array or pandas
DataFrame with the same number and ordering of variables.

## Running the Fully Profiled MIP

`MIP_profiled.py` profiles out both the environment-specific parameters and
the remaining baseline `Gamma` column for every candidate parent set. It then
solves a linear parent-set selection MILP and reconstructs `Gamma` and the
environment-specific intervention targets from the selected local optima.

```bash
python MIP_profiled.py --graph 2 --iteration 1
```

The formulation is exact when its reported local optima lie within the
specified `Gamma` bounds. The script checks this condition rather than silently
using an invalid closed-form score. `--max-parents` can restrict the candidate
parent-set size when the supplied superstructure is dense.

## Running the Naive MIP

`MIP_naive.py` implements Equation (4.2), retaining a separate `Gamma` matrix
for every environment. It uses the same defaults and output metrics as
`MIP.py`, but is expected to be substantially larger and slower.

```bash
python MIP_naive.py --time-limit 1000
```

## Running GNIES Rank

`GNIES.py` uses the same default dataset as `MIP.py`: graph 2 (`p = 20`) and
iteration 1. Its default GNIES penalty is 6, matching the earlier rank
experiments.

```bash
python GNIES.py
```

The runner prints the estimated I-CPDAG, the zero-based union of estimated
intervention targets, score, runtime, `d_cpdag`, union-target error, and
equivalence-class FDP/TDP. GNIES does not return a target indicator for every
environment-variable pair, so an environment-specific target error is not
available and should not be compared with the MIP target error.

## UT-IGSP comparison and standalone DCDI-G

The scheduled comparison adds UT-IGSP to the existing PS-MIP and GnIES
results. The checkpointed comparison runner supports only these three methods;
DCDI-G remains available separately through `DCDI.py`:

```bash
python -m experiments.run_tdp_fdp_experiments \
  --trial 1 \
  --graphs 1 \
  --methods utigsp \
  --penalties 0.001 \
  --output-dir experiment_results/tdp_fdp_utigsp/manual
```

The comparison uses UT-IGSP with `depth=4`, `nruns=10`, seed 42, and tied
`alpha_inv=alpha` over `{0.2,0.1,0.01,0.001,1e-5,1e-7,1e-9}`.

The comparison omits DCDI-G because this project's linear-Gaussian sparse-graph
setting matches the exception identified in the
[DCDI paper](https://proceedings.neurips.cc/paper/2020/file/f8b7aa3a0d349d9562b424160ad18612-Paper.pdf),
where DCDI shows no clear overall advantage over UT-IGSP. DCDI remains
available for standalone use and future nonlinear or denser experiments.

DCDI-G can be run directly through the project-local installation. Its
artifacts are retained below `experiment_results/dcdi_manual` by default, so
an interrupted or completed matching fit can be retried safely:

```bash
.venv-dcdi/bin/python DCDI.py \
  --graph 2 \
  --iteration 6 \
  --graph-penalty 0.1 \
  --target-penalty 0.001 \
  --mu-init 1e-2 \
  --hidden-dim 8
```

It prints the estimated DAG, I-CPDAG, environment-specific target matrix,
runtime, cache/artifact information, and `d_cpdag`/target-error/FDP/TDP.

UT-IGSP can also be run directly on one synthetic dataset. It prints the
estimated DAG, I-CPDAG, environment-specific target matrix, runtime, and the
same `d_cpdag`/target-error/FDP/TDP diagnostics as the MIP scripts:

```bash
python UTIGSP.py --graph 2 --iteration 6 --alpha 0.001
```

UT-IGSP therefore contributes a one-dimensional significance-level path. Its
environment-by-variable target matrix is converted with the same
`interventional_cpdag()` function and evaluated with the same exact FDP/TDP
metric as PS-MIP.

The new Quest array runs only UT-IGSP. Each of its ten array tasks handles one
complete trial across all three graphs and all seven tied significance levels.
Each task writes an isolated result fragment, so the completed files in
`experiment_results/tdp_fdp` are never rewritten and concurrent tasks cannot
overwrite one another. Per-fragment advisory locks make an accidental
overlapping submission fail fast instead of corrupting a retry:

```bash
mkdir -p experiments/quest_jobs/outlog
sbatch experiments/quest_jobs/tdp_fdp_utigsp_array.sh
```

The UT-IGSP array contains 10 tasks, one per trial. Each task sequentially runs
the 21 graph/significance-level combinations for that trial. Its default result
location is `experiment_results/tdp_fdp_utigsp`.

After the array finishes, aggregate the old and new result directories with:

```bash
python analysis/aggregate_tdp_fdp_results.py
```

Aggregation preserves every ingested row in `raw_results.csv`, validates it
against the declared comparison grids, and uses the validated rows for
`combined_results.csv`, summaries, and figures. Unknown or duplicate
combinations cause separate diagnostics rather than silently changing a curve.

Combined CSV summaries and a validation report are written to
`experiment_results/tdp_fdp_comparison`; the three-method figure is written to
`analysis/tdp_fdp.pdf` and `analysis/tdp_fdp.png`. The same aggregation command
also regenerates the manuscript's trial-wise best-penalty table data at
`experiment_results/tdp_fdp/best_dcpdag_summary.csv` and the LaTeX table included
by the writeup at `experiment_results/tdp_fdp/best_dcpdag_table.tex`.

## Synthetic Data

The main generator reproduces the procedure in `DataGeneration.R`:

- ordered random DAGs with edge probability `2 / p`;
- edge weights sampled from `{-0.8, -0.6, 0.6, 0.8}`;
- observational noise variances sampled from `{1, 2, 4}`;
- hard interventions that remove incoming edges at each target;
- five interventional environments by default; and
- ten repeated datasets per environment.

Review the available generator options before writing new data:

```bash
python data/DataGeneration.py --help
python data/DataGeneration_vary_n.py --help
```

Both generators use relative paths. They refuse to replace a nonempty output
directory unless `--overwrite` is explicitly supplied.

## Evaluation

`src/utils.py` contains the shared graph metrics. In particular,
`interventional_cpdag()` constructs an I-CPDAG from a DAG and its
environment-specific intervention targets. `cpdag_distance()` then computes
the entrywise L1 difference between the estimated and true I-CPDAG adjacency
matrices. This project calls that quantity `d_cpdag`; it is intentionally
different from structural Hamming distance (SHD).

## Project Status

The data and core method have been cleaned up, but the rewrite is still in
progress. The next work should focus on:

- validating `MIP.py` through small cases with known optima;
- writing experiment runners for the new formulation;
- adding Sachs-data preprocessing and evaluation;
- defining reproducible penalty-selection rules; and
- replacing or archiving the remaining legacy scripts and results.
