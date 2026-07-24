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

The implementation accepts unknown, known, or partially known intervention
targets. It can restrict candidate edges to a supplied moral graph or use the
complete graph as the superstructure.

The formulation requires bounded `Gamma` entries. In particular, `MIP.py`
places a positive lower bound on the diagonal of `Gamma`; without this bound,
no finite big-M constant can cover the logarithmic intervention-selection
costs. The selector bounds are computed automatically from the empirical
second-moment matrices and the chosen coefficient bounds.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `MIP.py` | Current Equation (4.4) implementation and command-line runner |
| `MIP_naive.py` | Original full joint-environment formulation from Equation (4.2) |
| `GNIES.py` | GNIES rank baseline using the same synthetic-data defaults |
| `MICP.py` | Legacy joint-environment formulation, retained for reference |
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
python -m pip install numpy causaldag gnies gurobipy "pgmpy==0.1.25"
```

Install `pandas` as well when using the legacy data-loading utilities:

```bash
python -m pip install pandas
```

The R dependencies are only needed for historical comparison and evaluation
scripts.

The `pgmpy` pin is required when using Python 3.9 because current `pgmpy`
releases require Python 3.10 or newer.

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
8. skeleton true-positive rate (TPR); and
9. skeleton false-positive rate (FPR).

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
skeleton TPR/FPR. GNIES does not return a target indicator for every
environment-variable pair, so an environment-specific target error is not
available and should not be compared with the MIP target error.

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
