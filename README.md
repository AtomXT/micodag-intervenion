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

The standalone MIP commands and the main experiment use the same persisted
datasets and the same graphical-lasso screening rule,
`5*sqrt(log(p)/n_screen)`. Imported optimization functions can still accept a
supplied superstructure or use the complete graph.

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
| `DCDI.py` | Adapter for the official DCDI-G perfect/unknown-target implementation |
| `GIES.py` | Thin adapter for the requested `juangamella/gies` Python package |
| `UTIGSP.py` | Adapter for UT-IGSP from `causaldag` |
| `data/DataGeneration.py` | Python translation of the original R synthetic-data generator |
| `experiments/generate_main_experiment_data.py` | Standalone generator for all persisted main-experiment datasets |
| `experiments/test_main_experiment_setup.py` | No-save, one-instance check of every main-experiment method |
| `experiments/run_main_experiment.py` | Main shared-data synthetic experiment driver |
| `analysis/aggregate_main_experiment.py` | Main TDP-FDP paths and best-`d_cpdag` potential tables |
| `experiments/quest_jobs/main_experiment_array.sh` | Ten-task Slurm array, one complete replicate per task |
| `src/main_experiment_data.py` | Main data schema, seed, serialization, and validation library |
| `src/main_experiment_cli.py` | Shared persisted-data options for standalone method commands |
| `src/utils.py` | Graph-evaluation utilities |
| `experiment_results/` | Generated outputs from the active main experiment |
| `archive/legacy_pipeline/` | Historical scripts, static datasets, and prior results |

The repository root now contains only the active experiment pipeline and its
runtime dependencies. Historical scripts, notebooks, static datasets, and
earlier result artifacts retain their original relative layout below
`archive/legacy_pipeline/`; see `archive/README.md` for the manifest and restore
notes.

## Installation

The current MIP requires:

- Python;
- NumPy and `causaldag`;
- Gurobi and `gurobipy`; and
- a working Gurobi license.

Install the tested dependencies into the Python environment that will run the
experiment (the same active environment is used for every method):

```bash
python3.9 -m pip install -r requirements-dcdi.txt
```

The versions in this file describe the tested compatibility environment; the
runtime does not require installed package versions to equal them. The runner
checks only that required packages and APIs are available. It records the
versions actually used and fingerprints the complete active Python and R
environments, so a changed environment cannot reuse an old checkpoint.
A working Gurobi license is still required. Install the R packages used only
by the unmodified DCDI reporting step:

```bash
Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); BiocManager::install(c("pcalg", "SID"), ask=FALSE)'
```

The `pgmpy` pin is required when using Python 3.9 because current `pgmpy`
releases require Python 3.10 or newer.

### DCDI installation

`DCDI.py` uses the official DCDI source pinned at commit
`594d328eae7795785e0d1a1138945e28a4fec037`. The minimal source snapshot needed
by this experiment is committed under `external/dcdi`; upstream example data
and unrelated methods are omitted. A normal GitHub clone or pull therefore
contains everything required to launch DCDI.

From the repository root:

```bash
python3 -m pip install -r requirements-dcdi.txt
```

`external/dcdi/VENDORED_SOURCE.json` records the official repository URL,
commit, and SHA-256 digest of every included upstream file. The adapter checks
that committed manifest and every source file before each fit, rejecting local
changes or unexpected files without requiring nested Git metadata.

The adapter verifies the exact source commit, confirms that required Python and
R imports work, records their actual versions, and rejects tracked changes or
untracked files before every fit. It then invokes the authors' `main.py`
directly. No DCDI model, objective, optimizer, or reporting function is copied
or changed. For current Matplotlib releases, a child-process-only compatibility
module translates the removed plotting keyword `padding` to its modern
`pad_inches` spelling. This affects only saved plot margins; the official DCDI
checkout and estimation code remain untouched.
The upstream report requires the R packages `pcalg` and `SID`.

Only file-format glue remains local. It stages sample values and environment
labels, and—because the upstream loader requires truth files—supplies zero/blank
placeholders. Simulation DAGs and targets therefore never enter DCDI's fitting
objective. The adapter reads the final author-produced DAG and learned target
probabilities and validates their shapes and conventions.

### Official competing implementations

The competing algorithms are not implemented in this repository:

| Method | Code used by the main experiment |
| --- | --- |
| DCDI-G | Authors' [`slachapelle/dcdi`](https://github.com/slachapelle/dcdi) checkout at commit `594d328eae7795785e0d1a1138945e28a4fec037`, executed through its official `main.py` |
| UT-IGSP | [`causaldag`](https://github.com/uhlerlab/causaldag) / `graphical-model-learning` author package API (tested with 0.1a163 / 0.1a8) |
| GnIES | [First-author `gnies`](https://github.com/juangamella/gnies) package using its full `approach="greedy"` search (tested with 0.3.3) |
| GIES oracle | Olga Kolotuhina and Juan L. Gamella's requested [`juangamella/gies`](https://github.com/juangamella/gies) Python package (tested with 0.0.3) |

`DCDI.py`, `UTIGSP.py`, and `GIES.py` only validate data and translate input or
output formats. PS-MIP is the proposed method in this project, so its local
implementation is intentionally outside this competitor-source rule. Every
result's `method_config` records the official source URL, installed package
versions, and the R/Python executables used. Missing packages or unusable APIs
stop before fitting, but version differences do not. Main experiment version 6
prevents older adapter results from being silently reused.

## Main synthetic experiment

The main experiment has a dedicated standalone generator. Running it once
materializes one graph and one shared set of datasets for every replicate in
the grid

```text
p = 10, 20, 50
e = 1, 4
10 replicates
```

Here `e` is an edge multiplier: the target expected number of directed edges
is `e * p`. The existing ordered Bernoulli DAG generator has `p * (p - 1) / 2`
possible forward edges, so the driver uses edge probability
`min(1, 2 * e / (p - 1))`. Every instance contains one observational
environment with 1000 samples and five stochastic hard-intervention
environments with 200 samples each, for 2000 samples total.

Generate all 60 required instances before running any method:

```bash
python3 experiments/generate_main_experiment_data.py
```

The generator accepts the installed NumPy version and records it in the
manifest for provenance. Different NumPy versions can produce different draws
from the same seed, which is why the generated arrays themselves are persisted,
checksummed, committed, and shared rather than regenerated on Quest.

The files are written under `data/main_experiment/` as one lossless NumPy
archive per `(p,e,replicate)`, plus `manifest.json`. A complete matching suite
is validated and reused when the command is run again. An invalid or different
existing suite is never replaced unless `--overwrite` is explicit. To perform
a read-only full validation:

```bash
python3 experiments/generate_main_experiment_data.py --verify-only
```

Generate this suite locally once, then commit `data/main_experiment/` and upload
it to GitHub. The Quest jobs never generate or modify datasets.

The fitting driver only loads and validates these persisted files; it has no
data-generation fallback. Every result records both the sample digest and an
instance digest covering samples, DAG, intervention targets, and identifying
metadata.

Unknown-target methods are PS-MIP, DCDI-G, UT-IGSP, and GnIES. GnIES uses the
authors' greedy search, not its faster rank approximation. `ps_mip_oracle`
fixes the existing PS-MIP intervention indicators to the generated target
matrix, while `gies_oracle` supplies that same matrix to the requested Python
`gies` package. The two oracle rows are graph-recovery references; their
target-recovery fields are left empty.

The main comparison records compact, predeclared tuning paths:

- PS-MIP unknown: tied graph/target multipliers `{0.5, 1, 2}` around
  `log(N)/N`;
- PS-MIP oracle: graph multipliers `{0.5, 1, 2}`, with its irrelevant fixed
  target penalty held constant;
- GnIES and GIES: `{0.25, 0.5, 1, 2, 4}` times their native
  `0.5*log(N)` BIC penalty;
- UT-IGSP: tied `alpha=alpha_inv` over
  `{1e-5, 1e-4, 1e-3, 1e-2, 0.05}`; and
- DCDI-G: a five-point cross around the reference `(lambda, lambda_R) =
  (0.1, 0.001)` setting.

These are compact predeclared candidate paths. Earlier exploratory smoke checks
informed their scales, but GnIES has since been changed from its rank
approximation to the authors' greedy search and GIES now uses the requested
`juangamella/gies` package. Therefore, rerun the documented all-settings smoke
command with the current implementation APIs before treating the paths as
smoke-validated. DCDI did not finish inside the earlier short smoke cap, so its
compact path remains centered on the reference configuration rather than tuned
using graph truth.

The TDP-FDP figure averages each setting over replicates. The potential table
then selects the smallest true `d_cpdag` separately for each replicate and
method, with the predeclared setting order breaking ties. This is deliberately
post-hoc oracle tuning and is labeled `posthoc_oracle_potential` in both the
selected-row audit and summary; it should not be interpreted as a deployable
hyperparameter-selection rule.
For DCDI's two-dimensional cross, the plot uses a solid graph-penalty branch
and a dotted target-penalty branch through the reference center.

DCDI receives the same persisted rows as every other method, then applies its
official implementation's own preprocessing: normalization from the training
split and the upstream randomized 80/20 train/validation split. These choices
are recorded in `method_config`; no graph or intervention truth is staged.
The five DCDI penalty settings and the width/`mu_init` values are declared
experiment configurations passed to the authors' CLI; they are not presented
as the authors' full tuning protocol.

PS-MIP screens the observational sample with
`alpha_screen = 5*sqrt(log(p)/n_screen)`, where `n_screen` is exactly the number
of observational rows used by graphical lasso. Parent sets are unrestricted
within that screened superstructure. The screen can omit true adjacencies, so
PS-MIP remains exact only conditional on the screen; `screen_alpha` and
`n_screen` are recorded with every setting, and successful PS-MIP rows also
record the realized `screen_edges` and `screen_parent_sets`. Across the 60
default seeded instances, the largest preflight screen contains 674,275 parent
sets, below the 1.1-million safety guard.
In the seeded `p=10,e=4` preflight cell, the screen had 27 edges and 528 parent
sets: 25 true adjacencies, two false adjacencies, and 13 of the 38 true
adjacencies omitted. This is why the screen statistics remain part of the
reported PS-MIP result rather than an invisible preprocessing detail.

Before launching experiments, run the repository setup check. It loads the
smallest official instance (`p=10`, `e=1`, replicate 1), checks the required
Python packages/APIs, the clean official DCDI checkout, DCDI's R reporting
dependencies (`pcalg` and `SID`), and the Gurobi setup, then runs one primary
setting for all six methods. It prints the
estimated graphs, targets, runtimes, objectives, `d_cpdag`, FDP, and TDP. It
uses cleaned temporary directories and saves no result or method artifact:

```bash
python3 experiments/test_main_experiment_setup.py
```

This is a real end-to-end fit, not a mocked unit test. DCDI can take much
longer than the other five methods; the default per-method limit is one hour
and can be changed with `--time-limit`. The setup check runs the five fast
methods first, runs DCDI last, and prints a one-minute heartbeat while DCDI is
working. A method failure is printed, the
remaining methods are still attempted, and the script exits nonzero.

For a checkpointed `p=10, e=1` experiment smoke run, use the main driver. It
loads the same persisted replicate with all 1000 observational and 200 rows in
each interventional environment, then runs one primary setting per method:

```bash
python3 experiments/run_main_experiment.py \
  --smoke-test \
  --output experiment_results/main_experiment/smoke_grid.csv
```

The official DCDI optimizer may not converge inside that deliberately short
limit; the driver records a clear error row and keeps the other method results.
A recorded method failure makes the command exit nonzero while preserving the
checkpoint CSV.
A full DCDI fit is not approximated by lowering its iteration cap because the
upstream code writes its final DAG only after convergence. Pass a larger
`--time-limit` to extend this same small smoke instance when validating DCDI on
faster hardware.

The non-DCDI tuning paths can be checked in seconds on the same smoke
instance:

```bash
python3 experiments/run_main_experiment.py \
  --smoke-test --all-settings \
  --methods ps_mip_unknown utigsp_unknown gnies_unknown \
            ps_mip_oracle gies_oracle \
  --output experiment_results/main_experiment/grid_smoke.csv
```

The serial complete-grid command is:

```bash
python3 experiments/run_main_experiment.py \
  --num-replicates 10 \
  --output experiment_results/main_experiment/results_grid.csv
```

The output is a checkpointed long-form CSV keyed by `p`, `e`, `replicate`,
`method`, and `setting_id`. The complete grid contains 1,560 rows, including
300 DCDI fits, so the serial command is primarily for debugging. The Slurm
array has exactly 10 tasks. Task `r` runs the entire 156-setting grid for
replicate `r`: 3 node counts, 2 graph densities, and 26 method/tuning settings.
Each fit has a six-hour cap and each exact metric evaluation has a one-hour cap
by default; DCDI is expected to dominate the total compute budget.

After cloning or pulling the repository on Quest, submit from the repository
root. The committed `data/main_experiment/manifest.json` must already exist:

```bash
mkdir -p experiments/quest_jobs/outlog experiment_results/main_experiment/parts
sbatch experiments/quest_jobs/main_experiment_array.sh
```

For a nondefault shared storage location or master seed, export the same values
before submitting every job:

```bash
export MAIN_DATA_ROOT=/shared/path/micodag/main_experiment
export MAIN_MASTER_SEED=20260823
```

Each task owns one checkpoint file, `parts/replicate_NNN.csv`. If a 24-hour
allocation ends before a replicate finishes, submit the same 10-task array
again; matching successful and failed rows are skipped so a recurring timeout
cannot starve all later settings, and each replicate continues from its
checkpoint. Set `MAIN_RETRY_FAILURES=1` only when you want the next submission
to retry matching failed rows. Increasing a time limit changes the checkpoint
identity and also retries the affected rows automatically. A fragment that
still contains a failed row exits nonzero even while it advances past that
checkpoint, so scheduler status never labels an incomplete replicate as clean.
The data root, fit
limits, Python executable, Rscript
executable, and environment can be overridden with `MAIN_DATA_ROOT`,
`MAIN_MASTER_SEED`, `MAIN_FIT_TIME_LIMIT`, `MAIN_METRIC_TIME_LIMIT`,
`MAIN_PYTHON`, `RSCRIPT`, `MAIN_R_MODULE`, `MAIN_RETRY_FAILURES`,
`MAIN_PREFLIGHT_TIMEOUT`, and `MAIN_CONDA_ENV`. If Quest makes R
available through a module, set `MAIN_R_MODULE`; otherwise set `RSCRIPT` to an
absolute executable whose basename is `Rscript`. For DCDI, the job checks that
`pcalg` and `SID` are available before fitting, without enforcing versions. DCDI uses
the same active Python interpreter as the main runner. The selected manifest
and fitting jobs must use the same master seed. Completed metric rows, including
time-limited MIP incumbents, are skipped when their declared budgets match.

To aggregate manually after the fragments finish:

```bash
python3 analysis/aggregate_main_experiment.py \
  --input experiment_results/main_experiment/parts \
  --output-dir experiment_results/main_experiment/summary \
  --data-root data/main_experiment \
  --seed 20260823 \
  --require-complete
```

This writes `main_tdp_fdp.{png,pdf}`, `curve_summary.csv`, the raw
best-setting selections, `best_dcpdag_summary.{csv,tex}`, and completeness/
shared-data diagnostics. Aggregation also checks every fragment against the
selected dataset manifest, so results from different generated suites cannot
be mixed accidentally. Its `--require-complete` check stops before creating a
final plot or table unless all 10 replicate files are complete.

`--time-limit` covers Gurobi's solve and the external baseline fits. PS-MIP's
screening and local-score precomputation happen before Gurobi and are included
in `fit_seconds`, but not in Gurobi's solver limit. To reuse an existing job
array, use the supplied script or run a single method/cell/replicate with
`--replicate`, `--p-values`, `--edge-multipliers`, and `--methods`.

## Running individual methods

Every active standalone method command reads the validated suite under
`data/main_experiment/`; none reads the archived legacy fixtures. Select an
instance with `--p`, `--edge-multiplier` (or `--e`), and `--replicate`. For
example, this runs Equation (4.4) on the first `p=20`, `e=1` instance:

```bash
python3 MIP.py --p 20 --e 1 --replicate 1 --time-limit 1000
```

When penalties are omitted, the standalone MIP uses the primary
`log(N)/N` graph and target penalties. Its default screen is the main
experiment rule `5*sqrt(log(p)/n_screen)`; change only the constant with
`--screen-constant`. Direct method commands default to the main experiment's
one-hour solver/external-fit and metric limits; PS-MIP also defaults to one
solver thread. As in the main runner, PS-MIP's screening and local-score
precomputation are included in reported runtime but occur before Gurobi's
`--time-limit`. Use `--time-limit`, `--metric-time-limit`, and `--threads` to
change these controls.

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
python3 MIP_profiled.py --p 20 --e 1 --replicate 1
```

The formulation is exact when its reported local optima lie within the
specified `Gamma` bounds. The script checks this condition rather than silently
using an invalid closed-form score. `--max-parents` can restrict the candidate
parent-set size when the supplied superstructure is dense.

The competing-method adapters use the same instance selectors and call the
official implementations described above:

```bash
python3 DCDI.py --p 10 --e 1 --replicate 1
python3 UTIGSP.py --p 10 --e 1 --replicate 1
python3 GIES.py --p 10 --e 1 --replicate 1
```

`DCDI.py` uses cleaned temporary storage unless `--artifact-dir` is supplied.
`GIES.py` is the known-target oracle and therefore reads the stored target
matrix for the selected instance. GnIES is called directly from its authors'
Python package by the main runner and the no-save setup check; there is no
local GnIES algorithm implementation.

## Legacy archive

The superseded naive MIP, standalone GNIES runner, old penalty-sweep jobs,
notebooks, static datasets, Sachs data, and all prior result artifacts are
preserved below `archive/legacy_pipeline/`. They are excluded from the active
pipeline but remain available for provenance and reproduction. See
`archive/README.md` before running an archived script; historical relative
paths and dependencies are intentionally preserved rather than modernized.

## Data generators

`data/DataGeneration.py` is the low-level sampling library and legacy-layout
CLI. It reproduces the procedure in the archived `DataGeneration.R` reference:

- ordered random DAGs with edge probability `2 / p`;
- edge weights sampled from `{-0.8, -0.6, 0.6, 0.8}`;
- observational noise variances sampled from `{1, 2, 4}`;
- hard interventions that remove incoming edges at each target;
- five interventional environments by default; and
- ten repeated datasets per environment.

Those are the legacy file-generation defaults. They are not the main experiment
design. `experiments/generate_main_experiment_data.py` is the only generator for
the main experiment: it uses the low-level functions with the `e*p` density
translation, 1000/200 sample sizes, exact per-instance seeds, and the persisted
manifest contract described above.

Review the low-level legacy-layout generator options only when reproducing that
older layout:

```bash
python3 data/DataGeneration.py --help
```

The low-level legacy-layout CLI refuses to replace existing intended output
files unless `--overwrite` is explicit. The static datasets formerly stored in
`data/` are preserved under `archive/legacy_pipeline/data/`.

## Evaluation

`src/utils.py` contains the shared graph metrics. In particular,
`interventional_cpdag()` constructs an I-CPDAG from a DAG and its
environment-specific intervention targets. `cpdag_distance()` then computes
the entrywise L1 difference between the estimated and true I-CPDAG adjacency
matrices. This project calls that quantity `d_cpdag`; it is intentionally
different from structural Hamming distance (SHD).

## Project Status

The active pipeline and historical material are now separated. The next work
should focus on:

- validating `MIP.py` through small cases with known optima;
- validating the redesigned main experiment at cluster scale;
- adding Sachs-data preprocessing and evaluation;
- validating the tuning paths and post-hoc potential summaries at scale; and
- validating the archived provenance bundle only when an older result must be
  reproduced.
