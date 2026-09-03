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

The standalone MIP commands and both PS-MIP variants in the main experiment use
the same persisted datasets and the same graphical-lasso screening rule,
`sqrt(log(p)/n_screen)`. The competing methods do not use this PS-MIP
screen. Imported optimization functions can still accept a
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
| `DCDI.py` | Adapter for the official DCDI-G perfect known/unknown-target implementations |
| `BaCaDI.py` | Adapter for the official joint linear-Gaussian BaCaDI unknown-target implementation |
| `GIES.py` | Thin adapter for the requested `juangamella/gies` Python package |
| `UTIGSP.py` | Adapter for UT-IGSP from `causaldag` |
| `IGSP.py` | Adapter for known-target IGSP from `causaldag` |
| `external/bacadi/` | Hash-verified minimal snapshot of the official BaCaDI Python package |
| `requirements-bacadi.txt` | Dedicated Quest environment for BaCaDI and the shared experiment code |
| `data/DataGeneration.py` | Python translation of the original R synthetic-data generator |
| `experiments/generate_main_experiment_data.py` | Standalone generator for all persisted main-experiment datasets |
| `experiments/test_main_experiment_setup.py` | No-save, one-instance check of the eight shared-environment methods, with optional BaCaDI inclusion |
| `experiments/run_main_experiment.py` | Main shared-data synthetic experiment driver |
| `experiments/SACHS_ROC_PLAN.md` | Exact-data, Quest, and evaluation plan for the Sachs biological benchmark |
| `experiments/prepare_sachs_data.py` | Checksum-pinned acquisition and validation of the six UT-IGSP Sachs files |
| `experiments/run_sachs_roc.py` | One-setting Sachs fitter for three unknown-target methods, contextual UT-IGSP, and three oracle references |
| `analysis/aggregate_sachs_roc.py` | Strict adjacency-based aggregation of Sachs directed and skeleton ROC points |
| `analysis/plot_sachs_roc.py` | Paper-compatible count plots and conventional normalized Sachs ROC plots |
| `analysis/plot_main_experiment_results.py` | Zero-argument plot of all currently available main-experiment results |
| `analysis/aggregate_main_experiment.py` | Main graph and target recovery paths and best-`d_cpdag` potential tables |
| `experiments/quest_jobs/main_experiment_<method>.sh` | Nine method-specific Slurm arrays, including the cell-wise BaCaDI array |
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

Install the tested shared dependencies into the Python environment used for
the eight non-BaCaDI methods:

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
# Quest only: make Rscript available first.
module load R/4.4.0
Rscript --vanilla -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install("pcalg", ask=FALSE, update=FALSE)'
Rscript --vanilla -e 'install.packages("SID", repos="https://cloud.r-project.org")'
```

Quest stores user-installed R packages separately for each R major/minor
version. The two DCDI submission scripts therefore load `R/4.4.0` explicitly;
install `pcalg` and `SID` once while that same module is loaded. The seven
non-DCDI method arrays do not load or use R.

The `pgmpy` pin is required when using Python 3.9 because current `pgmpy`
releases require Python 3.10 or newer.

Quest's Python 3.9 module is linked against the older system OpenSSL 1.0.2.
`urllib3` 2.x refuses that SSL runtime, so the tested requirements retain the
compatible 1.26 line. If an existing Quest environment reports this specific
OpenSSL error, repair that environment with:

```bash
python3 -m pip install --user 'urllib3==1.26.20'
```

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

The adapter verifies the exact source manifest, confirms that required Python
and R imports work, records their actual versions, and rejects modified,
missing, or unexpected source files before every fit. It then invokes the
authors' `main.py` directly. No DCDI model, objective, optimizer, or reporting function is copied
or changed. For current Matplotlib releases, a child-process-only compatibility
module translates the removed plotting keyword `padding` to its modern
`pad_inches` spelling. This affects only saved plot margins; the official DCDI
checkout and estimation code remain untouched.
The upstream report requires the R packages `pcalg` and `SID`.

Only file-format glue remains local. Unknown-target DCDI receives sample values,
environment labels, an all-zero DAG placeholder, and blank target rows, so
simulation truth never enters that fitting objective. Known-target DCDI receives
the same samples plus the true intervention masks, as required by the authors'
`intervention-knowledge=known` path, while the DAG remains hidden. The adapter
validates the author-produced graph and, for unknown mode, learned targets.

### BaCaDI installation

`BaCaDI.py` uses the official BaCaDI source pinned at commit
`b2c27d5fc560f7e0ec2aa282cd2e0a0e6e8637ea`. The complete upstream `bacadi`
Python package and its MIT license are committed under `external/bacadi`;
paper results, plotting code, baselines, and simulators are omitted. The
adapter validates the source commit and hashes the complete vendored Python
tree before every fit. Runtime compatibility aliases are installed in memory
for APIs removed after the authors' JAX release; the vendored source files are
not changed.

BaCaDI runs in a dedicated Quest environment because its tested modern JAX
runtime should not be mixed into the shared environment. Create that
environment once from the repository root:

```bash
module purge all
module load python-miniconda3
conda create --name bacadi39 python=3.9 -y
source activate bacadi39
python3 -m pip install -r requirements-bacadi.txt
python3 -c 'from BaCaDI import validate_runtime; print(validate_runtime())'
```

`requirements-bacadi.txt` includes the shared experiment dependencies and pins
JAX, `jaxlib`, `python-igraph`, and `tqdm` for the adapter. The provenance
manifest `external/bacadi/VENDORED_SOURCE.json` records the official URL,
commit, selected source tree, and its SHA-256 digest. The adapter records the
actual interpreter, JAX backend, installed versions, and compatibility layer in
each result's `method_config`.

### Official competing implementations

The competing algorithms are not implemented in this repository:

| Method | Code used by the main experiment |
| --- | --- |
| DCDI-G | Hash-verified minimal snapshot of the authors' [`slachapelle/dcdi`](https://github.com/slachapelle/dcdi) source at commit `594d328eae7795785e0d1a1138945e28a4fec037`, executed through its official `main.py` |
| UT-IGSP | [`causaldag`](https://github.com/uhlerlab/causaldag) / `graphical-model-learning` author package API (tested with 0.1a163 / 0.1a8) |
| BaCaDI | Hash-verified minimal snapshot of the authors' [`haeggee/bacadi`](https://github.com/haeggee/bacadi) source at commit `b2c27d5fc560f7e0ec2aa282cd2e0a0e6e8637ea`, using `bacadi.inference.bacadi_joint.BaCaDIJoint` |
| IGSP oracle | The known-target `igsp` API from the same `causaldag` / `graphical-model-learning` author packages |
| GnIES | [First-author `gnies`](https://github.com/juangamella/gnies) package using its full `approach="greedy"` search (tested with 0.3.3) |
| GIES oracle | Olga Kolotuhina and Juan L. Gamella's requested [`juangamella/gies`](https://github.com/juangamella/gies) Python package (tested with 0.0.3) |

`DCDI.py`, `BaCaDI.py`, `UTIGSP.py`, `IGSP.py`, and `GIES.py` validate data and
translate input or output formats. `BaCaDI.py` additionally applies
observational-only standardization and selects the highest-weight acyclic joint
particle to obtain one coherent graph-and-target point estimate for the shared
metrics. PS-MIP is the proposed method in this project, so its local
implementation is intentionally outside this competitor-source rule. Every
result's `method_config` records the official source URL, installed package
versions, and the R/Python executables used. Missing packages or unusable APIs
stop before fitting, but version differences do not. Main experiment version 10
prevents older adapter results from being silently reused.

## Main synthetic experiment

The main experiment has a dedicated standalone generator. Running it once
materializes one graph and one shared set of datasets for every replicate in
the grid

```text
p = 10, 20, 30
e = 1, 2
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

Unknown-target methods are PS-MIP, DCDI-G, UT-IGSP, BaCaDI, and GnIES. GnIES
uses the authors' greedy search, not its faster rank approximation. BaCaDI
receives only the stacked samples and environment labels. `ps_mip_oracle`,
`dcdi_g_oracle`, `igsp_oracle`, and `gies_oracle` each receive the generated
target matrix. DCDI and IGSP oracle use their authors' official
known-intervention pathways. These four oracle rows are graph-recovery
references; their target-recovery fields are left empty.

The main comparison records compact, predeclared tuning paths:

- PS-MIP unknown: tied graph/target multipliers `{0.5, 1, 2}` around
  `log(N)/N`;
- PS-MIP oracle: graph multipliers `{0.5, 1, 2}`, with its irrelevant fixed
  target penalty held constant;
- GnIES and GIES: `{0.25, 0.5, 1, 2, 4}` times their native
  `0.5*log(N)` BIC penalty;
- UT-IGSP and IGSP oracle: tied `alpha=alpha_inv` over
  `{1e-5, 1e-4, 1e-3, 1e-2, 0.05}`;
- BaCaDI: one fixed `official_repo_default_joint` setting using the joint
  linear-Gaussian model, 20 particles, 3000 SVGD steps, 128 gradient Monte
  Carlo samples, 32 acyclicity Monte Carlo samples, an Erdős–Rényi prior
  with two expected edges per node, and target regularization one;
- DCDI-G: a five-point cross around the reference `(lambda, lambda_R) =
  (0.1, 0.001)` setting; and
- DCDI-G oracle: graph penalties `{0.01, 0.1, 1}`, with no target penalty
  because the intervention masks are supplied.

BaCaDI is evaluated under a documented model mismatch. The generated SEMs use
node-specific observational noise variances and stochastic hard interventions
whose target variances vary by node and environment. After standardizing all
environments using observational moments, the adapter fits BaCaDI's joint
linear-Gaussian likelihood with fixed unit observational and intervention
noise variances and learned intervention means. The BaCaDI comparison therefore
measures the fixed declared model under misspecification; it is not a correctly
specified oracle comparison.

The particle, optimization, kernel, graph-prior, and target-regularization
settings follow the pinned repository parser and launcher defaults; the two
fixed likelihood-noise values are the standardization adaptation described
above. The paper's reported configurations were selected through a
hyperparameter search, so `official_repo_default_joint` is not presented as a
reproduction of the paper's tuned BaCaDI configuration.

The fixed BaCaDI configuration is computationally intensive and is designated
Quest-only for the full experiment. It is omitted from the normal local setup
check and from the fast smoke command below. The adapter, dedicated environment,
and checkpointed Quest jobs are complete, but no numerical conclusion should
be drawn until BaCaDI result fragments are available.

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
and a dotted target-penalty branch through the reference center. BaCaDI
contributes one point rather than a tuning path.

DCDI receives the same persisted rows as every other method, then applies its
official implementation's own preprocessing: normalization from the training
split and the upstream randomized 80/20 train/validation split. These choices
are recorded in `method_config`. The graph is never staged; intervention truth
is staged only for the explicitly labeled DCDI oracle. The DCDI penalty
settings and the width/`mu_init` values are declared
experiment configurations passed to the authors' CLI; they are not presented
as the authors' full tuning protocol.

PS-MIP screens the observational sample with
`alpha_screen = sqrt(log(p)/n_screen)`, where `n_screen` is exactly the number
of observational rows used by graphical lasso. Parent sets are unrestricted
within that screened superstructure. The screen can omit true adjacencies, so
PS-MIP remains exact only conditional on the screen; `screen_alpha` and
`n_screen` are recorded with every setting, and successful PS-MIP rows also
record the realized `screen_edges` and `screen_parent_sets`. The active grid
uses the sparse and moderately dense edge multipliers `e=1` and `e=2`. Across
the 60 committed instances, the largest screen has 3,064,768 parent sets, so
the safety guard is 4 million. Screen diagnostics are reported with the
results. The largest PS-MIP cells may spend substantial time profiling local
scores before Gurobi starts.

Before launching experiments, run the repository setup check. It loads the
smallest official instance (`p=10`, `e=1`, replicate 1), checks the required
Python packages/APIs, the hash-verified DCDI author snapshot, DCDI's R reporting
dependencies (`pcalg` and `SID`), and the Gurobi setup, then runs one primary
setting for the eight shared-environment methods. It prints the
estimated graphs, targets, runtimes, objectives, `d_cpdag`, FDP, and TDP. It
uses cleaned temporary directories and saves no result or method artifact:

```bash
python3 experiments/test_main_experiment_setup.py
```

This is a real end-to-end fit, not a mocked unit test. DCDI can take much
longer than the other methods; the default per-method limit is one hour
and can be changed with `--time-limit`. The setup check runs the six fast
methods first, runs both DCDI modes last, and prints a one-minute heartbeat
while DCDI is working. A method failure is printed, the
remaining methods are still attempted, and the script exits nonzero.
BaCaDI is excluded by default because one complete fit can use the full
one-hour cap. The `--include-bacadi` flag is available for a deliberate
Quest-side end-to-end check in an environment that contains both the BaCaDI
and shared dependencies.

For a checkpointed `p=10, e=1` experiment smoke run, use the main driver. It
loads the same persisted replicate with all 1000 observational and 200 rows in
each interventional environment, then runs one primary setting per method:

```bash
python3 experiments/run_main_experiment.py \
  --smoke-test \
  --methods ps_mip_unknown dcdi_g_unknown utigsp_unknown gnies_unknown \
            ps_mip_oracle dcdi_g_oracle igsp_oracle gies_oracle \
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

The fast non-DCDI tuning paths can be checked on the same smoke instance:

```bash
python3 experiments/run_main_experiment.py \
  --smoke-test --all-settings \
  --methods ps_mip_unknown utigsp_unknown gnies_unknown \
            ps_mip_oracle igsp_oracle gies_oracle \
  --output experiment_results/main_experiment/grid_smoke.csv
```

The serial complete-grid command is:

```bash
python3 experiments/run_main_experiment.py \
  --num-replicates 10 \
  --output experiment_results/main_experiment/results_grid.csv
```

The output is a checkpointed long-form CSV keyed by `p`, `e`, `replicate`,
`method`, and `setting_id`. The complete nine-method grid contains 2,100 rows:
the prior 2,040 rows plus one BaCaDI setting for each of the 60 instances. The
serial command is primarily for schema and checkpoint debugging; the complete
BaCaDI grid should be run through Quest.

The eight existing method arrays each launch ten tasks. Task `r` runs one
method on replicate `r` across all three node counts and both graph densities.
BaCaDI instead launches 60 tasks, one for each `(p,e,replicate)` cell, because
placing all six cells in one allocation would make recovery from a slow cell
unnecessarily coarse.

| Method array | Task unit | Fits per task | Partition | Wall time | Memory |
| --- | --- | ---: | --- | ---: | ---: |
| PS-MIP unknown | one replicate | 18 | `normal` | 24 hours | 16 GB |
| DCDI-G unknown | one replicate | 30 | `normal` | 36 hours | 16 GB |
| UT-IGSP unknown | one replicate | 30 | `short` | 1 hour | 16 GB |
| BaCaDI unknown | one design cell | 1 | `normal` | 3 hours | 32 GB |
| GnIES unknown | one replicate | 30 | `normal` | 12 hours | 16 GB |
| PS-MIP oracle | one replicate | 18 | `normal` | 24 hours | 16 GB |
| DCDI-G oracle | one replicate | 18 | `normal` | 24 hours | 16 GB |
| IGSP oracle | one replicate | 30 | `normal` | 12 hours | 16 GB |
| GIES oracle | one replicate | 30 | `normal` | 12 hours | 16 GB |

Every array uses a one-hour cap for each fit and a separate one-hour cap for
exact metric evaluation. The BaCaDI array requests three hours per one-fit task
to leave time for environment validation, JAX compilation, data loading,
metric evaluation, and checkpoint writing. Every task requests eight CPUs.
These requests are budgets rather than runtime claims, so the checkpoint files
remain the recovery mechanism when an allocation ends before a cell is
complete.

After cloning or pulling the repository on Quest, submit from the repository
root. The committed `data/main_experiment/manifest.json` must already exist:

```bash
sbatch experiments/quest_jobs/main_experiment_ps_mip_unknown.sh
sbatch experiments/quest_jobs/main_experiment_dcdi_g_unknown.sh
sbatch experiments/quest_jobs/main_experiment_utigsp_unknown.sh
sbatch experiments/quest_jobs/main_experiment_bacadi_unknown.sh
sbatch experiments/quest_jobs/main_experiment_gnies_unknown.sh
sbatch experiments/quest_jobs/main_experiment_ps_mip_oracle.sh
sbatch experiments/quest_jobs/main_experiment_dcdi_g_oracle.sh
sbatch experiments/quest_jobs/main_experiment_igsp_oracle.sh
sbatch experiments/quest_jobs/main_experiment_gies_oracle.sh
```

Submitting all nine scripts creates 140 array tasks: 80 replicate-level tasks
for the existing methods and 60 cell-level BaCaDI tasks. Because each array is
throttled with `%10`, at most 90 tasks are eligible to run concurrently; Quest
may keep some pending based on available resources and fairshare. Existing
methods write `parts/<method>/replicate_NNN.csv`. BaCaDI writes
`parts/bacadi_unknown/p_P_e_E_replicate_NNN.csv`, so each slow design cell has
an independent checkpoint. If an allocation ends before a task finishes,
submit only that method's script again. Matching successful and failed rows are
skipped so a recurring timeout cannot starve later settings, and each task
continues from its checkpoint. A fragment that still contains a failed row
exits nonzero. Completed metric rows, including time-limited MIP incumbents,
are skipped when the same script is submitted again.

To plot whatever results are currently available, run this zero-argument
script from the repository root:

```bash
python3 analysis/plot_main_experiment_results.py
```

It scans `experiment_results/main_experiment/parts/` automatically. The graph
plot contains every non-DCDI method for which structural metrics are currently
available, so it does not wait for BaCaDI. The target plot similarly uses the
available full target-matrix methods among PS-MIP, UT-IGSP, and BaCaDI. A method
is added to a cell once at least one replicate contains its complete setting
path; when several methods are present, the plot uses their common complete
replicates. Missing BaCaDI fragments therefore do not suppress the existing
PS-MIP and UT-IGSP comparison. The PNG and PDF files are written to
`experiment_results/main_experiment/summary/`.

For strict final aggregation that requires every planned row, run:

```bash
python3 analysis/aggregate_main_experiment.py \
  --input experiment_results/main_experiment/parts \
  --output-dir experiment_results/main_experiment/summary \
  --data-root data/main_experiment \
  --seed 20260823 \
  --require-complete
```

This writes `main_tdp_fdp.{png,pdf}`, `target_tpr_fpr.{png,pdf}`, their
setting-level summaries, the raw best-setting selections,
`best_dcpdag_summary.{csv,tex}`, and completeness/shared-data diagnostics. The
target comparison uses common complete replicates for the target-matrix
methods available in each cell. Aggregation also checks every fragment against
the selected dataset manifest, so results from different generated suites
cannot be mixed accidentally. Its `--require-complete` check stops before
creating a final graph plot or truth-selected table unless all 2,100 planned
rows represented by the 140 task-owned checkpoint fragments are complete.

`--time-limit` covers Gurobi's solve and the external baseline fits. PS-MIP's
screening and local-score precomputation happen before Gurobi and are included
in `fit_seconds`, but not in Gurobi's solver limit. To reuse an existing job
array, use the supplied method script or run a single method/cell/replicate with
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
experiment rule `sqrt(log(p)/n_screen)`; change only the constant with
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
python3 IGSP.py --p 10 --e 1 --replicate 1
python3 GIES.py --p 10 --e 1 --replicate 1
```

From the dedicated `bacadi39` environment, the corresponding standalone
BaCaDI command is:

```bash
python3 BaCaDI.py --p 10 --e 1 --replicate 1
```

This command is useful for Quest-side validation of one persisted instance; it
is not the recommended way to launch the complete BaCaDI grid.

`DCDI.py` uses cleaned temporary storage unless `--artifact-dir` is supplied.
`IGSP.py` and `GIES.py` are known-target oracles and therefore read the stored
target matrix for the selected instance. GnIES is called directly from its
authors' Python package by the main runner and the no-save setup check; there
is no local GnIES algorithm implementation. BaCaDI uses the hash-verified
vendored author package through `BaCaDI.py` and does not receive the generated
target matrix.

## Legacy archive

The superseded naive MIP, standalone GNIES runner, old penalty-sweep jobs,
notebooks, static datasets, Sachs data, and all prior result artifacts are
preserved below `archive/legacy_pipeline/`. They are excluded from the active
pipeline but remain available for provenance and reproduction. See
`archive/README.md` before running an archived script; historical relative
paths and dependencies are intentionally preserved rather than modernized.

## Sachs biological benchmark

The active real-data pipeline uses the exact six natural-log files, node
ordering, five intervention contexts, and 18-edge accepted DAG released for
the UT-IGSP Sachs analysis.  It does not reuse the archived all-condition
dataset or its mislabeled false-positive-discovery plot.  See
[`experiments/SACHS_ROC_PLAN.md`](experiments/SACHS_ROC_PLAN.md) for the pinned
source, the 5,846-row split, fixed method paths, Quest commands, and the strict
post-run plot contract.

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
- running the Quest-only BaCaDI array before reporting its comparison;
- validating the tuning paths and post-hoc potential summaries at scale; and
- validating the archived provenance bundle only when an older result must be
  reproduced.
