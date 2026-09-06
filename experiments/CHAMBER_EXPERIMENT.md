# Causal Chambers: scm_4 unknown-target / oracle comparison

Use `experiments/run_chamber_experiment.py` for preparation, fitting, resumption,
auditing and plotting. Five active Quest jobs follow the synthetic experiments'
target-information policy:

| Method key | Plot label | Information supplied | Settings |
|---|---|---|---:|
| `ps_mip_unknown` | PS-MIP: unknown targets | No intervention targets | 17 |
| `gies_oracle` | GIES (oracle) | Complete documented intervention lists | 15 |
| `utigsp_unknown` | UT-IGSP: unknown targets | No intervention targets | 8 |
| `igsp_oracle` | IGSP (oracle) | Complete documented intervention lists | 8 |
| `gnies_unknown` | GnIES: unknown targets | No target union or initial target set | 15 |

"Unknown" means the documented targets are deliberately withheld from the
algorithm, not unavailable to the experimenter. Environment groupings remain
known and the reference remains identified as observational. "Oracle" means the
documented intervention assignments are supplied as complete lists, not that a
method receives the reference graph. The physical reference does not guarantee
that every omitted physical effect is absent.

## Unchanged data and settings

Use `lt_camera_v1/scm_4`, a controlled physical/hybrid benchmark, not biological
validation. The programmed actuator chain is red -> green -> blue. Retain all
40,000 observations: 10,000 each in reference, red, green and blue environments,
in that order. Variables are red, green, blue, current, ir_1, vis_1, ir_2, vis_2,
ir_3, vis_3.

Every method receives the same reference-derived affine transformation:
subtract the reference mean and divide by its standard deviation, applying
the same constants to every environment. No rank transformation, filtering,
subsampling or screening. PS-MIP retains 45 candidate pairs and 5,120 parent
sets. Existing numerical bounds and internal mean handling are unchanged.

There are **63 settings**. PS-MIP uses `2**(k/2)*log(N)/N`, k=-8,...,8, with
target penalty `16*log(N)/N` per intervention environment. GnIES/GIES use
`2**k * 0.5*log(N)`, k=-4,...,10. Both IGSP methods use Gaussian invariance
tests, invariance level 1e-5, CI levels 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
0.5, depth four and ten starts. N=40,000, seed=20260901, eight solver threads,
one numerical-library thread.

Consistency with synthetic experiments refers to **target information**. The
chamber grids and fixed invariance level remain unchanged; preprocessing and
every tuning choice need not be identical across datasets.

## Setup

Upload the current repository, prepared `data/causal_chambers/lt_camera_v1/`,
and historical results if reusing them. Keep
`archive/causal_chambers/2026-09-06/known_target_quest_jobs/` for historical
code-identity checks. The old transfer bundle predates this comparison.

```bash
module purge all
module load python-miniconda3
source activate python39
module load gurobi
python3 -B experiments/run_chamber_experiment.py --dry-run
```

Use a compatible environment from `requirements-chambers.txt`. GnIES must
accept `center` and `known_targets`; PS-MIP needs a usable Gurobi license.
The dry run checks all 63 settings, data and APIs without fitting or creating
results. Use `--prepare` only if the numeric dataset is not already prepared.
Both configurations' numeric source files remain necessary for provenance,
but only scm_4 is fitted; no image archive is needed.

## Recommended: reuse unchanged fits and run two methods

The earlier 97-fit run remains at
`experiment_results/causal_chambers/scm4_unscreened_1h_v1/`. Unknown-target
PS-MIP, complete-list GIES and complete-list IGSP have unchanged fitting
conditions. Reuse these **40 fits**, including their nonoptimal/failure statuses.
Run this on Quest in the environment that will run the remaining jobs:

```bash
python3 -B experiments/run_chamber_experiment.py --reuse-from experiment_results/causal_chambers/scm4_unscreened_1h_v1
mkdir -p experiments/quest_jobs/outlog
sbatch experiments/quest_jobs/chamber_experiment_utigsp_unknown.sh
sbatch experiments/quest_jobs/chamber_experiment_gnies_unknown.sh
```

Reuse does not fit anything. It validates the historical run and stores unchanged
results in wrappers containing the complete original row and fitting manifest.
It does not edit source files, replace their recorded runtime, change graphs,
or relabel supplied-target UT-IGSP/GnIES as unknown-target results. Those two
methods require **23 new fits** (8 + 15). The new directory is self-contained
for auditing reused rows, even after transfer.

## Alternatively: run all five methods from scratch

If not reusing results, submit all five jobs from the repository root:

```bash
mkdir -p experiments/quest_jobs/outlog
sbatch experiments/quest_jobs/chamber_experiment_ps_mip_unknown.sh
sbatch experiments/quest_jobs/chamber_experiment_gies_oracle.sh
sbatch experiments/quest_jobs/chamber_experiment_utigsp_unknown.sh
sbatch experiments/quest_jobs/chamber_experiment_igsp_oracle.sh
sbatch experiments/quest_jobs/chamber_experiment_gnies_unknown.sh
```

Each array task is one setting, not a replicate. Every fit has a **one-hour
algorithm limit**. Slurm uses `short` and **01:10:00 per task**, not per whole
array, allowing startup and saving. The emergency watchdog is 3,720 seconds.
Each task requests eight CPUs and 16 GB; each array permits ten simultaneous
tasks. PS-MIP retains feasible incumbents at its native limit; baselines record
a timeout if no graph is returned.

For one setting or sequential paths:

```bash
python3 -B experiments/run_chamber_experiment.py --methods gnies_unknown --setting 1
python3 -B experiments/run_chamber_experiment.py --methods utigsp_unknown gnies_unknown
```

Omit `--methods` for all five paths. Successful, nonoptimal, failed and reused
results are preserved, not automatically retried. Keep code, packages and
settings fixed while fitting/resuming. Use a separate `--output-root` for a
different runtime or changed settings; never edit the frozen manifest. Run reuse
on the fitting machine, not locally before transferring a newly frozen run to
a different Quest environment.

## Audit and plot

New results go to
`experiment_results/causal_chambers/scm4_unknown_oracle_1h_v1/`.

```bash
python3 -B experiments/run_chamber_experiment.py --audit
python3 -B experiments/run_chamber_experiment.py --plot
```

These commands may run locally or on Quest: runtime/package differences warn
rather than stop postprocessing. Data, settings, bounds, target constraints,
graph classes, result identities and recomputed metrics remain checked. Reused
rows also validate their original fitting identity and unchanged payload.
Auditing is read-only. Exports record the postprocessing environment separately;
fitting/resumption stay strict. Missing results require `--allow-partial` for a
clearly marked preview, not a completed figure.

`summary/causal_chambers_main.pdf` contains the five-method full-range TP/FP,
0-25 FP and TPR/FPR plots. All 63 settings, including all 17 PS-MIP settings,
are retained. Nonoptimal PS-MIP points keep cross markers. No connecting lines,
envelope, AUC or selective removal of settings. The operating-point CSVs,
`RESULTS.md` and export identities preserve reuse provenance and completion
diagnostics. `causal_chambers.pdf` also includes PS-MIP-only views and model
diagnostics.

Metrics use the same 23-edge reference DAG: 21 physical and two programmed
edges, reported separately. Rate denominators are 23 true edges, 67 directed
nonedges and 22 skeleton nonedges. DAG extensions remain reference-independent;
orientation ambiguity is disclosed. Diagnostics do not justify removing
observations. No manuscript edits are made here.

The historical comparison remains auditable with
`--audit --output-root experiment_results/causal_chambers/scm4_unscreened_1h_v1`.
Its figures are historical, not results for the new unknown-target baseline
conditions. Superseded Quest jobs are archived, not discarded.

Sources: [dataset protocols](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_camera_v1),
[reference graphs](https://github.com/juangamella/causal-chamber-package/tree/main/causalchamber/ground_truth/adjacencies),
[original paper](https://www.nature.com/articles/s42256-024-00964-x).
