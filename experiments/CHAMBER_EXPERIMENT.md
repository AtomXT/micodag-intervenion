# Causal Chambers: scm_4 experiment

Use one entry point, `experiments/run_chamber_experiment.py`, for preparation,
fitting, resumption, validation, and plotting. Quest has one simple job file per
condition, following the existing `main_experiment_*.sh` style. No submission
helper, collection job, or separate preparation/plotting command is needed.

## Data and settings

This is a controlled physical/hybrid benchmark from `lt_camera_v1`, not a
biological dataset. Use **scm_4 only**, whose programmed actuator chain is
red -> green -> blue. Retain all 40,000 observations across reference, red,
green, and blue environments (10,000 each), in that order. Variables are red,
green, blue, current, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3.

All methods receive the same reference-derived affine transformation: subtract
the reference mean and divide by the reference standard deviation, using those
same constants in every environment. No rank transformation, filtering,
subsampling, or screening is used. PS-MIP retains all 45 candidate pairs and
5,120 parent sets. Existing numerical bounds, within-environment mean handling,
and intended-target constraints are unchanged. The reference environment is
observational for every method.

| Method key | Supplied target information | Settings |
|---|---|---:|
| `ps_mip_complete` | Documented targets present; all others absent | 17 |
| `ps_mip_present` | Documented targets present; others learnable | 17 |
| `ps_mip_unknown` | Targets withheld as an information diagnostic | 17 |
| `utigsp_present` | Documented targets supplied; others learnable | 8 |
| `gnies_present` | Documented target union supplied; additional members learnable | 15 |
| `gies_complete` | Documented target lists treated as complete | 15 |
| `igsp_complete` | Documented target lists treated as complete | 8 |

There are 97 fits. PS-MIP uses graph penalties `2**(k/2)*log(N)/N`, k=-8,...,8,
with target penalty `16*log(N)/N` in each intervention environment. GnIES/GIES
use `2**k * 0.5*log(N)`, k=-4,...,10. Both IGSP methods use Gaussian invariance
tests, invariance level 1e-5, CI levels 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
0.5, depth four, and ten starts. N=40,000, seed=20260901, eight solver threads,
one numerical-library thread. The optimizer and synthetic defaults are unchanged.

## Setup and check

Alternatively to syncing the repository, transfer `chamber_scm4_1h.tar.gz`,
extract it with `tar -xzf chamber_scm4_1h.tar.gz`, and enter `chamber_scm4_1h/`.
That self-contained bundle contains the current runner, seven jobs, internal
libraries, instructions, tests, and prepared numeric data; it includes no results
or obsolete launchers.

Upload the current code and the entire `data/causal_chambers/lt_camera_v1/`
directory to Quest. It contains only numeric data and metadata, not the image
collection. Both configurations' numeric source files remain necessary for the
existing provenance checks, but only scm_4 is fitted. Existing prepared data do
not need to be downloaded or transformed again.

Use the same `python39` environment and Gurobi module as the synthetic jobs.
`requirements-chambers.txt` lists the locally tested package versions. GnIES
must accept both `center` and `known_targets`; a usable Gurobi license is needed.

```bash
module purge all
module load python-miniconda3
source activate python39
module load gurobi
python3 -B experiments/run_chamber_experiment.py --dry-run
```

The dry run checks inputs, all 97 settings, and package APIs without fitting or
creating results. If the numeric dataset is not already prepared, run the same
entry point with `--prepare` once; it retrieves numeric archive ranges and
validates the recorded protocol, reference, diagnostics, and numerical bounds.

## Quest jobs: one file per condition

Submit from the repository root. Create the output-log directory first:

```bash
mkdir -p experiments/quest_jobs/outlog
sbatch experiments/quest_jobs/chamber_experiment_ps_mip_complete.sh
sbatch experiments/quest_jobs/chamber_experiment_ps_mip_present.sh
sbatch experiments/quest_jobs/chamber_experiment_ps_mip_unknown.sh
sbatch experiments/quest_jobs/chamber_experiment_utigsp_present.sh
sbatch experiments/quest_jobs/chamber_experiment_gnies_present.sh
sbatch experiments/quest_jobs/chamber_experiment_gies_complete.sh
sbatch experiments/quest_jobs/chamber_experiment_igsp_complete.sh
```

Each array index is one **one-based penalty/test setting**, not a new replicate.
Each fit has **`--time-limit 3600` (one hour)**, as in the synthetic experiments.
PS-MIP uses the native solver limit and saves a feasible incumbent when available;
baselines use an algorithm timer and record a timeout if no graph is returned.
The emergency worker watchdog is 3,720 seconds. Slurm requests **01:10:00** to
allow startup, solver return, and saving; this does not extend the fit budget.
Each task requests eight CPUs and 16 GB. Each method's array permits up to ten
concurrent settings, independently; submit fewer method files at once if desired.

For one setting without Slurm, or to run a method's complete path sequentially:

```bash
python3 -B experiments/run_chamber_experiment.py --methods ps_mip_present --setting 1 --time-limit 3600
python3 -B experiments/run_chamber_experiment.py --methods ps_mip_present --time-limit 3600
```

Omit `--methods` to run all seven paths sequentially. Reissuing the same command
resumes missing settings only; existing successful, nonoptimal, and failed
results are validated and preserved. Concurrent distinct settings are safe;
duplicate dispatch is rejected. Do not modify code, packages, or settings during
a run. Changed settings require a separate `--output-root`.

## Results and plots

New results go to `experiment_results/causal_chambers/scm4_unscreened_1h_v1/`.
They are never mixed with the earlier pilot or five-minute runs. Frozen settings,
input/code/runtime identities, graphs, targets, fit failures, solver bounds and
gaps, runtimes, and per-setting logs are saved there.

After the jobs finish, run in the **same Quest environment** used for fitting:

```bash
python3 -B experiments/run_chamber_experiment.py --audit
python3 -B experiments/run_chamber_experiment.py --plot
```

The final audit/plot requires all 97 settings to be accounted for. Failed fits
remain explicit, not silently retried or replaced. The plot command exports
available validated results when all settings are accounted for, but returns a
nonzero exit code if any failed. Jobs failing before the runner starts may leave
missing fragments; inspect their Slurm logs and rerun only those missing tasks.
Use `--audit --allow-partial` or `--plot --allow-partial` for an explicitly
incomplete check/preview; partial plots live in `summary_preview/`.

`summary/causal_chambers.pdf` contains seven-condition and PS-MIP-only full-range
TP/FP, TPR/FPR, and 0-25 FP views. PNGs, all operating points, and `RESULTS.md`
accompany it. Points remain unconnected; nonoptimal PS-MIP solutions are marked.
Directed and skeleton metrics use the same 23-edge reference DAG (21 physical
and two programmed edges); recovery of these two groups is reported separately.
Rate denominators are 23 true edges, 67 directed nonedges, and 22 skeleton
nonedges. DAG extensions are reference-independent, and ambiguous orientations
are disclosed. Diagnostics describe model mismatch without removing observations.
This experiment is exploratory and is not biological or perfect-model validation.

Bring back the entire result directory for inspection. The manuscript is not
edited. Superseded runners, screening studies, instructions, and the old transfer
bundle are preserved in `archive/causal_chambers/2026-09-06/`; their existing fit
results remain in the separate historical directories. Do not submit the old
five-minute bundle for this one-hour experiment.

Sources: [dataset protocols](https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_camera_v1),
[reference graphs](https://github.com/juangamella/causal-chamber-package/tree/main/causalchamber/ground_truth/adjacencies),
[original paper](https://www.nature.com/articles/s42256-024-00964-x).
