# Sachs biological-data ROC plan

This pipeline evaluates the current paper's three unknown-target methods
(PS-MIP, UT-IGSP*, and GnIES) on the biological benchmark used by Squires,
Wang, and Uhler (UAI 2020).  The original paper's UT-IGSP variant with intended
targets supplied is retained as a contextual curve. A matching partial-target
PS-MIP condition fixes each intended target present while leaving every other
target indicator unknown, so off-target effects can still be learned. Exact
intended-target PS-MIP, IGSP, and GIES are kept as separate oracle references,
matching the synthetic-experiment organization.

Run every command below from the repository root.  In particular, Slurm
resolves each repository-relative `#SBATCH --output` path before the job shell
starts, so submitting these arrays from another directory is not supported.

## 1. Acquire and freeze the exact analysis data

Run this once from a login node, before submitting compute jobs:

```bash
python3 experiments/prepare_sachs_data.py
```

The preparation command downloads the six analysis files from an immutable
commit of the UT-IGSP authors' repository, verifies a SHA-256 checksum for
every file, and writes a validated local archive plus a provenance manifest in
`data/sachs/`.  For an offline copy of those same six files, use:

```bash
python3 experiments/prepare_sachs_data.py --source-dir /path/to/six/files
```

The generated data are deliberately not committed.  The authors' experiment
repository does not declare a data license; downloading exact, checksum-pinned
inputs for a local reproduction avoids silently redistributing them.  The
manifest cites both the original Sachs et al. study and the UT-IGSP paper.

The reproducible benchmark is:

| Context | File | Rows | Intended target (0-based) |
| --- | --- | ---: | --- |
| observational | `iv=.txt` | 1,755 | none |
| Akt inhibitor | `iv=6.txt` | 911 | Akt (6) |
| PKC inhibitor | `iv=8.txt` | 723 | PKC (8) |
| PIP2 inhibitor | `iv=3.txt` | 810 | PIP2 (3) |
| Mek inhibitor | `iv=1.txt` | 799 | Mek (1) |
| PIP3 inhibitor | `iv=4.txt` | 848 | PIP3 (4) |

This gives 1,755 observational and 4,091 interventional samples (5,846 total)
over 11 proteins.  The UT-IGSP paper's sentence referring to “8
interventional datasets” is inconsistent with both that stated 4,091 total
and the six released analysis files.  The broader nine-condition collection
has 7,466 rows, but its PMA and beta2-cAMP contexts were not used in this
benchmark.  Loading all 7,466 rows would therefore be a different experiment.

The released values are already elementwise natural logarithms.  The active
archive preserves them without additional standardization, imputation, or
downsampling.  Methods that center internally retain their native behavior.
PS-MIP explicitly centers within each context and scales by the observational
population standard deviation; that transformation is recorded in every
result row.

## 2. Fixed graph, paths, and evaluation

The accepted graph is the exact 18-arc DAG in the UT-IGSP authors' metadata,
using the node order
`Raf, Mek, PLCg, PIP2, PIP3, Erk, Akt, PKA, PKC, p38, JNK`.
Common 17-, 20-, and cyclic Sachs graph variants are rejected by the loader.

The paths are declared before seeing results:

| Method | Path | Points |
| --- | --- | ---: |
| PS-MIP unknown | graph multiples 1/16 through 16 of `log(N)/N`; target penalty fixed at `16 log(N)/N` | 9 |
| PS-MIP intended-present | same graph and target-penalty path; intended entries fixed to 1 and all other target indicators left unknown | 9 |
| PS-MIP oracle | graph multiples 1/16 through 16 of `log(N)/N`; target penalty irrelevant because targets are fixed | 9 |
| UT-IGSP* / intended-target UT-IGSP / IGSP | Sachs CI-alpha path; Gaussian invariance alpha `1e-20` | 8 each |
| GnIES | authors' raw-Sachs multipliers `0.01,0.25,0.5,0.75,1,2`, each times `log(N)` | 6 |
| GIES | public UT-IGSP history's BIC multipliers `2,200,600,700,800,900` | 6 |

The public UT-IGSP repository is not a turnkey numerical reproduction of its
Figure 4: its paper-era runner constructs a Gaussian invariance tester and then
overwrites it with HSIC at alpha `1e-5`, while the later Gaussian adapter uses
alpha `1e-20`; the plotting filter and later runner also disagree, and generated
estimates were not committed.  The jobs therefore record the current Gaussian
adapter and exact parameters rather than claiming to recover unpublished points.

The released Figure 4 plotting code scores saved DAG arcs against the accepted
DAG and draws unconnected TP-versus-FP scatter points.  This pipeline preserves
that convention.  UT-IGSP, UT-IGSP*, IGSP, and PS-MIP already return DAGs;
GnIES and the current Python GIES package return I-CPDAGs, so their jobs save
both the original I-CPDAG and a deterministic consistent DAG extension.  Their
ROC coordinates use the saved extension, and every result records that basis.

The synthetic graphical-lasso screen is not reused: a data-dependent screen
would make this small benchmark unnecessarily sensitive to one additional
tuning choice.  At 11 variables the truth-independent complete superstructure
is tractable (55 undirected pairs and 11,264 parent sets), so every Sachs
PS-MIP job uses it and records that choice.

Every job saves the returned adjacency, any learned targets, runtime,
configuration, package version, solver metadata, and fingerprints of both the
local estimator code and resolved Python environment.  Aggregation rejects a
method path that mixes those identities.  Evaluation is performed twice:

- Paper-compatible plots use directed and skeleton **TP versus FP counts**.
- Conventional plots use TPR/FPR.  The directed universe has 18 positives and
  92 negatives; the skeleton universe has 18 positives and 37 negatives.

No AUC or monotone envelope is reported because these method paths are not
nested score thresholds.  Duplicate or dominated points remain in the data.

## 3. Validate and submit on Quest

On Quest, activate the same `python39` environment used by the main experiment
and expose Gurobi before validating the frozen input and enumerating every job
without fitting:

```bash
module purge all
module load python-miniconda3
source activate python39
module load gurobi
python3 experiments/test_sachs_setup.py
```

Submit the three unknown-target arrays:

```bash
sbatch experiments/quest_jobs/sachs_roc_ps_mip_unknown.sh
sbatch experiments/quest_jobs/sachs_roc_utigsp_unknown.sh
sbatch experiments/quest_jobs/sachs_roc_gnies_unknown.sh
```

Submit the PS-MIP intended-present condition separately. It uses the same
known-present interpretation as intended-target UT-IGSP: supplied intended
targets cannot be removed, but additional off-targets may be learned.

```bash
sbatch experiments/quest_jobs/sachs_roc_ps_mip_intended.sh
```

Submit the exact intended-target oracle references separately:

```bash
sbatch experiments/quest_jobs/sachs_roc_ps_mip_oracle.sh
sbatch experiments/quest_jobs/sachs_roc_igsp_oracle.sh
sbatch experiments/quest_jobs/sachs_roc_gies_oracle.sh
```

For a direct contextual comparison with the namesake UT-IGSP curve in the
2020 paper, also submit its partially informed variant.  Supplied ones are
known intended targets; UT-IGSP can still discover additional targets:

```bash
sbatch experiments/quest_jobs/sachs_roc_utigsp_intended.sh
```

Each Slurm task owns one `experiment_results/sachs/parts/<method>/setting_*.csv`
fragment.  A per-fragment advisory lock rejects overlapping submissions, and
successful and failed attempts are checkpointed atomically.  A
resubmission skips a matching successful checkpoint.  A matching failed
checkpoint exits nonzero so Slurm cannot report a false success; pass
`--retry-failures` directly to the Python runner only after inspecting the
corresponding log.  If code or the resolved environment changed, the embedded
identity fingerprint will no longer match, so use `--overwrite` deliberately
instead.  A feasible
PS-MIP incumbent found at the time limit is saved as `ok_nonoptimal`, retained
as a visibly marked ROC point, and skipped on ordinary resubmission.  Pass
`--retry-nonoptimal` to rerun that setting with the same declared configuration.

## 4. Return results and make the figures

When the arrays finish, bring back the complete
`experiment_results/sachs/parts/` directory.  The validation and plotting
commands are:

```bash
python3 analysis/aggregate_sachs_roc.py
python3 analysis/plot_sachs_roc.py
```

Aggregation refuses incomplete, conflicting-duplicate, checksum-mismatched, or malformed
paths by default and recomputes all metrics from the saved adjacencies.  The
plotting step writes count-based and normalized directed/skeleton figures as
both PNG and PDF under `experiment_results/sachs/summary/`.

Primary references:

- Sachs et al. (2005), DOI: <https://doi.org/10.1126/science.1105809>
- Squires, Wang, and Uhler (2020):
  <https://proceedings.mlr.press/v124/squires20a.html>
- Immutable UT-IGSP Sachs files:
  <https://github.com/csquires/utigsp/tree/dda019b4bd2708fce4d6e383f2c3d37297759bd3/real_data_analysis/sachs/data>
- Immutable Figure 4 scoring code:
  <https://github.com/csquires/utigsp/blob/dda019b4bd2708fce4d6e383f2c3d37297759bd3/real_data_analysis/sachs/sachs_plot.py>
