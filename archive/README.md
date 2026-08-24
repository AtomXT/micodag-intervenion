# Archive

This directory preserves material that is not part of the redesigned main
experiment. Nothing here was deleted during the cleanup.

## Layout

- `legacy_pipeline/` is the versioned historical bundle. Its subdirectories
  mirror the former repository layout: root-level methods and notebooks,
  `analysis/`, `data/`, `experiments/`, and `experiment_results/`.
- `local/` contains ignored machine-local artifacts such as earlier inspection
  files and solver logs. These remain on this workstation but are not intended
  for version control.

The active main experiment does not read historical results or static data; its
dedicated generator writes a separate validated suite under
`data/main_experiment/`. The standalone `MIP.py`, `MIP_profiled.py`, `DCDI.py`,
`UTIGSP.py`, and `GIES.py` commands also load that active persisted suite. Only
scripts inside this archive retain historical fixture paths.

## Move manifest

| Former location | Archived location |
| --- | --- |
| `MIP_naive.py`, `GNIES.py` | `legacy_pipeline/` |
| `evaluation.R`, `gies_test.R` | `legacy_pipeline/` |
| `reproduce.ipynb`, `temp.ipynb` | `legacy_pipeline/` |
| `analysis/aggregate_tdp_fdp_results.py` | `legacy_pipeline/analysis/` |
| `analysis/tdp_fdp.{pdf,png}` | `legacy_pipeline/analysis/` |
| `data/DataGeneration.R` | `legacy_pipeline/data/` |
| `data/DataGeneration_vary_n.py` | `legacy_pipeline/data/` |
| `data/SyntheticData/` | `legacy_pipeline/data/SyntheticData/` |
| `data/Vary_n/` | `legacy_pipeline/data/Vary_n/` |
| `data/RealData/` | `legacy_pipeline/data/RealData/` |
| `experiments/run_tdp_fdp_experiments.py` | `legacy_pipeline/experiments/` |
| `experiments/quest_jobs/tdp_fdp_*.sh` | `legacy_pipeline/experiments/quest_jobs/` |
| previous `experiment_results/*` | `legacy_pipeline/experiment_results/` |
| `tmp/`, `gurobi.log` | ignored `local/` machine archive |
| Python caches and Finder metadata | ignored `local/cache/` and `local/metadata/` |

## Restoring a historical path

To restore any item, remove the `archive/legacy_pipeline/` prefix. For example,
`archive/legacy_pipeline/data/RealData/` formerly lived at `data/RealData/`, and
`archive/legacy_pipeline/analysis/aggregate_tdp_fdp_results.py` formerly lived
at `analysis/aggregate_tdp_fdp_results.py`.

Archived scripts are snapshots of the earlier workflow. Some retain historical
dependencies and path assumptions, so they are not maintained as part of the
current pipeline.
