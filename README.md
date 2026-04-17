# Mixed-Integer Programming for Causal Learning with Interventions

This repository contains research code for learning directed acyclic graphs (DAGs) from a mixture of observational and interventional data, including settings where intervention targets are unknown. The codebase combines exact mixed-integer optimization, faster coordinate-descent style approximations, synthetic data generation, baseline comparisons, and experiment runners used to produce the saved results in `experiment_results/`.

The current repository is best understood as a research prototype rather than a polished package. It preserves the original algorithms, datasets, and experiment outputs that motivated an upcoming rewrite.

## Project Goal

The main problem studied here is:

- infer a causal DAG from multiple environments,
- use both observational and interventional samples,
- optionally recover which variables were intervened on,
- compare exact and approximate optimization strategies against existing baselines.

The accompanying write-up frames the method as a mixed-integer programming approach for causal structure learning with unknown interventions, together with an alternating optimization / coordinate-descent approximation for improved scalability.

## Statistical Model

The project studies multi-environment causal discovery under a linear Gaussian structural equation model. For each environment \(e\) and node \(j\),

\[
X_j^e = \sum_{k=1}^p \beta_{kj}^e X_k^e + \epsilon_j^e,
\qquad
\epsilon_j^e \sim \mathcal{N}(0, \nu_j^e).
\]

The observational environment plays the role of a baseline DAG, while interventional environments may modify a subset of node mechanisms. In the hard-intervention setting, intervening on node \(j\) removes the effect of its parents in that environment. More generally, the notes also discuss soft interventions and noise interventions.

A useful reparameterization in the write-up is

\[
\Gamma^e = (I - B^e)(D^e)^{1/2},
\]

or closely related variants depending on the intervention setting. In this parameterization, the off-diagonal sparsity pattern of \(\Gamma^e\) encodes the DAG structure, while environment-to-environment changes are concentrated in columns corresponding to intervention targets.

## Formulation at a Glance

For unknown intervention targets, the formulation introduces binary variables \(\delta_j^e\) indicating whether node \(j\) is intervened on in environment \(e\), together with binary edge-support variables for the baseline graph. At a high level, the estimator combines:

- a weighted Gaussian negative log-likelihood across environments,
- a sparsity penalty on the baseline DAG,
- a penalty on the number of intervened targets,
- acyclicity constraints for the baseline graph,
- and coupling constraints that force each interventional environment to match the baseline except at intervened columns.

In the exact mixed-integer formulation from the older write-up, the key coupling constraints are of the form

\[
\Gamma_{ij}^e = \Gamma_{ij}^{e_0}(1-\delta_j^e), \qquad i \neq j,
\]

with a separate diagonal relation allowing the variance term of an intervened node to change. Intuitively, if \(\delta_j^e = 1\), the incoming effects into node \(j\) are cut off in environment \(e\); otherwise that column is shared with the observational model.

The newer notes also sketch a more scalable regularized view: estimate one baseline observational DAG and let interventional environments deviate from it through column-sparse changes, using an \(\ell_0\) penalty for baseline sparsity and a group penalty such as \(\ell_{2,1}\) to encourage only a small number of changed columns. That perspective aligns well with the planned rewrite.

## Method Overview

At a high level, the project currently contains:

- an exact mixed-integer optimization approach for joint DAG and intervention-target recovery,
- a faster alternating optimization / coordinate-descent approximation,
- synthetic multi-environment benchmarks,
- comparisons against established causal discovery baselines,
- and evaluation code for structural and essential-graph recovery.

This description is intentionally high-level because the repository is about to be reorganized and many implementation files will change.

## Data Included in the Repo

The repository already includes several datasets and generated artifacts:

- `data/SyntheticData/`
  Synthetic multi-environment benchmark data used by the main experiments.

- `data/SyntheticData_preliminary/`
  Earlier synthetic datasets and graph instances.

- `data/RealData/`
  Sachs observational and interventional data, plus processed test cases.

- `experiment_results/`
  Saved estimated DAGs, estimated intervention targets, timing tables, and plots from previous runs.

## Dependencies

### Python

The Python scripts expect a scientific Python environment with at least:

- `numpy`
- `pandas`
- `matplotlib`
- `causaldag`
- `gnies`
- `gurobipy`

The exact optimization code requires:

- a working Gurobi installation,
- an active Gurobi license.

The Quest job scripts under `experiments/quest_jobs/` assume a Python 3.9 Conda environment plus a loaded Gurobi module.

### R

The R scripts rely on packages including:

- `pcalg`
- `igraph`
- `gRbase`
- `MASS`
- `glue`

## Current Usage

The current codebase is still script-driven and was developed as a research workflow rather than a stable package. Running experiments presently requires invoking the existing Python and R scripts directly, with Gurobi available for the exact optimization routines.

As part of the rewrite, this will likely be replaced by a cleaner interface with standardized configuration, entry points, and reproducible environments.

## Results and Artifacts

The repository already contains saved experimental outputs from substantial synthetic benchmarking, including estimated graphs, estimated intervention targets, timing summaries, and evaluation plots. These historical artifacts are useful as reference points for validating the rewrite.

## Important Caveats

- This is a research codebase, not a packaged library.
- Several scripts contain hard-coded file paths or machine-specific assumptions.
- Some files refer to `./Data/...` while the repository folder is `data/...`. On case-sensitive filesystems, those paths may need to be updated before scripts run successfully.
- The codebase mixes current implementations with older experiments and exploratory notebooks/scripts.
- There is no single end-to-end CLI yet; scripts are run directly.

## Why This Repository Exists

This codebase captures the original implementation behind a larger project on causal learning with interventions:

- exact mixed-integer formulations for structure learning,
- scalable coordinate-descent approximations,
- synthetic-data benchmarking,
- baseline comparison against GNIES and UT-IGSP,
- early experiments on real data such as Sachs.

It is a strong starting point for a rewrite because it already contains:

- the core optimization ideas,
- the dataset conventions,
- the experiment definitions,
- and the historical outputs needed for regression checking.

## Planned Rewrite Direction

If this repository is being used as the starting point for a rewrite, the most natural next steps are:

- separate legacy experiments from the core library,
- standardize data paths and configuration,
- package the optimization methods behind a clean API,
- unify evaluation and result writing,
- document reproducible environments for Python and R,
- and add tests around data loading, objective calculations, and graph recovery outputs.
