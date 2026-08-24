# Legacy pipeline snapshot

This bundle contains the superseded scripts, notebooks, generated datasets,
and experiment outputs that previously occupied the active project tree.
Original relative organization is preserved beneath this directory for
provenance. The active workflow is documented in the repository-root README.

Key groups are:

- root: the naive MIP, old GNIES wrapper, R evaluation scripts, and notebooks;
- `analysis/`: the earlier TDP-FDP aggregator and rendered figure;
- `data/`: the original R generators plus SyntheticData, Vary_n, and RealData;
- `experiments/`: the old sweep runner and Slurm jobs; and
- `experiment_results/`: all result artifacts that predate the main experiment.

These files should be treated as a historical snapshot, not as current entry
points. Copy an item back to its former path if exact legacy execution requires
its old working-directory assumptions.
