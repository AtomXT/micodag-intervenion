# Active data code

`DataGeneration.py` provides the shared low-level graph, intervention, and
Gaussian-sampling functions. Its command-line interface retains the older
file-layout defaults and is not the main-experiment entry point.

Generate the complete main-experiment suite with:

```bash
python3 experiments/generate_main_experiment_data.py
```

That dedicated script writes and validates 60 immutable archives for
`p={10,20,30}`, `e={1,2}`, and 10 replicates, plus
`data/main_experiment/manifest.json`. Model-fitting jobs only load those files;
they never generate data implicitly. Generation is not restricted to one NumPy
release; the actual version is recorded in the manifest. The generated suite
is intended to be committed and uploaded with the repository so compute jobs
receive exactly the same arrays.

Historical SyntheticData, Vary_n, RealData, and the older generator references
are preserved under `archive/legacy_pipeline/data/`.

The later six-context Sachs bundle is archived with its scripts and results
under `archive/sachs/2026-09-05/data/sachs/`; it is not an active input.
