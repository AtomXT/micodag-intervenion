#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=1-10%10
#SBATCH --job-name=micodag_main
#SBATCH --output=experiments/quest_jobs/outlog/main_%A_%a.out
#SBATCH --mail-type=FAIL,END

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"

module purge all
module load python-miniconda3
source activate "${MAIN_CONDA_ENV:-python39}"
module load gurobi
if [[ -n "${MAIN_R_MODULE:-}" ]]; then
  module load "${MAIN_R_MODULE}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/micodag-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" experiments/quest_jobs/outlog experiment_results/main_experiment/parts

data_root=${MAIN_DATA_ROOT:-data/main_experiment}
master_seed=${MAIN_MASTER_SEED:-20260823}
if [[ ! -f "${data_root}/manifest.json" ]]; then
  echo "missing ${data_root}/manifest.json; generate and commit the main datasets before submitting" >&2
  exit 2
fi

# Exactly one array task owns each replicate and runs its complete 156-setting
# grid: 3 p values x 2 densities x 26 method/tuning settings.
replicate=${SLURM_ARRAY_TASK_ID}
if (( replicate < 1 || replicate > 10 )); then
  echo "main-experiment replicate ${replicate} is outside 1..10" >&2
  exit 2
fi
result_path="experiment_results/main_experiment/parts/replicate_$(printf '%03d' "${replicate}").csv"

main_python=${MAIN_PYTHON:-python}
dcdi_root=${DCDI_ROOT:-external/dcdi}
rscript=${RSCRIPT:-Rscript}
fit_time_limit=${MAIN_FIT_TIME_LIMIT:-21600}
metric_time_limit=${MAIN_METRIC_TIME_LIMIT:-3600}
preflight_timeout=${MAIN_PREFLIGHT_TIMEOUT:-120}
retry_arguments=()
if [[ "${MAIN_RETRY_FAILURES:-0}" == "1" ]]; then
  retry_arguments+=(--retry-failures)
fi

if [[ ! -f "${dcdi_root}/main.py" ]]; then
  echo "missing tracked DCDI author source at ${dcdi_root}; restore external/dcdi from Git" >&2
  exit 2
fi

if [[ "${rscript}" == */* ]]; then
  if [[ ! -x "${rscript}" ]]; then
    echo "RSCRIPT is not executable: ${rscript}" >&2
    exit 2
  fi
elif ! command -v "${rscript}" >/dev/null 2>&1; then
  echo "Rscript is unavailable; export RSCRIPT=/absolute/path/to/Rscript or MAIN_R_MODULE" >&2
  exit 2
fi
if [[ "$(basename "${rscript}")" != "Rscript" ]]; then
  echo "RSCRIPT must point to an executable named Rscript for unmodified DCDI" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "the Quest job requires the standard timeout command for bounded preflights" >&2
  exit 2
fi
timeout "${preflight_timeout}" "${rscript}" --vanilla -e '
required <- c("pcalg", "SID")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop(paste("missing R packages:", paste(missing, collapse = ", ")))
'

"${main_python}" experiments/run_main_experiment.py \
  --data-root "${data_root}" \
  --seed "${master_seed}" \
  --replicate "${replicate}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --time-limit "${fit_time_limit}" \
  --metric-time-limit "${metric_time_limit}" \
  --dcdi-root "${dcdi_root}" \
  --rscript "${rscript}" \
  "${retry_arguments[@]}" \
  --output "${result_path}"
