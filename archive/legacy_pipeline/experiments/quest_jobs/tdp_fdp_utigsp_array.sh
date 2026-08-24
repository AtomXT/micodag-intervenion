#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH --job-name=tdp_fdp_utigsp
#SBATCH --output=experiments/quest_jobs/outlog/tdp_fdp_utigsp_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge all
module load python-miniconda3
source activate "${UTIGSP_CONDA_ENV:-python39}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=42
export MPLBACKEND=Agg

UTIGSP_PYTHON="${UTIGSP_PYTHON:-python3}"
UTIGSP_OUTPUT_ROOT="${UTIGSP_OUTPUT_ROOT:-experiment_results/tdp_fdp_utigsp/parts}"
tied_alphas=(0.2 0.1 0.01 0.001 1e-5 1e-7 1e-9)

# One array task runs one complete trial: all three graphs and all seven
# values shared by alpha and alpha_inv. The full array therefore has 10 jobs.
trial="${SLURM_ARRAY_TASK_ID}"
printf -v task_name "trial_%02d" "${trial}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/micodag-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${UTIGSP_PYTHON}" -m experiments.run_tdp_fdp_experiments \
  --trial "${trial}" \
  --graphs 1 2 3 \
  --methods utigsp \
  --penalties "${tied_alphas[@]}" \
  --seed 42 \
  --utigsp-depth 4 \
  --utigsp-nruns 10 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --time-limit 3600 \
  --metric-time-limit 3600 \
  --output-dir "${UTIGSP_OUTPUT_ROOT}/${task_name}"
