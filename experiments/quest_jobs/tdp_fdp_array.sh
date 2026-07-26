#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=1-10
#SBATCH --job-name=tdp_fdp
#SBATCH --output=experiments/quest_jobs/outlog/tdp_fdp_%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tongxu2027@u.northwestern.edu

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"

module purge all
module load python-miniconda3
source activate python39
module load gurobi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python3 -m experiments.run_tdp_fdp_experiments \
  --trial "${SLURM_ARRAY_TASK_ID}" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --time-limit 3600 \
  --metric-time-limit 3600 \
  --output-dir experiment_results/tdp_fdp
