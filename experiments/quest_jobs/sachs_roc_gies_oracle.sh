#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=1-6%6
#SBATCH --job-name=sachs_gies
#SBATCH --output=experiments/quest_jobs/outlog/sachs_gies_%A_%a.out

set -eo pipefail

module purge all
module load python-miniconda3
source activate python39

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

setting_index=$((SLURM_ARRAY_TASK_ID - 1))
python3 experiments/run_sachs_roc.py \
  --method gies_oracle \
  --setting-index "$setting_index" \
  --threads 8 \
  --time-limit 10800 \
  --output "experiment_results/sachs/parts/gies_oracle/setting_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").csv"
