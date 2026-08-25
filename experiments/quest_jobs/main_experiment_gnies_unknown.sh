#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --array=1-10%10
#SBATCH --job-name=gnies_unknown
#SBATCH --output=experiments/quest_jobs/outlog/gnies_unknown_%A_%a.out

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MPLBACKEND=Agg

replicate=$SLURM_ARRAY_TASK_ID
python3 experiments/run_main_experiment.py \
  --replicate "$replicate" \
  --methods gnies_unknown \
  --time-limit 3600 \
  --metric-time-limit 3600 \
  --output "experiment_results/main_experiment/parts/gnies_unknown/replicate_$(printf '%03d' "$replicate").csv"
