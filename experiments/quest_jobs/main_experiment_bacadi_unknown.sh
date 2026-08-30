#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --array=0-59%10
#SBATCH --job-name=bacadi_unknown
#SBATCH --output=experiments/quest_jobs/outlog/bacadi_unknown_%A_%a.out

set -e

module purge all
module load python-miniconda3
source activate bacadi39

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MPLBACKEND=Agg
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Each task runs one (p, e, replicate) instance. BaCaDI is too slow to place
# all six design cells for a replicate in one scheduler allocation.
task=$SLURM_ARRAY_TASK_ID
replicate=$((task / 6 + 1))
cell=$((task % 6))
p_index=$((cell / 2))
edge_multiplier=$((cell % 2 + 1))
p_values=(10 20 30)
p=${p_values[$p_index]}

python3 experiments/run_main_experiment.py \
  --replicate "$replicate" \
  --p-values "$p" \
  --edge-multipliers "$edge_multiplier" \
  --methods bacadi_unknown \
  --time-limit 3600 \
  --metric-time-limit 3600 \
  --output "experiment_results/main_experiment/parts/bacadi_unknown/p_${p}_e_${edge_multiplier}_replicate_$(printf '%03d' "$replicate").csv"
