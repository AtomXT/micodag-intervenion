#!/bin/bash
#SBATCH --account=p32811
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:10:00
#SBATCH --mem=16G
#SBATCH --array=1-17%10
#SBATCH --job-name=chamber_ps_mip_present
#SBATCH --output=experiments/quest_jobs/outlog/chamber_ps_mip_present_%A_%a.out

set -e

module purge all
module load python-miniconda3
source activate python39
module load gurobi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

setting=$SLURM_ARRAY_TASK_ID
python3 -B experiments/run_chamber_experiment.py \
  --methods ps_mip_present \
  --setting "$setting" \
  --time-limit 3600 \
  --output-root experiment_results/causal_chambers/scm4_unscreened_1h_v1
