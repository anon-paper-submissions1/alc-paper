#!/bin/bash
#SBATCH --job-name=compare_prior
#SBATCH --output=slurm_prior_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=60G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-44
arrayNum="${SLURM_ARRAY_TASK_ID}"
source ../.venv/bin/activate
python compare.py prior_measure $arrayNum