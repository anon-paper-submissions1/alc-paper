#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=120G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-44
arrayNum="${SLURM_ARRAY_TASK_ID}"
source ../.venv/bin/activate
python compare.py measure $arrayNum