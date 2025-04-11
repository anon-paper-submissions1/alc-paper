#!/bin/bash
#SBATCH --job-name=dependences
#SBATCH --output=slurm_out/out.%a.out
#SBATCH --time=7-0
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-2280
arrayNum="${SLURM_ARRAY_TASK_ID}"
source ../.venv/bin/activate
python compare.py measure $arrayNum