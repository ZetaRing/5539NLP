#!/bin/bash
#SBATCH --job-name=sst2-modernbert
#SBATCH --account=PAS3389
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

cd "$SLURM_SUBMIT_DIR"
PY=~/.conda/envs/ih_rl/bin/python

nvidia-smi
$PY 4.1.py
$PY 4.2.py
