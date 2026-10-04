#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --job-name=tabm-lora-eval
#SBATCH --output=slurm/logs/output_%x_%j.out
#SBATCH --error=slurm/logs/error_%x_%j.err

# Usage (run from the paper/ directory):
#   sbatch slurm/evaluate_tablora.sh <evaluation_dir>
#
# Example:
#   sbatch slurm/evaluate_tablora.sh exp/tabm-lora/california/0-evaluation


module load anaconda
eval "$(conda shell.bash hook)"
conda activate tabm

cd $HOME/DL_project/adapt-tabm/paper


python lib/data.py california 
