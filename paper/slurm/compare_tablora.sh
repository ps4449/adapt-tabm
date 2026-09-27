#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=tabm-lora-compare
#SBATCH --output=slurm/logs/output_%x_%j.out
#SBATCH --error=slurm/logs/error_%x_%j.err

# Generates a side-by-side TabM vs TabLoRA comparison table for all datasets.
# Run this after all evaluate + ensemble jobs have completed.
#
# Usage (from paper/ directory):
#   sbatch slurm/compare_tablora.sh
#   sbatch slurm/compare_tablora.sh adult higgs-small covtype2 diamond california

module load anaconda
eval "$(conda shell.bash hook)"
conda activate tabm

cd $HOME/adapt-tabm/paper

echo "========================================"
echo "TabM vs TabLoRA Comparison"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
echo "========================================"

if [ $# -gt 0 ]; then
    python tools/compare_results.py --datasets "$@"
else
    python tools/compare_results.py
fi

echo "Done: $(date)"
