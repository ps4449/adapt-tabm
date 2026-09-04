#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=tabm-lora-ensemble
#SBATCH --output=slurm/logs/output_%x_%j.out
#SBATCH --error=slurm/logs/error_%x_%j.err

# Runs ensemble evaluation and saves a documented summary to logs/results/.
#
# Usage (run from the paper/ directory):
#   sbatch slurm/ensemble_tablora.sh <evaluation_dir> [ensemble_size]
#
# Example:
#   sbatch slurm/ensemble_tablora.sh exp/tabm-lora/california/0-evaluation
#   sbatch slurm/ensemble_tablora.sh exp/tabm-lora/california/0-evaluation 5

EVAL_DIR=$1
ENSEMBLE_SIZE=${2:-5}   # default ensemble size = 5

if [ -z "$EVAL_DIR" ]; then
    echo "Error: evaluation_dir argument is required."
    echo "Usage: sbatch slurm/ensemble_tablora.sh <evaluation_dir> [ensemble_size]"
    exit 1
fi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate tabm

cd $HOME/adapt-tabm/paper

echo "========================================"
echo "TabLoRA Ensemble Job"
echo "Evaluation dir : $EVAL_DIR"
echo "Ensemble size  : $ENSEMBLE_SIZE"
echo "Node           : $SLURMD_NODENAME"
echo "GPU            : $CUDA_VISIBLE_DEVICES"
echo "Started        : $(date)"
echo "========================================"

# Step 1: Run ensemble evaluation
echo ""
echo "[1/2] Running bin/ensemble.py ..."
python bin/ensemble.py "$EVAL_DIR" \
    --ensemble_size "$ENSEMBLE_SIZE" \
    --force

# Step 2: Generate and save documented summary
echo ""
echo "[2/2] Generating results summary ..."
python bin/summarize_results.py "$EVAL_DIR" \
    --ensemble_size "$ENSEMBLE_SIZE"

echo ""
echo "Done: $(date)"
echo "Summary saved to: logs/results/"
