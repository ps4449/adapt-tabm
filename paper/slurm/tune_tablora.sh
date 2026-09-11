#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --job-name=tabm-lora-tune
#SBATCH --output=slurm/logs/output_%x_%j.out
#SBATCH --error=slurm/logs/error_%x_%j.err

# Usage (from paper/ directory):
#   sbatch slurm/tune_tablora.sh <tuning_config.toml>
#
# Example:
#   sbatch slurm/tune_tablora.sh exp/tabm-lora/california/0-tuning.toml

TUNING_CONFIG=$1

if [ -z "$TUNING_CONFIG" ]; then
    echo "Error: tuning config argument is required."
    echo "Usage: sbatch slurm/tune_tablora.sh <tuning_config.toml>"
    exit 1
fi

module load anaconda
eval "$(conda shell.bash hook)"
conda activate tabm

cd $HOME/adapt-tabm/paper

echo "========================================"
echo "TabLoRA Tuning Job"
echo "Config  : $TUNING_CONFIG"
echo "Node    : $SLURMD_NODENAME"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"
echo "========================================"

python bin/tune.py "$TUNING_CONFIG" --force

echo "Done: $(date)"
