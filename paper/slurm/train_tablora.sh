#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --job-name=tabm-lora
#SBATCH --output=slurm/logs/output_%x_%j.out
#SBATCH --error=slurm/logs/error_%x_%j.err

# Usage (run from the paper/ directory):
#   sbatch slurm/train_tablora.sh <config.toml> <output_dir>
#
# Example:
#   sbatch slurm/train_tablora.sh exp/tabm-lora/california/0-evaluation/0.toml exp/tabm-lora/california/0-evaluation/0

CONFIG=$1
OUTPUT=$2

if [ -z "$CONFIG" ] || [ -z "$OUTPUT" ]; then
    echo "Error: config and output arguments are required."
    echo "Usage: sbatch slurm/train_tablora.sh <config.toml> <output_dir>"
    exit 1
fi

# Activate conda environment
module load anaconda
eval "$(conda shell.bash hook)"
conda activate tabm

# Install tabm-specific packages if not already present
#pip install -q delu==0.0.25 rtdl_num_embeddings==0.0.11 rtdl_revisiting_models==0.0.2

# Must be run from the paper/ directory
cd $HOME/adapt-tabm/paper

echo "Config:  $CONFIG"
echo "Output:  $OUTPUT"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $CUDA_VISIBLE_DEVICES"

mkdir -p "$(dirname "$OUTPUT")"

python bin/model.py "$CONFIG" --output "$OUTPUT" --force
