#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH -p ug-gpu-small
#SBATCH --qos=normal
#SBATCH -t 00-08:00:00
#SBATCH --job-name=ssgg36
#SBATCH --mem=28G

# Purge existing modules first
module purge

# Load necessary modules
module load cuda
module load nvidia/cuda

# Initialize conda
source /home3/ssgg36/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate llm_env

# Add these diagnostic lines
echo "=== Environment Information ==="
echo "Modules loaded:"
module list
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "=== GPU Information ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"
echo ""

# Try different GPU detection methods
echo "Method 1: nvidia-smi"
nvidia-smi || echo "nvidia-smi failed"
echo ""

echo "Method 2: nvidia-debugdump"
nvidia-debugdump -l || echo "nvidia-debugdump failed"
echo "======================"

python step1.py