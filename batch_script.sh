#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH -p ug-gpu-small
#SBATCH --qos=normal
#SBATCH -t 02-00:00:00
#SBATCH --job-name=ssgg36
#SBATCH --mem=28G

module purge
# Load CUDA 11.8 with cuDNN 8.7 - specific version to match environment.yml
module load cuda/11.8-cudnn8.7


# Initialize conda
source /home3/ssgg36/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate new_env

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

# Run the script with increased CUDA memory settings
python step1.py
python step2.py
python step3.py
python step4.py
python fine_tune.py
