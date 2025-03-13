#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=short
#SBATCH -t 00-08:00:00
#SBATCH --job-name=Isla-Project
#SBATCH --mail-user=rui.carvalho@durham.ac.uk
#SBATCH --mem=56G

# Purge existing modules first
module purge

# Load necessary modules
module load cuda
module load nvidia/cuda

# Initialize conda
source /home3/grtq36/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate llm_env

# Add these diagnostic lines
echo "=== Environment Information ==="
echo "Modules loaded:"
module list
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "Operating System: $(uname -a)"
echo "CPU Info:"
lscpu | grep "Model name" || echo "CPU info not available"
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

# Add job start time logging
echo "Job started at: $(date)"
START_TIME=$(date +%s)

# Add memory monitoring
echo "Memory usage at start:"
free -h
echo ""

# Add disk space information
echo "Disk space information:"
df -h .
echo ""

# Add trap for cleanup if needed
trap 'echo "Job interrupted at $(date)"' SIGINT SIGTERM

# Function to check Python script execution
run_python_script() {
    local script=$1
    echo "Starting $script at $(date)"
    python "$script"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: $script failed with exit code $exit_code"
        exit $exit_code
    fi
    echo "Completed $script at $(date)"
    echo "Memory usage after $script:"
    free -h
    echo ""
}

# Run Python scripts with error checking
run_python_script step1.py
run_python_script step2.py
run_python_script step3.py
run_python_script step4.py

# At the end of your script
echo "Job completed at: $(date)"
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Total runtime: $((TOTAL_TIME/3600)) hours, $(((TOTAL_TIME%3600)/60)) minutes, $((TOTAL_TIME%60)) seconds"

# Final system state
echo "=== Final System State ==="
echo "Memory usage at end:"
free -h
echo "GPU status at end:"
nvidia-smi || echo "nvidia-smi failed"
echo "======================"