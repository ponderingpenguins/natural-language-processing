#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --array=0-5
#SBATCH --output=experiment_%a.out
#SBATCH --gpus-per-node=1

set -e # Exit immediately if a command exits with a non-zero status.

# Setup environment
module purge
module load CUDA/12.6.0
module load Python/3.13.1-GCCcore-14.2.0
module load Boost/1.79.0-GCC-11.3.0

# Define parameter arrays (indexed 0-5)
model_types=(cnn cnn cnn lstm lstm lstm)
seq_lengths=(64 128 256 64 128 256)

# Get this job's parameters
MODEL=${model_types[$SLURM_ARRAY_TASK_ID]}
SEQ=${seq_lengths[$SLURM_ARRAY_TASK_ID]}

echo "Running: model_type=${MODEL}, max_seq_length=${SEQ}"

# Load your modules / activate your environment here
cd /scratch/s5982960/nlp-hw/assignment2/src
uv sync
source .venv/bin/activate

uv run python main.py \
    model_type=${MODEL} \
    sample_size=100 \
    tuning_num_epochs=1 \
    num_epochs=2 \
    output_dir="experiment_${MODEL}_seq${SEQ}" \
    batch_size=128 \
    max_seq_length=${SEQ}