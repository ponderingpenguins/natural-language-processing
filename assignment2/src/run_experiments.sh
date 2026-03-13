#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --array=0-7
#SBATCH --output=experiment_%a.out
#SBATCH --gpus-per-node=1

set -e # Exit immediately if a command exits with a non-zero status.

# Setup environment
# module purge
# module load CUDA/12.6.0
# module load Python/3.13.1-GCCcore-14.2.0
# module load Boost/1.79.0-GCC-11.3.0

# Array layout:
#   0   -> tune cnn  (seq=128, shared reference)
#   1   -> tune lstm (seq=128, shared reference)
#   2-4 -> train cnn  with seq=64/128/256 (uses tuning from index 0)
#   5-7 -> train lstm with seq=64/128/256 (uses tuning from index 1)
model_types=(cnn  lstm cnn  cnn  cnn  lstm lstm lstm)
seq_lengths=(128  128  64   128  256  64   128  256)
run_tuning=(true true false false false false false false)

# Get this job's parameters
# Default to 0 if not provided as first argument
TASK_ID=${1:-0}
MODEL=${model_types[$TASK_ID]}
SEQ=${seq_lengths[$TASK_ID]}
TUNE=${run_tuning[$TASK_ID]}

echo "Running: model_type=${MODEL}, max_seq_length=${SEQ}, tuning=${TUNE}"

# Load your modules / activate your environment here
# cd /scratch/s5982960/nlp-hw/assignment2/src
cd "$(dirname "$0")"
uv sync

# Tuning jobs write results to a shared dir; training jobs read from it.
TUNING_DIR=experiment_${MODEL}_tuning

if [ "${TUNE}" = "true" ]; then
    # Phase 1: tune once at seq=128, then exit
    uv run python main.py \
        model_type=${MODEL} \
        sample_size=100 \
        tuning_num_epochs=1 \
        num_epochs=2 \
        output_dir=${TUNING_DIR} \
        batch_size=512 \
        vocab_size=20000 \
        max_seq_length=128 \
        run_tuning_only=True
else
    # Phase 2: ablation — vary only max_seq_length, reuse shared tuning results
    uv run python main.py \
        model_type=${MODEL} \
        sample_size=100 \
        tuning_num_epochs=1 \
        num_epochs=2 \
        output_dir=experiment_${MODEL}_seq${SEQ} \
        batch_size=512 \
        vocab_size=20000 \
        max_seq_length=${SEQ} \
        run_train_only=True \
        tuning_dir=${TUNING_DIR}
fi