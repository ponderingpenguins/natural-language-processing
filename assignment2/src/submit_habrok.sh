#!/bin/bash
# Two-phase submission script.
# Phase 1 (indices 0-1): hyperparameter tuning for CNN and LSTM.
# Phase 2 (indices 2-7): training ablation, held until both tuning jobs succeed.

set -e

# Submit tuning phase and capture the job ID.
TUNE_JOB=$(sbatch --array=0-1 --parsable run_experiments_habrok.sh)
echo "Submitted tuning phase: job ${TUNE_JOB}"

# Submit training phase, blocked until every task in the tuning job succeeds.
TRAIN_JOB=$(sbatch --array=2-7 --dependency=afterok:${TUNE_JOB} --parsable run_experiments_habrok.sh)
echo "Submitted training phase: job ${TRAIN_JOB} (depends on ${TUNE_JOB})"
