#!/usr/bin/env bash
set -euo pipefail

# Runs a fixed LSTM ablation sweep for overfitting analysis.

PYTHON_BIN="${PYTHON_BIN:-python}"
PENGUINLP_SRC="${PENGUINLP_SRC:-../../penguinlp/src}"

run_experiment() {
  local name="$1"
  shift
  echo "[ablation] running: ${name}"
  PYTHONPATH="${PENGUINLP_SRC}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" run_lstm_overfitting_experiment.py "$@"
}

run_experiment baseline \
  experiment.name=baseline \
  experiment.change_summary="Baseline LSTM config." \
  experiment.hypothesis="Reference run for train vs eval loss gap." \
  lstm_model.output_dir=./experiment_outputs/baseline \
  lstm_model.logging_dir=./experiment_outputs/baseline_logs

run_experiment early_stopping \
  experiment.name=early_stopping \
  experiment.change_summary="Enable early stopping, evaluate each epoch." \
  experiment.hypothesis="Stopping at best dev checkpoint should reduce overfitting." \
  lstm_model.output_dir=./experiment_outputs/early_stopping \
  lstm_model.logging_dir=./experiment_outputs/early_stopping_logs \
  training.eval_strategy=epoch \
  training.save_strategy=epoch \
  training.early_stopping_patience=2 \
  training.num_train_epochs=8 \
  training.metric_for_best_model=eval_loss \
  training.greater_is_better=false

run_experiment regularized_lstm \
  experiment.name=regularized_lstm \
  experiment.change_summary="Smaller 2-layer BiLSTM with recurrent dropout." \
  experiment.hypothesis="Smaller capacity should memorize less." \
  lstm_model.output_dir=./experiment_outputs/regularized_lstm \
  lstm_model.logging_dir=./experiment_outputs/regularized_lstm_logs \
  lstm_model.embed_dim=256 \
  lstm_model.init_embeddings_from_bert=false \
  lstm_model.hidden_dim=128 \
  lstm_model.num_layers=2 \
  lstm_model.dropout=0.5 \
  lstm_model.learning_rate=0.0003 \
  lstm_model.weight_decay=0.01

run_experiment shorter_sequence \
  experiment.name=shorter_sequence \
  experiment.change_summary="Sequence length 128 instead of 256." \
  experiment.hypothesis="Shorter context may reduce noise for AG News." \
  lstm_model.output_dir=./experiment_outputs/shorter_sequence \
  lstm_model.logging_dir=./experiment_outputs/shorter_sequence_logs \
  lstm_model.sequence_length=128

run_experiment more_data \
  experiment.name=more_data \
  experiment.change_summary="Use more data with stronger regularization." \
  experiment.hypothesis="Data scale should help generalization more than one regularizer." \
  lstm_model.output_dir=./experiment_outputs/more_data \
  lstm_model.logging_dir=./experiment_outputs/more_data_logs \
  dataset.max_samples=5000 \
  dataset.eval_max_samples=1000 \
  lstm_model.embed_dim=256 \
  lstm_model.init_embeddings_from_bert=false \
  lstm_model.hidden_dim=128 \
  lstm_model.num_layers=2 \
  lstm_model.dropout=0.5 \
  lstm_model.learning_rate=0.0003 \
  lstm_model.weight_decay=0.01 \
  training.eval_strategy=epoch \
  training.save_strategy=epoch \
  training.early_stopping_patience=2 \
  training.num_train_epochs=8 \
  training.metric_for_best_model=eval_loss \
  training.greater_is_better=false

echo "[ablation] done"
