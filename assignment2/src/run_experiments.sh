# !/bin/bash

# This script runs a series of experiments for different model types (CNN and LSTM) and sequence lengths (64, 128, 256).

echo "Starting experiments for model type and sequence length variations..."

echo "Configurations to be run:"
echo "1. CNN model with max sequence length 64"
echo "2. CNN model with max sequence length 128"
echo "3. CNN model with max sequence length 256"
echo "4. LSTM model with max sequence length 64"
echo "5. LSTM model with max sequence length 128"
echo "6. LSTM model with max sequence length 256"

# Run 6 configurations of modeltype x sequence length (model_type: lstm, cnn; max_seq_length: 64, 128, 256)

echo "Running experiments for CNN model with different sequence lengths..."

uv run python main.py model_type=cnn sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_cnn_seq64" batch_size=128 max_seq_length=64

uv run python main.py model_type=cnn sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_cnn_seq128" batch_size=128 max_seq_length=128

uv run python main.py model_type=cnn sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_cnn_seq256" batch_size=128 max_seq_length=256

echo "Running experiments for LSTM model with different sequence lengths..."

uv run python main.py model_type=lstm sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_lstm_seq64" batch_size=128 max_seq_length=64

uv run python main.py model_type=lstm sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_lstm_seq128" batch_size=128 max_seq_length=128

uv run python main.py model_type=lstm sample_size=100 tuning_num_epochs=1 num_epochs=2 output_dir="experiment_lstm_seq256" batch_size=128 max_seq_length=256

echo "All experiments completed!"
