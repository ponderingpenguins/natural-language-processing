# Assignment 2: Neural Model Comparison + Ablation

## Setup

To run the code, first make sure you have the required libraries installed. You can do this by running:

```bash
uv sync --no-dev
```

Run commands from `assignment2/src`.

## Execution modes

The pipeline is controlled by `model_type`, `run_tuning_only`, and `run_train_only`.

- `model_type`: `cnn` or `lstm`
- `run_tuning_only=true`: tune hyperparameters and exit
- `run_train_only=true`: skip tuning, load best config from `output/hyperparameter_tuning_<model_type>.json`, and train

Default behavior (`run_tuning_only=false`, `run_train_only=false`) is:
1) run tuning for `model_type`,
2) load best hyperparameters from JSON,
3) train on full data (`sample_size=None`) with that best config.

## Commands

### Full pipeline (tune + train)

```bash
uv run python main.py model_type=cnn
```

### Quick dev run

```bash
uv run python main.py model_type=cnn sample_size=100 tuning_num_epochs=1
```

### Hyperparameter tuning only

```bash
uv run python main.py model_type=cnn run_tuning_only=true sample_size=1000 batch_size=64 vocab_size=20000 tuning_num_epochs=5 early_stopping_patience=3 max_seq_len=64
```

### Training only (use previously saved best tuning config)

```bash
uv run python main.py model_type=cnn run_train_only=true
```