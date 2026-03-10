"""
Main training script for Assignment 2

Dev:
```bash
python main.py sample_size=100
```

Final training:
```bash
python3 main.py sample_size=1000 batch_size=64 vocab_size=20000 tuning_num_epochs=5 early_stopping_patience=3
```

"""

import os
import sys

from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from sklearn.model_selection import train_test_split
from utils.config import ModelConfig
from utils.data_utils import clear_cache_dirs, preprocess_data, setup_tokenizer
from utils.experiments import train_cnn_model, train_lstm_model
from utils.hyperparameter_tuning import save_tuning_results, tune_hyperparameters
from utils.plots import (
    plot_confusion_matrix,
    plot_parallel_coordinates,
    plot_training_curves,
)
from utils.training import set_seed


def main() -> None:
    """main function"""
    logger.info("Assignment 2: Training Neural Text Classifiers")

    # Load training configuration
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)  # type: ignore
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    # Load model configuration (uses defaults, can be overridden via CLI)
    model_cfg = ModelConfig()

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info(
        "Model configuration:\n%s", OmegaConf.to_yaml(OmegaConf.structured(model_cfg))
    )
    logger.info("Starting training pipeline...")

    # create output directory if it doesn't exist
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.clear_cache:
        clear_cache_dirs(cfg)

    # Train CNN model
    logger.info("\n" + "=" * 60)
    logger.info("Training CNN model")
    logger.info("=" * 60)
    cnn_results = train_cnn_model(cfg, model_cfg)
    logger.info(
        "CNN training completed. Final test F1: %.4f", cnn_results["test_metrics"]["f1"]
    )
    plot_training_curves(cfg, "cnn", cnn_results["history"])
    plot_confusion_matrix(cfg, "cnn", cnn_results["confusion_matrix"])

    # Train LSTM model later
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM model")
    logger.info("=" * 60)
    lstm_results = train_lstm_model(cfg, model_cfg)
    logger.info(
        "LSTM training completed. Final test F1: %.4f",
        lstm_results["test_metrics"]["f1"],
    )
    plot_training_curves(cfg, "lstm", lstm_results["history"])
    plot_confusion_matrix(cfg, "lstm", lstm_results["confusion_matrix"])


def tune_model_hyperparameters(model_type: str = "cnn") -> None:
    """
    Perform hyperparameter tuning for a specified model.

    Args:
        model_type: Either "cnn" or "lstm"

    This function will:
    1. Load and preprocess data
    2. Perform grid search over hyperparameters
    3. Optimize F1 score on the dev set
    4. Create heatmap visualization
    5. Save the best configuration
    """
    # Load training configuration
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)  # type: ignore
    except TypeError:
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING FOR %s", model_type.upper())
    logger.info("=" * 80)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.clear_cache:
        clear_cache_dirs(cfg)

    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    logger.info("Loading data...")
    data = load_data(cfg)
    data = preprocess_data(data)

    # Subsample for quick testing if sample_size is set
    if cfg.sample_size is not None:
        # Use stratified sampling to keep class ratios

        for split in ["train", "dev", "test"]:
            if len(data[split]) > cfg.sample_size:
                labels = data[split]["label"]
                indices = list(range(len(data[split])))
                _, selected_indices = train_test_split(
                    indices,
                    test_size=cfg.sample_size,
                    stratify=labels,
                    random_state=cfg.seed if hasattr(cfg, "seed") else 42,
                )
                data[split] = data[split].select(selected_indices)

    tokenizer = setup_tokenizer(cfg, data["train"])

    # Define hyperparameter grids
    if model_type == "cnn":
        param_grid = {
            "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
            "num_filters": [50, 100, 150, 200, 300],
            "kernel_sizes": [[3], [5], [3, 5], [3, 5, 7], [2, 3, 4, 5]],
            "embed_dim": [64, 128, 256, 512],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "dropout": [0.1, 0.3, 0.5],
        }
    elif model_type == "lstm":
        param_grid = {
            "lr": [0.001, 0.0001],
            "hidden_dim": [128, 256, 512],
            "embed_dim": [128, 256],
            "bidirectional": [False, True],
            "weight_decay": [0.0, 1e-5],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run hyperparameter tuning
    tuning_results = tune_hyperparameters(
        model_type=model_type,
        data=data,
        tokenizer=tokenizer,
        cfg=cfg,
        param_grid=param_grid,
        num_epochs=cfg.tuning_num_epochs,
        patience=cfg.early_stopping_patience,
    )

    # Save results
    save_tuning_results(cfg, model_type, tuning_results)

    # Create parallel coordinates plot
    plot_parallel_coordinates(cfg, model_type, tuning_results["results"])

    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETE!")
    logger.info("Best F1 score: %.4f", tuning_results["best_f1"])
    logger.info("Best configuration: %s", tuning_results["best_config"])
    logger.info(
        "Results saved to: %s/hyperparameter_tuning_%s.json", cfg.output_dir, model_type
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        # hyperparmeter tuning
        tune_model_hyperparameters("cnn")  # or "lstm"
        # tune_model_hyperparameters("lstm")

        # reconfigure for final training
        # main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
