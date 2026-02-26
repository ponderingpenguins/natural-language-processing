"""Main training script for Assignment 2"""

import os
import sys

from config import ModelConfig
from experiments import train_cnn_model, train_lstm_model
from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger
from plots import plot_confusion_matrix, plot_training_curves


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


if __name__ == "__main__":
    main()
