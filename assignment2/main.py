"""Main training script for Assignment 2"""

import sys

from experiments import train_cnn_model, train_lstm_model
from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger


def main() -> None:
    """main function"""
    logger.info("Assignment 2: Training a TF-IDF classifier")
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)  # type: ignore
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Starting training pipeline...")

    # Train CNN model
    # logger.info("\n" + "=" * 60)
    # logger.info("Training CNN model")
    # logger.info("=" * 60)
    # cnn_results = train_cnn_model(cfg)
    # logger.info(
    #     "CNN training completed. Final test F1: %.4f", cnn_results["test_metrics"]["f1"]
    # )

    # Uncomment to train LSTM model later
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM model")
    logger.info("=" * 60)
    lstm_results = train_lstm_model(cfg)
    logger.info(
        "LSTM training completed. Final test F1: %.4f",
        lstm_results["test_metrics"]["f1"],
    )


if __name__ == "__main__":
    main()
