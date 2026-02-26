"""Main training script for Assignment 2"""

import os
import sys

from config import ModelConfig
from experiments import train_cnn_model, train_lstm_model
from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger


def plot_training_curves(cfg: TrainingConfig, model_name: str, history: list) -> None:
    """Plot training curves for loss and accuracy."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(
        0, max(max(train_loss + val_loss) * 1.1, 1.0)
    )  # Set y-axis limit for better visualization
    plt.title("Training and Validation Loss for " + model_name.upper())
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(
        0, max(max(train_acc + val_acc) * 1.1, 1.0)
    )  # Set y-axis limit for better visualization
    plt.title("Training and Validation Accuracy for " + model_name.upper())
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{cfg.output_dir}/training_curves_{model_name}.png")
    logger.info(
        "Saved training curves to %s/training_curves_%s.png", cfg.output_dir, model_name
    )
    plt.close()


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
    # logger.info("\n" + "=" * 60)
    # logger.info("Training CNN model")
    # logger.info("=" * 60)
    # cnn_results = train_cnn_model(cfg, model_cfg)
    # logger.info(
    #     "CNN training completed. Final test F1: %.4f", cnn_results["test_metrics"]["f1"]
    # )

    # Uncomment to train LSTM model later
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM model")
    logger.info("=" * 60)
    lstm_results = train_lstm_model(cfg, model_cfg)
    logger.info(
        "LSTM training completed. Final test F1: %.4f",
        lstm_results["test_metrics"]["f1"],
    )
    plot_training_curves(cfg, "lstm", lstm_results["history"])


if __name__ == "__main__":
    main()
