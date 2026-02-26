"""Plotting utilities for training curves and confusion matrices."""

import matplotlib.pyplot as plt
import seaborn as sns
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger


def plot_training_curves(cfg: TrainingConfig, model_name: str, history: list) -> None:
    """Plot training curves for loss and accuracy."""

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


def plot_confusion_matrix(
    cfg: TrainingConfig, model_name: str, confusion_matrix: list
) -> None:
    """Plot confusion matrix."""

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg.label_names.values(),
        yticklabels=cfg.label_names.values(),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for " + model_name.upper())
    plt.tight_layout()

    plt.savefig(f"{cfg.output_dir}/confusion_matrix_{model_name}.png")
    logger.info(
        "Saved confusion matrix to %s/confusion_matrix_%s.png",
        cfg.output_dir,
        model_name,
    )
    plt.close()
