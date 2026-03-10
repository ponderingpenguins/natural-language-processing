"""Plotting utilities for training curves and confusion matrices."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def plot_parallel_coordinates(
    cfg: TrainingConfig, model_name: str, results: list
) -> None:
    """
    Plot parallel coordinates visualization of hyperparameter search.

    Args:
        cfg: Training configuration
        model_name: Name of the model (e.g., "cnn", "lstm")
        results: List of dicts with 'params' and 'dev_f1' keys
    """
    # Convert results to DataFrame
    data = []
    for r in results:
        row = r["params"].copy()
        row["f1_score"] = r["dev_f1"]
        data.append(row)

    df = pd.DataFrame(data)

    # Prepare dimensions for parallel coordinates
    dimensions = []
    for col in df.columns:
        if col == "f1_score":
            continue

        # Handle list values (like kernel_sizes)
        if df[col].dtype == object:
            # Convert lists to strings for visualization
            df[col] = df[col].astype(str)
            unique_vals = sorted(df[col].unique())
            dimensions.append(
                dict(
                    label=col,
                    values=pd.Categorical(df[col]).codes,
                    ticktext=unique_vals,
                    tickvals=list(range(len(unique_vals))),
                )
            )
        else:
            dimensions.append(dict(label=col, values=df[col]))

    # Add F1 score dimension (will also be used for color)
    dimensions.append(dict(label="F1 Score", values=df["f1_score"], range=[0, 1]))

    # Create parallel coordinates plot
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["f1_score"],
                colorscale="Viridis",
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(title="F1 Score"),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title=f"Hyperparameter Search Results for {model_name.upper()}",
        height=600,
        font=dict(size=12),
    )

    # Save as HTML for interactivity
    output_path = f"{cfg.output_dir}/hyperparameter_search_{model_name}.html"
    fig.write_html(output_path)
    logger.info("Saved parallel coordinates plot to %s", output_path)
    logger.info("Saved parallel coordinates plot to %s", output_path)
