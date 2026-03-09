"""Training and evaluation loops."""

import random

import numpy as np
import torch
from penguinlp.helpers import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)
logger.info("Using device: %s", DEVICE)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Evaluate the model on the given data loader and return metrics."""
    model.eval()
    all_y = []
    all_pred = []
    all_prob = []
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)

            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_prob)
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
) -> dict:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    correct = 0

    pbar = tqdm(loader, desc="Training batch", leave=False)
    try:
        for batch in pbar:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            lengths = batch["lengths"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())

            # Update progress bar with current loss and accuracy
            pbar.set_postfix(
                {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}
            )
    finally:
        pbar.close()

    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    num_epochs: int,
    gradient_clip_norm: float,
    weight_decay: float,
    early_stopping_patience: int | None = None,
) -> list:
    """
    Train the model for a fixed number of epochs.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        lr: Learning rate.
        num_epochs: Number of epochs to train.
        weight_decay: Weight decay for optimizer.
        early_stopping_patience: Stop if validation loss does not improve for this
            many consecutive epochs. Disabled when None.

    Returns:
        List of training history metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0

    logger.info("Starting training")
    logger.info("num_epochs=%s, lr=%s", num_epochs, lr)
    print(
        f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}"
    )
    print("-" * 60)

    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    try:
        for epoch in epoch_bar:
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, gradient_clip_norm
            )
            val_metrics = evaluate(model, val_loader)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                }
            )

            epoch_line = (
                f"{epoch:5d}  {train_metrics['loss']:10.4f}  {train_metrics['acc']:9.4f}  "
                f"{val_metrics['loss']:8.4f}  {val_metrics['acc']:7.4f}"
            )
            print(epoch_line)
            epoch_bar.set_postfix(
                {"val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"]}
            )

            if early_stopping_patience is not None:
                improved = val_metrics["loss"] < best_val_loss - 1e-4
                if improved:
                    best_val_loss = val_metrics["loss"]
                    bad_epochs = 0
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stopping_patience:
                        logger.info(
                            "Early stopping triggered at epoch %d (patience=%d)",
                            epoch,
                            early_stopping_patience,
                        )
                        break
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\nTraining interrupted by user.")
        raise
    finally:
        epoch_bar.close()

    if early_stopping_patience is not None and best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best model checkpoint (val_loss=%.4f)", best_val_loss)

    logger.info("Training completed")
    return history


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg,
) -> dict:
    """Run the complete training pipeline.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        cfg: Training configuration object.

    Returns:
        Dictionary containing training history and metrics.
    """
    # Move model to device
    model = model.to(DEVICE)
    logger.info("Model moved to %s", DEVICE)

    # Train model
    history = train(
        model,
        train_loader,
        val_loader,
        lr=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        gradient_clip_norm=cfg.gradient_clip_norm,
        weight_decay=cfg.weighted_decay,
    )

    # Evaluate on validation and test sets
    logger.info("Evaluating model...")
    val_metrics = evaluate(model, val_loader)
    test_metrics = evaluate(model, test_loader)

    logger.info(
        "Validation metrics: loss=%.4f, acc=%.4f, f1=%.4f",
        val_metrics["loss"],
        val_metrics["acc"],
        val_metrics["f1"],
    )
    logger.info(
        "Test metrics: loss=%.4f, acc=%.4f, f1=%.4f",
        test_metrics["loss"],
        test_metrics["acc"],
        test_metrics["f1"],
    )

    # Print classification report
    print("\n" + "=" * 60)
    print("Test Set Classification Report:")
    print("=" * 60)
    print(
        classification_report(test_metrics["y_true"], test_metrics["y_pred"], digits=4)
    )

    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])

    # Collect misclassified examples
    misclassified_indices = np.where(test_metrics["y_true"] != test_metrics["y_pred"])[
        0
    ]
    misclassified_labels = test_metrics["y_pred"][misclassified_indices]
    num_to_report = min(
        cfg.max_misclassifications_to_report, len(misclassified_indices)
    )
    logger.info(
        "Reporting %d misclassified examples (out of %d total misclassifications)",
        num_to_report,
        len(misclassified_indices),
    )
    for i in range(num_to_report):
        idx = misclassified_indices[i]
        true_label = test_metrics["y_true"][idx]
        pred_label = test_metrics["y_pred"][idx]
        prob = test_metrics["probs"][idx][pred_label]
        logger.info(
            "Misclassified example %d: true=%d, pred=%d, prob=%.4f",
            i + 1,
            true_label,
            pred_label,
            prob,
        )

    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "misclassified_examples": misclassified_indices[:num_to_report],
        "misclassified_labels": misclassified_labels[:num_to_report],
    }
