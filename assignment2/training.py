"""Training and evaluation loops."""

import numpy as np
import torch
from penguinlp.helpers import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)
logger.info("Using device: %s", DEVICE)


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
            x = batch.x.to(DEVICE)
            y = batch.y.to(DEVICE)

            logits = model(x)
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

    for batch in loader:
        x = batch.x.to(DEVICE)
        y = batch.y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())

    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    num_epochs: int,
    gradient_clip_norm: float,
) -> list:
    """
    Train the model for a fixed number of epochs.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        lr: Learning rate.
        num_epochs: Number of epochs to train.

    Returns:
        List of training history metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    logger.info("Starting training")
    logger.info("num_epochs=%s, lr=%s", num_epochs, lr)
    print(
        f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}"
    )
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
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

        print(
            f"{epoch:5d}  {train_metrics['loss']:10.4f}  {train_metrics['acc']:9.4f}  "
            f"{val_metrics['loss']:8.4f}  {val_metrics['acc']:7.4f}"
        )

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

    return {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
