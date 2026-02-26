"""Main training script for Assignment 2"""

import random
import sys

import numpy as np
import torch
from models.cnn import CNN
from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.tokanizer import build_tokenizer, load_tokenizer, save_tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else DEVICE)
logger.info(f"Using device: {DEVICE}")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Batch:
    """A simple Batch class to hold batch data."""

    def __init__(self, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.lengths = lengths
        self.y = y


class TokenizedDataset(Dataset):
    """A PyTorch Dataset that tokenizes text examples on the fly."""

    def __init__(self, data, tokenizer, max_seq_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = self.tokenizer(example["text"])
        # Truncate or pad to max_seq_len
        input_ids = tokens[: self.max_seq_len]
        if len(input_ids) < self.max_seq_len:
            input_ids.extend([0] * (self.max_seq_len - len(input_ids)))
        return {"input_ids": input_ids, "labels": example["label"]}


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
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
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
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())

    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 3,
) -> list:
    """
    Train with early stopping on validation loss.

    We track the best validation loss seen so far. If it does not improve for `patience`
    consecutive epochs, we stop training and restore the best model parameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0

    history = []
    print("Training with early stopping on validation loss")
    print(f"max_epochs={max_epochs}, patience={patience}, lr={lr}")
    print(
        f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}  {'note':>18}"
    )
    print("-" * 70)

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)

        improved = val_metrics["loss"] < best_val - 1e-4
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
            note = "new best val_loss"
        else:
            bad_epochs += 1
            note = f"no improve ({bad_epochs}/{patience})"

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
            f"{val_metrics['loss']:8.4f}  {val_metrics['acc']:7.4f}  {note:>18}"
        )

        if bad_epochs >= patience:
            print()
            print(
                f"Early stopping triggered at epoch {epoch} because validation loss did not improve "
                f"for {patience} consecutive epochs."
            )
            print(f"Best validation loss was {best_val:.4f} at epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored model parameters from epoch {best_epoch} (best val_loss={best_val:.4f})."
        )

    return history


def collate_fn(batch: list, vocab: dict[str, int]) -> Batch:
    """Collate function to convert a list of samples into a batch."""
    # batch: list of dicts with 'input_ids' and 'labels' keys
    PAD = vocab.get("<PAD>", 0)
    lengths = torch.tensor([len(item["input_ids"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(batch) > 0 else 0
    x = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    y = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return Batch(x=x, lengths=lengths, y=y)


def fooberino(cfg: TrainingConfig) -> None:
    """
    Placeholder function for the main training logic.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """
    # TODO: Implement the training logic here
    data = load_data(cfg)

    # load tokenizer if it exists, otherwise build and save a new one
    try:
        tokenizer = load_tokenizer(cfg.tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Building a new one.")

        tokenizer = build_tokenizer(
            data["train"], tokenizer_type=cfg.tokenizer_type, vocab_size=1000
        )
        logger.info("Vocabulary built with %d tokens", len(tokenizer.vocab))

        save_tokenizer(tokenizer, cfg.tokenizer_path)

    # Tokenize and collate data here (e.g., create DataLoader)

    # subsample for quick testing
    data["train"] = data["train"].select(range(10))
    data["dev"] = data["dev"].select(range(10))
    data["test"] = data["test"].select(range(10))

    train_ds = TokenizedDataset(data["train"], tokenizer)
    val_ds = TokenizedDataset(data["dev"], tokenizer)
    test_ds = TokenizedDataset(data["test"], tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )

    # Train CNN model here using the tokenizer and data
    model = CNN(
        vocab_size=len(tokenizer.vocab), embed_dim=128, num_classes=cfg.num_classes
    )
    set_seed(13)

    MAX_EPOCHS = 20
    PATIENCE = 3
    LEARNING_RATE = 1e-3

    model_base = CNN(
        vocab_size=len(tokenizer.vocab), embed_dim=128, num_classes=cfg.num_classes
    ).to(DEVICE)
    hist_base = fit_with_early_stopping(
        model_base,
        train_loader,
        val_loader,
        lr=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )

    base_val = evaluate(model_base, val_loader)
    base_test = evaluate(model_base, test_loader)

    breakpoint()


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
    logger.info("Data loaded successfully. Starting training...")
    fooberino(cfg)  # TODO: Replace with actual training function


if __name__ == "__main__":
    main()
