"""Main training script for Assignment 2"""

import random
import sys

import numpy as np
import torch
from models.cnn import CNN
from models.lstm import LSTM
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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    num_epochs: int = 20,
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
    logger.info(f"num_epochs={num_epochs}, lr={lr}")
    print(
        f"{'epoch':>5}  {'train_loss':>10}  {'train_acc':>9}  {'val_loss':>8}  {'val_acc':>7}"
    )
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer)
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


def setup_tokenizer(cfg: TrainingConfig, train_data) -> object:
    """Setup and return tokenizer, either by loading or building new one.

    Args:
        cfg: Training configuration.
        train_data: Training dataset for building tokenizer if needed.

    Returns:
        Tokenizer object.
    """
    try:
        tokenizer = load_tokenizer(cfg.tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Building a new one.")
        tokenizer = build_tokenizer(
            train_data, tokenizer_type=cfg.tokenizer_type, vocab_size=1000
        )
        logger.info("Vocabulary built with %d tokens", len(tokenizer.vocab))
        save_tokenizer(tokenizer, cfg.tokenizer_path)

    return tokenizer


def create_dataloaders(
    data: dict, tokenizer: object, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test sets.

    Args:
        data: Dictionary with 'train', 'dev', and 'test' datasets.
        tokenizer: Tokenizer object.
        batch_size: Batch size for dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = TokenizedDataset(data["train"], tokenizer)
    val_ds = TokenizedDataset(data["dev"], tokenizer)
    test_ds = TokenizedDataset(data["test"], tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )

    logger.info("DataLoaders created successfully")
    return train_loader, val_loader, test_loader


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
) -> dict:
    """Run the complete training pipeline.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        num_epochs: Number of epochs to train.
        learning_rate: Learning rate for optimizer.

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
        lr=learning_rate,
        num_epochs=num_epochs,
    )

    # TODO: plot training history (loss and accuracy curves) using matplotlib or seaborn

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


def train_cnn_model(cfg: TrainingConfig) -> dict:
    """Train a CNN model.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary containing training results.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    # Load data
    logger.info("Loading data...")
    data = load_data(cfg)

    # Subsample for quick testing (remove or adjust for full training)
    data["train"] = data["train"].select(range(10))
    data["dev"] = data["dev"].select(range(10))
    data["test"] = data["test"].select(range(10))

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg, data["train"])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data, tokenizer, cfg.batch_size
    )

    # Create CNN model with actual vocab size
    model = CNN(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_classes=cfg.num_classes,
    )
    logger.info("Created CNN model")

    return run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=1e-3,
    )


def train_lstm_model(cfg: TrainingConfig) -> dict:
    """Train an LSTM model.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary containing training results.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    # Load data
    logger.info("Loading data...")
    data = load_data(cfg)

    # Subsample for quick testing (remove or adjust for full training)
    data["train"] = data["train"].select(range(10))
    data["dev"] = data["dev"].select(range(10))
    data["test"] = data["test"].select(range(10))

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg, data["train"])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data, tokenizer, cfg.batch_size
    )

    # Create LSTM model
    model = LSTM(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        hidden_dim=256,
        num_classes=cfg.num_classes,
    )
    logger.info("Created LSTM model")

    return run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=20,
        learning_rate=1e-3,
    )


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
    logger.info("\n" + "=" * 60)
    logger.info("Training CNN model")
    logger.info("=" * 60)
    cnn_results = train_cnn_model(cfg)
    logger.info(
        "CNN training completed. Final test F1: %.4f", cnn_results["test_metrics"]["f1"]
    )

    # Uncomment to train LSTM model later
    # logger.info("\n" + "="*60)
    # logger.info("Training LSTM model")
    # logger.info("="*60)
    # lstm_results = train_lstm_model(cfg)
    # logger.info("LSTM training completed. Final test F1: %.4f", lstm_results["test_metrics"]["f1"])


if __name__ == "__main__":
    main()
