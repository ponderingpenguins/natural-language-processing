"""Model training entry points."""

from data_utils import create_dataloaders
from models.cnn import CNN
from models.lstm import LSTM
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from seed import set_seed
from tokenizer_utils import setup_tokenizer
from training import run_training_pipeline


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
