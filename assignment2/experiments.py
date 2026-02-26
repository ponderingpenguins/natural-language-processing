"""Model training entry points."""

import json

from config import ModelConfig
from data_utils import create_dataloaders
from models.cnn import CNN
from models.lstm import LSTM
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from seed import set_seed
from tokenizer_utils import setup_tokenizer
from training import run_training_pipeline


def save_misclassified_examples(
    cfg: TrainingConfig,
    model_name: str,
    misclassified_indices: list,
    misclassified_labels: list,
    test_data,
) -> None:
    """Save misclassified examples to a file."""
    output_path = f"{cfg.output_dir}/misclassified_examples_{model_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, label in zip(misclassified_indices, misclassified_labels):
            example = test_data[idx]
            record = {
                "index": int(idx),
                "text": example["text"],
                "true_label": int(example["label"]),
                "misclassified_as": int(label),
            }
            f.write(json.dumps(record) + "\n")
    logger.info(
        "Saved %d misclassified examples to %s",
        len(misclassified_indices),
        output_path,
    )


def train_cnn_model(cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Train a CNN model.

    Args:
        cfg: Training configuration.
        model_cfg: Model architecture configuration.

    Returns:
        Dictionary containing training results.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    # Load data
    logger.info("Loading data...")
    data = load_data(cfg)

    # Subsample for quick testing if sample_size is set
    if cfg.sample_size is not None:
        data["train"] = data["train"].select(
            range(min(cfg.sample_size, len(data["train"])))
        )
        data["dev"] = data["dev"].select(range(min(cfg.sample_size, len(data["dev"]))))
        data["test"] = data["test"].select(
            range(min(cfg.sample_size, len(data["test"])))
        )

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg, data["train"])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data, tokenizer, cfg)

    # Create CNN model with actual vocab size
    model = CNN(
        vocab_size=len(tokenizer.vocab),
        embed_dim=model_cfg.embed_dim,
        num_filters=model_cfg.cnn_num_filters,
        kernel_size=model_cfg.cnn_kernel_size,
        num_classes=cfg.num_classes,
    )
    logger.info("Created CNN model")

    results = run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
    )
    save_misclassified_examples(
        cfg,
        "cnn",
        results["misclassified_examples"],
        results["misclassified_labels"],
        data["test"],
    )
    return results


def train_lstm_model(cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Train an LSTM model.

    Args:
        cfg: Training configuration.
        model_cfg: Model architecture configuration.

    Returns:
        Dictionary containing training results.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    # Load data
    logger.info("Loading data...")
    data = load_data(cfg)

    # Subsample for quick testing if sample_size is set
    if cfg.sample_size is not None:
        data["train"] = data["train"].select(
            range(min(cfg.sample_size, len(data["train"])))
        )
        data["dev"] = data["dev"].select(range(min(cfg.sample_size, len(data["dev"]))))
        data["test"] = data["test"].select(
            range(min(cfg.sample_size, len(data["test"])))
        )

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg, data["train"])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data, tokenizer, cfg)

    # Create LSTM model
    model = LSTM(
        vocab_size=len(tokenizer.vocab),
        embed_dim=model_cfg.embed_dim,
        hidden_dim=model_cfg.lstm_hidden_dim,
        num_classes=cfg.num_classes,
        bidirectional=model_cfg.lstm_bidirectional,
    )
    logger.info("Created LSTM model")

    results = run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
    )
    save_misclassified_examples(
        cfg,
        "lstm",
        results["misclassified_examples"],
        results["misclassified_labels"],
        data["test"],
    )
    return results
