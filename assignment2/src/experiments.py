"""Model training entry points."""

import json

from utils.config import ModelConfig
from utils.data_utils import create_dataloaders, preprocess_data, setup_tokenizer
from models.cnn import CNN
from models.lstm import LSTM
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from training import run_training_pipeline, set_seed


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


def train_model(cfg: TrainingConfig, model_cfg: ModelConfig, model_type: str) -> dict:
    """Train a model (CNN or LSTM).

    Args:
        cfg: Training configuration.
        model_cfg: Model architecture configuration.
        model_type: Type of model to train ("cnn" or "lstm").

    Returns:
        Dictionary containing training results.
    """
    # Set seed for reproducibility
    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    # Load data
    logger.info("Loading data...")
    data = load_data(cfg)
    data = preprocess_data(data)

    # Subsample for quick testing if sample_size is set
    if cfg.sample_size is not None:
        for split in ["train", "dev", "test"]:
            data[split] = data[split].select(
                range(min(cfg.sample_size, len(data[split])))
            )

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg, data["train"])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data, tokenizer, cfg)

    # Create model based on type
    if model_type == "cnn":
        model = CNN(
            vocab_size=len(tokenizer.vocab),
            embed_dim=model_cfg.embed_dim,
            num_filters=model_cfg.cnn_num_filters,
            kernel_sizes=model_cfg.cnn_kernel_sizes,
            num_classes=cfg.num_classes,
        )
    elif model_type == "lstm":
        model = LSTM(
            vocab_size=len(tokenizer.vocab),
            embed_dim=model_cfg.embed_dim,
            hidden_dim=model_cfg.lstm_hidden_dim,
            num_classes=cfg.num_classes,
            num_layers=model_cfg.lstm_num_layers,
            bidirectional=model_cfg.lstm_bidirectional,
            dropout=model_cfg.lstm_dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Created %s model", model_type.upper())

    results = run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
    )
    save_misclassified_examples(
        cfg,
        model_type,
        results["misclassified_examples"],
        results["misclassified_labels"],
        data["test"],
    )
    return results


# Backward compatibility wrappers (need to refactor out the old train_cnn_model and train_lstm_model functions)
def train_cnn_model(cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Train a CNN model."""
    return train_model(cfg, model_cfg, "cnn")


def train_lstm_model(cfg: TrainingConfig, model_cfg: ModelConfig) -> dict:
    """Train an LSTM model."""
    return train_model(cfg, model_cfg, "lstm")
