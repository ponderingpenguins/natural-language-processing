"""Hyperparameter tuning utilities for neural text classifiers."""

import json
from itertools import product
from typing import Any, Dict, List

import torch
from utils.data_utils import create_dataloaders
from models.cnn import CNN
from models.lstm import LSTM
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger
from tqdm import tqdm
from .training import DEVICE, evaluate, set_seed, train


def tune_hyperparameters(
    model_type: str,
    data: dict,
    tokenizer,
    cfg: TrainingConfig,
    param_grid: Dict[str, List[Any]],
    num_epochs: int = 5,
    patience: int = 3,
) -> Dict[str, Any]:
    """
    Perform grid search over hyperparameters and evaluate on dev set. Implements early stopping based on dev loss.

    Args:
        model_type: Either "cnn" or "lstm"
        data: Dictionary with train/dev/test splits
        tokenizer: Tokenizer object
        cfg: Training configuration
        param_grid: Dictionary mapping parameter names to lists of values to try
        num_epochs: Number of epochs to train each configuration
        patience: Early stopping patience (epochs without val-loss improvement)

    Returns:
        Dictionary containing:
            - results: List of dicts with hyperparams and F1 scores
            - best_config: Best hyperparameter configuration
            - best_f1: Best F1 score achieved
    """
    logger.info("Starting hyperparameter tuning for %s", model_type.upper())
    logger.info("Parameter grid: %s", param_grid)

    # Create dataloaders once (data/tokenizer do not change across combinations)
    train_loader, val_loader, _ = create_dataloaders(
        data, tokenizer, cfg, include_test=False
    )

    # Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    logger.info("Testing %d hyperparameter combinations", len(param_combinations))

    results = []
    best_f1 = 0.0
    best_config = None

    combo_bar = tqdm(param_combinations, desc=f"{model_type.upper()} tuning")
    for i, param_combo in enumerate(combo_bar):
        model = None
        params = dict(zip(param_names, param_combo))
        logger.info("\n" + "=" * 80)
        logger.info("Configuration %d/%d: %s", i + 1, len(param_combinations), params)
        logger.info("=" * 80)

        try:
            # Set seed for reproducibility
            set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

            # Create model with current hyperparameters
            if model_type == "cnn":
                model = CNN(
                    vocab_size=len(tokenizer.vocab),
                    embed_dim=params.get("embed_dim", 128),
                    num_filters=params.get("num_filters", 100),
                    kernel_sizes=params.get("kernel_sizes", [3, 5, 7]),
                    num_classes=cfg.num_classes,
                )
            elif model_type == "lstm":
                model = LSTM(
                    vocab_size=len(tokenizer.vocab),
                    embed_dim=params.get("embed_dim", 128),
                    hidden_dim=params.get("hidden_dim", 256),
                    num_classes=cfg.num_classes,
                    num_layers=params.get("num_layers", 2),
                    bidirectional=params.get("bidirectional", False),
                    dropout=params.get("dropout", 0.5),
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model = model.to(DEVICE)

            _ = train(
                model,
                train_loader,
                val_loader,
                lr=params.get("lr", cfg.learning_rate),
                num_epochs=num_epochs,
                gradient_clip_norm=cfg.gradient_clip_norm,
                weight_decay=params.get("weight_decay", cfg.weighted_decay),
                early_stopping_patience=patience,
            )

            # Evaluate on dev set
            val_metrics = evaluate(model, val_loader)
            dev_f1 = val_metrics["f1"]

            logger.info("Dev F1 score: %.4f", dev_f1)

            result = {
                "params": params,
                "dev_f1": float(dev_f1),
                "dev_acc": float(val_metrics["acc"]),
                "dev_loss": float(val_metrics["loss"]),
            }
            results.append(result)

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_config = params.copy()
                logger.info("New best F1: %.4f with params: %s", best_f1, params)

            combo_bar.set_postfix({"best_f1": f"{best_f1:.4f}"})

        except KeyboardInterrupt:
            logger.warning("Tuning interrupted by user")
            raise

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to train with params %s: %s", params, str(e))
            results.append(
                {
                    "params": params,
                    "dev_f1": 0.0,
                    "dev_acc": 0.0,
                    "dev_loss": float("inf"),
                    "error": str(e),
                }
            )

        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    combo_bar.close()

    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter tuning completed!")
    logger.info("Best F1: %.4f", best_f1)
    logger.info("Best config: %s", best_config)
    logger.info("=" * 80)

    return {
        "results": results,
        "best_config": best_config,
        "best_f1": float(best_f1),
        "param_names": param_names,
    }


def save_tuning_results(
    cfg: TrainingConfig, model_name: str, tuning_results: Dict[str, Any]
) -> None:
    """Save hyperparameter tuning results to a JSON file."""
    output_path = f"{cfg.output_dir}/hyperparameter_tuning_{model_name}.json"

    # Format results for saving
    save_data = {
        "model_type": model_name,
        "best_config": tuning_results["best_config"],
        "best_f1": tuning_results["best_f1"],
        "all_results": tuning_results["results"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    logger.info("Saved tuning results to %s", output_path)
    logger.info("Saved tuning results to %s", output_path)
