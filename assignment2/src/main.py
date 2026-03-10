"""Main training script for Assignment 2."""

import json
import os
import sys
from copy import deepcopy

from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from sklearn.model_selection import train_test_split
from utils.config import ModelConfig
from utils.data_utils import clear_cache_dirs, preprocess_data, setup_tokenizer
from utils.experiments import train_model
from utils.hyperparameter_tuning import save_tuning_results, tune_hyperparameters
from utils.plots import (
    plot_confusion_matrix,
    plot_parallel_coordinates,
    plot_training_curves,
)
from utils.training import set_seed


def load_training_config() -> TrainingConfig:
    """Load and validate training configuration from defaults + CLI."""
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        return TrainingConfig(**cfg)  # type: ignore
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)


def get_param_grid(model_type: str) -> dict:
    """Return the hyperparameter grid for the requested model."""
    if model_type == "cnn":
        return {
            "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
            "num_filters": [50, 100, 150, 200, 300],
            "kernel_sizes": [[3], [5], [3, 5], [3, 5, 7], [2, 3, 4, 5]],
            "embed_dim": [64, 128, 256, 512],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "dropout": [0.1, 0.3, 0.5],
        }
    if model_type == "lstm":
        return {
            "lr": [0.001, 0.0001],
            "hidden_dim": [128, 256, 512],
            "embed_dim": [128, 256],
            "bidirectional": [False, True],
            "weight_decay": [0.0, 1e-5],
        }
    raise ValueError(f"Unknown model type: {model_type}")


def apply_best_hyperparameters(
    cfg: TrainingConfig, model_cfg: ModelConfig, model_type: str, best_config: dict
) -> tuple[TrainingConfig, ModelConfig]:
    """Apply best tuning hyperparameters to runtime training/model configs."""
    tuned_cfg = deepcopy(cfg)
    tuned_model_cfg = deepcopy(model_cfg)

    if "lr" in best_config:
        tuned_cfg.learning_rate = best_config["lr"]
    if "weight_decay" in best_config:
        tuned_cfg.weighted_decay = best_config["weight_decay"]
    if "embed_dim" in best_config:
        tuned_model_cfg.embed_dim = best_config["embed_dim"]

    if model_type == "cnn":
        if "num_filters" in best_config:
            tuned_model_cfg.cnn_num_filters = best_config["num_filters"]
        if "kernel_sizes" in best_config:
            tuned_model_cfg.cnn_kernel_sizes = best_config["kernel_sizes"]
    elif model_type == "lstm":
        if "hidden_dim" in best_config:
            tuned_model_cfg.lstm_hidden_dim = best_config["hidden_dim"]
        if "bidirectional" in best_config:
            tuned_model_cfg.lstm_bidirectional = best_config["bidirectional"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return tuned_cfg, tuned_model_cfg


def load_best_tuning_config(cfg: TrainingConfig, model_type: str) -> dict:
    """Load best hyperparameters from saved tuning JSON."""
    tuning_file = os.path.join(
        cfg.output_dir, f"hyperparameter_tuning_{model_type}.json"
    )
    if not os.path.exists(tuning_file):
        raise FileNotFoundError(
            f"Tuning results file not found: {tuning_file}. Run tuning first."
        )

    with open(tuning_file, "r", encoding="utf-8") as file_handle:
        tuning_payload = json.load(file_handle)

    best_config = tuning_payload.get("best_config")
    if not isinstance(best_config, dict) or not best_config:
        raise ValueError(f"No valid best_config found in {tuning_file}")

    return best_config


def run_hyperparameter_tuning(cfg: TrainingConfig, model_type: str) -> None:
    """Run hyperparameter tuning and save artifacts for one model type."""
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER TUNING FOR %s", model_type.upper())
    logger.info("=" * 80)

    set_seed(cfg.seed if hasattr(cfg, "seed") else 42)

    logger.info("Loading data...")
    data = load_data(cfg)
    data = preprocess_data(data)

    if cfg.sample_size is not None:
        for split in ["train", "dev", "test"]:
            if len(data[split]) > cfg.sample_size:
                labels = data[split]["label"]
                indices = list(range(len(data[split])))
                _, selected_indices = train_test_split(
                    indices,
                    test_size=cfg.sample_size,
                    stratify=labels,
                    random_state=cfg.seed if hasattr(cfg, "seed") else 42,
                )
                data[split] = data[split].select(selected_indices)

    tokenizer = setup_tokenizer(cfg, data["train"])

    tuning_results = tune_hyperparameters(
        model_type=model_type,
        data=data,
        tokenizer=tokenizer,
        cfg=cfg,
        param_grid=get_param_grid(model_type),
        num_epochs=cfg.tuning_num_epochs,
        patience=cfg.early_stopping_patience,
    )

    save_tuning_results(cfg, model_type, tuning_results)
    plot_parallel_coordinates(cfg, model_type, tuning_results["results"])


def main() -> None:
    """Run tuning/training pipeline for selected model based on config flags."""
    logger.info("Assignment 2: Training Neural Text Classifiers")
    cfg = load_training_config()
    model_cfg = ModelConfig()

    model_type = cfg.model_type.lower()
    if model_type not in {"cnn", "lstm"}:
        raise ValueError(
            f"Unsupported model_type '{cfg.model_type}'. Use 'cnn' or 'lstm'."
        )

    if cfg.run_tuning_only and cfg.run_train_only:
        raise ValueError(
            "Invalid configuration: run_tuning_only and run_train_only cannot both be True."
        )

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info(
        "Model configuration:\n%s", OmegaConf.to_yaml(OmegaConf.structured(model_cfg))
    )
    logger.info("Starting pipeline for %s", model_type.upper())

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.clear_cache:
        clear_cache_dirs(cfg)

    if not cfg.run_train_only:
        run_hyperparameter_tuning(cfg, model_type)
        if cfg.run_tuning_only:
            logger.info(
                "run_tuning_only=True: skipping training and exiting after tuning"
            )
            return

    best_config = load_best_tuning_config(cfg, model_type)
    logger.info(
        "Loaded best hyperparameters for %s: %s", model_type.upper(), best_config
    )

    tuned_cfg, tuned_model_cfg = apply_best_hyperparameters(
        cfg, model_cfg, model_type, best_config
    )

    tuned_cfg.sample_size = None
    logger.info(
        "Training on full data (sample_size=None) with best tuned hyperparameters"
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training %s model", model_type.upper())
    logger.info("=" * 60)
    results = train_model(tuned_cfg, tuned_model_cfg, model_type)
    logger.info(
        "%s training completed. Final test F1: %.4f",
        model_type.upper(),
        results["test_metrics"]["f1"],
    )
    plot_training_curves(tuned_cfg, model_type, results["history"])
    plot_confusion_matrix(tuned_cfg, model_type, results["confusion_matrix"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
