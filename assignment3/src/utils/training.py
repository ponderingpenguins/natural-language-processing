import json
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the BERT model."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
    }


def create_trainer(model_init, data, cfg):
    """Create a Trainer instance for hyperparameter search."""
    training_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        **{
            k: getattr(cfg, k)
            for k in (
                "output_dir",
                "learning_rate",
                "num_train_epochs",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "warmup_steps",
                "logging_dir",
                "save_total_limit",
            )
        },
    )
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
        ],
    )
    return trainer


def hp_space_fn(trial):
    """Define hyperparameter search space for Optuna (default backend)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
    }


def hyperparameter_tuning(cfg, data, model_fn):
    """
    Run hyperparameter search using trainer.hyperparameter_search().

    Args:
        cfg: Config object with grid_learning_rates, grid_max_lengths, etc.
        data: Dict with "train" and "dev" datasets.
        model_fn: Callable that returns a fresh model instance.
    """

    def model_init(trial):
        """Initialize a fresh model for each trial."""
        return model_fn()

    trainer = create_trainer(model_init, data, cfg)

    # Define search space based on cfg
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", cfg.grid_learning_rates
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", cfg.grid_batch_sizes
            ),
        }

    logger.info(
        "Starting hyperparameter search with %d learning rates and %d batch sizes",
        len(cfg.grid_learning_rates),
        len(cfg.grid_batch_sizes),
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=hp_space,
        backend="optuna",
        n_trials=len(cfg.grid_learning_rates) * len(cfg.grid_batch_sizes),
        direction="maximize",
        compute_objective=lambda metrics: metrics[
            "eval_f1"
        ],  # Use eval_f1 as the objective to maximize
    )

    logger.info(
        "Best trial: learning_rate=%s, batch_size=%s, eval_f1=%s",
        best_trial.hyperparameters.get("learning_rate"),
        best_trial.hyperparameters.get("per_device_train_batch_size"),
        best_trial.objective,
    )

    # Save config to output dir

    # Save config to output dir
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        # Convert OmegaConf to dict and save
        json.dump(OmegaConf.to_container(cfg), f, indent=2)
    logger.info("Config saved to %s", config_path)

    return {"best": best_trial, "trainer": trainer}
