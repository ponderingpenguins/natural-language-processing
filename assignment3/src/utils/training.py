"""
Training utilities for text classification models with hyperparameter tuning.

This module provides functions to set random seeds for reproducibility, compute evaluation metrics, create a Hugging Face Trainer instance, define the hyperparameter search space, and run hyperparameter tuning using the Trainer's hyperparameter_search() method with Optuna as the backend. The best hyperparameters are saved to a JSON config file in the output directory for later use during evaluation.
"""

import inspect
import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from penguinlp.helpers import logger
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


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
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
    }


def _save_learning_curve_artifacts(output_dir, log_history):
    """Persist per-step training and evaluation loss history for later inspection."""
    curve_rows = []
    for entry in log_history:
        if "step" not in entry:
            continue

        row = {"step": entry["step"]}
        if "epoch" in entry:
            row["epoch"] = entry["epoch"]
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
        if "learning_rate" in entry:
            row["learning_rate"] = entry["learning_rate"]

        if len(row) > 1:
            curve_rows.append(row)

    if not curve_rows:
        logger.info("No step-level logs available to save for %s", output_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    history_path = os.path.join(output_dir, "learning_curve.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(curve_rows, f, indent=2)

    train_steps = [row["step"] for row in curve_rows if "train_loss" in row]
    train_loss = [row["train_loss"] for row in curve_rows if "train_loss" in row]
    eval_steps = [row["step"] for row in curve_rows if "eval_loss" in row]
    eval_loss = [row["eval_loss"] for row in curve_rows if "eval_loss" in row]

    if train_steps or eval_steps:
        plt.figure(figsize=(9, 5))
        if train_steps:
            plt.plot(
                train_steps,
                train_loss,
                marker="o",
                linewidth=1.6,
                label="train_loss",
            )
        if eval_steps:
            plt.plot(
                eval_steps,
                eval_loss,
                marker="s",
                linewidth=1.8,
                label="eval_loss",
            )

        plt.xlabel("Step")
        plt.ylabel("Loss")

        # y-axis starts at 0 and has a small margin above the max loss for better visualization
        max_loss = max(train_loss + eval_loss) if (train_loss and eval_loss) else 1.0
        plt.ylim(0, max_loss * 1.1)

        plt.title("Learning Curve")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "learning_curve.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        logger.info(
            "Saved learning curve artifacts to %s and %s", history_path, plot_path
        )


class LearningCurveCallback(TrainerCallback):
    """Write the trainer's log history to disk when a run finishes."""

    def on_train_end(self, args, state, control, **kwargs):
        _save_learning_curve_artifacts(args.output_dir, state.log_history)
        return control


def create_trainer(model_init, data, cfg):
    """Create a Trainer instance for hyperparameter search."""
    training_args = TrainingArguments(
        **{
            k: getattr(cfg, k)
            for k in (
                "eval_strategy",
                "eval_steps",
                "save_strategy",
                "save_steps",
                "load_best_model_at_end",
                "metric_for_best_model",
                "greater_is_better",
                "output_dir",
                # "dropout",
                "weight_decay",
                "learning_rate",
                "num_train_epochs",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "warmup_steps",
                "max_grad_norm",
                "logging_dir",
                "save_total_limit",
                "logging_steps",
                "report_to",
                "logging_first_step",
            )
        },
    )

    logger.info("Data lengths: train=%d, dev=%d", len(data["train"]), len(data["dev"]))

    # Build a dynamic-padding collator when a tokenizer is available.
    probe_model = model_init(None)
    tokenizer = getattr(probe_model, "tokenizer", None)
    expects_lengths = "lengths" in inspect.signature(probe_model.forward).parameters

    data_collator = None
    if tokenizer is not None:
        base_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def dynamic_collator(features):
            batch = base_collator(features)
            if expects_lengths and "attention_mask" in batch:
                batch["lengths"] = batch["attention_mask"].sum(dim=1)
            return batch

        data_collator = dynamic_collator

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
        LearningCurveCallback(),
    ]

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    return trainer


def hp_space_fn(_trial):
    """Define hyperparameter search space for Optuna (default backend)."""
    return {
        "learning_rate": _trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "per_device_train_batch_size": _trial.suggest_categorical(
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

    def model_init(_trial):
        """Initialize a fresh model for each trial."""
        return model_fn()

    trainer = create_trainer(model_init, data, cfg)

    # Define search space based on cfg
    def hp_space(_trial):
        return {
            "learning_rate": _trial.suggest_categorical(
                "learning_rate", cfg.grid_learning_rates
            ),
            "per_device_train_batch_size": _trial.suggest_categorical(
                "per_device_train_batch_size", cfg.grid_batch_sizes
            ),
        }

    logger.info(
        "Starting hyperparameter search with %d learning rates and %d batch sizes",
        len(cfg.grid_learning_rates),
        len(cfg.grid_batch_sizes),
    )

    best_trial: Any = trainer.hyperparameter_search(
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

    # Update config with best hyperparameters before saving
    cfg.learning_rate = best_trial.hyperparameters["learning_rate"]
    cfg.per_device_train_batch_size = best_trial.hyperparameters[
        "per_device_train_batch_size"
    ]

    # Save config to output dir
    config_path = os.path.join(cfg.output_dir, "config.json")
    Path(config_path).write_text(
        json.dumps(OmegaConf.to_container(cfg), indent=2), encoding="utf-8"
    )
    logger.info(
        "Best config saved to %s with learning_rate=%s, batch_size=%s",
        config_path,
        cfg.learning_rate,
        cfg.per_device_train_batch_size,
    )

    return {"best": best_trial, "trainer": trainer}
