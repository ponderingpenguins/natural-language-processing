import random
from copy import deepcopy
from itertools import product

import numpy as np
import torch
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
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_bert(model, data, cfg):
    """Fine-tune a transformer-based model on the provided dataset."""
    training_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
            )
        },
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ],
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    return {
        "best_eval_loss": trainer.state.best_metric,
        "final_eval_loss": eval_metrics.get("eval_loss"),
    }


def _score(result):
    """Return the best available eval loss from a result dict."""
    return result.get("best_eval_loss") or result.get("final_eval_loss")


def hyperparameter_tuning(cfg, data, model):
    """Run a grid search over learning rate, max length, and batch size"""

    grid = list(product(cfg.grid_learning_rates, cfg.grid_max_lengths))
    logger.info("Starting hyperparameter search with %d combinations", len(grid))

    results = []
    for idx, (lr, max_len) in enumerate(grid, 1):
        trial_cfg = deepcopy(cfg)
        trial_cfg.learning_rate = lr
        trial_cfg.max_length = max_len
        trial_cfg.output_dir = f"{cfg.output_dir}/grid_lr{lr}_ml{max_len}"
        trial_cfg.logging_dir = f"{cfg.logging_dir}/grid_lr{lr}_ml{max_len}"

        logger.info(
            "[%d/%d] lr=%s  max_length=%d  batch_size=%d",
            idx,
            len(grid),
            lr,
            max_len,
            cfg.per_device_train_batch_size,
        )

        metrics = train_bert(model, data, trial_cfg)

        results.append({"learning_rate": lr, "max_length": max_len, **metrics})

    best = min((r for r in results if _score(r) is not None), key=_score, default=None)

    if best:
        logger.info(
            "Best trial: lr=%s  max_length=%d  batch_size=%d  best_eval_loss=%s",
            best["learning_rate"],
            best["max_length"],
            best["batch_size"],
            best["best_eval_loss"],
        )

    return {"best": best, "all_results": results}
    return {"best": best, "all_results": results}
