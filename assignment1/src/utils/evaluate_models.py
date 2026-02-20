"""
Functions to evaluate trained models on the test set and save misclassified examples.
"""

import json
import os
from typing import Any

from .config import TrainingConfig
from .helpers import logger, report_stats


def evaluate_models_on_test_set(
    cfg: TrainingConfig,
    best_models: dict[str, Any],
    X_test: Any,
    y_test: Any,
) -> dict[str, tuple[Any, list[int]]]:
    """
    Evaluate the best models on the test set and return misclassified indices.

    Args:
        cfg: The training configuration.
        best_models: A dictionary of the best models for each model type.
        X_test: The test set features.
        y_test: The test set labels.
    Returns:
        A dict mapping model name -> (y_pred, misclassified_indices).
    """

    results: dict[str, tuple[Any, list[int]]] = {}

    for name, clf in best_models.items():
        logger.info("Evaluating %s on test set...", name)
        y_pred_test = clf.predict(X_test)
        report_stats(name + " (Test)", y_test, y_pred_test)

        misclassified_indices = [
            i
            for i, (y_true, y_pred) in enumerate(zip(y_test, y_pred_test))
            if y_true != y_pred
        ]
        logger.info("Found %d misclassified examples", len(misclassified_indices))

        cfg.max_misclassifications_to_report = 20
        if len(misclassified_indices) >= cfg.max_misclassifications_to_report:
            logger.info("Categorizing misclassified examples...")
            for i in misclassified_indices[: cfg.max_misclassifications_to_report]:
                logger.info(
                    "Misclassified example %d: True=%s, Predicted=%s",
                    i,
                    y_test[i],
                    y_pred_test[i],
                )

        results[name] = (y_pred_test, misclassified_indices)

    return results


def save_misclassified(
    cfg: TrainingConfig,
    results: dict[str, tuple[Any, list[int]]],
    y_test: Any,
    test_ds: Any,
) -> None:
    """
    Save misclassified examples to JSONL files, looking up texts by index.

    Args:
        cfg: The training configuration.
        results: Dict of model name -> (y_pred, misclassified_indices).
        y_test: The test set labels.
        test_ds: The test dataset with 'raw_text' and 'text' fields.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    for name, (y_pred, indices) in results.items():
        safe_name = name.replace(" ", "_").lower()
        out_path = os.path.join(cfg.output_dir, f"misclassified_{safe_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i in indices:
                true_label = int(y_test[i])
                pred_label = int(y_pred[i])
                example = test_ds[i]
                record = {
                    "index": i,
                    "true_label": true_label,
                    "true_class": cfg.label_names.get(true_label, "Unknown"),
                    "predicted_label": pred_label,
                    "predicted_class": cfg.label_names.get(pred_label, "Unknown"),
                    "raw_text": example["raw_text"],
                    "preprocessed_text": example["text"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved %d misclassified examples to %s", len(indices), out_path)
