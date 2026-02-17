from typing import Any

from .config import TrainingConfig
from .helpers import logger, report_stats


def evaluate_models_on_test_set(
    cfg: TrainingConfig, best_models: dict[str, Any], X_test: Any, y_test: Any
) -> None:
    """
    Evaluate the best models on the test set and report misclassified examples.

    Args:
        cfg: The training configuration.
        best_models: A dictionary of the best models for each model type.
        X_test: The test set features.
        y_test: The test set labels.
    Returns:
        None
    """

    # Evaluate on test once for the final numbers.
    for name, clf in best_models.items():
        logger.info("Evaluating %s on test set...", name)
        y_pred_test = clf.predict(X_test)
        report_stats(name + " (Test)", y_test, y_pred_test)

        cfg.max_misclassifications_to_report = 20
        misclassified_indices = [
            i
            for i, (y_true, y_pred) in enumerate(zip(y_test, y_pred_test))
            if y_true != y_pred
        ]
        logger.info("Found %d misclassified examples", len(misclassified_indices))
        if len(misclassified_indices) >= cfg.max_misclassifications_to_report:
            logger.info("Categorizing misclassified examples...")
            for i in misclassified_indices[: cfg.max_misclassifications_to_report]:
                logger.info(
                    "Misclassified example %d: True=%s, Predicted=%s",
                    i,
                    y_test[i],
                    y_pred_test[i],
                )
