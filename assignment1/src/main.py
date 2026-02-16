import sys
from collections.abc import Callable
from typing import Any

from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from .utils.config import TrainingConfig
from .utils.data import load_and_preprocess_data
from .utils.helpers import logger, report_stats


def train_tfidf_classifier(cfg: TrainingConfig) -> None:
    """
    Train a TF-IDF classifier.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_and_preprocess_data(cfg)

    # Train two classical models (required):
    #     TF-IDF + Logistic Regression
    #     TF-IDF + Linear SVM

    param_grids: dict[str, list[dict[str, object]]] = {
        "Logistic Regression": [{"C": c} for c in [0.01, 0.1, 1.0, 10.0, 100.0]],
        "Linear SVM": [
            {"C": c, "loss": loss}
            for c in [0.01, 0.1, 1.0, 10.0, 100.0]
            for loss in ["hinge", "squared_hinge"]
        ],
    }

    model_classes: dict[str, Callable[[dict[str, Any]], Any]] = {
        "Logistic Regression": lambda params: LogisticRegression(
            max_iter=1000, **params
        ),
        "Linear SVM": lambda params: LinearSVC(**params),
    }

    best_models: dict[str, Any] = {}

    # Train both baseline models. Keep the dev split for model selection/tuning.
    for name, grid in param_grids.items():
        logger.info("Running grid search for %s...", name)
        best_score = -1
        best_clf = None
        best_params: dict[str, Any] | None = None

        for params in grid:
            clf = model_classes[name](params)
            clf.fit(X_train, y_train)
            y_pred_dev = clf.predict(X_dev)

            score = accuracy_score(y_dev, y_pred_dev)
            logger.info("  Params: %s -> Dev Accuracy: %.4f", params, score)

            if score > best_score:
                best_score = score
                best_clf = clf
                best_params = params

        assert best_clf is not None, f"No model was trained for {name}"
        best_models[name] = best_clf
        logger.info(
            "Best %s params: %s (Dev Accuracy: %.4f)", name, best_params, best_score
        )

        # Report Accuracy + Macro-F1 + confusion matrix.

        # Metrics (required)
        # - Primary: Accuracy
        # - Secondary: Macro-F1
        # - Also include: confusion matrix + 3–5 sentences interpreting it
        y_pred_dev = best_clf.predict(X_dev)
        report_stats(name, y_dev, y_pred_dev)

    # Evaluate on test once for the final numbers.
    for name, clf in best_models.items():
        logger.info("Evaluating %s on test set...", name)
        y_pred_test = clf.predict(X_test)
        report_stats(name + " (Test)", y_test, y_pred_test)

        # Collect ≥20 misclassified examples from test and categorize them into 3–5 error types.
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
    breakpoint()


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    train_tfidf_classifier(cfg)


if __name__ == "__main__":
    main()
