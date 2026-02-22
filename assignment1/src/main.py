"""
Main training script for TF-IDF classifiers.

This script loads the AG News dataset, preprocesses it,
trains two classical models (Logistic Regression and Linear SVM)
using TF-IDF features, evaluates them on a dev set for model selection,
and then evaluates the best models on the test set, reporting misclassified examples.
"""

import sys
from collections.abc import Callable
from typing import Any

from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore

from .utils.config import TrainingConfig
from .utils.data import load_and_preprocess_data
from .utils.evaluate_models import evaluate_models_on_test_set, save_misclassified
from .utils.helpers import (
    logger,
    plot_confusion_matrix,
    plot_prediction_map,
    plot_tfidf_clusters,
    plot_top_features,
    report_stats,
)


def train_tfidf_classifier(cfg: TrainingConfig) -> None:
    """
    Train a TF-IDF classifier.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """

    x_train, y_train, x_dev, y_dev, x_test, y_test, test_ds, tfidf = (
        load_and_preprocess_data(cfg)
    )

    # Visualise TF-IDF clusters (uses training data)
    plot_tfidf_clusters(
        x_train,
        y_train,
        label_names=cfg.label_names,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        file_name_prefix="dev",
    )

    # Train two classical models
    # 1. TF-IDF + Logistic Regression
    # 2. TF-IDF + Linear SVM

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
            max_iter=1000, random_state=cfg.seed, **params
        ),
        "Linear SVM": lambda params: LinearSVC(random_state=cfg.seed, **params),
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
            clf.fit(x_train, y_train)
            y_pred_dev = clf.predict(x_dev)

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

        if name == "Linear SVM":
            # plot top features for each class (requires coef_ attribute)
            feature_names = tfidf.get_feature_names_out().tolist()
            if hasattr(best_clf, "coef_"):
                plot_top_features(
                    best_clf,
                    feature_names,
                    label_names=cfg.label_names,
                    output_path=f"{cfg.output_dir}/{name}_top_features.png",
                )

        # Report Accuracy + Macro-F1 + confusion matrix.

        # Metrics (required)
        # - Primary: Accuracy
        # - Secondary: Macro-F1
        # - Also include: confusion matrix + 3–5 sentences interpreting it
        y_pred_dev = best_clf.predict(x_dev)
        report_stats(cfg, name, y_dev, y_pred_dev)

        plot_confusion_matrix(
            y_dev,
            y_pred_dev,
            label_names=cfg.label_names,
            title=f"Confusion Matrix for {name} (Dev Set)",
            output_path=f"{cfg.output_dir}/{name}_confusion_matrix.png",
        )

        # Also for the test set (only the best model)
        y_pred_test = best_clf.predict(x_test)
        report_stats(cfg, name, y_test, y_pred_test)

        plot_confusion_matrix(
            y_test,
            y_pred_test,
            label_names=cfg.label_names,
            title=f"Confusion Matrix for {name} (Test Set)",
            output_path=f"{cfg.output_dir}/{name}_confusion_matrix_test.png",
        )

        # Prediction map: colour = predicted class, shape = true class
        plot_prediction_map(
            x_data=x_test,
            y_true=y_test,
            y_pred=y_pred_test,
            label_names=cfg.label_names,
            title=f"Prediction Map for {name} (Test Set, t-SNE)",
            output_path=f"{cfg.output_dir}/{name}_prediction_map.png",
            seed=cfg.seed,
        )

    # Collect more than 20 misclassified examples from test
    # and categorize them into 3–5 error types.
    results = evaluate_models_on_test_set(cfg, best_models, x_test, y_test)
    save_misclassified(cfg, results, y_test, test_ds)

    # Visualise TF-IDF clusters (uses test data)
    plot_tfidf_clusters(
        x_test,
        y_test,
        label_names=cfg.label_names,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        file_name_prefix="test",
    )


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)  # type: ignore
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    train_tfidf_classifier(cfg)


if __name__ == "__main__":
    main()
