import sys

from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    for name, clf in models.items():
        logger.info(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred_dev = clf.predict(X_dev)

        report_stats(name, y_dev, y_pred_dev)

    # Train both baseline models. Keep the dev split for model selection/tuning.

    # Report Accuracy + Macro-F1 + confusion matrix.

    # Metrics (required)
    # - Primary: Accuracy
    # - Secondary: Macro-F1
    # - Also include: confusion matrix + 3–5 sentences interpreting it

    # Evaluate on test once for the final numbers.

    # Collect ≥20 misclassified examples from test and categorize them into 3–5 error types.
    # Collect ≥20 misclassified examples from test and categorize them into 3–5 error types.
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
