import logging
import sys
from dataclasses import dataclass

from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from .utils.preprocessing import preprocess_dataset, text_preprocessing_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Configuration dataclass
@dataclass
class TrainingConfig:
    """
    Configuration for training the TF-IDF classifier.

    Attributes:
        hf_dataset: The Hugging Face dataset to use (default: "sh0416/ag_news").
        model: The model type to train (default: "logistic_regression").
        dev_split: The percentage of training data to use as a dev set (default: 0.1 for 10%).
        max_features: The maximum number of features for TF-IDF vectorization (default: 10000).
        ngram_range: The n-gram range for TF-IDF vectorization (default: (1, 2) for unigrams and bigrams).

    """

    hf_dataset: str = "sh0416/ag_news"
    model: str = "logistic_regression"
    dev_split: float = (
        0.1  # percentage of training data to use as dev set (e.g., 0.1 for 10%)
    )
    max_features: int = 10000
    ngram_range: tuple = (1, 2)


def load_and_preprocess_data(cfg: TrainingConfig) -> tuple:
    """Load and preprocess the AG News dataset."""

    # Load AG News and create train/dev/test splits (dev from train).
    ds = load_dataset(cfg.hf_dataset)

    # Use the official train/test split.
    # Fix a random seed and report it.
    # Create a dev split from train (e.g., 90/10).
    # Keep the test set untouched until final reporting.
    train_ds = ds["train"]
    dev_size = int(len(train_ds) * cfg.dev_split)
    dev_ds = train_ds.select(range(dev_size))
    train_ds = train_ds.select(range(dev_size, len(train_ds)))

    # Implement preprocessing (tokenization, normalization) and document it.

    # Preprocess the datasets using the defined pipeline.
    train_ds = preprocess_dataset(train_ds, text_preprocessing_pipeline)
    dev_ds = preprocess_dataset(dev_ds, text_preprocessing_pipeline)
    test_ds = preprocess_dataset(ds["test"], text_preprocessing_pipeline)

    # print examples from the preprocessed datasets to verify the preprocessing steps.
    logger.info("(Before preprocessing) Example from original training set:")
    logger.info(ds["train"][len(ds["train"]) - 1])
    logger.info("Example from preprocessed training set:")
    logger.info(train_ds[len(train_ds) - 1])

    # Use word-level TF-IDF features (document the preprocessing choices).

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

    X_train = tfidf.fit_transform(train_ds["text"])
    y_train = train_ds["label"]
    X_dev = tfidf.transform(dev_ds["text"])
    y_dev = dev_ds["label"]
    X_test = tfidf.transform(test_ds["text"])
    y_test = test_ds["label"]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


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

    # TF-IDF + Logistic Regression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_dev = clf.predict(X_dev)

    logger.info("Logistic Regression, Dev Set Performance:")
    logger.info(classification_report(y_dev, y_pred_dev))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_dev, y_pred_dev))

    # TF-IDF + Linear SVM

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred_dev = clf.predict(X_dev)
    logger.info("Linear SVM, Dev Set Performance:")
    logger.info(classification_report(y_dev, y_pred_dev))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_dev, y_pred_dev))

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
