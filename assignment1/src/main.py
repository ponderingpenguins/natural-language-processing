import logging
import string
import sys
from dataclasses import dataclass

import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# download stopwords if not already downloaded
nltk.download("stopwords", quiet=True)


# Configuration dataclass
@dataclass
class TrainingConfig:
    hf_dataset: str = "sh0416/ag_news"
    model: str = "logistic_regression"
    dev_split: float = (
        0.1  # percentage of training data to use as dev set (e.g., 0.1 for 10%)
    )
    max_features: int = 10000
    ngram_range: tuple = (1, 2)


# Preprocessing functions


def remove_whitespace(text: str) -> str:
    """Remove extra whitespace from the text."""

    return " ".join(text.split())


def remove_punctuation(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def apply_preprocessing_pipeline(text: str, pipeline: dict) -> str:
    """Apply a preprocessing pipeline to the text."""
    for name, func in pipeline.items():
        text = func(text)
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords from the text."""
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)


def preprocess_dataset(dataset, pipeline):
    """Preprocess the dataset using the provided pipeline."""
    return dataset.map(
        lambda x: {
            "title": apply_preprocessing_pipeline(x["title"], pipeline),
            "description": apply_preprocessing_pipeline(x["description"], pipeline),
        }
    )


def train_tfidf_classifier(cfg: TrainingConfig) -> None:
    """
    Train a TF-IDF classifier.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """

    # Load AG News and create train/dev/test splits (dev from train).
    ds = load_dataset(cfg.hf_dataset)

    # Use the official train/test split.
    # Fix a random seed and report it.
    # Create a dev split from train (e.g., 90/10). Keep the test set untouched until final reporting.
    train_ds = ds["train"]
    dev_size = int(len(train_ds) * cfg.dev_split)
    dev_ds = train_ds.select(range(dev_size))
    train_ds = train_ds.select(range(dev_size, len(train_ds)))

    # Implement preprocessing (tokenization, normalization) and document it.

    pipeline = {
        "lowercase": lambda x: x.lower(),  # Convert text to lowercase
        "remove_whitespace": remove_whitespace,  # Remove extra whitespace
        "remove_punctuation": remove_punctuation,  # Remove punctuation
        "remove_stopwords": remove_stopwords,  # Remove stopwords
    }

    # Preprocess the datasets using the defined pipeline.
    train_ds = preprocess_dataset(train_ds, pipeline)
    dev_ds = preprocess_dataset(dev_ds, pipeline)
    test_ds = preprocess_dataset(ds["test"], pipeline)

    # print examples from the preprocessed datasets to verify the preprocessing steps.
    logger.info("(Before preprocessing) Example from original training set:")
    logger.info(ds["train"][len(ds["train"]) - 1])
    logger.info("Example from preprocessed training set:")
    logger.info(train_ds[len(train_ds) - 1])

    # Use word-level TF-IDF features (document the preprocessing choices).
    # Train two classical models (required):
    #     TF-IDF + Logistic Regression
    #     TF-IDF + Linear SVM

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
