"""
Data loading and preprocessing for the AG News dataset.
"""

from datasets import load_dataset  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from .config import TrainingConfig
from .helpers import logger
from .preprocessing import preprocess_dataset, text_preprocessing_pipeline


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
    # Each dataset now has both 'raw_text' and 'text' fields.
    train_ds = preprocess_dataset(train_ds, text_preprocessing_pipeline)
    dev_ds = preprocess_dataset(dev_ds, text_preprocessing_pipeline)
    test_ds = preprocess_dataset(ds["test"], text_preprocessing_pipeline)

    # print examples from the preprocessed datasets to verify the preprocessing steps.
    logger.info("(Before preprocessing) Example from original training set:")
    logger.info(ds["train"][len(ds["train"]) - 1])
    logger.info("Example from preprocessed training set:")
    logger.info(train_ds[len(train_ds) - 1])

    # Use word-level TF-IDF features (document the preprocessing choices).
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(
            cfg.ngram_min,
            cfg.ngram_max,
        ),  # unigrams and bigrams
        max_features=cfg.max_features,  # increase since bigrams add features
        min_df=cfg.min_df,  # filter rare n-grams
    )

    X_train = tfidf.fit_transform(train_ds["text"])
    y_train = train_ds["label"]
    X_dev = tfidf.transform(dev_ds["text"])
    y_dev = dev_ds["label"]
    X_test = tfidf.transform(test_ds["text"])
    y_test = test_ds["label"]

    return X_train, y_train, X_dev, y_dev, X_test, y_test, test_ds
