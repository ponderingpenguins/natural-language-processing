"""
Data loading and preprocessing for the AG News dataset.
"""

from datasets import DatasetDict, load_dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from .config import TrainingConfig


def load_data(cfg: TrainingConfig) -> tuple:
    """Load and preprocess the AG News dataset."""

    # Load AG News and create train/dev/test splits (dev from train).
    ds = load_dataset(cfg.hf_dataset)

    # Use the official train/test split.
    # Fix a random seed and report it.
    # Create a dev split from train (e.g., 90/10).
    # Keep the test set untouched until final reporting.
    # Even though AG News is balanced across classes, we use stratified sampling
    # to guarantee the dev set preserves the label distribution.
    full_train_ds = ds["train"]
    train_indices, dev_indices = train_test_split(
        range(len(full_train_ds)),
        test_size=cfg.dev_split,
        random_state=cfg.seed,
        stratify=full_train_ds["label"],
    )
    train_ds = full_train_ds.select(train_indices)
    dev_ds = full_train_ds.select(dev_indices)

    new_ds = DatasetDict(
        {
            "train": train_ds,
            "dev": dev_ds,
            "test": ds["test"],
        }
    )

    # Preprocess the dataset (e.g., tokenization, lowercasing).
    new_ds = preprocess_data(new_ds)

    return new_ds


def preprocess_data(ds: DatasetDict) -> DatasetDict:
    """Preprocess the dataset (e.g., tokenization, lowercasing)."""

    # combine title and description into a single text field
    def combine_title_description(example):
        example["text"] = example["title"] + " " + example["description"]
        return example

    ds = ds.map(combine_title_description)

    return ds
