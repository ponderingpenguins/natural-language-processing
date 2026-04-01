"""Dataset loading and preprocessing utilities for the AG News dataset."""

import html
from typing import Any, cast

from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore
from penguinlp.helpers import logger
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


def load_data(cfg: dict) -> DatasetDict:
    """Load the AG News dataset and create train/dev/test splits."""
    dataset = load_dataset(cfg.hf_dataset)
    # Even though AG News is balanced across classes, we use stratified sampling
    # to guarantee the dev set preserves the label distribution.
    full_train_ds = dataset["train"]
    train_indices, dev_indices = train_test_split(
        range(len(full_train_ds)),
        test_size=cfg.dev_split,
        random_state=cfg.seed,
        stratify=full_train_ds["label"],
    )
    train_ds = full_train_ds.select(train_indices)
    dev_ds = full_train_ds.select(dev_indices)
    test_ds = dataset["test"]

    # subsample for quick testing useing dataset.max_samples if set
    if cfg.max_samples is not None:
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(cfg.max_samples))
        dev_ds = dev_ds.shuffle(seed=cfg.seed).select(range(cfg.max_samples))
        test_ds = test_ds.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

    return DatasetDict({"train": train_ds, "dev": dev_ds, "test": test_ds})


def preprocess_data(dataset: DatasetDict) -> DatasetDict:
    """Preprocess the dataset by combining title and description, unescaping HTML entities, and normalizing whitespace."""
    dataset["train"] = dataset["train"].map(
        _preprocess_sample, batched=False, load_from_cache_file=False
    )
    dataset["dev"] = dataset["dev"].map(
        _preprocess_sample, batched=False, load_from_cache_file=False
    )
    dataset["test"] = dataset["test"].map(
        _preprocess_sample, batched=False, load_from_cache_file=False
    )

    # Check if labels are 1-based and convert to 0-based if needed
    all_labels = []
    for split in ["train", "dev", "test"]:
        all_labels.extend(int(label) for label in dataset[split]["label"])

    if all_labels and min(all_labels) == 1 and max(all_labels) > 1:
        logger.info(
            "Detected 1-based labels (min=%d, max=%d); converting to 0-based indices",
            min(all_labels),
            max(all_labels),
        )

        def _to_zero_based(example):
            example["label"] = int(example["label"]) - 1
            return example

        dataset["train"] = dataset["train"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
        dataset["dev"] = dataset["dev"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
        dataset["test"] = dataset["test"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )

    # Rename 'label' column to 'labels' for Hugging Face Trainer compatibility
    for split in ["train", "dev", "test"]:
        if "labels" not in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("label", "labels")

    return dataset


def _preprocess_sample(sample: dict) -> dict:
    """Preprocess a single sample if needed."""
    text = sample["title"] + " " + sample["description"]

    # html unescaping
    text = html.unescape(text)

    # Handle malformed HTML entities (missing leading &)
    text = text.replace("#36;", "$")  # &#36; = $
    text = text.replace("#38;", "&")  # &#38; = &
    text = text.replace("#39;", "'")  # &#39; = '
    text = text.replace("#34;", '"')  # &#34; = "
    text = text.replace("#35;", "#")  # &#35; = #
    text = text.replace("#37;", "%")  # &#37; = %

    # latex escape sequences - handle specific ones first
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    text = text.replace("\\&", "&")
    text = text.replace("\\%", "%")
    text = text.replace("\\$", "$")
    text = text.replace("\\#", "#")
    text = text.replace("\\_", "_")
    text = text.replace("\\{", "{")
    text = text.replace("\\}", "}")
    text = text.replace("\\~", "~")
    text = text.replace("\\^", "^")

    # Replace double backslash with single backslash
    text = text.replace("\\\\", "\\")

    # Replace remaining single backslashes (line continuations) with space
    text = text.replace("\\", " ")

    # Normalize whitespace
    text = " ".join(text.split())

    sample["text"] = text
    return sample


def dataset_prep(cfg: dict) -> DatasetDict:
    """Load and preprocess the dataset."""
    data = load_data(cfg)
    logger.info("Loaded dataset %s with splits: %s", cfg.hf_dataset, data.keys())
    # Preprocess the datasets using the defined pipeline.
    logger.info("Preprocessing splits...")
    data = preprocess_data(data)
    logger.info(
        "Dataset preprocessing complete. Sample from preprocessed dataset: %s",
        data["train"][0],
    )
    return data


def tokenize_data(data: DatasetDict, tokenization) -> DatasetDict:
    """Tokenize the datasets using the model's tokenizer."""
    # Tokenize the datasets using the model's tokenizer.
    for split_name in ["train", "dev", "test"]:
        logger.info("Tokenizing %s set...", split_name)
        data[split_name] = data[split_name].map(
            tokenization, batched=False, load_from_cache_file=False
        )
        logger.info(
            "Completed tokenization for %s set. Set size: %d",
            split_name,
            len(data[split_name]),
        )

    logger.info("Tokenization complete.")
    return data


def try_load_tokenized_data(tokenized_data_path, data, tokenization):
    """Try to load tokenized data from disk, if it fails, tokenize and save to disk for future runs."""
    try:
        logger.info("Attempting to load tokenized data from %s...", tokenized_data_path)
        data = load_from_disk(tokenized_data_path)
    except Exception as e:
        logger.warning(
            "Failed to load tokenized data from disk: %s. Tokenizing now...", e
        )
        data = tokenize_data(data, tokenization)
        # save to disk for future runs
        data.save_to_disk(tokenized_data_path)
        logger.info("Successfully loaded tokenized data from disk.")
    return data
