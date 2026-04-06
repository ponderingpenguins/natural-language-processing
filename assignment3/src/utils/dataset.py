"""Dataset loading and preprocessing utilities for the AG News dataset."""

import hashlib
import html
import json
from pathlib import Path

from datasets import (  # type: ignore
    ClassLabel,
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)
from penguinlp.helpers import logger
from sklearn.model_selection import train_test_split  # type: ignore


def load_data(cfg: dict) -> DatasetDict:
    """Load the AG News dataset and create train/dev/test splits."""
    try:
        dataset = load_dataset(cfg.hf_dataset)
    except Exception as exc:
        logger.warning("Primary dataset load failed for %s: %s", cfg.hf_dataset, exc)
        dataset = _load_cached_ag_news(cfg.hf_dataset)

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

    # Normalize 1-based labels before casting to ClassLabel (expects 0..num_classes-1).
    all_labels = [int(label) for label in train_ds["label"]]
    all_labels.extend(int(label) for label in dev_ds["label"])
    all_labels.extend(int(label) for label in test_ds["label"])
    num_classes = len(cfg.label_mapping)

    if all_labels and min(all_labels) == 1 and max(all_labels) == num_classes:
        logger.info(
            "Detected 1-based labels (min=%d, max=%d); converting to 0-based before cast",
            min(all_labels),
            max(all_labels),
        )

        def _to_zero_based(example):
            example["label"] = int(example["label"]) - 1
            return example

        train_ds = train_ds.map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
        dev_ds = dev_ds.map(_to_zero_based, batched=False, load_from_cache_file=False)
        test_ds = test_ds.map(_to_zero_based, batched=False, load_from_cache_file=False)

    # Convert labels to ClassLabel type for better compatibility with Hugging Face Trainer and metrics
    names = [cfg.label_mapping[k] for k in sorted(cfg.label_mapping)]
    train_ds = train_ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(cfg.label_mapping),
            names=names,
        ),
    )
    dev_ds = dev_ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(cfg.label_mapping),
            names=names,
        ),
    )
    test_ds = test_ds.cast_column(
        "label",
        ClassLabel(
            num_classes=len(cfg.label_mapping),
            names=names,
        ),
    )

    # Optionally perform stratified subsampling for faster experimentation during development.
    eval_max_samples = getattr(cfg, "eval_max_samples", None)

    if cfg.max_samples and len(train_ds) > cfg.max_samples:
        logger.info(
            "Subsampling the training set to %d examples for faster experimentation.",
            cfg.max_samples,
        )
        train_ds = train_ds.train_test_split(
            test_size=cfg.max_samples, stratify_by_column="label", seed=cfg.seed
        )["test"]

    if eval_max_samples and len(dev_ds) > eval_max_samples:
        logger.info(
            "Subsampling the dev set to %d examples for faster evaluation.",
            eval_max_samples,
        )
        dev_ds = dev_ds.train_test_split(
            test_size=eval_max_samples, stratify_by_column="label", seed=cfg.seed
        )["test"]

    if eval_max_samples and len(test_ds) > eval_max_samples:
        logger.info(
            "Subsampling the test set to %d examples for faster evaluation.",
            eval_max_samples,
        )
        test_ds = test_ds.train_test_split(
            test_size=eval_max_samples, stratify_by_column="label", seed=cfg.seed
        )["test"]

    return DatasetDict({"train": train_ds, "dev": dev_ds, "test": test_ds})


def _load_cached_ag_news(hf_dataset: str) -> DatasetDict:
    """Fallback loader for locally cached AG News Arrow files.

    TODO: Maybe clean this up a bit :)"""
    if hf_dataset != "sh0416/ag_news":
        raise RuntimeError(
            f"No offline fallback is configured for dataset {hf_dataset!r}."
        )

    cache_root = Path(
        "/home/ubuntu/.cache/huggingface/datasets/"
        "sh0416___ag_news/default/0.0.0/70e3fa1915be9a8daebec5e840f20df9a8e18793"
    )
    train_path = cache_root / "ag_news-train.arrow"
    test_path = cache_root / "ag_news-test.arrow"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Offline AG News cache not found. Expected arrow files at "
            f"{train_path} and {test_path}."
        )

    logger.info("Loading AG News from offline cached arrow files in %s", cache_root)
    return DatasetDict(
        {
            "train": Dataset.from_file(str(train_path), in_memory=True),
            "test": Dataset.from_file(str(test_path), in_memory=True),
        }
    )


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
    # text = sample["title"] + " " + sample["description"] # Old method (assignment 1 and 2)

    # Combine title and description with a separator for better readability, and to allow the model to learn from both fields. We do this because the the title and description often don't have a clear boundary.
    text = "Title: " + sample["title"] + " Content: " + sample["description"]

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


def get_config_hash(config_dict):
    """Generate hash of config to detect changes."""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def try_load_tokenized_data(tokenized_data_path, data, tokenization, config):
    """Try to load tokenized data from disk with config hash validation."""
    cache_dir = Path(tokenized_data_path)
    hash_file = cache_dir / ".config_hash"
    current_hash = get_config_hash(config)

    # Check if cache exists and hash matches
    if cache_dir.exists() and hash_file.exists():
        cached_hash = hash_file.read_text().strip()
        if cached_hash == current_hash:
            try:
                logger.info(
                    "Loading tokenized data from %s (hash: %s)...",
                    tokenized_data_path,
                    current_hash,
                )
                data = load_from_disk(tokenized_data_path)
                return data
            except Exception as e:
                logger.warning("Failed to load cached data: %s. Retokenizing...", e)
        else:
            logger.info(
                "Config changed (old: %s, new: %s). Retokenizing...",
                cached_hash,
                current_hash,
            )

    # Tokenize and save with hash
    logger.info("Tokenizing data...")
    data = tokenize_data(data, tokenization)
    data.save_to_disk(tokenized_data_path)
    hash_file.write_text(current_hash)
    logger.info("Saved tokenized data with config hash: %s", current_hash)

    return data
