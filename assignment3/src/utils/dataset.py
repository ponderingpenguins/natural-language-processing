import os
from datasets import load_dataset
import html
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader, Dataset as TorchDataset
from typing import Any, cast


from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger

def load_data(cfg: TrainingConfig):
    """Load the AG News dataset and create train/dev/test splits."""
    cfg = TrainingConfig()
    dataset = load_dataset(cfg.hf_dataset)
    # Use the official train/test split.
    # Fix a random seed and report it.
    # Create a dev split from train (e.g., 90/10).
    # Keep the test set untouched until final reporting.
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
    
    split_dataset = {
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds,
    }
    return split_dataset

def preprocess_data(dataset):
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
    return dataset

def _preprocess_sample(sample):
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

# Tokenization using dataloader.
# TODO: Debug it to avoid getting stuck. 
def tokenize_data(data, tokenize_fn):
    for split_name in ["train", "dev", "test"]:
        logger.info("Tokenizing %s set...", split_name)

        data_loader = DataLoader(
            cast(TorchDataset[Any], data[split_name]),
            batch_size=512,
            num_workers=0,
        )

        tokenized_columns = {}
        for batch in data_loader:
            tokenized_batch = cast(
                dict[str, list[Any]],
                tokenize_fn({"text": list(batch["text"])}),
            )

            if not tokenized_columns:
                tokenized_columns = {name: [] for name in tokenized_batch.keys()}

            for name, values in tokenized_batch.items():
                tokenized_columns[name].extend(values)

        for name, values in tokenized_columns.items():
            if name in data[split_name].column_names:
                data[split_name] = data[split_name].remove_columns(name)
            data[split_name] = data[split_name].add_column(name, values)

        logger.info("Tokenized %s set: %d samples", split_name, len(data[split_name]))
        
    return data
