from datasets import load_dataset
import html
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import AutoTokenizer
from penguinlp.src.penguinlp.config import TrainingConfig

from penguinlp.src.penguinlp.helpers import logger

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

    return train_ds, dev_ds, test_ds

def preprocess_data(dataset):
    preprocessed_dataset = dataset.map(_preprocess_sample, batched=False, load_from_cache_file=False)
    
    # Check if labels are 1-based and convert to 0-based if needed
    all_labels = []
    all_labels.extend(int(label) for label in dataset["label"])

    if all_labels and min(all_labels) == 1 and max(all_labels) > 1:
        logger.info(
            "Detected 1-based labels (min=%d, max=%d); converting to 0-based indices",
            min(all_labels),
            max(all_labels),
        )

        def _to_zero_based(example):
            example["label"] = int(example["label"]) - 1
            return example

        dataset = dataset.map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
    return preprocessed_dataset

def _preprocess_sample(sample):
    """Preprocess a single sample if needed."""
    text = sample["text"]

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
