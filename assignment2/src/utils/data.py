"""
Data loading and preprocessing for the AG News dataset.
"""

from datasets import load_dataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import html
from typing import Callable

from .config import TrainingConfig
from .helpers import logger
from .tokenizers import word_level_tokenizer

def load_data(cfg: TrainingConfig):
    """Load the AG News dataset and create train/dev/test splits."""
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
    test_ds = ds["test"]

    return train_ds, dev_ds, test_ds


def preprocess_text(sample: dict) -> dict:
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

    text = text.lower()
    
    sample["text"] = text
    return sample


def tokenize(text: str, tokenizer: Callable) -> list[str]:
    """Tokenize text using the provided tokenizer function."""
    PAD = "<pad>"
    UNK = "<unk>"
    vocab = {PAD: 0, UNK: 1}

    return tokenizer(text)

def preprocess_data(train_ds, dev_ds, test_ds, cfg):
    """Preprocess the datasets"""
    