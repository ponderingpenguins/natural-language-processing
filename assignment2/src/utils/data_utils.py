"""Dataset and DataLoader utilities."""

import hashlib
import html
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import torch
from penguinlp.helpers import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.tokanizer import build_tokenizer, load_tokenizer, save_tokenizer


def get_tokenizer_vocab_size(tokenizer: Any) -> int:
    """Return embedding-safe vocabulary size for a tokenizer."""
    vocab = getattr(tokenizer, "vocab", {})
    if not vocab:
        return 0
    values = [int(idx) for idx in vocab.values()]
    return max(values) + 1


def clear_cache_dirs(cfg: Any) -> None:
    """Clear tokenizer and tokenized dataset cache directories."""
    dirs = [
        Path(getattr(cfg, "tokenizer_cache_dir", "output/tokenizer_cache")),
        Path(getattr(cfg, "tokenized_cache_dir", "output/tokenized_cache")),
    ]
    for cache_dir in dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Cleared cache directory: %s", cache_dir)


def preprocess_sample(sample):
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


def preprocess_data(data):
    """Preprocess raw data if needed."""
    # Assuming data is already in the form of list of dicts with 'text' and 'label'

    data["train"] = data["train"].map(
        preprocess_sample, batched=False, load_from_cache_file=False
    )
    data["dev"] = data["dev"].map(
        preprocess_sample, batched=False, load_from_cache_file=False
    )
    data["test"] = data["test"].map(
        preprocess_sample, batched=False, load_from_cache_file=False
    )

    all_labels = []
    for split in ["train", "dev", "test"]:
        all_labels.extend(int(label) for label in data[split]["label"])

    if all_labels and min(all_labels) == 1 and max(all_labels) > 1:
        logger.info(
            "Detected 1-based labels (min=%d, max=%d); converting to 0-based indices",
            min(all_labels),
            max(all_labels),
        )

        def _to_zero_based(example):
            example["label"] = int(example["label"]) - 1
            return example

        data["train"] = data["train"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
        data["dev"] = data["dev"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )
        data["test"] = data["test"].map(
            _to_zero_based, batched=False, load_from_cache_file=False
        )

    return data


class TokenizedDataset(Dataset):
    """A PyTorch Dataset that tokenizes text examples on the fly."""

    def __init__(self, examples: list[tuple[list[int], int]], max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, label = self.examples[idx]
        # Truncate or pad to max_seq_length
        input_ids = tokens[: self.max_seq_length]
        real_length = len(input_ids)  # Track real length before padding
        if len(input_ids) < self.max_seq_length:
            input_ids.extend([0] * (self.max_seq_length - len(input_ids)))
        return {"input_ids": input_ids, "labels": label, "length": real_length}


def collate_fn(batch: list, vocab: dict[str, int]) -> dict:
    """Collate function to convert a list of samples into a batch.

    Returns:
        Dictionary with keys 'x', 'lengths', and 'y'.
    """
    # batch: list of dicts with 'input_ids', 'labels', and 'length' keys
    pad = vocab.get("<PAD>", 0)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(batch) > 0 else 0
    x = torch.full((len(batch), max_len), pad, dtype=torch.long)
    y = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        actual_len = min(len(ids), max_len)
        x[i, :actual_len] = torch.tensor(ids[:actual_len], dtype=torch.long)
    return {"x": x, "lengths": lengths, "y": y}


def _build_cache_key(cfg: Any, tokenizer: Any, split_name: str, split_len: int) -> str:
    """Build a stable cache key for tokenized examples."""
    tokenizer_path = str(getattr(cfg, "tokenizer_path", "tokenizer.pkl"))
    vocab_items = sorted(getattr(tokenizer, "vocab", {}).items())
    vocab_fingerprint = hashlib.sha256(str(vocab_items).encode("utf-8")).hexdigest()[
        :16
    ]
    key_parts = [
        "tokenized_schema_v3",
        str(getattr(cfg, "hf_dataset", "unknown_dataset")),
        split_name,
        str(split_len),
        str(getattr(cfg, "seed", "no_seed")),
        str(getattr(cfg, "sample_size", "full")),
        str(getattr(cfg, "max_seq_length", "no_max_seq_length")),
        str(getattr(cfg, "vocab_size", "no_vocab_size")),
        str(getattr(cfg, "min_freq", "no_min_freq")),
        str(getattr(cfg, "tokenizer_type", "no_tokenizer_type")),
        tokenizer_path,
        str(get_tokenizer_vocab_size(tokenizer)),
        vocab_fingerprint,
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_cache_file_path(
    cfg: Any, tokenizer: Any, split_name: str, split_len: int
) -> Path:
    """Get cache file path for a tokenized split."""
    cache_dir = Path(getattr(cfg, "tokenized_cache_dir", "output/tokenized_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_cache_key(cfg, tokenizer, split_name, split_len)
    return cache_dir / f"{split_name}_{cache_key}.pkl"


def _tokenize_split_examples(
    data_split, tokenizer, split_name: str
) -> list[tuple[list[int], int]]:
    """Tokenize one split into (token_ids, label) examples."""
    examples: list[tuple[list[int], int]] = []
    for ex in tqdm(data_split, desc=f"Tokenizing {split_name}", leave=False):
        examples.append((tokenizer(ex["text"]), int(ex["label"])))
    return examples


def _load_or_build_tokenized_examples(data_split, tokenizer, cfg: Any, split_name: str):
    """Load tokenized examples from cache, or build and save them."""
    split_len = len(data_split)
    cache_path = _get_cache_file_path(cfg, tokenizer, split_name, split_len)

    if cache_path.exists():
        logger.info("Loading tokenized %s split from cache: %s", split_name, cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Building tokenized %s split cache...", split_name)
    examples = _tokenize_split_examples(data_split, tokenizer, split_name)
    with open(cache_path, "wb") as f:
        pickle.dump(examples, f)
    logger.info("Saved tokenized %s cache to %s", split_name, cache_path)
    return examples


def _build_tokenizer_cache_key(cfg: Any, train_len: int) -> str:
    """Build a stable cache key for tokenizer artifacts."""
    key_parts = [
        "tokenizer_schema_v2",
        str(getattr(cfg, "hf_dataset", "unknown_dataset")),
        str(getattr(cfg, "seed", "no_seed")),
        str(getattr(cfg, "sample_size", "full")),
        str(train_len),
        str(getattr(cfg, "tokenizer_type", "no_tokenizer_type")),
        str(getattr(cfg, "vocab_size", "no_vocab_size")),
        str(getattr(cfg, "min_freq", "no_min_freq")),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _get_tokenizer_cache_path(cfg: Any, train_len: int) -> Path:
    """Get a config-aware tokenizer cache file path."""
    cache_dir = Path(getattr(cfg, "tokenizer_cache_dir", "output/tokenizer_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_tokenizer_cache_key(cfg, train_len)
    return cache_dir / f"tokenizer_{cache_key}.pkl"


def create_dataloaders(
    data: dict, tokenizer: object, cfg, include_test: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders for train, validation, and test sets.

    Args:
        data: Dictionary with 'train', 'dev', and 'test' datasets.
        tokenizer: Tokenizer object.
        cfg: Training configuration object.
        include_test: Whether to build the test DataLoader.

    Returns:
        Tuple of (train_loader, val_loader, test_loader). test_loader is None
        when include_test is False.
    """
    train_examples = _load_or_build_tokenized_examples(
        data["train"], tokenizer, cfg, "train"
    )
    val_examples = _load_or_build_tokenized_examples(data["dev"], tokenizer, cfg, "dev")
    train_ds = TokenizedDataset(train_examples, cfg.max_seq_length)
    val_ds = TokenizedDataset(val_examples, cfg.max_seq_length)
    test_ds = (
        TokenizedDataset(
            _load_or_build_tokenized_examples(data["test"], tokenizer, cfg, "test"),
            cfg.max_seq_length,
        )
        if include_test
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
        )
        if test_ds is not None
        else None
    )

    return train_loader, val_loader, test_loader


def compute_tokenizer_coverage(tokenizer, data: dict, output_dir: str) -> dict:
    """Measure tokenizer coverage on train, dev, and test splits.

    Coverage is measured on train (sanity check), dev, and test splits.
    The test split is the most meaningful: tokens were never seen during
    tokenizer fitting, so it reflects real-world OOV behaviour.

    Three complementary metrics are reported per split:
      - token_oov_rate:    fraction of token *instances* that are <UNK>
      - type_oov_rate:     fraction of unique token *types* not in vocab
      - document_oov_rate: fraction of documents containing >= 1 <UNK> token

    Args:
        tokenizer: Fitted tokenizer with .tokenize() and .vocab attributes.
        data: Dict with 'train', 'dev', 'test' HuggingFace Dataset splits.
        output_dir: Directory to write tokenizer_coverage.json.

    Returns:
        Dict with per-split coverage statistics.
    """
    vocab = tokenizer.vocab
    results: dict = {}

    # Text length bucket boundaries (in whitespace-split words)
    BUCKET_BOUNDARIES = [20, 50]  # short: <20, medium: 20-50, long: >50

    for split_name, split_data in data.items():
        total_tokens = 0
        oov_tokens = 0
        oov_docs = 0
        total_words = 0
        all_token_types: set = set()
        oov_types: set = set()
        all_word_types: set = set()  # for TTR preservation
        all_tok_types_ttr: set = set()  # tokenized types for TTR

        # Bucket accumulators: {bucket_name: [total_tokens, oov_tokens, n_docs]}
        buckets: dict = {
            "short": [0, 0, 0],
            "medium": [0, 0, 0],
            "long": [0, 0, 0],
        }

        for example in tqdm(split_data, desc=f"Coverage ({split_name})", leave=False):
            text = example["text"]
            tokens = tokenizer.tokenize(text)
            words = text.split()
            n_words = len(words)
            n_tokens = len(tokens)

            total_tokens += n_tokens
            total_words += n_words
            all_word_types.update(words)
            all_tok_types_ttr.update(tokens)

            example_has_oov = False
            for tok in tokens:
                all_token_types.add(tok)
                if tok not in vocab:
                    oov_tokens += 1
                    oov_types.add(tok)
                    example_has_oov = True
            if example_has_oov:
                oov_docs += 1

            # Assign to length bucket
            if n_words < BUCKET_BOUNDARIES[0]:
                bucket = "short"
            elif n_words <= BUCKET_BOUNDARIES[1]:
                bucket = "medium"
            else:
                bucket = "long"
            buckets[bucket][0] += n_tokens
            buckets[bucket][1] += sum(1 for tok in tokens if tok not in vocab)
            buckets[bucket][2] += 1

        n_docs = len(split_data)

        # TTR: type-token ratio of original words vs. BPE tokens
        word_ttr = round(len(all_word_types) / max(total_words, 1), 6)
        token_ttr = round(len(all_tok_types_ttr) / max(total_tokens, 1), 6)

        results[split_name] = {
            "total_tokens": total_tokens,
            "oov_tokens": oov_tokens,
            "token_oov_rate": round(oov_tokens / max(total_tokens, 1), 6),
            "total_types": len(all_token_types),
            "oov_types": len(oov_types),
            "type_oov_rate": round(len(oov_types) / max(len(all_token_types), 1), 6),
            "total_documents": n_docs,
            "oov_documents": oov_docs,
            "document_oov_rate": round(oov_docs / max(n_docs, 1), 6),
            # Fragmentation
            "total_words": total_words,
            "tokens_per_word": round(total_tokens / max(total_words, 1), 4),
            # TTR preservation
            "word_ttr": word_ttr,
            "token_ttr": token_ttr,
            "ttr_ratio": round(token_ttr / max(word_ttr, 1e-9), 6),
            # OOV by document length bucket
            "oov_by_length_bucket": {
                bucket: {
                    "total_documents": int(v[2]),
                    "total_tokens": int(v[0]),
                    "oov_tokens": int(v[1]),
                    "token_oov_rate": round(v[1] / max(v[0], 1), 6),
                }
                for bucket, v in buckets.items()
            },
        }
        logger.info(
            "Coverage [%s]: token_oov=%.2f%%, type_oov=%.2f%%, doc_oov=%.2f%%, "
            "tokens/word=%.2f, ttr_ratio=%.3f",
            split_name,
            results[split_name]["token_oov_rate"] * 100,
            results[split_name]["type_oov_rate"] * 100,
            results[split_name]["document_oov_rate"] * 100,
            results[split_name]["tokens_per_word"],
            results[split_name]["ttr_ratio"],
        )

    summary = {
        "vocab_size": len(vocab),
        "tokenizer_type": type(tokenizer).__name__,
        "splits": results,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "tokenizer_coverage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved tokenizer coverage report to %s", out_path)
    return summary


def setup_tokenizer(cfg, train_data) -> object:
    """Setup and return tokenizer, either by loading or building new one.

    Args:
        cfg: Training configuration.
        train_data: Training dataset for building tokenizer if needed.

    Returns:
        Tokenizer object.
    """
    train_len = len(train_data)
    tokenizer_cache_path = _get_tokenizer_cache_path(cfg, train_len)

    try:
        tokenizer = load_tokenizer(tokenizer_cache_path)
        logger.info(
            "Tokenizer loaded successfully from cache: %s", tokenizer_cache_path
        )
    except FileNotFoundError:
        logger.warning("Tokenizer cache miss. Building a new tokenizer.")
        tokenizer = build_tokenizer(
            train_data,
            tokenizer_type=cfg.tokenizer_type,
            vocab_size=cfg.vocab_size,
            min_freq=cfg.min_freq,
        )
        logger.info("Vocabulary built with %d tokens", len(tokenizer.vocab))
        save_tokenizer(tokenizer, tokenizer_cache_path)

    return tokenizer
    return tokenizer
