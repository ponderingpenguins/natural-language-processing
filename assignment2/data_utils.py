"""Dataset and DataLoader utilities."""

import html

import torch
from torch.utils.data import DataLoader, Dataset


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

    return data


class Batch:
    """A simple Batch class to hold batch data."""

    def __init__(self, x: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.lengths = lengths
        self.y = y


class TokenizedDataset(Dataset):
    """A PyTorch Dataset that tokenizes text examples on the fly."""

    def __init__(self, data, tokenizer, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = self.tokenizer(example["text"])
        # Truncate or pad to max_seq_len
        input_ids = tokens[: self.max_seq_len]
        if len(input_ids) < self.max_seq_len:
            input_ids.extend([0] * (self.max_seq_len - len(input_ids)))
        return {"input_ids": input_ids, "labels": example["label"]}


def collate_fn(batch: list, vocab: dict[str, int]) -> Batch:
    """Collate function to convert a list of samples into a batch."""
    # batch: list of dicts with 'input_ids' and 'labels' keys
    pad = vocab.get("<PAD>", 0)
    lengths = torch.tensor([len(item["input_ids"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(batch) > 0 else 0
    x = torch.full((len(batch), max_len), pad, dtype=torch.long)
    y = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return Batch(x=x, lengths=lengths, y=y)


def create_dataloaders(
    data: dict, tokenizer: object, cfg
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test sets.

    Args:
        data: Dictionary with 'train', 'dev', and 'test' datasets.
        tokenizer: Tokenizer object.
        cfg: Training configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = TokenizedDataset(data["train"], tokenizer, cfg.max_seq_len)
    val_ds = TokenizedDataset(data["dev"], tokenizer, cfg.max_seq_len)
    test_ds = TokenizedDataset(data["test"], tokenizer, cfg.max_seq_len)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.vocab),
    )

    return train_loader, val_loader, test_loader
