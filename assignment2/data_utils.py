"""Dataset and DataLoader utilities."""

import torch
from torch.utils.data import DataLoader, Dataset


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
