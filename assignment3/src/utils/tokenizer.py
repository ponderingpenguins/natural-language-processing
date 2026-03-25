"""
Tokenizer module for building a vocabulary from text data.
Copied from Assignment 2 and modified to only to support BPE Tokenizer.
"""
import pickle
from pathlib import Path
from typing import Literal
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class BPETokenizer():
    def __init__(self, min_freq: int = 2, vocab_size: int = 20000):
        self.min_freq = min_freq
        self.vocab_size = vocab_size

        # Hugging Face tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    def fit(self, texts: list[dict], text_field: str = "text") -> None:
        raw_texts = [doc[text_field] for doc in texts]

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=SPECIAL_TOKENS,
        )

        # train from iterator, directly from dataset (no temp file needed)
        self.tokenizer.train_from_iterator(raw_texts, trainer)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()


def build_tokenizer(
    texts: list[dict],
    min_freq: int = 2,
    vocab_size: int = 20000,
    text_field: str = "text",
) -> BPETokenizer:
    """Instantiate, fit, and return the appropriate tokenizer."""
    cls_map: dict[str, type[BPETokenizer]] = {
        "bpe": BPETokenizer,
    }
    tokenizer = cls_map["bpe"](min_freq=min_freq, vocab_size=vocab_size)
    tokenizer.fit(texts, text_field=text_field)
    return tokenizer


def save_tokenizer(tokenizer: BPETokenizer, filepath: str | Path) -> None:
    """Save a tokenizer to disk using pickle.

    Args:
        tokenizer: The tokenizer object to save.
        filepath: Path where to save the tokenizer.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath: str | Path) -> BPETokenizer:
    """Load a tokenizer from disk.

    Args:
        filepath: Path to the saved tokenizer.

    Returns:
        The loaded tokenizer object.

    Raises:
        FileNotFoundError: If the tokenizer file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {filepath}")

    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)

    print(f"Tokenizer loaded from {filepath}")
    return tokenizer