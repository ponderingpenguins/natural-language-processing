"""Tokenizer module for building a vocabulary from text data."""

import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from tqdm import tqdm

TokenizerType = Literal["word", "bpe", "char"]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    def __init__(self, min_freq: int = 2, vocab_size: int = 100):
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}

    def fit(self, texts: list[dict], text_field: str = "text") -> None:
        """Build vocabulary from a list of text documents."""
        raw_texts = [doc[text_field] for doc in texts]
        self._build_vocab(raw_texts)

    @abstractmethod
    def _build_vocab(self, texts: list[str]) -> None:
        """Tokenize texts and populate self.vocab."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single string into a list of tokens."""

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of vocabulary indices."""
        unk_idx = self.vocab.get(UNK_TOKEN, 0)
        return [self.vocab.get(token, unk_idx) for token in self.tokenize(text)]

    def _finalize_vocab(self, token_counts: Counter) -> None:
        """Apply min_freq filtering and vocab_size limit, then assign indices."""
        filtered = (
            (token, count)
            for token, count in token_counts.most_common(self.vocab_size)
            if count >= self.min_freq
        )
        self.vocab = {token: idx for idx, (token, _) in enumerate(filtered)}


class WordTokenizer(BaseTokenizer):
    """
    Whitespace-based word-level tokenizer.
    This implementation is based on the NLP tutorial of this course
    """

    def _build_vocab(self, texts: list[str]) -> None:
        token_counts: Counter = Counter()
        for text in texts:
            token_counts.update(text.split())
        self._finalize_vocab(token_counts)

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    This implementation is based on the NLP tutorial of this course
    """

    def _build_vocab(self, texts: list[str]) -> None:
        token_counts: Counter = Counter()
        for text in texts:
            token_counts.update(list(text))
        self._finalize_vocab(token_counts)

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer.

    This implementation is based on the huggingface tutorial: https://huggingface.co/learn/llm-course/en/chapter6/5#byte-pair-encoding-tokenization
    """

    def __init__(self, min_freq: int = 2, vocab_size: int = 100):
        super().__init__(min_freq, vocab_size)
        self.merges: dict[tuple[str, str], str] = {}

    @staticmethod
    def _compute_pair_freqs(
        splits: dict[str, list[str]], word_freqs: Counter
    ) -> dict[tuple[str, str], int]:
        """Compute frequencies of every consecutive token pair."""
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            for i in range(len(split) - 1):
                pair_freqs[(split[i], split[i + 1])] += freq
        return pair_freqs

    @staticmethod
    def _merge_pair(
        a: str, b: str, splits: dict[str, list[str]], word_freqs: Counter
    ) -> dict[str, list[str]]:
        """Merge all occurrences of pair (a, b) in the splits table."""
        merged = a + b
        for word in word_freqs:
            split = splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [merged] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def _build_vocab(self, texts: list[str]) -> None:
        word_counts: Counter = Counter()
        for text in texts:
            word_counts.update(text.split())

        alphabet = sorted({char for word in word_counts for char in word})
        vocab_list = SPECIAL_TOKENS + alphabet
        splits: dict[str, list[str]] = {word: list(word) for word in word_counts}

        # Build pair freqs once
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        for word, freq in word_counts.items():
            s = splits[word]
            for i in range(len(s) - 1):
                pair_freqs[(s[i], s[i + 1])] += freq

        pbar = tqdm(total=self.vocab_size - len(vocab_list), desc="BPE Training")
        while len(vocab_list) < self.vocab_size:
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            a, b = best_pair
            merged = a + b

            # Merge and incrementally update pair_freqs
            for word, freq in word_counts.items():
                s = splits[word]
                i = 0
                while i < len(s) - 1:
                    if s[i] == a and s[i + 1] == b:
                        # Remove old pairs touching this position
                        if i > 0:
                            pair_freqs[(s[i - 1], a)] -= freq
                        if i + 2 < len(s):
                            pair_freqs[(b, s[i + 2])] -= freq
                        # Merge
                        s = s[:i] + [merged] + s[i + 2 :]
                        # Add new pairs
                        if i > 0:
                            pair_freqs[(s[i - 1], merged)] += freq
                        if i + 1 < len(s):
                            pair_freqs[(merged, s[i + 1])] += freq
                        # Don't increment i — check for consecutive merges
                    else:
                        i += 1
                splits[word] = s

            del pair_freqs[best_pair]
            # Clean up zero entries periodically
            self.merges[best_pair] = merged
            vocab_list.append(merged)
            pbar.update(1)
        pbar.close()
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}

    def tokenize(self, text: str) -> list[str]:
        words = text.split()
        splits: list[list[str]] = [list(word) for word in words]

        for (a, b), merged in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split = split[:i] + [merged] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


def build_tokenizer(
    texts: list[dict],
    min_freq: int = 2,
    vocab_size: int = 100,
    tokenizer_type: TokenizerType = "word",
    text_field: str = "text",
) -> BaseTokenizer:
    """Instantiate, fit, and return the appropriate tokenizer."""
    cls_map: dict[str, type[BaseTokenizer]] = {
        "word": WordTokenizer,
        "bpe": BPETokenizer,
        "char": CharTokenizer,
    }
    if tokenizer_type not in cls_map:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    tokenizer = cls_map[tokenizer_type](min_freq=min_freq, vocab_size=vocab_size)
    tokenizer.fit(texts, text_field=text_field)
    return tokenizer


def save_tokenizer(tokenizer: BaseTokenizer, filepath: str | Path) -> None:
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


def load_tokenizer(filepath: str | Path) -> BaseTokenizer:
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
