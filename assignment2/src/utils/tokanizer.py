"""Tokenizer module for building a vocabulary from text data."""

import heapq
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from functools import lru_cache
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
        self._merge_priority: dict[tuple[str, str], int] = {}
        self._word_cache: dict[str, tuple[str, ...]] = {}

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

        # Build initial pair frequencies and inverted index
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set[str]] = defaultdict(set)
        for word, freq in word_counts.items():
            s = splits[word]
            for i in range(len(s) - 1):
                pair = (s[i], s[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word)

        # Build max-heap (negate freq for max behavior)
        heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
        heapq.heapify(heap)

        pbar = tqdm(total=self.vocab_size - len(vocab_list), desc="BPE Training")
        while len(vocab_list) < self.vocab_size:
            # Pop stale entries until we find a valid one
            best_pair = None
            while heap:
                neg_freq, candidate = heapq.heappop(heap)
                if pair_freqs.get(candidate, 0) == -neg_freq and -neg_freq > 0:
                    best_pair = candidate
                    break
            if best_pair is None:
                break

            a, b = best_pair
            merged = a + b

            # Only iterate over words that actually contain this pair
            changed_pairs: set[tuple[str, str]] = set()
            for word in list(pair_to_words.get(best_pair, set())):
                freq = word_counts[word]
                s = splits[word]
                i = 0
                while i < len(s) - 1:
                    if s[i] == a and s[i + 1] == b:
                        # Remove old neighboring pairs
                        if i > 0:
                            old_left = (s[i - 1], a)
                            pair_freqs[old_left] -= freq
                            pair_to_words[old_left].discard(word)
                            changed_pairs.add(old_left)
                        if i + 2 < len(s):
                            old_right = (b, s[i + 2])
                            pair_freqs[old_right] -= freq
                            pair_to_words[old_right].discard(word)
                            changed_pairs.add(old_right)

                        # Apply merge
                        s = s[:i] + [merged] + s[i + 2 :]

                        # Add new neighboring pairs
                        if i > 0:
                            new_left = (s[i - 1], merged)
                            pair_freqs[new_left] += freq
                            pair_to_words[new_left].add(word)
                            changed_pairs.add(new_left)
                        if i + 1 < len(s):
                            new_right = (merged, s[i + 1])
                            pair_freqs[new_right] += freq
                            pair_to_words[new_right].add(word)
                            changed_pairs.add(new_right)
                    else:
                        i += 1
                splits[word] = s

            # Clean up merged pair
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            # Push changed pairs onto the heap
            for pair in changed_pairs:
                freq = pair_freqs.get(pair, 0)
                if freq > 0:
                    heapq.heappush(heap, (-freq, pair))

            self.merges[best_pair] = merged
            vocab_list.append(merged)
            pbar.update(1)

        pbar.close()
        unique_vocab_list = list(dict.fromkeys(vocab_list))
        self.vocab = {token: idx for idx, token in enumerate(unique_vocab_list)}

        # Cache merge priority for faster inference
        self._merge_priority = {pair: i for i, pair in enumerate(self.merges)}

        # Cache pre-computed tokenizations for training vocabulary words
        self._word_cache = {word: tuple(split) for word, split in splits.items()}

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using the learned merges.

        Args:
            text: The input string to tokenize.
        Returns:
            A list of tokens resulting from applying the BPE merges to the input text.
        """
        return [token for w in text.split() for token in self._tokenize_word(w)]

    @lru_cache(maxsize=65536)
    def _tokenize_word(self, word: str) -> tuple[str, ...]:
        """
        Tokenize a single word using the learned merges.

        Args:
            word: The input word to tokenize.
        Returns:
            A tuple of tokens resulting from applying the BPE merges to the input word.
        """
        # Check if word was seen during training (O(1) lookup)
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached

        # Use pre-computed merge priority instead of rebuilding
        merge_priority = self._merge_priority
        split = list(word)
        while len(split) > 1:
            # Find the pair with the highest priority (lowest index)
            best_idx, best_pair = None, None
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                if pair in merge_priority:
                    if (
                        best_pair is None
                        or merge_priority[pair] < merge_priority[best_pair]
                    ):
                        best_idx, best_pair = i, pair
            if best_pair is None:
                break
            # Apply all occurrences of this merge
            merged = self.merges[best_pair]
            i = 0
            new_split = []
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
        return tuple(split)

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
