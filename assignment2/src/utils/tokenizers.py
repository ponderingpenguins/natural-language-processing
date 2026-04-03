from abc import ABC, abstractmethod
from collections import Counter

class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    
    def fit(self, texts: list[dict], text_field: str = "text") -> None:
        """Fit the tokenizer on a list of strings."""
        raw_texts = [sample[text_field] for sample in texts]
        self._build_vocab(raw_texts)

    def _build_vocab(self, texts: list[str]) -> None:
        """Tokenize texts and populate self.vocab."""
        token_count: Counter = Counter()
        for text in texts:
            token_count.update(self.tokenize(text))

        # Keep the most common tokens up to vocab_size
        for token, _ in token_count.most_common(self.vocab_size - 2):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single string into a list of tokens."""
    
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in self.tokenize(text)]

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)
        
    

class WordTokenizer(BaseTokenizer):
    """Tokenize text at the word level."""
    def tokenize(self, text: str) -> list[str]:
        return text.split()
            

class CharTokenizer(BaseTokenizer):
    """Tokenize text at the character level."""
    def tokenize(self, text: str) -> list[str]:
        return list(text)