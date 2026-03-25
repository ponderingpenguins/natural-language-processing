"""LSTM model definition for text classification."""

import torch
from torch import nn

from utils.tokenizer import BPETokenizer
from .base_model import BaseModel

class LSTM(BaseModel):
    """LSTM model for text classification.

    Supports both unidirectional and bidirectional LSTM with proper handling
    of variable-length sequences using pack_padded_sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.5,
    ) -> None:
        """Initialize LSTM model.

        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of word embeddings.
            hidden_dim: LSTM hidden dimension.
            num_classes: Number of output classes.
            num_layers: Number of stacked LSTM layers (default: 2).
            bidirectional: Whether to use bidirectional LSTM (default: False).
            dropout: Dropout probability applied to hidden states (default: 0.5).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.tokenizer = BPETokenizer()
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
            lengths: Real sequence lengths before padding (batch_size,).
                    If None, assumes all sequences are used fully.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        embedded = self.embedding(x)

        if lengths is not None:
            # Pack sequences to ignore padding tokens
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        else:
            # No packing - process all positions
            _, (hidden, _) = self.lstm(embedded)

        # Extract final hidden state
        if self.lstm.bidirectional:
            # Concatenate final forward and backward hidden states
            # hidden[-2] is the last forward layer, hidden[-1] is the last backward layer
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            # Just take the last layer's hidden state
            last_hidden = hidden[-1]

        # Apply dropout and classification layer
        return self.fc(self.dropout(last_hidden))
    
    def tokenize(self, dataset: dict):
        """Tokenize input dataset using the provided BPE tokenizer."""
        return self.tokenizer(dataset["text"])