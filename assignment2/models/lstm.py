"""LSTM and BiLSTM model definitions for text classification."""

import torch
from torch import nn


class LSTM(nn.Module):
    """LSTM model for text classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim * (2 if bidirectional else 1), num_classes
        )  # (bidirectional doubles hidden size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


class BiLSTM(LSTM):
    """Bi-directional LSTM model for text classification."""

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int
    ) -> None:
        super().__init__(
            vocab_size, embed_dim, hidden_dim, num_classes, bidirectional=True
        )
