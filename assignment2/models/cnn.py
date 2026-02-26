"""CNN model definition for text classification."""

import torch
from torch import nn


class CNN(nn.Module):
    """A simple CNN for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        """Forward pass of the CNN."""
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        x = self.conv(x)  # (batch_size, 100, seq_len-2)
        x = self.pool(x).squeeze(-1)  # (batch_size, 100)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

    def predict(self, x):
        """Predict class labels for input x."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
