"""CNN model definition for text classification."""

import torch
from torch import nn


class CNN(nn.Module):
    """A simple CNN for text classification."""

    def __init__(
        self, vocab_size, embed_dim, num_filters, kernel_sizes, num_classes, dropout=0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths=None):
        """Forward pass of the CNN.

        Args:
            x: Input tensor of shape (batch_size, seq_len).
            lengths: Sequence lengths (ignored, for compatibility with LSTM).
        """
        x = self.embedding(x).permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)
        return x
