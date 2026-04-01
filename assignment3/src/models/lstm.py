"""LSTM model definition for text classification."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base_model import BaseModel


class LSTMClassifier(BaseModel):
    """LSTM model for text classification.

    Supports both unidirectional and bidirectional LSTM with proper handling
    of variable-length sequences using pack_padded_sequence.
    """

    def __init__(self, config, device=None, **kwargs):
        """Initialize LSTM model with config object (LSTMConfig or DictConfig)."""
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=0
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(
            config.hidden_dim * (2 if config.bidirectional else 1), config.num_classes
        )

    def forward(self, input_ids, labels=None, lengths=None, **kwargs):
        """Forward pass through the model. Handles variable-length sequences if lengths are provided."""
        # For LSTM, we need to handle variable-length sequences. If lengths are provided, we use pack_padded_sequence.
        embedded = self.embedding(input_ids)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # For bidirectional LSTM, concatenate the final forward and backward hidden states
        if self.lstm.bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]

        logits = self.fc(self.dropout(last_hidden))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None

        # Trainer expects a ModelOutput or a tuple of (loss, logits, ...)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def tokenize(self, dataset: dict, **kwargs) -> dict:
        """Tokenize input dataset using the provided BPE tokenizer."""
        return self.tokenizer(dataset["text"], **kwargs)
