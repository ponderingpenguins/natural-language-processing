"""LSTM model definition for text classification."""

import torch
import torch.nn as nn
from penguinlp.helpers import logger
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base_model import BaseModel


class LSTMClassifier(BaseModel):
    """LSTM model for text classification.

    Supports both unidirectional and bidirectional LSTM with
    attention-mask-aware pooling over token-level hidden states.
    """

    def __init__(self, config, device=None, **kwargs):
        """Initialize LSTM model with config object (LSTMConfig or DictConfig)."""
        super().__init__()
        self.sequence_length = config.sequence_length
        self.init_embeddings_from_bert = bool(
            getattr(config, "init_embeddings_from_bert", False)
        )
        self.bert_init_model_name = getattr(
            config, "bert_init_model_name", config.tokenizer_name
        )

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

        if self.init_embeddings_from_bert:
            self._init_embeddings_from_bert()

    def _init_embeddings_from_bert(self) -> None:
        """Initialize the embedding matrix from a pretrained BERT-like model."""
        logger.info("Initializing LSTM embeddings from %s", self.bert_init_model_name)
        bert_model = AutoModel.from_pretrained(self.bert_init_model_name)
        bert_embeddings = bert_model.get_input_embeddings().weight.detach()

        if bert_embeddings.shape != self.embedding.weight.shape:
            raise ValueError(
                "Embedding shape mismatch for BERT initialization: "
                f"LSTM={tuple(self.embedding.weight.shape)} vs "
                f"BERT={tuple(bert_embeddings.shape)}. "
                "Set lstm_model.embed_dim and lstm_model.vocab_size to match the "
                "selected BERT model, or disable init_embeddings_from_bert."
            )

        with torch.no_grad():
            self.embedding.weight.copy_(bert_embeddings)
            # Keep PAD embedding neutral when padding_idx=0.
            self.embedding.weight[0].zero_()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass with masked mean pooling over sequence outputs."""
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = output.mean(dim=1)

        logits = self.fc(self.dropout(pooled))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None

        # Trainer expects a ModelOutput or a tuple of (loss, logits, ...)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def tokenize(self, dataset: dict, **kwargs) -> dict:
        """Tokenize input dataset using the provided BPE tokenizer."""
        return self.tokenizer(
            dataset["text"],
            truncation=True,
            max_length=self.sequence_length,
            return_attention_mask=True,
            **kwargs,
        )
