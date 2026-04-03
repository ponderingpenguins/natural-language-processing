import torch.nn as nn
from omegaconf import DictConfig
from penguinlp.helpers import logger
from transformers import AutoModel, BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base_model import BaseModel


class BertClassifier(BaseModel):
    def __init__(self, config: DictConfig, device=None, **kwargs):
        """BERT-based classifier for text classification tasks."""
        super().__init__()
        self.model_name = config.model_name
        self.num_labels = config.num_labels
        self.max_length = config.max_length

        self.bert = AutoModel.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        hidden_size = self.bert.config.hidden_size
        self.classifier = self._build_head(hidden_size, config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.to(device)

        # Optionally freeze BERT layers to speed up training and reduce memory usage.
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

            frozen = sum(
                p.numel() for p in self.bert.parameters() if not p.requires_grad
            )
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info("Frozen: %d  Trainable: %d", frozen, trainable)

    def _build_head(self, hidden_size: int, config: DictConfig) -> nn.Module:
        """Build the classification head on top of BERT's pooled output."""
        dims = [hidden_size] + list(config.get("head_dims", [])) + [self.num_labels]
        dropout = config.get("head_dropout", 0.1)
        layers = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers = layers[:-2]  # drop final ReLU + Dropout
        return nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """Forward pass through the model."""
        pooled = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        ).pooler_output
        logits = self.classifier(pooled)

        loss = self.loss_fn(logits, labels) if labels is not None else None

        # Trainer expects a ModelOutput or a tuple of (loss, logits, ...)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def tokenize(self, example: dict, **kwargs) -> dict:
        """Tokenize a single example using the model's tokenizer."""
        return self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            **kwargs,
        )
