from transformers import AutoModelForSequenceClassification, BertTokenizer
from torch import nn

from utils.config import BERTConfig

cfg = BERTConfig()

class BertClassifier(nn.Module):
    """
    A simple BERT-based classifier for text classification tasks.
    This model uses a pre-trained BERT model as the base and adds a linear layer on top for classification.
    """
    def __init__(self, model_name: str = cfg.model_name, num_labels: int = cfg.num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize(self, dataset: dict):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return tokenizer(dataset["text"], padding="max_length", truncation=True)
