from dataclasses import dataclass, field

@dataclass
class LSTMConfig:
    """
    Hyperparameter configuration for the best LSTM model architecture from Assignment 2.

    Attributes:
        embed_dim: Embedding dimension for all models.
        lstm_hidden_dim: Hidden dimension for LSTM models.
        lstm_bidirectional: Whether to use bidirectional LSTM.
    """

    # TODO: Migrate the best hyperparameters from Assignment 2's LSTM tuning results here. Double-check needed by at least another team member.
    vocab_size: int = 30522  # vocabulary size for LSTM (same as BERT's tokenizer)
    embed_dim: int = 256  # embedding dimension for LSTM
    hidden_dim: int = 512  # hidden dimension for LSTM
    num_classes: int = 4  # number of output classes for classification
    num_layers: int = 2  # number of stacked LSTM layers
    bidirectional: bool = True  # whether to use bidirectional LSTM
    learning_rate: float = 1e-3  # learning rate for LSTM training
    sequence_length: int = 256  # maximum sequence length for LSTM input
    dropout: float = 0.5  # dropout probability for LSTM
    early_stopping_patience: int = 3  # number of evaluation steps with no improvement to wait before stopping training

@dataclass
class BERTConfig:
    """
    Hyperparameter configuration for the best BERT-based model architecture for Assignment 3.

    Attributes:
        model_name: Name of the pre-trained BERT model to use.
    """
    model_name: str = "bert-base-uncased"  # pre-trained BERT model name
    num_labels: int = 4  # number of output classes for classification
    output_dir: str = "./bert_output"  # directory to save BERT training outputs
    num_train_epochs: int = 15  # number of training epochs for BERT
    learning_rate: float = 2e-5  # learning rate for AdamW in BERT fine-tuning
    max_length: int = 256  # max sequence length used during tokenization
    per_device_train_batch_size: int = 16  # batch size for training BERT
    per_device_eval_batch_size: int = 16  # batch size for evaluating BERT
    warmup_steps: int = 500  # number of warmup steps for learning rate scheduler
    logging_dir: str = "./bert_logs"  # directory for BERT training logs
    early_stopping_patience: int = 3  # number of evaluation steps with no improvement to wait before stopping training
    grid_learning_rates: list[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    grid_max_lengths: list[int] = field(default_factory=lambda: [128, 256, 512])
    grid_batch_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])