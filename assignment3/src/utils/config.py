from transformers import dataclass

@dataclass
class LSTMConfig:
    """
    Hyperparameter configuration for the best LSTM model architecture from Assignment 2.

    Attributes:
        embed_dim: Embedding dimension for all models.
        lstm_hidden_dim: Hidden dimension for LSTM models.
        lstm_bidirectional: Whether to use bidirectional LSTM.
    """

    # TODO: Migrate the best hyperparameters from Assignment 2's LSTM tuning results here. Below are placeholders.
    embed_dim: int = 128  # embedding dimension for all models
    lstm_hidden_dim: int = 256  # hidden dimension for LSTM
    lstm_num_layers: int = 2  # number of stacked LSTM layers
    lstm_bidirectional: bool = False  # whether to use bidirectional LSTM
    lstm_dropout: float = 0.5  # dropout probability for LSTM    

class BERTConfig:
    """
    Hyperparameter configuration for the best BERT-based model architecture for Assignment 3.

    Attributes:
        model_name: Name of the pre-trained BERT model to use.
    """
    model_name: str = "bert-base-uncased"  # pre-trained BERT model name
    num_labels: int = 4  # number of output classes for classification