"""
Configuration dataclass for Assignment 2 model architectures.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Configuration for model architectures in Assignment 2.

    These are model-specific hyperparameters that may not be reusable
    across different assignments.

    Attributes:
        embed_dim: Embedding dimension for all models.
        cnn_num_filters: Number of output channels for CNN conv layer.
        cnn_kernel_sizes: Kernel sizes for CNN conv layers.
        lstm_hidden_dim: Hidden dimension for LSTM models.
        lstm_bidirectional: Whether to use bidirectional LSTM.
    """

    # Shared parameters
    embed_dim: int = 128  # embedding dimension for all models

    # CNN-specific parameters
    cnn_num_filters: int = 100  # number of convolutional filters
    cnn_kernel_sizes: list = field(
        default_factory=lambda: [3, 5, 7]
    )  # kernel sizes for convolution

    # LSTM-specific parameters
    lstm_hidden_dim: int = 256  # hidden dimension for LSTM
    lstm_num_layers: int = 2  # number of stacked LSTM layers
    lstm_bidirectional: bool = False  # whether to use bidirectional LSTM
    lstm_dropout: float = 0.5  # dropout probability for LSTM
