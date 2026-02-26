"""
Configuration dataclass for Assignment 2 model architectures.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for model architectures in Assignment 2.

    These are model-specific hyperparameters that may not be reusable
    across different assignments.

    Attributes:
        embed_dim: Embedding dimension for all models.
        cnn_num_filters: Number of output channels for CNN conv layer.
        cnn_kernel_size: Kernel size for CNN conv layer.
        lstm_hidden_dim: Hidden dimension for LSTM models.
        lstm_bidirectional: Whether to use bidirectional LSTM.
    """

    # Shared parameters
    embed_dim: int = 128  # embedding dimension for all models

    # CNN-specific parameters
    cnn_num_filters: int = 100  # number of convolutional filters
    cnn_kernel_size: int = 3  # kernel size for convolution

    # LSTM-specific parameters
    lstm_hidden_dim: int = 256  # hidden dimension for LSTM
    lstm_bidirectional: bool = False  # whether to use bidirectional LSTM
