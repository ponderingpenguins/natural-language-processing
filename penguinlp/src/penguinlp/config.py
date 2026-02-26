"""
Configuration dataclass for Assignment 2
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    Configuration for training the TF-IDF classifier.

    Attributes:
        hf_dataset: The Hugging Face dataset to use (default: "sh0416/ag_news").
        dev_split: The percentage of training data to use as a dev set (default: 0.1 for 10%).
        max_misclassifications_to_report: The maximum number of misclassified examples to report
            (default: 20).
        label_names: Mapping from integer label to human-readable class name.
        num_classes: The number of classes in the dataset (default: 4).
        seed: The random seed for reproducibility (default: 67).
        tokenizer_type: The type of tokenizer to use ("word", "bpe", or "char", default: "bpe").
        tokenizer_path: The path to save/load the tokenizer (default: "tokenizer.pkl").
        learning_rate: The learning rate for the optimizer (default: 1e-3).
        num_epochs: The number of training epochs (default: 20).
        gradient_clip_norm: The maximum norm for gradient clipping (default: 5.0).
        batch_size: The batch size for training (default: 4).
        max_seq_len: The maximum sequence length for tokenization (default: 512).
        vocab_size: The vocabulary size for the tokenizer (default: 1000).
        min_freq: The minimum frequency for tokens to be included in the vocabulary (default: 2).
        sample_size: If set, subsample data to this size for quick testing (default: None).
    """

    hf_dataset: str = "sh0416/ag_news"
    dev_split: float = (
        0.1  # percentage of training data to use as dev set (e.g., 0.1 for 10%)
    )
    max_misclassifications_to_report: int = 20
    output_dir: str = "output"  # directory to save misclassified examples
    label_names: dict[int, str] = field(
        default_factory=lambda: {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech",
        }
    )
    num_classes: int = 4
    seed: int = 67

    # Assignment 2 specific parameters
    tokenizer_type: str = "bpe"  # type of tokenizer to use ("word", "bpe", or "char")
    tokenizer_path: str = "tokenizer.pkl"  # path to save/load the tokenizer

    # Training hyperparameters (general, reusable across assignments)
    learning_rate: float = 1e-3  # learning rate for optimizer
    num_epochs: int = 20  # number of training epochs
    gradient_clip_norm: float = 5.0  # max norm for gradient clipping
    batch_size: int = 4  # batch size for training

    # Data processing parameters
    max_seq_len: int = 512  # maximum sequence length for tokenization
    vocab_size: int = 1000  # vocabulary size for tokenizer
    min_freq: int = 2  # minimum frequency for tokens to be included in vocab

    # Debug/testing parameters
    sample_size: int | None = (
        None  # if set, subsample data to this size for quick testing
    )
