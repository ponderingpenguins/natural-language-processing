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
        tokenizer_cache_dir: Directory for cached tokenizers (default: "output/tokenizer_cache").
        tokenized_cache_dir: Directory for cached tokenized splits (default: "output/tokenized_cache").
        clear_cache: Whether to clear tokenizer/tokenized caches before run (default: False).
        learning_rate: The learning rate for the optimizer (default: 1e-3).
        num_epochs: The number of training epochs (default: 20).
        tuning_num_epochs: Number of epochs per hyperparameter configuration
            during tuning (default: 1).
        gradient_clip_norm: The maximum norm for gradient clipping (default: 5.0).
        early_stopping_patience: Patience for validation-loss early stopping during
            training/tuning (default: 3).
        batch_size: The batch size for training (default: 4).
        weighted_decay: The weight decay for the optimizer (default: 1e-4).
        max_seq_length: The maximum sequence length for tokenization (default: 512).
        vocab_size: The vocabulary size for the tokenizer (default: 5000).
        min_freq: The minimum frequency for tokens to be included in the vocabulary (default: 2).
        sample_size: If set, subsample data to this size for quick testing (default: None).
        run_tuning_only: If True, run only hyperparameter tuning and exit (default: False).
        run_train_only: If True, skip tuning and train using best saved tuning config
            (default: False).
    """

    hf_dataset: str = "sh0416/ag_news"
    dev_split: float = (
        0.1  # percentage of training data to use as dev set (e.g., 0.1 for 10%)
    )
    max_misclassifications_to_report: int = 20
    output_dir: str = "output"  # directory to save misclassified examples
    label_names: dict[int, str] = field(
        default_factory=lambda: {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }
    )
    num_classes: int = 4
    seed: int = 67

    # Assignment 2 specific parameters
    model_type: str = "cnn"  # type of model to train ("cnn" or "lstm")
    tokenizer_type: str = "bpe"  # type of tokenizer to use ("word", "bpe", or "char")
    tokenizer_cache_dir: str = "output/tokenizer_cache"
    tokenized_cache_dir: str = "output/tokenized_cache"
    clear_cache: bool = False

    # Training hyperparameters (general, reusable across assignments)
    learning_rate: float = 1e-3  # learning rate for optimizer
    num_epochs: int = 20  # number of training epochs
    tuning_num_epochs: int = 1  # epochs per trial during hyperparameter tuning
    gradient_clip_norm: float = 5.0  # max norm for gradient clipping
    early_stopping_patience: int = 3  # patience for early stopping on val loss
    batch_size: int = 4  # batch size for training
    weighted_decay: float = 1e-4  # weight decay for optimizer

    # Data processing parameters
    max_seq_length: int = 512  # maximum sequence length for tokenization
    vocab_size: int = 5000  # vocabulary size for tokenizer
    min_freq: int = 2  # minimum frequency for tokens to be included in vocab

    # Debug/testing parameters
    sample_size: int | None = (
        None  # if set, subsample data to this size for quick testing
    )

    # Pipeline execution control
    run_tuning_only: bool = False
    run_train_only: bool = False
