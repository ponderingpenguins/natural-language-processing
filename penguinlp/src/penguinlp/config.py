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
        seed: The random seed for reproducibility (default: 67).
        tokenizer_type: The type of tokenizer to use ("word", "bpe", or "char", default: "bpe").
        tokenizer_path: The path to save/load the tokenizer (default: "tokenizer.pkl").
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
    seed: int = 67

    # Assignment 2 specific parameters
    tokenizer_type: str = "bpe"  # type of tokenizer to use ("word", "bpe", or "char")
    tokenizer_path: str = "tokenizer.pkl"  # path to save/load the tokenizer
