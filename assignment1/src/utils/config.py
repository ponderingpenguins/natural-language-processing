from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration for training the TF-IDF classifier.

    Attributes:
        hf_dataset: The Hugging Face dataset to use (default: "sh0416/ag_news").
        model: The model type to train (default: "logistic_regression").
        dev_split: The percentage of training data to use as a dev set (default: 0.1 for 10%).
        max_features: The maximum number of features for TF-IDF vectorization (default: 10000).
        max_misclassifications_to_report: The maximum number of misclassified examples to report
            (default: 20).
    """

    hf_dataset: str = "sh0416/ag_news"
    model: str = "logistic_regression"
    dev_split: float = (
        0.1  # percentage of training data to use as dev set (e.g., 0.1 for 10%)
    )
    max_features: int = 10000
    max_misclassifications_to_report: int = 20
