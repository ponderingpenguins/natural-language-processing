from abc import ABC, abstractmethod
from torch import nn

class BaseModel(nn.Module, ABC):
    """Base model class for all models in this project.

    This class can be extended to implement specific models like LSTM, BERT, etc.
    It provides a common interface and can include shared utilities or methods in the future.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(self, dataset: dict):
        """Tokenize input dataset. This method should be overridden by subclasses to implement model-specific tokenization."""
        pass