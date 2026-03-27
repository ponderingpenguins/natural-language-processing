from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BaseModel(nn.Module, ABC):
    """Base model class for all models in this project."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model. This method should be overridden by subclasses to implement the model's forward logic."""

    @abstractmethod
    def tokenize(self, dataset, **kwargs) -> Any:
        """Tokenize input dataset. This method should be overridden by subclasses to implement model-specific tokenization."""
