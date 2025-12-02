"""Base classes for board detection models."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class BaseBoardDetector(ABC, nn.Module):
    """
    Abstract base class for board detection models.
    
    All board detectors should inherit from this class and implement
    the forward and predict methods.
    """

    def __init__(self, input_size: int = 640):
        """
        Initialize board detector.

        Args:
            input_size: Expected input image size (square)
        """
        super().__init__()
        self.input_size = input_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Model-specific output tensor
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: torch.Tensor,
        return_corners: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict board location from image.

        Args:
            image: Input image tensor (3, H, W) or (B, 3, H, W)
            return_corners: Whether to extract and return corner coordinates

        Returns:
            Dictionary with model-specific outputs, e.g.:
                - 'mask': Binary segmentation mask (B, 1, H, W)
                - 'corners': Corner coordinates (B, 4, 2) if return_corners=True
                - 'corners_normalized': Normalized corners (B, 4, 2)
        """
        pass

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'state_dict': self.state_dict(),
            'input_size': self.input_size,
            'model_type': self.__class__.__name__,
        }, path)

    def load(self, path: str, device: str = 'cpu'):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        self.input_size = checkpoint.get('input_size', self.input_size)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device
