"""Metrics for evaluating board detection and position recognition."""

from typing import Dict, Optional
import torch
import numpy as np


def compute_mask_iou(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute IoU between predicted and target masks.

    Args:
        pred_mask: Predicted mask probabilities (B, 1, H, W) or logits
        target_mask: Ground truth binary mask (B, 1, H, W)
        threshold: Threshold for binarizing predictions

    Returns:
        IoU value(s)
    """
    # Binarize predictions
    if pred_mask.max() > 1.0:
        # Logits
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_mask.view(target_mask.size(0), -1)
    
    # Compute IoU
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    # Avoid division by zero
    iou = intersection / (union + 1e-6)
    
    return iou


def compute_mask_accuracy(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute pixel-wise accuracy.

    Args:
        pred_mask: Predicted mask probabilities (B, 1, H, W) or logits
        target_mask: Ground truth binary mask (B, 1, H, W)
        threshold: Threshold for binarizing predictions

    Returns:
        Accuracy value(s)
    """
    # Binarize predictions
    if pred_mask.max() > 1.0:
        # Logits
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    # Compute accuracy
    correct = (pred_binary == target_mask).float()
    accuracy = correct.mean(dim=[1, 2, 3])
    
    return accuracy


def compute_dice_coefficient(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute Dice coefficient.

    Args:
        pred_mask: Predicted mask probabilities (B, 1, H, W) or logits
        target_mask: Ground truth binary mask (B, 1, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor

    Returns:
        Dice coefficient value(s)
    """
    # Binarize predictions
    if pred_mask.max() > 1.0:
        # Logits
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_mask.view(target_mask.size(0), -1)
    
    # Compute Dice
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def compute_corner_error(
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    image_size: Optional[tuple] = None,
) -> Dict[str, float]:
    """
    Compute corner localization error in pixels.

    Args:
        pred_corners: Predicted corners (B, 4, 2) or (4, 2) 
        target_corners: Ground truth corners (B, 4, 2) or (4, 2)
        image_size: Optional (H, W) for denormalizing if corners are normalized

    Returns:
        Dictionary with error metrics
    """
    if pred_corners.ndim == 2:
        pred_corners = pred_corners.unsqueeze(0)
        target_corners = target_corners.unsqueeze(0)
    
    # Denormalize if needed
    if image_size is not None and pred_corners.max() <= 1.0:
        h, w = image_size
        pred_corners = pred_corners.clone()
        target_corners = target_corners.clone()
        pred_corners[..., 0] *= w
        pred_corners[..., 1] *= h
        target_corners[..., 0] *= w
        target_corners[..., 1] *= h
    
    # Compute Euclidean distance
    distances = torch.sqrt(((pred_corners - target_corners) ** 2).sum(dim=-1))
    
    return {
        'mean': distances.mean().item(),
        'max': distances.max().item(),
        'min': distances.min().item(),
        'per_corner_mean': distances.mean(dim=0).cpu().numpy(),
    }


def compute_position_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute accuracy for position recognition.

    Args:
        predictions: Predicted logits (B, C, 19, 19)
        targets: Ground truth class indices (B, 19, 19)

    Returns:
        Dictionary with accuracy metrics
    """
    pred_classes = predictions.argmax(dim=1)
    
    correct = (pred_classes == targets).float()
    
    accuracy = correct.mean().item()
    
    # Per-class accuracy
    per_class = {}
    for cls in range(3):
        mask = targets == cls
        if mask.sum() > 0:
            per_class[f'class_{cls}'] = correct[mask].mean().item()
    
    return {
        'overall': accuracy,
        **per_class,
    }


class MetricsTracker:
    """Track and aggregate metrics during training/validation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric name -> value
            count: Number of samples this metric is computed over
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * count
            self.counts[key] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        averaged = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averaged[key] = self.metrics[key] / self.counts[key]
            else:
                averaged[key] = 0.0
        
        return averaged
