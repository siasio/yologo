"""Loss functions for training board detection and position recognition networks."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBoardDetectionLoss(ABC, nn.Module):
    """Abstract base class for board detection losses."""

    @abstractmethod
    def forward(
        self,
        predictions: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: Model predictions (format depends on model type)
            targets: Ground truth dictionary

        Returns:
            Dictionary with 'total_loss' and component losses
        """
        pass


class SegmentationLoss(BaseBoardDetectionLoss):
    """
    Loss for segmentation-based board detection.
    
    Combines Binary Cross-Entropy and Dice loss for robust segmentation.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        """
        Initialize segmentation loss.

        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss (prevents division by zero)
        """
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Dice loss value
        """
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss is 1 - Dice coefficient
        return (1.0 - dice).mean()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.

        Args:
            predictions: Logits of shape (B, 1, H, W)
            targets: Dictionary containing 'mask' of shape (B, 1, H, W)

        Returns:
            Dictionary with loss components
        """
        target_mask = targets['mask']
        
        # Ensure predictions and targets have the same spatial dimensions
        if predictions.shape[-2:] != target_mask.shape[-2:]:
            predictions = F.interpolate(
                predictions,
                size=target_mask.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Compute losses
        bce = self.bce_loss(predictions, target_mask)
        dice = self.dice_loss(predictions, target_mask)
        
        # Combined loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce,
            'dice_loss': dice,
        }


class PositionRecognitionLoss(nn.Module):
    """
    Loss for position recognition (stone classification on 19x19 grid).
    
    Multi-class classification: empty (0), black (1), white (2).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = False,
    ):
        """
        Initialize position recognition loss.

        Args:
            class_weights: Optional weights for each class (empty, black, white)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_focal: Whether to use focal loss (better for imbalanced classes)
        """
        super().__init__()
        
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if not use_focal:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.

        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Ground truth class indices (B, H, W)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute position recognition loss.

        Args:
            predictions: Logits of shape (B, 3, 19, 19)
            targets: Ground truth class indices of shape (B, 19, 19)

        Returns:
            Dictionary with loss value
        """
        if self.use_focal:
            loss = self.focal_loss(predictions, targets)
        else:
            loss = self.ce_loss(predictions, targets)
        
        return {
            'total_loss': loss,
            'ce_loss': loss,
        }


def create_board_detection_loss(config: Dict) -> BaseBoardDetectionLoss:
    """
    Create board detection loss from config.

    Args:
        config: Configuration dictionary

    Returns:
        Board detection loss instance
    """
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "segmentation")
    
    if loss_type == "segmentation":
        return SegmentationLoss(
            bce_weight=loss_config.get("bce_weight", 0.5),
            dice_weight=loss_config.get("dice_weight", 0.5),
            smooth=loss_config.get("smooth", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_position_recognition_loss(config: Dict) -> PositionRecognitionLoss:
    """
    Create position recognition loss from config.

    Args:
        config: Configuration dictionary

    Returns:
        Position recognition loss instance
    """
    loss_config = config.get("loss", {})
    
    # Optional class weights: [empty, black, white]
    class_weights = loss_config.get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    return PositionRecognitionLoss(
        class_weights=class_weights,
        focal_alpha=loss_config.get("focal_alpha", 0.25),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
        use_focal=loss_config.get("use_focal", False),
    )
