"""Segmentation-based board detector using MobileNetV3 encoder."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np

from .base import BaseBoardDetector


class SegmentationBoardDetector(BaseBoardDetector):
    """
    Board detector using segmentation approach.
    
    Architecture:
    - MobileNetV3-Small encoder (pretrained on ImageNet)
    - Lightweight decoder with skip connections
    - Output: Binary mask of board region (B, 1, H, W)
    
    Corners are extracted from the mask using contour detection during inference.
    This approach is rotation-invariant - no need for ordered corner labels.
    """

    def __init__(
        self,
        input_size: int = 640,
        mask_size: int = 160,
        pretrained: bool = True,
    ):
        """
        Initialize segmentation board detector.

        Args:
            input_size: Expected input image size (square)
            mask_size: Output mask resolution (will be upsampled to input_size for loss)
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__(input_size=input_size)
        
        self.mask_size = mask_size
        
        # Load MobileNetV3-Small as encoder
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Extract encoder layers (remove classifier)
        self.encoder = nn.Sequential(*list(mobilenet.features.children()))
        
        # Get feature dimensions at different scales for skip connections
        # MobileNetV3-Small feature dimensions:
        # After block 0: 16 channels
        # After block 3: 24 channels  
        # After block 8: 48 channels
        # After block 12: 576 channels (final)
        
        # Decoder with upsampling and skip connections
        self.decoder = nn.ModuleDict({
            # Reduce final features
            'reduce': nn.Sequential(
                nn.Conv2d(576, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            
            # Upsample blocks
            'up1': self._make_upsample_block(256, 128),
            'up2': self._make_upsample_block(128, 64),
            'up3': self._make_upsample_block(64, 32),
            'up4': self._make_upsample_block(32, 16),
            
            # Final prediction head
            'head': nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=1),
            ),
        })
        
    def _make_upsample_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create an upsample block with conv."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits of shape (B, 1, H_mask, W_mask) where H_mask = W_mask = mask_size
        """
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = self.decoder['reduce'](features)
        x = self.decoder['up1'](x)
        x = self.decoder['up2'](x)
        x = self.decoder['up3'](x)
        x = self.decoder['up4'](x)
        logits = self.decoder['head'](x)
        
        return logits
    
    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        return_corners: bool = True,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict board mask and optionally extract corners.

        Args:
            image: Input image tensor (3, H, W) or (B, 3, H, W)
            return_corners: Whether to extract corner coordinates from mask
            threshold: Threshold for binary mask (0-1)

        Returns:
            Dictionary containing:
                - 'logits': Raw logits (B, 1, H_mask, W_mask)
                - 'mask': Binary mask (B, 1, H, W) at input resolution
                - 'mask_prob': Probability mask (B, 1, H, W)
                - 'corners': Corner pixel coordinates (B, 4, 2) if return_corners=True
                - 'corners_normalized': Normalized corners (B, 4, 2) if return_corners=True
        """
        self.eval()
        
        # Handle single image
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image.shape
        
        # Forward pass
        logits = self.forward(image)
        
        # Upsample to input resolution
        logits_upsampled = F.interpolate(
            logits, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Get probabilities
        mask_prob = torch.sigmoid(logits_upsampled)
        
        # Binary mask
        mask = (mask_prob > threshold).float()
        
        result = {
            'logits': logits,
            'mask': mask,
            'mask_prob': mask_prob,
        }
        
        # Extract corners if requested
        if return_corners:
            corners_list = []
            corners_normalized_list = []
            
            for i in range(B):
                mask_np = mask[i, 0].cpu().numpy().astype(np.uint8)
                corners = self._extract_corners_from_mask(mask_np, image_size=(H, W))
                
                if corners is not None:
                    corners_tensor = torch.from_numpy(corners).float().to(image.device)
                    corners_normalized = corners_tensor.clone()
                    corners_normalized[:, 0] /= W
                    corners_normalized[:, 1] /= H
                else:
                    # Return invalid corners if extraction fails
                    corners_tensor = torch.zeros(4, 2, device=image.device)
                    corners_normalized = torch.zeros(4, 2, device=image.device)
                
                corners_list.append(corners_tensor)
                corners_normalized_list.append(corners_normalized)
            
            result['corners'] = torch.stack(corners_list)
            result['corners_normalized'] = torch.stack(corners_normalized_list)
        
        if squeeze_output:
            for key in result:
                if result[key].shape[0] == 1:
                    result[key] = result[key].squeeze(0)
        
        return result
    
    def _extract_corners_from_mask(
        self, 
        mask: np.ndarray, 
        image_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract 4 corners from binary mask using contour detection.

        Args:
            mask: Binary mask (H, W) with 0/1 values
            image_size: (H, W) of the image

        Returns:
            Numpy array of shape (4, 2) with corner coordinates [x, y]
            Ordered clockwise from top-left based on angle from centroid
            Returns None if corner extraction fails
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get largest contour (should be the board)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we get exactly 4 points, use them
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Otherwise, use minimum area rectangle or convex hull
            if len(approx) < 4:
                # Use minimum bounding rectangle
                rect = cv2.minAreaRect(largest_contour)
                corners = cv2.boxPoints(rect).astype(np.float32)
            else:
                # Use convex hull and get 4 extreme points
                hull = cv2.convexHull(largest_contour)
                
                # Get 4 corners by finding extreme points
                corners = self._get_four_extreme_points(hull.reshape(-1, 2))
        
        # Order corners consistently: clockwise from top-left
        corners = self._order_corners(corners)
        
        return corners
    
    def _get_four_extreme_points(self, points: np.ndarray) -> np.ndarray:
        """
        Get 4 extreme points from a set of points.
        
        Args:
            points: Array of shape (N, 2)
            
        Returns:
            Array of shape (4, 2) with 4 corner points
        """
        # Get extreme points
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left.
        Based on clockwise ordering from centroid.

        Args:
            corners: Array of shape (4, 2)

        Returns:
            Ordered array of shape (4, 2)
        """
        # Calculate centroid
        centroid = corners.mean(axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(
            corners[:, 1] - centroid[1],
            corners[:, 0] - centroid[0]
        )
        
        # Sort by angle (clockwise from top-left)
        # Adjust angles so top-left is first
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        
        # Find the point closest to top-left (smallest x+y)
        top_left_idx = np.argmin(corners[:, 0] + corners[:, 1])
        
        # Sort by angle starting from top-left
        sorted_indices = np.argsort(angles)
        
        # Rotate sorted indices to start from top_left_idx
        start_pos = np.where(sorted_indices == top_left_idx)[0][0]
        sorted_indices = np.roll(sorted_indices, -start_pos)
        
        return corners[sorted_indices]


def create_segmentation_detector(config: Dict) -> SegmentationBoardDetector:
    """
    Create segmentation board detector from config.

    Args:
        config: Configuration dictionary

    Returns:
        SegmentationBoardDetector instance
    """
    model_config = config.get("model", {})
    
    input_size = model_config.get("input_size", 640)
    mask_size = model_config.get("mask_size", 160)
    pretrained = model_config.get("pretrained", True)
    
    return SegmentationBoardDetector(
        input_size=input_size,
        mask_size=mask_size,
        pretrained=pretrained,
    )
