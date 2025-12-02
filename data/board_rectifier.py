"""Board rectification utilities for processing board detector outputs.

This module handles the transformation from detected board corners to rectified
square images suitable for position recognition. It includes:
- Corner fitting and validation
- Padding application before rectification
- Perspective transformation to square format
- Corner mapping tracking for ground truth rotation
"""

from typing import Tuple, Dict, Optional
from enum import Enum

import cv2
import numpy as np

from utils.geometry import order_points


class Corner(Enum):
    """Enum for board corners."""
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3


class BoardRectifier:
    """
    Handles rectification of detected board regions to square images.
    
    This class processes board detector predictions and prepares images
    for the position recognition network. It handles:
    1. Corner validation and ordering
    2. Padding addition (configurable, default 2%)
    3. Perspective transformation to square
    4. Corner mapping tracking for ground truth rotation
    """
    
    def __init__(
        self,
        output_size: int = 224,
        padding_ratio: float = 0.02,
        min_board_size: int = 50,
    ):
        """
        Initialize board rectifier.
        
        Args:
            output_size: Size of output square image (width and height)
            padding_ratio: Fraction of board size to add as padding (default 2%)
            min_board_size: Minimum board size in pixels to consider valid
        """
        self.output_size = output_size
        self.padding_ratio = padding_ratio
        self.min_board_size = min_board_size
    
    def validate_corners(self, corners: np.ndarray) -> bool:
        """
        Validate that corners form a reasonable quadrilateral.
        
        Args:
            corners: Array of shape (4, 2) with corner coordinates
            
        Returns:
            True if corners are valid, False otherwise
        """
        if corners.shape != (4, 2):
            return False
        
        # Check all coordinates are finite
        if not np.all(np.isfinite(corners)):
            return False
        
        # Check board is large enough
        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        
        if width < self.min_board_size or height < self.min_board_size:
            return False
        
        # Check corners form a proper quadrilateral (no self-intersections)
        # Using cross product to check if corners are in consistent order
        ordered = order_points(corners)
        
        # Compute area using shoelace formula
        area = 0.5 * abs(
            ordered[0, 0] * ordered[1, 1] - ordered[1, 0] * ordered[0, 1] +
            ordered[1, 0] * ordered[2, 1] - ordered[2, 0] * ordered[1, 1] +
            ordered[2, 0] * ordered[3, 1] - ordered[3, 0] * ordered[2, 1] +
            ordered[3, 0] * ordered[0, 1] - ordered[0, 0] * ordered[3, 1]
        )
        
        # Area should be positive and reasonable
        if area < self.min_board_size ** 2:
            return False
        
        return True
    
    def add_padding(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Add padding around detected board corners.
        
        This ensures the full board is visible after rectification,
        accounting for potential detection inaccuracies.
        
        Args:
            corners: Array of shape (4, 2) with corner coordinates [x, y]
            image_shape: (height, width) of the source image
            
        Returns:
            Padded corners array of shape (4, 2)
        """
        h, w = image_shape
        
        # Order corners
        ordered = order_points(corners)
        
        # Compute center of the quadrilateral
        center = ordered.mean(axis=0)
        
        # Compute padding amount based on average distance from center
        distances = np.linalg.norm(ordered - center, axis=1)
        avg_distance = distances.mean()
        padding_px = avg_distance * self.padding_ratio
        
        # Move each corner away from center by padding amount
        padded = np.zeros_like(ordered)
        for i in range(4):
            direction = ordered[i] - center
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                padded[i] = ordered[i] + direction * padding_px
            else:
                padded[i] = ordered[i]
        
        # Clip to image bounds
        padded[:, 0] = np.clip(padded[:, 0], 0, w - 1)
        padded[:, 1] = np.clip(padded[:, 1], 0, h - 1)
        
        return padded
    
    def get_corner_mapping(
        self,
        src_corners: np.ndarray,
        original_corners_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Corner]:
        """
        Determine which original corner maps to which position in rectified image.
        
        This is crucial for correctly rotating ground truth position arrays
        when the board is viewed from different angles.
        
        Args:
            src_corners: Ordered source corners (4, 2) in TL, TR, BR, BL order
            original_corners_dict: Optional dict with keys 'top_left', 'top_right',
                'bottom_left', 'bottom_right' from manifest
                
        Returns:
            Dictionary mapping original corner names to Corner enum values,
            indicating where each original corner ended up in the rectified image.
            
        Example:
            If the board is rotated 180°, the mapping would be:
            {'top_left': Corner.BOTTOM_RIGHT, 'top_right': Corner.BOTTOM_LEFT, ...}
        """
        if original_corners_dict is None:
            # No rotation tracking needed, assume identity mapping
            return {
                'top_left': Corner.TOP_LEFT,
                'top_right': Corner.TOP_RIGHT,
                'bottom_right': Corner.BOTTOM_RIGHT,
                'bottom_left': Corner.BOTTOM_LEFT,
            }
        
        # Convert original corners dict to array in standard order
        original_ordered = np.array([
            original_corners_dict['top_left'],
            original_corners_dict['top_right'],
            original_corners_dict['bottom_right'],
            original_corners_dict['bottom_left'],
        ], dtype=np.float32)
        
        # Find which original corner corresponds to each position in src_corners
        # by finding the closest match
        mapping = {}
        corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        
        used_indices = set()
        for src_idx, src_corner in enumerate(src_corners):
            # Find closest original corner
            distances = np.linalg.norm(original_ordered - src_corner, axis=1)
            
            # Exclude already used corners
            for used_idx in used_indices:
                distances[used_idx] = np.inf
            
            orig_idx = np.argmin(distances)
            used_indices.add(orig_idx)
            
            # Map original corner name to its destination
            mapping[corner_names[orig_idx]] = Corner(src_idx)
        
        return mapping
    
    def rectify(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        add_padding: bool = True,
        original_corners_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Corner]]:
        """
        Rectify board region to square image.
        
        Args:
            image: Source image (H, W, C)
            corners: Detected corner coordinates (4, 2) [x, y]
            add_padding: Whether to add padding before rectification
            original_corners_dict: Optional original corners from manifest for tracking
            
        Returns:
            Tuple of:
                - rectified_image: Square image (output_size, output_size, C)
                - padded_corners: Corners after padding (4, 2)
                - corner_mapping: Dict mapping original corners to rectified positions
                
        Raises:
            ValueError: If corners are invalid
        """
        if not self.validate_corners(corners):
            raise ValueError("Invalid corners provided for rectification")
        
        h, w = image.shape[:2]
        
        # Add padding if requested
        if add_padding:
            padded_corners = self.add_padding(corners, (h, w))
        else:
            padded_corners = order_points(corners)
        
        # Get corner mapping for ground truth rotation
        corner_mapping = self.get_corner_mapping(padded_corners, original_corners_dict)
        
        # Define destination points (square)
        dst_points = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1]
        ], dtype=np.float32)
        
        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(padded_corners, dst_points)
        
        # Apply transform
        rectified = cv2.warpPerspective(
            image,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        
        return rectified, padded_corners, corner_mapping
    
    def rectify_batch(
        self,
        images: np.ndarray,
        corners_batch: np.ndarray,
        add_padding: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Rectify a batch of images.
        
        Args:
            images: Batch of images (B, H, W, C)
            corners_batch: Batch of corners (B, 4, 2)
            add_padding: Whether to add padding
            
        Returns:
            Tuple of:
                - rectified_images: (B, output_size, output_size, C)
                - padded_corners_batch: (B, 4, 2)
                - corner_mappings: List of corner mapping dicts
        """
        batch_size = images.shape[0]
        c = images.shape[3] if len(images.shape) == 4 else 1
        
        rectified_images = np.zeros(
            (batch_size, self.output_size, self.output_size, c),
            dtype=images.dtype,
        )
        padded_corners_batch = np.zeros((batch_size, 4, 2), dtype=np.float32)
        corner_mappings = []
        
        for i in range(batch_size):
            try:
                rectified, padded, mapping = self.rectify(
                    images[i],
                    corners_batch[i],
                    add_padding=add_padding,
                )
                rectified_images[i] = rectified
                padded_corners_batch[i] = padded
                corner_mappings.append(mapping)
            except ValueError as e:
                # Invalid corners, return black image and identity mapping
                print(f"Warning: Failed to rectify image {i}: {e}")
                padded_corners_batch[i] = order_points(corners_batch[i])
                corner_mappings.append({
                    'top_left': Corner.TOP_LEFT,
                    'top_right': Corner.TOP_RIGHT,
                    'bottom_right': Corner.BOTTOM_RIGHT,
                    'bottom_left': Corner.BOTTOM_LEFT,
                })
        
        return rectified_images, padded_corners_batch, corner_mappings


def rotate_position_grid(
    position_grid: np.ndarray,
    corner_mapping: Dict[str, Corner],
) -> np.ndarray:
    """
    Rotate position grid according to corner mapping.
    
    When the board is viewed from different angles, we need to rotate
    the ground truth grid to match the rectified image orientation.
    
    Args:
        position_grid: Position array of shape (C, 19, 19) or (2, 19, 19)
        corner_mapping: Mapping from original corners to rectified positions
        
    Returns:
        Rotated position grid with same shape as input
    """
    # Determine rotation by checking where top_left ended up
    dest_corner = corner_mapping['top_left']
    
    # Calculate number of 90° rotations needed
    if dest_corner == Corner.TOP_LEFT:
        k = 0  # No rotation
    elif dest_corner == Corner.TOP_RIGHT:
        k = 3  # 270° CCW = 90° CW
    elif dest_corner == Corner.BOTTOM_RIGHT:
        k = 2  # 180°
    elif dest_corner == Corner.BOTTOM_LEFT:
        k = 1  # 90° CCW
    else:
        k = 0
    
    # Rotate grid (rotate in the spatial dimensions)
    # np.rot90 rotates counter-clockwise by default
    rotated = np.rot90(position_grid, k=k, axes=(1, 2))
    
    # Make a contiguous copy to avoid negative strides
    # (PyTorch doesn't support negative strides)
    return np.ascontiguousarray(rotated)


def calculate_padding_pixels(
    corners: np.ndarray,
    padding_ratio: float = 0.02,
) -> float:
    """
    Calculate the padding in pixels for given corners.
    
    Args:
        corners: Corner coordinates (4, 2)
        padding_ratio: Padding as fraction of board size
        
    Returns:
        Padding amount in pixels
    """
    ordered = order_points(corners)
    center = ordered.mean(axis=0)
    distances = np.linalg.norm(ordered - center, axis=1)
    avg_distance = distances.mean()
    return avg_distance * padding_ratio
