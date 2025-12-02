"""Image preprocessing and data conversion utilities."""

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch


def load_image(image_path: str, color_mode: str = "RGB") -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to image file
        color_mode: 'RGB' or 'BGR'

    Returns:
        Image as numpy array (H, W, C)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    if color_mode == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normalize image with ImageNet statistics.

    Args:
        image: Image array (H, W, C) in range [0, 255]
        mean: Mean values per channel
        std: Std values per channel

    Returns:
        Normalized image in range approximately [-2, 2]
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std
    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image to PyTorch tensor.

    Args:
        image: Image array (H, W, C)

    Returns:
        Tensor of shape (C, H, W)
    """
    # Convert to float32 if needed
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # Transpose to (C, H, W)
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    return tensor


def convert_position_array_to_grid(position_array: np.ndarray) -> np.ndarray:
    """
    Convert position array from (2, 19, 19) to (4, 19, 19) format.

    Input format:
        Channel 0: Black stones (binary)
        Channel 1: White stones (binary)

    Output format:
        Channel 0: Empty intersections
        Channel 1: Black stones
        Channel 2: White stones
        Channel 3: Occluded (all zeros for now)

    Args:
        position_array: Array of shape (2, 19, 19)

    Returns:
        Array of shape (4, 19, 19) with one-hot encoding per intersection
    """
    assert position_array.shape == (
        2,
        19,
        19,
    ), f"Expected shape (2, 19, 19), got {position_array.shape}"

    # Create output array
    grid = np.zeros((4, 19, 19), dtype=np.float32)

    # Extract black and white positions
    black_positions = position_array[0]
    white_positions = position_array[1]

    # Channel 1: Black stones
    grid[1] = black_positions

    # Channel 2: White stones
    grid[2] = white_positions

    # Channel 0: Empty intersections (neither black nor white)
    occupied = black_positions + white_positions
    grid[0] = 1.0 - occupied

    # Channel 3: Occluded (all zeros for now - no occlusion in dataset)
    # grid[3] remains zero

    return grid


def grid_to_class_indices(grid: np.ndarray) -> np.ndarray:
    """
    Convert one-hot grid to class indices.

    Args:
        grid: Array of shape (4, 19, 19) with one-hot encoding

    Returns:
        Array of shape (19, 19) with class indices (0=empty, 1=black, 2=white, 3=occluded)
    """
    return np.argmax(grid, axis=0).astype(np.int64)


def class_indices_to_grid(indices: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Convert class indices to one-hot grid.

    Args:
        indices: Array of shape (19, 19) with class indices
        num_classes: Number of classes

    Returns:
        Array of shape (num_classes, 19, 19) with one-hot encoding
    """
    grid = np.zeros((num_classes, indices.shape[0], indices.shape[1]), dtype=np.float32)

    for c in range(num_classes):
        grid[c] = (indices == c).astype(np.float32)

    return grid


def get_stone_positions(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract stone positions from grid.

    Args:
        grid: Array of shape (4, 19, 19) or (19, 19) with class indices

    Returns:
        Tuple of (black_positions, white_positions) as (N, 2) arrays with (row, col)
    """
    if grid.ndim == 3:
        # One-hot format
        black_mask = grid[1]
        white_mask = grid[2]
    else:
        # Class indices format
        black_mask = grid == 1
        white_mask = grid == 2

    black_positions = np.argwhere(black_mask > 0.5)
    white_positions = np.argwhere(white_mask > 0.5)

    return black_positions, white_positions


def count_stones(grid: np.ndarray) -> Tuple[int, int]:
    """
    Count black and white stones in grid.

    Args:
        grid: Array of shape (4, 19, 19) or (19, 19)

    Returns:
        Tuple of (num_black, num_white)
    """
    black_pos, white_pos = get_stone_positions(grid)
    return len(black_pos), len(white_pos)
