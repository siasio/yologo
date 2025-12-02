"""Tests for preprocessor utilities."""

import pytest
import numpy as np
import torch

from data.preprocessor import (
    normalize_image,
    image_to_tensor,
    convert_position_array_to_grid,
    grid_to_class_indices,
    class_indices_to_grid,
    get_stone_positions,
    count_stones,
)


def test_normalize_image():
    """Test image normalization."""
    # Create random image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    normalized = normalize_image(image)

    # Check output range (approximately [-2, 2] due to ImageNet stats)
    assert normalized.min() >= -3
    assert normalized.max() <= 3
    assert normalized.dtype == np.float32


def test_image_to_tensor():
    """Test image to tensor conversion."""
    image = np.random.rand(100, 100, 3).astype(np.float32)

    tensor = image_to_tensor(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.float32


def test_convert_position_array_to_grid():
    """Test conversion from (2, 19, 19) to (4, 19, 19) format."""
    # Create test position array
    position_array = np.zeros((2, 19, 19), dtype=np.float32)

    # Add some black stones
    position_array[0, 3, 3] = 1
    position_array[0, 15, 15] = 1

    # Add some white stones
    position_array[1, 3, 15] = 1
    position_array[1, 15, 3] = 1

    grid = convert_position_array_to_grid(position_array)

    assert grid.shape == (4, 19, 19)
    assert grid.dtype == np.float32

    # Check black stones (channel 1)
    assert grid[1, 3, 3] == 1
    assert grid[1, 15, 15] == 1

    # Check white stones (channel 2)
    assert grid[2, 3, 15] == 1
    assert grid[2, 15, 3] == 1

    # Check empty intersections (channel 0)
    assert grid[0, 0, 0] == 1  # Empty intersection
    assert grid[0, 3, 3] == 0  # Occupied by black

    # Check occluded (channel 3) - should be all zeros
    assert np.all(grid[3] == 0)

    # Each intersection should have exactly one active channel
    assert np.allclose(grid.sum(axis=0), 1.0)


def test_grid_to_class_indices():
    """Test conversion from one-hot to class indices."""
    # Create one-hot grid
    grid = np.zeros((4, 19, 19), dtype=np.float32)
    grid[0] = 1  # All empty initially

    # Set some stones
    grid[0, 3, 3] = 0
    grid[1, 3, 3] = 1  # Black

    grid[0, 15, 15] = 0
    grid[2, 15, 15] = 1  # White

    indices = grid_to_class_indices(grid)

    assert indices.shape == (19, 19)
    assert indices.dtype == np.int64
    assert indices[3, 3] == 1  # Black
    assert indices[15, 15] == 2  # White
    assert indices[0, 0] == 0  # Empty


def test_class_indices_to_grid():
    """Test conversion from class indices to one-hot."""
    # Create class indices
    indices = np.zeros((19, 19), dtype=np.int64)
    indices[3, 3] = 1  # Black
    indices[15, 15] = 2  # White
    indices[10, 10] = 3  # Occluded

    grid = class_indices_to_grid(indices, num_classes=4)

    assert grid.shape == (4, 19, 19)
    assert grid[1, 3, 3] == 1  # Black
    assert grid[2, 15, 15] == 1  # White
    assert grid[3, 10, 10] == 1  # Occluded
    assert grid[0, 0, 0] == 1  # Empty


def test_get_stone_positions():
    """Test extracting stone positions."""
    # Create grid with some stones
    grid = np.zeros((4, 19, 19), dtype=np.float32)
    grid[0] = 1  # All empty

    # Add 3 black stones
    for i, j in [(3, 3), (3, 15), (15, 3)]:
        grid[0, i, j] = 0
        grid[1, i, j] = 1

    # Add 2 white stones
    for i, j in [(15, 15), (9, 9)]:
        grid[0, i, j] = 0
        grid[2, i, j] = 1

    black_pos, white_pos = get_stone_positions(grid)

    assert len(black_pos) == 3
    assert len(white_pos) == 2
    assert black_pos.shape[1] == 2
    assert white_pos.shape[1] == 2


def test_count_stones():
    """Test stone counting."""
    # Create grid
    grid = np.zeros((4, 19, 19), dtype=np.float32)
    grid[0] = 1

    # Add 5 black and 3 white stones
    num_black = 5
    num_white = 3

    for idx in range(num_black):
        i, j = idx * 2, idx * 2
        grid[0, i, j] = 0
        grid[1, i, j] = 1

    for idx in range(num_white):
        i, j = idx * 2 + 1, idx * 2 + 1
        grid[0, i, j] = 0
        grid[2, i, j] = 1

    black_count, white_count = count_stones(grid)

    assert black_count == num_black
    assert white_count == num_white


def test_convert_position_array_wrong_shape():
    """Test that wrong shape raises assertion."""
    wrong_shape = np.zeros((3, 19, 19))

    with pytest.raises(AssertionError):
        convert_position_array_to_grid(wrong_shape)
