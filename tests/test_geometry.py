"""Tests for geometry utilities."""

import numpy as np
import pytest

from utils.geometry import (
    order_points,
    compute_output_size,
    rectify_board,
    compute_grid_intersections,
    distance_between_points,
    points_to_list,
)


def test_order_points():
    """Test ordering of corner points."""
    # Create unordered points
    points = np.array(
        [
            [100, 200],  # bottom-left
            [200, 100],  # top-right
            [100, 100],  # top-left
            [200, 200],  # bottom-right
        ],
        dtype=np.float32,
    )

    ordered = order_points(points)

    # Check order: top-left, top-right, bottom-right, bottom-left
    assert np.allclose(ordered[0], [100, 100])  # top-left
    assert np.allclose(ordered[1], [200, 100])  # top-right
    assert np.allclose(ordered[2], [200, 200])  # bottom-right
    assert np.allclose(ordered[3], [100, 200])  # bottom-left


def test_compute_output_size():
    """Test computation of output size from corners."""
    # Square corners
    points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

    width, height = compute_output_size(points)

    assert width == height
    assert width == 100


def test_compute_output_size_non_square():
    """Test output size computation for non-square."""
    points = np.array([[0, 0], [200, 0], [200, 100], [0, 100]], dtype=np.float32)

    width, height = compute_output_size(points)

    # Should use the larger dimension
    assert width == height
    assert width == 200


def test_rectify_board():
    """Test board rectification."""
    # Create a simple test image
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # Define corners (slightly rotated square)
    corners = np.array([[50, 60], [240, 50], [250, 240], [60, 250]], dtype=np.float32)

    rectified = rectify_board(image, corners, output_size=(224, 224))

    assert rectified.shape == (224, 224, 3)
    assert rectified.dtype == np.uint8


def test_compute_grid_intersections():
    """Test computation of grid intersection coordinates."""
    intersections = compute_grid_intersections(
        board_size=19, image_size=(512, 512), margin=0.05
    )

    assert intersections.shape == (19, 19, 2)

    # Check corners are within bounds
    assert intersections[0, 0, 0] > 0  # Top-left x
    assert intersections[0, 0, 1] > 0  # Top-left y
    assert intersections[18, 18, 0] < 512  # Bottom-right x
    assert intersections[18, 18, 1] < 512  # Bottom-right y

    # Check spacing is roughly uniform
    spacing_x = intersections[0, 1, 0] - intersections[0, 0, 0]
    spacing_y = intersections[1, 0, 1] - intersections[0, 0, 1]

    assert spacing_x > 0
    assert spacing_y > 0

    # Spacing should be similar for all intersections
    for i in range(18):
        next_spacing_x = intersections[0, i + 1, 0] - intersections[0, i, 0]
        assert np.isclose(spacing_x, next_spacing_x, rtol=0.01)


def test_compute_grid_intersections_different_sizes():
    """Test grid computation with different board and image sizes."""
    # Test 9x9 board
    intersections_9x9 = compute_grid_intersections(board_size=9, image_size=(256, 256))

    assert intersections_9x9.shape == (9, 9, 2)

    # Test 13x13 board
    intersections_13x13 = compute_grid_intersections(
        board_size=13, image_size=(384, 384)
    )

    assert intersections_13x13.shape == (13, 13, 2)


def test_distance_between_points():
    """Test distance calculation."""
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])

    distance = distance_between_points(p1, p2)

    assert np.isclose(distance, 5.0)


def test_distance_between_points_same():
    """Test distance between same points."""
    p1 = np.array([10, 20])
    p2 = np.array([10, 20])

    distance = distance_between_points(p1, p2)

    assert np.isclose(distance, 0.0)


def test_points_to_list():
    """Test conversion of numpy array to list."""
    points = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    points_list = points_to_list(points)

    assert isinstance(points_list, list)
    assert len(points_list) == 2
    assert points_list[0] == [1.5, 2.5]
    assert points_list[1] == [3.5, 4.5]


def test_order_points_random():
    """Test ordering with random points."""
    # Generate random quadrilateral
    np.random.seed(42)

    for _ in range(10):
        # Random corners in different quadrants
        tl = np.random.uniform(0, 50, 2)
        tr = (
            np.random.uniform(150, 200, 1).tolist()
            + np.random.uniform(0, 50, 1).tolist()
        )
        br = np.random.uniform(150, 200, 2)
        bl = (
            np.random.uniform(0, 50, 1).tolist()
            + np.random.uniform(150, 200, 1).tolist()
        )

        points = np.array([tl, tr, br, bl], dtype=np.float32)
        np.random.shuffle(points)

        ordered = order_points(points)

        # Top points should have smaller y than bottom points
        assert ordered[0, 1] < ordered[3, 1]
        assert ordered[1, 1] < ordered[2, 1]

        # Left points should have smaller x than right points
        assert ordered[0, 0] < ordered[1, 0]
        assert ordered[3, 0] < ordered[2, 0]
