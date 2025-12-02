"""Geometric transformations for Go board processing."""

from typing import List, Tuple

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order starting from top-left.

    Args:
        points: Array of shape (4, 2) containing corner coordinates

    Returns:
        Ordered points as (top-left, top-right, bottom-right, bottom-left)
    """
    # Sort by y-coordinate
    sorted_pts = points[np.argsort(points[:, 1])]

    # Top two points (smaller y)
    top_pts = sorted_pts[:2]
    # Bottom two points (larger y)
    bottom_pts = sorted_pts[2:]

    # Sort top points by x (left, right)
    top_left, top_right = top_pts[np.argsort(top_pts[:, 0])]

    # Sort bottom points by x (left, right)
    bottom_left, bottom_right = bottom_pts[np.argsort(bottom_pts[:, 0])]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def compute_output_size(
    points: np.ndarray, aspect_ratio: float = 1.0
) -> Tuple[int, int]:
    """
    Compute output size for perspective transform based on corner points.

    Args:
        points: Ordered corner points (4, 2)
        aspect_ratio: Desired aspect ratio (width/height)

    Returns:
        Tuple of (width, height)
    """
    # Compute width as max of top and bottom edge lengths
    width_top = np.linalg.norm(points[1] - points[0])
    width_bottom = np.linalg.norm(points[2] - points[3])
    width = max(int(width_top), int(width_bottom))

    # Compute height as max of left and right edge lengths
    height_left = np.linalg.norm(points[3] - points[0])
    height_right = np.linalg.norm(points[2] - points[1])
    height = max(int(height_left), int(height_right))

    # Adjust for aspect ratio
    if aspect_ratio != 1.0:
        size = max(width, height)
        width = int(size * aspect_ratio)
        height = int(size / aspect_ratio)
    else:
        # For Go board, use the larger dimension to avoid cropping
        size = max(width, height)
        width = height = size

    return width, height


def rectify_board(
    image: np.ndarray, corners: np.ndarray, output_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Apply perspective transform to rectify a Go board to a square.

    Args:
        image: Input image
        corners: Board corner coordinates (4, 2)
        output_size: Desired output size (width, height)

    Returns:
        Rectified board image
    """
    # Order the points
    src_points = order_points(corners)

    # Define destination points (square)
    width, height = output_size
    dst_points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transform
    rectified = cv2.warpPerspective(image, matrix, output_size)

    return rectified


def get_perspective_transform_matrix(
    src_corners: np.ndarray, dst_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Compute perspective transform matrix from source corners to destination.

    Args:
        src_corners: Source corner coordinates (4, 2)
        dst_size: Destination image size (width, height)

    Returns:
        3x3 perspective transform matrix
    """
    src_points = order_points(src_corners)

    width, height = dst_size
    dst_points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(src_points, dst_points)


def apply_perspective_transform(
    image: np.ndarray, matrix: np.ndarray, output_size: Tuple[int, int]
) -> np.ndarray:
    """
    Apply a pre-computed perspective transform matrix to an image.

    Args:
        image: Input image
        matrix: 3x3 perspective transform matrix
        output_size: Output size (width, height)

    Returns:
        Transformed image
    """
    return cv2.warpPerspective(image, matrix, output_size)


def compute_grid_intersections(
    board_size: int = 19, image_size: Tuple[int, int] = (512, 512), margin: float = 0.05
) -> np.ndarray:
    """
    Compute pixel coordinates of grid intersections for a rectified board.

    Args:
        board_size: Number of intersections per side (19 for standard Go)
        image_size: Size of the rectified image (width, height)
        margin: Margin as fraction of image size

    Returns:
        Array of shape (board_size, board_size, 2) with (x, y) coordinates
    """
    width, height = image_size

    # Calculate usable space after margins
    margin_px_x = int(width * margin)
    margin_px_y = int(height * margin)

    usable_width = width - 2 * margin_px_x
    usable_height = height - 2 * margin_px_y

    # Calculate spacing between intersections
    spacing_x = usable_width / (board_size - 1)
    spacing_y = usable_height / (board_size - 1)

    # Generate grid coordinates
    intersections = np.zeros((board_size, board_size, 2), dtype=np.float32)

    for i in range(board_size):
        for j in range(board_size):
            x = margin_px_x + j * spacing_x
            y = margin_px_y + i * spacing_y
            intersections[i, j] = [x, y]

    return intersections


def distance_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        Distance
    """
    return float(np.linalg.norm(p1 - p2))


def points_to_list(points: np.ndarray) -> List[List[float]]:
    """
    Convert numpy array of points to list format.

    Args:
        points: Numpy array of shape (N, 2)

    Returns:
        List of [x, y] coordinates
    """
    return points.tolist()
