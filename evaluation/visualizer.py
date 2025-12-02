"""Visualization utilities for dataset and predictions."""

from typing import Optional, Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 5,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw corner points on an image.

    Args:
        image: Image array (H, W, 3)
        corners: Corner coordinates array (4, 2) in format [x, y]
        color: Circle color (R, G, B)
        radius: Circle radius
        thickness: Circle thickness (-1 for filled)

    Returns:
        Image with corners drawn
    """
    image = image.copy()

    for i, (x, y) in enumerate(corners):
        cv2.circle(image, (int(x), int(y)), radius, color, thickness)
        # Add label
        cv2.putText(
            image,
            str(i),
            (int(x) + 10, int(y) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return image


def draw_grid(
    image: np.ndarray,
    intersections: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """
    Draw grid intersections on an image.

    Args:
        image: Image array (H, W, 3)
        intersections: Grid intersection coordinates (19, 19, 2) in format [x, y]
        color: Point color (R, G, B)
        radius: Point radius

    Returns:
        Image with grid drawn
    """
    image = image.copy()

    # Draw horizontal and vertical lines
    for i in range(19):
        # Horizontal lines
        start = intersections[i, 0]
        end = intersections[i, 18]
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, 1)

        # Vertical lines
        start = intersections[0, i]
        end = intersections[18, i]
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, 1)

    # Draw star points (3-3, 3-15, etc.)
    star_points = [
        (3, 3),
        (3, 9),
        (3, 15),
        (9, 3),
        (9, 9),
        (9, 15),
        (15, 3),
        (15, 9),
        (15, 15),
    ]
    for i, j in star_points:
        x, y = intersections[i, j]
        cv2.circle(image, (int(x), int(y)), radius + 2, color, -1)

    return image


def visualize_position_grid(grid: np.ndarray, size: int = 400) -> np.ndarray:
    """
    Render position grid as an image.

    Args:
        grid: Position grid (4, 19, 19) or (19, 19) with class indices
        size: Output image size

    Returns:
        Rendered image (size, size, 3)
    """
    # Convert to class indices if one-hot
    if grid.ndim == 3:
        if isinstance(grid, torch.Tensor):
            grid = grid.cpu().numpy()
        class_grid = np.argmax(grid, axis=0)
    else:
        class_grid = grid

    # Create image
    cell_size = size // 19
    image = np.ones((size, size, 3), dtype=np.uint8) * 220  # Light gray background

    # Draw grid lines
    for i in range(20):
        pos = i * cell_size
        cv2.line(image, (pos, 0), (pos, size), (100, 100, 100), 1)
        cv2.line(image, (0, pos), (size, pos), (100, 100, 100), 1)

    # Draw stones
    stone_radius = cell_size // 3
    for i in range(19):
        for j in range(19):
            class_idx = class_grid[i, j]
            center = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)

            if class_idx == 1:  # Black stone
                cv2.circle(image, center, stone_radius, (0, 0, 0), -1)
            elif class_idx == 2:  # White stone
                cv2.circle(image, center, stone_radius, (255, 255, 255), -1)
                cv2.circle(image, center, stone_radius, (0, 0, 0), 1)
            elif class_idx == 3:  # Occluded
                cv2.circle(image, center, stone_radius, (255, 0, 0), 2)

    return image


def visualize_prediction_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: str = "Prediction vs Ground Truth",
) -> np.ndarray:
    """
    Create side-by-side visualization of prediction and ground truth.

    Args:
        image: Original board image (H, W, 3)
        ground_truth: Ground truth grid (4, 19, 19) or (19, 19)
        prediction: Predicted grid (4, 19, 19) or (19, 19)
        title: Plot title

    Returns:
        Combined visualization image
    """
    gt_vis = visualize_position_grid(ground_truth)
    pred_vis = visualize_position_grid(prediction)

    # Resize image to match visualizations
    image_resized = cv2.resize(image, (gt_vis.shape[1], gt_vis.shape[0]))

    # Combine horizontally
    combined = np.hstack([image_resized, gt_vis, pred_vis])

    # Add labels
    h, w = combined.shape[:2]
    label_height = 30
    labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8) + 255
    labeled[label_height:] = combined

    # Add text labels
    cv2.putText(
        labeled, "Original", (w // 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    cv2.putText(
        labeled,
        "Ground Truth",
        (w // 2, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        labeled,
        "Prediction",
        (5 * w // 6 - 50, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )

    return labeled


def log_images_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    images: List[np.ndarray],
    global_step: int,
    max_images: int = 8,
):
    """
    Log images to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the images
        images: List of images (H, W, 3) in RGB format
        global_step: Global step number
        max_images: Maximum number of images to log
    """
    images = images[:max_images]

    # Convert to tensor format (N, C, H, W)
    image_tensors = []
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        image_tensors.append(img_tensor)

    # Stack and log
    if image_tensors:
        image_grid = torch.stack(image_tensors)
        writer.add_images(tag, image_grid, global_step)


def log_prediction_to_tensorboard(
    writer: SummaryWriter,
    images: torch.Tensor,
    ground_truths: torch.Tensor,
    predictions: torch.Tensor,
    global_step: int,
    prefix: str = "val",
):
    """
    Log predictions to TensorBoard with ground truth comparison.

    Args:
        writer: TensorBoard SummaryWriter
        images: Batch of images (B, 3, H, W)
        ground_truths: Ground truth grids (B, 4, 19, 19)
        predictions: Predicted grids (B, 4, 19, 19)
        global_step: Global step number
        prefix: Prefix for tag (e.g., 'train' or 'val')
    """
    batch_size = min(4, images.shape[0])  # Log at most 4 images

    vis_images = []
    for i in range(batch_size):
        # Convert tensors to numpy
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)

        gt = ground_truths[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        # Create visualization
        vis = visualize_prediction_comparison(img, gt, pred)
        vis_images.append(vis)

    log_images_to_tensorboard(writer, f"{prefix}/predictions", vis_images, global_step)


def compute_accuracy_per_class(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> dict:
    """
    Compute per-class accuracy.

    Args:
        predictions: Predicted grids (B, 4, 19, 19)
        ground_truth: Ground truth grids (B, 4, 19, 19)

    Returns:
        Dictionary with per-class accuracies
    """
    # Convert to class indices
    pred_classes = torch.argmax(predictions, dim=1)  # (B, 19, 19)
    gt_classes = torch.argmax(ground_truth, dim=1)  # (B, 19, 19)

    accuracies = {}
    class_names = ["empty", "black", "white", "occluded"]

    for class_idx, class_name in enumerate(class_names):
        mask = gt_classes == class_idx
        if mask.sum() > 0:
            correct = (pred_classes[mask] == class_idx).sum().item()
            total = mask.sum().item()
            accuracies[class_name] = correct / total
        else:
            accuracies[class_name] = 0.0

    # Overall accuracy
    accuracies["overall"] = (pred_classes == gt_classes).float().mean().item()

    return accuracies
