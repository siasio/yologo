#!/usr/bin/env python
"""Visualize dataset samples for inspection and debugging."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.config_loader import load_training_config
from data.manifest_parser import find_manifests
from data.board_detection_dataset import BoardDetectionDataset
from data.position_recognition_dataset import PositionRecognitionDataset
from data.transforms import get_simple_resize_transform
from evaluation.visualizer import (
    draw_corners,
    draw_grid,
    visualize_position_grid,
)
from utils.geometry import compute_grid_intersections


def visualize_board_detection(dataset, num_samples: int, output_dir: Path):
    """Visualize board detection samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing {num_samples} board detection samples...")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Convert tensor back to numpy
        image = sample["image"].numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        grid_corners = sample["grid_corners"].numpy()

        # Draw corners
        vis_image = draw_corners(image, grid_corners, color=(0, 255, 0), radius=8)

        # Save
        output_path = output_dir / f"board_detection_{i:03d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")


def visualize_position_recognition(dataset, num_samples: int, output_dir: Path):
    """Visualize position recognition samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing {num_samples} position recognition samples...")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Convert tensor back to numpy
        image = sample["image"].numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        position_grid = sample["position_grid"].numpy()
        num_stones = sample["num_stones"]

        # Create grid visualization
        grid_vis = visualize_position_grid(position_grid, size=image.shape[0])

        # Draw grid on original image
        intersections = compute_grid_intersections(
            board_size=19, image_size=(image.shape[1], image.shape[0]), margin=0.05
        )
        image_with_grid = draw_grid(image.copy(), intersections, color=(255, 255, 0))

        # Combine images side by side
        combined = np.hstack([image, image_with_grid, grid_vis])

        # Add text label
        label = f"Sample {i} | Stones: {num_stones}"
        cv2.putText(
            combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Save
        output_path = output_dir / f"position_recognition_{i:03d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing manifest files",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["board_detection", "position_recognition", "both"],
        default="both",
        help="Which task to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to visualize per task",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*_manifest.json",
        help="Glob pattern for finding manifests",
    )

    args = parser.parse_args()

    # Find manifests
    print(f"Searching for manifests in: {args.data_root}")
    print(f"Pattern: {args.pattern}")
    manifest_paths = find_manifests(args.data_root, args.pattern)
    print(f"Found {len(manifest_paths)} manifests")

    if len(manifest_paths) == 0:
        print("No manifests found!")
        return

    # Limit to num_samples for faster iteration
    manifest_paths = manifest_paths[: args.num_samples * 2]

    output_dir = Path(args.output_dir)

    # Visualize board detection
    if args.task in ["board_detection", "both"]:
        print("\n=== Board Detection ===")
        transform = get_simple_resize_transform(640)
        dataset = BoardDetectionDataset(manifest_paths, transform=transform)
        visualize_board_detection(
            dataset, args.num_samples, output_dir / "board_detection"
        )

    # Visualize position recognition
    if args.task in ["position_recognition", "both"]:
        print("\n=== Position Recognition ===")
        transform = get_simple_resize_transform(224)
        dataset = PositionRecognitionDataset(
            manifest_paths, transform=transform, rectify=True, output_size=224
        )
        visualize_position_recognition(
            dataset, args.num_samples, output_dir / "position_recognition"
        )

    print(f"\n✅ Done! Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
