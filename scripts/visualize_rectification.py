#!/usr/bin/env python3
"""Visualize the board rectification pipeline.

This script demonstrates the complete rectification process:
1. Original image with detected corners
2. Original image with padded corners
3. Rectified square image
4. Position grid overlay on rectified image
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from data.manifest_parser import (
    find_manifests,
    parse_manifest,
    get_image_path,
    get_grid_corners,
    load_ground_truth_array,
)
from data.preprocessor import load_image, convert_position_array_to_grid
from data.board_rectifier import BoardRectifier, rotate_position_grid
from evaluation.visualizer import visualize_position_grid


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw corners and connecting lines on image.
    
    Args:
        image: Image to draw on
        corners: Corner coordinates (4, 2)
        color: Line color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with corners drawn
    """
    img = image.copy()
    
    # Draw lines connecting corners
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw corner points
    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int))
        cv2.circle(img, pt, 5, color, -1)
        
        # Label corners
        label_pos = (pt[0] + 10, pt[1] - 10)
        labels = ['TL', 'TR', 'BR', 'BL']
        cv2.putText(
            img,
            labels[i],
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    
    return img


def visualize_rectification(
    manifest_path: str,
    output_size: int = 224,
    padding_ratio: float = 0.02,
    save_path: str = None,
):
    """
    Visualize the complete rectification pipeline for a single image.
    
    Args:
        manifest_path: Path to manifest JSON
        output_size: Size of rectified image
        padding_ratio: Padding ratio for rectification
        save_path: Optional path to save figure
    """
    # Load manifest and data
    manifest = parse_manifest(manifest_path)
    image_path = get_image_path(manifest)
    image = load_image(image_path, color_mode="RGB")
    corners = get_grid_corners(manifest)
    position_array = load_ground_truth_array(manifest)
    
    # Get original corners from manifest
    corners_dict = manifest["ground_truth"]["grid_corners"]["pixel_coordinates"]
    original_corners = {
        'top_left': np.array(corners_dict['top_left']),
        'top_right': np.array(corners_dict['top_right']),
        'bottom_right': np.array(corners_dict['bottom_right']),
        'bottom_left': np.array(corners_dict['bottom_left']),
    }
    
    # Initialize rectifier
    rectifier = BoardRectifier(
        output_size=output_size,
        padding_ratio=padding_ratio,
    )
    
    # Get padded corners (without rectification)
    h, w = image.shape[:2]
    padded_corners = rectifier.add_padding(corners, (h, w))
    
    # Perform rectification
    rectified, _, corner_mapping = rectifier.rectify(
        image,
        corners,
        add_padding=True,
        original_corners_dict=original_corners,
    )
    
    # Convert and rotate position grid
    position_grid = convert_position_array_to_grid(position_array)
    rotated_grid = rotate_position_grid(position_grid, corner_mapping)
    
    # Create visualization overlays
    img_with_corners = draw_corners(image, corners, color=(0, 255, 0))
    img_with_padded = draw_corners(image, padded_corners, color=(255, 0, 0))
    
    # Visualize position grid on rectified image
    # visualize_position_grid returns the visualization, not an overlay
    # Let's create a simple overlay ourselves
    grid_vis = visualize_position_grid(rotated_grid, size=output_size)
    # Blend with rectified image
    alpha = 0.6
    rectified_with_grid = cv2.addWeighted(rectified, alpha, grid_vis, 1 - alpha, 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Original with detected corners
    axes[0, 0].imshow(img_with_corners)
    axes[0, 0].set_title('1. Detected Corners (Green)')
    axes[0, 0].axis('off')
    
    # 2. Original with padded corners
    axes[0, 1].imshow(img_with_padded)
    axes[0, 1].set_title(f'2. Padded Corners (Red, {padding_ratio*100:.1f}%)')
    axes[0, 1].axis('off')
    
    # 3. Rectified image
    axes[1, 0].imshow(rectified)
    axes[1, 0].set_title(f'3. Rectified ({output_size}×{output_size})')
    axes[1, 0].axis('off')
    
    # 4. Rectified with position grid
    axes[1, 1].imshow(rectified_with_grid)
    axes[1, 1].set_title('4. With Position Grid Overlay')
    axes[1, 1].axis('off')
    
    # Add info text
    num_stones = manifest.get("metadata", {}).get("num_stones", 0)
    rotation_info = f"Corner mapping: top_left → {corner_mapping['top_left'].name}"
    fig.suptitle(
        f'Board Rectification Pipeline\n'
        f'Image: {Path(manifest_path).stem}\n'
        f'Stones: {num_stones} | {rotation_info}',
        fontsize=12,
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize board rectification pipeline'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='example_assets',
        help='Directory containing manifest files',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/rectification_vis',
        help='Directory to save visualizations',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize',
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=224,
        help='Size of rectified output image',
    )
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.02,
        help='Padding ratio (fraction of board size)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show visualizations interactively instead of saving',
    )
    
    args = parser.parse_args()
    
    # Find manifests
    manifests = find_manifests(args.data_dir)
    print(f"Found {len(manifests)} manifests in {args.data_dir}")
    
    if len(manifests) == 0:
        print("No manifests found!")
        return
    
    # Select random samples
    np.random.seed(42)
    selected = np.random.choice(
        manifests,
        size=min(args.num_samples, len(manifests)),
        replace=False,
    )
    
    # Create output directory
    if not args.show:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to {output_dir}")
    
    # Visualize each sample
    for i, manifest_path in enumerate(selected):
        print(f"\nProcessing {i+1}/{len(selected)}: {Path(manifest_path).name}")
        
        save_path = None
        if not args.show:
            save_path = output_dir / f"rectification_{i+1:02d}.png"
        
        try:
            visualize_rectification(
                manifest_path,
                output_size=args.output_size,
                padding_ratio=args.padding_ratio,
                save_path=save_path,
            )
        except Exception as e:
            print(f"Error processing {manifest_path}: {e}")
            continue
    
    print(f"\nCompleted visualization of {len(selected)} samples")


if __name__ == '__main__':
    main()
