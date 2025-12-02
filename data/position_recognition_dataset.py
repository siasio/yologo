"""Dataset for position recognition task."""

from typing import Dict, List, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from data.manifest_parser import (
    parse_manifest,
    get_image_path,
    get_grid_corners,
    load_ground_truth_array,
    get_num_stones,
)
from data.preprocessor import (
    load_image,
    convert_position_array_to_grid,
)
from data.board_rectifier import BoardRectifier, rotate_position_grid


class PositionRecognitionDataset(Dataset):
    """
    Dataset for training position recognition network.

    Returns rectified board images with position grids.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        transform: Optional[Callable] = None,
        rectify: bool = True,
        output_size: int = 224,
        padding_ratio: float = 0.02,
    ):
        """
        Initialize dataset.

        Args:
            manifest_paths: List of paths to manifest JSON files
            transform: Optional albumentations transform (applied after rectification)
            rectify: Whether to rectify boards using corner points
            output_size: Size of output square image
            padding_ratio: Padding ratio for BoardRectifier (default 2%)
        """
        self.manifest_paths = manifest_paths
        self.transform = transform
        self.rectify = rectify
        self.output_size = output_size
        
        # Initialize board rectifier
        self.rectifier = BoardRectifier(
            output_size=output_size,
            padding_ratio=padding_ratio,
        )

        # Load all manifests
        self.manifests = []
        for path in manifest_paths:
            try:
                manifest = parse_manifest(path)
                self.manifests.append(manifest)
            except Exception as e:
                print(f"Warning: Failed to load manifest {path}: {e}")

        print(f"Loaded {len(self.manifests)} manifests for position recognition")

    def __len__(self) -> int:
        return len(self.manifests)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W) - rectified board
                - position_grid: Tensor of shape (4, 19, 19) - one-hot encoded positions
                - num_stones: int - total number of stones
                - image_path: str
                - corner_mapping: Dict mapping original corners to rectified positions
        """
        manifest = self.manifests[idx]

        # Load image
        image_path = get_image_path(manifest)
        image = load_image(image_path, color_mode="RGB")

        # Load ground truth
        position_array = load_ground_truth_array(manifest)  # (2, 19, 19)

        # Rectify board if needed
        corner_mapping = None
        if self.rectify:
            grid_corners = get_grid_corners(manifest)
            
            # Get original corners from manifest for tracking
            original_corners_dict = manifest["ground_truth"]["grid_corners"]["pixel_coordinates"]
            original_corners = {
                'top_left': np.array(original_corners_dict['top_left']),
                'top_right': np.array(original_corners_dict['top_right']),
                'bottom_right': np.array(original_corners_dict['bottom_right']),
                'bottom_left': np.array(original_corners_dict['bottom_left']),
            }
            
            # Rectify with corner tracking
            image, _, corner_mapping = self.rectifier.rectify(
                image,
                grid_corners,
                add_padding=True,
                original_corners_dict=original_corners,
            )

        # Convert position array to 4-channel grid
        position_grid = convert_position_array_to_grid(position_array)  # (4, 19, 19)
        
        # Rotate position grid if board was rotated during rectification
        if corner_mapping is not None:
            position_grid = rotate_position_grid(position_grid, corner_mapping)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert image to tensor
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1] and convert to (C, H, W)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Convert position grid to tensor
        position_grid = torch.from_numpy(position_grid)

        # Get number of stones
        num_stones = get_num_stones(manifest)

        return {
            "image": image,
            "position_grid": position_grid,
            "num_stones": num_stones,
            "image_path": image_path,
            "corner_mapping": corner_mapping,
        }
