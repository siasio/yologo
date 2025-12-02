"""Dataset for board detection task."""

from typing import Dict, List, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.manifest_parser import (
    parse_manifest,
    get_image_path,
    get_grid_corners,
    get_board_corners,
)
from data.preprocessor import load_image


def create_mask_from_corners(
    corners: np.ndarray,
    image_shape: tuple,
) -> np.ndarray:
    """
    Create a binary mask from corner coordinates.

    Args:
        corners: Array of shape (4, 2) with corner coordinates [x, y]
        image_shape: (H, W) of the image

    Returns:
        Binary mask of shape (H, W) with board region marked as 1
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert corners to int32 for cv2.fillPoly
    corners_int = corners.astype(np.int32).reshape((-1, 1, 2))
    
    # Fill polygon
    cv2.fillPoly(mask, [corners_int], color=1)
    
    return mask


class BoardDetectionDataset(Dataset):
    """
    Dataset for training segmentation-based board detection network.

    Returns images with binary masks marking the board region.
    """

    def __init__(
        self,
        manifest_paths: List[str],
        transform: Optional[Callable] = None,
        use_grid_corners: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            manifest_paths: List of paths to manifest JSON files
            transform: Optional albumentations transform (should support masks)
            use_grid_corners: If True, use grid corners; if False, use board corners
        """
        self.manifest_paths = manifest_paths
        self.transform = transform
        self.use_grid_corners = use_grid_corners

        # Load all manifests
        self.manifests = []
        for path in manifest_paths:
            try:
                manifest = parse_manifest(path)
                self.manifests.append(manifest)
            except Exception as e:
                print(f"Warning: Failed to load manifest {path}: {e}")

        print(f"Loaded {len(self.manifests)} manifests for board detection")

    def __len__(self) -> int:
        return len(self.manifests)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - mask: Binary mask tensor of shape (1, H, W)
                - corners: Tensor of shape (4, 2) with corner coordinates
                - image_path: str
        """
        manifest = self.manifests[idx]

        # Load image
        image_path = get_image_path(manifest)
        image = load_image(image_path, color_mode="RGB")

        # Get corners
        if self.use_grid_corners:
            corners = get_grid_corners(manifest)
        else:
            corners = get_board_corners(manifest)

        # Create mask from corners
        h, w = image.shape[:2]
        mask = create_mask_from_corners(corners, (h, w))

        # Apply transforms
        if self.transform is not None:
            # Albumentations handles both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image to tensor
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1] and convert to (C, H, W)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Convert mask to tensor (1, H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        # Note: We don't transform corners through augmentations
        # They're kept for reference but not used in training
        corners = torch.from_numpy(corners)

        return {
            "image": image,
            "mask": mask,
            "corners": corners,
            "image_path": image_path,
        }
