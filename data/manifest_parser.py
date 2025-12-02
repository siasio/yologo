"""Manifest parsing utilities for loading dataset metadata."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


def parse_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Parse a single manifest JSON file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Dictionary containing manifest data

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        json.JSONDecodeError: If manifest is invalid JSON
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    return manifest


def find_manifests(data_root: str, pattern: str = "**/*_manifest.json") -> List[str]:
    """
    Find all manifest files matching a pattern.

    Args:
        data_root: Root directory to search
        pattern: Glob pattern for manifest files

    Returns:
        List of absolute paths to manifest files
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    manifest_paths = sorted(data_root.glob(pattern))
    return [str(p.absolute()) for p in manifest_paths]


def load_ground_truth_array(manifest: Dict[str, Any]) -> np.ndarray:
    """
    Load ground truth position array from manifest.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        Numpy array of shape (2, 19, 19) with black and white stone positions
    """
    gt_path = manifest["ground_truth"]["position_array"]["path"]

    # Try absolute path first, then relative to manifest
    gt_path = Path(gt_path)
    if not gt_path.exists():
        # Try absolute_path field if available
        abs_path = manifest["ground_truth"]["position_array"].get("absolute_path")
        if abs_path:
            gt_path = Path(abs_path)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    return np.load(gt_path)


def get_image_path(manifest: Dict[str, Any]) -> str:
    """
    Get the image path from manifest, trying absolute path first.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        Absolute path to the image file
    """
    # Try absolute path first
    abs_path = manifest["rendered_image"].get("absolute_path")
    if abs_path and Path(abs_path).exists():
        return abs_path

    # Fall back to relative path
    rel_path = manifest["rendered_image"]["path"]
    if Path(rel_path).exists():
        return rel_path

    raise FileNotFoundError(f"Image file not found: {rel_path}")


def get_grid_corners(manifest: Dict[str, Any]) -> np.ndarray:
    """
    Extract grid corner coordinates from manifest.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        Numpy array of shape (4, 2) with corner coordinates [x, y]
        Order: top_left, top_right, bottom_right, bottom_left
    """
    corners_dict = manifest["ground_truth"]["grid_corners"]["pixel_coordinates"]

    corners = np.array(
        [
            corners_dict["top_left"],
            corners_dict["top_right"],
            corners_dict["bottom_right"],
            corners_dict["bottom_left"],
        ],
        dtype=np.float32,
    )

    return corners


def get_board_corners(manifest: Dict[str, Any]) -> np.ndarray:
    """
    Extract board corner coordinates from manifest.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        Numpy array of shape (4, 2) with corner coordinates [x, y]
        Order: top_left, top_right, bottom_right, bottom_left
    """
    corners_dict = manifest["ground_truth"]["wooden_board_corners"]["pixel_coordinates"]

    corners = np.array(
        [
            corners_dict["top_left"],
            corners_dict["top_right"],
            corners_dict["bottom_right"],
            corners_dict["bottom_left"],
        ],
        dtype=np.float32,
    )

    return corners


def validate_manifest(manifest: Dict[str, Any]) -> bool:
    """
    Validate that manifest has required fields.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        ["rendered_image", "path"],
        ["ground_truth", "position_array", "path"],
        ["ground_truth", "grid_corners", "pixel_coordinates"],
        ["ground_truth", "wooden_board_corners", "pixel_coordinates"],
    ]

    for field_path in required_fields:
        current = manifest
        for field in field_path:
            if field not in current:
                return False
            current = current[field]

    return True


def get_num_stones(manifest: Dict[str, Any]) -> int:
    """
    Get number of stones from manifest metadata.

    Args:
        manifest: Parsed manifest dictionary

    Returns:
        Number of stones in the position
    """
    return manifest.get("metadata", {}).get("num_stones", 0)
