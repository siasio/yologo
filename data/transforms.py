"""Data augmentation pipelines using Albumentations."""

from typing import Dict, Any, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_board_detection_transforms(
    config: Dict[str, Any], train: bool = True
) -> A.Compose:
    """
    Get augmentation pipeline for board detection task.

    Args:
        config: Configuration dictionary with augmentation settings
        train: Whether this is for training (applies augmentations) or validation

    Returns:
        Albumentations Compose object
    """
    aug_config = config.get("data", {}).get("board_detection_augmentation", {})

    if train:
        settings = aug_config.get("train", {})
        resize = settings.get("resize", {"height": 640, "width": 640})

        transforms = [
            A.Resize(height=resize["height"], width=resize["width"]),
            A.Rotate(limit=settings.get("rotation_limit", 15), border_mode=0, p=0.7),
            (
                A.Perspective(scale=(0.05, 0.1), p=0.5)
                if settings.get("perspective", True)
                else A.NoOp()
            ),
            A.ColorJitter(
                brightness=settings.get("brightness_limit", 0.2),
                contrast=settings.get("contrast_limit", 0.2),
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=settings.get("blur_limit", 7), p=0.3),
        ]
    else:
        # Validation: only resize
        settings = aug_config.get("val", {})
        resize = settings.get("resize", {"height": 640, "width": 640})
        transforms = [
            A.Resize(height=resize["height"], width=resize["width"]),
        ]

    # For segmentation, we need to handle masks not keypoints
    return A.Compose(transforms)


def get_position_recognition_transforms(
    config: Dict[str, Any], train: bool = True
) -> A.Compose:
    """
    Get augmentation pipeline for position recognition task.

    Args:
        config: Configuration dictionary with augmentation settings
        train: Whether this is for training or validation

    Returns:
        Albumentations Compose object
    """
    aug_config = config.get("data", {}).get("position_recognition_augmentation", {})

    if train:
        settings = aug_config.get("train", {})
        resize = settings.get("resize", {"height": 224, "width": 224})

        transforms = [
            A.Resize(height=resize["height"], width=resize["width"]),
            A.Rotate(
                limit=settings.get(
                    "rotation_limit", 5
                ),  # Small rotation for rectified boards
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=settings.get("brightness_limit", 0.3),
                contrast=settings.get("contrast_limit", 0.3),
                saturation=settings.get("saturation_limit", 0.2),
                hue=settings.get("hue_limit", 10),
                p=0.6,
            ),
            A.GaussianBlur(blur_limit=settings.get("blur_limit", 5), p=0.2),
            (
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
                if settings.get("gaussian_noise", 0.01)
                else A.NoOp()
            ),
        ]
    else:
        # Validation: only resize
        settings = aug_config.get("val", {})
        resize = settings.get("resize", {"height": 224, "width": 224})
        transforms = [
            A.Resize(height=resize["height"], width=resize["width"]),
        ]

    return A.Compose(transforms)


def get_simple_resize_transform(size: int = 224) -> A.Compose:
    """
    Get a simple resize transform (useful for testing).

    Args:
        size: Output size (square)

    Returns:
        Albumentations Compose object
    """
    return A.Compose(
        [
            A.Resize(height=size, width=size),
        ]
    )
