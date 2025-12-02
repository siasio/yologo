"""Tests for manifest parser."""

import pytest
import numpy as np
from pathlib import Path

from data.manifest_parser import (
    parse_manifest,
    find_manifests,
    load_ground_truth_array,
    get_image_path,
    get_grid_corners,
    get_board_corners,
    validate_manifest,
    get_num_stones,
)


# Use the example manifest from the project
EXAMPLE_MANIFEST = "example_assets/1ccb6d11-6aff-41de-b825-0c895df69016_manifest.json"


def test_parse_manifest():
    """Test parsing the example manifest."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)

    assert "rendered_image" in manifest
    assert "ground_truth" in manifest
    assert "metadata" in manifest


def test_validate_manifest():
    """Test manifest validation."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)
    assert validate_manifest(manifest) is True


def test_get_image_path():
    """Test getting image path from manifest."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)

    # Should try absolute path first
    image_path = get_image_path(manifest)
    assert isinstance(image_path, str)


def test_get_grid_corners():
    """Test extracting grid corners."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)
    corners = get_grid_corners(manifest)

    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32

    # Check that we have 4 distinct points
    assert len(np.unique(corners, axis=0)) == 4


def test_get_board_corners():
    """Test extracting board corners."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)
    corners = get_board_corners(manifest)

    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32


def test_load_ground_truth_array():
    """Test loading ground truth array."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)

    # This will fail if the absolute path doesn't exist, which is expected
    # since the example is from a different machine
    try:
        gt_array = load_ground_truth_array(manifest)
        assert gt_array.shape == (2, 19, 19)
    except FileNotFoundError:
        pytest.skip("Ground truth file not at absolute path (expected for example)")


def test_get_num_stones():
    """Test getting number of stones from metadata."""
    if not Path(EXAMPLE_MANIFEST).exists():
        pytest.skip("Example manifest not found")

    manifest = parse_manifest(EXAMPLE_MANIFEST)
    num_stones = get_num_stones(manifest)

    assert isinstance(num_stones, int)
    assert num_stones >= 0
    assert manifest["metadata"]["num_stones"] == 107


def test_find_manifests():
    """Test finding manifest files."""
    # Test with example_assets directory
    if not Path("example_assets").exists():
        pytest.skip("example_assets directory not found")

    manifests = find_manifests("example_assets", pattern="*_manifest.json")

    assert isinstance(manifests, list)
    assert len(manifests) >= 1  # Should find at least the example manifest


def test_find_manifests_missing_dir():
    """Test finding manifests in nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        find_manifests("/nonexistent/path")


def test_parse_nonexistent_manifest():
    """Test parsing nonexistent manifest."""
    with pytest.raises(FileNotFoundError):
        parse_manifest("nonexistent_manifest.json")
