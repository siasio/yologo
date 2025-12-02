"""Tests for board rectification utilities."""

import numpy as np
import pytest

from data.board_rectifier import (
    BoardRectifier,
    Corner,
    rotate_position_grid,
    calculate_padding_pixels,
)


class TestBoardRectifier:
    """Tests for BoardRectifier class."""
    
    def test_init(self):
        """Test initialization with default and custom parameters."""
        rectifier = BoardRectifier()
        assert rectifier.output_size == 224
        assert rectifier.padding_ratio == 0.02
        assert rectifier.min_board_size == 50
        
        rectifier = BoardRectifier(output_size=512, padding_ratio=0.05, min_board_size=100)
        assert rectifier.output_size == 512
        assert rectifier.padding_ratio == 0.05
        assert rectifier.min_board_size == 100
    
    def test_validate_corners_valid(self):
        """Test corner validation with valid corners."""
        rectifier = BoardRectifier()
        
        # Valid square corners
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        assert rectifier.validate_corners(corners)
    
    def test_validate_corners_invalid_shape(self):
        """Test corner validation with invalid shape."""
        rectifier = BoardRectifier()
        
        # Wrong shape
        corners = np.array([[100, 100], [400, 100]], dtype=np.float32)
        assert not rectifier.validate_corners(corners)
    
    def test_validate_corners_too_small(self):
        """Test corner validation with too small board."""
        rectifier = BoardRectifier(min_board_size=100)
        
        # Board too small
        corners = np.array([
            [100, 100],
            [120, 100],
            [120, 120],
            [100, 120],
        ], dtype=np.float32)
        
        assert not rectifier.validate_corners(corners)
    
    def test_validate_corners_nan(self):
        """Test corner validation with NaN values."""
        rectifier = BoardRectifier()
        
        corners = np.array([
            [100, 100],
            [np.nan, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        assert not rectifier.validate_corners(corners)
    
    def test_add_padding(self):
        """Test padding addition."""
        rectifier = BoardRectifier(padding_ratio=0.1)
        
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        image_shape = (512, 512)
        padded = rectifier.add_padding(corners, image_shape)
        
        # Check shape
        assert padded.shape == (4, 2)
        
        # Check that corners moved outward
        center = corners.mean(axis=0)
        for i in range(4):
            dist_orig = np.linalg.norm(corners[i] - center)
            dist_padded = np.linalg.norm(padded[i] - center)
            assert dist_padded > dist_orig
        
        # Check within bounds
        assert np.all(padded[:, 0] >= 0)
        assert np.all(padded[:, 0] < 512)
        assert np.all(padded[:, 1] >= 0)
        assert np.all(padded[:, 1] < 512)
    
    def test_add_padding_near_edge(self):
        """Test padding when corners are near image edges."""
        rectifier = BoardRectifier(padding_ratio=0.2)
        
        # Corners near edges
        corners = np.array([
            [10, 10],
            [500, 10],
            [500, 500],
            [10, 500],
        ], dtype=np.float32)
        
        image_shape = (512, 512)
        padded = rectifier.add_padding(corners, image_shape)
        
        # Should be clipped to image bounds
        assert np.all(padded[:, 0] >= 0)
        assert np.all(padded[:, 0] < 512)
        assert np.all(padded[:, 1] >= 0)
        assert np.all(padded[:, 1] < 512)
    
    def test_get_corner_mapping_identity(self):
        """Test corner mapping with no rotation."""
        rectifier = BoardRectifier()
        
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        mapping = rectifier.get_corner_mapping(corners)
        
        assert mapping['top_left'] == Corner.TOP_LEFT
        assert mapping['top_right'] == Corner.TOP_RIGHT
        assert mapping['bottom_right'] == Corner.BOTTOM_RIGHT
        assert mapping['bottom_left'] == Corner.BOTTOM_LEFT
    
    def test_get_corner_mapping_with_original(self):
        """Test corner mapping with original corners provided."""
        rectifier = BoardRectifier()
        
        # Detected corners (in order: TL, TR, BR, BL)
        detected = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        # Original corners (same as detected)
        original = {
            'top_left': np.array([100, 100]),
            'top_right': np.array([400, 100]),
            'bottom_right': np.array([400, 400]),
            'bottom_left': np.array([100, 400]),
        }
        
        mapping = rectifier.get_corner_mapping(detected, original)
        
        assert mapping['top_left'] == Corner.TOP_LEFT
        assert mapping['top_right'] == Corner.TOP_RIGHT
        assert mapping['bottom_right'] == Corner.BOTTOM_RIGHT
        assert mapping['bottom_left'] == Corner.BOTTOM_LEFT
    
    def test_get_corner_mapping_rotated_180(self):
        """Test corner mapping with 180° rotation."""
        rectifier = BoardRectifier()
        
        # Detected corners after 180° rotation
        # What was bottom_right is now top_left
        detected = np.array([
            [100, 100],  # was bottom_right
            [400, 100],  # was bottom_left
            [400, 400],  # was top_left
            [100, 400],  # was top_right
        ], dtype=np.float32)
        
        # Original corners (before rotation)
        original = {
            'top_left': np.array([400, 400]),
            'top_right': np.array([100, 400]),
            'bottom_right': np.array([100, 100]),
            'bottom_left': np.array([400, 100]),
        }
        
        mapping = rectifier.get_corner_mapping(detected, original)
        
        assert mapping['top_left'] == Corner.BOTTOM_RIGHT
        assert mapping['top_right'] == Corner.BOTTOM_LEFT
        assert mapping['bottom_right'] == Corner.TOP_LEFT
        assert mapping['bottom_left'] == Corner.TOP_RIGHT
    
    def test_rectify_square_board(self):
        """Test rectification of a square board."""
        rectifier = BoardRectifier(output_size=224, padding_ratio=0.0)
        
        # Create test image
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Draw square region in middle
        image[150:350, 150:350] = [0, 255, 0]  # Green square
        
        # Corners of the green square
        corners = np.array([
            [150, 150],
            [350, 150],
            [350, 350],
            [150, 350],
        ], dtype=np.float32)
        
        rectified, padded_corners, mapping = rectifier.rectify(
            image, corners, add_padding=False
        )
        
        # Check output shape
        assert rectified.shape == (224, 224, 3)
        
        # Check that center is green
        center_pixel = rectified[112, 112]
        assert center_pixel[1] > 200  # Green channel should be high
        
        # Check corner mapping is identity
        assert mapping['top_left'] == Corner.TOP_LEFT
    
    def test_rectify_with_padding(self):
        """Test rectification with padding."""
        rectifier = BoardRectifier(output_size=224, padding_ratio=0.05)
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        image[200:300, 200:300] = [255, 0, 0]  # Red square
        
        corners = np.array([
            [200, 200],
            [300, 200],
            [300, 300],
            [200, 300],
        ], dtype=np.float32)
        
        rectified, padded_corners, mapping = rectifier.rectify(
            image, corners, add_padding=True
        )
        
        assert rectified.shape == (224, 224, 3)
        
        # Check that padded corners are different from original
        assert not np.allclose(padded_corners, corners)
        
        # Padded corners should be further from center
        center = corners.mean(axis=0)
        for i in range(4):
            orig_dist = np.linalg.norm(corners[i] - center)
            padded_dist = np.linalg.norm(padded_corners[i] - center)
            assert padded_dist >= orig_dist
    
    def test_rectify_perspective_distorted(self):
        """Test rectification of perspective-distorted board."""
        rectifier = BoardRectifier(output_size=224, padding_ratio=0.0)
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Trapezoid (perspective-distorted rectangle)
        corners = np.array([
            [150, 100],
            [400, 120],
            [450, 400],
            [100, 380],
        ], dtype=np.float32)
        
        rectified, padded_corners, mapping = rectifier.rectify(
            image, corners, add_padding=False
        )
        
        # Should produce square output
        assert rectified.shape == (224, 224, 3)
        assert padded_corners.shape == (4, 2)
    
    def test_rectify_invalid_corners(self):
        """Test rectification with invalid corners."""
        rectifier = BoardRectifier(min_board_size=100)
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        # Corners too small
        corners = np.array([
            [250, 250],
            [260, 250],
            [260, 260],
            [250, 260],
        ], dtype=np.float32)
        
        with pytest.raises(ValueError):
            rectifier.rectify(image, corners)
    
    def test_rectify_batch(self):
        """Test batch rectification."""
        rectifier = BoardRectifier(output_size=224, padding_ratio=0.0)
        
        # Create batch of images
        batch_size = 3
        images = np.ones((batch_size, 512, 512, 3), dtype=np.uint8) * 255
        
        # Different corners for each image
        corners_batch = np.array([
            [[100, 100], [400, 100], [400, 400], [100, 400]],
            [[150, 150], [350, 150], [350, 350], [150, 350]],
            [[200, 100], [450, 120], [430, 450], [180, 430]],
        ], dtype=np.float32)
        
        rectified_batch, padded_batch, mappings = rectifier.rectify_batch(
            images, corners_batch, add_padding=False
        )
        
        assert rectified_batch.shape == (batch_size, 224, 224, 3)
        assert padded_batch.shape == (batch_size, 4, 2)
        assert len(mappings) == batch_size


class TestRotatePositionGrid:
    """Tests for position grid rotation."""
    
    def test_rotate_no_rotation(self):
        """Test with no rotation (identity mapping)."""
        grid = np.random.rand(4, 19, 19).astype(np.float32)
        
        mapping = {
            'top_left': Corner.TOP_LEFT,
            'top_right': Corner.TOP_RIGHT,
            'bottom_right': Corner.BOTTOM_RIGHT,
            'bottom_left': Corner.BOTTOM_LEFT,
        }
        
        rotated = rotate_position_grid(grid, mapping)
        
        assert np.allclose(rotated, grid)
    
    def test_rotate_90_ccw(self):
        """Test 90° counter-clockwise rotation."""
        # Create identifiable grid
        grid = np.zeros((2, 19, 19), dtype=np.float32)
        grid[0, 0, 0] = 1.0  # Mark top-left
        grid[1, 0, 18] = 1.0  # Mark top-right
        
        # Top-left moved to bottom-left (90° CCW)
        mapping = {
            'top_left': Corner.BOTTOM_LEFT,
            'top_right': Corner.TOP_LEFT,
            'bottom_right': Corner.TOP_RIGHT,
            'bottom_left': Corner.BOTTOM_RIGHT,
        }
        
        rotated = rotate_position_grid(grid, mapping)
        
        # Check marked positions moved correctly
        assert rotated[0, 18, 0] == 1.0  # Top-left -> bottom-left
        assert rotated[1, 0, 0] == 1.0   # Top-right -> top-left
    
    def test_rotate_180(self):
        """Test 180° rotation."""
        grid = np.zeros((2, 19, 19), dtype=np.float32)
        grid[0, 0, 0] = 1.0  # Top-left
        
        # Top-left moved to bottom-right
        mapping = {
            'top_left': Corner.BOTTOM_RIGHT,
            'top_right': Corner.BOTTOM_LEFT,
            'bottom_right': Corner.TOP_LEFT,
            'bottom_left': Corner.TOP_RIGHT,
        }
        
        rotated = rotate_position_grid(grid, mapping)
        
        assert rotated[0, 18, 18] == 1.0  # Top-left -> bottom-right
    
    def test_rotate_270_ccw(self):
        """Test 270° counter-clockwise (90° clockwise) rotation."""
        grid = np.zeros((2, 19, 19), dtype=np.float32)
        grid[0, 0, 0] = 1.0  # Top-left
        
        # Top-left moved to top-right
        mapping = {
            'top_left': Corner.TOP_RIGHT,
            'top_right': Corner.BOTTOM_RIGHT,
            'bottom_right': Corner.BOTTOM_LEFT,
            'bottom_left': Corner.TOP_LEFT,
        }
        
        rotated = rotate_position_grid(grid, mapping)
        
        assert rotated[0, 0, 18] == 1.0  # Top-left -> top-right
    
    def test_rotate_preserves_shape(self):
        """Test that rotation preserves grid shape."""
        for channels in [2, 4]:
            grid = np.random.rand(channels, 19, 19).astype(np.float32)
            
            mapping = {
                'top_left': Corner.BOTTOM_RIGHT,
                'top_right': Corner.BOTTOM_LEFT,
                'bottom_right': Corner.TOP_LEFT,
                'bottom_left': Corner.TOP_RIGHT,
            }
            
            rotated = rotate_position_grid(grid, mapping)
            
            assert rotated.shape == grid.shape


class TestCalculatePaddingPixels:
    """Tests for padding calculation."""
    
    def test_padding_calculation(self):
        """Test padding calculation for square corners."""
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        padding = calculate_padding_pixels(corners, padding_ratio=0.1)
        
        # For a square with side 300, distance from center to corner is ~212
        # Padding should be about 21.2 pixels
        assert 15 < padding < 30
    
    def test_padding_different_ratios(self):
        """Test that larger ratio gives more padding."""
        corners = np.array([
            [100, 100],
            [400, 100],
            [400, 400],
            [100, 400],
        ], dtype=np.float32)
        
        padding1 = calculate_padding_pixels(corners, padding_ratio=0.02)
        padding2 = calculate_padding_pixels(corners, padding_ratio=0.1)
        
        assert padding2 > padding1
        assert padding2 / padding1 == pytest.approx(5.0, rel=0.01)
