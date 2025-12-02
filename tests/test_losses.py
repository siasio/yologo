"""Tests for loss functions."""

import pytest
import torch

from training.losses import (
    CornerLoss,
    BoardDetectionLoss,
    PositionRecognitionLoss,
    create_board_detection_loss,
    create_position_recognition_loss,
)


class TestCornerLoss:
    """Tests for CornerLoss."""

    def test_mse_loss(self):
        """Test MSE loss variant."""
        loss_fn = CornerLoss(loss_type="mse")
        
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_smooth_l1_loss(self):
        """Test Smooth L1 loss variant."""
        loss_fn = CornerLoss(loss_type="smooth_l1")
        
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_wing_loss(self):
        """Test Wing loss variant."""
        loss_fn = CornerLoss(loss_type="wing")
        
        pred = torch.randn(4, 8)
        target = torch.randn(4, 8)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_loss_with_3d_input(self):
        """Test loss with (B, 4, 2) input."""
        loss_fn = CornerLoss(loss_type="mse")
        
        pred = torch.randn(4, 4, 2)
        target = torch.randn(4, 4, 2)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_zero_loss(self):
        """Test that identical predictions give zero loss."""
        loss_fn = CornerLoss(loss_type="mse")
        
        pred = torch.randn(4, 8)
        
        loss = loss_fn(pred, pred)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestBoardDetectionLoss:
    """Tests for BoardDetectionLoss."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        loss_fn = BoardDetectionLoss(corner_loss_type="mse", shape_regularization=False)
        
        pred = torch.rand(4, 8)  # Normalized [0, 1]
        target = torch.rand(4, 8)
        
        losses = loss_fn(pred, target)
        
        assert "corner_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"].item() >= 0

    def test_loss_with_shape_regularization(self):
        """Test loss with shape regularization."""
        loss_fn = BoardDetectionLoss(
            corner_loss_type="smooth_l1",
            shape_regularization=True,
            shape_weight=0.1,
        )
        
        pred = torch.rand(4, 8)
        target = torch.rand(4, 8)
        
        losses = loss_fn(pred, target)
        
        assert "corner_loss" in losses
        assert "shape_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"].item() >= 0

    def test_shape_loss_computation(self):
        """Test shape regularization loss."""
        loss_fn = BoardDetectionLoss(shape_regularization=True)
        
        # Create square-ish corners
        pred = torch.tensor([
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],  # Roughly square
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],
        ])
        target = torch.tensor([
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9],
        ])
        
        losses = loss_fn(pred, target)
        assert losses["shape_loss"].item() >= 0


class TestPositionRecognitionLoss:
    """Tests for PositionRecognitionLoss."""

    def test_basic_loss(self):
        """Test basic cross-entropy loss."""
        loss_fn = PositionRecognitionLoss()
        
        batch_size = 4
        pred = torch.randn(batch_size, 4, 19, 19)  # Logits
        target = torch.randint(0, 4, (batch_size, 19, 19))  # Class indices
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_loss_with_class_weights(self):
        """Test loss with class weights."""
        weights = torch.tensor([1.0, 1.0, 1.0, 2.0])  # Higher weight for occluded
        loss_fn = PositionRecognitionLoss(class_weights=weights)
        
        batch_size = 4
        pred = torch.randn(batch_size, 4, 19, 19)
        target = torch.randint(0, 4, (batch_size, 19, 19))
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_loss_with_label_smoothing(self):
        """Test loss with label smoothing."""
        loss_fn = PositionRecognitionLoss(label_smoothing=0.1)
        
        batch_size = 4
        pred = torch.randn(batch_size, 4, 19, 19)
        target = torch.randint(0, 4, (batch_size, 19, 19))
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestLossFactoryFunctions:
    """Tests for loss factory functions."""

    def test_create_board_detection_loss(self):
        """Test creating board detection loss from config."""
        config = {
            "training": {
                "loss": {
                    "corner_loss_type": "smooth_l1",
                    "shape_regularization": True,
                    "shape_weight": 0.1,
                }
            }
        }
        
        loss_fn = create_board_detection_loss(config)
        assert isinstance(loss_fn, BoardDetectionLoss)

    def test_create_position_recognition_loss(self):
        """Test creating position recognition loss from config."""
        config = {
            "training": {
                "loss": {
                    "class_weights": [1.0, 1.0, 1.0, 2.0],
                    "label_smoothing": 0.0,
                }
            }
        }
        
        device = torch.device("cpu")
        loss_fn = create_position_recognition_loss(config, device)
        assert isinstance(loss_fn, PositionRecognitionLoss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLossCUDA:
    """Tests for losses on CUDA."""

    def test_board_detection_loss_cuda(self):
        """Test board detection loss on CUDA."""
        loss_fn = BoardDetectionLoss().cuda()
        
        pred = torch.rand(4, 8).cuda()
        target = torch.rand(4, 8).cuda()
        
        losses = loss_fn(pred, target)
        assert losses["total_loss"].is_cuda

    def test_position_recognition_loss_cuda(self):
        """Test position recognition loss on CUDA."""
        weights = torch.tensor([1.0, 1.0, 1.0, 2.0]).cuda()
        loss_fn = PositionRecognitionLoss(class_weights=weights)
        
        pred = torch.randn(2, 4, 19, 19).cuda()
        target = torch.randint(0, 4, (2, 19, 19)).cuda()
        
        loss = loss_fn(pred, target)
        assert loss.is_cuda
