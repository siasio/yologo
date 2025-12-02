"""Tests for board detection model."""

import pytest
import torch
import torch.nn as nn

from models.board_detector import MobileNetBoardDetector, create_board_detector


class TestMobileNetBoardDetector:
    """Tests for MobileNetBoardDetector model."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = MobileNetBoardDetector(pretrained=False)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'corner_head')

    def test_model_output_shape(self):
        """Test model output shape."""
        model = MobileNetBoardDetector(pretrained=False, input_size=640)
        model.eval()
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 640, 640)
        
        # Forward pass
        output = model(x)
        
        # Check output shape (B, 8)
        assert output.shape == (batch_size, 8)
        
        # Check output range (should be [0, 1] due to sigmoid)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_predict_method(self):
        """Test predict method."""
        model = MobileNetBoardDetector(pretrained=False)
        model.eval()
        
        # Single image
        x = torch.randn(3, 640, 640)
        corners = model.predict(x)
        
        # Should return (4, 2) tensor
        assert corners.shape == (4, 2)
        assert torch.all(corners >= 0) and torch.all(corners <= 1)

    def test_predict_with_denormalization(self):
        """Test predict with denormalization."""
        model = MobileNetBoardDetector(pretrained=False)
        model.eval()
        
        x = torch.randn(3, 640, 640)
        image_size = (640, 640)
        corners = model.predict(x, image_size=image_size)
        
        # Should return denormalized coordinates
        assert corners.shape == (4, 2)
        assert torch.all(corners >= 0) and torch.all(corners <= 640)

    def test_predict_batch(self):
        """Test predict method with batch."""
        model = MobileNetBoardDetector(pretrained=False)
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 640, 640)
        corners = model.predict(x)
        
        # Should return (B, 4, 2) tensor
        assert corners.shape == (batch_size, 4, 2)

    def test_save_load(self, tmp_path):
        """Test model save and load."""
        model = MobileNetBoardDetector(pretrained=False)
        
        # Save
        save_path = tmp_path / "model.pth"
        model.save(str(save_path))
        assert save_path.exists()
        
        # Load
        model2 = MobileNetBoardDetector(pretrained=False)
        model2.load(str(save_path))
        
        # Check if weights match
        model.eval()
        model2.eval()
        x = torch.randn(1, 3, 640, 640)
        out1 = model(x)
        out2 = model2(x)
        
        assert torch.allclose(out1, out2)


class TestCreateBoardDetector:
    """Tests for create_board_detector factory function."""

    def test_create_from_config(self):
        """Test creating model from config."""
        config = {
            "model": {
                "pretrained": False,
                "input_size": 640,
            }
        }
        
        model = create_board_detector(config)
        assert isinstance(model, MobileNetBoardDetector)
        assert model.input_size == 640

    def test_create_with_defaults(self):
        """Test creating model with default config."""
        config = {}
        model = create_board_detector(config)
        assert isinstance(model, MobileNetBoardDetector)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestModelCUDA:
    """Tests for model on CUDA device."""

    def test_model_on_cuda(self):
        """Test model on CUDA."""
        model = MobileNetBoardDetector(pretrained=False)
        model = model.cuda()
        
        x = torch.randn(2, 3, 640, 640).cuda()
        output = model(x)
        
        assert output.is_cuda
        assert output.shape == (2, 8)
