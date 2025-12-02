"""Tests for evaluation metrics."""

import pytest
import torch
import numpy as np

from training.metrics import (
    compute_corner_error,
    compute_board_iou,
    compute_position_accuracy,
    compute_confusion_matrix,
    MetricsTracker,
)


class TestComputeCornerError:
    """Tests for compute_corner_error function."""

    def test_zero_error(self):
        """Test with identical predictions."""
        pred = torch.rand(4, 8)
        target = pred.clone()
        image_size = (640, 640)
        
        metrics = compute_corner_error(pred, target, image_size)
        
        assert "mean_corner_error" in metrics
        assert metrics["mean_corner_error"] == 0.0

    def test_error_computation(self):
        """Test error computation with different predictions."""
        pred = torch.tensor([[0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6]])
        target = torch.tensor([[0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.6, 0.6]]) + 0.1
        image_size = (640, 640)
        
        metrics = compute_corner_error(pred, target, image_size)
        
        assert metrics["mean_corner_error"] > 0
        assert "top_left_error" in metrics
        assert "top_right_error" in metrics
        assert "bottom_left_error" in metrics
        assert "bottom_right_error" in metrics

    def test_with_3d_input(self):
        """Test with (B, 4, 2) input shape."""
        pred = torch.rand(4, 4, 2)
        target = torch.rand(4, 4, 2)
        image_size = (640, 640)
        
        metrics = compute_corner_error(pred, target, image_size)
        
        assert "mean_corner_error" in metrics
        assert metrics["mean_corner_error"] >= 0


class TestComputeBoardIoU:
    """Tests for compute_board_iou function."""

    def test_perfect_iou(self):
        """Test with identical corners."""
        corners = torch.rand(4, 4, 2)
        iou = compute_board_iou(corners, corners)
        
        assert iou == pytest.approx(1.0, abs=1e-5)

    def test_zero_iou(self):
        """Test with non-overlapping corners."""
        pred = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]])
        target = torch.tensor([[[0.9, 0.9], [1.0, 0.9], [0.9, 1.0], [1.0, 1.0]]])
        
        iou = compute_board_iou(pred, target)
        
        assert iou >= 0 and iou < 0.1

    def test_partial_overlap(self):
        """Test with partially overlapping corners."""
        pred = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]])
        target = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]])
        
        iou = compute_board_iou(pred, target)
        
        assert iou > 0 and iou < 1

    def test_single_sample(self):
        """Test with single sample (2D input)."""
        pred = torch.rand(4, 2)
        target = torch.rand(4, 2)
        
        iou = compute_board_iou(pred, target)
        
        assert isinstance(iou, float)
        assert iou >= 0 and iou <= 1


class TestComputePositionAccuracy:
    """Tests for compute_position_accuracy function."""

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        batch_size = 4
        pred = torch.zeros(batch_size, 4, 19, 19)
        pred[:, 0, :, :] = 10.0  # All empty (class 0)
        target = torch.zeros(batch_size, 19, 19, dtype=torch.long)
        
        metrics = compute_position_accuracy(pred, target)
        
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["empty_accuracy"] == 1.0

    def test_zero_accuracy(self):
        """Test with completely wrong predictions."""
        batch_size = 2
        pred = torch.zeros(batch_size, 4, 19, 19)
        pred[:, 0, :, :] = 10.0  # Predict all empty
        target = torch.ones(batch_size, 19, 19, dtype=torch.long)  # All black
        
        metrics = compute_position_accuracy(pred, target)
        
        assert metrics["overall_accuracy"] == 0.0

    def test_per_class_accuracy(self):
        """Test per-class accuracy computation."""
        batch_size = 2
        pred = torch.zeros(batch_size, 4, 19, 19)
        # Half empty, half black
        pred[:, 0, :10, :] = 10.0
        pred[:, 1, 10:, :] = 10.0
        
        target = torch.zeros(batch_size, 19, 19, dtype=torch.long)
        target[:, 10:, :] = 1
        
        metrics = compute_position_accuracy(pred, target)
        
        assert "empty_accuracy" in metrics
        assert "black_accuracy" in metrics
        assert metrics["empty_accuracy"] == 1.0
        assert metrics["black_accuracy"] == 1.0


class TestComputeConfusionMatrix:
    """Tests for compute_confusion_matrix function."""

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        batch_size = 2
        pred = torch.randn(batch_size, 4, 19, 19)
        target = torch.randint(0, 4, (batch_size, 19, 19))
        
        cm = compute_confusion_matrix(pred, target, num_classes=4)
        
        assert cm.shape == (4, 4)
        assert cm.dtype == np.int64

    def test_confusion_matrix_values(self):
        """Test confusion matrix values."""
        batch_size = 1
        pred = torch.zeros(batch_size, 4, 19, 19)
        pred[:, 0, :, :] = 10.0  # All predict class 0
        target = torch.zeros(batch_size, 19, 19, dtype=torch.long)  # All class 0
        
        cm = compute_confusion_matrix(pred, target, num_classes=4)
        
        # All should be in (0, 0) position
        assert cm[0, 0] == 19 * 19
        assert cm.sum() == 19 * 19


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert len(tracker.metrics) == 0
        assert len(tracker.counts) == 0

    def test_update(self):
        """Test updating metrics."""
        tracker = MetricsTracker()
        
        metrics = {"loss": 0.5, "accuracy": 0.9}
        tracker.update(metrics, count=4)
        
        assert "loss" in tracker.metrics
        assert "accuracy" in tracker.metrics

    def test_compute_average(self):
        """Test computing average metrics."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 1.0}, count=2)
        tracker.update({"loss": 2.0}, count=2)
        
        averaged = tracker.compute()
        assert averaged["loss"] == 1.5

    def test_reset(self):
        """Test resetting tracker."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 1.0}, count=1)
        tracker.reset()
        
        assert len(tracker.metrics) == 0
        assert len(tracker.counts) == 0

    def test_weighted_average(self):
        """Test weighted averaging with different counts."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 1.0}, count=1)
        tracker.update({"loss": 2.0}, count=3)
        
        averaged = tracker.compute()
        # (1.0 * 1 + 2.0 * 3) / 4 = 1.75
        assert averaged["loss"] == 1.75

    def test_summary_string(self):
        """Test getting summary string."""
        tracker = MetricsTracker()
        
        tracker.update({"loss": 0.5, "accuracy": 0.9}, count=1)
        
        summary = tracker.get_summary_string()
        assert "loss" in summary
        assert "accuracy" in summary
