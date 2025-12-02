"""Checkpoint management for model training."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            monitor: Metric to monitor for saving best checkpoints
            mode: 'min' or 'max' - whether lower or higher metric is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode

        self.best_checkpoints = []  # List of (score, path) tuples
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save a checkpoint if it's better than existing ones.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
            extra_state: Optional additional state to save

        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        current_score = metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return None

        # Check if this is a better checkpoint
        is_better = (self.mode == "min" and current_score < self.best_score) or (
            self.mode == "max" and current_score > self.best_score
        )

        if is_better:
            self.best_score = current_score

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "monitor": self.monitor,
            "score": current_score,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update best checkpoints list
        self._update_best_checkpoints(current_score, checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def save_last_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save the last checkpoint (always saved, not subject to max_checkpoints).

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
            extra_state: Optional additional state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        checkpoint_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved last checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def _update_best_checkpoints(self, score: float, path: Path) -> None:
        """Update the list of best checkpoints."""
        self.best_checkpoints.append((score, path))

        # Sort by score
        reverse = self.mode == "max"
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.best_checkpoints) <= self.max_checkpoints:
            return

        # Keep only top max_checkpoints
        checkpoints_to_remove = self.best_checkpoints[self.max_checkpoints :]
        self.best_checkpoints = self.best_checkpoints[: self.max_checkpoints]

        # Delete files
        for _, path in checkpoints_to_remove:
            if path.exists() and path.name != "last.pth":
                path.unlink()
                logger.debug(f"Removed old checkpoint: {path}")

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0][1]
