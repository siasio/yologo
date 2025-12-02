"""Trainer for board detection models."""

from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.tensorboard import SummaryWriter

from models.board_detector.base import BaseBoardDetector
from training.losses import BaseBoardDetectionLoss
from training.metrics import (
    compute_mask_iou,
    compute_mask_accuracy,
    compute_dice_coefficient,
    compute_corner_error,
    MetricsTracker,
)


class BoardDetectorTrainer:
    """Trainer for segmentation-based board detection."""

    def __init__(
        self,
        model: BaseBoardDetector,
        criterion: BaseBoardDetectionLoss,
        optimizer: Optimizer,
        device: str = "cuda",
        scheduler: Optional[_LRScheduler] = None,
        grad_clip: float = 1.0,
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Board detection model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Optional learning rate scheduler
            grad_clip: Gradient clipping value
            tensorboard_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        # TensorBoard
        self.writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            log_interval: Log metrics every N steps

        Returns:
            Dictionary of averaged training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            logits = self.model(images)

            # Compute loss (loss function handles upsampling internally)
            loss_dict = self.criterion(logits, {"mask": masks})
            total_loss = loss_dict["total_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Compute metrics (upsample logits to match mask resolution)
            with torch.no_grad():
                logits_upsampled = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                iou = compute_mask_iou(logits_upsampled, masks).mean()
                accuracy = compute_mask_accuracy(logits_upsampled, masks).mean()
                dice = compute_dice_coefficient(logits_upsampled, masks).mean()

            # Update metrics
            batch_metrics = {
                "total_loss": total_loss.item(),
                **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
                "iou": iou.item(),
                "accuracy": accuracy.item(),
                "dice": dice.item(),
            }
            metrics_tracker.update(batch_metrics, count=images.size(0))

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "iou": f"{iou.item():.4f}",
            })

            # Log to TensorBoard
            if self.writer and (batch_idx % log_interval == 0):
                self.writer.add_scalar("train_step/total_loss", total_loss.item(), self.global_step)
                self.writer.add_scalar("train_step/iou", iou.item(), self.global_step)
                self.writer.add_scalar("train_step/accuracy", accuracy.item(), self.global_step)

            self.global_step += 1

        return metrics_tracker.compute()

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        max_viz_samples: int = 12,
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            max_viz_samples: Maximum number of samples to visualize

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        # Collect samples for visualization
        viz_samples = []

        pbar = tqdm(val_loader, desc=f"Epoch {self.epoch} [Val]")

        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # Forward pass
            logits = self.model(images)

            # Compute loss (loss function handles upsampling internally)
            loss_dict = self.criterion(logits, {"mask": masks})

            # Compute metrics (upsample logits to match mask resolution)
            logits_upsampled = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            iou = compute_mask_iou(logits_upsampled, masks).mean()
            accuracy = compute_mask_accuracy(logits_upsampled, masks).mean()
            dice = compute_dice_coefficient(logits_upsampled, masks).mean()

            # Update metrics
            batch_metrics = {
                "total_loss": loss_dict["total_loss"].item(),
                **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
                "iou": iou.item(),
                "accuracy": accuracy.item(),
                "dice": dice.item(),
            }
            metrics_tracker.update(batch_metrics, count=images.size(0))

            # Collect samples for visualization
            if len(viz_samples) < max_viz_samples:
                # Get predictions
                pred_dict = self.model.predict(images, return_corners=True)

                for i in range(min(images.size(0), max_viz_samples - len(viz_samples))):
                    viz_samples.append({
                        "image": images[i].cpu(),
                        "mask_gt": masks[i].cpu(),
                        "mask_pred": pred_dict["mask"][i].cpu(),
                        "mask_prob": pred_dict["mask_prob"][i].cpu(),
                        "corners_pred": pred_dict.get("corners", [None])[i].cpu() if "corners" in pred_dict else None,
                        "corners_gt": batch.get("corners", [None])[i].cpu() if "corners" in batch else None,
                    })

            pbar.set_postfix({"loss": f"{batch_metrics['total_loss']:.4f}", "iou": f"{iou.item():.4f}"})

        # Log visualizations
        if self.writer and viz_samples:
            self._log_visualizations(viz_samples)

        return metrics_tracker.compute()

    def _log_visualizations(self, samples: list):
        """
        Create and log visualizations to TensorBoard.

        Args:
            samples: List of sample dictionaries
        """
        n_samples = len(samples)
        n_cols = 4  # Image, GT Mask, Pred Mask, Overlay
        n_rows = n_samples

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, sample in enumerate(samples):
            # Denormalize image
            image = sample["image"].permute(1, 2, 0).numpy()
            image = (image * 255).astype("uint8")

            mask_gt = sample["mask_gt"][0].numpy()
            mask_pred = sample["mask_pred"][0].numpy()
            mask_prob = sample["mask_prob"][0].numpy()

            # Image
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Input Image")
            axes[idx, 0].axis("off")

            # Ground truth mask
            axes[idx, 1].imshow(image)
            axes[idx, 1].imshow(mask_gt, alpha=0.5, cmap="Greens")
            axes[idx, 1].set_title("GT Mask")
            axes[idx, 1].axis("off")

            # Predicted mask (binary)
            axes[idx, 2].imshow(image)
            axes[idx, 2].imshow(mask_pred, alpha=0.5, cmap="Blues")
            axes[idx, 2].set_title("Pred Mask")
            axes[idx, 2].axis("off")

            # Overlay with corners
            axes[idx, 3].imshow(image)
            axes[idx, 3].imshow(mask_prob, alpha=0.3, cmap="hot")

            # Draw corners if available
            corners_gt = sample.get("corners_gt")
            corners_pred = sample.get("corners_pred")

            if corners_gt is not None and corners_gt.sum() > 0:
                corners_gt_np = corners_gt.numpy()
                axes[idx, 3].scatter(corners_gt_np[:, 0], corners_gt_np[:, 1], 
                                    c="green", s=100, marker="o", label="GT")

            if corners_pred is not None and corners_pred.sum() > 0:
                corners_pred_np = corners_pred.numpy()
                axes[idx, 3].scatter(corners_pred_np[:, 0], corners_pred_np[:, 1],
                                    c="red", s=80, marker="x", linewidths=3, label="Pred")

            axes[idx, 3].set_title("Probability + Corners")
            axes[idx, 3].axis("off")
            if corners_gt is not None or corners_pred is not None:
                axes[idx, 3].legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        # Log to TensorBoard
        self.writer.add_figure("val/predictions", fig, self.epoch)
        plt.close(fig)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary of training history
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": [], "val_iou": []}
        patience_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log epoch metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, IoU: {train_metrics['iou']:.4f}")
            print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, IoU: {val_metrics['iou']:.4f}")

            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_metrics["total_loss"], epoch)
                self.writer.add_scalar("epoch/val_loss", val_metrics["total_loss"], epoch)
                self.writer.add_scalar("epoch/val_iou", val_metrics["iou"], epoch)
                self.writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("epoch/val_dice", val_metrics["dice"], epoch)
                if self.scheduler:
                    self.writer.add_scalar("epoch/learning_rate", self.scheduler.get_last_lr()[0], epoch)

            # Save checkpoint
            if checkpoint_dir:
                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self.save_checkpoint(checkpoint_dir / "best_model.pt")
                    patience_counter = 0
                    print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
                else:
                    patience_counter += 1

                # Save last checkpoint
                self.save_checkpoint(checkpoint_dir / "last_model.pt")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Update history
            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["total_loss"])
            history["val_iou"].append(val_metrics["iou"])

        if self.writer:
            self.writer.close()

        return history

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.epoch}")
