"""Training script for segmentation-based board detector."""

import argparse
from pathlib import Path
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.manifest_parser import find_manifests
from data.board_detection_dataset import BoardDetectionDataset
from data.transforms import get_board_detection_transforms
from models.board_detector import create_segmentation_detector
from training.losses import create_board_detection_loss
from training.board_detector_trainer import BoardDetectorTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train segmentation-based board detector")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/board_detector.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="example_assets",
        help="Path to data directory containing manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use small subset of data)",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_data_loaders(
    config: dict,
    data_dir: Path,
    debug: bool = False,
) -> tuple:
    """
    Create train and validation data loaders.

    Args:
        config: Configuration dictionary
        data_dir: Path to data directory with manifests
        debug: If True, use small subset of data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Find all manifest files
    manifest_files = find_manifests(str(data_dir), pattern="**/*_manifest.json")
    
    print(f"Found {len(manifest_files)} manifest files in {data_dir}")
    
    if len(manifest_files) == 0:
        raise ValueError(f"No manifest files found in {data_dir}")
    
    # Debug mode: use subset
    if debug:
        manifest_files = manifest_files[:20]
        print(f"Debug mode: using only {len(manifest_files)} samples")
    
    # Split dataset: 80% train, 20% val
    train_size = int(0.8 * len(manifest_files))
    train_manifests = manifest_files[:train_size]
    val_manifests = manifest_files[train_size:]
    
    # Create transforms
    train_transform = get_board_detection_transforms(config, train=True)
    val_transform = get_board_detection_transforms(config, train=False)
    
    # Get dataset config
    data_config = config.get("data", {})
    use_grid_corners = data_config.get("use_grid_corners", False)
    
    # Create datasets
    train_dataset = BoardDetectionDataset(
        manifest_paths=train_manifests,
        transform=train_transform,
        use_grid_corners=use_grid_corners,
    )
    
    val_dataset = BoardDetectionDataset(
        manifest_paths=val_manifests,
        transform=val_transform,
        use_grid_corners=use_grid_corners,
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 16)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_batch_size = config.get("validation", {}).get("batch_size", batch_size * 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"board_detector_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Segmentation-Based Board Detector Training")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Debug mode: {args.debug}")
    
    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Data directory: {data_dir}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        config=config,
        data_dir=data_dir,
        debug=args.debug,
    )
    
    # Create model
    print("\nCreating model...")
    model = create_segmentation_detector(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Create loss function
    print("\nCreating loss function...")
    criterion = create_board_detection_loss(config)
    
    # Create optimizer
    train_config = config["training"]
    optimizer_config = train_config["optimizer"]
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("lr", 0.0005),
        weight_decay=optimizer_config.get("weight_decay", 0.0001),
        betas=optimizer_config.get("betas", [0.9, 0.999]),
    )
    
    # Create scheduler
    scheduler_config = train_config.get("scheduler", {})
    if scheduler_config.get("type") == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 1e-5),
        )
    else:
        scheduler = None
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = BoardDetectorTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=str(device),
        scheduler=scheduler,
        grad_clip=train_config.get("grad_clip", 1.0),
        tensorboard_dir=str(output_dir / "tensorboard"),
    )
    
    # Train
    try:
        epochs = train_config.get("epochs", 100)
        early_stopping_config = train_config.get("early_stopping", {})
        patience = early_stopping_config.get("patience", 10)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Tensorboard: tensorboard --logdir {output_dir / 'tensorboard'}")
        print("-" * 80)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            checkpoint_dir=str(output_dir / "checkpoints"),
            early_stopping_patience=patience,
        )
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val IoU: {history['val_iou'][-1]:.4f}")
        print(f"Best model saved at: {output_dir / 'checkpoints' / 'best_model.pt'}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
