# YoloGo Quick Reference

## Project Status

**Phase 1:** ✅ Complete (Project Setup)  
**Phase 2:** ✅ Complete (Data Pipeline)  
**Phase 3:** 📋 Ready to start (Model Implementation)  
**Lines of Code:** ~1,950 (utilities + data + eval + tests)  
**Test Coverage:** 44 tests (37 passed, 7 skipped)

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run tests
make test

# Format code
make format

# Run linting
make lint
```

## Key Modules

### Configuration
```python
from utils.config_loader import load_training_config

# Load complete config (base + data + model)
config = load_training_config('position_recognizer')
```

### Geometry
```python
from utils.geometry import rectify_board, compute_grid_intersections

# Rectify board from corner points
rectified = rectify_board(image, corners, output_size=(512, 512))

# Get 19x19 grid intersection coordinates
intersections = compute_grid_intersections(board_size=19, image_size=(512, 512))
```

### Reproducibility
```python
from utils.seed import set_seed

# Set all random seeds
set_seed(42, deterministic=True)
```

### Logging
```python
from utils.logger import setup_logger, get_tensorboard_writer

logger = setup_logger('training', log_dir='logs/', level='INFO')
writer = get_tensorboard_writer('logs/tensorboard', 'experiment_1')
```

### Checkpointing
```python
from utils.checkpoint import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='outputs/checkpoints',
    max_checkpoints=3,
    monitor='val_loss',
    mode='min'
)

# Save checkpoint
manager.save_checkpoint(model, optimizer, epoch, metrics)

# Load checkpoint
metadata = manager.load_checkpoint('last.pth', model, optimizer, device='cuda')
```

## Configuration Files

- `configs/base_config.yaml` - Project-wide settings
- `configs/data.yaml` - Dataset and augmentation config
- `configs/board_detector.yaml` - YOLO board detector config
- `configs/position_recognizer.yaml` - MobileNet position recognizer config

## Dataset Format

Manifests are JSON files containing:
```json
{
  "rendered_image": {"path": "...", "absolute_path": "..."},
  "ground_truth": {
    "position_array": {
      "path": "ground_truth.npy",
      "shape": [2, 19, 19]  // [black, white]
    },
    "grid_corners": {
      "pixel_coordinates": {
        "top_left": [x, y], "top_right": [x, y],
        "bottom_left": [x, y], "bottom_right": [x, y]
      }
    }
  }
}
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_geometry.py -v

# Run specific test
pytest tests/test_geometry.py::test_rectify_board -v
```

## Code Quality

```bash
# Format with black
make format

# Lint with ruff
make lint

# Clean artifacts
make clean
```

## Data Pipeline (Phase 2)

### Load Datasets
```python
from data.manifest_parser import find_manifests
from data.board_detection_dataset import BoardDetectionDataset
from data.position_recognition_dataset import PositionRecognitionDataset
from data.transforms import get_board_detection_transforms, get_position_recognition_transforms
from utils.config_loader import load_training_config

# Find manifests
config = load_training_config('position_recognizer')
manifests = find_manifests(config['paths']['data_root'], pattern="**/*_manifest.json")

# Board detection dataset
transform = get_board_detection_transforms(config, train=True)
board_dataset = BoardDetectionDataset(manifests, transform=transform)

# Position recognition dataset
transform = get_position_recognition_transforms(config, train=True)
position_dataset = PositionRecognitionDataset(manifests, transform=transform, rectify=True)
```

### Visualize Dataset
```bash
python scripts/visualize_dataset.py \
    --data-root /path/to/dataset \
    --task both \
    --num-samples 10 \
    --output-dir outputs/visualizations
```

## Next Steps (Phase 3)

1. Implement board detector model (YOLOv8-based)
2. Implement position recognizer model (MobileNetV3-based)  
3. Create training loop with TensorBoard logging
4. Add evaluation metrics and callbacks
5. Train both models on synthetic data

See `copilot_reports/phase2_completion.md` for Phase 2 summary.

## Important Notes

- Always activate venv before running scripts: `source .venv/bin/activate`
- Position array shape is (2, 19, 19): [black_stones, white_stones]
- Target output is (4, 19, 19): [empty, black, white, occluded]
- Grid corners are in pixel (x, y) coordinates from top-left
- CUDA is available: PyTorch 2.9.1 with CUDA 12.8
- Use config files for all hyperparameters
- Follow TDD approach: write tests first
