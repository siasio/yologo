# YoloGo: Real-time Go Board Recognition

Deep learning system for recognizing Go board positions from video feeds, designed to run on mobile devices.

## Overview

YoloGo consists of two lightweight neural networks with a preprocessing pipeline:
1. **Board Detector**: Locates the Go board in an image by detecting 4 corners (~1.1M params, MobileNetV3-Small)
2. **Board Rectification Pipeline**: Transforms detected board regions to square images with proper orientation
3. **Position Recognizer**: Classifies the state of each of the 19×19 intersections (Empty, Black, White, Occluded)

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd yologo

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
make test

# Train board detector
python scripts/train_board_detector.py --config board_detector --epochs 100 --batch-size 32

# Visualize board rectification pipeline
python scripts/visualize_rectification.py --num-samples 5

# Monitor training with TensorBoard
tensorboard --logdir outputs/board_detector_*/tensorboard --port 6006

# Train position recognizer (Phase 4 - coming soon)
# python scripts/train_position_recognizer.py --config position_recognizer
```

## Project Structure

```
yologo/
├── configs/          # YAML configuration files
├── data/             # Dataset loading and augmentation
├── models/           # Neural network architectures
├── training/         # Training loops and optimization
├── evaluation/       # Model evaluation and visualization
├── export/           # ONNX/TFLite export utilities
├── utils/            # Helper functions
├── scripts/          # Executable training/inference scripts
├── tests/            # Unit and integration tests
└── outputs/          # Trained models and predictions
```

## Dataset Format

The training data consists of synthetic Go board images generated with Blender. Each datapoint includes:
- Rendered image
- Ground truth position array (2, 19, 19) with black/white stone locations
- Grid corner coordinates for perspective transformation
- Board corner coordinates

See `example_assets/` for a sample manifest.

## Training

### Board Detector (Phase 3 - ✅ Complete)

The board detector uses MobileNetV3-Small to predict 4 corner coordinates:

```bash
# Basic training
python scripts/train_board_detector.py

# With custom config
python scripts/train_board_detector.py \
    --config board_detector \
    --data-dir /path/to/synthetic/data \
    --epochs 100 \
    --batch-size 32 \
    --device cuda

# Debug mode (small dataset)
python scripts/train_board_detector.py --debug --epochs 5
```

**Model Architecture**: MobileNetV3-Small backbone + MLP regression head  
**Output**: 8 values (4 corners × 2 coordinates) normalized to [0,1]  
**Loss**: Smooth L1 + optional shape regularization  
**Training Time**: ~4 hours (100 epochs, batch size 32, on GPU)

### Board Rectification Pipeline (Phase 3.5 - ✅ Complete)

The rectification pipeline transforms detected boards to square images ready for position recognition:

```bash
# Visualize rectification on sample data
python scripts/visualize_rectification.py \
    --data-dir example_assets \
    --num-samples 5 \
    --output-dir outputs/rectification_vis
```

**Key Features**:
- Validates detected corners for proper quadrilateral shape
- Adds configurable padding (default 2%) to ensure full board visibility
- Performs perspective transformation to square format (default 224×224)
- Tracks corner mapping for automatic ground truth rotation
- Handles arbitrary board orientations (0°, 90°, 180°, 270°)

**Integration**: Seamlessly integrated with `PositionRecognitionDataset` for automatic rectification during training.

### Position Recognizer (Phase 4 - Coming Soon)

Models are configured via YAML files in `configs/`. Key parameters:
- Model architecture (MobileNetV3)
- Training hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Loss functions and class weights

## Export

Export trained models for mobile deployment:

```bash
python scripts/export_models.py --format onnx --quantize int8
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

See `requirements.txt` for complete dependencies.

## License

[License information]
