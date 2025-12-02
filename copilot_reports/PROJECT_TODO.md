# YoloGo Project - TODO List

This document tracks all implementation phases and tasks for the YoloGo real-time Go board recognition project.

## Legend
- ✅ Completed
- 🔄 In Progress
- ⏳ Pending
- ⏸️ Blocked/Deferred

---

## Phase 1: Project Setup ✅

- ✅ Create virtual environment
- ✅ Install dependencies (PyTorch, Ultralytics, Albumentations, etc.)
- ✅ Set up project structure (directories: data/, models/, training/, evaluation/, utils/, tests/, configs/, scripts/)
- ✅ Initialize git, .gitignore
- ✅ Create base configuration files (base_config.yaml, data.yaml, board_detector.yaml, position_recognizer.yaml)
- ✅ Implement configuration loader (utils/config_loader.py)
- ✅ Implement logger utilities (utils/logger.py)
- ✅ Implement seed utilities (utils/seed.py)
- ✅ Implement geometry utilities (utils/geometry.py)
- ✅ Implement checkpoint manager (utils/checkpoint.py)
- ✅ Write utility tests (26 tests)
- ✅ Create Makefile with common commands
- ✅ Create setup.py for package installation
- ✅ Document Phase 1 completion

**Status**: Complete (74% test coverage)

---

## Phase 2: Data Pipeline ✅

### Core Data Components
- ✅ Implement manifest parser (data/manifest_parser.py)
  - ✅ Parse JSON manifests
  - ✅ Find manifest files with glob patterns
  - ✅ Load ground truth arrays (.npy files)
  - ✅ Extract image paths, grid corners, board corners
  - ✅ Validate manifest structure
- ✅ Implement preprocessor (data/preprocessor.py)
  - ✅ Load and normalize images (ImageNet normalization)
  - ✅ Convert position arrays: (2,19,19) → (4,19,19) with occluded channel as zeros
  - ✅ Grid/class index conversions
  - ✅ Stone extraction utilities
- ✅ Implement BoardDetectionDataset (data/board_detection_dataset.py)
  - ✅ PyTorch Dataset class
  - ✅ Load images + corner annotations
  - ✅ Support Albumentations transforms with keypoints
- ✅ Implement PositionRecognitionDataset (data/position_recognition_dataset.py)
  - ✅ PyTorch Dataset class
  - ✅ Load rectified boards + position grids
  - ✅ Automatic board rectification using perspective transform
  - ✅ Configurable output size (default 224×224)

### Augmentation & Visualization
- ✅ Add augmentation pipeline (data/transforms.py)
  - ✅ Board detection transforms (rotation ±15°, perspective, color jitter, blur)
  - ✅ Position recognition transforms (rotation ±5°, lighter augmentation)
  - ✅ Config-driven parameters
- ✅ Create visualization tools (evaluation/visualizer.py)
  - ✅ Draw corners and grid overlays
  - ✅ Visualize position grids with stones
  - ✅ Prediction comparison views
  - ✅ TensorBoard logging integration
  - ✅ Per-class accuracy computation
- ✅ Create CLI visualization script (scripts/visualize_dataset.py)

### Testing & Documentation
- ✅ Write dataset tests (18 new tests: manifest parser, preprocessor)
- ✅ Test geometry transformations
- ✅ Document Phase 2 completion

**Status**: Complete (44 tests total: 37 passing, 7 skipped appropriately)

---

## Phase 3: Board Detector ✅

### Model Architecture
- ✅ Create board detector module structure (models/board_detector/)
- ✅ Implement MobileNetV3-Small wrapper (models/board_detector/yolo_detector.py)
  - ✅ Load pretrained MobileNetV3-Small backbone (ImageNet weights)
  - ✅ Add custom regression head for 4 corners (8 coordinates)
  - ✅ Configure input size (640×640)
- ✅ Implement corner loss function (training/losses.py)
  - ✅ MSE loss for corner coordinates
  - ✅ Smooth L1 loss variant
  - ✅ Wing loss for robustness
  - ✅ Normalized coordinate handling [0,1]
  - ✅ Optional shape regularization

### Training Infrastructure
- ✅ Create training loop (training/board_detector_trainer.py)
  - ✅ Load BoardDetectionDataset with augmentations
  - ✅ Training/validation split (80/20)
  - ✅ AdamW optimizer + CosineAnnealingLR scheduler
  - ✅ Checkpoint saving (best + latest)
  - ✅ TensorBoard logging (loss, metrics, visualizations)
  - ✅ Early stopping logic
  - ✅ Gradient accumulation and clipping
- ✅ Implement evaluation metrics (training/metrics.py)
  - ✅ IoU for board bounding box
  - ✅ Corner localization error (pixel distance)
  - ✅ Mean corner error per image
  - ✅ Per-corner error breakdown
  - ✅ MetricsTracker for aggregation
- ✅ Create training script (scripts/train_board_detector.py)
  - ✅ CLI with argparse
  - ✅ Config file support
  - ✅ Debug mode for testing
  - ✅ Resume from checkpoint

### Testing & Validation
- ✅ Write model architecture tests (9 tests)
- ✅ Write loss function tests (15 tests)
- ✅ Write metrics tests (18 tests)
- ✅ Test training loop components
- ✅ Validate on example data (20 samples, debug mode)

### Training & Optimization
- ⏳ Train on full synthetic dataset (not just example_assets)
- ⏳ Hyperparameter tuning (learning rate, batch size, augmentation strength)
- ⏳ Evaluate performance on validation set (full dataset)
- ⏳ Error analysis (visualize failures)
- ⏳ Optimize for speed/accuracy tradeoff

### Export
- ⏳ Export trained model to ONNX format
- ⏳ Verify ONNX model outputs match PyTorch
- ⏳ Benchmark ONNX inference speed

**Status**: Complete (infrastructure ready, needs training on full dataset)  
**Test Coverage**: 42 tests (all passing)  
**Components**: Model (~1.1M params), Trainer, Losses, Metrics, CLI Script

---

## Phase 3.5: Board Rectification Pipeline ✅

### Core Rectification Module
- ✅ Implement BoardRectifier class (data/board_rectifier.py)
  - ✅ Corner validation (shape, size, finite values, proper quadrilateral)
  - ✅ Padding addition (configurable ratio, default 2%)
  - ✅ Perspective transformation to square output
  - ✅ Corner mapping tracking for ground truth rotation
  - ✅ Batch processing support
- ✅ Implement position grid rotation (data/board_rectifier.py)
  - ✅ Automatic rotation based on corner mapping
  - ✅ Handles 0°, 90°, 180°, 270° rotations
  - ✅ PyTorch tensor compatibility (contiguous arrays)

### Integration
- ✅ Update PositionRecognitionDataset (data/position_recognition_dataset.py)
  - ✅ Initialize BoardRectifier with configurable parameters
  - ✅ Track original corners from manifest
  - ✅ Automatic ground truth rotation
  - ✅ Return corner mapping in sample dict

### Visualization & Testing
- ✅ Create visualization script (scripts/visualize_rectification.py)
  - ✅ 4-panel visualization (original, padded, rectified, with grid)
  - ✅ Batch processing support
  - ✅ Save to disk or interactive display
  - ✅ Show rotation information and stone count
- ✅ Write comprehensive tests (tests/test_board_rectifier.py)
  - ✅ 22 tests covering all functionality
  - ✅ Corner validation tests
  - ✅ Padding tests (standard, edge cases)
  - ✅ Corner mapping tests (identity, rotations)
  - ✅ Rectification tests (square, perspective, batch)
  - ✅ Position grid rotation tests
- ✅ Document Phase 3.5 completion

**Status**: Complete (22 tests, all passing)  
**Components**: BoardRectifier class, rotation utilities, visualization tools  
**Integration**: Seamlessly integrated with PositionRecognitionDataset

---

## Phase 4: Position Recognizer ⏳

### Model Architecture
- ⏳ Create position recognizer module structure (models/position_recognizer/)
- ⏳ Implement MobileNetV3-small backbone (models/position_recognizer/mobilenet_recognizer.py)
  - ⏳ Load pretrained MobileNetV3-small
  - ⏳ Create custom spatial output head
  - ⏳ Output shape: (batch, 4, 19, 19) for 4 classes
  - ⏳ Input size: 224×224
- ⏳ Implement position recognition loss (training/losses.py)
  - ⏳ Weighted CrossEntropyLoss (occluded class weight = 2.0)
  - ⏳ Optional: Focal Loss for hard examples
  - ⏳ Optional: Label smoothing

### Training Infrastructure
- ⏳ Create training loop (training/position_recognizer_trainer.py)
  - ⏳ Load PositionRecognitionDataset with augmentations
  - ⏳ Training/validation split (80/10/10)
  - ⏳ AdamW optimizer + StepLR or CosineAnnealing
  - ⏳ Checkpoint saving (best + latest)
  - ⏳ TensorBoard logging (loss, accuracy, confusion matrix)
  - ⏳ Early stopping logic
  - ⏳ Curriculum learning (optional: start with clear positions, add occlusion)
- ⏳ Implement evaluation metrics (training/metrics.py)
  - ⏳ Per-class accuracy (Empty, Black, White, Occluded)
  - ⏳ Per-intersection accuracy
  - ⏳ Confusion matrix
  - ⏳ Stone count accuracy
- ⏳ Create training script (scripts/train_position_recognizer.py)

### Testing & Validation
- ⏳ Write model architecture tests
- ⏳ Write loss function tests
- ⏳ Test training loop components
- ⏳ Test metric calculations
- ⏳ Validate on example data

### Training & Optimization
- ⏳ Train MobileNetV3-small on full dataset
- ⏳ Implement curriculum learning stages (if beneficial)
  - ⏳ Stage 1: 0-50 stones (clear positions)
  - ⏳ Stage 2: 50-150 stones (medium complexity)
  - ⏳ Stage 3: All positions (including edge cases)
- ⏳ Hyperparameter tuning
- ⏳ Evaluate performance on validation set
- ⏳ Error analysis (confusion matrix, failure cases)
- ⏳ Add temporal smoothing logic (optional, for video sequences)

### Export
- ⏳ Export trained model to ONNX format
- ⏳ Verify ONNX model outputs match PyTorch
- ⏳ Benchmark ONNX inference speed

**Status**: Not started

---

## Phase 5: Integration & Optimization ⏳

### End-to-End Pipeline
- ⏳ Create inference pipeline (scripts/inference.py)
  - ⏳ Load both models (board detector + position recognizer)
  - ⏳ Full pipeline: image → board detection → position recognition → output
  - ⏳ Handle multiple boards in single image (optional)
  - ⏳ Output in SGF format or JSON
- ⏳ Add video processing support (scripts/process_video.py)
  - ⏳ Frame-by-frame processing
  - ⏳ Temporal smoothing across frames
  - ⏳ Change detection (only update when stones placed/removed)

### Model Compression
- ⏳ Implement post-training quantization (PTQ)
  - ⏳ Quantize board detector to INT8
  - ⏳ Quantize position recognizer to INT8
  - ⏳ Calibrate on validation data
  - ⏳ Verify accuracy degradation < 2%
- ⏳ Model pruning (if needed to reach <10MB per model)
  - ⏳ Structured pruning
  - ⏳ Fine-tune pruned models
- ⏳ Knowledge distillation (optional, if quantization hurts accuracy)
  - ⏳ Train smaller student models
  - ⏳ Distill from full-precision teachers

### Benchmarking
- ⏳ Benchmark on CPU (simulate mobile environment)
  - ⏳ Measure inference time per model
  - ⏳ Measure memory usage
  - ⏳ Measure model file sizes
  - ⏳ Target: <15MB total, <200ms per frame
- ⏳ Profile bottlenecks
- ⏳ Optimize slow operations

### Error Analysis
- ⏳ Systematic error analysis on test set
  - ⏳ Board detection failures (corners mislocalized)
  - ⏳ Position recognition errors (misclassified stones)
  - ⏳ Edge case analysis (occlusion, blur, unusual angles)
- ⏳ Create error visualization reports
- ⏳ Iterate on problem areas (data augmentation, model architecture)

### Demo & Validation
- ⏳ Create demo notebook (notebooks/demo.ipynb)
- ⏳ Create demo script with webcam support
- ⏳ Test on real photos/videos (not just synthetic data)

**Status**: Not started

---

## Phase 6: Documentation & Deployment Prep ⏳

### Documentation
- ⏳ Write comprehensive README.md
  - ⏳ Project overview
  - ⏳ Installation instructions
  - ⏳ Quick start guide
  - ⏳ Usage examples
  - ⏳ Model performance metrics
- ⏳ Create API documentation
  - ⏳ Document all public interfaces
  - ⏳ Add usage examples for each module
  - ⏳ Create architecture diagrams
- ⏳ Write Android integration guide
  - ⏳ Model loading in Android
  - ⏳ Preprocessing pipeline for mobile
  - ⏳ Inference optimization tips
  - ⏳ Example Android code snippets
- ⏳ Create model card
  - ⏳ Training data description
  - ⏳ Performance metrics
  - ⏳ Limitations and biases
  - ⏳ Intended use cases

### Model Versioning
- ⏳ Create model versioning system
  - ⏳ Version numbering scheme
  - ⏳ Model registry structure
  - ⏳ Metadata tracking (training date, metrics, config)
- ⏳ Package models for release
  - ⏳ ONNX models + metadata
  - ⏳ Example inference code
  - ⏳ Preprocessing parameters

### Testing & Quality Assurance
- ⏳ Achieve >80% test coverage
- ⏳ Integration tests for full pipeline
- ⏳ Performance regression tests
- ⏳ Cross-platform validation (Linux, macOS, Windows)

### Release Preparation
- ⏳ Create release checklist
- ⏳ Tag stable version in git
- ⏳ Prepare demo materials (videos, screenshots)
- ⏳ Write blog post or technical report (optional)

**Status**: Not started

---

## Future Enhancements (Post-v1.0)

- ⏸️ Real-time video optimization (frame skipping, motion detection)
- ⏸️ Multi-board detection in single frame
- ⏸️ Support for different board sizes (9×9, 13×13)
- ⏸️ Adversarial training for robustness
- ⏸️ Active learning pipeline for continuous improvement
- ⏸️ Cloud-based model updates
- ⏸️ Game state validation (detect illegal moves)
- ⏸️ Automatic game transcription with move sequence

---

## Notes

### Key Technical Decisions
- **Dataset Format**: (2,19,19) → (4,19,19) with occluded channel as zeros (no occlusion in current synthetic data)
- **Architecture**: YOLOv8-nano (board detector) + MobileNetV3-small (position recognizer)
- **Target Size**: <15MB total model size for mobile deployment
- **Input Sizes**: 640×640 (detector), 224×224 (recognizer)
- **Deployment**: PyTorch → ONNX → TFLite (or ONNX Runtime on Android)
- **TDD Approach**: Write tests before/alongside implementation
- **Code Quality**: Type hints, docstrings, black formatting, modular structure

### Current Status
- **Phase 1**: ✅ Complete (74% coverage, 26 tests)
- **Phase 2**: ✅ Complete (44 tests total, ~1,034 lines of code)
- **Phase 3**: ✅ Complete (42 new tests, board detector training infrastructure)
- **Phase 3.5**: ✅ Complete (22 new tests, board rectification pipeline)
- **Phase 4**: ⏳ Next priority - Position recognizer model and training
- **Overall Progress**: ~58% complete (3.5/6 phases)

### Priority Order for Phase 3-4
1. Position recognizer model (simpler, fewer dependencies)
2. Position recognizer training loop
3. Board detector model (YOLO integration more complex)
4. Board detector training loop
5. Evaluation and optimization

### Phase 3 Notes

**Architecture Decision**: Used MobileNetV3-Small with direct coordinate regression instead of YOLO-pose:
- Simpler and more suitable for single-board, 4-corner detection
- Smaller model size (~1.1M params vs ~3-5M for YOLO)
- Direct regression more stable than keypoint detection format
- Better for mobile deployment

**Model Output**: 8 values (4 corners × 2 coords) normalized to [0,1]

**Loss Function**: Smooth L1 for corners + optional shape regularization

**Training Infrastructure**: Complete with TensorBoard logging, checkpointing, early stopping, and visual predictions

**Tests**: 42 new tests covering model, losses, and metrics (100% passing)

### Phase 3.5 Notes

**Board Rectification Pipeline**: Complete infrastructure for converting detected corners to rectified square images:
- Validates detected corners for proper quadrilateral shape and size
- Adds configurable padding (default 2%) to ensure full board visibility
- Performs perspective transformation to square format (default 224×224)
- Tracks corner mapping for automatic ground truth rotation
- Handles arbitrary board orientations (0°, 90°, 180°, 270°)

**Key Components**:
- `BoardRectifier` class with validation, padding, and rectification
- `rotate_position_grid()` function for ground truth alignment
- Integration with `PositionRecognitionDataset`
- Visualization script for pipeline debugging

**Benefits for Position Recognizer**:
- Consistent 224×224 square inputs (no perspective distortion)
- Properly oriented ground truth (automatic rotation handling)
- Full board visibility (padding prevents edge cutoff)
- Works with any camera angle/perspective

**Tests**: 22 new tests covering all rectification functionality (100% passing)

---

Last Updated: November 16, 2025
