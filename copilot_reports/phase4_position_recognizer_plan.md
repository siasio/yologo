# Phase 4 Implementation Plan: Position Recognizer

**Status**: ⏳ Ready to Start  
**Priority**: High  
**Dependencies**: Phase 3 (Board Detector) Complete

---

## Overview

Phase 4 will implement the position recognition model that takes a rectified Go board image and outputs a 19×19 grid predicting the state of each intersection: Empty (0), Black stone (1), White stone (2), or Occluded (3).

## Architecture Design

### Model: MobileNetV3PositionRecognizer

**Backbone**: MobileNetV3-Small (pretrained on ImageNet)
- Input: 224×224 rectified board image
- Feature extraction: MobileNetV3-Small features

**Spatial Head**:
- Upsample features to 19×19 spatial resolution
- Conv layers to predict per-intersection classes
- Output: (B, 4, 19, 19) logits for 4 classes

**Alternative Approaches to Consider**:
1. **Direct MobileNet + Upsample** (simple, fast)
2. **U-Net style** with skip connections (better spatial resolution)
3. **DeepLabV3-style ASPP** (better context)

**Recommended**: Start with #1 (simplest), can upgrade if needed.

## Implementation Steps

### 1. Model Architecture (1-2 hours)

**File**: `models/position_recognizer/mobilenet_recognizer.py`

```python
class MobileNetPositionRecognizer(nn.Module):
    """
    Position recognizer using MobileNetV3-Small.
    
    Architecture:
    - MobileNetV3-Small backbone (pretrained)
    - Feature extraction at multiple scales
    - Spatial upsampling to 19×19
    - Per-intersection classification (4 classes)
    """
```

**Key Components**:
- Load pretrained MobileNetV3-Small
- Remove classifier head
- Add spatial decoder:
  - ConvTranspose2d or Upsample layers
  - 1×1 Conv for class prediction
  - Output shape: (B, 4, 19, 19)

**Design Considerations**:
- Input size: 224×224 (standard for MobileNetV3)
- Feature map size after backbone: ~7×7
- Need to upsample 7×7 → 19×19
- Use bilinear interpolation + Conv refinement

### 2. Training Infrastructure (2-3 hours)

**File**: `training/position_recognizer_trainer.py`

**Similar to BoardDetectorTrainer** but with:
- PositionRecognitionDataset (already implemented)
- PositionRecognitionLoss (already implemented)
- Position accuracy metrics (already implemented)
- Confusion matrix visualization
- Per-class accuracy logging
- Stone count comparison

**Special Considerations**:
- Class imbalance: Empty >> Black/White >> Occluded
- Use weighted CrossEntropyLoss
- Optional: Focal loss for hard examples
- Optional: Label smoothing

### 3. Training Script (1 hour)

**File**: `scripts/train_position_recognizer.py`

**Command-Line Interface**:
```bash
python scripts/train_position_recognizer.py \
    --config position_recognizer \
    --data-dir /path/to/data \
    --batch-size 64 \
    --epochs 50 \
    --debug
```

**Features** (same as board detector):
- Config file support
- CLI overrides
- Train/val split
- Logging and checkpointing
- Resume from checkpoint

### 4. Configuration (30 mins)

**Update**: `configs/position_recognizer.yaml`

```yaml
model:
  type: mobilenet  # or efficientnet, resnet
  size: small
  pretrained: true
  input_size: 224
  num_classes: 4
  
training:
  epochs: 50
  batch_size: 64
  
  optimizer:
    type: AdamW
    lr: 0.001
    weight_decay: 0.0001
  
  scheduler:
    type: CosineAnnealingLR
    T_max: 50
  
  loss:
    type: CrossEntropyLoss
    class_weights: [1.0, 1.0, 1.0, 2.0]  # Higher weight for occluded
    label_smoothing: 0.0
```

### 5. Testing (2-3 hours)

**File**: `tests/test_position_recognizer_model.py`

**Test Cases**:
- Model initialization
- Forward pass output shape
- Predict method
- Save/load
- CUDA compatibility

**File**: `tests/test_position_recognizer_trainer.py` (optional)

**Integration Tests**:
- Full training loop on small dataset
- Checkpoint saving/loading
- Metrics computation

### 6. Visualization Enhancements (1-2 hours)

**Update**: `evaluation/visualizer.py`

**New Functions**:
- `visualize_position_predictions()`: Overlay predicted stones on board
- `plot_confusion_matrix()`: Heatmap for confusion matrix
- `compare_stone_counts()`: Bar chart comparing pred vs target stone counts

**TensorBoard Logging**:
- Confusion matrix as image
- Per-class accuracy curves
- Example predictions (side-by-side: input, target, predicted)

### 7. Training and Evaluation (4-8 hours)

**Steps**:
1. Train on example_assets (debug mode, 5-10 epochs)
2. Verify TensorBoard logs and visualizations
3. Analyze confusion matrix and per-class accuracy
4. Train on full synthetic dataset (50-100 epochs)
5. Hyperparameter tuning if needed
6. Error analysis: which positions are difficult?

**Expected Results**:
- Overall accuracy: >95% (empty intersections dominate)
- Black stone accuracy: >90%
- White stone accuracy: >90%
- Occluded accuracy: >80% (harder, less data)

### 8. Optional Enhancements

**Curriculum Learning**:
- Stage 1: Empty boards + few stones (easy)
- Stage 2: Medium complexity (50-150 stones)
- Stage 3: Full complexity + occlusions

**Data Augmentation**:
- Already have transforms for rectified boards
- May need to tune augmentation strength

**Temporal Smoothing** (for video):
- Exponential moving average of predictions
- Hysteresis: require N consistent frames before change

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Model architecture | 1-2h | Critical |
| Trainer implementation | 2-3h | Critical |
| Training script | 1h | Critical |
| Configuration | 0.5h | Critical |
| Testing | 2-3h | High |
| Visualization | 1-2h | High |
| Training (debug) | 0.5h | Critical |
| Training (full) | 4-8h | High |
| Documentation | 1h | Medium |
| **Total** | **13-21h** | |

## Success Criteria

1. ✅ Model implemented and tested
2. ✅ Training script runs without errors
3. ✅ TensorBoard logging shows metrics and visualizations
4. ✅ Overall accuracy >95% on validation set
5. ✅ Per-class accuracy >85% for stones
6. ✅ Confusion matrix shows good separation
7. ✅ Tests pass (>90% coverage)
8. ✅ Documentation complete

## Potential Issues and Solutions

### Issue 1: Class Imbalance
**Problem**: Empty intersections ~70%, stones ~30%, occluded <5%  
**Solution**: Weighted loss, focal loss, or balanced sampling

### Issue 2: Small Stones at Low Resolution
**Problem**: 224×224 → 19×19 means ~12 pixels per intersection  
**Solution**: Use higher input resolution (448×448) or multi-scale features

### Issue 3: Occlusion Detection
**Problem**: Current synthetic data has NO occlusions (all zeros in channel 3)  
**Solution**: 
- Phase 4a: Train without occlusion (3 classes only)
- Phase 4b: Add synthetic occlusions or real data augmentation
- Future: Fine-tune on real data with occlusions

### Issue 4: Perspective Distortion in Rectification
**Problem**: Corner errors propagate to rectification  
**Solution**: Robust perspective transform, maybe add slight warping augmentation

## Testing Strategy

### Unit Tests
- Model forward pass
- Loss computation
- Metrics calculation
- Predict method

### Integration Tests
- End-to-end training loop (1-2 epochs)
- Checkpoint save/load
- TensorBoard logging

### Functional Tests
- Train on small subset (20 samples, 5 epochs)
- Verify metrics improve over epochs
- Check confusion matrix structure
- Validate predictions on known samples

## Next Phase Preview: Phase 5

After Phase 4, we'll have:
- Board detector (Phase 3) ✅
- Position recognizer (Phase 4) ⏳

Phase 5 will integrate both:
1. End-to-end pipeline script
2. Video processing support
3. Temporal smoothing
4. SGF output format
5. Model compression (quantization, pruning)
6. Benchmarking and optimization

---

**Ready to implement!** Start with model architecture, then training infrastructure, then tests.
