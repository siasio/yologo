# Position Recognizer: Implementation-Ready Summary

**Date**: November 16, 2025  
**Status**: Ready to Implement  
**Full Analysis**: See `phase4_architecture_analysis.md`

---

## Final Architecture Decision: GridNet

A hybrid FCN architecture that combines pretrained features with explicit grid sampling.

### Model Structure

```
Input: (B, 3, 224, 224) - Rectified board images
    ↓
MobileNetV3-Small Backbone (pretrained)
    ↓
Multi-scale feature extraction:
    - Stage 2: 56×56 (low-level: edges, stone boundaries)
    - Stage 3: 28×28 (mid-level: stone groups)
    - Stage 4: 14×14 (high-level: board patterns)
    ↓
Feature Pyramid Fusion → 38×38 feature map
    ↓
Grid Sampling at 361 known intersection locations
    ↓
Lightweight Conv Head (3×3 + 1×1)
    ↓
Output: (B, 4, 19, 19) - Class logits per intersection
```

### Why This Architecture?

1. **Leverages Known Structure**: Uses fixed 19×19 grid positions explicitly
2. **Multi-Scale Features**: Combines edge detection + semantic understanding
3. **Parameter Efficient**: ~2.5M parameters (fits mobile constraint)
4. **Pretrained Initialization**: Faster training, better generalization
5. **Interpretable**: Per-intersection predictions with spatial context

---

## Dataset Statistics (example_assets, 99 samples)

```
Total Intersections: 35,739
Class Distribution:
  - Empty:    70.76% (25,290 intersections)
  - Black:    14.65% (5,237 intersections)
  - White:    14.58% (5,212 intersections)
  - Occluded:  0.00% (0 intersections - NO DATA)

Stone Count Distribution:
  - Mean: 105.5 stones per position
  - Range: 6 to 145 stones
  - Std Dev: 32.7
```

**Critical Insight**: No occlusion data exists. Train 3-class model initially.

---

## Training Configuration

### Loss Function

**Weighted Cross-Entropy Loss**

Based on inverse frequency weighting:

```yaml
loss:
  type: weighted_cross_entropy
  class_weights: [1.0, 4.8, 4.9, 5.0]  # [empty, black, white, occluded]
  label_smoothing: 0.05  # Prevent overconfidence
```

Rationale:
- Empty intersections dominate (70%) → weight 1.0
- Stones are important but less common (~15% each) → weight ~5×
- Occluded has no data → high weight (5.0) for future use

### Hyperparameters

```yaml
training:
  epochs: 50
  batch_size: 32  # 64 if GPU memory allows
  
  optimizer:
    type: AdamW
    lr: 0.001  # Start higher, backbone is pretrained
    weight_decay: 0.0001
    
  scheduler:
    type: CosineAnnealingLR
    T_max: 50
    eta_min: 0.00001
    
  early_stopping:
    patience: 10
    min_delta: 0.001
```

### Data Pipeline

```python
# Already implemented and tested ✅
from data.position_recognition_dataset import PositionRecognitionDataset

dataset = PositionRecognitionDataset(
    manifest_paths=manifests,
    rectify=True,          # Use BoardRectifier
    output_size=224,       # Standard input size
    padding_ratio=0.02,    # 2% padding
    transform=train_transforms,  # From data/transforms.py
)

# Returns:
# - image: (3, 224, 224) tensor, rectified and normalized
# - position_grid: (4, 19, 19) tensor, one-hot encoded
# - num_stones: int
# - corner_mapping: dict (for debugging)
```

---

## Implementation Checklist

### Phase 4a: Core Implementation (Days 1-3)

- [ ] **Model Architecture** (`models/position_recognizer/gridnet.py`)
  - [ ] GridNet class implementation
  - [ ] Multi-scale feature extraction from MobileNetV3
  - [ ] Feature pyramid fusion module
  - [ ] Grid sampling layer
  - [ ] Classification head
  - [ ] Forward pass test
  
- [ ] **Tests** (`tests/test_gridnet.py`)
  - [ ] Model initialization
  - [ ] Forward pass shape: (B,3,224,224) → (B,4,19,19)
  - [ ] Gradient flow verification
  - [ ] Save/load checkpoint
  - [ ] CUDA compatibility
  
- [ ] **Loss Function** (`training/losses.py` - extend existing)
  - [ ] Weighted cross-entropy (already exists, verify)
  - [ ] Optional: Focal loss implementation
  - [ ] Optional: Auxiliary losses (stone count)
  
- [ ] **Trainer** (`training/position_recognizer_trainer.py`)
  - [ ] PositionRecognizerTrainer class
  - [ ] Training loop with metrics
  - [ ] Validation loop
  - [ ] TensorBoard logging:
    - Loss curves
    - Per-class accuracy
    - Confusion matrix
    - Example predictions
  - [ ] Checkpointing (best + latest)
  - [ ] Early stopping
  
- [ ] **Training Script** (`scripts/train_position_recognizer.py`)
  - [ ] CLI with argparse
  - [ ] Config loading (position_recognizer.yaml)
  - [ ] Debug mode (small dataset)
  - [ ] Train/val split (80/20)
  - [ ] Logging setup

### Phase 4b: Training & Validation (Days 4-5)

- [ ] **Debug Training**
  - [ ] Train on 20 samples, 5 epochs
  - [ ] Verify loss decreases
  - [ ] Check TensorBoard visualizations
  - [ ] Fix any bugs
  
- [ ] **Full Training**
  - [ ] Train on all data, 50 epochs
  - [ ] Monitor metrics closely
  - [ ] Save best checkpoints
  
- [ ] **Evaluation**
  - [ ] Overall accuracy >95%
  - [ ] Per-class accuracy >90% (for stones)
  - [ ] Confusion matrix analysis
  - [ ] Visualize failure cases

### Phase 4c: Refinement (Days 6-7)

- [ ] **Error Analysis**
  - [ ] Which positions are hardest?
  - [ ] Common failure patterns?
  - [ ] Edge/corner vs center accuracy
  
- [ ] **Hyperparameter Tuning** (if needed)
  - [ ] Learning rate adjustment
  - [ ] Loss weight tuning
  - [ ] Augmentation strength
  
- [ ] **Documentation**
  - [ ] Update phase4_completion.md
  - [ ] Update PROJECT_TODO.md
  - [ ] Update README.md

---

## Success Criteria

| Metric | Target | Critical |
|--------|--------|----------|
| Overall Accuracy | >95% | Yes |
| Black Stone Accuracy | >90% | Yes |
| White Stone Accuracy | >90% | Yes |
| Empty Accuracy | >98% | No (easy) |
| Stone Count MAE | <2.0 | Yes |
| Model Size | <5MB | Yes |
| Inference Time (CPU) | <50ms | No (optimize later) |
| Test Coverage | >85% | Yes |

---

## Key Implementation Details

### Grid Intersection Positions

```python
# Use existing utility
from utils.geometry import compute_grid_intersections

# Compute 361 intersection positions in 224×224 space
intersections = compute_grid_intersections(
    board_size=19,
    image_size=(224, 224),
    margin=0.05  # 5% margin (matches rectification padding)
)
# Shape: (19, 19, 2) with (x, y) coordinates
```

### Feature Sampling

```python
import torch.nn.functional as F

# Given feature map (B, C, H, W) and grid positions (19, 19, 2)
# Normalize grid to [-1, 1] for grid_sample
grid_normalized = 2.0 * (intersections / image_size) - 1.0

# Sample features at each intersection
sampled = F.grid_sample(
    features,
    grid_normalized,
    mode='bilinear',
    align_corners=True
)
# Output: (B, C, 19, 19)
```

### Multi-Scale Feature Fusion

```python
# Extract features at multiple scales from backbone
feat_56 = backbone.stage2(x)   # (B, 24, 56, 56)
feat_28 = backbone.stage3(...)  # (B, 40, 28, 28)
feat_14 = backbone.stage4(...)  # (B, 96, 14, 14)

# Upsample all to 38×38
feat_56_up = F.interpolate(feat_56, size=38, mode='bilinear')
feat_28_up = F.interpolate(feat_28, size=38, mode='bilinear')
feat_14_up = F.interpolate(feat_14, size=38, mode='bilinear')

# Concatenate or add
fused = torch.cat([feat_56_up, feat_28_up, feat_14_up], dim=1)

# Reduce channels
fused = self.fusion_conv(fused)  # (B, 128, 38, 38)
```

---

## Risk Mitigation Strategies

### If Accuracy is Too Low (<90% on stones)

1. Try focal loss instead of weighted CE
2. Increase model capacity (add more channels)
3. Use U-Net skip connections
4. Strengthen augmentation
5. Check data quality (are labels correct?)

### If Model is Too Large (>5MB)

1. Use MobileNetV3-Small (not Large)
2. Reduce decoder channels (128 → 64)
3. Apply structured pruning
4. Quantize to INT8 (2-4× reduction)

### If Training is Unstable

1. Reduce learning rate (1e-3 → 1e-4)
2. Add gradient clipping (max_norm=1.0)
3. Use smaller batch size
4. Add more weight decay
5. Try curriculum learning

---

## Post-Training Tasks

### Model Export

```bash
python scripts/export_model.py \
    --checkpoint outputs/.../best_model.pth \
    --format onnx \
    --quantize int8
```

### Benchmarking

```python
# Inference speed test
import time

model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 224, 224)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Measure
    start = time.time()
    for _ in range(100):
        _ = model(x)
    elapsed = time.time() - start
    
    print(f"Inference: {elapsed/100*1000:.2f} ms per image")
```

### Integration Test

```python
# End-to-end pipeline test
from data.board_rectifier import BoardRectifier
from models.position_recognizer import GridNet

# Load models
board_detector = load_model('board_detector.pth')
position_recognizer = load_model('position_recognizer.pth')
rectifier = BoardRectifier()

# Process image
image = load_image('test.jpg')
corners = board_detector.predict(image)
rectified, _, _ = rectifier.rectify(image, corners)
position_grid = position_recognizer.predict(rectified)

# Validate output
assert position_grid.shape == (4, 19, 19)
print(f"Detected {position_grid[1].sum():.0f} black stones")
print(f"Detected {position_grid[2].sum():.0f} white stones")
```

---

## Next Phase Preview: Phase 5

After Phase 4 completion, Phase 5 will integrate everything:

1. **End-to-End Pipeline**
   - Board detection → Rectification → Position recognition
   - Single inference script
   - Video processing support

2. **Model Optimization**
   - Quantization to INT8
   - ONNX export for deployment
   - Speed benchmarking

3. **SGF Output**
   - Convert position grid to SGF format
   - Game record generation

4. **Mobile Preparation**
   - TFLite conversion (for Android)
   - On-device testing

---

## Files to Create/Modify

### New Files
- `models/position_recognizer/gridnet.py` (main model)
- `models/position_recognizer/__init__.py` (exports)
- `training/position_recognizer_trainer.py` (trainer)
- `scripts/train_position_recognizer.py` (CLI)
- `tests/test_gridnet.py` (unit tests)
- `configs/position_recognizer.yaml` (config)
- `copilot_reports/phase4_completion.md` (docs)

### Modified Files
- `training/losses.py` (verify/extend)
- `training/metrics.py` (already has needed metrics)
- `copilot_reports/PROJECT_TODO.md` (update status)
- `README.md` (add Phase 4 info)

---

**Status**: All prerequisites complete. Ready to implement GridNet architecture.

**Estimated Time**: 5-7 days for full implementation, training, and validation.

**Confidence**: High. Clear architecture, tested data pipeline, comprehensive analysis complete.
