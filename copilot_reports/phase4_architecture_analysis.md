# Phase 4: Position Recognizer Architecture Analysis

**Date**: November 16, 2025  
**Status**: Pre-implementation Deep Analysis

---

## Problem Statement

Design a neural network that takes a **224×224 rectified Go board image** and outputs a **(4, 19, 19) tensor** representing one-hot encoded predictions for each of the 361 intersections.

**Classes**:
- 0: Empty intersection
- 1: Black stone
- 2: White stone  
- 3: Occluded (hand/body blocking view)

**Key Constraints**:
- Must be lightweight (<5MB, ideally <3MB) for mobile deployment
- Fast inference (<50ms on CPU)
- High accuracy (>95% overall, >90% per stone class)
- Must handle the synthetic dataset with NO occlusion data initially

---

## Critical Analysis: Why This is NOT a Standard Classification Problem

### The Spatial Structure Matters

This is fundamentally a **dense spatial prediction** problem with unique characteristics:

1. **Grid Structure is Known**: We're predicting on a fixed 19×19 grid, not arbitrary spatial locations
2. **Spatial Relationships Matter**: Stones rarely appear in isolation; patterns matter (groups, chains, territories)
3. **Resolution Mismatch**: 224×224 pixels → 19×19 predictions means ~12 pixels per intersection
4. **Perspective Already Corrected**: Rectification pipeline ensures square, aligned boards

### Key Insight: This is More Like Semantic Segmentation Than Classification

Standard image classification extracts global features and predicts a single label. We need **per-intersection predictions** with spatial structure preserved.

---

## Architecture Options: Deep Analysis

### Option 1: Fully Convolutional Network (FCN) - Direct Approach

**Architecture**:
```
Input: (B, 3, 224, 224)
↓
CNN Backbone → Feature maps (B, C, 28, 28)
↓
Spatial Decoder → (B, C', 19, 19)
↓
1×1 Conv → (B, 4, 19, 19)
```

**Pros**:
- Simple, direct approach
- Naturally preserves spatial structure
- Can use pretrained backbone (MobileNetV3, EfficientNet)
- Small model size possible

**Cons**:
- 224→19 is unusual (not power of 2 downsampling)
- May lose fine spatial detail in backbone
- Need careful upsampling strategy

**Best for**: Lightweight mobile deployment, simple baseline

---

### Option 2: U-Net Style with Skip Connections

**Architecture**:
```
Input: (B, 3, 224, 224)
↓
Encoder (MobileNetV3) with intermediate features
  - Level 1: 112×112
  - Level 2: 56×56
  - Level 3: 28×28
  - Level 4: 14×14
  - Level 5: 7×7
↓
Decoder with skip connections
  - Upsample + concatenate encoder features
  - Refine with conv layers
  - Final: 19×19×4
```

**Pros**:
- Skip connections preserve fine spatial details
- State-of-art for dense prediction tasks
- Better boundary detection (important for stone edges)
- Can recover from backbone downsampling

**Cons**:
- Larger model (more parameters in decoder)
- More complex implementation
- May be overkill for this problem

**Best for**: Maximum accuracy, when model size is less critical

---

### Option 3: Spatial Transformer Network (STN) + Grid-Aligned Prediction

**Key Insight**: We know EXACTLY where each intersection should be in the rectified image!

**Architecture**:
```
Input: (B, 3, 224, 224)
↓
CNN Feature Extractor → (B, C, H, W)
↓
For each of 361 intersections:
  - Sample features at known (x, y) location
  - Small MLP: features → 4-class logit
↓
Output: (B, 4, 19, 19)
```

**Implementation Detail**:
- Compute intersection positions: `compute_grid_intersections(19, 224)`
- Use grid_sample to extract features at each intersection
- Apply shared MLP to classify each intersection

**Pros**:
- **Explicitly uses known grid structure** 
- Can use higher resolution features (no need for 19×19 upsampling)
- More interpretable (per-intersection predictions)
- Can handle varying board sizes easily
- Parameter efficient (shared MLP across intersections)

**Cons**:
- Less common architecture (more implementation work)
- May not capture context as well as fully convolutional
- Requires careful feature sampling

**Best for**: This specific problem where grid structure is known and fixed

---

### Option 4: Attention-Based Grid Prediction

**Architecture**:
```
Input: (B, 3, 224, 224)
↓
CNN Backbone → Global features (B, C, H, W)
↓
Learnable Grid Queries: 19×19 = 361 queries
↓
Cross-Attention: Queries attend to image features
↓
Per-query MLP → (B, 361, 4)
↓
Reshape → (B, 4, 19, 19)
```

**Inspired by**: DETR, ViT detection approaches

**Pros**:
- Can capture long-range dependencies (important for Go strategy)
- Very flexible, can learn optimal feature aggregation
- State-of-art in many dense prediction tasks

**Cons**:
- Larger model (transformer attention is expensive)
- Needs more training data
- Slower inference
- Overkill for this problem

**Best for**: Research, when accuracy is paramount and model size doesn't matter

---

## Recommended Architecture: Hybrid FCN with Grid Alignment

After deep analysis, I propose a **hybrid approach** that combines the best aspects:

### Proposed Architecture: "GridNet"

```python
class GridNet(nn.Module):
    """
    Hybrid architecture optimized for Go board position recognition.
    
    Key ideas:
    1. Use pretrained MobileNetV3-Small for feature extraction
    2. Extract features at multiple scales
    3. Upsample to intermediate resolution (56×56 or 38×38)
    4. Sample features at known grid intersection locations
    5. Use lightweight head to predict per-intersection classes
    """
    
    def __init__(self):
        # Backbone: MobileNetV3-Small (pretrained)
        # Remove final pooling and classifier
        # Keep features at multiple scales
        
        # Feature fusion: combine multi-scale features
        # Target resolution: 38×38 (2× the grid size)
        
        # Grid sampler: sample at 361 known locations
        # Interpolate features using bilinear sampling
        
        # Classification head: lightweight MLP per intersection
        # Input: sampled features (C,)
        # Output: logits (4,)
```

### Why This Architecture?

1. **Leverages Known Structure**: Uses grid intersection positions explicitly
2. **Multi-Scale Features**: Combines low-level (edges, stones) and high-level (context) features
3. **Parameter Efficient**: Shared classification head, minimal decoder
4. **Maintains Spatial Context**: 38×38 intermediate resolution preserves neighborhood information
5. **Mobile-Friendly**: Based on MobileNetV3, ~2-3M parameters total

### Technical Details

**Backbone Output Scales** (MobileNetV3-Small):
- Input: 224×224
- Stage 1: 112×112 (early features, edges)
- Stage 2: 56×56 (mid-level features)
- Stage 3: 28×28 (high-level features)
- Stage 4: 14×14 (deep features)
- Stage 5: 7×7 (semantic features)

**Feature Fusion Strategy**:
- Upsample stages 3,4,5 to 38×38
- Concatenate or add features
- Apply 1×1 conv to reduce channels
- Result: (B, 128, 38, 38) feature map

**Grid Sampling**:
- Compute 361 intersection positions in 224×224 space
- Scale to 38×38 feature map space
- Use `F.grid_sample` to extract features at each point
- Result: (B, 128, 19, 19)

**Classification Head**:
- Apply conv layers: (B, 128, 19, 19) → (B, 4, 19, 19)
- Options:
  - Single 1×1 conv (simplest)
  - 3×3 conv + 1×1 conv (adds local context)
  - Small residual block + 1×1 conv (best accuracy)

---

## Loss Function Analysis

### Standard Cross-Entropy Loss

**Pros**:
- Simple, well-understood
- Works well with one-hot encoded targets
- Efficient to compute

**Cons**:
- Doesn't handle class imbalance well
- Treats all errors equally

### Weighted Cross-Entropy Loss

**Recommended for Initial Training**

```python
class_weights = torch.tensor([
    0.5,   # Empty (most common, lower weight)
    2.0,   # Black stone (important)
    2.0,   # White stone (important)
    5.0,   # Occluded (rare but critical)
])

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Rationale**:
- Empty intersections: ~70% of data (downweight)
- Stones: ~30% of data (normal weight)
- Occluded: <1% in synthetic data (high weight when it exists)

### Focal Loss

**Alternative for Hard Examples**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        # FL(p_t) = -α(1-p_t)^γ log(p_t)
        # Focuses on hard examples
```

**Pros**:
- Automatically focuses on hard intersections
- Reduces weight of easy empty intersections
- State-of-art for imbalanced classification

**Cons**:
- More hyperparameters to tune
- Slightly slower

**Recommendation**: Start with weighted CE, try focal loss if accuracy plateaus

### Auxiliary Losses (Optional)

**Stone Count Loss**:
```python
# Predict total stone counts as auxiliary task
predicted_stones = output.sum(dim=(2,3))  # (B, 4)
target_stones = target.sum(dim=(2,3))
aux_loss = F.mse_loss(predicted_stones, target_stones)

total_loss = main_loss + 0.1 * aux_loss
```

**Spatial Consistency Loss**:
```python
# Encourage smoothness (stones tend to cluster)
# Penalize isolated stone predictions
diff_h = output[:, :, :-1, :] - output[:, :, 1:, :]
diff_w = output[:, :, :, :-1] - output[:, :, :, 1:]
consistency_loss = (diff_h**2).mean() + (diff_w**2).mean()
```

**Recommendation**: Start without auxiliary losses, add if needed

---

## Data Considerations

### Class Distribution (Expected)

Based on typical Go positions:
- Empty: ~65-75% of intersections
- Black stones: ~12-18%
- White stones: ~12-18%
- Occluded: 0% in current synthetic data

### Critical Issue: No Occlusion Data

**Problem**: Synthetic dataset has NO occlusion (channel 3 is all zeros)

**Solutions**:

1. **Phase 4a: Train 3-class model initially**
   - Ignore occlusion class
   - Output: (B, 3, 19, 19)
   - Achieve high accuracy on stones

2. **Phase 4b: Add synthetic occlusions**
   - Generate random masks during training
   - Overlay on rectified boards
   - Label occluded regions as class 3
   - Fine-tune from 3-class model

3. **Phase 4c: Real data fine-tuning**
   - Collect real photos with occlusions
   - Fine-tune on real data
   - Transfer learning from synthetic

**Recommendation**: Implement 3-class model first, design architecture to easily extend to 4 classes

### Augmentation Strategy

Already have augmentation in `data/transforms.py`, but specific considerations:

**Keep**:
- Small rotations (±5°) - boards won't be perfectly aligned
- Color jitter - lighting variations
- Blur - out-of-focus stones

**Avoid**:
- Large perspective transforms (already rectified)
- Heavy rotations (rectification handles this)
- Cutout (don't want to simulate occlusion yet)

**Add** (optional):
- Random grid shift (±2 pixels) - account for rectification errors
- Stone size variation - simulate different camera distances
- Synthetic occlusion masks (Phase 4b)

---

## Training Strategy

### Curriculum Learning (Optional but Recommended)

**Stage 1: Empty boards + sparse positions** (0-50 stones)
- Easiest cases
- Learn stone appearance
- 10-20 epochs

**Stage 2: Medium complexity** (50-150 stones)
- More realistic positions
- Learn spatial patterns
- 20-30 epochs

**Stage 3: Full dataset** (all positions)
- All difficulties
- Fine-tune
- 20-30 epochs

**Benefits**:
- Faster convergence
- Better generalization
- Avoids local minima

**Drawback**: More complex training pipeline

**Recommendation**: Try without curriculum first, add if training is unstable

### Hyperparameters

**Learning Rate**:
- Initial: 1e-3 (with pretrained backbone)
- Or: 1e-4 for backbone, 1e-3 for head (differential learning rates)
- Scheduler: CosineAnnealingLR or ReduceLROnPlateau

**Batch Size**:
- 32-64 (depending on GPU memory)
- Larger batch → more stable gradients for dense prediction

**Optimizer**:
- AdamW (standard, works well with transformers and CNNs)
- SGD with momentum (alternative, sometimes better generalization)

**Regularization**:
- Weight decay: 1e-4
- Dropout in head: 0.1-0.2
- Label smoothing: 0.05-0.1 (helps with overconfident predictions)

---

## Evaluation Metrics

### Primary Metrics

1. **Overall Accuracy**: Percentage of correctly classified intersections
   - Target: >95%
   - Easy to achieve due to empty majority

2. **Per-Class Accuracy**:
   - Empty: >98%
   - Black: >92%
   - White: >92%
   - Occluded: >85% (when data available)

3. **Mean Intersection-wise F1**: Better than accuracy for imbalanced data

4. **Stone Count MAE**: Mean absolute error in total stone count
   - Should be <1 stone per position

### Secondary Metrics

5. **Confusion Matrix**: Visualize misclassification patterns

6. **Per-Intersection Difficulty**: Which intersections are hardest?
   - Corners vs edges vs center
   - Near other stones vs isolated

7. **Edge Accuracy**: How well are stone boundaries detected?

8. **Spatial Error Analysis**: Do errors cluster or scatter?

---

## Implementation Phases

### Phase 4a: Core Implementation (3-5 days)

1. **Day 1: Architecture**
   - Implement GridNet model
   - Test forward pass and output shape
   - Verify gradient flow

2. **Day 2: Training Infrastructure**
   - PositionRecognizerTrainer class
   - Loss computation
   - Metrics tracking

3. **Day 3: Training Script & Config**
   - CLI script
   - Configuration files
   - Debug mode for quick testing

4. **Day 4: Initial Training**
   - Train on example_assets (debug)
   - Fix bugs and issues
   - Verify TensorBoard logging

5. **Day 5: Full Training**
   - Train on full synthetic dataset
   - Monitor metrics
   - Basic error analysis

### Phase 4b: Refinement (2-3 days)

6. **Hyperparameter Tuning**
   - Learning rate search
   - Augmentation strength
   - Loss weights

7. **Error Analysis**
   - Which positions fail?
   - Failure mode categorization
   - Targeted improvements

8. **Optimization**
   - Model pruning/quantization exploration
   - Inference speed testing

### Phase 4c: Occlusion Support (1-2 days)

9. **Synthetic Occlusion Generation**
   - Add occlusion augmentation
   - Retrain/fine-tune

10. **Validation**
    - Test on synthetic occlusions
    - Prepare for real data

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | GridNet (FCN + Grid Sampling) | Leverages known structure, parameter efficient |
| **Backbone** | MobileNetV3-Small (pretrained) | Lightweight, proven, mobile-friendly |
| **Loss** | Weighted CrossEntropyLoss | Simple, handles imbalance, easy to tune |
| **Output** | (B, 4, 19, 19) logits | Direct one-hot format, no post-processing |
| **Initial Training** | 3-class (no occlusion) | Match available data, extend later |
| **Augmentation** | Moderate (from transforms.py) | Rectification already normalizes |
| **Curriculum** | Optional (try without first) | May help but adds complexity |
| **Metrics** | Multi-faceted (accuracy, F1, count) | Comprehensive evaluation |

---

## Risk Mitigation

### Risk 1: Poor Accuracy on Stones

**Mitigation**:
- Use multi-scale features (combine low/high level)
- Try focal loss to focus on hard examples
- Increase model capacity if needed
- Add more augmentation

### Risk 2: Model Too Large

**Mitigation**:
- Start with MobileNetV3-Small (smallest)
- Use depthwise separable convs in decoder
- Apply pruning post-training
- Quantization (INT8) for deployment

### Risk 3: Slow Inference

**Mitigation**:
- Profile model (identify bottlenecks)
- Use ONNX Runtime (faster than PyTorch)
- Optimize image preprocessing
- Consider smaller input size (192×192)

### Risk 4: Occlusion Class Never Learns

**Mitigation**:
- Start with 3-class model (ignore occlusion)
- Add synthetic occlusions gradually
- Use very high class weight (10×)
- Fine-tune on real data with occlusions

---

## Next Steps

1. **Save dataset statistics**: Compute class distributions on full dataset
2. **Implement GridNet architecture**: Start with clean, modular code
3. **Write comprehensive tests**: TDD approach, test each component
4. **Set up training pipeline**: Trainer + script + config
5. **Debug mode validation**: Verify on 20 samples before full training
6. **Full training run**: Monitor carefully, be ready to adjust
7. **Error analysis**: Understand failure modes
8. **Iterate**: Improve based on results

---

**Confidence Level**: High. This architecture is well-suited for the problem, balances accuracy and efficiency, and has clear path to implementation and optimization.
