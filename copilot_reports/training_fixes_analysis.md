# Training Setup Analysis and Fixes

## Issues Identified

### 1. **Shape Regularization Was Interfering with Learning**

**Problem**: The shape loss was falling nicely but corner localization wasn't improving because:
- Shape regularization (0.1 weight) was too strong
- It's easy to minimize shape loss by predicting similar side lengths
- But this doesn't help the model learn actual corner positions
- The gradient signal was dominated by shape loss, not corner localization

**Solution**: 
- **Disabled shape regularization by default** (set to `false` in config)
- Reduced weight from 0.1 → 0.01 if enabled
- Shape regularization is NOT standard for corner detection tasks

### 2. **Standard Loss Functions for Corner Detection**

For keypoint/corner localization, the standard approach is:
- **Primary loss**: Smooth L1 or L1 loss on normalized coordinates
- **Why Smooth L1**: More robust to outliers than MSE, smooth gradients near zero
- **No auxiliary losses**: Direct localization loss is most effective

**What NOT to use**:
- Shape regularization (not standard, adds complexity)
- High loss weights on auxiliary terms
- Complex multi-term losses that dilute the main signal

### 3. **Insufficient Visualizations**

**Problem**: Only 4 images logged per epoch made it hard to see patterns

**Solution**: Increased to 12 images with dynamic grid layout
- Better coverage of validation set
- Easier to spot systematic errors
- Grid adjusts automatically to number of samples

### 4. **Cluttered TensorBoard Logs**

**Problem**: Too many redundant metrics made it hard to see what matters

**Solution**: Cleaned up logging to focus on key metrics:

**Per-step logs** (every 10 steps):
- `train_step/total_loss`
- `train_step/corner_loss`
- `train_step/mean_corner_error_px`

**Per-epoch logs**:
- `epoch/train_loss`, `epoch/val_loss`
- `epoch/train_corner_loss`, `epoch/val_corner_loss`
- `epoch/train_corner_error_px`, `epoch/val_corner_error_px`
- `epoch/val_iou`
- `epoch/learning_rate`

**What was removed**:
- Duplicate loss logging (was logging both "loss" and "total_loss")
- Per-corner error logs (cluttered, mean is sufficient)
- Max/median errors (mean is most important)

## Updated Configuration

```yaml
loss:
  corner_loss_type: smooth_l1  # Standard for corner detection
  shape_regularization: false  # Disabled - interferes with learning
  shape_weight: 0.01  # Only if enabled (was 0.1)
```

## What to Look For in TensorBoard

### Key Metrics (should all improve):

1. **`epoch/val_loss` and `epoch/val_corner_loss`**
   - Should decrease steadily
   - These are now the SAME (since shape regularization is disabled)
   - This is the main metric to monitor

2. **`epoch/val_corner_error_px`**
   - Mean error in pixels
   - Should decrease from ~200px toward <20px
   - This tells you actual localization accuracy

3. **`epoch/val_iou`**
   - Board bounding box overlap
   - Should increase from ~0.01 toward >0.8
   - Low IoU indicates poor corner localization

4. **`val/corner_predictions` images**
   - Now shows 12 samples instead of 4
   - Green circles = ground truth
   - Red X = predictions
   - **Look for**: predictions moving toward green circles over epochs

### Signs of Good Training:

✅ Corner loss decreasing smoothly  
✅ Corner error (px) decreasing  
✅ IoU increasing  
✅ Visual predictions getting closer to green circles  
✅ Learning rate decreasing with cosine schedule  

### Signs of Problems:

❌ Corner loss not decreasing after 10+ epochs  
❌ Corner error stuck around 200px  
❌ IoU not increasing above 0.1  
❌ Predictions far from targets in visualizations  

## Expected Training Behavior

### Early Training (Epochs 1-10):
- Loss drops rapidly from ~0.05 to ~0.01
- Corner error drops from ~200px to ~50-100px
- IoU increases from ~0 to ~0.3-0.5
- Visualizations show predictions moving toward targets

### Mid Training (Epochs 10-50):
- Loss continues decreasing slower
- Corner error reaches ~20-50px
- IoU increases to ~0.6-0.8
- Predictions mostly accurate but some outliers

### Late Training (Epochs 50-100):
- Loss plateaus around ~0.001-0.005
- Corner error reaches ~5-20px (acceptable)
- IoU reaches ~0.8-0.9 (good overlap)
- Most predictions very close to ground truth

## Debugging Tips

### If corner error isn't decreasing:

1. **Check learning rate**: Might be too high or too low
   - Default: 0.001 with cosine annealing
   - Try: 0.0005 or 0.002

2. **Check data augmentation**: Too aggressive augmentation can hurt
   - Rotation: ±15° (reasonable)
   - Perspective: 0.05-0.1 (reasonable)
   - Consider reducing if corners are too distorted

3. **Check batch size**: Small batches can be noisy
   - Minimum: 16
   - Recommended: 32-64
   - Use gradient accumulation if GPU memory limited

4. **Check input normalization**: Images should be in [0,1] range
   - Already handled in dataset

### If loss is NaN or exploding:

1. Lower learning rate
2. Check for invalid data (NaN in images or labels)
3. Increase gradient clipping (currently 1.0)

## Next Steps

1. **Run training with new config**:
```bash
python scripts/train_board_detector.py \
    --data-dir example_assets \
    --epochs 20 \
    --batch-size 4 \
    --output-dir outputs/fixed_training
```

2. **Monitor TensorBoard**:
```bash
tensorboard --logdir outputs/fixed_training/tensorboard --port 6006
```

3. **Watch for**:
   - Corner loss decreasing (most important)
   - Corner error dropping below 100px by epoch 10
   - IoU increasing above 0.3 by epoch 10
   - Predictions visibly moving toward targets

4. **If training looks good**:
   - Train on full dataset (not just example_assets)
   - Use batch size 32, epochs 100
   - Expect corner error <20px, IoU >0.8

## Summary of Changes

| Component | Before | After | Reason |
|-----------|--------|-------|---------|
| Shape regularization | Enabled (0.1 weight) | Disabled | Interferes with learning |
| Loss focus | Mixed signals | Pure corner localization | Standard practice |
| Visualizations | 4 images | 12 images | Better coverage |
| TensorBoard logs | ~15 metrics | 8 key metrics | Reduce clutter |
| Logged metrics | Duplicates | Clean, focused | Easier to interpret |

The training is now properly configured for corner detection with standard losses and clear monitoring.
