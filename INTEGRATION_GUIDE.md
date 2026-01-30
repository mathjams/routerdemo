# Integration Guide - Using Your Existing Models

This guide explains how to integrate your trained router and branch models with the demo application.

## Your Model Architecture

Based on your existing code, your models use:

**Router Model** ([router_models.py](router_models.py)):
- Input: 32×32 RGB images (normalized with CIFAR-100 stats)
- Output: Logits over N branches
- Architecture: 3-layer CNN with ~224 channels

**Branch Models** ([router_models.py](router_models.py)):
- Input: 32×32 RGB images (normalized with CIFAR-100 stats)
- Output: Logits over 100 classes (CIFAR-100)
- Architecture: SimpleCNN with configurable channels and layers

## How the Demo Works with Your Models

### 1. Model Export Process

The [export_models_for_demo.py](export_models_for_demo.py) script:

1. Loads your checkpoint or weight files
2. Reconstructs `RouterModel` and `SimpleCNN` instances
3. Loads state_dicts into these models
4. Saves as full PyTorch models (not just state_dicts)
5. Exports to `backend/models/` directory

**Why full models?**
- Simpler loading in production
- No need to reconstruct architecture at runtime
- Backend doesn't need training code dependencies

### 2. Backend Loading Process

The backend ([backend/app/inference.py](backend/app/inference.py)):

1. Imports model architectures from `backend/router_models.py`
2. Loads full models with `torch.load()`
3. Sets models to eval mode
4. Calculates parameters and FLOPs

### 3. Inference Process

For each batch of images:

```python
# 1. Preprocess images
images = resize_and_normalize(uploaded_images)  # -> [N, 3, 32, 32]

# 2. Router decides which branch for each image
router_logits = router(images)  # -> [N, num_branches]
routes = torch.argmax(router_logits, dim=1)  # -> [N]

# 3. Each selected branch predicts its images
for i in range(batch_size):
    route_idx = routes[i]
    branch = branches[route_idx]
    logits = branch(images[i:i+1])  # -> [1, 100]
    predicted_class = torch.argmax(logits, dim=1)
```

## Step-by-Step Integration

### Step 1: Verify Your Models Are Trained

Check that you have one of:
- A checkpoint file from training (e.g., `checkpoint.pt`)
- Individual weight files (e.g., `router_20240128.pth`, `branch_0_20240128.pth`, ...)

### Step 2: Run Verification Script

```bash
python3 verify_setup.py
```

This checks:
- Python dependencies installed
- Directory structure correct
- Model files present
- Configuration matches

### Step 3: Export Models

**If you have a checkpoint:**
```bash
python3 export_models_for_demo.py --checkpoint path/to/checkpoint.pt
```

**If you have separate weight files:**
```bash
python3 export_models_for_demo.py --model_dir ./models --timestamp 20240128_123456
```

This creates:
- `backend/models/router.pth`
- `backend/models/branch_0.pth`
- `backend/models/branch_1.pth`
- ... (one per branch)

### Step 4: Update Configuration

Edit [backend/config.py](backend/config.py):

```python
# Update this to match the number of branches you exported
BRANCH_PATHS = [
    MODELS_DIR / "branch_0.pth",
    MODELS_DIR / "branch_1.pth",
    MODELS_DIR / "branch_2.pth",
    MODELS_DIR / "branch_3.pth",
    # Add or remove as needed
]
```

### Step 5: Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Step 6: Test Backend

```bash
cd backend
python3 run.py
```

Look for:
```
✓ Loaded router from backend/models/router.pth
✓ Loaded branch 0 from backend/models/branch_0.pth
✓ Loaded branch 1 from backend/models/branch_1.pth
...
✓ All models loaded successfully!
```

### Step 7: Start Frontend

In a new terminal:
```bash
cd frontend
npm run dev
```

### Step 8: Test the Demo

1. Open `http://localhost:5173`
2. Upload some test images
3. Click "Run Inference"
4. Verify predictions and routing

## Troubleshooting

### "Router model not found"

**Cause:** Models haven't been exported yet

**Fix:**
```bash
python3 export_models_for_demo.py --checkpoint your_checkpoint.pt
```

### "Failed to load router: ..."

**Cause:** Model architecture mismatch or corrupted file

**Fix:**
1. Check that `backend/router_models.py` matches your architecture
2. Re-export models
3. Check console for detailed error

### "Number of branches mismatch"

**Cause:** `config.py` has wrong number of branches

**Fix:**
Edit `backend/config.py` and update `BRANCH_PATHS` to match your model

### "FLOPs showing as null"

**Cause:** fvcore not installed

**Fix (optional):**
```bash
pip install fvcore
```

Note: FLOPs calculation is optional. The app works fine with just parameters.

### "ImportError: No module named 'router_models'"

**Cause:** Backend can't find model definitions

**Fix:**
- `backend/router_models.py` should exist (created automatically)
- If missing, copy from root: `cp router_models.py backend/`

## Model Architecture Compatibility

### Required Model Interfaces

Your models must support these interfaces:

**Router:**
```python
router = RouterModel(num_branches=4)
logits = router(images)  # images: [N, 3, 32, 32] -> logits: [N, 4]
```

**Branch:**
```python
branch = SimpleCNN(num_classes=100)
logits = branch(images)  # images: [N, 3, 32, 32] -> logits: [N, 100]
```

### Supported Configurations

- **Number of branches**: Any (tested with 4)
- **Number of classes**: Any (default 100 for CIFAR-100)
- **Branch architectures**: Can vary per branch
- **Input size**: Must be 32×32 RGB
- **Normalization**: CIFAR-100 mean/std

### Customizing for Different Architectures

If your models differ, update:

1. **Model definitions**: Edit `backend/router_models.py`
2. **Number of classes**: Edit `backend/config.py` -> `NUM_CLASSES`
3. **Class names**: Edit `backend/config.py` -> `CIFAR100_CLASSES`
4. **Input size**: Edit `backend/config.py` -> `INPUT_SIZE`
5. **Normalization**: Edit `backend/config.py` -> `CIFAR_MEAN`, `CIFAR_STD`

## Advanced: Using Different Model Formats

### If you have state_dicts only

Modify `backend/app/inference.py` `load_models()` method:

```python
from router_models import RouterModel, SimpleCNN

# For router
router = RouterModel(num_branches=4)
router.load_state_dict(torch.load(router_path, map_location=device))

# For branches
branch = SimpleCNN(num_classes=100, base_channels=32, num_layers=3)
branch.load_state_dict(torch.load(branch_path, map_location=device))
```

### If you use different model classes

1. Copy your model class file to `backend/`
2. Import in `backend/app/inference.py`
3. Pass model classes to `load_models()`

## Testing Your Integration

### Minimal Test

```python
# Test loading
python3 -c "
import torch
from pathlib import Path
router = torch.load('backend/models/router.pth', map_location='cpu')
print('Router loaded:', router)
"
```

### Full Test with Dummy Data

```python
# Test inference
python3 -c "
import torch
from pathlib import Path

# Load models
router = torch.load('backend/models/router.pth', map_location='cpu')
branch = torch.load('backend/models/branch_0.pth', map_location='cpu')

# Test with dummy data
x = torch.randn(1, 3, 32, 32)
route_logits = router(x)
print('Router output shape:', route_logits.shape)

class_logits = branch(x)
print('Branch output shape:', class_logits.shape)
"
```

## Performance Considerations

### Memory Usage

- Router: ~0.5M parameters (~2 MB)
- Branches: Varies (1-10M parameters, 4-40 MB each)
- Total: Expect 20-100 MB for full system

### Inference Speed

- CPU: ~50-200 ms per batch (depends on batch size)
- GPU: ~5-20 ms per batch
- Router overhead: ~2-5 ms

### Optimization Tips

1. **Use GPU**: Set `USE_CUDA=1` when running backend
2. **Batch processing**: Upload multiple images at once
3. **Model optimization**: Use `torch.jit.script()` for production

## Next Steps

Once integrated:

1. Test with real CIFAR-100 test images
2. Compare routing decisions with your training results
3. Analyze savings metrics
4. Customize the frontend visualization
5. Deploy to production (see [README.md](README.md))

## Support

If you encounter issues:

1. Run `python3 verify_setup.py` to diagnose
2. Check backend console logs for errors
3. Verify model architecture matches
4. Test models independently first

## Summary

✅ Your models work with the demo as-is
✅ Export script handles the conversion
✅ Backend loads and serves inference
✅ Frontend visualizes results

No code changes needed to your training code or model definitions!
