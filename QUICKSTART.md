# Quick Start Guide

Get your adaptive router demo running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] PyTorch 2.0+ installed
- [ ] Trained router and branch models ready

## Step 0: Verify Setup (Optional but Recommended)

Run the verification script to check your setup:

```bash
python3 verify_setup.py
```

This checks:
- Dependencies installed
- Directory structure correct
- Model files present
- Configuration valid

Fix any issues before proceeding.

## Step 1: Export Your Models

First, export your trained models to the demo-compatible format:

```bash
# If you have a checkpoint file:
python export_models_for_demo.py --checkpoint ./checkpoints/your_checkpoint.pt

# OR if you have individual weight files:
python export_models_for_demo.py --model_dir ./models --timestamp YOUR_TIMESTAMP
```

This will create files in `backend/models/`:
- `router.pth`
- `branch_0.pth`, `branch_1.pth`, etc.

## Step 2: Configure Backend

Update `backend/config.py` to match your model count:

```python
# Example for 4 branches:
BRANCH_PATHS = [
    MODELS_DIR / "branch_0.pth",
    MODELS_DIR / "branch_1.pth",
    MODELS_DIR / "branch_2.pth",
    MODELS_DIR / "branch_3.pth",
]
```

## Step 3: Install Dependencies

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

## Step 4: Start the Application

Open **two terminal windows**.

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python run.py
```

Wait for: `✓ All models loaded successfully!`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Step 5: Use the Demo

1. Open `http://localhost:5173` in your browser
2. Drag and drop some images (or click to browse)
3. Click "Run Inference"
4. View your results!

## What You'll See

- **Predictions**: Class predictions with confidence scores
- **Router Graph**: Visual diagram showing which branches were used
- **Metrics Cards**:
  - Total images processed
  - Average confidence
  - Accuracy (if you provide ground truth)
- **Savings Metrics**:
  - Parameters saved vs. always using largest model
  - FLOPs saved (if fvcore is installed)
  - Routing distribution across branches

## Common Issues

### "Models not loaded"
- Check that model files exist in `backend/models/`
- Verify `config.py` has correct number of branches
- Check backend console for specific error messages

### FLOPs showing as null
- Install fvcore: `pip install fvcore`
- If still not working, app will show parameters only (still useful!)

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check that both servers are running
- Look for CORS errors in browser console

### Import errors in export script
- Make sure `router_models.py` is in the same directory
- Activate your Python environment where PyTorch is installed

## Advanced Usage

### Using GPU
```bash
USE_CUDA=1 python run.py
```

### Adding Ground Truth Labels

Edit `frontend/src/App.jsx` around line 45:

```javascript
// Add your ground truth labels
const labels = [0, 1, 42, 15, ...];  // One per image
formData.append('labels', JSON.stringify(labels));
```

### Testing with Sample Images

CIFAR-100 uses 32×32 images but the app accepts any size (automatically resized).

Test with:
- Images from CIFAR-100 test set
- Your own images (will be resized to 32×32)
- Batch uploads (upload 10-50 images at once)

## Production Deployment

For production use:

1. **Backend**: Set `reload=False` in `backend/run.py`
2. **Frontend**:
   ```bash
   cd frontend
   npm run build
   npm run preview
   ```

3. Consider using:
   - Gunicorn or similar for FastAPI
   - Nginx to serve frontend static files
   - Docker containers for easier deployment

## Need Help?

- Check [README.md](README.md) for full documentation
- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Look at backend console logs for detailed error messages
- Check browser console for frontend errors

## What's Next?

Once the demo is running, try:

1. Upload a batch of images and analyze routing patterns
2. Compare savings metrics - how much computation is saved?
3. Explore which types of images route to which branches
4. Modify the router graph visualization in `frontend/src/components/RouterGraph.jsx`
5. Add custom metrics or visualizations

Enjoy your demo! 🚀
