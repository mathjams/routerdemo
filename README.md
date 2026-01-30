# Adaptive Router Demo

A full-stack web application for visualizing adaptive routing in neural networks using PyTorch. Upload images, run inference through a router-based model system, and see real-time metrics on computational savings.

## Features

- 🚀 **Dynamic Routing**: Router model selects optimal branch for each input
- 📊 **Interactive Visualization**: SVG-based router graph showing model architecture
- 💾 **Computational Savings**: Track FLOPs and parameters saved vs baseline
- 📸 **Batch Processing**: Upload and process multiple images at once
- 📈 **Detailed Metrics**: Routing distribution, accuracy, confidence scores
- 🎨 **Modern UI**: React + Tailwind CSS with responsive design

## Architecture

- **Backend**: FastAPI + PyTorch
- **Frontend**: React + Vite + Tailwind CSS
- **Models**: Router model + N branch models (CIFAR-100 classification)

## Prerequisites

- Python 3.8+
- Node.js 16+
- PyTorch 2.0+
- Your trained models (router.pth, branch_0.pth, branch_1.pth, ...)

## Quick Start

### 1. Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your model files in backend/models/
# - router.pth
# - branch_0.pth
# - branch_1.pth
# - branch_2.pth
# - ... (as many branches as you have)
```

**Important**: Update `config.py` to match your setup:
- Set the correct number of branch models in `BRANCH_PATHS`
- Adjust `NUM_CLASSES` if not using CIFAR-100
- Update `CIFAR100_CLASSES` or replace with your class names

### 2. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 3. Run the Application

**Terminal 1 - Start Backend:**
```bash
cd backend
python run.py
```

The backend will start on `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:5173`

### 4. Use the Application

1. Open `http://localhost:5173` in your browser
2. Upload one or more images (JPG, PNG, etc.)
3. Click "Run Inference"
4. View results:
   - Predictions with confidence scores
   - Router graph showing which branches were used
   - Computational savings metrics
   - Routing distribution

## Project Structure

```
routerdemo/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── inference.py         # Model loading & inference
│   │   ├── preprocessing.py     # Image preprocessing
│   │   ├── utils.py             # FLOPs/params calculation
│   │   └── models.py            # API schemas
│   ├── models/                  # Place your .pth files here
│   ├── config.py                # Configuration
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── RouterGraph.jsx
│   │   │   ├── ResultsTable.jsx
│   │   │   └── MetricsCards.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## API Endpoints

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "num_branches": 4
}
```

### GET `/model-info`
Get model architecture information.

**Response:**
```json
{
  "num_branches": 4,
  "num_classes": 100,
  "branches": [...],
  "router": {...}
}
```

### POST `/predict`
Run inference on uploaded images.

**Request:**
- Form data with multiple image files
- Optional: JSON array of ground truth labels

**Response:**
```json
{
  "results": [...],
  "metrics": {
    "routing_distribution": {...},
    "params_savings": {...},
    "flops_savings": {...}
  }
}
```

## Configuration

### Backend Configuration

Edit `backend/config.py`:

```python
# Model paths
BRANCH_PATHS = [
    MODELS_DIR / "branch_0.pth",
    MODELS_DIR / "branch_1.pth",
    # Add more branches as needed
]

# Number of classes (e.g., 100 for CIFAR-100)
NUM_CLASSES = 100

# Class names
CIFAR100_CLASSES = [...]  # Update with your class names

# Device
DEVICE = "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
```

To use GPU:
```bash
USE_CUDA=1 python run.py
```

### Frontend Configuration

Create `frontend/.env` (optional):
```
VITE_API_URL=http://localhost:8000
```

## Model Loading

The application expects PyTorch models saved as full models (not just state_dicts). If your models are saved differently, modify `inference.py`:

### If you have state_dicts only:

1. Define your model architecture classes
2. Modify `load_models()` in `inference.py`:

```python
# Example for state_dict loading
router = YourRouterModel()
router.load_state_dict(torch.load(router_path, map_location=self.device))

branch = YourBranchModel()
branch.load_state_dict(torch.load(branch_path, map_location=self.device))
```

### If you have full models:

The current code should work as-is:
```python
self.router = torch.load(router_path, map_location=self.device)
branch = torch.load(branch_path, map_location=self.device)
```

## Customization

### Adding Ground Truth Labels

To compare predictions against ground truth:

1. Prepare a JSON array of labels: `[0, 1, 2, ...]`
2. Update `App.jsx` to include labels in the form data:

```javascript
const labels = [0, 1, 2, ...]; // Your ground truth labels
formData.append('labels', JSON.stringify(labels));
```

### Changing Image Size

If your models use a different input size than 32×32:

1. Update `INPUT_SIZE` in `backend/config.py`
2. Update normalization stats if needed

### Custom Class Names

Replace `CIFAR100_CLASSES` in `backend/config.py` with your class names.

## Troubleshooting

### Models Not Loading

**Problem**: "Models not loaded" warning on startup

**Solution**:
- Check that model files exist in `backend/models/`
- Verify file names match those in `config.py`
- Check console output for specific error messages
- Ensure models are saved in compatible PyTorch format

### FLOPs Calculation Not Working

**Problem**: FLOPs showing as `null`

**Solution**:
- Install fvcore: `pip install fvcore`
- If still not working, the app will fall back to showing only parameters

### CORS Errors

**Problem**: Frontend can't connect to backend

**Solution**:
- Check that backend is running on port 8000
- Check that frontend is running on port 5173
- Verify CORS settings in `backend/app/main.py`

### Import Errors

**Problem**: Module not found errors

**Solution**:
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- For frontend: `npm install`

## Performance Tips

1. **GPU Acceleration**: Set `USE_CUDA=1` environment variable
2. **Batch Size**: Adjust `BATCH_SIZE` in `config.py` for larger batches
3. **Model Optimization**: Use `torch.jit.script()` for production deployment

## Development

### Backend Development

With auto-reload enabled:
```bash
python run.py  # uvicorn auto-reload is enabled by default
```

### Frontend Development

Vite dev server with hot reload:
```bash
npm run dev
```

### Building for Production

**Backend:**
```bash
# Set reload=False in run.py
python run.py
```

**Frontend:**
```bash
npm run build
npm run preview
```

## License

MIT License - feel free to use this for your projects!

## Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Lucide Icons](https://lucide.dev/)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review console output for error messages
3. Ensure all dependencies are installed correctly
