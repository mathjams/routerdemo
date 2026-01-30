# Adaptive Router Demo - Architecture

## Folder Structure

```
routerdemo/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── models.py            # Pydantic models for API
│   │   ├── inference.py         # Model loading & inference
│   │   ├── preprocessing.py     # Image preprocessing
│   │   └── utils.py             # FLOPs/params calculation
│   ├── models/                  # Saved PyTorch models
│   │   ├── router.pth
│   │   ├── branch_0.pth
│   │   ├── branch_1.pth
│   │   └── ...
│   ├── config.py                # Configuration (model paths, classes, etc.)
│   ├── requirements.txt
│   └── run.py                   # Entry point
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── RouterGraph.jsx
│   │   │   ├── ResultsTable.jsx
│   │   │   └── MetricsCards.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── README.md
└── load_model.py               # Your existing model loading script
```

## API Schema

### Endpoints

#### 1. GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "num_branches": 4
}
```

#### 2. GET `/model-info`
Get model architecture information.

**Response:**
```json
{
  "num_branches": 4,
  "num_classes": 100,
  "branches": [
    {
      "id": 0,
      "name": "Branch 0",
      "params": 1234567,
      "flops": 987654321
    },
    ...
  ],
  "router": {
    "params": 50000,
    "flops": 10000000
  }
}
```

#### 3. POST `/predict`
Run inference on uploaded images.

**Request:**
- Content-Type: `multipart/form-data`
- Files: Multiple image files
- Optional field: `labels` (JSON array of ground truth labels)

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "predicted_class": 42,
      "predicted_label": "cat",
      "confidence": 0.95,
      "route_chosen": 2,
      "route_confidence": 0.87,
      "ground_truth": 42,
      "is_correct": true,
      "image_preview": "data:image/png;base64,..."
    },
    ...
  ],
  "metrics": {
    "routing_distribution": {
      "0": 5,
      "1": 10,
      "2": 15,
      "3": 20
    },
    "total_images": 50,
    "accuracy": 0.92,
    "avg_confidence": 0.89,
    "flops_saved": {
      "total_flops_routed": 1234567890,
      "total_flops_baseline": 9876543210,
      "savings_percent": 87.5
    },
    "params_saved": {
      "avg_params_routed": 2345678,
      "max_params_baseline": 5432109,
      "savings_percent": 56.8
    }
  }
}
```

## Data Flow

1. **User uploads images** → Frontend
2. **Frontend sends files** → POST `/predict` (multipart/form-data)
3. **Backend receives files** → Validates file types
4. **Preprocessing** → Resize to 32×32, normalize with CIFAR mean/std
5. **Router inference** → Outputs logits over branches, select argmax
6. **Branch inference** → Selected branch outputs class logits
7. **Compute metrics** → FLOPs/params saved, routing distribution
8. **Return results** → JSON response with predictions + metrics
9. **Frontend displays** → Results table, router graph, metrics cards

## Key Assumptions

1. **Model Format**: All models are saved as `.pth` or `.pt` files containing state_dict
2. **Input Format**: RGB images (any size) → resized to 32×32
3. **Normalization**: CIFAR-100 mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
4. **Router Output**: Logits over N branches (shape: [batch_size, num_branches])
5. **Branch Output**: Logits over 100 classes (shape: [batch_size, 100])
6. **FLOPs Calculation**: Using `fvcore` library with dummy input (1, 3, 32, 32)
7. **Params Calculation**: Sum of `p.numel()` for all parameters
8. **Baseline**: Always using the largest branch (by FLOPs or params)
9. **Device**: CPU by default, GPU if available

## Tech Stack

### Backend
- **FastAPI**: Web framework
- **PyTorch**: Model inference
- **Pillow**: Image preprocessing
- **fvcore**: FLOPs calculation
- **uvicorn**: ASGI server

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **Axios**: HTTP client
- **Lucide React**: Icons

## Performance Considerations

1. **Model Loading**: Load once at startup, keep in memory
2. **Batch Processing**: Process all uploaded images in a single batch
3. **Image Caching**: Return base64-encoded 32×32 previews to avoid re-preprocessing
4. **CORS**: Enable CORS for development (backend on :8000, frontend on :5173)
