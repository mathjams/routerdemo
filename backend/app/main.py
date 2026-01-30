"""FastAPI application for adaptive router demo."""
import io
import json
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from config import ROUTER_PATH, BRANCH_PATHS, ALLOWED_EXTENSIONS
from app.models import (
    HealthResponse, ModelInfoResponse, PredictionResponse,
    PredictionResult, ErrorResponse
)
from app.inference import AdaptiveRouterInference
from app.preprocessing import ImagePreprocessor

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Router Demo API",
    description="API for adaptive routing inference with PyTorch models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and Next.js defaults
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
inference_engine = AdaptiveRouterInference()
preprocessor = ImagePreprocessor()
models_loaded = False


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global models_loaded

    print("Starting up...")
    print(f"Router path: {ROUTER_PATH}")
    print(f"Branch paths: {BRANCH_PATHS}")

    # Check if model files exist
    if not ROUTER_PATH.exists():
        print(f"WARNING: Router model not found at {ROUTER_PATH}")
        print("Please place your router.pth file in the backend/models/ directory")
        return

    missing_branches = [p for p in BRANCH_PATHS if not p.exists()]
    if missing_branches:
        print(f"WARNING: Some branch models not found:")
        for p in missing_branches:
            print(f"  - {p}")
        print("Please place your branch_*.pth files in the backend/models/ directory")

    try:
        # Load models (exported format from export_models_for_demo.py)
        print("Loading models...")
        inference_engine.load_models(
            router_path=ROUTER_PATH,
            branch_paths=BRANCH_PATHS
        )
        models_loaded = True
        print("✓ All models loaded successfully!")

    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("The API will start but /predict endpoint will not work.")
        print("\nTroubleshooting:")
        print("1. Run: python export_models_for_demo.py --checkpoint your_checkpoint.pt")
        print("2. Check that all .pth files exist in backend/models/")
        print("3. Verify BRANCH_PATHS in backend/config.py matches your model count")
        import traceback
        traceback.print_exc()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        num_branches=inference_engine.num_branches
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model architecture information."""
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs for details."
        )

    info = inference_engine.get_model_info_dict()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None)
):
    """Run inference on uploaded images.

    Args:
        files: List of uploaded image files
        labels: Optional JSON string of ground truth labels (list of integers)

    Returns:
        Prediction results and metrics
    """
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs and ensure model files are in backend/models/"
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Parse ground truth labels if provided
    ground_truth_labels = None
    if labels:
        try:
            ground_truth_labels = json.loads(labels)
            if not isinstance(ground_truth_labels, list):
                raise ValueError("Labels must be a JSON array")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid labels format: {e}"
            )

    # Validate and load images
    images = []
    filenames = []
    invalid_files = []

    for file in files:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            invalid_files.append(f"{file.filename} (unsupported format)")
            continue

        try:
            # Read file
            contents = await file.read()

            # Validate image
            if not preprocessor.validate_image(contents):
                invalid_files.append(f"{file.filename} (invalid image)")
                continue

            # Load image
            img = Image.open(io.BytesIO(contents))
            images.append(img)
            filenames.append(file.filename)

        except Exception as e:
            invalid_files.append(f"{file.filename} (error: {str(e)})")

    if invalid_files:
        detail = "Invalid files: " + ", ".join(invalid_files)
        if not images:
            raise HTTPException(status_code=400, detail=detail)
        # Continue with valid images but warn about invalid ones
        print(f"Warning: {detail}")

    if not images:
        raise HTTPException(status_code=400, detail="No valid images found")

    try:
        # Preprocess images
        batch_tensor, previews = preprocessor.preprocess_batch(images)

        # Run inference
        results, metrics = inference_engine.predict(batch_tensor)

        # Build response
        prediction_results = []
        correct_count = 0

        for i, result in enumerate(results):
            pred_result = PredictionResult(
                filename=filenames[i],
                predicted_class=result['predicted_class'],
                predicted_label=result['predicted_label'],
                confidence=result['confidence'],
                route_chosen=result['route_chosen'],
                route_confidence=result['route_confidence'],
                branch4_predicted_class=result['branch4_predicted_class'],
                branch4_predicted_label=result['branch4_predicted_label'],
                branch4_confidence=result['branch4_confidence'],
                image_preview=previews[i]
            )

            # Add ground truth if provided
            if ground_truth_labels and i < len(ground_truth_labels):
                gt_label = ground_truth_labels[i]
                pred_result.ground_truth = gt_label
                pred_result.is_correct = (result['predicted_class'] == gt_label)
                if pred_result.is_correct:
                    correct_count += 1

            prediction_results.append(pred_result)

        # Add accuracy to metrics if ground truth provided
        if ground_truth_labels:
            metrics['accuracy'] = round(correct_count / len(results), 4)

        return PredictionResponse(
            results=prediction_results,
            metrics=metrics
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
