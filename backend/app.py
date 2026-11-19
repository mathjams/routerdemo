"""HTTP API for the demo. Run from the repo root:  uvicorn backend.app:app --reload"""
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from .inference import RoutedClassifier, load_image

MAX_FILES = 5000  # per request; the frontend enforces the same limit

app = FastAPI(title="Adaptive Router Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
classifier = RoutedClassifier()


class ModelStats(BaseModel):
    name: str
    params: int
    latency_ms: float  # single-image forward pass, measured at startup


class ModelInfo(BaseModel):
    router: ModelStats
    branches: List[ModelStats]
    baseline_branch: int  # index of the largest branch, the comparison baseline


class Prediction(BaseModel):
    filename: str
    image_preview: str  # the 32x32 input the models saw, as a PNG data URI
    predicted_class: int
    predicted_label: str
    confidence: float
    route_chosen: int  # index of the branch the router picked
    route_confidence: float
    baseline_class: int  # what the largest branch says about the same image
    baseline_label: str
    baseline_confidence: float


class BranchUsage(BaseModel):
    count: int
    top_labels: List[str]  # most common predictions among the images routed here


class Savings(BaseModel):
    routed: float
    baseline: float
    savings_percent: float


class Metrics(BaseModel):
    total_images: int
    avg_confidence: float
    routing: List[BranchUsage]  # one entry per branch
    params_savings: Savings  # parameters of the branch that ran, per image
    time_savings: Savings  # milliseconds per image, router included


class PredictResponse(BaseModel):
    results: List[Prediction]
    metrics: Metrics
    skipped: List[str]  # uploads that could not be read as images


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return classifier.info


# FastAPI's own file parameters are capped at 1000 files per request, so the form is
# read by hand; this tells the interactive docs at /docs what the endpoint expects.
UPLOAD_BODY = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object",
    "required": ["files"],
    "properties": {"files": {"type": "array", "items": {"type": "string", "format": "binary"}}},
}}}}}


@app.post("/predict", response_model=PredictResponse, openapi_extra=UPLOAD_BODY)
async def predict(request: Request):
    """Route and classify the images uploaded in the multipart field `files`."""
    async with request.form(max_files=MAX_FILES) as form:
        uploads = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
        if not uploads:
            raise HTTPException(400, "No files uploaded")
        return await run_in_threadpool(classify, uploads)


def classify(uploads):
    names, images, skipped = [], [], []
    for upload in uploads:
        try:
            images.append(load_image(upload.file))
            names.append(upload.filename)
        except Exception:  # Pillow raises many different errors for files it cannot read
            skipped.append(upload.filename)
    if not images:
        raise HTTPException(400, "None of the uploaded files could be read as images")

    results, metrics = classifier.predict(images)
    for name, result in zip(names, results):
        result["filename"] = name
    return {"results": results, "metrics": metrics, "skipped": skipped}
