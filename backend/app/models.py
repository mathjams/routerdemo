"""Pydantic models for API request/response validation."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    num_branches: int


class BranchInfo(BaseModel):
    """Information about a branch model."""
    id: int
    name: str
    params: int
    flops: Optional[int] = None
    params_m: float = Field(description="Parameters in millions")
    flops_m: Optional[float] = Field(None, description="FLOPs in millions")


class RouterInfo(BaseModel):
    """Information about the router model."""
    name: str
    params: int
    flops: Optional[int] = None
    params_m: float
    flops_m: Optional[float] = None


class ModelInfoResponse(BaseModel):
    """Model architecture information response."""
    num_branches: int
    num_classes: int
    branches: List[BranchInfo]
    router: RouterInfo


class PredictionResult(BaseModel):
    """Single image prediction result."""
    filename: str
    predicted_class: int
    predicted_label: str
    confidence: float
    route_chosen: int
    route_confidence: float
    branch4_predicted_class: int = Field(description="Branch 4 (largest) prediction")
    branch4_predicted_label: str = Field(description="Branch 4 (largest) prediction label")
    branch4_confidence: float = Field(description="Branch 4 (largest) confidence")
    ground_truth: Optional[int] = None
    is_correct: Optional[bool] = None
    image_preview: str = Field(description="Base64 encoded preview image")


class SavingsMetrics(BaseModel):
    """Resource savings metrics."""
    total_routed: int
    total_baseline: int
    savings: int
    savings_percent: float
    avg_routed: int
    baseline_value: int


class AggregateMetrics(BaseModel):
    """Aggregate metrics for the batch."""
    routing_distribution: Dict[int, int]
    total_images: int
    avg_confidence: float
    accuracy: Optional[float] = None
    params_savings: SavingsMetrics
    time_savings: SavingsMetrics


class PredictionResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictionResult]
    metrics: AggregateMetrics


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
