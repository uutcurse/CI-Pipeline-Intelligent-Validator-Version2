from pydantic import BaseModel, Field, validator
from typing import Dict

class PredictRequest(BaseModel):
    workflow_yaml: str = Field(
        ..., 
        description="Raw GitHub Actions workflow YAML string.",
        min_length=1
    )

class PredictionProbabilities(BaseModel):
    LOW: float
    MEDIUM: float
    HIGH: float

class PredictionResult(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: PredictionProbabilities

class WorkflowMetadata(BaseModel):
    name: str | None
    job_count: int
    step_count: int

class ExplanationSignal(BaseModel):
    name: str
    detail: str

class ExplanationResult(BaseModel):
    title: str = "Key Risk Signals Detected"
    signals: list[ExplanationSignal] = []

class PredictResponse(BaseModel):
    prediction: PredictionResult
    workflow: WorkflowMetadata
    explanation: ExplanationResult | None = None
    inference_duration_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool

class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    feature_count: int
    classes: Dict[str, str]
    frozen: bool

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
