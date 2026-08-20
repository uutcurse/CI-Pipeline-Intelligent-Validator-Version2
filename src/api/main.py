import uuid
import yaml
import time
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.schemas import (
    PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse, ErrorResponse
)
from src.inference.workflow_preprocessor import WorkflowPreprocessor
from src.inference.service import E06InferenceService

def load_config():
    with open('configs/api.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
MAX_SIZE = config['security']['max_workflow_size_bytes']

# Global state
app_state = {
    "preprocessor": None,
    "service": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app_state["preprocessor"] = WorkflowPreprocessor()
        app_state["service"] = E06InferenceService()
        # Bind service to preprocessor so it doesn't initialize it lazily later
        app_state["preprocessor"].service = app_state["service"]
    except Exception as e:
        # Fails clearly if model missing
        print(f"Failed to load models: {e}")
        # Not exiting here so that Health check can report DOWN
    yield
    # Shutdown
    app_state["preprocessor"] = None
    app_state["service"] = None

app = FastAPI(
    title="GitHub Actions Workflow Classifier",
    description="Inference API for the frozen E06 Hybrid Logistic Regression model.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config['security']['allowed_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id_and_size_limit(request: Request, call_next):
    req_id = str(uuid.uuid4())
    
    # Simple size limit check if Content-Length provided
    cl = request.headers.get('content-length')
    if cl and int(cl) > MAX_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Workflow exceeds maximum allowed size."}}
        )
        
    # Inject request_id into state
    request.state.req_id = req_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": {"code": "INVALID_WORKFLOW", "message": str(exc)}}
    )
    
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred during inference."}}
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    svc = app_state.get("service")
    prep = app_state.get("preprocessor")
    
    is_up = svc is not None and prep is not None
    if is_up:
        svc_health = svc.health()
        if svc_health["status"] != "UP":
            is_up = False
            
    return {
        "status": "UP" if is_up else "DOWN",
        "model_loaded": bool(svc and svc._model is not None),
        "preprocessor_loaded": prep is not None
    }

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    svc = app_state.get("service")
    if not svc:
        raise HTTPException(status_code=503, detail="Model service not ready")
        
    info = svc.model_info()
    return {
        "model_id": info.get("model_id", "E06"),
        "model_type": info.get("model_type", "HybridBaseline_LogisticRegression"),
        "feature_count": info.get("feature_count", 80),
        "classes": info.get("label_mapping", {"0": "LOW", "1": "MEDIUM", "2": "HIGH"}),
        "frozen": True
    }

from src.inference.explainer import generate_explanation

@app.post("/predict", response_model=PredictResponse)
async def predict_workflow(req: PredictRequest, request: Request):
    if len(req.workflow_yaml.encode('utf-8')) > MAX_SIZE:
        raise ValueError("Workflow exceeds maximum allowed size.")
        
    prep = app_state.get("preprocessor")
    if not prep:
        raise HTTPException(status_code=503, detail="Preprocessor not ready")
        
    # Preprocessor handles empty/malformed/invalid logic internally and raises ValueError
    result = prep.predict(req.workflow_yaml)
    
    # Generate explanation from structural features
    explanation = None
    struct_feats = result["preprocessing"].get("structural_features")
    if struct_feats:
        try:
            explanation = generate_explanation(result["prediction"]["predicted_label"], struct_feats)
        except Exception:
            pass
    
    return {
        "prediction": {
            "predicted_class": result["prediction"]["predicted_class"],
            "predicted_label": result["prediction"]["predicted_label"],
            "probabilities": result["prediction"]["probabilities"]
        },
        "workflow": {
            "name": result["preprocessing"]["workflow_name"],
            "job_count": result["preprocessing"]["job_count"],
            "step_count": result["preprocessing"]["step_count"]
        },
        "explanation": explanation,
        "inference_duration_ms": result["inference_duration"] * 1000.0
    }
