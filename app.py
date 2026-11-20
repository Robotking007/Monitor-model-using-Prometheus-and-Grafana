"""
app.py
------
FastAPI Prediction Service instrumented with Prometheus Client.
Assignment 8: Monitor a Machine Learning Model using Prometheus and Grafana
Student: Siddhant Udavant (Roll No: 20, PRN: 12414447, Branch: AI&DS, Div: F)

Features:
- Machine Learning inference using serialized Random Forest model.
- Complete Prometheus metrics instrumentation adhering to assignment specification.
- Real-time feature data drift detection via rolling z-scores.
- Ground-truth feedback loop for online accuracy monitoring.
- High-performance, low-cardinality label design.
"""

import collections
import json
import os
import threading
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------
# 1. Model & Reference Metadata Initialization
# ---------------------------------------------------------
MODEL_PATH = os.path.join("model", "model.joblib")
STATS_PATH = os.path.join("model", "training_stats.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
    raise RuntimeError("Model or stats file not found! Please run train.py first.")

model = joblib.load(MODEL_PATH)
with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = json.load(f)

FEATURES: List[str] = stats["features"]
CLASSES: List[str] = stats["classes"]
MODEL_VERSION: str = stats.get("version", "1.0.0")
APP_START_TIME = time.time()

# Sliding window buffer for drift analysis (retains last 100 feature vectors)
SLIDING_WINDOW_SIZE = 100
INPUT_WINDOW = collections.deque(maxlen=SLIDING_WINDOW_SIZE)
INPUT_WINDOW_LOCK = threading.Lock()

# ---------------------------------------------------------
# 2. Prometheus Metrics Instrumentation
# ---------------------------------------------------------
# Required Assignment Specification Metrics
ML_PREDICTION_REQUESTS = Counter(
    "ml_prediction_requests_total",
    "Total prediction requests received by the service",
    ["method", "endpoint", "status"]
)

ML_PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Total failed prediction requests",
    ["endpoint", "error_type"]
)

ML_PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Distribution of prediction request latency in seconds",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

ML_PREDICTIONS = Counter(
    "ml_predictions_total",
    "Total successfully generated predictions categorized by class",
    ["predicted_class"]
)

# Additional Production MLOps Metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received across all endpoints",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Overall HTTP request duration in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

MODEL_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Distribution of prediction confidence (max predicted probability)",
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
)

MODEL_FEATURE_DRIFT_ZSCORE = Gauge(
    "model_feature_drift_zscore",
    "Z-score of rolling feature mean versus baseline training mean",
    ["feature"]
)

MODEL_FEEDBACK = Counter(
    "model_feedback_total",
    "Ground-truth feedback records received for evaluating online accuracy",
    ["outcome"]
)

MODEL_INFO = Gauge(
    "model_info",
    "Model metadata and serving version information",
    ["version", "model_type"]
)
MODEL_INFO.labels(version=MODEL_VERSION, model_type="RandomForestClassifier").set(1)

for code in ("200", "422", "500"):
    ML_PREDICTION_REQUESTS.labels(method="POST", endpoint="/predict", status=code)
for error_type in ("validation_error", "inference_failure"):
    ML_PREDICTION_ERRORS.labels(endpoint="/predict", error_type=error_type)
for outcome in ("correct", "incorrect"):
    MODEL_FEEDBACK.labels(outcome=outcome)
for label in CLASSES:
    ML_PREDICTIONS.labels(predicted_class=label)
ML_PREDICTION_LATENCY.labels(endpoint="/predict")

# Initialize drift gauges to zero
for feat in FEATURES:
    MODEL_FEATURE_DRIFT_ZSCORE.labels(feature=feat).set(0.0)

# ---------------------------------------------------------
# 3. FastAPI Application & Pydantic Schemas
# ---------------------------------------------------------
app = FastAPI(
    title="ML Model Serving API with Prometheus Monitoring",
    description="Production-ready REST API for Iris classification with real-time operational metrics.",
    version=MODEL_VERSION,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=20.0, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0.0, le=20.0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0.0, le=20.0, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.0, le=20.0, description="Petal width in cm")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class FeedbackPayload(BaseModel):
    predicted: str
    actual: str


# ---------------------------------------------------------
# 4. Middleware & Error Handling
# ---------------------------------------------------------
@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    route = request.scope.get("route")
    endpoint = getattr(route, "path", "unmatched")
    status_str = str(response.status_code)

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=endpoint,
        status=status_str
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle 422 Unprocessable Entity requests and record metrics."""
    if request.scope.get("endpoint") is predict:
        ML_PREDICTION_REQUESTS.labels(
            method="POST", endpoint="/predict", status="422"
        ).inc()
        ML_PREDICTION_ERRORS.labels(
            endpoint="/predict", error_type="validation_error"
        ).inc()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body}
    )


def compute_feature_drift():
    """Computes z-scores of rolling feature means against training baselines."""
    if len(INPUT_WINDOW) < 15:
        return

    data = np.array(INPUT_WINDOW)
    n = len(data)
    for i, feature_name in enumerate(FEATURES):
        train_mean = stats["mean"][i]
        train_std = stats["std"][i]
        rolling_mean = float(data[:, i].mean())
        
        # Standard error of the mean
        standard_error = train_std / np.sqrt(n)
        z_score = (rolling_mean - train_mean) / standard_error if standard_error > 0 else 0.0
        MODEL_FEATURE_DRIFT_ZSCORE.labels(feature=feature_name).set(round(float(z_score), 3))


# ---------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------
@app.get("/", summary="API Root & Info")
def root():
    return {
        "service": "MLOps Prediction Service",
        "model": "Iris Random Forest Classifier",
        "version": MODEL_VERSION,
        "endpoints": {
            "predict": "/predict (POST)",
            "feedback": "/feedback (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)"
        },
        "student": "Siddhant Udavant",
        "roll_no": 20,
        "div": "F"
    }


@app.get("/health", summary="Health Check")
def health():
    uptime = time.time() - APP_START_TIME
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(uptime, 2),
        "buffered_samples": len(INPUT_WINDOW)
    }


@app.post("/predict", response_model=PredictionResponse, summary="Perform Model Inference")
def predict(features: IrisFeatures):
    start_time = time.perf_counter()
    x = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    try:
        probabilities = model.predict_proba(x)[0]
    except Exception as exc:
        ML_PREDICTION_REQUESTS.labels(method="POST", endpoint="/predict", status="500").inc()
        ML_PREDICTION_ERRORS.labels(endpoint="/predict", error_type="inference_failure").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(exc)}"
        )

    duration = time.perf_counter() - start_time
    class_idx = int(np.argmax(probabilities))
    predicted_label = CLASSES[class_idx]
    confidence_val = float(probabilities[class_idx])

    # Record Prometheus metrics
    ML_PREDICTION_REQUESTS.labels(method="POST", endpoint="/predict", status="200").inc()
    ML_PREDICTIONS.labels(predicted_class=predicted_label).inc()
    ML_PREDICTION_LATENCY.labels(endpoint="/predict").observe(duration)
    MODEL_CONFIDENCE.observe(confidence_val)

    # Track in sliding window and update drift gauges
    with INPUT_WINDOW_LOCK:
        INPUT_WINDOW.append(x[0])
        compute_feature_drift()

    prob_dict = {name: round(float(prob), 4) for name, prob in zip(CLASSES, probabilities)}

    return PredictionResponse(
        prediction=predicted_label,
        class_id=class_idx,
        confidence=round(confidence_val, 4),
        probabilities=prob_dict,
        model_version=MODEL_VERSION
    )


@app.post("/feedback", summary="Submit Prediction Ground Truth")
def feedback(fb: FeedbackPayload):
    outcome = "correct" if fb.predicted.lower() == fb.actual.lower() else "incorrect"
    MODEL_FEEDBACK.labels(outcome=outcome).inc()
    return {
        "status": "success",
        "outcome_recorded": outcome,
        "predicted": fb.predicted,
        "actual": fb.actual
    }


@app.get("/metrics", summary="Prometheus Metrics Scrape Endpoint")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

