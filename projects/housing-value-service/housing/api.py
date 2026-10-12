"""Load a released model once, validate requests, serve and observe predictions."""

from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Literal
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from housing.artifact import Predictor
from housing.preprocessing import input_frame

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("housing.requests")


class District(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    longitude: float = Field(ge=-125, le=-114)
    latitude: float = Field(ge=32, le=42)
    housing_median_age: float = Field(ge=1, le=52)
    total_rooms: float = Field(gt=0, le=1000000)
    total_bedrooms: float | None = Field(default=None, ge=0, le=1000000)
    population: float = Field(gt=0, le=1000000)
    households: float = Field(gt=0, le=1000000)
    median_income: float = Field(gt=0, le=20)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.total_bedrooms is not None and self.total_bedrooms > self.total_rooms:
            raise ValueError("total_bedrooms cannot exceed total_rooms")
        if self.households > self.population:
            raise ValueError("households cannot exceed population")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[District] = Field(min_length=1, max_length=50)
    include_trace: bool = False


class Observation:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}
        self.predictions = 0
        self.duration = 0.0
        self.started = time.time()

    def record(self, path, status, seconds):
        allowed = {"/", "/health", "/ready", "/metrics", "/v1/model", "/v1/predict", "/v1/example", "/docs", "/openapi.json"}
        path = path if path in allowed else "other"
        with self.lock:
            key = (path, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            self.duration += seconds

    def increment_predictions(self, n):
        with self.lock:
            self.predictions += n

    def prometheus(self):
        with self.lock:
            lines = ["# TYPE housing_requests_total counter"]
            lines += [f'housing_requests_total{{path="{p}",status="{s}"}} {n}'
                      for (p, s), n in sorted(self.requests.items())]
            lines += ["# TYPE housing_predictions_total counter", f"housing_predictions_total {self.predictions}",
                      "# TYPE housing_request_duration_seconds_sum counter", f"housing_request_duration_seconds_sum {self.duration}",
                      "# TYPE housing_process_uptime_seconds gauge", f"housing_process_uptime_seconds {time.time() - self.started}"]
            return "\n".join(lines) + "\n"


def check_key(x_api_key: str | None = Header(default=None)):
    configured = os.getenv("API_KEY", "")
    if configured and (not x_api_key or not hmac.compare_digest(x_api_key, configured)):
        raise HTTPException(status_code=401, detail="Valid X-API-Key required")


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.getenv("MODEL_PATH", str(ROOT / "artifacts/model.json")))
        app.state.predictor = Predictor.load(path)
        app.state.model_path = path
        yield
        app.state.predictor = None

    app = FastAPI(title="Housing Value Service", version="1.0.0", lifespan=lifespan)
    app.state.observation = Observation()

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        size = request.headers.get("content-length")
        try:
            if size and int(size) > 65536:
                response = JSONResponse({"detail": "Request body exceeds 64 KiB"}, status_code=413)
            else:
                response = await call_next(request)
        except Exception:
            logger.exception(json.dumps({"request_id": request_id, "event": "request_failed"}))
            response = JSONResponse({"detail": "Internal service error", "request_id": request_id}, status_code=500)
        elapsed = time.perf_counter() - started
        app.state.observation.record(request.url.path, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.3f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        logger.info(json.dumps({"request_id": request_id, "path": request.url.path,
                               "status": response.status_code, "duration_ms": round(elapsed * 1000, 3)}))
        return response

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(503, "No model loaded")
        return {"status": "ready", "model_version": predictor.payload["model_version"]}

    @app.get("/v1/model")
    def model_info():
        payload = app.state.predictor.payload
        history = []
        path = app.state.model_path.parent / "releases" / payload["model_version"] / "loss-history.json"
        if path.exists():
            history = json.loads(path.read_text())
        return {"model_version": payload["model_version"], "target": payload["target"],
                "algorithm": "NumPy ridge regression with batch gradient descent",
                "evaluation": payload["evaluation"], "split_counts": payload["split_counts"],
                "history": history, "limitations": ["Historical census data", "District median, not an individual house price", "Random district split is not a spatial generalization test", "Targets are capped near USD 500001"]}

    @app.get("/v1/example")
    def example():
        return json.loads((ROOT / "examples/request.json").read_text())

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(check_key)])
    def monitoring():
        return PlainTextResponse(app.state.observation.prometheus(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/predict", dependencies=[Depends(check_key)])
    def predict(body: PredictionRequest, request: Request):
        predictor = app.state.predictor
        records = [r.model_dump() for r in body.records]
        values = predictor.predict(records)
        app.state.observation.increment_predictions(len(values))
        result = {
            "request_id": request.state.request_id, "model_version": predictor.payload["model_version"],
            "target": "district_median_house_value", "unit": "USD",
            "predictions": [{"value": round(float(value), 2), "warnings": predictor.warnings(record)}
                            for value, record in zip(values, records)],
        }
        if body.include_trace:
            x = predictor.preprocessor.transform(input_frame(records))[0]
            contributions = x * predictor.weights * 100000
            result["trace_first_record"] = {
                "features": [{"name": name, "transformed_value": float(v), "weight": float(w), "contribution_usd": float(c)}
                             for name, v, w, c in zip(predictor.preprocessor.feature_names_, x, predictor.weights, contributions)],
                "bias_usd": predictor.bias * 100000,
                "raw_value_usd": float(contributions.sum() + predictor.bias * 100000),
                "note": "Linear contributions explain this calculation; they are not causal effects or uncertainty bounds.",
            }
        return result

    return app


app = create_app()
